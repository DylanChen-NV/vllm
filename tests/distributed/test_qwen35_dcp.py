# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Decode Context Parallel (DCP) support on Qwen3.5 hybrid attention
models (GatedDeltaNet + GQA).

Test 1 (correctness): DCP-enabled outputs must match non-DCP baseline.
Test 2 (KV cache sharding): DCP must only shard FullAttention KV cache,
    leaving Mamba/GDN recurrent state untouched.
"""

import os
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use local model if available, otherwise fall back to HuggingFace ID
QWEN35_MODEL_LOCAL = "/llm-models_ci/Qwen3.5-35B-A3B/"
QWEN35_MODEL_HF = "Qwen/Qwen3.5-0.8B"

# Qwen3.5 GQA layer defaults (from qwen3_5.py / qwen3_5_moe.py)
QWEN35_NUM_KV_HEADS = 2
QWEN35_HEAD_DIM = 256
BLOCK_SIZE = 16

# Qwen3.5 GDN layer defaults
GDN_CONV_DIM = 8192  # head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
GDN_CONV_KERNEL = 4
GDN_NUM_V_HEADS = 32
GDN_V_DIM = 128
GDN_K_DIM = 128


def _make_mock_vllm_config(
    max_model_len: int = 4096,
    dcp_size: int = 1,
    pcp_size: int = 1,
    mamba_cache_mode: str = "align",
):
    """Minimal mock of VllmConfig for max_memory_usage_bytes tests."""
    cfg = MagicMock()
    cfg.model_config.max_model_len = max_model_len
    cfg.parallel_config.decode_context_parallel_size = dcp_size
    cfg.parallel_config.prefill_context_parallel_size = pcp_size
    cfg.cache_config.mamba_cache_mode = mamba_cache_mode
    cfg.scheduler_config.max_num_batched_tokens = 1024
    return cfg


def _make_full_attention_spec(block_size: int = BLOCK_SIZE) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=QWEN35_NUM_KV_HEADS,
        head_size=QWEN35_HEAD_DIM,
        dtype=torch.bfloat16,
    )


def _make_mamba_spec(block_size: int = 1) -> MambaSpec:
    """MambaSpec matching Qwen3.5 GDN state shapes."""
    conv_state_shape = (GDN_CONV_DIM, GDN_CONV_KERNEL - 1)  # (8192, 3)
    ssm_state_shape = (GDN_NUM_V_HEADS, GDN_V_DIM, GDN_K_DIM)  # (32, 128, 128)
    return MambaSpec(
        block_size=block_size,
        shapes=(conv_state_shape, ssm_state_shape),
        dtypes=(torch.bfloat16, torch.bfloat16),
    )


# ===========================================================================
# Test 1: Output correctness — DCP vs non-DCP must produce identical output
# ===========================================================================


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="Need at least 4 GPUs for TP=4 DCP=2",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9,
    reason="GQA+DCP requires compute capability >= 9.0 (H100+)",
)
def test_qwen35_dcp_output_correctness(num_gpus_available):
    """
    Compare greedy-decoded outputs between:
      - baseline: TP=4, DCP=1  (no context parallel)
      - target:   TP=4, DCP=2  (KV cache sharded across 2 ranks)

    With deterministic weights and greedy decoding, both settings must
    produce identical token sequences.

    Qwen3.5-35B-A3B: num_kv_heads=2, so DCP upper bound = TP/num_kv_heads.
    With TP=4, DCP max = 2.
    """
    from tests.utils import compare_two_settings

    if num_gpus_available < 4:
        pytest.skip("Need at least 4 GPUs")

    # Prefer local model path; fall back to HF ID (uses dummy weights)
    if os.path.isdir(QWEN35_MODEL_LOCAL):
        model = QWEN35_MODEL_LOCAL
    else:
        model = QWEN35_MODEL_HF

    common_args = [
        "--dtype", "bfloat16",
        "--max-model-len", "512",
        "--max-num-seqs", "8",
        "--tensor-parallel-size", "4",
        "--enforce-eager",
        "--enable-chunked-prefill",
    ]

    args_baseline = common_args + [
        "--decode-context-parallel-size", "1",
    ]

    args_dcp = common_args + [
        "--decode-context-parallel-size", "2",
        "--dcp-kv-cache-interleave-size", "16",
        "--attention-backend", "FLASH_ATTN",
    ]

    compare_two_settings(
        model,
        args_baseline,
        args_dcp,
        method="generate",
        max_wait_seconds=600,
    )


# ===========================================================================
# Test 2: KV cache sharding — verify DCP only shards FullAttention cache
# ===========================================================================


class TestDCPKVCacheShardingConfig:
    """
    Unit tests verifying that DCP correctly differentiates between
    FullAttention (GQA) and Mamba (GDN) cache groups in a hybrid model.
    """

    # ----- 2a: FullAttention memory must decrease with DCP -----

    def test_full_attention_memory_halved_by_dcp2(self):
        """With DCP=2, each rank stores half the FullAttention KV tokens."""
        spec = _make_full_attention_spec()
        cfg_no_dcp = _make_mock_vllm_config(max_model_len=4096, dcp_size=1)
        cfg_dcp2 = _make_mock_vllm_config(max_model_len=4096, dcp_size=2)

        mem_no_dcp = spec.max_memory_usage_bytes(cfg_no_dcp)
        mem_dcp2 = spec.max_memory_usage_bytes(cfg_dcp2)

        assert mem_dcp2 < mem_no_dcp, (
            f"DCP=2 memory ({mem_dcp2}) should be less than "
            f"DCP=1 memory ({mem_no_dcp})"
        )
        # Allow small rounding difference from cdiv
        assert mem_dcp2 == pytest.approx(mem_no_dcp / 2, rel=0.05)

    def test_full_attention_memory_quartered_by_dcp4(self):
        """With DCP=4, each rank stores one quarter of FullAttention KV."""
        spec = _make_full_attention_spec()
        cfg_no_dcp = _make_mock_vllm_config(max_model_len=4096, dcp_size=1)
        cfg_dcp4 = _make_mock_vllm_config(max_model_len=4096, dcp_size=4)

        mem_no_dcp = spec.max_memory_usage_bytes(cfg_no_dcp)
        mem_dcp4 = spec.max_memory_usage_bytes(cfg_dcp4)

        assert mem_dcp4 < mem_no_dcp
        assert mem_dcp4 == pytest.approx(mem_no_dcp / 4, rel=0.05)

    # ----- 2b: Mamba/GDN memory must NOT change with DCP -----

    def test_mamba_memory_unchanged_by_dcp(self):
        """GDN recurrent state is fixed-size; DCP must not affect it."""
        spec = _make_mamba_spec()
        cfg_no_dcp = _make_mock_vllm_config(dcp_size=1)
        cfg_dcp2 = _make_mock_vllm_config(dcp_size=2)
        cfg_dcp4 = _make_mock_vllm_config(dcp_size=4)

        mem_base = spec.max_memory_usage_bytes(cfg_no_dcp)
        mem_dcp2 = spec.max_memory_usage_bytes(cfg_dcp2)
        mem_dcp4 = spec.max_memory_usage_bytes(cfg_dcp4)

        assert mem_base == mem_dcp2, (
            f"MambaSpec memory changed with DCP=2: {mem_base} -> {mem_dcp2}"
        )
        assert mem_base == mem_dcp4, (
            f"MambaSpec memory changed with DCP=4: {mem_base} -> {mem_dcp4}"
        )

    # ----- 2c: HybridKVCacheCoordinator must accept DCP > 1 -----

    def test_hybrid_coordinator_accepts_dcp(self):
        """
        HybridKVCacheCoordinator must not crash when dcp_world_size > 1.

        Currently blocked by:
            kv_cache_coordinator.py:406
            assert dcp_world_size == 1, "DCP not support hybrid attn now."

        After our changes, this test should pass.
        """
        from vllm.v1.core.kv_cache_coordinator import (
            HybridKVCacheCoordinator,
        )

        full_attn_spec = _make_full_attention_spec(block_size=BLOCK_SIZE)
        mamba_spec = _make_mamba_spec(block_size=1)

        kv_cache_config = KVCacheConfig(
            num_blocks=1000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[f"layers.{i}.self_attn" for i in range(10)],
                    kv_cache_spec=full_attn_spec,
                ),
                KVCacheGroupSpec(
                    layer_names=[f"layers.{i}.linear_attn" for i in range(30)],
                    kv_cache_spec=mamba_spec,
                ),
            ],
        )

        # hash_block_size=1 because it must divide both FullAttention
        # (block_size=16) and Mamba (block_size=1) block sizes.
        # This should NOT raise AssertionError
        coordinator = HybridKVCacheCoordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=4096,
            use_eagle=False,
            enable_caching=False,
            enable_kv_cache_events=False,
            dcp_world_size=2,
            pcp_world_size=1,
            hash_block_size=1,
        )

        # Verify per-group block_size:
        # FullAttention manager block_size should be multiplied by dcp
        # Mamba manager block_size should remain unchanged
        for manager in coordinator.single_type_managers:
            spec = manager.kv_cache_spec
            if isinstance(spec, FullAttentionSpec):
                expected = BLOCK_SIZE * 2  # dcp_world_size = 2
                assert manager.block_size == expected, (
                    f"FullAttention block_size should be {expected}, "
                    f"got {manager.block_size}"
                )
            elif isinstance(spec, MambaSpec):
                expected = 1  # Mamba block_size unchanged
                assert manager.block_size == expected, (
                    f"Mamba block_size should be {expected}, "
                    f"got {manager.block_size}"
                )
