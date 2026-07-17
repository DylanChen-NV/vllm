#!/usr/bin/env python3
import argparse
import py_compile
from pathlib import Path


def replace_once(text: str, old: str, new: str, marker: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1), True
    raise RuntimeError(f"expected one {marker!r} target, found {count}")


def replace_all_exact(
    text: str, old: str, new: str, expected: int, marker: str
) -> tuple[str, bool]:
    count = text.count(old)
    if count == expected:
        return text.replace(old, new), True
    if count == 0 and text.count(new) == expected:
        return text, False
    raise RuntimeError(f"expected {expected} {marker!r} targets, found {count}")


def patch_scheduler(path: Path) -> bool:
    text = path.read_text()
    original_text = text
    changed = False
    text = text.replace(
        "[h.hex() for h in request.block_hashes[:3]]",
        "[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[:3]]",
    )
    text = text.replace(
        "[h.hex() for h in request.block_hashes[-3:]]",
        "[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[-3:]]",
    )
    changed = text != original_text
    lookup_trace_old = (
        "            hit_tokens=num_external_hit_tokens,\n"
        "        )\n"
    )
    lookup_trace_new = (
        "            hit_tokens=num_external_hit_tokens,\n"
        "            block_hash_count=len(request.block_hashes),\n"
        "            block_hash_head=[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[:3]],\n"
        "            block_hash_tail=[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[-3:]],\n"
        "        )\n"
    )
    if lookup_trace_old in text:
        text = text.replace(lookup_trace_old, lookup_trace_new, 1)
        changed = True

    final_trace_old = (
        "                total_blocks=total_blocks,\n"
        "            )\n"
    )
    final_trace_new = (
        "                total_blocks=total_blocks,\n"
        "                block_hash_count=len(request.block_hashes),\n"
        "                block_hash_head=[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[:3]],\n"
        "                block_hash_tail=[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[-3:]],\n"
        "            )\n"
    )
    if final_trace_old in text:
        text = text.replace(final_trace_old, final_trace_new, 1)
        changed = True

    replacements = [
        (
            "from typing import Any\n",
            "import json\nimport os\nimport time\nfrom typing import Any\n",
            "def _abort_kv_event(",
        ),
        (
            "from vllm.v1.request import Request\n\nlogger = init_logger(__name__)\n",
            "from vllm.v1.request import Request, RequestStatus\n\n"
            "logger = init_logger(__name__)\n\n\n"
            "def _abort_kv_event(phase: str, **fields: Any) -> None:\n"
            "    if os.getenv(\"VLLM_MOONCAKE_ABORT_KV_TRACE\", \"0\") != \"1\":\n"
            "        return\n"
            "    payload = {\"phase\": phase, \"wall_ns\": time.time_ns(), **fields}\n"
            "    logger.warning(\"MOONCAKE_ABORT_KV_EVENT %s\", json.dumps(payload, sort_keys=True))\n",
            "MOONCAKE_ABORT_KV_EVENT",
        ),
        (
            "        self._unfinished_request_ids: set[str] = set()\n",
            "        self._unfinished_request_ids: set[str] = set()\n"
            "        self._finished_save_metas: dict[str, ReqMeta] = {}\n",
            "self._finished_save_metas",
        ),
        (
            "        if token_len < self._block_size:\n"
            "            return 0, False\n\n"
            "        num_external_hit_tokens = self.client.lookup(token_len, request.block_hashes)\n",
            "        if token_len < self._block_size:\n"
            "            return 0, False\n"
            "        if os.getenv(\"VLLM_MOONCAKE_FORCE_MISS\", \"0\") == \"1\":\n"
            "            _abort_kv_event(\n"
            "                \"LOOKUP_FORCED_MISS\",\n"
            "                request_id=request.request_id,\n"
            "                token_len=token_len,\n"
            "            )\n"
            "            return 0, False\n\n"
            "        num_external_hit_tokens = self.client.lookup(token_len, request.block_hashes)\n"
            "        _abort_kv_event(\n"
            "            \"LOOKUP\",\n"
            "            request_id=request.request_id,\n"
            "            token_len=token_len,\n"
            "            hit_tokens=num_external_hit_tokens,\n"
            "            block_hash_count=len(request.block_hashes),\n"
            "            block_hash_head=[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[:3]],\n"
            "            block_hash_tail=[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[-3:]],\n"
            "        )\n",
            "LOOKUP_FORCED_MISS",
        ),
        (
            "        force_skip_save = self.kv_role == \"kv_consumer\"\n\n"
            "        for finished_req_id in scheduler_output.finished_req_ids:\n",
            "        force_skip_save = self.kv_role == \"kv_consumer\"\n"
            "        preempted_ids = scheduler_output.preempted_req_ids or set()\n"
            "        meta = MooncakeStoreConnectorMetadata(\n"
            "            self._unfinished_request_ids,\n"
            "            preempted_ids,\n"
            "        )\n\n"
            "        for finished_req_id in scheduler_output.finished_req_ids:\n"
            "            final_save_meta = self._finished_save_metas.pop(finished_req_id, None)\n"
            "            if final_save_meta is not None:\n"
            "                meta.add_request(final_save_meta)\n"
            "                _abort_kv_event(\n"
            "                    \"FINAL_SAVE_DISPATCHED\",\n"
            "                    request_id=finished_req_id,\n"
            "                    saved_tokens=final_save_meta.token_len_chunk,\n"
            "                    total_blocks=sum(\n"
            "                        len(group) for group in final_save_meta.block_ids\n"
            "                    ),\n"
            "                )\n",
            "final_save_meta = self._finished_save_metas.pop",
        ),
        (
            "        preempted_ids = scheduler_output.preempted_req_ids or set()\n"
            "        self._preempted_req_ids.update(preempted_ids)\n",
            "        self._preempted_req_ids.update(preempted_ids)\n",
            "final-save-meta-preempted-cleanup",
        ),
        (
            "        meta = MooncakeStoreConnectorMetadata(\n"
            "            self._unfinished_request_ids,\n"
            "            preempted_ids,\n"
            "        )\n\n"
            "        # Handle new requests\n",
            "        # Handle new requests\n",
            "final-save-meta-created-before-cleanup",
        ),
        (
            "        if self.kv_role == \"kv_consumer\":\n"
            "            return False, None\n"
            "        tracker = self._request_trackers.get(request.request_id)\n",
            "        if self.kv_role == \"kv_consumer\":\n"
            "            return False, None\n"
            "        if (\n"
            "            request.status == RequestStatus.FINISHED_ABORTED\n"
            "            and os.getenv(\"VLLM_MOONCAKE_ABORT_KV_REUSE\", \"0\") == \"1\"\n"
            "        ):\n"
            "            saved_tokens = (\n"
            "                request.num_computed_tokens // self._block_size * self._block_size\n"
            "            )\n"
            "            if saved_tokens <= 0 or not request.block_hashes:\n"
            "                _abort_kv_event(\n"
            "                    \"FINAL_SAVE_SKIPPED\",\n"
            "                    request_id=request.request_id,\n"
            "                    num_computed_tokens=request.num_computed_tokens,\n"
            "                    reason=\"no_complete_block\",\n"
            "                )\n"
            "                return False, None\n"
            "            final_tracker = RequestTracker(\n"
            "                req_id=request.request_id,\n"
            "                token_len=saved_tokens,\n"
            "                allocated_block_ids=tuple(group.copy() for group in block_ids),\n"
            "                num_saved_tokens=0,\n"
            "                token_ids=list(request.all_token_ids[:saved_tokens]),\n"
            "                prefill_end_tokens=saved_tokens,\n"
            "            )\n"
            "            final_save_meta = ReqMeta.from_request_tracker(\n"
            "                final_tracker,\n"
            "                self._block_size,\n"
            "                block_hashes=request.block_hashes,\n"
            "                is_last_chunk=True,\n"
            "            )\n"
            "            if final_save_meta is None:\n"
            "                return False, None\n"
            "            self._finished_save_metas[request.request_id] = final_save_meta\n"
            "            total_blocks = sum(len(group) for group in block_ids)\n"
            "            _abort_kv_event(\n"
            "                \"FINAL_SAVE_QUEUED\",\n"
            "                request_id=request.request_id,\n"
            "                num_computed_tokens=request.num_computed_tokens,\n"
            "                saved_tokens=saved_tokens,\n"
            "                total_blocks=total_blocks,\n"
            "                block_hash_count=len(request.block_hashes),\n"
            "                block_hash_head=[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[:3]],\n"
            "                block_hash_tail=[h.hex() if hasattr(h, 'hex') else repr(h) for h in request.block_hashes[-3:]],\n"
            "            )\n"
            "            return total_blocks > 0, None\n"
            "        tracker = self._request_trackers.get(request.request_id)\n",
            "FINAL_SAVE_QUEUED",
        ),
    ]
    cleanup_markers = {
        "final-save-meta-preempted-cleanup",
        "final-save-meta-created-before-cleanup",
    }
    for old, new, marker in replacements:
        if marker in cleanup_markers:
            count = text.count(old)
            if count == 1:
                text = text.replace(old, new, 1)
                changed = True
                continue
            if (
                count == 0
                and "final_save_meta = self._finished_save_metas.pop" in text
            ):
                continue
            raise RuntimeError(f"expected one {marker!r} target, found {count}")
        text, did_change = replace_once(text, old, new, marker)
        changed |= did_change
    if changed:
        path.write_text(text)
    return changed


def patch_core(path: Path) -> bool:
    text = path.read_text()
    changed = False
    replacements = [
        (
            '    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:\n',
            '    def sleep(\n'
            '        self,\n'
            '        level: int = 1,\n'
            '        mode: PauseMode = "abort",\n'
            '        reset_connector: bool = True,\n'
            '    ) -> None | Future:\n',
            "    def sleep(\n        self,\n        level: int = 1,",
        ),
        (
            "        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)\n",
            "        pause_future = self.pause_scheduler(\n"
            "            mode=mode,\n"
            "            clear_cache=clear_prefix_cache,\n"
            "            reset_connector=reset_connector,\n"
            "        )\n",
            "reset_connector=reset_connector",
        ),
        (
            "    def pause_scheduler(\n"
            "        self, mode: PauseMode = \"abort\", clear_cache: bool = True\n"
            "    ) -> Future | None:\n",
            "    def pause_scheduler(\n"
            "        self,\n"
            "        mode: PauseMode = \"abort\",\n"
            "        clear_cache: bool = True,\n"
            "        reset_connector: bool = True,\n"
            "    ) -> Future | None:\n",
            "pause-scheduler-reset-connector",
        ),
        (
            "        if clear_cache:\n"
            "            self._reset_caches()\n\n"
            "        return None\n",
            "        if clear_cache:\n"
            "            self._reset_caches(reset_connector=reset_connector)\n\n"
            "        return None\n",
            "self._reset_caches(reset_connector=reset_connector)",
        ),
    ]
    for old, new, marker in replacements:
        if marker == "pause-scheduler-reset-connector":
            text, did_change = replace_all_exact(text, old, new, 2, marker)
        elif marker == "self._reset_caches(reset_connector=reset_connector)":
            text, did_change = replace_all_exact(text, old, new, 1, marker)
        else:
            text, did_change = replace_once(text, old, new, marker)
        changed |= did_change

    callback_old = "            if clear_cache:\n                engine._reset_caches()\n"
    callback_new = (
        "            if clear_cache:\n"
        "                engine._reset_caches(reset_connector=reset_connector)\n"
    )
    text, did_change = replace_once(
        text, callback_old, callback_new, "DP idle reset_connector"
    )
    changed |= did_change

    immediate_old = (
        "        if self._pause_complete():\n"
        "            if clear_cache:\n"
        "                self._reset_caches()\n"
        "            return None\n"
    )
    immediate_new = (
        "        if self._pause_complete():\n"
        "            if clear_cache:\n"
        "                self._reset_caches(reset_connector=reset_connector)\n"
        "            return None\n"
    )
    text, did_change = replace_once(
        text, immediate_old, immediate_new, "DP immediate reset_connector"
    )
    changed |= did_change
    if changed:
        path.write_text(text)
    return changed


def patch_async_llm(path: Path) -> bool:
    text = path.read_text()
    old = (
        '    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:\n'
        "        if level >= 1:\n"
        "            await self.renderer.clear_mm_cache_async()\n"
        "        await self.engine_core.sleep_async(level, mode)\n"
    )
    new = (
        "    async def sleep(\n"
        "        self,\n"
        "        level: int = 1,\n"
        '        mode: PauseMode = "abort",\n'
        "        reset_connector: bool = True,\n"
        "    ) -> None:\n"
        "        if level >= 1:\n"
        "            await self.renderer.clear_mm_cache_async()\n"
        "        await self.engine_core.sleep_async(level, mode, reset_connector)\n"
    )
    text, changed = replace_once(
        text, old, new, "await self.engine_core.sleep_async(level, mode, reset_connector)"
    )
    if changed:
        path.write_text(text)
    return changed


def patch_core_client(path: Path) -> bool:
    text = path.read_text()
    changed = False
    replacements = [
        (
            '    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:\n'
            '        raise NotImplementedError\n',
            '    async def sleep_async(\n'
            '        self,\n'
            '        level: int = 1,\n'
            '        mode: PauseMode = "abort",\n'
            '        reset_connector: bool = True,\n'
            '    ) -> None:\n'
            '        raise NotImplementedError\n',
            "core-client-interface-reset-connector",
        ),
        (
            '    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:\n'
            '        await self.call_utility_async("sleep", level, mode)\n',
            '    async def sleep_async(\n'
            '        self,\n'
            '        level: int = 1,\n'
            '        mode: PauseMode = "abort",\n'
            '        reset_connector: bool = True,\n'
            '    ) -> None:\n'
            '        await self.call_utility_async("sleep", level, mode, reset_connector)\n',
            "core-client-utility-reset-connector",
        ),
    ]
    for old, new, marker in replacements:
        text, did_change = replace_once(text, old, new, marker)
        changed |= did_change
    if changed:
        path.write_text(text)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/vllm"))
    args = parser.parse_args()

    files = {
        "scheduler": args.root / "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py",
        "core": args.root / "vllm/v1/engine/core.py",
        "core_client": args.root / "vllm/v1/engine/core_client.py",
        "async_llm": args.root / "vllm/v1/engine/async_llm.py",
    }
    changed = {
        "scheduler": patch_scheduler(files["scheduler"]),
        "core": patch_core(files["core"]),
        "core_client": patch_core_client(files["core_client"]),
        "async_llm": patch_async_llm(files["async_llm"]),
    }
    for path in files.values():
        py_compile.compile(str(path), doraise=True)
    print(f"vllm_mooncake_abort_reuse_patch={changed}")


if __name__ == "__main__":
    main()
