import inspect

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, DPEngineCoreProc
from vllm.v1.engine.core_client import AsyncMPClient, EngineCoreClient
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import KVTransferThread


def test_sleep_connector_lifecycle_signatures() -> None:
    assert "reset_connector" in inspect.signature(AsyncLLM.sleep).parameters
    assert "reset_connector" in inspect.signature(EngineCoreClient.sleep_async).parameters
    assert "reset_connector" in inspect.signature(AsyncMPClient.sleep_async).parameters
    assert "reset_connector" in inspect.signature(EngineCore.sleep).parameters
    assert "reset_connector" in inspect.signature(EngineCore.pause_scheduler).parameters
    assert "reset_connector" in inspect.signature(DPEngineCoreProc.pause_scheduler).parameters


def test_mooncake_transfer_thread_restores_cuda_device() -> None:
    source = inspect.getsource(KVTransferThread)
    assert "self._mooncake_cuda_device_index" in source
    assert "torch.cuda.current_device()" in source
    assert "torch.cuda.set_device(self._mooncake_cuda_device_index)" in source
