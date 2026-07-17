import inspect

import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker as mooncake_worker


def test_mooncake_tcp_uses_cpu_staging() -> None:
    source = inspect.getsource(mooncake_worker)
    assert "_mooncake_tcp_staged_batch_put" in source
    assert "_mooncake_tcp_staged_batch_get" in source
    assert "Mooncake TCP staging D2H failed" in source
    assert "Mooncake TCP staging H2D failed" in source
