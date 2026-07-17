from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.v1.request import RequestStatus


def make_scheduler() -> MooncakeStoreScheduler:
    scheduler = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler._block_size = 16
    scheduler._request_trackers = {}
    scheduler._finished_save_metas = {}
    scheduler._unfinished_request_ids = set()
    scheduler._unfinished_requests = {}
    scheduler._preempted_req_ids = set()
    scheduler.load_specs = {}
    return scheduler


def make_aborted_request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-1",
        status=RequestStatus.FINISHED_ABORTED,
        num_computed_tokens=35,
        block_hashes=[object(), object()],
        all_token_ids=list(range(35)),
    )


def test_aborted_request_queues_complete_blocks_and_delays_free(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_KV_REUSE", "1")
    scheduler = make_scheduler()

    delay_free, params = scheduler.request_finished(
        make_aborted_request(), ([10, 11],)
    )

    assert delay_free is True
    assert params is None
    meta = scheduler._finished_save_metas["req-1"]
    assert meta.req_id == "req-1"
    assert meta.token_len_chunk == 32
    assert meta.block_ids == ([10, 11],)
    assert meta.token_ids == list(range(32))
    assert meta.is_last_chunk is True


def test_aborted_request_without_reuse_does_not_delay_free(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_MOONCAKE_ABORT_KV_REUSE", raising=False)
    scheduler = make_scheduler()

    delay_free, params = scheduler.request_finished(
        make_aborted_request(), ([10, 11],)
    )

    assert delay_free is False
    assert params is None
    assert scheduler._finished_save_metas == {}


def test_aborted_final_save_is_dispatched_in_finished_only_step(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_KV_REUSE", "1")
    scheduler = make_scheduler()
    scheduler.request_finished(make_aborted_request(), ([10, 11],))
    output = SimpleNamespace(
        finished_req_ids={"req-1"},
        preempted_req_ids=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
    )

    meta = scheduler.build_connector_meta(output)

    assert [request.req_id for request in meta.requests] == ["req-1"]
    assert meta.requests[0].token_len_chunk == 32
    assert scheduler._finished_save_metas == {}
