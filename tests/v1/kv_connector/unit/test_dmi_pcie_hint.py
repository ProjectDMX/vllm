from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from vllm.distributed.kv_transfer import dmi_pcie_hint
from vllm.distributed.kv_transfer.kv_connector.v1 import lmcache_mp_connector
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
    LMCacheMPConnector,
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)
from vllm.v1.worker.gpu import kv_connector as gpu_kv_connector


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000


def capture_hints(monkeypatch):
    hints = []
    monkeypatch.setattr(
        dmi_pcie_hint,
        "_emit_hint",
        lambda **kwargs: hints.append(kwargs),
    )
    return hints


def test_d2h_hint_lease_emits_bounded_begin_renew_and_end(monkeypatch):
    calls = capture_hints(monkeypatch)
    clock = FakeClock()
    lease = dmi_pcie_hint.D2HHintLease(
        "lmcache_store",
        renew_interval_ns=100_000_000,
        clock_ns=clock,
    )

    lease.start(est_bytes=1024)
    lease.start(est_bytes=2048)
    clock.advance_ms(99)
    lease.renew(est_bytes=4096)

    assert lease.active is True
    assert len(calls) == 1
    assert calls[0] == {
        "direction": "D2H",
        "source": "lmcache_store",
        "est_bytes": 1024,
        "valid_until_ns": 0,
    }

    clock.advance_ms(1)
    lease.renew(est_bytes=4096)
    lease.finish()
    lease.finish()

    assert lease.active is False
    assert len(calls) == 3
    assert calls[1]["est_bytes"] == 4096
    assert calls[1]["valid_until_ns"] == 0
    assert calls[2]["valid_until_ns"] == clock()


def test_d2h_hint_lease_is_fail_open_when_bridge_raises(monkeypatch):
    def raise_on_emit(**_kwargs):
        raise RuntimeError("synthetic bridge failure")

    monkeypatch.setattr(dmi_pcie_hint, "_emit_hint", raise_on_emit)
    lease = dmi_pcie_hint.D2HHintLease("lmcache_store")

    lease.start()
    assert lease.active is True

    lease.finish()
    assert lease.active is False


def test_connector_hint_capability_is_best_effort():
    class Managed:
        manages_dmi_pcie_hints = True

    class Broken:
        @property
        def manages_dmi_pcie_hints(self):
            raise RuntimeError("synthetic property failure")

    assert dmi_pcie_hint.connector_manages_dmi_pcie_hints(Managed()) is True
    assert dmi_pcie_hint.connector_manages_dmi_pcie_hints(object()) is False
    assert dmi_pcie_hint.connector_manages_dmi_pcie_hints(Broken()) is False


class FakeLMCacheAdapter:
    def __init__(self, connector, calls, *, raise_on_save: bool = False) -> None:
        self.connector = connector
        self.calls = calls
        self.raise_on_save = raise_on_save
        self.kv_role = "kv_both"

    def save_kv_layer(self, *_args, **_kwargs) -> None:
        self.calls.append(("save", self.connector._dmi_store_hint.active))
        if self.raise_on_save:
            raise RuntimeError("synthetic layerwise save failure")

    def wait_for_save(self) -> None:
        self.calls.append(("wait", self.connector._dmi_store_hint.active))


def make_lmcache_connector(*, layerwise: bool, can_save: bool = True):
    connector = object.__new__(LMCacheConnectorV1)
    connector._role = KVConnectorRole.WORKER
    connector._kv_transfer_config = SimpleNamespace(kv_role="kv_both")
    connector._connector_metadata = None
    connector._dmi_use_layerwise = layerwise
    connector._dmi_store_hint = dmi_pcie_hint.D2HHintLease("lmcache_store")
    calls = []
    connector._lmcache_engine = FakeLMCacheAdapter(connector, calls)
    metadata = SimpleNamespace(
        requests=[
            SimpleNamespace(
                token_ids=[1, 2, 3],
                save_spec=SimpleNamespace(
                    can_save=can_save,
                    skip_leading_tokens=0,
                ),
            )
        ]
    )
    connector.bind_connector_metadata(metadata)
    return connector, calls


def test_non_layerwise_lmcache_brackets_wait_for_save(monkeypatch):
    hints = capture_hints(monkeypatch)
    connector, calls = make_lmcache_connector(layerwise=False)

    connector.save_kv_layer("layer.0", object(), object())
    connector.wait_for_save()

    assert calls == [("save", False), ("wait", True)]
    assert [hint["valid_until_ns"] == 0 for hint in hints] == [True, False]
    assert connector._dmi_store_hint.active is False


def test_layerwise_lmcache_covers_first_layer_through_final_wait(monkeypatch):
    hints = capture_hints(monkeypatch)
    connector, calls = make_lmcache_connector(layerwise=True)

    connector.save_kv_layer("layer.0", object(), object())
    connector.save_kv_layer("layer.1", object(), object())
    connector.wait_for_save()

    assert calls == [("save", True), ("save", True), ("wait", True)]
    assert [hint["valid_until_ns"] == 0 for hint in hints] == [True, False]
    assert connector._dmi_store_hint.active is False


def test_lmcache_does_not_hint_when_step_has_no_store(monkeypatch):
    hints = capture_hints(monkeypatch)
    connector, calls = make_lmcache_connector(layerwise=True, can_save=False)

    connector.save_kv_layer("layer.0", object(), object())
    connector.wait_for_save()

    assert calls == [("save", False), ("wait", False)]
    assert hints == []


def test_layerwise_lmcache_ends_hint_when_save_raises(monkeypatch):
    hints = capture_hints(monkeypatch)
    connector, _ = make_lmcache_connector(layerwise=True)
    connector._lmcache_engine.raise_on_save = True

    try:
        connector.save_kv_layer("layer.0", object(), object())
    except RuntimeError as exc:
        assert "synthetic layerwise save failure" in str(exc)
    else:
        raise AssertionError("save_kv_layer should have propagated the adapter error")

    assert [hint["valid_until_ns"] == 0 for hint in hints] == [True, False]
    assert connector._dmi_store_hint.active is False


class FakeCudaEvent:
    def __init__(self, *, interprocess: bool) -> None:
        assert interprocess is True

    def record(self) -> None:
        pass


class FakeMPWorkerAdapter:
    def __init__(self, connector) -> None:
        self.connector = connector
        self.store_futures = {}
        self.completed = set()
        self.submission_active_states = []

    def batched_submit_store_requests(self, request_ids, _ops, _event) -> None:
        self.submission_active_states.append(connector_hint_active(self.connector))
        self.store_futures[request_ids[0]] = object()

    def get_finished(self, _finished_req_ids):
        finished = self.completed.intersection(self.store_futures)
        for request_id in finished:
            self.store_futures.pop(request_id)
        return set(finished), set()

    def shutdown(self) -> None:
        self.store_futures.clear()


def connector_hint_active(connector) -> bool:
    return connector._dmi_store_hint.active


def make_lmcache_mp_connector():
    connector = object.__new__(LMCacheMPConnector)
    connector._role = KVConnectorRole.WORKER
    connector._dmi_store_hint = dmi_pcie_hint.D2HHintLease("lmcache_mp_store")
    connector.worker_adapter = FakeMPWorkerAdapter(connector)
    metadata = LMCacheMPConnectorMetadata()
    metadata.add_request_metadata(
        LMCacheMPRequestMetadata(
            request_id="request-0",
            direction="STORE",
            op=object(),
        )
    )
    connector._connector_metadata = metadata
    return connector


def patch_cuda_event_recording(monkeypatch) -> None:
    monkeypatch.setattr(lmcache_mp_connector.torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(
        lmcache_mp_connector.torch.cuda, "current_stream", lambda: object()
    )
    monkeypatch.setattr(
        lmcache_mp_connector.torch.cuda, "stream", lambda _stream: nullcontext()
    )


def test_lmcache_mp_hint_stays_active_until_store_future_finishes(monkeypatch):
    hints = capture_hints(monkeypatch)
    patch_cuda_event_recording(monkeypatch)
    connector = make_lmcache_mp_connector()

    connector.wait_for_save()

    assert connector.worker_adapter.submission_active_states == [True]
    assert connector._dmi_store_hint.active is True
    assert [hint["valid_until_ns"] == 0 for hint in hints] == [True]

    connector.get_finished(set())
    assert connector._dmi_store_hint.active is True
    assert len(hints) == 1

    connector.worker_adapter.completed.add("request-0")
    connector.get_finished({"request-0"})

    assert connector._dmi_store_hint.active is False
    assert [hint["valid_until_ns"] == 0 for hint in hints] == [True, False]


def test_lmcache_mp_does_not_end_while_another_store_future_is_pending(monkeypatch):
    hints = capture_hints(monkeypatch)
    patch_cuda_event_recording(monkeypatch)
    connector = make_lmcache_mp_connector()

    connector.wait_for_save()
    connector.worker_adapter.store_futures["request-1"] = object()
    connector.worker_adapter.completed.add("request-0")
    connector.get_finished({"request-0"})

    assert connector._dmi_store_hint.active is True
    assert [hint["valid_until_ns"] == 0 for hint in hints] == [True]

    connector.worker_adapter.completed.add("request-1")
    connector.get_finished({"request-1"})

    assert connector._dmi_store_hint.active is False
    assert [hint["valid_until_ns"] == 0 for hint in hints] == [True, False]


class FakeWorkerConnector:
    def __init__(self, *, managed: bool) -> None:
        self.manages_dmi_pcie_hints = managed
        self.wait_calls = 0
        self.clear_calls = 0

    def wait_for_save(self) -> None:
        self.wait_calls += 1

    def get_finished(self, _finished_req_ids):
        return None, None

    def get_block_ids_with_load_errors(self):
        return set()

    def get_kv_connector_stats(self):
        return None

    def get_kv_connector_kv_cache_events(self):
        return None

    def clear_connector_metadata(self) -> None:
        self.clear_calls += 1


def run_active_worker_post_forward(*, managed: bool):
    active = object.__new__(gpu_kv_connector.ActiveKVConnector)
    active._disabled = False
    active.kv_connector = FakeWorkerConnector(managed=managed)
    scheduler_output = SimpleNamespace(finished_req_ids=set())
    output = active.post_forward(scheduler_output)
    return active.kv_connector, output


def test_worker_skips_coarse_hint_for_lifecycle_managed_connector(monkeypatch):
    hints = []
    monkeypatch.setattr(
        gpu_kv_connector,
        "_emit_dmi_pcie_hint",
        lambda **kwargs: hints.append(kwargs),
    )

    connector, output = run_active_worker_post_forward(managed=True)

    assert output is not None
    assert connector.wait_calls == 1
    assert connector.clear_calls == 1
    assert hints == []


def test_worker_keeps_coarse_hint_for_unmanaged_connector(monkeypatch):
    hints = []
    monkeypatch.setattr(
        gpu_kv_connector,
        "_emit_dmi_pcie_hint",
        lambda **kwargs: hints.append(kwargs),
    )

    connector, output = run_active_worker_post_forward(managed=False)

    assert output is not None
    assert connector.wait_calls == 1
    assert connector.clear_calls == 1
    assert [hint.get("valid_until_ns", 0) == 0 for hint in hints] == [True, False]
