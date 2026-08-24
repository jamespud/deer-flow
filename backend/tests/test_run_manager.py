"""Tests for RunManager."""

import asyncio
import logging
import re
import sqlite3
from typing import Any

import pytest
from sqlalchemy.exc import DatabaseError as SQLAlchemyDatabaseError

import deerflow.runtime.runs.manager as manager_module
from deerflow.config.run_ownership_config import RunOwnershipConfig
from deerflow.runtime import DisconnectMode, RunManager, RunStatus, ThreadOperationKind
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.runs.manager import CancelOutcome, ConflictError, PersistenceRetryPolicy, RunStartOutcome
from deerflow.runtime.runs.store.memory import MemoryRunStore

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


@pytest.fixture
def manager() -> RunManager:
    return RunManager()


class FlakyStatusRunStore(MemoryRunStore):
    """Memory run store that simulates transient SQLite status-write failures."""

    def __init__(self, *, status_failures: int) -> None:
        super().__init__()
        self.status_failures = status_failures
        self.status_update_attempts = 0

    async def update_status(self, run_id, status, *, error=None, stop_reason=None):
        self.status_update_attempts += 1
        if self.status_failures > 0:
            self.status_failures -= 1
            raise sqlite3.OperationalError("database is locked")
        return await super().update_status(run_id, status, error=error, stop_reason=stop_reason)


class MissingRowStatusRunStore(MemoryRunStore):
    """Memory run store that reports a missing row for status updates."""

    async def update_status(self, run_id, status, *, error=None, stop_reason=None):
        await super().update_status(run_id, status, error=error, stop_reason=stop_reason)
        return False


class PermanentStatusRunStore(MemoryRunStore):
    """Memory run store that simulates a permanent SQLAlchemy write failure."""

    def __init__(self) -> None:
        super().__init__()
        self.status_update_attempts = 0

    async def update_status(self, run_id, status, *, error=None, stop_reason=None):
        self.status_update_attempts += 1
        raise SQLAlchemyDatabaseError(
            "UPDATE runs SET status = :status WHERE run_id = :run_id",
            {"status": status, "run_id": run_id},
            sqlite3.DatabaseError("no such table: runs"),
        )


class FailingTakeoverRunStore(MemoryRunStore):
    """Memory run store that always fails takeover claims."""

    def __init__(self) -> None:
        super().__init__()
        self.takeover_attempts = 0

    async def claim_for_takeover(self, run_id, *, grace_seconds, error, stop_reason=None):
        self.takeover_attempts += 1
        raise sqlite3.OperationalError("database is locked")


class MissingCompletionRunStore(MemoryRunStore):
    """Memory run store that reports one missing row for completion updates."""

    def __init__(self) -> None:
        super().__init__()
        self.completion_update_attempts = 0

    async def update_run_completion(self, run_id, *, status, **kwargs):
        self.completion_update_attempts += 1
        if self.completion_update_attempts == 1:
            return False
        return await super().update_run_completion(run_id, status=status, **kwargs)


class AlwaysMissingCompletionRunStore(MemoryRunStore):
    """Memory run store that keeps reporting missing rows for completion updates."""

    def __init__(self) -> None:
        super().__init__()
        self.completion_update_attempts = 0

    async def update_run_completion(self, run_id, *, status, **kwargs):
        self.completion_update_attempts += 1
        return False


class FailingDeleteRunStore(MemoryRunStore):
    """Run store that cannot release a persisted thread-operation row."""

    async def delete(self, run_id, *, user_id=None):
        raise RuntimeError("delete failed")


class LostLeaseRunStore(MemoryRunStore):
    """Run store that reports a reservation was taken over."""

    async def update_lease(self, run_id, *, owner_worker_id, lease_expires_at):
        return False


class PausedLostLeaseRunStore(MemoryRunStore):
    """Run store whose failed renewal can be released after reservation cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self.renewal_started = asyncio.Event()
        self.finish_renewal = asyncio.Event()

    async def update_lease(self, run_id, *, owner_worker_id, lease_expires_at):
        self.renewal_started.set()
        await self.finish_renewal.wait()
        return False


class BlockingFinalizationRunStore(MemoryRunStore):
    def __init__(self) -> None:
        super().__init__()
        self.finalization_started = asyncio.Event()

    async def finalize_if_not_cancelled(self, *args, **kwargs):
        self.finalization_started.set()
        await asyncio.Event().wait()


class FailingFinalizationRunStore(MemoryRunStore):
    def __init__(self) -> None:
        super().__init__()
        self.finalization_failed = asyncio.Event()

    async def finalize_if_not_cancelled(self, *args, **kwargs):
        self.finalization_failed.set()
        raise RuntimeError("finalization unavailable")


async def _stored_statuses(store: MemoryRunStore, *run_ids: str) -> dict[str, Any]:
    rows = {}
    for run_id in run_ids:
        row = await store.get(run_id)
        rows[run_id] = row["status"] if row else None
    return rows


@pytest.mark.anyio
async def test_repeated_cancellation_during_finalization_still_fences_local_run():
    store = BlockingFinalizationRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)
    task = asyncio.create_task(manager.set_status_if_not_cancelled(record.run_id, RunStatus.success))
    await store.finalization_started.wait()
    await manager._lock.acquire()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    manager._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert record.ownership_lost is True
    assert record.abort_event.is_set() is True
    assert record.status == RunStatus.error


@pytest.mark.anyio
async def test_finalization_fence_failure_preserves_caller_cancellation(monkeypatch, caplog):
    store = BlockingFinalizationRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)

    async def fail_fence(*_args, **_kwargs):
        raise RuntimeError("fence unavailable")

    monkeypatch.setattr(manager, "_mark_ownership_lost", fail_fence)
    task = asyncio.create_task(manager.set_status_if_not_cancelled(record.run_id, RunStatus.success))
    await store.finalization_started.wait()
    task.cancel()

    with caplog.at_level(logging.ERROR), pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert "fence unavailable" in caplog.text


@pytest.mark.anyio
async def test_cancellation_during_normal_error_finalization_fence_still_fences_local_run():
    store = FailingFinalizationRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
        persistence_retry_policy=PersistenceRetryPolicy(max_attempts=1, initial_delay=0),
    )
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)
    await manager._lock.acquire()
    task = asyncio.create_task(manager.set_status_if_not_cancelled(record.run_id, RunStatus.success))
    await store.finalization_failed.wait()
    await asyncio.sleep(0)
    task.cancel()
    manager._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert record.ownership_lost is True
    assert record.abort_event.is_set() is True


@pytest.mark.anyio
async def test_hung_finalization_fence_transfers_to_background(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(manager_module, "_CANCELLATION_DRAIN_TIMEOUT_SECONDS", 0.01)
    store = BlockingFinalizationRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)
    fence_started = asyncio.Event()
    finish_fence = asyncio.Event()
    fence_calls = 0
    original_mark = manager._mark_ownership_lost

    async def hung_mark(*args, **kwargs):
        nonlocal fence_calls
        fence_calls += 1
        fence_started.set()
        await finish_fence.wait()
        await original_mark(*args, **kwargs)

    monkeypatch.setattr(manager, "_mark_ownership_lost", hung_mark)
    caller = asyncio.create_task(manager.set_status_if_not_cancelled(record.run_id, RunStatus.success))
    await store.finalization_started.wait()
    caller.cancel()
    await fence_started.wait()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=0.2)
    assert len(manager._cancellation_cleanup_tasks) == 1

    finish_fence.set()
    await asyncio.gather(*tuple(manager._cancellation_cleanup_tasks), return_exceptions=True)
    await asyncio.sleep(0)
    assert record.ownership_lost is True
    assert fence_calls == 1
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_hung_cancelled_admission_cleanup_transfers_to_background(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(manager_module, "_CANCELLATION_DRAIN_TIMEOUT_SECONDS", 0.01)
    store = MemoryRunStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)
    old_persist_started = asyncio.Event()
    release_old_persist = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_calls = 0
    original_persist = manager._persist_status
    original_close = manager._close_cancelled_admission

    async def block_old_persist(record, status, **kwargs):
        if record.run_id == old.run_id:
            old_persist_started.set()
            await release_old_persist.wait()
        return await original_persist(record, status, **kwargs)

    async def hung_close(record):
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_started.set()
        await finish_cleanup.wait()
        await original_close(record)

    monkeypatch.setattr(manager, "_persist_status", block_old_persist)
    monkeypatch.setattr(manager, "_close_cancelled_admission", hung_close)
    caller = asyncio.create_task(manager.create_or_reject("thread-1", multitask_strategy="interrupt"))
    await old_persist_started.wait()
    replacement = next(record for record in manager._runs.values() if record.run_id != old.run_id)
    caller.cancel()
    release_old_persist.set()
    await cleanup_started.wait()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=0.2)
    assert len(manager._cancellation_cleanup_tasks) == 1

    finish_cleanup.set()
    await asyncio.gather(*tuple(manager._cancellation_cleanup_tasks), return_exceptions=True)
    await asyncio.sleep(0)
    assert replacement.status == RunStatus.interrupted
    assert cleanup_calls == 1
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
@pytest.mark.parametrize("cleanup_error", [RuntimeError("fence unavailable"), asyncio.CancelledError()], ids=["failure", "self-cancel"])
async def test_late_cancellation_cleanup_outcome_is_consumed_without_replacing_cancel(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cleanup_error: BaseException,
):
    monkeypatch.setattr(manager_module, "_CANCELLATION_DRAIN_TIMEOUT_SECONDS", 0.01)
    store = BlockingFinalizationRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)
    finish_cleanup = asyncio.Event()

    async def late_cleanup(*_args, **_kwargs):
        await finish_cleanup.wait()
        raise cleanup_error

    monkeypatch.setattr(manager, "_mark_ownership_lost", late_cleanup)
    caller = asyncio.create_task(manager.set_status_if_not_cancelled(record.run_id, RunStatus.success))
    await store.finalization_started.wait()
    caller.cancel()

    with caplog.at_level(logging.ERROR), pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=0.2)
    assert len(manager._cancellation_cleanup_tasks) == 1

    finish_cleanup.set()
    cleanup = next(iter(manager._cancellation_cleanup_tasks))
    await asyncio.gather(cleanup, return_exceptions=True)
    await asyncio.sleep(0)
    assert not manager._cancellation_cleanup_tasks
    assert caplog.text.count(f"run_id={record.run_id}") == 1
    assert "Run cancellation cleanup" in caplog.text


@pytest.mark.anyio
async def test_repeated_cancellation_uses_one_cleanup_task_and_absolute_deadline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(manager_module, "_CANCELLATION_DRAIN_TIMEOUT_SECONDS", 0.03)
    store = BlockingFinalizationRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)
    finish_cleanup = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_calls = 0
    observed = []
    original_wait = manager_module.wait_for_task_until

    async def record_wait(task, *, deadline):
        observed.append((task, deadline))
        return await original_wait(task, deadline=deadline)

    async def late_cleanup(*_args, **_kwargs):
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_started.set()
        await finish_cleanup.wait()

    monkeypatch.setattr(manager_module, "wait_for_task_until", record_wait)
    monkeypatch.setattr(manager, "_mark_ownership_lost", late_cleanup)
    caller = asyncio.create_task(manager.set_status_if_not_cancelled(record.run_id, RunStatus.success))
    await store.finalization_started.wait()
    caller.cancel()
    await cleanup_started.wait()
    await asyncio.sleep(0.005)
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=0.2)
    assert len(observed) == 1
    assert len(manager._cancellation_cleanup_tasks) == 1
    cleanup = observed[0][0]
    finish_cleanup.set()
    await asyncio.gather(cleanup, return_exceptions=True)
    await asyncio.sleep(0)
    assert cleanup_calls == 1
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_shutdown_observes_supervised_cleanup_only_within_budget(caplog: pytest.LogCaptureFixture):
    manager = RunManager()
    finish_cleanup = asyncio.Event()

    async def wait_for_cleanup() -> None:
        await finish_cleanup.wait()

    cleanup = asyncio.create_task(wait_for_cleanup())
    manager._track_cancellation_cleanup(cleanup, action="test cleanup", run_id="run-1")
    loop = asyncio.get_running_loop()
    started = loop.time()

    with caplog.at_level(logging.WARNING):
        await manager.shutdown(timeout=0.01)

    assert loop.time() - started < 0.2
    assert cleanup.done() is False
    assert cleanup in manager._cancellation_cleanup_tasks
    assert "cancellation cleanup task" in caplog.text
    finish_cleanup.set()
    await cleanup
    await asyncio.sleep(0)
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_shutdown_observes_cleanup_registered_by_cancelled_run_before_deadline():
    manager = RunManager()
    cleanup_finished = asyncio.Event()
    cleanup_registered = asyncio.Event()
    release_producer = manager._begin_cancellation_cleanup_producer()

    async def run() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(cleanup_finished.wait())
            manager._track_cancellation_cleanup(cleanup, action="dynamic cleanup", run_id="run-1")
            cleanup_registered.set()
        finally:
            release_producer()

    record = await manager.create("thread-1")
    record.task = asyncio.create_task(run())
    shutdown_task = asyncio.create_task(manager.shutdown(timeout=0.2))
    await cleanup_registered.wait()
    assert shutdown_task.done() is False
    cleanup_finished.set()
    await shutdown_task
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_shutdown_keeps_dynamic_cleanup_supervised_after_deadline(caplog: pytest.LogCaptureFixture):
    manager = RunManager()
    cleanup_registered = asyncio.Event()
    cleanup_finished = asyncio.Event()
    release_producer = manager._begin_cancellation_cleanup_producer()

    async def run() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(cleanup_finished.wait())
            manager._track_cancellation_cleanup(cleanup, action="stalled dynamic cleanup", run_id="run-1")
            cleanup_registered.set()
        finally:
            release_producer()

    record = await manager.create("thread-1")
    record.task = asyncio.create_task(run())
    with caplog.at_level(logging.WARNING):
        shutdown_task = asyncio.create_task(manager.shutdown(timeout=0.05))
        await cleanup_registered.wait()
        await shutdown_task

    assert any("cancellation cleanup task" in message for message in caplog.messages)
    cleanup = next(iter(manager._cancellation_cleanup_tasks))
    assert cleanup.done() is False
    cleanup_finished.set()
    await cleanup
    await asyncio.sleep(0)
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_shutdown_waits_for_delayed_run_completion_cleanup_registration():
    manager = RunManager()
    cleanup_finished = asyncio.Event()
    cleanup_registered = asyncio.Event()
    loop = asyncio.get_running_loop()
    release_producer = manager._begin_cancellation_cleanup_producer()

    async def register_cleanup() -> None:
        await asyncio.sleep(0)
        cleanup = asyncio.create_task(cleanup_finished.wait())
        manager._track_cancellation_cleanup(cleanup, action="delayed cleanup", run_id="run-1")
        cleanup_registered.set()
        release_producer()

    async def run() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    record = await manager.create("thread-1")
    record.task = asyncio.create_task(run())

    def schedule_register() -> None:
        asyncio.create_task(register_cleanup())

    def second_barrier() -> None:
        loop.call_soon(schedule_register)

    def first_barrier() -> None:
        loop.call_soon(second_barrier)

    def after_run(_completed: asyncio.Future[None]) -> None:
        loop.call_soon(first_barrier)

    record.task.add_done_callback(after_run)
    shutdown_task = asyncio.create_task(manager.shutdown(timeout=0.05))
    await asyncio.wait_for(cleanup_registered.wait(), timeout=0.2)
    assert shutdown_task.done() is False
    cleanup = next(iter(manager._cancellation_cleanup_tasks))
    cleanup_finished.set()
    await shutdown_task
    await asyncio.sleep(0)
    assert cleanup.done() is True
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_shutdown_keeps_delayed_stalled_cleanup_supervised_until_deadline(caplog: pytest.LogCaptureFixture):
    manager = RunManager()
    cleanup_finished = asyncio.Event()
    cleanup_registered = asyncio.Event()
    loop = asyncio.get_running_loop()
    release_producer = manager._begin_cancellation_cleanup_producer()

    async def register_cleanup() -> None:
        await asyncio.sleep(0)
        cleanup = asyncio.create_task(cleanup_finished.wait())
        manager._track_cancellation_cleanup(cleanup, action="delayed stalled cleanup", run_id="run-1")
        cleanup_registered.set()
        release_producer()

    async def run() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    record = await manager.create("thread-1")
    record.task = asyncio.create_task(run())

    def schedule_register() -> None:
        asyncio.create_task(register_cleanup())

    def second_barrier() -> None:
        loop.call_soon(schedule_register)

    def first_barrier() -> None:
        loop.call_soon(second_barrier)

    def after_run(_completed: asyncio.Future[None]) -> None:
        loop.call_soon(first_barrier)

    record.task.add_done_callback(after_run)
    with caplog.at_level(logging.WARNING):
        shutdown_task = asyncio.create_task(manager.shutdown(timeout=0.05))
        await asyncio.wait_for(cleanup_registered.wait(), timeout=0.2)
        assert shutdown_task.done() is False
        await shutdown_task

    cleanup = next(iter(manager._cancellation_cleanup_tasks))
    assert cleanup.done() is False
    assert any("cancellation cleanup task" in message for message in caplog.messages)
    cleanup_finished.set()
    await cleanup
    await asyncio.sleep(0)
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_shutdown_waits_for_registered_cleanup_producer_through_deep_callback_chain():
    manager = RunManager()
    cleanup_finished = asyncio.Event()
    cleanup_registered = asyncio.Event()
    loop = asyncio.get_running_loop()
    release_producer = manager._begin_cancellation_cleanup_producer()

    def callback(depth: int) -> None:
        if depth < 100:
            loop.call_soon(callback, depth + 1)
            return
        cleanup = asyncio.create_task(cleanup_finished.wait())
        manager._track_cancellation_cleanup(cleanup, action="deep delayed cleanup", run_id="run-1")
        cleanup_registered.set()
        release_producer()

    loop.call_soon(callback, 0)
    shutdown_task = asyncio.create_task(manager.shutdown(timeout=0.05))
    await asyncio.wait_for(cleanup_registered.wait(), timeout=0.2)
    assert shutdown_task.done() is False
    cleanup_finished.set()
    await shutdown_task
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_shutdown_waits_for_cleanup_producer_that_finishes_without_registration():
    manager = RunManager()
    producer_released = asyncio.Event()
    loop = asyncio.get_running_loop()
    release_producer = manager._begin_cancellation_cleanup_producer()

    def callback(depth: int) -> None:
        if depth < 100:
            loop.call_soon(callback, depth + 1)
            return
        release_producer()
        producer_released.set()

    loop.call_soon(callback, 0)
    shutdown_task = asyncio.create_task(manager.shutdown(timeout=0.2))
    await asyncio.wait_for(producer_released.wait(), timeout=0.2)
    await shutdown_task
    assert not manager._cancellation_cleanup_tasks


@pytest.mark.anyio
async def test_shutdown_cancellation_propagates_while_waiting_for_cleanup_producer():
    manager = RunManager()
    release_producer = manager._begin_cancellation_cleanup_producer()
    manager._cancellation_cleanup_state_changed.clear()
    shutdown_task = asyncio.create_task(manager.shutdown(timeout=1.0))
    await asyncio.sleep(0)
    assert shutdown_task.done() is False

    shutdown_task.cancel("shutdown cancelled")
    with pytest.raises(asyncio.CancelledError) as caught:
        await shutdown_task
    assert caught.value.args == ("shutdown cancelled",)
    assert manager._cancellation_cleanup_producers
    release_producer()
    await asyncio.sleep(0)


@pytest.mark.anyio
async def test_shutdown_bounds_initial_manager_lock_wait(caplog: pytest.LogCaptureFixture):
    manager = RunManager()
    await manager._lock.acquire()
    started = asyncio.get_running_loop().time()

    try:
        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(manager.shutdown(timeout=0.01), timeout=0.2)
    finally:
        manager._lock.release()

    assert asyncio.get_running_loop().time() - started < 0.2
    assert "could not acquire manager lock" in caplog.text
    await asyncio.wait_for(manager._lock.acquire(), timeout=0.1)
    manager._lock.release()
    assert not getattr(manager._lock, "_waiters", ())


@pytest.mark.anyio
async def test_shutdown_bounds_post_wait_manager_lock_wait(caplog: pytest.LogCaptureFixture):
    manager = RunManager()
    run_started = asyncio.Event()
    lock_held = asyncio.Event()
    release_lock = asyncio.Event()

    async def run() -> None:
        run_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await manager._lock.acquire()
            lock_held.set()
            await release_lock.wait()
            manager._lock.release()

    record = await manager.create("thread-1")
    record.task = asyncio.create_task(run())
    await run_started.wait()
    shutdown_task = asyncio.create_task(manager.shutdown(timeout=0.03))

    try:
        await asyncio.wait_for(lock_held.wait(), timeout=0.2)
        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(shutdown_task, timeout=0.2)
    finally:
        release_lock.set()
        await record.task

    assert "could not acquire manager lock" in caplog.text
    await asyncio.wait_for(manager._lock.acquire(), timeout=0.1)
    manager._lock.release()
    assert not getattr(manager._lock, "_waiters", ())


@pytest.mark.anyio
async def test_shutdown_observes_completed_run_after_post_wait_lock_timeout():
    store = MemoryRunStore()
    manager = RunManager(store=store)
    run_started = asyncio.Event()
    holder_started = asyncio.Event()
    release_holder = asyncio.Event()
    holder_task: asyncio.Task[None] | None = None

    async def hold_lock() -> None:
        await manager._lock.acquire()
        holder_started.set()
        await release_holder.wait()
        manager._lock.release()

    async def run() -> None:
        nonlocal holder_task
        run_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            holder_task = asyncio.create_task(hold_lock())
            await holder_started.wait()
            raise RuntimeError("run failed during shutdown")

    record = await manager.create("thread-1")
    record.task = asyncio.create_task(run())
    await run_started.wait()

    try:
        await asyncio.wait_for(manager.shutdown(timeout=0.03), timeout=0.2)
        assert record.task.done()
        assert getattr(record.task, "_log_traceback", True) is False
        assert record.status == RunStatus.pending
        stored = await store.get(record.run_id)
        assert stored is not None
        assert stored["status"] == RunStatus.pending.value
    finally:
        release_holder.set()
        if holder_task is not None:
            await holder_task


@pytest.mark.anyio
async def test_shutdown_cancellation_observes_completed_run_during_post_wait_lock():
    store = MemoryRunStore()
    manager = RunManager(store=store)
    run_started = asyncio.Event()
    holder_started = asyncio.Event()
    release_holder = asyncio.Event()
    second_lock_wait_started = asyncio.Event()
    holder_task: asyncio.Task[None] | None = None
    acquire_calls = 0
    original_acquire = manager._acquire_lock_until

    async def observe_second_lock_wait(deadline: float) -> bool:
        nonlocal acquire_calls
        acquire_calls += 1
        if acquire_calls == 2:
            second_lock_wait_started.set()
        return await original_acquire(deadline)

    async def hold_lock() -> None:
        await manager._lock.acquire()
        holder_started.set()
        await release_holder.wait()
        manager._lock.release()

    async def run() -> None:
        nonlocal holder_task
        run_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            holder_task = asyncio.create_task(hold_lock())
            await holder_started.wait()
            raise RuntimeError("run failed during shutdown cancellation")

    manager._acquire_lock_until = observe_second_lock_wait
    record = await manager.create("thread-1")
    record.task = asyncio.create_task(run())
    await run_started.wait()
    shutdown_task = asyncio.create_task(manager.shutdown(timeout=1.0))

    try:
        await asyncio.wait_for(second_lock_wait_started.wait(), timeout=0.2)
        shutdown_task.cancel("caller cancelled shutdown")
        with pytest.raises(asyncio.CancelledError) as caught:
            await shutdown_task
        assert caught.value.args == ("caller cancelled shutdown",)
        assert record.task.done()
        assert getattr(record.task, "_log_traceback", True) is False
        assert record.status == RunStatus.pending
        stored = await store.get(record.run_id)
        assert stored is not None
        assert stored["status"] == RunStatus.pending.value
    finally:
        release_holder.set()
        if holder_task is not None:
            await holder_task


@pytest.mark.anyio
@pytest.mark.parametrize("late_failure", [False, True], ids=["success", "failure"])
async def test_shutdown_supervises_late_status_persistence_without_cancelling_or_retrying(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    late_failure: bool,
):
    manager = RunManager(store=MemoryRunStore())
    run_started = asyncio.Event()
    release_persist = asyncio.Event()
    persist_started = asyncio.Event()
    persist_calls = 0

    async def run() -> None:
        run_started.set()
        await asyncio.Event().wait()

    async def stubborn_persist(_record: Any, _status: RunStatus, **_kwargs: Any) -> bool:
        nonlocal persist_calls
        persist_calls += 1
        persist_started.set()
        try:
            await release_persist.wait()
        except asyncio.CancelledError:
            raise AssertionError("shutdown must not cancel status persistence")
        if late_failure:
            raise RuntimeError("late shutdown persistence failed")
        return True

    monkeypatch.setattr(manager, "_persist_status", stubborn_persist)
    record = await manager.create("thread-1")
    record.task = asyncio.create_task(run())
    await run_started.wait()

    with caplog.at_level(logging.WARNING):
        shutdown_task = asyncio.create_task(manager.shutdown(timeout=0.03))
        await asyncio.wait_for(persist_started.wait(), timeout=0.2)
        persistence_task = next(iter(manager._shutdown_persistence_tasks))
        await asyncio.wait_for(shutdown_task, timeout=0.15)

        assert persistence_task.done() is False
        assert persistence_task in manager._shutdown_persistence_tasks
        assert persist_calls == 1
        release_persist.set()
        await persistence_task
        await asyncio.sleep(0)

    assert not manager._shutdown_persistence_tasks
    assert persist_calls == 1
    if late_failure:
        assert caplog.text.count("late shutdown persistence failed") == 1


@pytest.mark.anyio
async def test_heartbeat_does_not_schedule_orphans_after_stop_during_renewal(
    monkeypatch: pytest.MonkeyPatch,
):
    manager = RunManager(
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=5,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    renewal_started = asyncio.Event()
    renewal_cancelled = asyncio.Event()
    release_renewal = asyncio.Event()
    renewal_calls = 0
    scheduled_orphans: list[None] = []
    real_asyncio = manager_module.asyncio

    class FastAsyncio:
        def __getattr__(self, name: str) -> Any:
            return getattr(real_asyncio, name)

        async def wait_for(self, awaitable: Any, timeout: float) -> Any:
            if timeout == 1:
                awaitable.close()
                raise TimeoutError
            return await real_asyncio.wait_for(awaitable, timeout)

    async def renew_leases() -> None:
        nonlocal renewal_calls
        renewal_calls += 1
        if renewal_calls != 3:
            return
        renewal_started.set()
        try:
            await release_renewal.wait()
        except asyncio.CancelledError:
            renewal_cancelled.set()
            await release_renewal.wait()

    monkeypatch.setattr(manager_module, "asyncio", FastAsyncio())
    monkeypatch.setattr(manager, "_renew_leases", renew_leases)
    monkeypatch.setattr(
        manager,
        "_schedule_orphan_reconciliation",
        lambda: scheduled_orphans.append(None),
    )

    await manager.start_heartbeat()
    task = manager._heartbeat_task
    assert task is not None
    await renewal_started.wait()

    await manager.stop_heartbeat(timeout=0.01)
    await asyncio.wait_for(renewal_cancelled.wait(), timeout=0.2)
    release_renewal.set()
    await task
    await asyncio.sleep(0)

    assert renewal_calls == 3
    assert scheduled_orphans == []


@pytest.mark.anyio
async def test_stop_heartbeat_stubborn_task_keeps_one_background_owner_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    manager = RunManager(
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    heartbeat_started = asyncio.Event()
    heartbeat_cancelled = asyncio.Event()
    finish_heartbeat = asyncio.Event()
    cancel_calls = 0

    async def stubborn_heartbeat() -> None:
        nonlocal cancel_calls
        heartbeat_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_calls += 1
            heartbeat_cancelled.set()
            await finish_heartbeat.wait()
            raise RuntimeError("late heartbeat failure")

    monkeypatch.setattr(manager, "_heartbeat_loop", stubborn_heartbeat)
    await manager.start_heartbeat()
    task = manager._heartbeat_task
    assert task is not None
    await heartbeat_started.wait()
    started = asyncio.get_running_loop().time()

    with caplog.at_level(logging.WARNING):
        await manager.stop_heartbeat(timeout=0.01)
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.2
    await asyncio.wait_for(heartbeat_cancelled.wait(), timeout=0.2)
    assert heartbeat_cancelled.is_set()
    assert cancel_calls == 1
    assert manager._heartbeat_task is task
    assert "background" in caplog.text

    started = asyncio.get_running_loop().time()
    await manager.stop_heartbeat(timeout=0.01)
    assert asyncio.get_running_loop().time() - started < 0.05
    assert cancel_calls == 1

    finish_heartbeat.set()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert manager._heartbeat_task is None
    assert "late heartbeat failure" in caplog.text


@pytest.mark.anyio
async def test_orphan_recovery_stubborn_task_keeps_one_background_owner_after_deadline(caplog: pytest.LogCaptureFixture):
    manager = RunManager()
    recovery_started = asyncio.Event()
    recovery_cancelled = asyncio.Event()
    finish_recovery = asyncio.Event()
    cancel_calls = 0

    async def stubborn_recovery() -> None:
        nonlocal cancel_calls
        recovery_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_calls += 1
            recovery_cancelled.set()
            await finish_recovery.wait()
            raise RuntimeError("late orphan recovery failure")

    task = asyncio.create_task(stubborn_recovery())
    manager._orphan_recovery_task = task
    task.add_done_callback(manager._orphan_reconciliation_done)
    await recovery_started.wait()
    started = asyncio.get_running_loop().time()

    with caplog.at_level(logging.WARNING):
        await manager._drain_orphan_recovery_task(timeout=0.01)
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.2
    await asyncio.wait_for(recovery_cancelled.wait(), timeout=0.2)
    assert recovery_cancelled.is_set()
    assert cancel_calls == 1
    assert manager._orphan_recovery_task is task
    assert "background" in caplog.text

    started = asyncio.get_running_loop().time()
    await manager._drain_orphan_recovery_task(timeout=0.01)
    assert asyncio.get_running_loop().time() - started < 0.05
    assert cancel_calls == 1

    finish_recovery.set()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert manager._orphan_recovery_task is None
    assert "late orphan recovery failure" in caplog.text


@pytest.mark.anyio
async def test_stop_heartbeat_consumes_completed_failure_once(caplog: pytest.LogCaptureFixture):
    manager = RunManager(
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )

    async def failed_heartbeat() -> None:
        raise RuntimeError("heartbeat failure")

    manager._heartbeat_loop = failed_heartbeat
    await manager.start_heartbeat()
    await asyncio.sleep(0)

    with caplog.at_level(logging.WARNING):
        await manager.stop_heartbeat(timeout=0)
        await asyncio.sleep(0)

    assert caplog.messages.count("Run lease heartbeat failed; its task has stopped") == 1
    assert manager._heartbeat_task is None


@pytest.mark.anyio
async def test_reservation_delete_failure_preserves_body_error_and_clears_local_record(caplog):
    store = FailingDeleteRunStore()
    manager = RunManager(
        store=store,
        persistence_retry_policy=PersistenceRetryPolicy(max_attempts=1, initial_delay=0),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="body failed"):
        async with manager.reserve_thread_operation(
            "thread-1",
            kind=ThreadOperationKind.checkpoint_write,
        ):
            raise ValueError("body failed")

    assert not await manager.has_inflight("thread-1")
    assert manager._runs == {}
    assert manager._runs_by_thread == {}
    assert len(await store.list_inflight()) == 1
    assert "leaving it for orphan reconciliation" in caplog.text


@pytest.mark.anyio
async def test_reservation_lease_loss_surfaces_as_conflict_after_cancelling_body():
    store = LostLeaseRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    entered = asyncio.Event()

    async def hold_reservation() -> None:
        async with manager.reserve_thread_operation(
            "thread-1",
            kind=ThreadOperationKind.checkpoint_write,
        ):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_reservation())
    await entered.wait()

    await manager._renew_leases()

    with pytest.raises(ConflictError, match="reservation lease was lost"):
        await task
    assert not await manager.has_inflight("thread-1")
    assert await store.list_inflight() == []


@pytest.mark.anyio
async def test_reservation_cancelled_while_attaching_task_is_released(monkeypatch):
    store = MemoryRunStore()
    manager = RunManager(store=store)
    admitted = asyncio.Event()
    return_from_admission = asyncio.Event()
    original_admit = manager._admit_thread_operation

    async def pause_after_admission(*args, **kwargs):
        record = await original_admit(*args, **kwargs)
        admitted.set()
        await return_from_admission.wait()
        return record

    monkeypatch.setattr(manager, "_admit_thread_operation", pause_after_admission)

    async def reserve() -> None:
        async with manager.reserve_thread_operation(
            "thread-1",
            kind=ThreadOperationKind.checkpoint_write,
        ):
            raise AssertionError("cancelled reservation must not enter its body")

    task = asyncio.create_task(reserve())
    await admitted.wait()
    await manager._lock.acquire()
    return_from_admission.set()
    await asyncio.sleep(0)
    task.cancel()
    manager._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not await manager.has_inflight("thread-1")
    assert manager._runs == {}
    assert manager._runs_by_thread == {}
    assert await store.list_inflight() == []


@pytest.mark.anyio
async def test_late_failed_renewal_does_not_cancel_released_reservation():
    store = PausedLostLeaseRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(
            lease_seconds=30,
            grace_seconds=10,
            heartbeat_enabled=True,
        ),
    )
    entered = asyncio.Event()
    leave_body = asyncio.Event()
    context_exited = asyncio.Event()
    finish_request = asyncio.Event()

    async def request() -> None:
        async with manager.reserve_thread_operation(
            "thread-1",
            kind=ThreadOperationKind.checkpoint_write,
        ):
            entered.set()
            await leave_body.wait()
        context_exited.set()
        await finish_request.wait()

    request_task = asyncio.create_task(request())
    await entered.wait()
    renewal_task = asyncio.create_task(manager._renew_leases())
    await store.renewal_started.wait()

    leave_body.set()
    await context_exited.wait()
    assert not await manager.has_inflight("thread-1")

    store.finish_renewal.set()
    await renewal_task
    assert not request_task.done()

    finish_request.set()
    await request_task


@pytest.mark.anyio
async def test_create_and_get(manager: RunManager):
    """Created run should be retrievable with new fields."""
    record = await manager.create(
        "thread-1",
        "lead_agent",
        metadata={"key": "val"},
        kwargs={"input": {}},
        multitask_strategy="reject",
    )
    assert record.status == RunStatus.pending
    assert record.thread_id == "thread-1"
    assert record.assistant_id == "lead_agent"
    assert record.metadata == {"key": "val"}
    assert record.kwargs == {"input": {}}
    assert record.multitask_strategy == "reject"
    assert ISO_RE.match(record.created_at)
    assert ISO_RE.match(record.updated_at)

    fetched = await manager.get(record.run_id)
    assert fetched is record


@pytest.mark.anyio
async def test_status_transitions(manager: RunManager):
    """Status should transition pending -> running -> success."""
    record = await manager.create("thread-1")
    assert record.status == RunStatus.pending

    await manager.set_status(record.run_id, RunStatus.running)
    assert record.status == RunStatus.running
    assert ISO_RE.match(record.updated_at)

    await manager.set_status(record.run_id, RunStatus.success)
    assert record.status == RunStatus.success


@pytest.mark.anyio
async def test_cancel(manager: RunManager):
    """Cancel should set abort_event and transition to interrupted."""
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)

    cancelled = await manager.cancel(record.run_id)
    assert cancelled == CancelOutcome.cancelled
    assert record.abort_event.is_set()
    assert record.status == RunStatus.interrupted


@pytest.mark.anyio
async def test_cancel_persists_interrupted_status_to_store():
    """Cancel should persist interrupted status to the backing store."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)

    cancelled = await manager.cancel(record.run_id)

    stored = await store.get(record.run_id)
    assert cancelled == CancelOutcome.cancelled
    assert stored is not None
    assert stored["status"] == "interrupted"


@pytest.mark.anyio
async def test_status_persistence_retries_transient_sqlite_lock():
    """Transient SQLite lock errors should not leave a final status stale."""
    store = FlakyStatusRunStore(status_failures=2)
    manager = RunManager(store=store)
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)

    await manager.set_status(record.run_id, RunStatus.success)

    stored = await store.get(record.run_id)
    assert stored is not None
    assert stored["status"] == "success"
    assert store.status_update_attempts >= 4


@pytest.mark.anyio
async def test_status_persistence_recreates_missing_store_row():
    """A final status update should recreate a run row if initial persistence was lost."""
    store = MissingRowStatusRunStore()
    manager = RunManager(store=store)
    record = await manager.create("thread-1")
    await store.delete(record.run_id)

    await manager.set_status(record.run_id, RunStatus.error, error="boom")

    stored = await store.get(record.run_id)
    assert stored is not None
    assert stored["status"] == "error"
    assert stored["error"] == "boom"


@pytest.mark.anyio
async def test_status_persistence_does_not_retry_permanent_sqlalchemy_errors():
    """Permanent SQLAlchemy failures should not be retried as SQLite pressure."""
    store = PermanentStatusRunStore()
    manager = RunManager(
        store=store,
        persistence_retry_policy=PersistenceRetryPolicy(max_attempts=5, initial_delay=0),
    )
    record = await manager.create("thread-1")

    await manager.set_status(record.run_id, RunStatus.error, error="boom")

    assert store.status_update_attempts == 1


@pytest.mark.anyio
async def test_try_start_respects_durable_and_racing_cancels():
    """Startup must not resurrect durable or locally racing cancels."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    record = await manager.create_or_reject("thread-1")
    await store.update_status(record.run_id, RunStatus.interrupted.value)

    assert await manager.try_start(record.run_id) == RunStartOutcome.cancelled
    assert record.status == RunStatus.interrupted
    assert (await store.get(record.run_id))["status"] == RunStatus.interrupted.value

    record = await manager.create_or_reject("thread-2")
    original_start_run = store.start_run

    async def start_then_cancel(run_id):
        updated = await original_start_run(run_id)
        await manager.cancel(record.run_id)
        return updated

    store.start_run = start_then_cancel

    assert await manager.try_start(record.run_id) == RunStartOutcome.cancelled
    assert record.status == RunStatus.interrupted
    assert (await store.get(record.run_id))["status"] == RunStatus.interrupted.value


@pytest.mark.anyio
async def test_fail_start_if_pending_marks_pending_run_error_and_persists():
    """Worker attach failures should finalize only runs still pending startup."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    record = await manager.create_or_reject("thread-1")
    error = "Failed to attach run worker: boom"

    assert await manager.fail_start_if_pending(record.run_id, error=error) is True

    stored = await store.get(record.run_id)
    assert record.status == RunStatus.error
    assert record.error == error
    assert record.abort_event.is_set()
    assert stored is not None
    assert stored["status"] == RunStatus.error.value
    assert stored["error"] == error

    running = await manager.create_or_reject("thread-2")
    assert await manager.try_start(running.run_id) == RunStartOutcome.started

    assert await manager.fail_start_if_pending(running.run_id, error="late") is False

    stored_running = await store.get(running.run_id)
    assert running.status == RunStatus.running
    assert running.error is None
    assert stored_running is not None
    assert stored_running["status"] == RunStatus.running.value
    assert stored_running["error"] is None


@pytest.mark.anyio
async def test_completion_persistence_recreates_missing_store_row():
    """Completion updates should recreate a missing row and persist final counters."""
    store = MissingCompletionRunStore()
    manager = RunManager(store=store)
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)
    await manager.set_status(record.run_id, RunStatus.success)
    await store.delete(record.run_id)

    await manager.update_run_completion(
        record.run_id,
        status="success",
        total_tokens=42,
        llm_call_count=2,
        last_ai_message="done",
    )

    stored = await store.get(record.run_id)
    assert stored is not None
    assert stored["status"] == "success"
    assert stored["total_tokens"] == 42
    assert stored["llm_call_count"] == 2
    assert stored["last_ai_message"] == "done"
    assert store.completion_update_attempts == 2


@pytest.mark.anyio
async def test_completion_persistence_warns_when_recreated_row_still_missing(caplog):
    """A second zero-row completion update after recreation should not be silent."""
    store = AlwaysMissingCompletionRunStore()
    manager = RunManager(store=store)
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.success)
    caplog.set_level(logging.WARNING, logger="deerflow.runtime.runs.manager")

    await manager.update_run_completion(record.run_id, status="success", total_tokens=42)

    assert store.completion_update_attempts == 2
    assert "affected no rows after row recreation" in caplog.text


@pytest.mark.anyio
async def test_reconcile_orphaned_inflight_runs_marks_stale_rows_error():
    """Startup recovery should turn persisted active rows into explicit errors."""
    store = MemoryRunStore()
    await store.put("pending-run", thread_id="thread-1", status="pending", created_at="2026-01-01T00:00:00+00:00")
    await store.put("running-run", thread_id="thread-1", status="running", created_at="2026-01-01T00:00:01+00:00")
    await store.put("success-run", thread_id="thread-1", status="success", created_at="2026-01-01T00:00:02+00:00")
    manager = RunManager(store=store)

    recovered = await manager.reconcile_orphaned_inflight_runs(
        error="Gateway restarted before this run reached a durable final state.",
        before="2026-01-01T00:00:02+00:00",
    )

    assert {record.run_id for record in recovered} == {"pending-run", "running-run"}
    assert await _stored_statuses(store, "pending-run", "running-run", "success-run") == {
        "pending-run": "error",
        "running-run": "error",
        "success-run": "success",
    }


@pytest.mark.anyio
async def test_reconcile_orphaned_run_backfills_delivery_after_atomic_takeover():
    """Lease recovery must durably backfill the terminal receipt exactly once."""
    store = MemoryRunStore()
    events = MemoryRunEventStore()
    await store.put("running-run", thread_id="thread-1", status="running", created_at="2026-01-01T00:00:00+00:00")
    manager = RunManager(store=store, event_store=events)

    first = await manager.reconcile_orphaned_inflight_runs(error="worker crashed", before="2026-01-01T00:00:01+00:00")
    second = await manager.reconcile_orphaned_inflight_runs(error="worker crashed", before="2026-01-01T00:00:01+00:00")

    assert [record.run_id for record in first] == ["running-run"]
    assert second == []
    delivery = await events.list_events("thread-1", "running-run", event_types=["run.delivery"])
    assert len(delivery) == 1
    assert delivery[0]["content"] == {"presented": 0, "paths": [], "by_tool": {}}
    assert (await store.get("running-run"))["status"] == "error"


@pytest.mark.anyio
async def test_reconcile_preserves_delivery_written_before_worker_crash():
    """A crash after the receipt but before status persistence keeps its facts."""
    store = MemoryRunStore()
    events = MemoryRunEventStore()
    await store.put("running-run", thread_id="thread-1", status="running", created_at="2026-01-01T00:00:00+00:00")
    await events.put_if_absent(
        thread_id="thread-1",
        run_id="running-run",
        event_type="run.delivery",
        category="outputs",
        content={"presented": 1, "paths": ["report.md"], "by_tool": {"present_files": ["report.md"]}},
    )
    manager = RunManager(store=store, event_store=events)

    recovered = await manager.reconcile_orphaned_inflight_runs(error="worker crashed", before="2026-01-01T00:00:01+00:00")

    assert [record.run_id for record in recovered] == ["running-run"]
    delivery = await events.list_events("thread-1", "running-run", event_types=["run.delivery"])
    assert len(delivery) == 1
    assert delivery[0]["content"]["presented"] == 1


@pytest.mark.anyio
async def test_reconcile_preserves_terminal_takeover_when_delivery_backfill_fails():
    """A receipt-store outage must not undo an atomically claimed orphan."""

    class FailingReceiptStore(MemoryRunEventStore):
        async def put_if_absent(self, **kwargs):
            raise RuntimeError("event store unavailable")

    store = MemoryRunStore()
    await store.put("running-run", thread_id="thread-1", status="running", created_at="2026-01-01T00:00:00+00:00")
    manager = RunManager(store=store, event_store=FailingReceiptStore())

    recovered = await manager.reconcile_orphaned_inflight_runs(error="worker crashed", before="2026-01-01T00:00:01+00:00")

    assert [record.run_id for record in recovered] == ["running-run"]
    assert (await store.get("running-run"))["status"] == "error"


@pytest.mark.anyio
async def test_reconcile_orphaned_inflight_runs_skips_live_local_run():
    """Startup recovery should not mark an active row orphaned when this worker owns it."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.running)

    recovered = await manager.reconcile_orphaned_inflight_runs(
        error="Gateway restarted before this run reached a durable final state.",
    )

    stored = await store.get(record.run_id)
    assert recovered == []
    assert stored["status"] == "running"


@pytest.mark.anyio
async def test_reconcile_orphaned_inflight_runs_skips_rows_when_takeover_claim_fails():
    """Startup recovery must not report a row as recovered if the takeover claim failed."""
    store = FailingTakeoverRunStore()
    await store.put("running-run", thread_id="thread-1", status="running", created_at="2026-01-01T00:00:00+00:00")
    manager = RunManager(
        store=store,
        persistence_retry_policy=PersistenceRetryPolicy(max_attempts=2, initial_delay=0),
    )

    recovered = await manager.reconcile_orphaned_inflight_runs(
        error="Gateway restarted before this run reached a durable final state.",
        before="2026-01-01T00:00:01+00:00",
    )

    stored = await store.get("running-run")
    assert recovered == []
    assert stored["status"] == "running"
    assert store.takeover_attempts == 2


@pytest.mark.anyio
async def test_cancel_not_inflight(manager: RunManager):
    """Cancelling a completed run should return not_cancellable."""
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.success)

    cancelled = await manager.cancel(record.run_id)
    assert cancelled == CancelOutcome.not_cancellable


@pytest.mark.anyio
async def test_list_by_thread(manager: RunManager):
    """Same thread should return multiple runs."""
    r1 = await manager.create("thread-1")
    r2 = await manager.create("thread-1")
    await manager.create("thread-2")

    runs = await manager.list_by_thread("thread-1")
    assert len(runs) == 2
    # Newest first: r2 was created after r1.
    assert runs[0].run_id == r2.run_id
    assert runs[1].run_id == r1.run_id


@pytest.mark.anyio
async def test_list_by_thread_is_stable_when_timestamps_tie(manager: RunManager, monkeypatch: pytest.MonkeyPatch):
    """Ordering should be stable (insertion order) even when timestamps tie."""
    monkeypatch.setattr("deerflow.runtime.runs.manager._now_iso", lambda: "2026-01-01T00:00:00+00:00")

    r1 = await manager.create("thread-1")
    r2 = await manager.create("thread-1")

    runs = await manager.list_by_thread("thread-1")
    assert [run.run_id for run in runs] == [r1.run_id, r2.run_id]


@pytest.mark.anyio
async def test_has_inflight(manager: RunManager):
    """has_inflight should be True when a run is pending or running."""
    record = await manager.create("thread-1")
    assert await manager.has_inflight("thread-1") is True

    await manager.set_status(record.run_id, RunStatus.success)
    assert await manager.has_inflight("thread-1") is False


@pytest.mark.anyio
async def test_has_inflight_ignores_checkpoint_write_reservation(manager: RunManager):
    """Internal checkpoint writers are not user-visible runs."""
    async with manager.reserve_thread_operation(
        "thread-1",
        kind=ThreadOperationKind.checkpoint_write,
    ):
        assert await manager.has_inflight("thread-1") is False


@pytest.mark.anyio
async def test_cleanup(manager: RunManager):
    """After cleanup, the run should be gone."""
    record = await manager.create("thread-1")
    run_id = record.run_id

    await manager.cleanup(run_id, delay=0)
    assert await manager.get(run_id) is None


@pytest.mark.anyio
async def test_set_status_with_error(manager: RunManager):
    """Error message should be stored on the record."""
    record = await manager.create("thread-1")
    await manager.set_status(record.run_id, RunStatus.error, error="Something went wrong")
    assert record.status == RunStatus.error
    assert record.error == "Something went wrong"


@pytest.mark.anyio
async def test_get_nonexistent(manager: RunManager):
    """Getting a nonexistent run should return None."""
    assert await manager.get("does-not-exist") is None


@pytest.mark.anyio
async def test_get_hydrates_store_only_run():
    """Store-only runs should be readable after process restart."""
    store = MemoryRunStore()
    await store.put(
        "run-store-only",
        thread_id="thread-1",
        assistant_id="lead_agent",
        status="success",
        multitask_strategy="reject",
        metadata={"source": "store"},
        kwargs={"input": "value"},
        created_at="2026-01-01T00:00:00+00:00",
        model_name="model-a",
    )
    manager = RunManager(store=store)

    record = await manager.get("run-store-only")

    assert record is not None
    assert record.run_id == "run-store-only"
    assert record.thread_id == "thread-1"
    assert record.assistant_id == "lead_agent"
    assert record.status == RunStatus.success
    assert record.on_disconnect == DisconnectMode.cancel
    assert record.metadata == {"source": "store"}
    assert record.kwargs == {"input": "value"}
    assert record.model_name == "model-a"
    assert record.task is None
    assert record.store_only is True


@pytest.mark.anyio
async def test_get_hydrates_run_with_null_enum_fields():
    """Rows with NULL status/on_disconnect must hydrate with safe defaults, not raise."""
    store = MemoryRunStore()
    # Simulate a SQL row where the nullable status column is NULL
    await store.put(
        "run-null-status",
        thread_id="thread-1",
        status=None,
        created_at="2026-01-01T00:00:00+00:00",
    )
    manager = RunManager(store=store)

    record = await manager.get("run-null-status")

    assert record is not None
    assert record.status == RunStatus.pending
    assert record.on_disconnect == DisconnectMode.cancel
    assert record.store_only is True


@pytest.mark.anyio
async def test_list_by_thread_hydrates_run_with_null_enum_fields():
    """list_by_thread must not skip rows with NULL status; applies safe defaults."""
    store = MemoryRunStore()
    await store.put(
        "run-null-status-list",
        thread_id="thread-null",
        status=None,
        created_at="2026-01-01T00:00:00+00:00",
    )
    manager = RunManager(store=store)

    runs = await manager.list_by_thread("thread-null")

    assert len(runs) == 1
    assert runs[0].run_id == "run-null-status-list"
    assert runs[0].status == RunStatus.pending
    assert runs[0].on_disconnect == DisconnectMode.cancel


@pytest.mark.anyio
async def test_create_record_is_not_store_only(manager: RunManager):
    """In-memory records created via create() must have store_only=False."""
    record = await manager.create("thread-1")
    assert record.store_only is False


@pytest.mark.anyio
async def test_create_rolls_back_in_memory_record_on_store_failure():
    """create() must fail and hide the run when the initial store write fails."""
    from unittest.mock import AsyncMock

    store = MemoryRunStore()
    store.put = AsyncMock(side_effect=RuntimeError("db down"))
    manager = RunManager(store=store)

    with pytest.raises(RuntimeError, match="db down"):
        await manager.create("thread-1")

    assert manager._runs == {}
    assert await manager.list_by_thread("thread-1") == []


@pytest.mark.anyio
async def test_create_rolls_back_in_memory_record_on_store_cancellation():
    """create() must also roll back when cancelled during the initial store write."""
    store = MemoryRunStore()

    async def cancelled_put(run_id, **kwargs):
        raise asyncio.CancelledError

    store.put = cancelled_put
    manager = RunManager(store=store)

    with pytest.raises(asyncio.CancelledError):
        await manager.create("thread-1")

    assert manager._runs == {}
    assert await manager.list_by_thread("thread-1") == []


@pytest.mark.anyio
async def test_create_does_not_expose_run_until_store_persist_completes():
    """Concurrent readers must wait until the new run has been persisted."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    original_put = store.put
    put_started = asyncio.Event()
    allow_put = asyncio.Event()

    async def blocking_put(run_id, **kwargs):
        put_started.set()
        await allow_put.wait()
        return await original_put(run_id, **kwargs)

    store.put = blocking_put
    create_task = asyncio.create_task(manager.create("thread-1"))
    list_task = None

    try:
        await put_started.wait()
        list_task = asyncio.create_task(manager.list_by_thread("thread-1"))
        await asyncio.sleep(0)
        assert not list_task.done()

        allow_put.set()
        record = await create_task
        runs = await list_task

        assert [run.run_id for run in runs] == [record.run_id]
    finally:
        allow_put.set()
        cleanup_tasks = []
        for task in (list_task, create_task):
            if task is None:
                continue
            if not task.done():
                task.cancel()
            cleanup_tasks.append(task)
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_get_prefers_in_memory_record_over_store():
    """In-memory records retain task/control state when store has same run."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    record = await manager.create("thread-1")
    await store.update_status(record.run_id, "success")

    fetched = await manager.get(record.run_id)

    assert fetched is record
    assert fetched.status == RunStatus.pending


@pytest.mark.anyio
async def test_list_by_thread_merges_store_runs_newest_first():
    """list_by_thread should merge memory and store rows with memory precedence."""
    store = MemoryRunStore()
    await store.put("old-store", thread_id="thread-1", status="success", created_at="2026-01-01T00:00:00+00:00")
    await store.put("other-thread", thread_id="thread-2", status="success", created_at="2026-01-03T00:00:00+00:00")
    manager = RunManager(store=store)
    memory_record = await manager.create("thread-1")

    runs = await manager.list_by_thread("thread-1")

    assert [run.run_id for run in runs] == [memory_record.run_id, "old-store"]
    assert runs[0] is memory_record


@pytest.mark.anyio
async def test_list_by_thread_limit_does_not_let_old_memory_hide_new_store_run():
    """A local row must not consume the store query's newest-run limit."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    old_memory = await manager.create("thread-1")
    old_memory.created_at = "2026-01-01T00:00:00+00:00"
    await store.put(
        "new-store",
        thread_id="thread-1",
        status="success",
        created_at="2026-01-02T00:00:00+00:00",
    )

    runs = await manager.list_by_thread("thread-1", limit=1)

    assert [run.run_id for run in runs] == ["new-store"]


@pytest.mark.anyio
async def test_create_defaults(manager: RunManager):
    """Create with no optional args should use defaults."""
    record = await manager.create("thread-1")
    assert record.metadata == {}
    assert record.kwargs == {}
    assert record.multitask_strategy == "reject"
    assert record.assistant_id is None


@pytest.mark.anyio
async def test_model_name_create_or_reject():
    """create_or_reject should accept and persist model_name."""
    from deerflow.runtime.runs.schemas import DisconnectMode

    store = MemoryRunStore()
    mgr = RunManager(store=store)

    record = await mgr.create_or_reject(
        "thread-1",
        assistant_id="lead_agent",
        on_disconnect=DisconnectMode.cancel,
        metadata={"key": "val"},
        kwargs={"input": {}},
        multitask_strategy="reject",
        model_name="anthropic.claude-sonnet-4-20250514-v1:0",
    )
    assert record.model_name == "anthropic.claude-sonnet-4-20250514-v1:0"
    assert record.status == RunStatus.pending

    # Verify model_name was persisted to store
    stored = await store.get(record.run_id)
    assert stored is not None
    assert stored["model_name"] == "anthropic.claude-sonnet-4-20250514-v1:0"

    # Verify retrieval returns the model_name via in-memory record
    fetched = await mgr.get(record.run_id)
    assert fetched is not None
    assert fetched.model_name == "anthropic.claude-sonnet-4-20250514-v1:0"


@pytest.mark.anyio
async def test_create_or_reject_interrupt_persists_interrupted_status_to_store():
    """interrupt strategy should persist interrupted status for old runs."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)

    new = await manager.create_or_reject("thread-1", multitask_strategy="interrupt")

    stored_old = await store.get(old.run_id)
    assert new.run_id != old.run_id
    assert old.status == RunStatus.interrupted
    assert stored_old is not None
    assert stored_old["status"] == "interrupted"


@pytest.mark.anyio
async def test_create_or_reject_does_not_interrupt_old_run_when_new_run_store_write_fails():
    """A failed new-run persist must not cancel the existing inflight run."""
    from unittest.mock import AsyncMock

    store = MemoryRunStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)
    store.create_thread_operation_atomic = AsyncMock(side_effect=RuntimeError("db down"))

    with pytest.raises(RuntimeError, match="db down"):
        await manager.create_or_reject("thread-1", multitask_strategy="interrupt")

    stored_old = await store.get(old.run_id)
    assert list(manager._runs) == [old.run_id]
    assert old.status == RunStatus.running
    assert old.abort_event.is_set() is False
    assert stored_old is not None
    assert stored_old["status"] == "running"


@pytest.mark.anyio
async def test_create_or_reject_does_not_interrupt_old_run_when_new_run_store_write_is_cancelled():
    """Cancellation during new-run persist must not cancel the existing run."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)

    async def cancelled_create(run_id, **kwargs):
        raise asyncio.CancelledError

    store.create_thread_operation_atomic = cancelled_create

    with pytest.raises(asyncio.CancelledError):
        await manager.create_or_reject("thread-1", multitask_strategy="interrupt")

    stored_old = await store.get(old.run_id)
    assert list(manager._runs) == [old.run_id]
    assert old.status == RunStatus.running
    assert old.abort_event.is_set() is False
    assert stored_old is not None
    assert stored_old["status"] == "running"


@pytest.mark.anyio
@pytest.mark.parametrize("strategy", ["interrupt", "rollback"])
async def test_create_or_reject_cancellation_after_registration_interrupts_replacement(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    """Cancellation after admission must not leave the replacement active."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)
    persist_started = asyncio.Event()
    release_persist = asyncio.Event()
    original_persist_status = manager._persist_status

    async def blocking_persist_status(record: Any, status: RunStatus, **kwargs: Any) -> bool:
        persist_started.set()
        await asyncio.wait_for(release_persist.wait(), timeout=1)
        return await original_persist_status(record, status, **kwargs)

    monkeypatch.setattr(manager, "_persist_status", blocking_persist_status)
    create_task = asyncio.create_task(manager.create_or_reject("thread-1", multitask_strategy=strategy))
    await asyncio.wait_for(persist_started.wait(), timeout=1)
    create_task.cancel()
    release_persist.set()

    with pytest.raises(asyncio.CancelledError):
        _ = await create_task

    records = await manager.list_by_thread("thread-1")
    replacement = next(record for record in records if record.run_id != old.run_id)
    stored_replacement = await store.get(replacement.run_id)
    assert not await manager.has_inflight("thread-1")
    assert replacement.status == RunStatus.interrupted
    assert replacement.abort_event.is_set()
    assert stored_replacement is not None
    assert stored_replacement["status"] == RunStatus.interrupted.value


@pytest.mark.anyio
async def test_create_or_reject_repeated_cancellation_drains_replacement_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated cancellation must not abandon the durable cleanup task."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)
    old_persist_started = asyncio.Event()
    release_old_persist = asyncio.Event()
    replacement_persist_started = asyncio.Event()
    release_replacement_persist = asyncio.Event()
    original_persist_status = manager._persist_status

    async def staged_persist_status(record: Any, status: RunStatus, **kwargs: Any) -> bool:
        if record.run_id == old.run_id:
            old_persist_started.set()
            await asyncio.wait_for(release_old_persist.wait(), timeout=1)
        else:
            replacement_persist_started.set()
            await asyncio.wait_for(release_replacement_persist.wait(), timeout=1)
        return await original_persist_status(record, status, **kwargs)

    monkeypatch.setattr(manager, "_persist_status", staged_persist_status)
    create_task = asyncio.create_task(manager.create_or_reject("thread-1", multitask_strategy="interrupt"))
    await asyncio.wait_for(old_persist_started.wait(), timeout=1)
    replacement = next(record for record in manager._runs.values() if record.run_id != old.run_id)

    create_task.cancel()
    release_old_persist.set()
    await asyncio.wait_for(replacement_persist_started.wait(), timeout=1)
    create_task.cancel()
    await asyncio.sleep(0)
    assert not create_task.done()

    release_replacement_persist.set()
    with pytest.raises(asyncio.CancelledError):
        _ = await asyncio.wait_for(create_task, timeout=1)

    stored_replacement = await store.get(replacement.run_id)
    assert replacement.status == RunStatus.interrupted
    assert stored_replacement is not None
    assert stored_replacement["status"] == RunStatus.interrupted.value


@pytest.mark.anyio
async def test_create_or_reject_retries_replacement_when_cancel_status_cannot_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed best-effort update must get a strict durable retry."""

    class FailFirstReplacementInterruptStore(MemoryRunStore):
        failed = False

        async def get(self, run_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
            raw = await super().get(run_id)
            if raw is not None and raw.get("status") == RunStatus.pending.value and raw.get("user_id") is not None and user_id != raw.get("user_id"):
                raise RuntimeError("replacement lookup was not owner-scoped")
            return await super().get(run_id, user_id=user_id)

        async def update_status(self, run_id: str, status: str, **kwargs: Any) -> bool:
            row = await super().get(run_id)
            if not self.failed and status == RunStatus.interrupted.value and row is not None and row.get("status") == RunStatus.pending.value:
                self.failed = True
                raise RuntimeError("replacement status write failed")
            return await super().update_status(run_id, status, **kwargs)

    store = FailFirstReplacementInterruptStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1", user_id="owner-1")
    await manager.set_status(old.run_id, RunStatus.running)
    old_persist_started = asyncio.Event()
    release_old_persist = asyncio.Event()
    original_persist_status = manager._persist_status

    async def block_old_persist(record: Any, status: RunStatus, **kwargs: Any) -> bool:
        if record.run_id != old.run_id:
            return await original_persist_status(record, status, **kwargs)
        old_persist_started.set()
        await asyncio.wait_for(release_old_persist.wait(), timeout=1)
        return await original_persist_status(record, status, **kwargs)

    monkeypatch.setattr(manager, "_persist_status", block_old_persist)
    create_task = asyncio.create_task(manager.create_or_reject("thread-1", multitask_strategy="interrupt", user_id="owner-1"))
    await asyncio.wait_for(old_persist_started.wait(), timeout=1)
    replacement = next(record for record in manager._runs.values() if record.run_id != old.run_id)
    create_task.cancel()
    release_old_persist.set()

    with pytest.raises(asyncio.CancelledError):
        _ = await asyncio.wait_for(create_task, timeout=1)

    stored_replacement = await store.get(replacement.run_id, user_id="owner-1")
    assert stored_replacement is not None
    assert stored_replacement["status"] == RunStatus.interrupted.value
    assert replacement.status == RunStatus.interrupted
    assert not await manager.has_inflight("thread-1")


@pytest.mark.anyio
async def test_create_or_reject_cleanup_failure_preserves_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup IO failure must not replace the caller's CancelledError."""

    class FailingReplacementCleanupStore(MemoryRunStore):
        replacement_update_failed = False

        async def update_status(self, run_id: str, status: str, **kwargs: Any) -> bool:
            row = await super().get(run_id)
            if status == RunStatus.interrupted.value and row is not None and row.get("status") == RunStatus.pending.value:
                self.replacement_update_failed = True
                raise RuntimeError("replacement status unavailable")
            return await super().update_status(run_id, status, **kwargs)

        async def get(self, run_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
            row = await super().get(run_id, user_id=user_id)
            if self.replacement_update_failed and row is not None and row.get("status") == RunStatus.pending.value:
                raise RuntimeError("replacement verification unavailable")
            return row

    store = FailingReplacementCleanupStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)
    old_persist_started = asyncio.Event()
    release_old_persist = asyncio.Event()
    original_persist_status = manager._persist_status

    async def block_old_persist(record: Any, status: RunStatus, **kwargs: Any) -> bool:
        if record.run_id != old.run_id:
            return await original_persist_status(record, status, **kwargs)
        old_persist_started.set()
        await asyncio.wait_for(release_old_persist.wait(), timeout=1)
        return await original_persist_status(record, status, **kwargs)

    monkeypatch.setattr(manager, "_persist_status", block_old_persist)
    create_task = asyncio.create_task(manager.create_or_reject("thread-1", multitask_strategy="interrupt"))
    await asyncio.wait_for(old_persist_started.wait(), timeout=1)
    create_task.cancel()
    release_old_persist.set()

    with pytest.raises(asyncio.CancelledError):
        _ = await asyncio.wait_for(create_task, timeout=1)


@pytest.mark.anyio
async def test_create_or_reject_preserves_peer_terminal_status_during_cancel_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peer terminal transition must win the strict cancellation retry."""

    class PeerWinsReplacementInterruptStore(MemoryRunStore):
        replacement_attempts = 0

        async def update_status(self, run_id: str, status: str, **kwargs: Any) -> bool:
            row = await self.get(run_id)
            if status == RunStatus.interrupted.value and row is not None and row.get("status") == RunStatus.pending.value:
                self.replacement_attempts += 1
                if self.replacement_attempts == 1:
                    raise RuntimeError("replacement status write failed")
                await super().update_status(run_id, RunStatus.error.value, error="peer takeover")
                return False
            return await super().update_status(run_id, status, **kwargs)

    store = PeerWinsReplacementInterruptStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)
    old_persist_started = asyncio.Event()
    release_old_persist = asyncio.Event()
    original_persist_status = manager._persist_status

    async def block_old_persist(record: Any, status: RunStatus, **kwargs: Any) -> bool:
        if record.run_id != old.run_id:
            return await original_persist_status(record, status, **kwargs)
        old_persist_started.set()
        await asyncio.wait_for(release_old_persist.wait(), timeout=1)
        return await original_persist_status(record, status, **kwargs)

    monkeypatch.setattr(manager, "_persist_status", block_old_persist)
    create_task = asyncio.create_task(manager.create_or_reject("thread-1", multitask_strategy="interrupt"))
    await asyncio.wait_for(old_persist_started.wait(), timeout=1)
    replacement = next(record for record in manager._runs.values() if record.run_id != old.run_id)
    create_task.cancel()
    release_old_persist.set()

    with pytest.raises(asyncio.CancelledError):
        _ = await asyncio.wait_for(create_task, timeout=1)

    stored_replacement = await store.get(replacement.run_id)
    assert stored_replacement is not None
    assert stored_replacement["status"] == RunStatus.error.value
    assert stored_replacement["error"] == "peer takeover"
    assert replacement.status == RunStatus.error
    assert replacement.error == "peer takeover"
    assert not await manager.has_inflight("thread-1")


@pytest.mark.anyio
async def test_create_or_reject_rollback_persists_interrupted_status_to_store():
    """rollback strategy should persist interrupted status for old runs."""
    store = MemoryRunStore()
    manager = RunManager(store=store)
    old = await manager.create("thread-1")
    await manager.set_status(old.run_id, RunStatus.running)

    new = await manager.create_or_reject("thread-1", multitask_strategy="rollback")

    stored_old = await store.get(old.run_id)
    assert new.run_id != old.run_id
    assert old.status == RunStatus.interrupted
    assert stored_old is not None
    assert stored_old["status"] == "interrupted"


@pytest.mark.anyio
async def test_model_name_default_is_none():
    """create_or_reject without model_name should default to None."""
    from deerflow.runtime.runs.schemas import DisconnectMode

    store = MemoryRunStore()
    mgr = RunManager(store=store)

    record = await mgr.create_or_reject(
        "thread-1",
        on_disconnect=DisconnectMode.cancel,
        model_name=None,
    )
    assert record.model_name is None

    stored = await store.get(record.run_id)
    assert stored["model_name"] is None


# ---------------------------------------------------------------------------
# Store fallback tests (simulates gateway restart scenario)
# ---------------------------------------------------------------------------


@pytest.fixture
def manager_with_store() -> RunManager:
    """RunManager backed by a MemoryRunStore."""
    return RunManager(store=MemoryRunStore())


@pytest.mark.anyio
async def test_list_by_thread_returns_store_records_after_restart(manager_with_store: RunManager):
    """After in-memory state is cleared (simulating restart), list_by_thread
    should still return runs from the persistent store."""
    mgr = manager_with_store
    r1 = await mgr.create("thread-1", "agent-1")
    await mgr.set_status(r1.run_id, RunStatus.success)
    r2 = await mgr.create("thread-1", "agent-2")
    await mgr.set_status(r2.run_id, RunStatus.error, error="boom")

    # Clear in-memory dict to simulate a restart
    mgr._runs.clear()

    runs = await mgr.list_by_thread("thread-1")
    assert len(runs) == 2
    statuses = {r.run_id: r.status for r in runs}
    assert statuses[r1.run_id] == RunStatus.success
    assert statuses[r2.run_id] == RunStatus.error
    # Verify other fields survive the round-trip
    for r in runs:
        assert r.thread_id == "thread-1"
        assert ISO_RE.match(r.created_at)


@pytest.mark.anyio
async def test_list_by_thread_merges_in_memory_and_store(manager_with_store: RunManager):
    """In-memory runs should be included alongside store-only records."""
    mgr = manager_with_store

    # Create a run and let it complete (will be in both memory and store)
    r1 = await mgr.create("thread-1")
    await mgr.set_status(r1.run_id, RunStatus.success)

    # Simulate restart: clear memory, then create a new in-memory run
    mgr._runs.clear()
    r2 = await mgr.create("thread-1")

    runs = await mgr.list_by_thread("thread-1")
    assert len(runs) == 2
    run_ids = {r.run_id for r in runs}
    assert r1.run_id in run_ids
    assert r2.run_id in run_ids

    # r2 should be the in-memory record (has live state)
    r2_record = next(r for r in runs if r.run_id == r2.run_id)
    assert r2_record is r2  # same object reference


@pytest.mark.anyio
async def test_list_by_thread_no_store():
    """Without a store, list_by_thread should only return in-memory runs."""
    mgr = RunManager()
    await mgr.create("thread-1")

    mgr._runs.clear()
    runs = await mgr.list_by_thread("thread-1")
    assert runs == []


@pytest.mark.anyio
async def test_aget_returns_in_memory_record(manager_with_store: RunManager):
    """aget should return the in-memory record when available."""
    mgr = manager_with_store
    r1 = await mgr.create("thread-1", "agent-1")

    result = await mgr.aget(r1.run_id)
    assert result is r1  # same object


@pytest.mark.anyio
async def test_aget_falls_back_to_store(manager_with_store: RunManager):
    """aget should return a record from the store when not in memory."""
    mgr = manager_with_store
    r1 = await mgr.create("thread-1", "agent-1")
    await mgr.set_status(r1.run_id, RunStatus.success)

    mgr._runs.clear()

    result = await mgr.aget(r1.run_id)
    assert result is not None
    assert result.run_id == r1.run_id
    assert result.status == RunStatus.success
    assert result.thread_id == "thread-1"
    assert result.assistant_id == "agent-1"


@pytest.mark.anyio
async def test_aget_falls_back_to_store_with_user_filter():
    """aget should honor user_id when reading store-only records."""
    store = MemoryRunStore()
    await store.put("run-1", thread_id="thread-1", user_id="user-1", status="success")
    mgr = RunManager(store=store)

    allowed = await mgr.aget("run-1", user_id="user-1")
    denied = await mgr.aget("run-1", user_id="user-2")
    assert allowed is not None
    assert denied is None


@pytest.mark.anyio
async def test_aget_returns_none_for_unknown(manager_with_store: RunManager):
    """aget should return None for a run ID that doesn't exist anywhere."""
    result = await manager_with_store.aget("nonexistent-run-id")
    assert result is None


@pytest.mark.anyio
async def test_aget_store_failure_is_graceful():
    """If the store raises, aget should return None instead of propagating."""
    from unittest.mock import AsyncMock

    store = MemoryRunStore()
    store.get = AsyncMock(side_effect=RuntimeError("db down"))
    mgr = RunManager(store=store)

    result = await mgr.aget("some-id")
    assert result is None


@pytest.mark.anyio
async def test_get_can_surface_store_failure_for_lifecycle_callers():
    """Lifecycle code must distinguish a missing run from an unavailable store."""
    from unittest.mock import AsyncMock

    store = MemoryRunStore()
    store.get = AsyncMock(side_effect=RuntimeError("db down"))
    mgr = RunManager(store=store)

    with pytest.raises(RuntimeError, match="db down"):
        await mgr.get("some-id", raise_on_store_error=True)


@pytest.mark.anyio
async def test_list_by_thread_store_failure_is_graceful():
    """If the store raises, list_by_thread should return only in-memory runs."""
    from unittest.mock import AsyncMock

    store = MemoryRunStore()
    store.list_by_thread = AsyncMock(side_effect=RuntimeError("db down"))
    mgr = RunManager(store=store)

    r1 = await mgr.create("thread-1")
    runs = await mgr.list_by_thread("thread-1")
    assert len(runs) == 1
    assert runs[0].run_id == r1.run_id


@pytest.mark.anyio
async def test_list_by_thread_falls_back_to_store_with_user_filter():
    """list_by_thread should return only the requesting user's store records."""
    store = MemoryRunStore()
    await store.put("run-1", thread_id="thread-1", user_id="user-1", status="success")
    await store.put("run-2", thread_id="thread-1", user_id="user-2", status="success")
    mgr = RunManager(store=store)

    runs = await mgr.list_by_thread("thread-1", user_id="user-1")
    assert [r.run_id for r in runs] == ["run-1"]


# ---------------------------------------------------------------------------
# Per-thread index (thread_id -> run_ids): keeps per-thread queries
# O(runs-in-thread) instead of scanning every in-memory run, and stays
# consistent with ``_runs`` across create / cleanup / rollback.
# ---------------------------------------------------------------------------


class _FailingPutRunStore(MemoryRunStore):
    """Memory run store whose every ``put`` and atomic operation create fails."""

    async def put(self, run_id, **kwargs):
        raise ValueError("simulated persist failure")

    async def create_thread_operation_atomic(self, run_id, **kwargs):
        raise ValueError("simulated persist failure")


@pytest.mark.anyio
async def test_thread_index_scopes_runs_per_thread(manager: RunManager):
    a1 = await manager.create("thread-a")
    a2 = await manager.create("thread-a")
    b1 = await manager.create("thread-b")

    # The index mirrors _runs membership, bucketed by thread.
    assert set(manager._runs_by_thread["thread-a"]) == {a1.run_id, a2.run_id}
    assert set(manager._runs_by_thread["thread-b"]) == {b1.run_id}

    # Per-thread queries return only that thread's runs (no cross-thread leak).
    assert {r.run_id for r in await manager.list_by_thread("thread-a")} == {a1.run_id, a2.run_id}
    assert {r.run_id for r in await manager.list_by_thread("thread-b")} == {b1.run_id}
    assert await manager.list_by_thread("thread-missing") == []


@pytest.mark.anyio
async def test_thread_index_preserves_insertion_order(manager: RunManager):
    # The index is insertion-ordered (dict-as-ordered-set) so list_by_thread
    # keeps the stable tie-breaking the full-scan implementation guaranteed.
    first = await manager.create("thread-a")
    second = await manager.create("thread-a")
    assert list(manager._runs_by_thread["thread-a"]) == [first.run_id, second.run_id]


@pytest.mark.anyio
async def test_thread_index_cleanup_prunes_run_and_empty_bucket(manager: RunManager):
    a1 = await manager.create("thread-a")
    a2 = await manager.create("thread-a")

    await manager.cleanup(a1.run_id, delay=0)
    assert a1.run_id not in manager._runs
    assert set(manager._runs_by_thread["thread-a"]) == {a2.run_id}

    await manager.cleanup(a2.run_id, delay=0)
    # Empty buckets are pruned so the index cannot grow without bound.
    assert "thread-a" not in manager._runs_by_thread
    assert await manager.list_by_thread("thread-a") == []


@pytest.mark.anyio
async def test_has_inflight_reflects_index(manager: RunManager):
    record = await manager.create("thread-a")
    assert await manager.has_inflight("thread-a") is True
    assert await manager.has_inflight("thread-b") is False

    await manager.set_status(record.run_id, RunStatus.success)
    assert await manager.has_inflight("thread-a") is False


@pytest.mark.anyio
async def test_create_or_reject_inflight_is_thread_scoped(manager: RunManager):
    await manager.create_or_reject("thread-a", multitask_strategy="reject")
    # A different thread is unaffected by thread-a's active run.
    await manager.create_or_reject("thread-b", multitask_strategy="reject")
    # A second active run on the same thread is rejected.
    with pytest.raises(ConflictError):
        await manager.create_or_reject("thread-a", multitask_strategy="reject")


@pytest.mark.anyio
async def test_failed_create_unindexes_run():
    manager = RunManager(store=_FailingPutRunStore())
    with pytest.raises(ValueError):
        await manager.create("thread-a")
    # A rolled-back run must leave no trace in either _runs or the index.
    assert manager._runs == {}
    assert "thread-a" not in manager._runs_by_thread


@pytest.mark.anyio
async def test_failed_create_or_reject_unindexes_run():
    # Symmetric to test_failed_create_unindexes_run: create_or_reject has its own
    # insert + rollback-unindex site, so a persist failure there must also leave
    # neither _runs nor the index holding the rolled-back run. This closes the last
    # mutation path not exercised by an index-consistency test.
    manager = RunManager(store=_FailingPutRunStore())
    with pytest.raises(ValueError):
        await manager.create_or_reject("thread-a", multitask_strategy="reject")
    assert manager._runs == {}
    assert "thread-a" not in manager._runs_by_thread
