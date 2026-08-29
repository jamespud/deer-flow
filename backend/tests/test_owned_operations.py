import asyncio
import gc
import weakref

import pytest

from deerflow.runtime.owned_operations import OwnedTaskSet


@pytest.mark.anyio
async def test_registry_exposes_stable_read_only_collection_view():
    release = asyncio.Event()
    first = asyncio.create_task(release.wait())
    second = asyncio.create_task(release.wait())
    owned = OwnedTaskSet()

    def on_settled(outcome):
        pass

    owned.retain(first, on_settled=on_settled)
    owned.retain(second, on_settled=on_settled)
    snapshot = owned.snapshot()

    assert len(snapshot) == 2
    assert first in snapshot
    assert snapshot == (first, second)

    release.set()
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)

    assert owned.snapshot() == ()
    assert first not in owned.snapshot()


@pytest.mark.anyio
async def test_retain_reports_completed_success_with_terminal_outcome():
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    future.set_result("done")
    outcomes = []

    owned = OwnedTaskSet()
    owned.retain(future, on_settled=outcomes.append)

    assert len(outcomes) == 1
    assert outcomes[0].future is future
    assert outcomes[0].result == "done"
    assert outcomes[0].error is None
    assert outcomes[0].cancelled is False


@pytest.mark.anyio
async def test_completed_future_keeps_its_first_callback_binding_without_ownership():
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    future.set_result("done")
    calls = []
    owned = OwnedTaskSet()

    def first_callback(outcome):
        calls.append(outcome.result)

    def second_callback(outcome):
        calls.append("second")

    owned.retain(future, on_settled=first_callback)

    assert owned.retain(future, on_settled=first_callback) is future
    with pytest.raises(ValueError, match="different settlement callback"):
        owned.retain(future, on_settled=second_callback)
    assert calls == ["done"]
    assert future not in owned._owned


@pytest.mark.anyio
async def test_completed_future_binding_does_not_keep_future_alive():
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_result(None)
    future_reference = weakref.ref(future)
    owned = OwnedTaskSet()

    owned.retain(future, on_settled=lambda outcome: None)
    await asyncio.sleep(0)
    del future
    gc.collect()

    assert future_reference() is None


@pytest.mark.anyio
async def test_completed_future_binding_does_not_keep_capturing_callback_or_future_alive():
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_result(None)
    future_reference = weakref.ref(future)
    owned = OwnedTaskSet()

    def callback(outcome, held_future=future):
        assert outcome.future is held_future

    callback_reference = weakref.ref(callback)
    owned.retain(future, on_settled=callback)
    await asyncio.sleep(0)
    del callback
    del future
    gc.collect()

    assert callback_reference() is None
    assert future_reference() is None


@pytest.mark.anyio
async def test_retain_rejects_non_weak_callback_before_id_reuse_can_weaken_identity():
    class NonWeakCallback:
        __slots__ = ()

        def __call__(self, outcome):
            pass

    callback = None
    for _ in range(10):
        first_callback = NonWeakCallback()
        first_callback_id = id(first_callback)
        del first_callback
        gc.collect()
        candidate = NonWeakCallback()
        if id(candidate) == first_callback_id:
            callback = candidate
            break
        del candidate

    assert callback is not None

    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_result(None)
    owned = OwnedTaskSet()

    with pytest.raises(TypeError, match="weak-referenceable"):
        owned.retain(future, on_settled=callback)

    assert future not in owned._owned
    assert future not in owned._settlement_callbacks


@pytest.mark.anyio
async def test_start_rejects_non_weak_callback_before_creating_task():
    class NonWeakCallback:
        __slots__ = ()

        def __call__(self, outcome):
            pass

    operation_started = False

    async def operation():
        nonlocal operation_started
        operation_started = True

    operation_coroutine = operation()
    owned = OwnedTaskSet()
    try:
        with pytest.raises(TypeError, match="weak-referenceable"):
            owned.start(operation_coroutine, on_settled=NonWeakCallback())
    finally:
        for task in tuple(owned._owned):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        if not operation_coroutine.cr_running:
            operation_coroutine.close()
    await asyncio.sleep(0)

    assert operation_started is False


@pytest.mark.anyio
async def test_retain_rejects_async_callback_without_creating_an_unawaited_coroutine(recwarn):
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_result(None)
    callback_ran = False

    async def on_settled(outcome):
        nonlocal callback_ran
        callback_ran = True

    owned = OwnedTaskSet()

    with pytest.raises(TypeError, match="synchronous"):
        owned.retain(future, on_settled=on_settled)
    await asyncio.sleep(0)

    assert callback_ran is False
    assert future not in owned._owned
    assert not [warning for warning in recwarn if warning.category is RuntimeWarning]


@pytest.mark.anyio
async def test_awaitable_callback_result_is_consumed_and_logged(recwarn, caplog):
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_result(None)
    owned = OwnedTaskSet()

    async def unexpected_async_finalization():
        return None

    def on_settled(outcome):
        return unexpected_async_finalization()

    with caplog.at_level("ERROR", logger="deerflow.runtime.owned_operations"):
        owned.retain(future, on_settled=on_settled)
        gc.collect()

    assert future not in owned._owned
    assert "must be synchronous" in caplog.text
    assert not [warning for warning in recwarn if warning.category is RuntimeWarning]


@pytest.mark.anyio
async def test_callback_returned_failing_task_has_its_exception_consumed():
    loop = asyncio.get_running_loop()
    exception_contexts = []
    failed = asyncio.Event()
    returned_task_reference = None
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: exception_contexts.append(context))

    async def fail():
        await asyncio.sleep(0)
        failed.set()
        raise RuntimeError("returned task failed")

    def on_settled(outcome):
        nonlocal returned_task_reference
        task = asyncio.create_task(fail())
        returned_task_reference = weakref.ref(task)
        return task

    future: asyncio.Future[None] = loop.create_future()
    future.set_result(None)
    try:
        OwnedTaskSet().retain(future, on_settled=on_settled)
        await failed.wait()
        await asyncio.sleep(0)
        gc.collect()
    finally:
        loop.set_exception_handler(previous_handler)

    assert returned_task_reference() is None
    assert not exception_contexts


@pytest.mark.anyio
async def test_retain_reports_base_exception_failure_with_terminal_outcome():
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()
    error = KeyboardInterrupt("stop")
    future.set_exception(error)
    outcomes = []

    OwnedTaskSet().retain(future, on_settled=outcomes.append)

    assert len(outcomes) == 1
    assert outcomes[0].future is future
    assert outcomes[0].result is None
    assert outcomes[0].error is error
    assert outcomes[0].cancelled is False


@pytest.mark.anyio
async def test_start_consumes_self_cancellation_in_terminal_outcome():
    async def self_cancel():
        asyncio.current_task().cancel("self")
        await asyncio.sleep(0)

    outcomes = []
    owned = OwnedTaskSet()
    task = owned.start(self_cancel(), on_settled=outcomes.append)

    result = await owned.wait_until(task, deadline=asyncio.get_running_loop().time() + 1)

    assert result.completed is True
    assert result.cancellation_args is None
    assert len(outcomes) == 1
    assert outcomes[0].future is task
    assert outcomes[0].result is None
    assert isinstance(outcomes[0].error, asyncio.CancelledError)
    assert outcomes[0].error.args == ("self",)
    assert outcomes[0].cancelled is True


@pytest.mark.anyio
async def test_wait_reports_caller_cancellation_without_marking_child_cancelled():
    child_started = asyncio.Event()
    release_child = asyncio.Event()
    outcomes = []

    async def child():
        child_started.set()
        await release_child.wait()
        return "late"

    owned = OwnedTaskSet()
    task = owned.start(child(), on_settled=outcomes.append)
    waiter = asyncio.create_task(owned.wait_until(task, deadline=asyncio.get_running_loop().time() + 1))
    await child_started.wait()
    waiter.cancel("caller")
    result = await waiter

    assert result.completed is False
    assert result.cancellation_args == ("caller",)
    assert task.cancelled() is False
    assert outcomes == []

    release_child.set()
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)


@pytest.mark.anyio
async def test_start_creates_and_owns_coroutine_task():
    outcomes = []
    owned = OwnedTaskSet()

    task = owned.start(asyncio.sleep(0, result=7), on_settled=outcomes.append)
    result = await owned.wait_until(task, deadline=asyncio.get_running_loop().time() + 1)

    assert isinstance(task, asyncio.Task)
    assert result.completed is True
    assert outcomes[0].result == 7


@pytest.mark.anyio
async def test_retain_same_future_with_same_callback_is_idempotent():
    release = asyncio.Event()
    future = asyncio.create_task(release.wait())
    outcomes = []
    on_settled = outcomes.append
    owned = OwnedTaskSet()

    assert owned.retain(future, on_settled=on_settled) is future
    assert owned.retain(future, on_settled=on_settled) is future
    release.set()
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)

    assert len(outcomes) == 1


@pytest.mark.anyio
async def test_retain_same_future_with_different_callback_raises():
    future = asyncio.create_task(asyncio.sleep(10))
    owned = OwnedTaskSet()

    try:
        owned.retain(future, on_settled=lambda outcome: None)
        with pytest.raises(ValueError, match="different settlement callback"):
            owned.retain(future, on_settled=lambda outcome: None)
    finally:
        future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await future


@pytest.mark.anyio
async def test_retain_rejects_none_callback_before_registration():
    release = asyncio.Event()
    future = asyncio.create_task(release.wait())
    owned = OwnedTaskSet()

    try:
        with pytest.raises(ValueError, match="on_settled"):
            owned.retain(future, on_settled=None)
        assert future not in owned._owned
    finally:
        future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await future


@pytest.mark.anyio
async def test_start_rejects_none_callback_before_creating_task():
    operation_started = False

    async def operation():
        nonlocal operation_started
        operation_started = True

    operation_coroutine = operation()
    try:
        with pytest.raises(ValueError, match="on_settled"):
            OwnedTaskSet().start(operation_coroutine, on_settled=None)
    finally:
        operation_coroutine.close()
    await asyncio.sleep(0)

    assert operation_started is False


@pytest.mark.anyio
async def test_settlement_callback_runs_before_registry_removal():
    observations = []
    owned = OwnedTaskSet()
    future = asyncio.create_task(asyncio.sleep(0, result="done"))

    def on_settled(outcome):
        observations.append(outcome.future in owned._owned)

    owned.retain(future, on_settled=on_settled)
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)

    assert observations == [True]
    assert future not in owned._owned


@pytest.mark.anyio
async def test_callback_failure_is_logged_and_consumed_before_ownership_removal(caplog):
    owned = OwnedTaskSet()
    future = asyncio.create_task(asyncio.sleep(0, result="done"))

    def on_settled(outcome):
        raise RuntimeError("callback failed")

    owned.retain(future, on_settled=on_settled)
    with caplog.at_level("ERROR", logger="deerflow.runtime.owned_operations"):
        result = await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)

    assert result.pending == frozenset()
    assert future not in owned._owned
    assert "Owned task settlement callback failed" in caplog.messages


@pytest.mark.anyio
async def test_drain_never_cancels_owned_task():
    release = asyncio.Event()

    class CancelGuardTask(asyncio.Task[None]):
        cancel_called = False

        def cancel(self, *args, **kwargs):
            self.cancel_called = True
            raise AssertionError("OwnedTaskSet must not cancel owned work")

    task = CancelGuardTask(release.wait(), loop=asyncio.get_running_loop())
    owned = OwnedTaskSet()
    owned.retain(task, on_settled=lambda outcome: None)

    result = await owned.drain_all_until(deadline=asyncio.get_running_loop().time())

    assert result.pending == frozenset({task})
    assert task.cancel_called is False
    release.set()
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)


@pytest.mark.anyio
async def test_wait_until_reports_late_failure_after_settlement_callback():
    release = asyncio.Event()
    error = RuntimeError("late")
    outcomes = []

    async def fail_late():
        await release.wait()
        raise error

    owned = OwnedTaskSet()
    future = owned.start(fail_late(), on_settled=outcomes.append)
    release.set()
    result = await owned.wait_until(future, deadline=asyncio.get_running_loop().time() + 1)

    assert result.completed is True
    assert outcomes[0].error is error


@pytest.mark.anyio
async def test_wait_until_keeps_first_of_repeated_cancellation_arguments():
    release = asyncio.Event()
    future = asyncio.create_task(release.wait())
    owned = OwnedTaskSet()
    owned.retain(future, on_settled=lambda outcome: None)
    waiter = asyncio.create_task(owned.wait_until(future, deadline=asyncio.get_running_loop().time() + 1))

    await asyncio.sleep(0)
    waiter.cancel("first")
    await asyncio.sleep(0)
    waiter.cancel("second")
    result = await waiter

    assert result.completed is False
    assert result.cancellation_args == ("first",)

    release.set()
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)


@pytest.mark.anyio
async def test_wait_until_handles_empty_cancellation_arguments():
    release = asyncio.Event()
    future = asyncio.create_task(release.wait())
    owned = OwnedTaskSet()
    owned.retain(future, on_settled=lambda outcome: None)
    waiter = asyncio.create_task(owned.wait_until(future, deadline=asyncio.get_running_loop().time() + 1))

    await asyncio.sleep(0)
    waiter.cancel()
    result = await waiter

    assert result.cancellation_args == ()

    release.set()
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)


@pytest.mark.anyio
async def test_drain_returns_unsettled_owned_futures_after_expired_deadline():
    release = asyncio.Event()
    future = asyncio.create_task(release.wait())
    owned = OwnedTaskSet()
    owned.retain(future, on_settled=lambda outcome: None)

    result = await owned.drain_all_until(deadline=asyncio.get_running_loop().time())

    assert result.pending == frozenset({future})
    assert isinstance(result.pending, frozenset)
    assert result.cancellation_args is None

    release.set()
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)


@pytest.mark.anyio
async def test_drain_reaches_fixed_point_for_task_registered_by_callback():
    outcomes = []
    owned = OwnedTaskSet()

    def on_first_settled(outcome):
        outcomes.append(outcome.result)
        owned.start(asyncio.sleep(0, result="second"), on_settled=on_second_settled)

    def on_second_settled(outcome):
        outcomes.append(outcome.result)

    owned.start(asyncio.sleep(0, result="first"), on_settled=on_first_settled)
    result = await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)

    assert result.pending == frozenset()
    assert outcomes == ["first", "second"]


@pytest.mark.anyio
async def test_drain_returns_after_synchronous_callback_completes():
    callback_completed = False
    owned = OwnedTaskSet()

    def on_settled(outcome):
        nonlocal callback_completed
        callback_completed = True

    owned.start(asyncio.sleep(0), on_settled=on_settled)
    result = await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)

    assert result.pending == frozenset()
    assert callback_completed is True


@pytest.mark.anyio
async def test_drain_reports_caller_cancellation_and_keeps_ownership():
    release = asyncio.Event()
    future = asyncio.create_task(release.wait())
    owned = OwnedTaskSet()
    owned.retain(future, on_settled=lambda outcome: None)
    drain = asyncio.create_task(owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1))

    await asyncio.sleep(0)
    drain.cancel("caller")
    result = await drain

    assert result.pending == frozenset({future})
    assert result.cancellation_args == ("caller",)

    release.set()
    await owned.drain_all_until(deadline=asyncio.get_running_loop().time() + 1)
