import asyncio

import pytest

from deerflow.runtime.cancellation import wait_for_task_until


@pytest.mark.anyio
async def test_wait_for_task_until_reports_completion():
    child = asyncio.create_task(asyncio.sleep(0, result="done"))

    completed = await wait_for_task_until(child, deadline=asyncio.get_running_loop().time() + 1)

    assert completed is True
    assert child.result() == "done"


@pytest.mark.anyio
async def test_wait_for_task_until_times_out_without_cancelling_child():
    event = asyncio.Event()
    child = asyncio.create_task(event.wait())

    completed = await wait_for_task_until(child, deadline=asyncio.get_running_loop().time() + 0.01)

    assert completed is False
    assert child.done() is False
    event.set()
    await child


@pytest.mark.anyio
async def test_wait_for_task_until_zero_budget_returns_immediately():
    event = asyncio.Event()
    child = asyncio.create_task(event.wait())

    completed = await wait_for_task_until(child, deadline=asyncio.get_running_loop().time())

    assert completed is False
    assert child.done() is False
    child.cancel()
    with pytest.raises(asyncio.CancelledError):
        await child


@pytest.mark.anyio
async def test_wait_for_task_until_repeated_cancellation_keeps_original_deadline(monkeypatch):
    loop = asyncio.get_running_loop()
    clock = iter((0.0, 0.01, 0.02, 0.05))
    monkeypatch.setattr(loop, "time", lambda: next(clock, 0.05))

    wait_timeouts = []

    async def fake_wait(tasks, *, timeout):
        del tasks
        wait_timeouts.append(timeout)
        if len(wait_timeouts) < 3:
            raise asyncio.CancelledError
        return set(), set()

    monkeypatch.setattr(asyncio, "wait", fake_wait)
    event = asyncio.Event()
    child = asyncio.create_task(event.wait())

    completed = await wait_for_task_until(child, deadline=0.05)

    assert completed is False
    assert wait_timeouts == pytest.approx([0.05, 0.04, 0.03])
    assert child.done() is False
    event.set()
    await child
