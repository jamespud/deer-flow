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
async def test_wait_for_task_until_repeated_cancellation_keeps_original_deadline():
    event = asyncio.Event()
    child = asyncio.create_task(event.wait())
    deadline = asyncio.get_running_loop().time() + 0.05
    waiter = asyncio.create_task(wait_for_task_until(child, deadline=deadline))

    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0.02)
    waiter.cancel()

    completed = await asyncio.wait_for(waiter, timeout=0.2)

    assert completed is False
    assert asyncio.get_running_loop().time() < deadline + 0.1
    assert child.done() is False
    event.set()
    await child
