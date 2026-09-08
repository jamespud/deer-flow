from __future__ import annotations

import asyncio
from typing import TypeVar

T = TypeVar("T")


async def wait_for_task_until(  # noqa: UP047
    task: asyncio.Future[T], *, deadline: float
) -> bool:
    """Wait through repeated caller cancellation without cancelling task."""
    loop = asyncio.get_running_loop()
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        try:
            done, _ = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            continue
        if task in done:
            return True
    return True


async def wait_for_task_until_capturing_cancellation(  # noqa: UP047
    task: asyncio.Future[T], *, deadline: float
) -> tuple[bool, BaseException | None]:
    """Wait through repeated caller cancellation without cancelling *task*.

    Like :func:`wait_for_task_until`, it absorbs caller cancellation; unlike it,
    it returns the first cancellation absorbed so the caller can re-raise the
    original cancellation after a bounded compensation drain. Use this when the
    cancellation must not outrun a cleanup the caller is guaranteed to finish (or
    bound) before the cancellation re-raises.
    """
    captured: BaseException | None = None
    loop = asyncio.get_running_loop()
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False, captured
        try:
            done, _ = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError as exc:
            if captured is None:
                captured = exc
            continue
        if task in done:
            return True, captured
    return True, captured
