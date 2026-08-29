from __future__ import annotations

import asyncio
from typing import TypeVar

from deerflow.runtime.owned_operations import OwnedTaskSet

T = TypeVar("T")


async def wait_for_task_until(  # noqa: UP047
    task: asyncio.Future[T], *, deadline: float
) -> bool:
    """Compatibility wrapper for the shared absolute-deadline wait loop."""
    return (await OwnedTaskSet().wait_until(task, deadline=deadline)).completed
