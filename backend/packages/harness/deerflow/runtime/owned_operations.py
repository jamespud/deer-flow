from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar
from weakref import ReferenceType, WeakKeyDictionary, ref

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SettledOutcome[T]:
    future: asyncio.Future[T]
    result: T | None
    error: BaseException | None
    cancelled: bool


SettledCallback = Callable[[SettledOutcome[T]], None]


@dataclass(frozen=True, slots=True)
class WaitResult:
    completed: bool
    cancellation_args: tuple[object, ...] | None


@dataclass(frozen=True, slots=True)
class DrainResult:
    pending: frozenset[asyncio.Future[Any]]
    cancellation_args: tuple[object, ...] | None


@dataclass(slots=True)
class _OwnedOperation:
    on_settled: SettledCallback[Any]
    settling: bool = False


@dataclass(frozen=True, slots=True)
class _CallbackBinding:
    callback_ref: ReferenceType[object]

    @classmethod
    def from_callback(cls, callback: SettledCallback[Any]) -> _CallbackBinding:
        return cls(callback_ref=ref(callback))

    def matches(self, callback: SettledCallback[Any]) -> bool:
        return self.callback_ref() is callback


class OwnedTaskSet:
    """Strongly own asynchronous operations until their settlement callback runs."""

    def __init__(self) -> None:
        self._owned: dict[asyncio.Future[Any], _OwnedOperation] = {}
        self._settlement_callbacks: WeakKeyDictionary[asyncio.Future[Any], _CallbackBinding] = WeakKeyDictionary()

    def start(
        self,
        operation: Coroutine[Any, Any, T],
        *,
        on_settled: SettledCallback[T],
    ) -> asyncio.Task[T]:
        self._validate_callback(on_settled)
        task = asyncio.create_task(operation)
        self.retain(task, on_settled=on_settled)
        return task

    def retain(
        self,
        future: asyncio.Future[T],
        *,
        on_settled: SettledCallback[T],
    ) -> asyncio.Future[T]:
        self._validate_callback(on_settled)

        bound_callback = self._settlement_callbacks.get(future)
        if bound_callback is not None:
            if not bound_callback.matches(on_settled):
                raise ValueError("future already has a different settlement callback")
            return future

        self._settlement_callbacks[future] = _CallbackBinding.from_callback(on_settled)
        self._owned[future] = _OwnedOperation(on_settled=on_settled)
        future.add_done_callback(self._settle)
        if future.done():
            self._settle(future)
        return future

    async def wait_until(
        self,
        task: asyncio.Future[T],
        *,
        deadline: float,
    ) -> WaitResult:
        cancellation_args: tuple[object, ...] | None = None
        loop = asyncio.get_running_loop()

        while not task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return WaitResult(completed=False, cancellation_args=cancellation_args)
            try:
                shielded = asyncio.shield(task)
                shielded.add_done_callback(self._consume_shielded_exception)
                await asyncio.wait({shielded}, timeout=remaining)
            except asyncio.CancelledError as error:
                if cancellation_args is None:
                    cancellation_args = error.args

        self._settle(task)
        return WaitResult(completed=True, cancellation_args=cancellation_args)

    async def drain_all_until(self, *, deadline: float) -> DrainResult:
        cancellation_args: tuple[object, ...] | None = None

        while self._owned:
            for task in tuple(self._owned):
                result = await self.wait_until(task, deadline=deadline)
                if cancellation_args is None and result.cancellation_args is not None:
                    cancellation_args = result.cancellation_args
                if not result.completed:
                    self._settle_completed()
                    return DrainResult(pending=self._pending(), cancellation_args=cancellation_args)

        return DrainResult(pending=frozenset(), cancellation_args=cancellation_args)

    def _settle_completed(self) -> None:
        for task in tuple(self._owned):
            if task.done():
                self._settle(task)

    def _pending(self) -> frozenset[asyncio.Future[Any]]:
        return frozenset(task for task in self._owned if not task.done())

    @staticmethod
    def _validate_callback(on_settled: SettledCallback[Any]) -> None:
        if on_settled is None:
            raise ValueError("on_settled is required")
        if inspect.iscoroutinefunction(on_settled) or inspect.iscoroutinefunction(getattr(on_settled, "__call__", None)):
            raise TypeError("on_settled must be synchronous and non-blocking")
        try:
            ref(on_settled)
        except TypeError as error:
            raise TypeError("on_settled must be weak-referenceable") from error

    @staticmethod
    def _consume_awaitable(awaitable: Any) -> None:
        if isinstance(awaitable, asyncio.Future):
            awaitable.add_done_callback(OwnedTaskSet._consume_shielded_exception)
            return
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _consume_shielded_exception(future: asyncio.Future[Any]) -> None:
        try:
            future.exception()
        except asyncio.CancelledError:
            pass

    def _settle(self, future: asyncio.Future[Any]) -> None:
        owned = self._owned.get(future)
        if owned is None or owned.settling:
            return

        owned.settling = True
        try:
            outcome = self._outcome(future)
            try:
                callback_result = owned.on_settled(outcome)
                if inspect.isawaitable(callback_result):
                    self._consume_awaitable(callback_result)
                    logger.error("Owned task settlement callback returned an awaitable; on_settled must be synchronous and non-blocking")
            except BaseException:
                logger.exception("Owned task settlement callback failed")
        finally:
            self._owned.pop(future, None)

    @staticmethod
    def _outcome(future: asyncio.Future[T]) -> SettledOutcome[T]:
        if future.cancelled():
            try:
                future.exception()
            except asyncio.CancelledError as error:
                return SettledOutcome(
                    future=future,
                    result=None,
                    error=error,
                    cancelled=True,
                )

        error = future.exception()
        if error is not None:
            return SettledOutcome(
                future=future,
                result=None,
                error=error,
                cancelled=False,
            )
        return SettledOutcome(
            future=future,
            result=future.result(),
            error=None,
            cancelled=False,
        )
