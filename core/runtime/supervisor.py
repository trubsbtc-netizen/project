from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(slots=True)
class ManagedTask:
    name: str
    task: asyncio.Task[object]
    critical: bool = True


class TaskSupervisor:
    def __init__(self) -> None:
        self._tasks: dict[str, ManagedTask] = {}
        self._stop = asyncio.Event()
        self._errors: list[BaseException] = []

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_stop)

    def request_stop(self) -> None:
        self._stop.set()

    def create(self, name: str, coro: Awaitable[object], critical: bool = True) -> asyncio.Task[object]:
        task = asyncio.create_task(coro, name=name)
        managed = ManagedTask(name=name, task=task, critical=critical)
        self._tasks[name] = managed
        task.add_done_callback(lambda done, n=name: self._on_done(n, done))
        return task

    def _on_done(self, name: str, task: asyncio.Task[object]) -> None:
        managed = self._tasks.get(name)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._errors.append(exc)
            if managed is None or managed.critical:
                self.request_stop()

    async def sleep_until_stop(self) -> None:
        await self._stop.wait()

    async def cancel_all(self, timeout: float = 5.0) -> None:
        for managed in self._tasks.values():
            managed.task.cancel()
        if not self._tasks:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*(m.task for m in self._tasks.values()), return_exceptions=True),
                timeout=timeout,
            )

    async def run(self, starter: Callable[["TaskSupervisor"], Awaitable[None]]) -> None:
        self.install_signal_handlers()
        await starter(self)
        await self.sleep_until_stop()
        await self.cancel_all()
        if self._errors:
            raise self._errors[0]

