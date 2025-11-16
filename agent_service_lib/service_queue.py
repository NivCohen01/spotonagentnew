from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from typing import Awaitable, Callable, Optional


class QueueManager:
    """
    FIFO queue backed by a deque + N worker tasks.
    - start(): spawn N workers
    - stop():  cancel workers
    - enqueue(sid): append to queue; returns 1-based position
    - position(sid): get 1-based position or None
    """

    def __init__(
        self,
        workers_count: int,
        session_runner: Callable[[object], Awaitable[None]],
        session_lookup: Callable[[str], object],
    ):
        self.workers_count = workers_count
        self._session_runner = session_runner
        self._session_lookup = session_lookup
        self._queue: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._workers: list[asyncio.Task] = []
        self._stopping = False

    async def start(self):
        for i in range(self.workers_count):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def stop(self):
        self._stopping = True
        async with self._not_empty:
            self._not_empty.notify_all()
        for task in self._workers:
            task.cancel()
        for task in self._workers:
            with contextlib.suppress(Exception):
                await task

    async def enqueue(self, sid: str) -> int:
        async with self._not_empty:
            self._queue.append(sid)
            pos = len(self._queue)
            self._not_empty.notify()
            return pos

    async def remove_if_present(self, sid: str) -> bool:
        async with self._lock:
            try:
                self._queue.remove(sid)
                return True
            except ValueError:
                return False

    async def position(self, sid: str) -> Optional[int]:
        async with self._lock:
            try:
                return list(self._queue).index(sid) + 1
            except ValueError:
                return None

    async def _pop(self) -> Optional[str]:
        async with self._not_empty:
            while not self._queue and not self._stopping:
                await self._not_empty.wait()
            if self._stopping:
                return None
            return self._queue.popleft()

    async def _worker_loop(self, worker_id: int):
        log = logging.getLogger("service")
        while True:
            sid = await self._pop()
            if sid is None:
                return
            sess = self._session_lookup(sid)
            if not sess:
                continue
            try:
                await self._session_runner(sess)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("Worker %s crashed running %s: %r", worker_id, sid, exc)

