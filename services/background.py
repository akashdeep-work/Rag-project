from __future__ import annotations
import asyncio
from typing import Optional

from indexer import Indexer


class BackgroundIndexer:
    """Periodically scans the data directory for new files and indexes them."""

    def __init__(self, indexer: Indexer, interval_seconds: int = 60) -> None:
        self.indexer = indexer
        self.interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task:
            return
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.indexer.index_new_files()
            except Exception as exc:  # pragma: no cover - log/monitor in production
                print(f"Background indexing failed: {exc}")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        if self._task:
            self._stop_event.set()
            await self._task
            self._task = None
