"""Placeholder recording watchdog (CDP screencast removed)."""

from __future__ import annotations

import logging
from typing import ClassVar

from bubus import BaseEvent

from browser_use.browser.events import BrowserConnectedEvent, BrowserStopEvent
from browser_use.browser.watchdog_base import BaseWatchdog


class RecordingWatchdog(BaseWatchdog):
	"""
	Recording is now handled via Playwright video capture in the replay service.
	This watchdog remains attached to avoid breaking session wiring but performs no CDP screencast work.
	"""

	LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [BrowserConnectedEvent, BrowserStopEvent]
	EMITS: ClassVar[list[type[BaseEvent]]] = []

	async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:  # pragma: no cover - passive
		if self.browser_session.browser_profile.record_video_dir:
			self.logger.info(
				"[recording] Playwright-based recording is handled externally; CDP screencast is disabled for this session."
			)

	async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:  # pragma: no cover - passive
		self.logger.debug("[recording] stop event received; no recorder to flush.")

	async def pause_recording(self) -> None:
		self.logger.debug("[recording] pause requested; no-op (screencast disabled).")

	async def resume_recording(self) -> None:
		self.logger.debug("[recording] resume requested; no-op (screencast disabled).")
