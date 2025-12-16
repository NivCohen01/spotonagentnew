from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .playwright_replayer import ReplayProfile, replay_trace_to_video
from .service_config import RECORDINGS_BASE
from .service_models import ActionTraceEntry, DeviceType

logger = logging.getLogger("service")


def _pick_initial_url(entries: list[ActionTraceEntry]) -> Optional[str]:
    for entry in entries:
        action = (entry.action or "").lower()
        if action == "navigate":
            params = entry.params or {}
            url = params.get("url") or entry.value or entry.page_url
            if url:
                return str(url)
    for entry in entries:
        if entry.page_url:
            return str(entry.page_url)
    return None


def _sort_entries(entries: list[ActionTraceEntry]) -> list[ActionTraceEntry]:
    try:
        return sorted(entries, key=lambda e: (e.order if e.order is not None else 0))
    except Exception:
        return entries


async def replay_action_trace_to_video(
    session_id: str,
    entries: list[ActionTraceEntry],
    *,
    device_type: DeviceType = "desktop",
    viewport_width: Optional[int] = None,
    viewport_height: Optional[int] = None,
) -> tuple[Optional[Path], int, list[str]]:
    if not entries:
        return None, 0, []

    log = logger
    sorted_entries = _sort_entries(entries)
    initial_url = _pick_initial_url(sorted_entries)

    output_dir = RECORDINGS_BASE / session_id / "video"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = ReplayProfile(
        device_type=device_type,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )

    log.info(
        "[replay] starting Playwright run session=%s entries=%s initial_url=%s dir=%s",
        session_id,
        len(sorted_entries),
        initial_url,
        output_dir,
    )
    mp4_path, applied, skipped = await replay_trace_to_video(
        sorted_entries,
        initial_url,
        output_dir,
        profile,
        logger_instance=log,
    )
    return mp4_path, applied, skipped
