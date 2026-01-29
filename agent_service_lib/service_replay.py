from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .playwright_replayer import ReplayProfile, RequiredActionError, replay_trace_to_video
from .service_config import RECORDINGS_BASE
from .service_models import ActionTraceEntry, DeviceType

logger = logging.getLogger("service")


def _replace_placeholders(obj, email: Optional[str], password: Optional[str]):
    """
    Replace sanitized placeholders in strings/containers with provided credentials.
    Only used for replay; does not mutate original objects.
    """
    if obj is None:
        return obj

    def _swap(val: str) -> str:
        out = val
        if email:
            out = out.replace("<your email>", email)
        if password:
            out = out.replace("<your password>", password)
        return out

    if isinstance(obj, str):
        return _swap(obj)
    if isinstance(obj, list):
        return [_replace_placeholders(item, email, password) for item in obj]
    if isinstance(obj, dict):
        return {k: _replace_placeholders(v, email, password) for k, v in obj.items()}
    return obj


def _unsanitize_entries(entries: list[ActionTraceEntry], email: Optional[str], password: Optional[str]) -> list[ActionTraceEntry]:
    """
    Clone entries and replace placeholders with provided credentials for replay purposes.
    """
    if not email and not password:
        return entries

    cleaned: list[ActionTraceEntry] = []
    for entry in entries:
        try:
            data = entry.model_dump()
            data = _replace_placeholders(data, email, password)
            cleaned.append(ActionTraceEntry.model_validate(data))
        except Exception:
            cleaned.append(entry)
    return cleaned


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
    full_entries: Optional[list[ActionTraceEntry]] = None,
    otp_email: Optional[str] = None,
    otp_password: Optional[str] = None,
) -> tuple[Optional[Path], int, list[str]]:
    base_entries = full_entries or entries
    if not base_entries:
        logger.warning("[replay] no action trace entries for session=%s", session_id)
        return None, 0, []

    log = logger
    ordered_entries = _sort_entries(_unsanitize_entries(base_entries, otp_email, otp_password))
    optional_orders = {e.order for e in ordered_entries if int(getattr(e, "relevance", 1) or 1) == 0}
    included_optional: set[int] = set()
    max_attempts = len(optional_orders) + 1

    output_dir = RECORDINGS_BASE / session_id / "video"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = ReplayProfile(
        device_type=device_type,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )

    attempt = 0
    last_error: Optional[Exception] = None
    while attempt < max_attempts:
        attempt += 1
        candidate_entries = [
            e for e in ordered_entries if int(getattr(e, "relevance", 1) or 1) == 1 or e.order in included_optional
        ]
        if not candidate_entries:
            candidate_entries = list(ordered_entries)
        candidate_entries = _sort_entries(candidate_entries)
        initial_url = _pick_initial_url(candidate_entries)

        log.info(
            "[replay] attempt=%s session=%s entries=%s initial_url=%s dir=%s",
            attempt,
            session_id,
            len(candidate_entries),
            initial_url,
            output_dir,
        )
        try:
            mp4_path, applied, skipped = await replay_trace_to_video(
                candidate_entries,
                initial_url,
                output_dir,
                profile,
                logger_instance=log,
                stop_on_required_failure=True,
                otp_email=otp_email,
                otp_password=otp_password,
            )
            return mp4_path, applied, skipped
        except RequiredActionError as exc:
            last_error = exc
            next_optional = None
            for entry in reversed(ordered_entries):
                if entry.order >= exc.order:
                    continue
                if int(getattr(entry, "relevance", 1) or 1) != 0:
                    continue
                if entry.order in included_optional:
                    continue
                next_optional = entry
                break
            if not next_optional:
                log.warning("[replay] required action failed with no optional fallback: %s", exc)
                break
            included_optional.add(next_optional.order)
            log.info(
                "[replay] adding optional action order=%s before retry (failed order=%s)",
                next_optional.order,
                exc.order,
            )
        except Exception as exc:  # pragma: no cover - fallback
            last_error = exc
            log.warning("[replay] replay failed: %s", exc)
            break

    if last_error:
        log.warning("[replay] final failure after retries: %s", last_error)
    return None, 0, []
