from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Optional

from browser_use import Browser
from browser_use.browser.events import ClickElementEvent, NavigateToUrlEvent, ScrollEvent, SendKeysEvent, TypeTextEvent

from .service_agent import Session
from .service_browser import launch_chrome
from .service_config import RECORDINGS_BASE
from .service_models import ActionTraceEntry, StartReq


async def _refresh_dom_snapshot(browser: Browser) -> None:
    """Best-effort DOM refresh. In replay mode we skip the snapshot to avoid slow handlers."""
    return  # Skip heavy DOM snapshot during replay to prevent timeouts


async def _apply_replay_action(browser: Browser, entry: ActionTraceEntry, logger: logging.Logger) -> tuple[bool, Optional[str]]:
    """Apply a single recorded action against the browser."""
    action = (entry.action or "").lower()
    params = entry.params or {}

    async def _ensure_on_page() -> None:
        """Navigate to the recorded page_url if provided and different from current."""
        if not entry.page_url:
            return
        with contextlib.suppress(Exception):
            current_url = await browser.get_current_page_url()
            if current_url != entry.page_url:
                nav_event = browser.event_bus.dispatch(NavigateToUrlEvent(url=entry.page_url, new_tab=False))
                await nav_event
                with contextlib.suppress(Exception):
                    await nav_event.event_result(raise_if_any=False, raise_if_none=False)
                await asyncio.sleep(1.5)

    async def _click_via_xpath(xpath: str) -> tuple[bool, Optional[str]]:
        """Fallback click using XPath evaluation inside the page."""
        try:
            cdp_session = await browser.get_or_create_cdp_session()
            expr = f"""
            (function() {{
                const xp = {json.dumps(xpath)};
                const res = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                const el = res.singleNodeValue;
                if (!el) return {{ok:false, reason:'xpath not found'}};
                try {{
                    el.scrollIntoView({{block:'center', inline:'center', behavior:'instant'}});
                    const rect = el.getBoundingClientRect();
                    const x = rect.left + rect.width / 2;
                    const y = rect.top + rect.height / 2;
                    const evt = new MouseEvent('click', {{bubbles:true, cancelable:true, view:window, clientX:x, clientY:y}});
                    el.dispatchEvent(evt);
                    return {{ok:true}};
                }} catch (e) {{
                    return {{ok:false, reason:String(e)}};
                }}
            }})();"""
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True}, session_id=cdp_session.session_id
            )
            val = (res.get("result") or {}).get("value") or {}
            return bool(val.get("ok")), val.get("reason")
        except Exception as exc:  # pragma: no cover - best effort
            return False, str(exc)

    async def _input_via_xpath(xpath: str, text: str, clear: bool) -> tuple[bool, Optional[str]]:
        """Fallback input using XPath evaluation inside the page."""
        try:
            cdp_session = await browser.get_or_create_cdp_session()
            expr = f"""
            (function() {{
                const xp = {json.dumps(xpath)};
                const text = {json.dumps(text)};
                const clear = {json.dumps(bool(clear))};
                const res = document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                const el = res.singleNodeValue;
                if (!el) return {{ok:false, reason:'xpath not found'}};
                try {{
                    el.scrollIntoView({{block:'center', inline:'center', behavior:'instant'}});
                    el.focus();
                    if (clear && 'value' in el) el.value = '';
                    if ('value' in el) {{
                        el.value = text;
                        el.dispatchEvent(new Event('input', {{bubbles:true}}));
                        el.dispatchEvent(new Event('change', {{bubbles:true}}));
                    }} else {{
                        el.textContent = text;
                    }}
                    return {{ok:true}};
                }} catch (e) {{
                    return {{ok:false, reason:String(e)}};
                }}
            }})();"""
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True}, session_id=cdp_session.session_id
            )
            val = (res.get("result") or {}).get("value") or {}
            return bool(val.get("ok")), val.get("reason")
        except Exception as exc:  # pragma: no cover - best effort
            return False, str(exc)

    async def _wait_for_xpath(xpath: str, timeout: float = 10.0, interval: float = 0.5) -> bool:
        """Wait until an element matching the XPath exists or timeout."""
        try:
            cdp_session = await browser.get_or_create_cdp_session()
        except Exception:
            return False

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            try:
                res = await cdp_session.cdp_client.send.Runtime.evaluate(
                    params={
                        "expression": f"Boolean(document.evaluate({json.dumps(xpath)}, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue)",
                        "returnByValue": True,
                    },
                    session_id=cdp_session.session_id,
                )
                val = (res.get("result") or {}).get("value")
                if bool(val):
                    return True
            except Exception:
                pass
            if asyncio.get_event_loop().time() >= deadline:
                return False
            await asyncio.sleep(interval)

    try:
        if action == "navigate":
            url = params.get("url") or entry.value or entry.page_url
            if not url:
                return False, "missing url"
            event = browser.event_bus.dispatch(NavigateToUrlEvent(url=str(url), new_tab=bool(params.get("new_tab"))))
            await event
            with contextlib.suppress(Exception):
                await event.event_result(raise_if_any=False, raise_if_none=False)
            await asyncio.sleep(1.0)
            await _refresh_dom_snapshot(browser)
            return True, None

        if action == "scroll":
            await _ensure_on_page()
            direction = "down" if params.get("down", True) else "up"
            pages_raw = params.get("pages", 1) or 1
            try:
                pages = float(pages_raw)
            except Exception:
                pages = 1.0
            amount = max(200, int(pages * 800))
            node = None
            if params.get("index") is not None:
                with contextlib.suppress(Exception):
                    node = await browser.get_element_by_index(int(params["index"]))
            event = browser.event_bus.dispatch(ScrollEvent(direction=direction, amount=amount, node=node))
            await event
            with contextlib.suppress(Exception):
                await event.event_result(raise_if_any=False, raise_if_none=False)
            return True, None

        if action == "click":
            await _ensure_on_page()
            idx = params.get("index")
            node = None
            if idx is not None:
                node = await browser.get_element_by_index(int(idx))
            if node:
                event = browser.event_bus.dispatch(ClickElementEvent(node=node))
                await event
                with contextlib.suppress(Exception):
                    await event.event_result(raise_if_any=False, raise_if_none=False)
                return True, None
            if entry.xpath:
                if not await _wait_for_xpath(entry.xpath, timeout=10.0, interval=0.5):
                    return False, f"xpath {entry.xpath}: not found after wait"
                ok, reason = await _click_via_xpath(entry.xpath)
                if ok:
                    return True, None
                return False, f"xpath {entry.xpath}: {reason or 'failed'}"
            return False, f"index {idx} not found"

        if action in ("input", "type", "write"):
            await _ensure_on_page()
            idx = params.get("index")
            text = params.get("text") or entry.value
            clear = params.get("clear", True)
            if text is None:
                return False, "missing input text"
            node = None
            if idx is not None:
                node = await browser.get_element_by_index(int(idx))
            if node:
                event = browser.event_bus.dispatch(TypeTextEvent(node=node, text=str(text), clear=bool(clear), is_sensitive=False))
                await event
                with contextlib.suppress(Exception):
                    await event.event_result(raise_if_any=False, raise_if_none=False)
                return True, None
            if entry.xpath:
                if not await _wait_for_xpath(entry.xpath, timeout=10.0, interval=0.5):
                    return False, f"xpath {entry.xpath}: not found after wait"
                ok, reason = await _input_via_xpath(entry.xpath, str(text), bool(clear))
                if ok:
                    return True, None
                return False, f"xpath {entry.xpath}: {reason or 'failed'}"
            return False, f"index {idx} not found"

        if action == "wait":
            seconds_raw = params.get("seconds") or params.get("duration") or 1
            try:
                seconds = float(seconds_raw)
            except Exception:
                seconds = 1.0
            await asyncio.sleep(max(0.0, seconds))
            return True, None

        if action == "send_keys":
            keys = params.get("keys") or entry.value
            if not keys:
                return False, "missing keys"
            event = browser.event_bus.dispatch(SendKeysEvent(keys=str(keys)))
            await event
            with contextlib.suppress(Exception):
                await event.event_result(raise_if_any=False, raise_if_none=False)
            return True, None

        if action in ("extract", "done"):
            return True, None

        return False, f"unsupported action {entry.action}"

    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Replay action %s (order %s) failed: %s", entry.action, entry.order, exc)
        return False, str(exc)


async def _start_replay_browser(session_id: str, headless: bool = True) -> tuple[Browser, Session, Path]:
    """Launch a fresh Chrome instance for replay and return the Browser + temp session."""
    dummy_req = StartReq(task=f"replay {session_id}", headless=headless)
    replay_session = Session(session_id=f"{session_id}-replay", req=dummy_req)
    ws = await launch_chrome(replay_session)

    video_dir = RECORDINGS_BASE / session_id / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    browser = Browser(
        cdp_url=ws,
        headless=headless,
        record_video_dir=video_dir,
        highlight_elements=False,
        wait_between_actions=0.2,
    )
    await browser.start()
    await _refresh_dom_snapshot(browser)
    return browser, replay_session, video_dir


def _detect_new_video_file(video_dir: Path, before: set[Path]) -> Optional[Path]:
    """Determine which video file was created after replay."""
    after = {p.resolve() for p in video_dir.glob("*") if p.is_file()}
    new_files = [p for p in after if p not in before]
    if new_files:
        with contextlib.suppress(Exception):
            new_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return new_files[0]
    if after:
        with contextlib.suppress(Exception):
            latest = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
            return latest[0]
    return None


async def replay_action_trace_to_video(session_id: str, entries: list[ActionTraceEntry]) -> tuple[Optional[Path], int, list[str]]:
    """Replay the recorded actions into a video. Returns (video_path, applied_count, skipped_actions)."""
    logger = logging.getLogger("service")
    if not entries:
        return None, 0, []

    browser: Optional[Browser] = None
    replay_session: Optional[Session] = None
    video_dir: Optional[Path] = None
    applied = 0
    skipped: list[str] = []

    try:
        browser, replay_session, video_dir = await _start_replay_browser(session_id, headless=True)
        before_files = {p.resolve() for p in video_dir.glob("*") if p.is_file()}

        initialized_url = False
        for entry in entries:
            if not initialized_url and entry.page_url:
                try:
                    await browser.goto(entry.page_url)
                    initialized_url = True
                except Exception:
                    pass
            ok, reason = await _apply_replay_action(browser, entry, logger)
            if ok:
                applied += 1
            else:
                skipped.append(f"{entry.action} (step {entry.step}): {reason or 'failed'}")

        await asyncio.sleep(1.0)
        await browser.stop()
        browser = None

        video_path = _detect_new_video_file(video_dir, before_files) if video_dir else None
        return video_path, applied, skipped

    except Exception as exc:  # pragma: no cover - best effort
        logger.error("replay video failed: %s", exc)
        return None, applied, skipped
    finally:
        with contextlib.suppress(Exception):
            if browser:
                await browser.stop()
        with contextlib.suppress(Exception):
            if replay_session and replay_session.chrome_proc and replay_session.chrome_proc.returncode is None:
                replay_session.chrome_proc.terminate()
        if video_dir and video_dir.exists():
            pass
