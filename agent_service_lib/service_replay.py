from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Literal, Optional

from browser_use import Browser
from browser_use.browser.events import ClickElementEvent, NavigateToUrlEvent, ScrollEvent, SendKeysEvent, TypeTextEvent

from .service_agent import Session
from .service_browser import launch_chrome
from .service_config import RECORDINGS_BASE
from .service_models import ActionTraceEntry, DeviceType, StartReq


async def _refresh_dom_snapshot(browser: Browser) -> None:
    """Best-effort DOM refresh. In replay mode we skip the snapshot to avoid slow handlers."""
    return  # Skip heavy DOM snapshot during replay to prevent timeouts


async def _apply_replay_action(browser: Browser, entry: ActionTraceEntry, logger: logging.Logger) -> tuple[bool, Optional[str]]:
    """Apply a single recorded action against the browser."""
    action = (entry.action or "").lower()
    params = entry.params or {}
    metadata_attrs = {k: v for k, v in (entry.element_attributes or {}).items() if v not in (None, "")}
    metadata_text = (entry.element_text or "").strip()
    metadata_tag = (entry.element_tag or "").lower().strip()

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

    async def _wait_for_metadata(timeout: float = 10.0, interval: float = 0.5) -> bool:
        """Wait for an element that matches recorded tag/text/attributes when XPath is unreliable."""
        if not (metadata_tag or metadata_text or metadata_attrs):
            return False
        try:
            cdp_session = await browser.get_or_create_cdp_session()
        except Exception:
            return False

        deadline = asyncio.get_event_loop().time() + timeout
        query = f"""
        (function() {{
            const tag = {json.dumps(metadata_tag)} || '*';
            const attrs = {json.dumps(metadata_attrs)};
            const normText = {json.dumps(metadata_text)}.toLowerCase();
            const candidates = Array.from(document.querySelectorAll(tag)).slice(0, 500);
            function score(el) {{
                let s = 0;
                if (normText) {{
                    const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                    if (t === normText) s += 3;
                    else if (t.includes(normText)) s += 1;
                }}
                for (const [k, v] of Object.entries(attrs || {{}})) {{
                    if (!v) continue;
                    const val = el.getAttribute(k);
                    if (val === null) continue;
                    const a = String(val).toLowerCase();
                    const b = String(v).toLowerCase();
                    if (a === b) s += 2;
                    else if (a.includes(b)) s += 1;
                }}
                return s;
            }}
            for (const el of candidates) {{
                if (score(el) > 0) return true;
            }}
            return false;
        }})();
        """

        while True:
            try:
                res = await cdp_session.cdp_client.send.Runtime.evaluate(
                    params={"expression": query, "returnByValue": True}, session_id=cdp_session.session_id
                )
                val = (res.get("result") or {}).get("value")
                if bool(val):
                    return True
            except Exception:
                pass
            if asyncio.get_event_loop().time() >= deadline:
                return False
            await asyncio.sleep(interval)

    async def _click_via_metadata() -> tuple[bool, Optional[str]]:
        """Fallback click using recorded tag/text/attributes when XPath/index are invalid."""
        if not (metadata_tag or metadata_text or metadata_attrs):
            return False, "no metadata available"
        try:
            cdp_session = await browser.get_or_create_cdp_session()
            expr = f"""
            (function() {{
                const tag = {json.dumps(metadata_tag)} || '*';
                const attrs = {json.dumps(metadata_attrs)};
                const normText = {json.dumps(metadata_text)}.toLowerCase();
                const candidates = Array.from(document.querySelectorAll(tag)).slice(0, 800);
                function score(el) {{
                    let s = 0;
                    if (normText) {{
                        const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                        if (t === normText) s += 4;
                        else if (t.includes(normText)) s += 2;
                    }}
                    for (const [k, v] of Object.entries(attrs || {{}})) {{
                        if (!v) continue;
                        const val = el.getAttribute(k);
                        if (val === null) continue;
                        const a = String(val).toLowerCase();
                        const b = String(v).toLowerCase();
                        if (a === b) s += 3;
                        else if (a.includes(b)) s += 1;
                    }}
                    const tagName = (el.tagName || '').toLowerCase();
                    if (tagName === 'button' || tagName === 'a' || el.onclick) s += 1;
                    return s;
                }}
                let best = null;
                let bestScore = -1;
                for (const el of candidates) {{
                    const sc = score(el);
                    if (sc > bestScore) {{ best = el; bestScore = sc; }}
                }}
                if (!best || bestScore < 1) return {{ok:false, reason:'no metadata match'}};
                try {{
                    best.scrollIntoView({{block:'center', inline:'center', behavior:'instant'}});
                    const rect = best.getBoundingClientRect();
                    const x = rect.left + rect.width / 2;
                    const y = rect.top + rect.height / 2;
                    const evt = new MouseEvent('click', {{bubbles:true, cancelable:true, view:window, clientX:x, clientY:y}});
                    best.dispatchEvent(evt);
                    return {{ok:true}};
                }} catch (e) {{
                    return {{ok:false, reason:String(e)}};
                }}
            }})();
            """
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True}, session_id=cdp_session.session_id
            )
            val = (res.get("result") or {}).get("value") or {}
            return bool(val.get("ok")), val.get("reason")
        except Exception as exc:  # pragma: no cover - best effort
            return False, str(exc)

    async def _input_via_metadata(text: str, clear: bool) -> tuple[bool, Optional[str]]:
        """Fallback input using recorded metadata when XPath/index are invalid."""
        if not (metadata_tag or metadata_text or metadata_attrs):
            return False, "no metadata available"
        try:
            cdp_session = await browser.get_or_create_cdp_session()
            expr = f"""
            (function() {{
                const tag = {json.dumps(metadata_tag)} || '*';
                const attrs = {json.dumps(metadata_attrs)};
                const normText = {json.dumps(metadata_text)}.toLowerCase();
                const newValue = {json.dumps(text)};
                const shouldClear = {json.dumps(bool(clear))};
                const candidates = Array.from(document.querySelectorAll(tag)).slice(0, 800);
                function score(el) {{
                    let s = 0;
                    if (normText) {{
                        const t = (el.innerText || el.textContent || '').trim().toLowerCase();
                        if (t === normText) s += 4;
                        else if (t.includes(normText)) s += 2;
                    }}
                    for (const [k, v] of Object.entries(attrs || {{}})) {{
                        if (!v) continue;
                        const val = el.getAttribute(k);
                        if (val === null) continue;
                        const a = String(val).toLowerCase();
                        const b = String(v).toLowerCase();
                        if (a === b) s += 3;
                        else if (a.includes(b)) s += 1;
                    }}
                    const tagName = (el.tagName || '').toLowerCase();
                    if (tagName === 'input' || tagName === 'textarea' || el.isContentEditable) s += 1;
                    return s;
                }}
                let best = null;
                let bestScore = -1;
                for (const el of candidates) {{
                    const sc = score(el);
                    if (sc > bestScore) {{ best = el; bestScore = sc; }}
                }}
                if (!best || bestScore < 1) return {{ok:false, reason:'no metadata match'}};
                try {{
                    best.scrollIntoView({{block:'center', inline:'center', behavior:'instant'}});
                    best.focus();
                    if (shouldClear && 'value' in best) best.value = '';
                    if ('value' in best) {{
                        best.value = newValue;
                        best.dispatchEvent(new Event('input', {{bubbles:true}}));
                        best.dispatchEvent(new Event('change', {{bubbles:true}}));
                    }} else {{
                        best.textContent = newValue;
                    }}
                    return {{ok:true}};
                }} catch (e) {{
                    return {{ok:false, reason:String(e)}};
                }}
            }})();
            """
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True}, session_id=cdp_session.session_id
            )
            val = (res.get("result") or {}).get("value") or {}
            return bool(val.get("ok")), val.get("reason")
        except Exception as exc:  # pragma: no cover - best effort
            return False, str(exc)

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
            xpath_reason = None
            if entry.xpath:
                if not await _wait_for_xpath(entry.xpath, timeout=10.0, interval=0.5):
                    xpath_reason = f"xpath {entry.xpath}: not found after wait"
                else:
                    ok, reason = await _click_via_xpath(entry.xpath)
                    if ok:
                        return True, None
                    xpath_reason = f"xpath {entry.xpath}: {reason or 'failed'}"
            if await _wait_for_metadata(timeout=10.0, interval=0.5):
                ok, reason = await _click_via_metadata()
                if ok:
                    return True, None
                return False, reason or xpath_reason or f"index {idx} not found"
            if xpath_reason:
                return False, xpath_reason
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
            xpath_reason = None
            if entry.xpath:
                if not await _wait_for_xpath(entry.xpath, timeout=10.0, interval=0.5):
                    xpath_reason = f"xpath {entry.xpath}: not found after wait"
                else:
                    ok, reason = await _input_via_xpath(entry.xpath, str(text), bool(clear))
                    if ok:
                        return True, None
                    xpath_reason = f"xpath {entry.xpath}: {reason or 'failed'}"
            if await _wait_for_metadata(timeout=10.0, interval=0.5):
                ok, reason = await _input_via_metadata(str(text), bool(clear))
                if ok:
                    return True, None
                return False, reason or xpath_reason or f"index {idx} not found"
            if xpath_reason:
                return False, xpath_reason
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


def _resolve_replay_viewport(device_type: DeviceType, viewport_width: Optional[int], viewport_height: Optional[int]) -> tuple[dict | None, float | None]:
    """Return viewport and device scale settings to mirror the original run."""
    if device_type == "desktop":
        return {"width": 1920, "height": 1080}, 1.0
    if device_type == "mobile":
        return {"width": 390, "height": 844}, 3.0
    if device_type == "custom":
        if viewport_width and viewport_height:
            return {"width": int(viewport_width), "height": int(viewport_height)}, 1.0
        # Fallback if custom requested without dims
        return {"width": 1280, "height": 720}, 1.0
    return None, None


async def _start_replay_browser(
    session_id: str,
    *,
    headless: bool = True,
    device_type: DeviceType = "desktop",
    viewport_width: Optional[int] = None,
    viewport_height: Optional[int] = None,
) -> tuple[Browser, Session, Path]:
    """Launch a fresh Chrome instance for replay and return the Browser + temp session."""
    dummy_req = StartReq(task=f"replay {session_id}", headless=headless)
    replay_session = Session(session_id=f"{session_id}-replay", req=dummy_req)
    ws = await launch_chrome(replay_session)

    video_dir = RECORDINGS_BASE / session_id / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    viewport, device_scale_factor = _resolve_replay_viewport(device_type, viewport_width, viewport_height)
    browser = Browser(
        cdp_url=ws,
        headless=headless,
        record_video_dir=video_dir,
        highlight_elements=False,
        wait_between_actions=0.2,
        viewport=viewport,
        device_scale_factor=device_scale_factor,
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


async def replay_action_trace_to_video(
    session_id: str,
    entries: list[ActionTraceEntry],
    *,
    device_type: DeviceType = "desktop",
    viewport_width: Optional[int] = None,
    viewport_height: Optional[int] = None,
) -> tuple[Optional[Path], int, list[str]]:
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
        browser, replay_session, video_dir = await _start_replay_browser(
            session_id,
            headless=True,
            device_type=device_type,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        before_files = {p.resolve() for p in video_dir.glob("*") if p.is_file()}

        # Navigate to the best initial URL before replaying actions so the recording starts with the real page load
        initial_url: Optional[str] = None
        for candidate in entries:
            if (candidate.action or "").lower() == "navigate":
                initial_url = str(candidate.params.get("url") or candidate.value or candidate.page_url or "")
                if initial_url:
                    break
            if not initial_url and candidate.page_url:
                initial_url = candidate.page_url
                break

        initialized_url = False
        if initial_url:
            with contextlib.suppress(Exception):
                await browser.goto(initial_url)
                initialized_url = True

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
