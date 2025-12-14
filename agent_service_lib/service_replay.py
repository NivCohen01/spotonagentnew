from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
import time
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


async def _install_screencast_heartbeat(browser: Browser, interval_ms: int = 250) -> None:
    """Inject a tiny off-screen element that toggles to force regular repaints for screencast."""
    try:
        cdp_session = await browser.get_or_create_cdp_session()
        expr = f"""
        (function() {{
            try {{
                if (window.__pathixHeartbeatInstalled) return;
                window.__pathixHeartbeatInstalled = true;

                if (!document.getElementById('__pathix_cursor_styles')) {{
                    var style = document.createElement('style');
                    style.id = '__pathix_cursor_styles';
                    style.textContent = `@keyframes pathixCursorPulse {{
  0%   {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.45); transform: scale(1); }}
  60%  {{ box-shadow: 0 0 0 12px rgba(239,68,68,0); transform: scale(1.05); }}
  100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0); transform: scale(1); }}
}}

#__pathix_cursor {{
  position: fixed;
  width: 24px;
  height: 24px;
  border-radius: 9999px;
  border: 2px solid #ef4444;
  background: rgba(239,68,68,0.1);
  z-index: 2147483647;
  pointer-events: none;
  box-shadow: 0 0 0 0 rgba(239,68,68,0.35);
  animation: pathixCursorPulse 1.1s infinite ease-out;
}}

#__pathix_click_overlay {{
  position: fixed;
  border: 2px solid #ef4444;
  border-radius: 6px;
  background: rgba(239,68,68,0.14);
  pointer-events: none;
  z-index: 2147483646;
  opacity: 0;
  transition: opacity 0.25s ease-out;
}}`;
                    document.head.appendChild(style);
                }}

                var hb = document.getElementById('__pathix_heartbeat');
                if (!hb) {{
                    hb = document.createElement('div');
                    hb.id = '__pathix_heartbeat';
                    hb.style.position = 'fixed';
                    hb.style.left = '-9999px';
                    hb.style.top = '0';
                    hb.style.width = '1px';
                    hb.style.height = '1px';
                    hb.style.backgroundColor = 'transparent';
                    hb.style.pointerEvents = 'none';
                    document.documentElement.appendChild(hb);
                }}
                (function() {{
                    var toggle = false;
                    setInterval(function() {{
                        toggle = !toggle;
                        hb.style.transform = toggle ? 'translateX(0)' : 'translateX(0.0001px)';
                    }}, {interval_ms});
                }})();

                window.__pathixEnsureCursor = function() {{
                    var cur = document.getElementById('__pathix_cursor');
                    if (!cur) {{
                        cur = document.createElement('div');
                        cur.id = '__pathix_cursor';
                        document.documentElement.appendChild(cur);
                        var startX = window.innerWidth / 2 - 12;
                        var startY = window.innerHeight / 2 - 12;
                        cur.style.left = startX + 'px';
                        cur.style.top = startY + 'px';
                    }}
                    return cur;
                }};

                window.__pathixMoveCursor = function(x, y) {{
                    var cur = window.__pathixEnsureCursor();
                    if (!cur) return;
                    cur.style.left = (x - 12) + 'px';
                    cur.style.top = (y - 12) + 'px';
                }};

                window.__pathixAnimateCursorTo = function(targetX, targetY, durationMs) {{
                    try {{
                        var el = window.__pathixEnsureCursor();
                        if (!el) return;
                        var rect = el.getBoundingClientRect();
                        var startLeft = rect.left;
                        var startTop = rect.top;
                        var destLeft = targetX - rect.width / 2;
                        var destTop = targetY - rect.height / 2;
                        var startTime = performance.now();
                        var d = durationMs || 450;
                        function ease(t) {{ return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; }}
                        function step(now) {{
                            var t = Math.min(1, (now - startTime) / d);
                            var e = ease(t);
                            el.style.left = (startLeft + (destLeft - startLeft) * e) + 'px';
                            el.style.top = (startTop + (destTop - startTop) * e) + 'px';
                            if (t < 1) requestAnimationFrame(step);
                        }}
                        requestAnimationFrame(step);
                    }} catch (e) {{}}
                }};

                window.__pathixHighlightAt = function(x, y) {{
                    var overlay = document.getElementById('__pathix_click_overlay');
                    if (!overlay) {{
                        overlay = document.createElement('div');
                        overlay.id = '__pathix_click_overlay';
                        document.documentElement.appendChild(overlay);
                    }}
                    var el = document.elementFromPoint(x, y);
                    var r = el ? el.getBoundingClientRect() : {{ left: x - 20, top: y - 20, width: 40, height: 40 }};
                    overlay.style.left = r.left + 'px';
                    overlay.style.top = r.top + 'px';
                    overlay.style.width = r.width + 'px';
                    overlay.style.height = r.height + 'px';
                    overlay.style.opacity = '1';
                    setTimeout(function() {{ overlay.style.opacity = '0'; }}, 320);
                }};
            }} catch (e) {{}}
        }})();
        """
        await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": expr, "returnByValue": False},
            session_id=cdp_session.session_id,
        )
    except Exception:
        return


async def _wait_for_page_ready(
    browser: Browser,
    logger: logging.Logger,
    *,
    timeout: float = 15.0,
    poll_interval: float = 0.3,
    idle_after_ready: float = 1.2,
) -> None:
    """
    Wait until the current page is reasonably "ready" for recording.

    Conditions:
    - document.readyState === 'complete'
    - document.body exists
    - body has measurable size and at least one child element

    After readiness, pause for idle_after_ready seconds to show a neutral pause.
    Never raises; logs a warning on timeout.
    """
    try:
        cdp_session = await browser.get_or_create_cdp_session()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("[replay/page] skipping ready wait (cdp session error): %s", exc)
        return

    start = time.monotonic()
    deadline = start + max(0.0, timeout)
    expr = """
    (function() {
        try {
            var ready = document.readyState === 'complete';
            var body = document.body;
            if (!body) return { ok: false, reason: 'no body yet' };
            var rect = body.getBoundingClientRect();
            var hasSize = rect.width > 0 && rect.height > 0;
            var hasChildren = !!body.querySelector('*');
            return { ok: ready && hasSize && hasChildren };
        } catch (e) {
            return { ok: false, reason: String(e) };
        }
    })();
    """

    while True:
        try:
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True},
                session_id=cdp_session.session_id,
            )
            val = (res.get("result") or {}).get("value") or {}
            if bool(val.get("ok")):
                elapsed = time.monotonic() - start
                logger.info("[replay/page] page ready after %.1fs", elapsed)
                await asyncio.sleep(max(0.0, idle_after_ready))
                return
        except Exception:
            pass

        now = time.monotonic()
        if now >= deadline:
            logger.warning("[replay/page] page not ready after %.1fs (timeout %.1fs)", now - start, timeout)
            return
        await asyncio.sleep(poll_interval)


async def _wait_for_navigation_or_idle(
    browser: Browser,
    logger: logging.Logger,
    *,
    previous_url: Optional[str],
    timeout: float = 10.0,
    idle_if_same_url: float = 1.5,
    poll_interval: float = 0.3,
) -> tuple[bool, Optional[str]]:
    """
    After an interaction, wait to see if navigation occurs.

    Returns (navigated, new_url). If no navigation occurs within timeout, returns (False, previous_url).
    """
    if previous_url is None:
        await asyncio.sleep(max(0.0, idle_if_same_url))
        return False, None

    start = time.monotonic()
    deadline = start + max(0.0, timeout)
    current_url = previous_url

    while time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            current_url = await browser.get_current_page_url()
        if current_url and current_url != previous_url:
            logger.info("[replay/nav] url changed after click: %s -> %s", previous_url, current_url)
            return True, current_url
        await asyncio.sleep(poll_interval)

    logger.info("[replay/nav] url unchanged after click (%.1fs), idling %.1fs", timeout, idle_if_same_url)
    await asyncio.sleep(max(0.0, idle_if_same_url))
    return False, current_url


async def _simulate_mouse_move_and_click(browser: Browser, x: float, y: float, logger: logging.Logger) -> None:
    """
    Move a visible cursor to (x, y) and perform a left click so it appears in the screencast.
    All visuals (cursor + red highlight) are handled here before dispatching real CDP click events.
    """
    # Ensure the guide cursor + overlay are present and animated from the heartbeat helper.
    with contextlib.suppress(Exception):
        await _install_screencast_heartbeat(browser)

    cdp_session = await browser.get_or_create_cdp_session()
    client = cdp_session.cdp_client
    session_id = cdp_session.session_id

    cursor_js = f"""
    (function() {{
      try {{
        if (window.__pathixEnsureCursor) window.__pathixEnsureCursor();
        if (window.__pathixAnimateCursorTo) {{
          window.__pathixAnimateCursorTo({x}, {y}, 500);
        }} else if (window.__pathixMoveCursor) {{
          window.__pathixMoveCursor({x}, {y});
        }}
        if (window.__pathixHighlightAt) window.__pathixHighlightAt({x}, {y});
      }} catch (e) {{}}
    }})();
    """

    with contextlib.suppress(Exception):
        await client.send.Runtime.evaluate(
            params={"expression": cursor_js, "returnByValue": False},
            session_id=session_id,
        )

    # Let the animation play before sending the click
    await asyncio.sleep(0.6)

    logger.info("[replay/cursor] moving + clicking at x=%s y=%s", x, y)

    for ev_type, button, click_count, delay in [
        ("mouseMoved", "none", 0, 0.05),
        ("mousePressed", "left", 1, 0.15),
        ("mouseReleased", "left", 1, 0.0),
    ]:
        with contextlib.suppress(Exception):
            await client.send.Input.dispatchMouseEvent(
                params={"type": ev_type, "x": x, "y": y, "button": button, "clickCount": click_count},
                session_id=session_id,
            )
        if delay:
            await asyncio.sleep(delay)

    # Keep cursor + overlay visible a bit after click
    await asyncio.sleep(0.4)

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
                with contextlib.suppress(Exception):
                    await _install_screencast_heartbeat(browser)
                await _wait_for_page_ready(browser, logger, timeout=15.0, idle_after_ready=1.2)

    async def _click_via_xpath(xpath: str) -> tuple[bool, Optional[str], Optional[tuple[float, float]]]:
        """Locate an element via XPath and return its center coordinates."""
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
                    return {{ok:true, x:x, y:y}};
                }} catch (e) {{
                    return {{ok:false, reason:String(e)}};
                }}
            }})();"""
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True}, session_id=cdp_session.session_id
            )
            val = (res.get("result") or {}).get("value") or {}
            coords = None
            if "x" in val and "y" in val:
                coords = (float(val["x"]), float(val["y"]))
            return bool(val.get("ok")), val.get("reason"), coords
        except Exception as exc:  # pragma: no cover - best effort
            return False, str(exc), None

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

    async def _click_via_metadata() -> tuple[bool, Optional[str], Optional[tuple[float, float]]]:
        """Locate an element via recorded tag/text/attributes and return its center coordinates."""
        if not (metadata_tag or metadata_text or metadata_attrs):
            return False, "no metadata available", None
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
                    return {{ok:true, x:x, y:y}};
                }} catch (e) {{
                    return {{ok:false, reason:String(e)}};
                }}
            }})();
            """
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True}, session_id=cdp_session.session_id
            )
            val = (res.get("result") or {}).get("value") or {}
            coords = None
            if "x" in val and "y" in val:
                coords = (float(val["x"]), float(val["y"]))
            return bool(val.get("ok")), val.get("reason"), coords
        except Exception as exc:  # pragma: no cover - best effort
            return False, str(exc), None

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
            logger.info("[replay/nav] performing navigate to %s", url)
            event = browser.event_bus.dispatch(NavigateToUrlEvent(url=str(url), new_tab=bool(params.get("new_tab"))))
            await event
            with contextlib.suppress(Exception):
                await event.event_result(raise_if_any=False, raise_if_none=False)

            # Ensure heartbeat exists on the navigated page
            with contextlib.suppress(Exception):
                await _install_screencast_heartbeat(browser)

            # Wait for readiness and then a brief idle to ensure the page is fully rendered
            await _wait_for_page_ready(browser, logger, timeout=15.0, idle_after_ready=1.2)
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
            previous_url = None
            with contextlib.suppress(Exception):
                previous_url = await browser.get_current_page_url()
            with contextlib.suppress(Exception):
                current_url = await browser.get_current_page_url()
                logger.info(
                    "[replay/click] step=%s order=%s ensure_on_page done, current_url=%s (target_page_url=%s)",
                    entry.step,
                    entry.order,
                    current_url,
                    entry.page_url,
                )
            logger.info(
                "[replay/click] step=%s order=%s trying index=%s",
                entry.step,
                entry.order,
                idx,
            )
            if idx is not None:
                node = await browser.get_element_by_index(int(idx))
            logger.info(
                "[replay/click] step=%s order=%s get_element_by_index -> %s",
                entry.step,
                entry.order,
                "FOUND" if node else "NONE",
            )

            async def _post_click_handling(prev_url: Optional[str]) -> None:
                # Keep recording active during navigation so the destination page shows up in the video.
                navigated, new_url = await _wait_for_navigation_or_idle(
                    browser,
                    logger,
                    previous_url=prev_url,
                    timeout=10.0,
                    idle_if_same_url=1.5,
                )
                if navigated:
                    with contextlib.suppress(Exception):
                        await _wait_for_page_ready(browser, logger, timeout=15.0, idle_after_ready=1.3)
                # Leave a short stable pause so the screencast captures the final state.
                await asyncio.sleep(0.7)

            if node:
                # Prefer a visible CDP click when we have XPath metadata for coordinates
                if entry.xpath:
                    ok, reason, coords = await _click_via_xpath(entry.xpath)
                    logger.info(
                        "[replay/click] step=%s order=%s node+xpath click prep ok=%s reason=%s coords=%s",
                        entry.step,
                        entry.order,
                        ok,
                        reason,
                        coords,
                    )
                    if ok and coords:
                        with contextlib.suppress(Exception):
                            await _simulate_mouse_move_and_click(browser, coords[0], coords[1], logger)
                        await _post_click_handling(previous_url)
                        return True, None
                event = browser.event_bus.dispatch(ClickElementEvent(node=node))
                await event
                with contextlib.suppress(Exception):
                    await event.event_result(raise_if_any=False, raise_if_none=False)
                await _post_click_handling(previous_url)
                return True, None
            xpath_reason = None
            if entry.xpath:
                logger.info(
                    "[replay/click] step=%s order=%s waiting for xpath=%s",
                    entry.step,
                    entry.order,
                    entry.xpath,
                )
                found = await _wait_for_xpath(entry.xpath, timeout=10.0, interval=0.5)
                logger.info(
                    "[replay/click] step=%s order=%s wait_for_xpath result=%s",
                    entry.step,
                    entry.order,
                    found,
                )
                if not found:
                    xpath_reason = f"xpath {entry.xpath}: not found after wait"
                else:
                    ok, reason, coords = await _click_via_xpath(entry.xpath)
                    logger.info(
                        "[replay/click] step=%s order=%s _click_via_xpath ok=%s reason=%s coords=%s",
                        entry.step,
                        entry.order,
                        ok,
                        reason,
                        coords,
                    )
                    if ok:
                        if coords:
                            with contextlib.suppress(Exception):
                                await _simulate_mouse_move_and_click(browser, coords[0], coords[1], logger)
                        await _post_click_handling(previous_url)
                        return True, None
                    xpath_reason = f"xpath {entry.xpath}: {reason or 'failed'}"
            if await _wait_for_metadata(timeout=10.0, interval=0.5):
                logger.info(
                    "[replay/click] step=%s order=%s using metadata fallback",
                    entry.step,
                    entry.order,
                )
                ok, reason, coords = await _click_via_metadata()
                logger.info(
                    "[replay/click] step=%s order=%s metadata click ok=%s reason=%s coords=%s",
                    entry.step,
                    entry.order,
                    ok,
                    reason,
                    coords,
                )
                if ok:
                    if coords:
                        with contextlib.suppress(Exception):
                            await _simulate_mouse_move_and_click(browser, coords[0], coords[1], logger)
                    await _post_click_handling(previous_url)
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

    wall_start = time.monotonic()
    wall_logged = False

    try:
        entries = sorted(entries, key=lambda e: (e.order if e.order is not None else 0))
    except Exception:
        pass

    logger.info(
        "[replay] Entries sequence: %s",
        [(e.action, e.step, e.order, e.page_url) for e in entries],
    )
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
            ok = False
            try:
                logger.info("[replay/nav] performing initial navigate to %s", initial_url)
                nav = browser.event_bus.dispatch(NavigateToUrlEvent(url=initial_url, new_tab=False))
                await nav
                with contextlib.suppress(Exception):
                    await nav.event_result(raise_if_any=False, raise_if_none=False)
                ok = True
            except Exception as exc:
                logger.warning("Initial navigate(%s) failed: %s", initial_url, exc)

            if ok:
                logger.info(
                    "[service] Initial navigate(%s) completed, installing heartbeat + waiting for page readiness",
                    initial_url,
                )
                with contextlib.suppress(Exception):
                    await _install_screencast_heartbeat(browser)
                # Wait for DOM readiness and give the page a short moment to stabilize
                await _wait_for_page_ready(browser, logger, timeout=15.0, idle_after_ready=1.2)
                # Minimal pause so first frames show a fully loaded page
                await asyncio.sleep(0.5)
                initialized_url = True

        for entry in entries:
            action_lower = (entry.action or "").lower()

            # Only honor the first explicit navigate/go-to-url action. Subsequent navigations
            # should happen naturally via the recorded clicks to keep the replay faithful.
            if initialized_url and action_lower == "navigate":
                logger.info(
                    "[replay/nav] skipping explicit navigate(action=%s, step=%s) because initial_url already loaded",
                    entry.action,
                    entry.step,
                )
                skipped.append(f"{entry.action} (step {entry.step}): skipped extra navigate after initial load")
                continue

            if not initialized_url and entry.page_url:
                try:
                    nav = browser.event_bus.dispatch(NavigateToUrlEvent(url=entry.page_url, new_tab=False))
                    await nav
                    with contextlib.suppress(Exception):
                        await nav.event_result(raise_if_any=False, raise_if_none=False)
                    with contextlib.suppress(Exception):
                        await _install_screencast_heartbeat(browser)
                    await _wait_for_page_ready(browser, logger, timeout=15.0, idle_after_ready=1.2)
                    initialized_url = True
                except Exception:
                    pass
            logger.info(
                "[replay] Applying action=%s step=%s order=%s page_url=%s",
                entry.action,
                entry.step,
                entry.order,
                entry.page_url,
            )
            ok, reason = await _apply_replay_action(browser, entry, logger)
            logger.info(
                "[replay] Result for action=%s step=%s order=%s: ok=%s reason=%s",
                entry.action,
                entry.step,
                entry.order,
                ok,
                reason,
            )
            if ok:
                applied += 1
                if action_lower == "navigate":
                    initialized_url = True
            else:
                skipped.append(f"{entry.action} (step {entry.step}): {reason or 'failed'}")

        await asyncio.sleep(0.5)
        await browser.stop()
        browser = None
        wall_elapsed = time.monotonic() - wall_start
        logger.info("[service] Replay wall-clock duration: %.3fs", wall_elapsed)
        wall_logged = True

        video_path = _detect_new_video_file(video_dir, before_files) if video_dir else None
        logger.info("[replay] Applied actions count: %s", applied)
        logger.info("[replay] Skipped actions: %s", skipped)
        return video_path, applied, skipped

    except Exception as exc:  # pragma: no cover - best effort
        logger.error("replay video failed: %s", exc)
        return None, applied, skipped
    finally:
        if not wall_logged:
            with contextlib.suppress(Exception):
                logger.info("[service] Replay wall-clock duration: %.3fs", time.monotonic() - wall_start)
        with contextlib.suppress(Exception):
            if browser:
                await browser.stop()
        with contextlib.suppress(Exception):
            if replay_session and replay_session.chrome_proc and replay_session.chrome_proc.returncode is None:
                replay_session.chrome_proc.terminate()
        if video_dir and video_dir.exists():
            pass
