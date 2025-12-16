from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

try:
    from playwright.async_api import (  # type: ignore[import-not-found]
        Locator,
        Page,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Playwright is required for video replay. Install with `uv add playwright` and run `playwright install chromium`."
    ) from exc

from browser_use.browser.video_recorder import convert_webm_to_mp4, ensure_ffmpeg_available

from .service_config import CHROME_BIN, MAX_CHROME_CONCURRENCY
from .service_models import ActionTraceEntry, DeviceType

logger = logging.getLogger("service")

_CHROME_SEMAPHORE = asyncio.Semaphore(MAX_CHROME_CONCURRENCY)


# -------------------------------------------------------------------
# Humanization profiles
# -------------------------------------------------------------------
class HumanizationProfile(BaseModel):
    """Timing profile for human-like replay."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_delay: float = Field(default=0.35)
    action_jitter: float = Field(default=0.25)

    cursor_min_ms: int = Field(default=450)
    cursor_max_ms: int = Field(default=650)
    cursor_steps_min: int = Field(default=18)
    cursor_steps_max: int = Field(default=34)
    highlight_ms: int = Field(default=240)

    type_delay_ms: int = Field(default=70)
    type_jitter_ms: int = Field(default=35)
    pre_type_pause_range: Tuple[float, float] = Field(default=(0.18, 0.42))
    post_type_pause_range: Tuple[float, float] = Field(default=(0.22, 0.55))
    clear_pause_range: Tuple[float, float] = Field(default=(0.10, 0.22))

    post_nav_idle_range: Tuple[float, float] = Field(default=(0.6, 1.25))
    post_idle_range: Tuple[float, float] = Field(default=(0.45, 0.9))
    network_idle_timeout_ms: int = Field(default=3500)
    navigation_timeout_ms: int = Field(default=12000)

    scroll_amount: int = Field(default=800)


class ReplayProfile(BaseModel):
    """Viewport and video sizing profile."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    device_type: DeviceType = Field(default="desktop")
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None


@dataclass
class ReplayState:
    mouse_x: float = 0.0
    mouse_y: float = 0.0


# -------------------------------------------------------------------
# Cursor + highlight JS (IMPORTANT: no transform animation!)
# -------------------------------------------------------------------
_CURSOR_HELPER_SCRIPT = r"""
(() => {
  if (window.__pathixCursorHelpersInstalled) return;
  window.__pathixCursorHelpersInstalled = true;

  const STYLE_ID = '__pathix_cursor_styles';
  const CURSOR_ID = '__pathix_cursor';
  const HIGHLIGHT_ID = '__pathix_click_overlay';

  function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      @keyframes pathixCursorPulse {
        0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.45); }
        70%  { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
        100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
      }
      #${CURSOR_ID} {
        position: fixed;
        left: 0;
        top: 0;
        width: 24px;
        height: 24px;
        border-radius: 9999px;
        border: 2px solid #ef4444;
        background: rgba(239,68,68,0.12);
        z-index: 2147483647;
        pointer-events: none;
        opacity: 1 !important;
        display: block !important;
        will-change: transform;
        animation: pathixCursorPulse 1.2s infinite ease-out;
      }
      #${HIGHLIGHT_ID} {
        position: fixed;
        border: 2px solid #ef4444;
        border-radius: 8px;
        background: rgba(239,68,68,0.12);
        pointer-events: none;
        z-index: 2147483646;
        opacity: 0;
        transition: opacity 0.18s ease-out;
      }
    `;
    document.head.appendChild(style);
  }

  function ensureCursor() {
    ensureStyles();
    let cur = document.getElementById(CURSOR_ID);
    if (!cur) {
      cur = document.createElement('div');
      cur.id = CURSOR_ID;
      document.documentElement.appendChild(cur);

      // init at center ONCE per document
      const startX = window.innerWidth / 2;
      const startY = window.innerHeight / 2;
      cur.dataset.x = String(startX);
      cur.dataset.y = String(startY);
      cur.style.transform = `translate3d(${startX - 12}px, ${startY - 12}px, 0)`;
    } else {
      cur.style.display = 'block';
      cur.style.opacity = '1';
    }
    return cur;
  }

  function setCursorPos(x, y) {
    const cur = ensureCursor();
    cur.dataset.x = String(x);
    cur.dataset.y = String(y);
    cur.style.transform = `translate3d(${x - 12}px, ${y - 12}px, 0)`;
  }

  // --- NEW: Keep the visual cursor synced with real mouse moves (Playwright emits mousemove) ---
  let __pathixCursorRAF = 0;

  function __pathixSyncCursorFromEvent(e) {
    // prefer pointer coordinates if available
    const x = (typeof e.clientX === 'number') ? e.clientX : null;
    const y = (typeof e.clientY === 'number') ? e.clientY : null;
    if (x === null || y === null) return;

    if (__pathixCursorRAF) cancelAnimationFrame(__pathixCursorRAF);
    __pathixCursorRAF = requestAnimationFrame(() => {
      setCursorPos(x, y);
    });
  }

  document.addEventListener('mousemove', __pathixSyncCursorFromEvent, { capture: true, passive: true });
  document.addEventListener('pointermove', __pathixSyncCursorFromEvent, { capture: true, passive: true });
  // ------------------------------------------------------------------------------------------

  function animateCursorTo(x, y, durationMs) {
    const cur = ensureCursor();
    const startX = parseFloat(cur.dataset.x || String(window.innerWidth / 2));
    const startY = parseFloat(cur.dataset.y || String(window.innerHeight / 2));
    const targetX = typeof x === 'number' ? x : startX;
    const targetY = typeof y === 'number' ? y : startY;
    const d = Math.max(0, durationMs || 450);
    const start = performance.now();
    const ease = (t) => (1 - Math.cos(Math.PI * Math.min(1, t))) / 2;

    function step(now) {
      const t = Math.min(1, (now - start) / d);
      const e = ease(t);
      const cx = startX + (targetX - startX) * e;
      const cy = startY + (targetY - startY) * e;
      setCursorPos(cx, cy);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function ensureOverlay() {
    ensureStyles();
    let overlay = document.getElementById(HIGHLIGHT_ID);
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = HIGHLIGHT_ID;
      document.documentElement.appendChild(overlay);
    }
    return overlay;
  }

  function highlightRect(x, y, w, h, durationMs) {
    const overlay = ensureOverlay();
    if (![x,y,w,h].every(v => typeof v === 'number')) return;
    overlay.style.left = `${x - 4}px`;
    overlay.style.top = `${y - 4}px`;
    overlay.style.width = `${w + 8}px`;
    overlay.style.height = `${h + 8}px`;
    overlay.style.opacity = '1';
    const dur = Math.max(60, durationMs || 220);
    setTimeout(() => { overlay.style.opacity = '0'; }, dur);
  }

  window.__pathixEnsureCursor = ensureCursor;
  window.__pathixSetCursorPos = setCursorPos;
  window.__pathixAnimateCursorTo = animateCursorTo;
  window.__pathixHighlightRect = highlightRect;

  ensureCursor();
})();
"""

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def _resolve_viewport(profile: ReplayProfile) -> dict:
    if profile.viewport_width and profile.viewport_height:
        return {"width": int(profile.viewport_width), "height": int(profile.viewport_height)}
    if profile.device_type == "mobile":
        return {"width": 390, "height": 844}
    if profile.device_type == "custom":
        return {"width": 1280, "height": 720}
    return {"width": 1920, "height": 1080}


def _resolve_video_size(profile: ReplayProfile, viewport: dict) -> dict:
    width = int(profile.video_width or viewport["width"])
    height = int(profile.video_height or viewport["height"])
    return {"width": width, "height": height}


def _human_sleep(profile: HumanizationProfile) -> float:
    return max(0.05, profile.action_delay + random.uniform(-profile.action_jitter, profile.action_jitter))


def _norm_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    try:
        p = urlsplit(str(u))
        scheme = (p.scheme or "https").lower()
        netloc = p.netloc.lower()
        path = (p.path or "/").rstrip("/") or "/"
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return str(u).rstrip("/")


def _extract_coordinates(entry: ActionTraceEntry) -> Optional[Tuple[float, float]]:
    params = entry.params or {}
    for candidate in ("coordinates", "point", "position"):
        value = params.get(candidate)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return float(value[0]), float(value[1])
            except Exception:
                continue
        if isinstance(value, dict) and "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except Exception:
                continue
    if "x" in params and "y" in params:
        try:
            return float(params["x"]), float(params["y"])
        except Exception:
            return None
    return None


def _candidate_text(entry: ActionTraceEntry) -> Optional[str]:
    for cand in (entry.element_text, entry.value):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    text = entry.params.get("text") if isinstance(entry.params, dict) else None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


async def _ensure_cursor_helpers(page: Page) -> None:
    with contextlib.suppress(Exception):
        await page.evaluate(_CURSOR_HELPER_SCRIPT)
    with contextlib.suppress(Exception):
        await page.evaluate("() => window.__pathixEnsureCursor && window.__pathixEnsureCursor()")


async def _set_cursor(page: Page, x: float, y: float) -> None:
    await _ensure_cursor_helpers(page)
    with contextlib.suppress(Exception):
        await page.evaluate(
            "(x,y)=>window.__pathixSetCursorPos && window.__pathixSetCursorPos(x,y)",
            x,
            y,
        )


async def _animate_cursor(page: Page, x: float, y: float, duration_ms: int) -> None:
    await _ensure_cursor_helpers(page)
    with contextlib.suppress(Exception):
        await page.evaluate(
            "(x,y,ms)=>window.__pathixAnimateCursorTo && window.__pathixAnimateCursorTo(x,y,ms)",
            x,
            y,
            duration_ms,
        )
    await asyncio.sleep(max(0.0, duration_ms / 1000.0))


async def _highlight_rect(page: Page, rect: dict, duration_ms: int) -> None:
    await _ensure_cursor_helpers(page)
    with contextlib.suppress(Exception):
        await page.evaluate(
            "(r,ms)=>window.__pathixHighlightRect && window.__pathixHighlightRect(r.x,r.y,r.w,r.h,ms)",
            {"x": rect["x"], "y": rect["y"], "w": rect["width"], "h": rect["height"]},
            duration_ms,
        )


async def _human_mouse_move(
    page: Page,
    state: ReplayState,
    target_x: float,
    target_y: float,
    duration_ms: int,
    human: HumanizationProfile,
) -> None:
    """Move the *real* Playwright mouse in small eased steps."""
    if state.mouse_x == 0 and state.mouse_y == 0:
        with contextlib.suppress(Exception):
            vp = page.viewport_size
            if vp:
                state.mouse_x = vp["width"] / 2
                state.mouse_y = vp["height"] / 2

    steps = max(6, random.randint(human.cursor_steps_min, human.cursor_steps_max))
    total_s = max(0.08, duration_ms / 1000.0)
    step_s = total_s / steps

    sx, sy = state.mouse_x, state.mouse_y

    def ease(t: float) -> float:
        import math

        t = max(0.0, min(1.0, t))
        return 0.5 - 0.5 * math.cos(math.pi * t)

    for i in range(1, steps + 1):
        t = i / steps
        e = ease(t)
        cx = sx + (target_x - sx) * e
        cy = sy + (target_y - sy) * e
        with contextlib.suppress(Exception):
            await page.mouse.move(cx, cy)
        await asyncio.sleep(step_s)

    state.mouse_x = target_x
    state.mouse_y = target_y


async def _get_absolute_rect(locator: Locator) -> Optional[dict]:
    """
    Try to compute a reliable absolute rect even when element is inside SAME-ORIGIN iframes.
    Falls back to None if not possible.
    """
    js = r"""
(el) => {
  if (!el) return null;

  const rect = el.getBoundingClientRect();
  let x = rect.left;
  let y = rect.top;

  // Walk up iframe chain (same-origin only). If cross-origin, stop.
  try {
    let win = el.ownerDocument.defaultView;
    while (win && win.frameElement) {
      const fr = win.frameElement.getBoundingClientRect();
      x += fr.left;
      y += fr.top;
      // accessing parent may be restricted in cross-origin; this will throw and break.
      win = win.parent;
    }
  } catch (e) {}

  return {
    x,
    y,
    width: rect.width,
    height: rect.height
  };
}
"""
    with contextlib.suppress(Exception):
        rect = await locator.evaluate(js)
        if rect and all(k in rect for k in ("x", "y", "width", "height")):
            # Ensure numbers
            return {
                "x": float(rect["x"]),
                "y": float(rect["y"]),
                "width": float(rect["width"]),
                "height": float(rect["height"]),
            }
    return None


async def _resolve_locator(page: Page, entry: ActionTraceEntry, log: logging.Logger) -> Optional[Locator]:
    params = entry.params or {}
    xpaths = [entry.xpath, params.get("xpath")]
    selectors = [
        params.get("selector"),
        params.get("css"),
        params.get("css_selector"),
        params.get("aria_label"),
        params.get("data_testid"),
    ]

    for xp in [x for x in xpaths if x]:
        try:
            loc = page.locator(f"xpath={xp}").first
            await loc.wait_for(state="visible", timeout=5000)
            return loc
        except PlaywrightTimeoutError:
            log.debug("[replay/locate] xpath timed out: %s", xp)
        except Exception as exc:
            log.debug("[replay/locate] xpath failed: %s", exc)

    for sel in [s for s in selectors if s]:
        try:
            loc = page.locator(str(sel)).first
            await loc.wait_for(state="visible", timeout=5000)
            return loc
        except PlaywrightTimeoutError:
            log.debug("[replay/locate] selector timed out: %s", sel)
        except Exception as exc:
            log.debug("[replay/locate] selector failed: %s", exc)

    txt = _candidate_text(entry)
    if txt:
        try:
            loc = page.get_by_text(txt, exact=False).first
            await loc.wait_for(state="visible", timeout=5000)
            return loc
        except PlaywrightTimeoutError:
            log.debug("[replay/locate] text timed out: %s", txt)
        except Exception as exc:
            log.debug("[replay/locate] text failed: %s", exc)

    return None


async def _wait_after_interaction(
    page: Page,
    *,
    previous_url: Optional[str],
    human: HumanizationProfile,
    log: logging.Logger,
) -> Optional[str]:
    current_url: Optional[str] = None
    with contextlib.suppress(Exception):
        current_url = page.url

    if previous_url and current_url and previous_url != current_url:
        log.info("[replay/nav] url changed: %s -> %s", previous_url, current_url)
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("domcontentloaded", timeout=human.navigation_timeout_ms)
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("load", timeout=human.navigation_timeout_ms)
        await asyncio.sleep(random.uniform(*human.post_nav_idle_range))
        await _ensure_cursor_helpers(page)
        return current_url

    with contextlib.suppress(Exception):
        await page.wait_for_load_state("networkidle", timeout=human.network_idle_timeout_ms)
    await asyncio.sleep(random.uniform(*human.post_idle_range))
    await _ensure_cursor_helpers(page)

    with contextlib.suppress(Exception):
        return page.url
    return current_url


async def _perform_navigation(page: Page, url: str, human: HumanizationProfile, log: logging.Logger) -> Optional[str]:
    await _ensure_cursor_helpers(page)
    log.info("[replay/nav] navigating to %s", url)
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=human.navigation_timeout_ms)
    except PlaywrightTimeoutError:
        log.warning("[replay/nav] domcontentloaded timeout for %s", url)
    except Exception as exc:
        log.warning("[replay/nav] navigate failed for %s: %s", url, exc)
        return None

    with contextlib.suppress(Exception):
        await page.wait_for_load_state("load", timeout=human.navigation_timeout_ms)

    await asyncio.sleep(random.uniform(*human.post_nav_idle_range))
    await _ensure_cursor_helpers(page)

    with contextlib.suppress(Exception):
        return page.url
    return url


async def _human_type_text(page: Page, text: str, human: HumanizationProfile) -> None:
    for ch in str(text):
        if ch == "\n":
            with contextlib.suppress(Exception):
                await page.keyboard.press("Enter")
        else:
            with contextlib.suppress(Exception):
                await page.keyboard.insert_text(ch)

        delay_ms = max(12, human.type_delay_ms + random.randint(-human.type_jitter_ms, human.type_jitter_ms))
        await asyncio.sleep(delay_ms / 1000.0)


async def _move_cursor_to_rect_center(
    page: Page,
    state: ReplayState,
    rect: dict,
    human: HumanizationProfile,
) -> None:
    cx = float(rect["x"] + rect["width"] / 2)
    cy = float(rect["y"] + rect["height"] / 2)
    duration = random.randint(human.cursor_min_ms, human.cursor_max_ms)

    # Visual cursor + real mouse move together
    await asyncio.gather(
        _animate_cursor(page, cx, cy, duration),
        _human_mouse_move(page, state, cx, cy, duration, human),
    )


async def _perform_click(
    page: Page,
    state: ReplayState,
    entry: ActionTraceEntry,
    human: HumanizationProfile,
    previous_url: Optional[str],
    log: logging.Logger,
) -> tuple[bool, Optional[str]]:
    await _ensure_cursor_helpers(page)
    locator = await _resolve_locator(page, entry, log)
    coords = _extract_coordinates(entry)

    # Prefer locator rect (more reliable than trace coords)
    rect = None
    if locator:
        with contextlib.suppress(Exception):
            await locator.scroll_into_view_if_needed(timeout=5000)
        rect = await _get_absolute_rect(locator)

    if rect:
        await _move_cursor_to_rect_center(page, state, rect, human)
        await _highlight_rect(page, rect, human.highlight_ms)

        try:
            with contextlib.suppress(Exception):
                await locator.hover(timeout=2000)  # type: ignore[union-attr]
            await locator.click(timeout=human.navigation_timeout_ms, delay=40)  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("[replay/click] click failed: %s", exc)
            return False, previous_url

    elif coords:
        cx, cy = coords
        duration = random.randint(human.cursor_min_ms, human.cursor_max_ms)
        await asyncio.gather(
            _animate_cursor(page, cx, cy, duration),
            _human_mouse_move(page, state, cx, cy, duration, human),
        )
        with contextlib.suppress(Exception):
            await page.mouse.click(cx, cy, delay=35)
    else:
        return False, previous_url

    new_url = await _wait_after_interaction(page, previous_url=previous_url, human=human, log=log)
    await asyncio.sleep(_human_sleep(human))
    return True, new_url


async def _perform_type_like(
    page: Page,
    state: ReplayState,
    entry: ActionTraceEntry,
    human: HumanizationProfile,
    previous_url: Optional[str],
    log: logging.Logger,
) -> tuple[bool, Optional[str]]:
    await _ensure_cursor_helpers(page)
    locator = await _resolve_locator(page, entry, log)
    if not locator:
        return False, previous_url

    with contextlib.suppress(Exception):
        await locator.scroll_into_view_if_needed(timeout=5000)

    rect = await _get_absolute_rect(locator)
    if rect:
        await _move_cursor_to_rect_center(page, state, rect, human)
        await _highlight_rect(page, rect, human.highlight_ms)

    try:
        # focus
        await locator.click(timeout=human.navigation_timeout_ms, delay=35)

        await asyncio.sleep(random.uniform(*human.pre_type_pause_range))

        # clear if requested (default: clear)
        clear_flag = True
        if isinstance(entry.params, dict) and entry.params.get("clear") is False:
            clear_flag = False

        if clear_flag:
            with contextlib.suppress(Exception):
                await page.keyboard.press("Control+A")
            await asyncio.sleep(random.uniform(*human.clear_pause_range))
            with contextlib.suppress(Exception):
                await page.keyboard.press("Delete")
            await asyncio.sleep(random.uniform(*human.clear_pause_range))

        # text source: params.text > value
        text = ""
        if isinstance(entry.params, dict):
            text = str(entry.params.get("text") or "")
        if not text:
            text = str(entry.value or "")

        await _human_type_text(page, text, human)
        await asyncio.sleep(random.uniform(*human.post_type_pause_range))

    except Exception as exc:
        log.warning("[replay/type] typing failed: %s", exc)
        return False, previous_url

    new_url = await _wait_after_interaction(page, previous_url=previous_url, human=human, log=log)
    await asyncio.sleep(_human_sleep(human))
    return True, new_url


async def _perform_scroll(page: Page, entry: ActionTraceEntry, human: HumanizationProfile, log: logging.Logger) -> None:
    params = entry.params or {}
    direction = (params.get("direction") or params.get("dir") or "down").lower()
    magnitude = params.get("pixels") or params.get("amount") or human.scroll_amount
    try:
        delta = int(magnitude)
    except Exception:
        delta = human.scroll_amount
    dy = abs(delta) if direction in ("down", "next") else -abs(delta)

    with contextlib.suppress(Exception):
        await page.mouse.wheel(0, dy)
    log.info("[replay/scroll] direction=%s amount=%s", direction, dy)

    await asyncio.sleep(random.uniform(*human.post_idle_range))
    await _ensure_cursor_helpers(page)


# -------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------
async def replay_trace_to_video(
    trace: list[ActionTraceEntry],
    initial_url: Optional[str],
    output_dir: Path,
    profile: ReplayProfile,
    *,
    humanization: Optional[HumanizationProfile] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> tuple[Optional[Path], int, list[str]]:
    """
    Replay an action trace in Playwright Chromium and return (mp4_path, applied_count, skipped_msgs).
    Rules:
      - Perform ONLY the first navigation:
          * If initial_url is provided => navigate to it once.
          * Else => use the first 'navigate' action from the trace.
      - Ignore all subsequent navigate actions.
      - Treat action 'input' as 'type'.
    """
    if not trace:
        return None, 0, []

    log = logger_instance or logger
    human = humanization or HumanizationProfile()

    entries = sorted(trace, key=lambda e: (e.order if e.order is not None else 0))

    viewport = _resolve_viewport(profile)
    video_size = _resolve_video_size(profile, viewport)

    run_id = uuid.uuid4().hex[:10]
    temp_dir = Path(output_dir) / f"{run_id}_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = Path(output_dir) / f"{run_id}.mp4"

    applied = 0
    skipped: list[str] = []
    mp4_path: Optional[Path] = None
    page_video = None

    ensure_ffmpeg_available()

    async with _CHROME_SEMAPHORE:
        log.info("[replay] acquiring chromium slot (max=%s)", MAX_CHROME_CONCURRENCY)

        playwright = None
        browser = None
        context = None
        page = None

        state = ReplayState()

        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=CHROME_BIN if CHROME_BIN else None,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )

            context = await browser.new_context(
                record_video_dir=str(temp_dir),
                record_video_size=video_size,
                viewport=viewport,
            )

            # Ensure helpers exist in every future document:
            await context.add_init_script(_CURSOR_HELPER_SCRIPT)

            page = await context.new_page()
            await _ensure_cursor_helpers(page)

            # Initialize state/mouse/cursor to center once
            vp = page.viewport_size
            if vp:
                state.mouse_x = vp["width"] / 2
                state.mouse_y = vp["height"] / 2
            else:
                state.mouse_x, state.mouse_y = 640, 360

            with contextlib.suppress(Exception):
                await page.mouse.move(state.mouse_x, state.mouse_y)
            await _set_cursor(page, state.mouse_x, state.mouse_y)
            await asyncio.sleep(0.25)

            last_url: Optional[str] = None
            did_first_nav = False

            # 1) If initial_url is given, that's the ONLY navigation we do.
            if initial_url:
                last_url = await _perform_navigation(page, initial_url, human, log)
                did_first_nav = True
                await _ensure_cursor_helpers(page)

            i = 0
            while i < len(entries):
                entry = entries[i]
                action = (entry.action or "").lower()

                log.info("[replay] action=%s step=%s order=%s page=%s", entry.action, entry.step, entry.order, entry.page_url)
                await _ensure_cursor_helpers(page)

                # 2) If no initial_url, allow exactly ONE 'navigate' from trace (the first one), skip the rest.
                if action == "navigate":
                    if did_first_nav:
                        skipped.append(f"navigate (step {entry.step}): skipped (only first navigation is executed)")
                        i += 1
                        continue

                    # Use trace URL for the *single* navigation
                    url = None
                    if isinstance(entry.params, dict):
                        url = entry.params.get("url")
                    url = url or entry.value or entry.page_url
                    if not url:
                        skipped.append(f"navigate (step {entry.step}): missing url")
                        i += 1
                        continue

                    last_url = await _perform_navigation(page, str(url), human, log)
                    did_first_nav = True
                    applied += 1
                    await asyncio.sleep(_human_sleep(human))
                    i += 1
                    continue

                # 3) Click
                if action == "click":
                    ok, last_url = await _perform_click(page, state, entry, human, last_url, log)
                    if ok:
                        applied += 1
                    else:
                        skipped.append(f"click (step {entry.step}): locator/coords not found or click failed")
                    i += 1
                    continue

                # 4) Type OR Input (same mechanics)
                if action in ("type", "input"):
                    ok, last_url = await _perform_type_like(page, state, entry, human, last_url, log)
                    if ok:
                        applied += 1
                    else:
                        skipped.append(f"{action} (step {entry.step}): locator not found or typing failed")
                    i += 1
                    continue

                # 5) Scroll
                if action == "scroll":
                    await _perform_scroll(page, entry, human, log)
                    applied += 1
                    i += 1
                    continue

                # 6) Wait / Sleep
                if action in ("wait", "sleep"):
                    seconds_raw = entry.params.get("seconds") if isinstance(entry.params, dict) else entry.value
                    try:
                        seconds = float(seconds_raw)
                    except Exception:
                        seconds = 0.5
                    await asyncio.sleep(max(0.0, seconds))
                    applied += 1
                    i += 1
                    continue

                # 7) Extract / Done (no-op but keep timing natural)
                if action in ("extract", "done"):
                    await asyncio.sleep(random.uniform(*human.post_idle_range))
                    applied += 1
                    i += 1
                    continue

                skipped.append(f"{entry.action} (step {entry.step}): unsupported action")
                i += 1

            # Small tail so final state is visible in video
            await page.wait_for_timeout(600)
            page_video = page.video if page else None

        finally:
            with contextlib.suppress(Exception):
                if context:
                    await context.close()
            with contextlib.suppress(Exception):
                if browser:
                    await browser.close()
            with contextlib.suppress(Exception):
                if playwright:
                    await playwright.stop()

    # Convert and cleanup
    if page_video:
        webm_path_str = await page_video.path()
        webm_path = Path(webm_path_str)
        mp4_path = final_mp4

        try:
            convert_webm_to_mp4(webm_path, mp4_path)
            log.info("[replay] converted %s -> %s", webm_path, mp4_path)
        except Exception as exc:
            log.error("[replay] ffmpeg conversion failed: %s", exc)
            skipped.append(f"conversion failed: {exc}")
            mp4_path = None

        with contextlib.suppress(Exception):
            webm_path.unlink()
        with contextlib.suppress(Exception):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return mp4_path, applied, skipped
