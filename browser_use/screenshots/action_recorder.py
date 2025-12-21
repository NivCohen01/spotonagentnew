from __future__ import annotations

import base64
import io
import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from PIL import Image, ImageDraw

from browser_use.browser.python_highlights import get_viewport_info_from_cdp
from browser_use.observability import observe_debug
from browser_use.screenshots.models import ActionScreenshotSettings

if TYPE_CHECKING:
    from browser_use.browser import BrowserProfile, BrowserSession
    from browser_use.tools.registry.views import ActionModel


def _draw_solid_rect(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int], color: str = '#FF3B30', width: int = 3) -> None:
    x1, y1, x2, y2 = bbox
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color, width=1)


def _draw_short_arrow(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    color: str = '#FF3B30',
    shaft: int = 28,
    head: int = 10,
    offset: int = 12,
) -> None:
    """Draw an unobtrusive arrow pointing toward the interacted element."""

    x1, y1, x2, y2 = bbox
    target_x = x1
    target_y = (y1 + y2) // 2
    start_x = target_x - offset - shaft
    start_y = target_y + offset + shaft
    end_x = target_x - offset
    end_y = target_y + offset

    draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=3)
    angle = math.atan2(end_y - start_y, end_x - start_x)
    left = (end_x - head * math.cos(angle - math.pi / 6), end_y - head * math.sin(angle - math.pi / 6))
    right = (end_x - head * math.cos(angle + math.pi / 6), end_y - head * math.sin(angle + math.pi / 6))
    draw.line([left, (end_x, end_y), right], fill=color, width=3)


def _apply_spotlight(img: Image.Image, bbox: tuple[int, int, int, int], pad: int = 12, opacity: int = 230) -> None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.width, x2 + pad)
    y2 = min(img.height, y2 + pad)

    overlay = Image.new('RGBA', img.size, (0, 0, 0, opacity))
    window = Image.new('RGBA', (x2 - x1, y2 - y1), (0, 0, 0, 0))
    overlay.paste(window, (x1, y1))
    img.alpha_composite(overlay)


def _bbox_from_coordinates(
    img: Image.Image, dpr: float, x: float, y: float, size: int = 80
) -> tuple[int, int, int, int]:
    """Return a square bounding box centered around viewport coordinates."""

    center_x = int(max(0, min(img.width, x * dpr)))
    center_y = int(max(0, min(img.height, y * dpr)))
    half = size // 2

    return (
        max(0, center_x - half),
        max(0, center_y - half),
        min(img.width, center_x + half),
        min(img.height, center_y + half),
    )


def _shrink_bbox(bbox: tuple[int, int, int, int], max_inset: int = 4) -> tuple[int, int, int, int]:
    """Slightly shrink a bbox to avoid over-drawing element edges."""
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width <= 2 or height <= 2:
        return bbox

    inset = min(max_inset, max(1, int(min(width, height) * 0.02)))
    return (x1 + inset, y1 + inset, x2 - inset, y2 - inset)


def _clamp_bbox_to_image(bbox: tuple[int, int, int, int], img: Image.Image) -> tuple[int, int, int, int]:
    """Ensure bbox stays within image bounds."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img.width))
    y1 = max(0, min(y1, img.height))
    x2 = max(0, min(x2, img.width))
    y2 = max(0, min(y2, img.height))
    return (x1, y1, x2, y2)


def _normalize_coordinates_to_viewport(
    coord_x: float | None,
    coord_y: float | None,
    *,
    browser_session: 'BrowserSession',
    viewport_width: int | None,
    viewport_height: int | None,
    screenshot_width: int | None,
    screenshot_height: int | None,
) -> tuple[float | None, float | None]:
    """Scale coordinates from screenshot/device space back to CSS viewport space."""
    if coord_x is None or coord_y is None:
        return coord_x, coord_y

    x, y = coord_x, coord_y

    # If screenshots were resized for the LLM, map back using the recorded viewport size
    if browser_session.llm_screenshot_size and browser_session._original_viewport_size:
        llm_w, llm_h = browser_session.llm_screenshot_size
        orig_w, orig_h = browser_session._original_viewport_size
        if llm_w and llm_h and orig_w and orig_h:
            x = (x / llm_w) * orig_w
            y = (y / llm_h) * orig_h
    # Otherwise, if the model likely returned device-pixel coordinates (mobile/high-DPI), scale down
    elif (
        viewport_width
        and viewport_height
        and screenshot_width
        and screenshot_height
        and (x > viewport_width or y > viewport_height)
    ):
        x = (x / screenshot_width) * viewport_width
        y = (y / screenshot_height) * viewport_height

    return x, y


class ActionScreenshotRecorder:
    """Captures annotated screenshots for each interactive action."""

    _CLICK_ACTION_NAMES = {'click'}

    def __init__(self, settings: ActionScreenshotSettings, agent_directory: Path, session_id: str):
        self.settings = settings
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        self.enabled = bool(settings.enabled)

        self.base_dir = self._resolve_base_dir(agent_directory)
        self.manifest_path = self.base_dir / 'manifest.jsonl'
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_base_dir(self, agent_directory: Path) -> Path:
        if self.settings.output_dir:
            base = Path(self.settings.output_dir)
        else:
            base = Path(agent_directory) / 'action_screenshots'

        if self.settings.session_subdirectories:
            base = base / self.session_id
        return base

    @observe_debug(ignore_output=True, name='action_screenshot_capture')
    async def capture(
        self,
        *,
        action: 'ActionModel',
        step_number: int,
        run_step_id: int | None = None,
        action_index: int,
        phase: str = 'before',
        browser_session: 'BrowserSession',
        browser_profile: 'BrowserProfile',
    ) -> str | None:
        if not self.enabled or browser_session is None:
            return None

        action_data = action.model_dump(exclude_unset=True)
        action_name, action_params = next(iter(action_data.items()), (None, {}))
        action_name = action_name or 'action'
        action_params = action_params or {}

        if action_name not in self._CLICK_ACTION_NAMES:
            return None

        target_index = action.get_index()
        if target_index is not None:
            try:
                target_index = int(target_index)
            except (TypeError, ValueError):
                self.logger.debug(f'Skipping action screenshot: invalid index {target_index!r}')
                target_index = None

        coordinate_x = action_params.get('coordinate_x')
        coordinate_y = action_params.get('coordinate_y')
        try:
            coordinate_x = float(coordinate_x) if coordinate_x is not None else None
            coordinate_y = float(coordinate_y) if coordinate_y is not None else None
        except (TypeError, ValueError):
            coordinate_x = coordinate_y = None

        old_highlight = browser_profile.highlight_elements
        browser_profile.highlight_elements = False
        try:
            state = await browser_session.get_browser_state_summary(include_screenshot=True)
        finally:
            browser_profile.highlight_elements = old_highlight

        dom_state = getattr(state, 'dom_state', None)
        selector_map = getattr(dom_state, 'selector_map', None) if dom_state else None
        screenshot_b64 = getattr(state, 'screenshot', None)
        if not screenshot_b64:
            self.logger.debug('Skipping action screenshot: screenshot data missing.')
            return None

        image_bytes = base64.b64decode(screenshot_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGBA')

        page_info = getattr(state, 'page_info', None)
        page_url = getattr(state, 'url', None)
        viewport_width = getattr(page_info, 'viewport_width', None) if page_info else None
        viewport_height = getattr(page_info, 'viewport_height', None) if page_info else None
        if viewport_width is None and browser_profile.viewport:
            vp = browser_profile.viewport
            viewport_width = vp.get('width') if isinstance(vp, dict) else getattr(vp, 'width', None)
            viewport_height = vp.get('height') if isinstance(vp, dict) else getattr(vp, 'height', None)

        # Normalize coordinates back to viewport space so annotations line up on mobile/high-DPI captures
        coordinate_x, coordinate_y = _normalize_coordinates_to_viewport(
            coordinate_x,
            coordinate_y,
            browser_session=browser_session,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            screenshot_width=image.width,
            screenshot_height=image.height,
        )

        cdp_session = await browser_session.get_or_create_cdp_session()
        device_pixel_ratio, _, _ = await get_viewport_info_from_cdp(cdp_session)
        dpr = device_pixel_ratio or 1.0

        # Prefer per-axis scale from screenshot vs viewport; fall back to CDP DPR.
        scale_x = scale_y = dpr
        if viewport_width and viewport_height:
            scale_x = image.width / float(viewport_width)
            scale_y = image.height / float(viewport_height)
            self.logger.debug(
                f'Using screenshot-derived scale for annotations: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}, cdp_dpr={dpr:.3f}'
            )
        elif viewport_width:
            scale_x = scale_y = image.width / float(viewport_width)
            self.logger.debug(f'Using screenshot-derived scale (width only) for annotations: {scale_x:.3f}')

        bbox: tuple[int, int, int, int] | None = None
        if selector_map and target_index is not None and target_index in selector_map:
            element = selector_map[target_index]
            if element and element.absolute_position:
                rect = element.absolute_position
                bbox = (
                    round(rect.x * scale_x),
                    round(rect.y * scale_y),
                    round((rect.x + rect.width) * scale_x),
                    round((rect.y + rect.height) * scale_y),
                )
            else:
                self.logger.debug(f'Skipping annotation: element missing position for index {target_index}.')
        elif target_index is not None:
            self.logger.debug(f'Selector map missing index {target_index}; falling back to coordinates.')

        if bbox is None and coordinate_x is not None and coordinate_y is not None:
            bbox = _bbox_from_coordinates(image, scale_x, coordinate_x, coordinate_y)
            self.logger.debug(
                f'Annotating screenshot for action "{action_name}" using coordinates ({coordinate_x}, {coordinate_y}).'
            )
        elif bbox is None:
            self.logger.debug(
                f'Skipping action screenshot for "{action_name}": no selector index or coordinates available.'
            )
            return None

        if bbox and self.settings.annotate:
            # Clamp to image bounds to avoid spillover, then only shrink when bbox is coordinate-derived
            bbox = _clamp_bbox_to_image(bbox, image)
            if target_index is None:
                bbox = _shrink_bbox(bbox)
            if self.settings.spotlight:
                _apply_spotlight(image, bbox)
            draw = ImageDraw.Draw(image)
            _draw_solid_rect(draw, bbox)
            _draw_short_arrow(draw, bbox)

        filename = self._build_filename(action, step_number, action_index, target_index)
        output_path = self.base_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await anyio.to_thread.run_sync(image.save, output_path, 'PNG')
        self.logger.info(f'Saved action screenshot -> {output_path}')
        self._write_manifest_entry(
            run_step_id=run_step_id if run_step_id is not None else step_number,
            action_type=action_name,
            phase=phase,
            action_index=action_index,
            click_index=target_index,
            page_url=page_url,
            file_path=output_path,
        )
        return str(output_path)

    def _build_filename(
        self, action: 'ActionModel', step_number: int, action_index: int, target_index: int | None
    ) -> str:
        action_data = action.model_dump(exclude_unset=True)
        action_name = next(iter(action_data.keys()), 'action')
        timestamp_ms = int(time.time() * 1000)
        index_fragment = str(target_index) if target_index is not None else 'na'
        return f'step_{step_number:03d}_{action_name}_{action_index}_{index_fragment}_{timestamp_ms}.png'

    def _write_manifest_entry(
        self,
        *,
        run_step_id: int,
        action_type: str,
        phase: str,
        action_index: int,
        click_index: int | None,
        page_url: str | None,
        file_path: Path,
    ) -> None:
        """Persist lightweight metadata for deterministic screenshot mapping."""
        if not self.enabled:
            return
        entry = {
            'run_step_id': run_step_id,
            'action_type': action_type,
            'phase': phase or 'before',
            'action_index': action_index,
            'click_index': click_index,
            'page_url': page_url,
            'ts': int(time.time() * 1000),
            'file': str(file_path),
        }
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.manifest_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self.logger.info(
                f'Manifest entry | run_step_id={run_step_id} action={action_type} click_index={click_index} file={file_path}'
            )
        except Exception as exc:  # pragma: no cover - best-effort logging only
            self.logger.debug(f'Failed to write screenshot manifest entry: {type(exc).__name__}: {exc}')
