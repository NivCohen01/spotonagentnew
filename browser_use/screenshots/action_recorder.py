from __future__ import annotations

import base64
import io
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


class ActionScreenshotRecorder:
	"""Captures annotated screenshots for each interactive action."""

	def __init__(self, settings: ActionScreenshotSettings, agent_directory: Path, session_id: str):
		self.settings = settings
		self.session_id = session_id
		self.logger = logging.getLogger(__name__)
		self.enabled = bool(settings.enabled)

		self.base_dir = self._resolve_base_dir(agent_directory)
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
		action_index: int,
		browser_session: 'BrowserSession',
		browser_profile: 'BrowserProfile',
	) -> str | None:
		if not self.enabled or browser_session is None:
			return None

		target_index = action.get_index()
		if target_index is None:
			return None

		cached_state = getattr(browser_session, '_cached_browser_state_summary', None)
		dom_state = getattr(cached_state, 'dom_state', None)
		selector_map = getattr(dom_state, 'selector_map', None) if dom_state else None
		if not selector_map or target_index not in selector_map:
			return None

		element = selector_map[target_index]
		if element is None or element.absolute_position is None:
			return None

		rect = element.absolute_position
		old_highlight = browser_profile.highlight_elements
		browser_profile.highlight_elements = False
		try:
			state = await browser_session.get_browser_state_summary(include_screenshot=True)
		finally:
			browser_profile.highlight_elements = old_highlight

		screenshot_b64 = getattr(state, 'screenshot', None)
		if not screenshot_b64:
			return None

		image_bytes = base64.b64decode(screenshot_b64)
		image = Image.open(io.BytesIO(image_bytes)).convert('RGBA')

		cdp_session = await browser_session.get_or_create_cdp_session()
		device_pixel_ratio, _, _ = await get_viewport_info_from_cdp(cdp_session)
		dpr = device_pixel_ratio or 1.0

		bbox = (
			int(rect.x * dpr),
			int(rect.y * dpr),
			int((rect.x + rect.width) * dpr),
			int((rect.y + rect.height) * dpr),
		)

		if self.settings.annotate:
			if self.settings.spotlight:
				_apply_spotlight(image, bbox)
			draw = ImageDraw.Draw(image)
			_draw_solid_rect(draw, bbox)
			_draw_short_arrow(draw, bbox)

		filename = self._build_filename(action, step_number, action_index, target_index)
		output_path = self.base_dir / filename
		await anyio.to_thread.run_sync(image.save, output_path, 'PNG')
		return str(output_path)

	def _build_filename(self, action: 'ActionModel', step_number: int, action_index: int, target_index: int) -> str:
		action_data = action.model_dump(exclude_unset=True)
		action_name = next(iter(action_data.keys()), 'action')
		timestamp_ms = int(time.time() * 1000)
		return f'step_{step_number:03d}_{action_name}_{action_index}_{target_index}_{timestamp_ms}.png'
