"""Video Recording Service for Browser Use Sessions."""

import base64
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Optional

from browser_use.browser.profile import ViewportSize

try:
	import imageio.v2 as iio  # type: ignore[import-not-found]
	import imageio_ffmpeg  # type: ignore[import-not-found]
	import numpy as np  # type: ignore[import-not-found]
	from imageio.core.format import Format  # type: ignore[import-not-found]

	IMAGEIO_AVAILABLE = True
except ImportError:
	IMAGEIO_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_padded_size(size: ViewportSize, macro_block_size: int = 16) -> ViewportSize:
	"""Calculates the dimensions padded to the nearest multiple of macro_block_size."""
	width = int(math.ceil(size['width'] / macro_block_size)) * macro_block_size
	height = int(math.ceil(size['height'] / macro_block_size)) * macro_block_size
	return ViewportSize(width=width, height=height)


class VideoRecorderService:
	"""
	Handles the video encoding process for a browser session using imageio.

	This service captures individual frames from the CDP screencast, decodes them,
	and appends them to a video file using a pip-installable ffmpeg backend.
	It automatically resizes frames to match the target video dimensions.
	"""

	def __init__(self, output_path: Path, size: ViewportSize, framerate: int, warmup_seconds: float = 0.0):
		"""
		Initializes the video recorder.

		Args:
		    output_path: The full path where the video will be saved.
		    size: A ViewportSize object specifying the width and height of the video.
		    framerate: The desired framerate for the output video.
		    warmup_seconds: Optional initial window to skip frames (avoids blank/half-rendered frames).
		"""
		self.output_path = output_path
		self.size = size
		self.framerate = framerate
		self._writer: Optional['Format.Writer'] = None
		self._is_active = False
		self._is_paused = False
		self.padded_size = _get_padded_size(self.size)
		self._frame_count = 0
		self._first_frame_ts: float | None = None
		self._last_frame_ts: float | None = None
		self._start_wall_time: float | None = None
		self._target_fps = max(1, int(self.framerate or 5))
		self._warmup_seconds = max(0.0, float(warmup_seconds))
		# Keep replication modest so gaps don't over-inflate the output duration.
		self._max_replicate_per_frame = 4

	def pause(self) -> None:
		"""Temporarily pause adding frames to the recording."""
		self._is_paused = True

	def resume(self) -> None:
		"""Resume adding frames to the recording."""
		self._is_paused = False

	def start(self) -> None:
		"""
		Prepares and starts the video writer.

		If the required optional dependencies are not installed, this method will
		log an error and do nothing.
		"""
		if not IMAGEIO_AVAILABLE:
			logger.error(
				'MP4 recording requires optional dependencies. Please install them with: pip install "browser-use[video]"'
			)
			return

		try:
			self.output_path.parent.mkdir(parents=True, exist_ok=True)
			self._start_wall_time = time.monotonic()
			self._frame_count = 0
			self._first_frame_ts = None
			self._last_frame_ts = None
			# The macro_block_size is set to None because we handle padding ourselves
			self._writer = iio.get_writer(
				str(self.output_path),
				fps=self._target_fps,
				codec='libx264',
				quality=8,  # A good balance of quality and file size (1-10 scale)
				pixelformat='yuv420p',  # Ensures compatibility with most players
				macro_block_size=None,
			)
			self._is_paused = False
			self._is_active = True
			logger.debug(f'Video recorder started. Output will be saved to {self.output_path}')
		except Exception as e:
			logger.error(f'Failed to initialize video writer: {e}')
			self._is_active = False

	def add_frame(self, frame_data_b64: str) -> None:
		"""
		Decodes a base64-encoded PNG frame, resizes it, pads it to be codec-compatible,
		and appends it to the video.

		Args:
		    frame_data_b64: A base64-encoded string of the PNG frame data.
		"""
		if not self._is_active or not self._writer or self._is_paused:
			return

		try:
			if self._start_wall_time is not None and self._warmup_seconds > 0:
				if (time.monotonic() - self._start_wall_time) < self._warmup_seconds:
					return

			now = time.monotonic()
			if self._first_frame_ts is None:
				self._first_frame_ts = now
			dt = 0.0
			if self._last_frame_ts is not None:
				dt = max(0.0, min(now - self._last_frame_ts, 1.0))
			self._last_frame_ts = now

			# Replicate frames so video duration tracks the real elapsed time even if CDP frames are sparse.
			replicate = max(1, int(round(dt * self._target_fps)))
			replicate = min(self._max_replicate_per_frame, replicate)

			frame_bytes = base64.b64decode(frame_data_b64)

			# Build a filter chain for ffmpeg:
			# 1. scale: Resizes the frame to the user-specified dimensions.
			# 2. pad: Adds black bars to meet codec's macro-block requirements,
			#    centering the original content.
			vf_chain = (
				f'scale={self.size["width"]}:{self.size["height"]},'
				f'pad={self.padded_size["width"]}:{self.padded_size["height"]}:(ow-iw)/2:(oh-ih)/2:color=black'
			)

			output_pix_fmt = 'rgb24'
			command = [
				imageio_ffmpeg.get_ffmpeg_exe(),
				'-f',
				'image2pipe',  # Input format from a pipe
				'-c:v',
				'png',  # Specify input codec is PNG
				'-i',
				'-',  # Input from stdin
				'-vf',
				vf_chain,  # Video filter for resizing and padding
				'-f',
				'rawvideo',  # Output format is raw video
				'-pix_fmt',
				output_pix_fmt,  # Output pixel format
				'-',  # Output to stdout
			]

			# Execute ffmpeg as a subprocess
			proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			out, err = proc.communicate(input=frame_bytes)

			if proc.returncode != 0:
				err_msg = err.decode(errors='ignore').strip()
				if 'deprecated pixel format used' not in err_msg.lower():
					raise OSError(f'ffmpeg error during resizing/padding: {err_msg}')
				else:
					logger.debug(f'ffmpeg warning during resizing/padding: {err_msg}')

			# Convert the raw output bytes to a numpy array with the padded dimensions
			img_array = np.frombuffer(out, dtype=np.uint8).reshape((self.padded_size['height'], self.padded_size['width'], 3))

			for _ in range(replicate):
				self._writer.append_data(img_array)
			self._frame_count += replicate
		except Exception as e:
			logger.warning(f'Could not process and add video frame: {e}')

	def stop_and_save(self) -> None:
		"""
		Finalizes the video file by closing the writer.

		This method should be called when the recording session is complete.
		"""
		if not self._is_active or not self._writer:
			return

		try:
			self._writer.close()
			wall = (time.monotonic() - self._start_wall_time) if self._start_wall_time else None
			video_duration = (self._frame_count / float(self._target_fps)) if self._target_fps else None
			logger.info(
				'[video_recorder] frames=%d, first_ts=%s, last_ts=%s, wall_clock_duration=%s, fps=%s, video_duration_estimate=%s, output=%s',
				self._frame_count,
				self._first_frame_ts,
				self._last_frame_ts,
				f'{wall:.3f}s' if wall is not None else 'unknown',
				self._target_fps,
				f'{video_duration:.3f}s' if video_duration is not None else 'unknown',
				self.output_path,
			)
			logger.info(f'📹 Video recording saved successfully to: {self.output_path}')
		except Exception as e:
			logger.error(f'Failed to finalize and save video: {e}')
		finally:
			self._is_active = False
			self._writer = None
