"""Lightweight video helpers for Playwright recordings."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_ffmpeg_available(explicit_path: Optional[str] = None) -> str:
    """Return a usable ffmpeg binary path or raise if not found."""
    candidates = [explicit_path, os.getenv("FFMPEG_BIN"), "ffmpeg"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) if not Path(candidate).exists() else candidate
        if resolved:
            return str(resolved)
    raise RuntimeError("ffmpeg not found. Install ffmpeg or set FFMPEG_BIN to its path.")


def convert_webm_to_mp4(
    input_webm: Path | str,
    output_mp4: Path | str,
    *,
    ffmpeg_bin: Optional[str] = None,
    fps: int = 60,
    trim_start_seconds: Optional[float] = None,
) -> Path:
    """Convert a WebM video to MP4 (libx264 + yuv420p) with faststart flags."""
    input_path = Path(input_webm)
    output_path = Path(output_mp4)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ensure_ffmpeg_available(ffmpeg_bin)

    cmd = [ffmpeg, "-y"]
    if trim_start_seconds is not None and trim_start_seconds > 0:
        cmd.extend(["-ss", f"{trim_start_seconds:.3f}"])
    cmd.extend(
        [
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
    )

    logger.info("[video] ffmpeg command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        logger.error("[video] ffmpeg failed: %s", stderr)
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}")

    logger.info("[video] converted %s -> %s", input_path, output_path)
    return output_path
