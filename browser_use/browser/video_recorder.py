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


def _compute_video_segments(
    action_timestamps: list[tuple[float, float]],
    trim_start: float,
    buffer_before_action: float,
    min_idle_duration: float,
) -> list[tuple[str, float, float]]:
    """Classify video time ranges as 'normal' or 'fast' for idle speedup.

    Returns a list of (type, start_sec, end_sec) tuples in the ORIGINAL
    video's time domain (not trimmed).  The trim_start is used to skip
    the pre-page-load content.  The final segment uses a sentinel end
    value (99999) meaning "until EOF".
    """
    # Keep timestamps in original video domain; discard anything before trim
    adjusted: list[tuple[float, float]] = []
    for start, end in sorted(action_timestamps):
        if end <= trim_start:
            continue
        adjusted.append((max(trim_start, start), end))

    if not adjusted:
        return []

    # Merge overlapping / nearly-adjacent action windows (within 0.3s)
    merged: list[list[float]] = [[adjusted[0][0], adjusted[0][1]]]
    for s, e in adjusted[1:]:
        if s <= merged[-1][1] + 0.3:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    segments: list[tuple[str, float, float]] = []
    # Start from trim_start — everything before this is discarded
    pos = trim_start

    for action_start, action_end in merged:
        buffered_start = max(pos, action_start - buffer_before_action)
        idle_gap = buffered_start - pos

        if idle_gap >= min_idle_duration:
            segments.append(("fast", pos, buffered_start))
        elif idle_gap > 0.05:
            segments.append(("normal", pos, buffered_start))

        segments.append(("normal", buffered_start, action_end))
        pos = action_end

    # Trailing content after the last action (until EOF)
    segments.append(("normal", pos, 99999.0))

    # Merge adjacent segments of the same type
    compacted: list[tuple[str, float, float]] = [segments[0]]
    for seg in segments[1:]:
        if seg[0] == compacted[-1][0]:
            compacted[-1] = (seg[0], compacted[-1][1], seg[2])
        else:
            compacted.append(seg)

    # Drop zero-duration segments (except the trailing sentinel)
    return [
        s for s in compacted
        if s[2] > s[1] + 0.01 or s[2] >= 99999.0
    ]


def convert_with_idle_speedup(
    input_webm: Path | str,
    output_mp4: Path | str,
    action_timestamps: list[tuple[float, float]],
    *,
    ffmpeg_bin: str | None = None,
    fps: int = 60,
    trim_start_seconds: float | None = None,
    idle_speed: float = 2.0,
    buffer_before_action: float = 0.75,
    min_idle_duration: float = 1.5,
) -> Path:
    """Convert WebM to MP4, speeding up idle segments between actions.

    Uses ffmpeg's trim+setpts+concat filter to compress gaps where nothing
    visible is happening (no cursor movement, typing, or scrolling).
    Falls back to simple conversion if no idle segments qualify.
    """
    input_path = Path(input_webm)
    output_path = Path(output_mp4)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trim_start = trim_start_seconds or 0.0

    segments = _compute_video_segments(
        action_timestamps, trim_start, buffer_before_action, min_idle_duration,
    )

    has_fast = any(t == "fast" for t, _, _ in segments)
    if not segments or not has_fast:
        logger.info("[video] no idle segments to speed up (%d segments, has_fast=%s), using simple conversion", len(segments), has_fast)
        return convert_webm_to_mp4(
            input_path, output_path, ffmpeg_bin=ffmpeg_bin,
            fps=fps, trim_start_seconds=trim_start_seconds,
        )

    for seg_type, seg_start, seg_end in segments:
        dur = (seg_end - seg_start) if seg_end < 99999 else -1
        logger.info("[video] segment: %s %.3f-%.3f (%.1fs)", seg_type, seg_start, seg_end, dur)
    logger.info("[video] idle speedup: %d segments (%d fast), trim_start=%.3f", len(segments), sum(1 for t, _, _ in segments if t == "fast"), trim_start)

    # Build ffmpeg filter_complex:
    #   [0:v]trim=A:B,setpts=PTS-STARTPTS[s0]; ...
    #   [s0][s1]...concat=n=N:v=1:a=0[out]
    ffmpeg = ensure_ffmpeg_available(ffmpeg_bin)
    speed_factor = 1.0 / idle_speed
    parts: list[str] = []
    labels: list[str] = []

    for idx, (seg_type, start, end) in enumerate(segments):
        label = f"s{idx}"
        pts = f"{speed_factor}*(PTS-STARTPTS)" if seg_type == "fast" else "PTS-STARTPTS"

        if end >= 99999.0:
            trim_expr = f"trim=start={start:.3f}"
        else:
            trim_expr = f"trim={start:.3f}:{end:.3f}"

        parts.append(f"[0:v]{trim_expr},setpts={pts}[{label}]")
        labels.append(f"[{label}]")

    concat_str = "".join(labels)
    parts.append(f"{concat_str}concat=n={len(segments)}:v=1:a=0[out]")
    filter_complex = ";".join(parts)

    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-r", str(fps),
        str(output_path),
    ]

    logger.info("[video] ffmpeg idle-speedup command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        logger.error("[video] ffmpeg idle-speedup failed: %s", stderr)
        # Fall back to simple conversion on failure
        logger.info("[video] falling back to simple conversion")
        return convert_webm_to_mp4(
            input_path, output_path, ffmpeg_bin=ffmpeg_bin,
            fps=fps, trim_start_seconds=trim_start_seconds,
        )

    logger.info("[video] converted with idle speedup %s -> %s", input_path, output_path)
    return output_path
