from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ActionScreenshotSettings(BaseModel):
	"""Configuration for capturing annotated screenshots before browser actions."""

	enabled: bool = Field(
		default=False,
		description='Enable capturing screenshots before executing click-like actions.',
	)
	output_dir: str | None = Field(
		default=None,
		description='Optional base directory for storing action screenshots. Defaults to the agent directory.',
	)
	annotate: bool = Field(
		default=True,
		description='Draw borders/arrows around the interacted element in the stored screenshot.',
	)
	spotlight: bool = Field(
		default=False,
		description='Dim everything except the interacted element in the annotated screenshot.',
	)
	session_subdirectories: bool = Field(
		default=True,
		description='Store screenshots under a session-specific folder even when output_dir is shared.',
	)
	include_in_available_files: bool = Field(
		default=False,
		description='Append generated screenshot paths to Agent.available_file_paths for downstream usage.',
	)

	@field_validator('output_dir')
	@classmethod
	def _expand_output_dir(cls, value: str | None) -> str | None:
		if value is None:
			return None
		return str(Path(value).expanduser())
