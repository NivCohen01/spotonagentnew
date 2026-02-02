"""Pydantic models for captcha solving."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CaptchaType(str, Enum):
	"""Supported captcha types."""

	RECAPTCHA_V2 = 'recaptcha_v2'
	RECAPTCHA_V3 = 'recaptcha_v3'
	HCAPTCHA = 'hcaptcha'
	TURNSTILE = 'turnstile'
	IMAGE = 'image'
	FUNCAPTCHA = 'funcaptcha'


class SolveCaptchaParams(BaseModel):
	"""Parameters for solving a captcha."""

	model_config = ConfigDict(extra='forbid')

	captcha_type: CaptchaType = Field(
		description='Type of captcha: recaptcha_v2, recaptcha_v3, hcaptcha, turnstile, image, funcaptcha'
	)
	sitekey: str | None = Field(
		default=None,
		description='Site key for reCAPTCHA/hCaptcha/Turnstile/FunCaptcha. Found in page HTML data-sitekey attribute.',
	)
	page_url: str | None = Field(
		default=None,
		description='URL of page containing captcha. Auto-detected from browser if not provided.',
	)
	action: str | None = Field(
		default=None,
		description='Action parameter for reCAPTCHA v3 (check site JS for the action string)',
	)
	min_score: float | None = Field(
		default=None,
		ge=0.1,
		le=0.9,
		description='Minimum score for reCAPTCHA v3 (0.1-0.9, higher = more human-like)',
	)
	image_base64: str | None = Field(
		default=None,
		description='Base64 encoded image for image-based captchas',
	)


class DetectCaptchaParams(BaseModel):
	"""Parameters for detecting captcha type and sitekey on current page."""

	model_config = ConfigDict(extra='forbid')
