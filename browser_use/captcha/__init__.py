"""Captcha solving integration for browser-use."""

from browser_use.captcha.service import CaptchaSolver, get_detection_script, get_injection_script
from browser_use.captcha.views import CaptchaType, DetectCaptchaParams, SolveCaptchaParams

__all__ = [
	'CaptchaSolver',
	'CaptchaType',
	'SolveCaptchaParams',
	'DetectCaptchaParams',
	'get_detection_script',
	'get_injection_script',
]
