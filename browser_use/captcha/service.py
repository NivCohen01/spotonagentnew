"""Captcha solving service using 2Captcha API."""

import logging
import os
from typing import Any

from browser_use.captcha.views import CaptchaType, SolveCaptchaParams

logger = logging.getLogger(__name__)


class CaptchaSolver:
	"""Service for solving captchas using 2Captcha API."""

	def __init__(self, api_key: str | None = None):
		self.api_key = api_key or os.getenv('TWOCAPTCHA_API_KEY') or os.getenv('CAPTCHA2_API_KEY')
		if not self.api_key:
			raise ValueError(
				'2Captcha API key not configured. Set TWOCAPTCHA_API_KEY environment variable or pass api_key parameter.'
			)

		# Import here to avoid import errors if twocaptcha is not installed
		try:
			from twocaptcha import AsyncTwoCaptcha
		except ImportError:
			raise ImportError('twocaptcha package not installed. Run: pip install 2captcha-python')

		self.solver = AsyncTwoCaptcha(self.api_key)

	async def solve(self, params: SolveCaptchaParams, page_url: str | None = None) -> dict[str, Any]:
		"""Solve captcha based on type and return solution.

		Args:
			params: Captcha parameters including type and sitekey
			page_url: URL of page with captcha (used if params.page_url is None)

		Returns:
			Dict with 'success' bool and either 'code' (token) or 'error' message
		"""
		url = params.page_url or page_url
		if not url and params.captcha_type != CaptchaType.IMAGE:
			return {'success': False, 'error': 'page_url is required for non-image captchas'}

		try:
			if params.captcha_type == CaptchaType.RECAPTCHA_V2:
				if not params.sitekey:
					return {'success': False, 'error': 'sitekey is required for reCAPTCHA v2'}
				result = await self.solver.recaptcha(sitekey=params.sitekey, url=url)

			elif params.captcha_type == CaptchaType.RECAPTCHA_V3:
				if not params.sitekey:
					return {'success': False, 'error': 'sitekey is required for reCAPTCHA v3'}
				kwargs: dict[str, Any] = {'sitekey': params.sitekey, 'url': url, 'version': 'v3'}
				if params.action:
					kwargs['action'] = params.action
				if params.min_score:
					kwargs['min_score'] = params.min_score
				result = await self.solver.recaptcha(**kwargs)

			elif params.captcha_type == CaptchaType.HCAPTCHA:
				if not params.sitekey:
					return {'success': False, 'error': 'sitekey is required for hCaptcha'}
				result = await self.solver.hcaptcha(sitekey=params.sitekey, url=url)

			elif params.captcha_type == CaptchaType.TURNSTILE:
				if not params.sitekey:
					return {'success': False, 'error': 'sitekey is required for Turnstile'}
				result = await self.solver.turnstile(sitekey=params.sitekey, url=url)

			elif params.captcha_type == CaptchaType.FUNCAPTCHA:
				if not params.sitekey:
					return {'success': False, 'error': 'sitekey (publickey) is required for FunCaptcha'}
				result = await self.solver.funcaptcha(sitekey=params.sitekey, url=url)

			elif params.captcha_type == CaptchaType.IMAGE:
				if not params.image_base64:
					return {'success': False, 'error': 'image_base64 is required for image captchas'}
				result = await self.solver.normal(params.image_base64)

			else:
				return {'success': False, 'error': f'Unsupported captcha type: {params.captcha_type}'}

			# 2captcha returns dict with 'code' key containing the solution
			code = result.get('code') if isinstance(result, dict) else result
			logger.info(f'Captcha solved successfully: {params.captcha_type.value}')
			return {'success': True, 'code': code}

		except Exception as e:
			error_msg = str(e)
			logger.error(f'Failed to solve captcha ({params.captcha_type.value}): {error_msg}')
			return {'success': False, 'error': error_msg}


def get_injection_script(captcha_type: CaptchaType, token: str) -> str:
	"""Generate JavaScript to inject captcha token into page.

	Args:
		captcha_type: Type of captcha that was solved
		token: Solution token from 2Captcha

	Returns:
		JavaScript code to inject the token (arrow function format for CDP)
	"""
	# Escape token for safe inclusion in JS string
	safe_token = token.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')

	if captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3):
		return f'''() => {{
			// Set token in textarea(s)
			const textareas = document.querySelectorAll('[name="g-recaptcha-response"], #g-recaptcha-response, textarea.g-recaptcha-response');
			textareas.forEach(textarea => {{
				textarea.value = "{safe_token}";
				textarea.innerHTML = "{safe_token}";
				// Make visible if hidden (some sites hide it)
				if (textarea.style.display === 'none') {{
					textarea.style.display = 'block';
				}}
			}});

			// Try to trigger callback via grecaptcha config
			try {{
				if (typeof window.___grecaptcha_cfg !== 'undefined') {{
					const clients = window.___grecaptcha_cfg.clients;
					for (let key in clients) {{
						const client = clients[key];
						for (let subkey in client) {{
							const item = client[subkey];
							if (item && item.callback) {{
								item.callback("{safe_token}");
								return 'Callback triggered';
							}}
						}}
					}}
				}}
			}} catch(e) {{ console.log('Callback trigger failed:', e); }}

			return 'Token injected into textarea';
		}}'''

	elif captcha_type == CaptchaType.HCAPTCHA:
		return f'''() => {{
			// Set token in textarea
			const textareas = document.querySelectorAll('[name="h-captcha-response"], [name="g-recaptcha-response"]');
			textareas.forEach(textarea => {{
				textarea.value = "{safe_token}";
			}});

			// Set in iframe response attribute
			const iframes = document.querySelectorAll('iframe[data-hcaptcha-response]');
			iframes.forEach(iframe => {{
				iframe.setAttribute('data-hcaptcha-response', "{safe_token}");
			}});

			return 'Token injected';
		}}'''

	elif captcha_type == CaptchaType.TURNSTILE:
		return f'''() => {{
			// Set token in input
			const inputs = document.querySelectorAll('[name="cf-turnstile-response"], input[name*="turnstile"]');
			inputs.forEach(input => {{
				input.value = "{safe_token}";
			}});

			// Also try hidden input inside turnstile container
			const containers = document.querySelectorAll('.cf-turnstile');
			containers.forEach(container => {{
				const hidden = container.querySelector('input[type="hidden"]');
				if (hidden) hidden.value = "{safe_token}";
			}});

			return 'Token injected';
		}}'''

	elif captcha_type == CaptchaType.FUNCAPTCHA:
		return f'''() => {{
			const inputs = document.querySelectorAll('#fc-token, [name="fc-token"], input[name*="funcaptcha"]');
			inputs.forEach(input => {{
				input.value = "{safe_token}";
			}});
			return 'Token injected';
		}}'''

	else:  # IMAGE captcha - the token is the text answer
		return f'''() => {{
			// For image captchas, find and fill the input field
			const inputs = document.querySelectorAll(
				'input[name*="captcha"], input[id*="captcha"], ' +
				'input[placeholder*="captcha" i], input[placeholder*="code" i], ' +
				'input[aria-label*="captcha" i]'
			);
			inputs.forEach(input => {{
				input.value = "{safe_token}";
				input.dispatchEvent(new Event('input', {{ bubbles: true }}));
				input.dispatchEvent(new Event('change', {{ bubbles: true }}));
			}});
			return 'Text injected into ' + inputs.length + ' input(s)';
		}}'''


def get_detection_script() -> str:
	"""Generate JavaScript to detect captcha type and extract sitekey from current page.

	Returns:
		JavaScript code that returns detection result object (arrow function format for CDP)
	"""
	return """() => {
		const result = { type: null, sitekey: null, url: window.location.href };

		// reCAPTCHA v2/v3 detection
		const recaptchaElement = document.querySelector('[data-sitekey].g-recaptcha, .g-recaptcha[data-sitekey], [data-sitekey][data-callback]');
		if (recaptchaElement) {
			result.sitekey = recaptchaElement.getAttribute('data-sitekey');
			// Check for v3 indicators
			const isV3 = recaptchaElement.getAttribute('data-size') === 'invisible' ||
						 document.querySelector('script[src*="recaptcha/api.js?render="]') !== null ||
						 recaptchaElement.classList.contains('grecaptcha-badge');
			result.type = isV3 ? 'recaptcha_v3' : 'recaptcha_v2';

			// Try to get action for v3
			if (isV3) {
				const actionAttr = recaptchaElement.getAttribute('data-action');
				if (actionAttr) result.action = actionAttr;
			}
			return result;
		}

		// reCAPTCHA Enterprise detection (invisible/behavioral - check BEFORE regular script detection)
		const recaptchaEnterpriseScript = document.querySelector('script[src*="recaptcha/enterprise"]');
		if (recaptchaEnterpriseScript) {
			const src = recaptchaEnterpriseScript.src;
			const renderMatch = src.match(/render=([^&]+)/);
			result.type = 'recaptcha_enterprise';
			result.sitekey = renderMatch ? renderMatch[1] : null;
			result.is_invisible = true;
			result.unsolvable = true;  // Enterprise invisible cannot be solved by captcha services
			return result;
		}

		// Check for grecaptcha.enterprise global object
		if (typeof grecaptcha !== 'undefined' && typeof grecaptcha.enterprise !== 'undefined') {
			result.type = 'recaptcha_enterprise';
			result.is_invisible = true;
			result.unsolvable = true;
			return result;
		}

		// Also check for reCAPTCHA loaded via script (regular v3)
		const recaptchaScript = document.querySelector('script[src*="recaptcha"]');
		if (recaptchaScript) {
			const src = recaptchaScript.src;
			const renderMatch = src.match(/render=([^&]+)/);
			if (renderMatch && renderMatch[1] !== 'explicit') {
				result.type = 'recaptcha_v3';
				result.sitekey = renderMatch[1];
				return result;
			}
		}

		// hCaptcha detection
		const hcaptchaElement = document.querySelector('[data-sitekey].h-captcha, .h-captcha[data-sitekey]');
		if (hcaptchaElement) {
			result.type = 'hcaptcha';
			result.sitekey = hcaptchaElement.getAttribute('data-sitekey');
			return result;
		}

		// Cloudflare Turnstile detection
		const turnstileElement = document.querySelector('[data-sitekey].cf-turnstile, .cf-turnstile[data-sitekey]');
		if (turnstileElement) {
			result.type = 'turnstile';
			result.sitekey = turnstileElement.getAttribute('data-sitekey');
			return result;
		}

		// FunCaptcha detection
		const funcaptchaElement = document.querySelector('[data-pkey], #FunCaptcha');
		if (funcaptchaElement) {
			result.type = 'funcaptcha';
			result.sitekey = funcaptchaElement.getAttribute('data-pkey');
			return result;
		}

		// Image captcha detection (generic)
		const imgCaptcha = document.querySelector('img[src*="captcha" i], img[alt*="captcha" i], img[id*="captcha" i]');
		if (imgCaptcha) {
			result.type = 'image';
			result.image_src = imgCaptcha.src;
			return result;
		}

		// No captcha found
		return result;
	}"""
