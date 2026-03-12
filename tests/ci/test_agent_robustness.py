"""
tests/ci/test_agent_robustness.py

Tests for the agent hardening features:
- Page readiness detection
- Blocking overlay detection and banner
- Action grounding validation
- Conceptual loop detection
- Guardrails OTP domain generalisation

These tests use real objects throughout. Browser-requiring tests use pytest-httpserver.
LLM calls are not needed for these scenarios.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_use.agent.prompts import AgentMessagePrompt
from browser_use.browser.views import BrowserStateSummary, OverlayInfo, PageReadinessInfo, TabInfo
from browser_use.dom.views import SerializedDOMState
from browser_use.filesystem.file_system import FileSystem


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_empty_dom_state() -> SerializedDOMState:
	"""Build a minimal SerializedDOMState with no elements."""
	return SerializedDOMState(_root=None, selector_map={})


def _make_browser_state(
	url: str = 'https://example.com',
	page_readiness: PageReadinessInfo | None = None,
	blocking_overlays: list[OverlayInfo] | None = None,
) -> BrowserStateSummary:
	tab = TabInfo(url=url, title='Test', target_id='AAAA')  # type: ignore[arg-type]
	return BrowserStateSummary(
		dom_state=_make_empty_dom_state(),
		url=url,
		title='Test',
		tabs=[tab],
		page_readiness=page_readiness,
		blocking_overlays=blocking_overlays or [],
	)


def _make_prompt(browser_state: BrowserStateSummary) -> AgentMessagePrompt:
	"""Build an AgentMessagePrompt wrapping the given state, without needing a real FileSystem."""
	fs = MagicMock(spec=FileSystem)
	fs.get_state.return_value = None
	return AgentMessagePrompt(
		browser_state_summary=browser_state,
		file_system=fs,
		include_attributes=[],
	)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PageReadinessInfo data model
# ─────────────────────────────────────────────────────────────────────────────

def test_page_readiness_info_fields():
	"""PageReadinessInfo should store all fields correctly."""
	info = PageReadinessInfo(is_ready=False, ready_state='loading', stable_for_ms=0)
	assert info.is_ready is False
	assert info.ready_state == 'loading'
	assert info.stable_for_ms == 0

	info2 = PageReadinessInfo(is_ready=True, ready_state='complete', stable_for_ms=450)
	assert info2.is_ready is True
	assert info2.stable_for_ms == 450


# ─────────────────────────────────────────────────────────────────────────────
# 2. OverlayInfo data model
# ─────────────────────────────────────────────────────────────────────────────

def test_overlay_info_fields():
	"""OverlayInfo should store all fields correctly."""
	overlay = OverlayInfo(
		is_blocking=True,
		description='modal dialog',
		close_index=42,
		z_index=9999,
		coverage_pct=0.75,
	)
	assert overlay.is_blocking is True
	assert overlay.description == 'modal dialog'
	assert overlay.close_index == 42
	assert overlay.z_index == 9999
	assert overlay.coverage_pct == pytest.approx(0.75)


def test_overlay_info_no_close_button():
	"""OverlayInfo should handle missing close button (close_index=None)."""
	overlay = OverlayInfo(
		is_blocking=True,
		description='loading spinner',
		close_index=None,
		z_index=1000,
		coverage_pct=0.25,
	)
	assert overlay.close_index is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Page readiness banner in AgentMessagePrompt
# ─────────────────────────────────────────────────────────────────────────────

def test_readiness_banner_not_ready():
	"""Banner should warn when page is not ready."""
	readiness = PageReadinessInfo(is_ready=False, ready_state='loading', stable_for_ms=0)
	state = _make_browser_state(page_readiness=readiness)
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert 'PAGE NOT READY' in banner
	assert 'loading' in banner


def test_readiness_banner_ready_no_overlay():
	"""No readiness banner should appear when page is ready and no overlays exist."""
	readiness = PageReadinessInfo(is_ready=True, ready_state='complete', stable_for_ms=500)
	state = _make_browser_state(page_readiness=readiness)
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert 'PAGE NOT READY' not in banner
	assert 'OVERLAY' not in banner
	assert banner == ''


def test_readiness_banner_no_page_readiness_field():
	"""Banner should be empty when page_readiness is None (backward compat)."""
	state = _make_browser_state(page_readiness=None)
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert banner == ''


# ─────────────────────────────────────────────────────────────────────────────
# 4. Blocking overlay banner in AgentMessagePrompt
# ─────────────────────────────────────────────────────────────────────────────

def test_overlay_banner_blocking():
	"""Banner should show BLOCKING OVERLAY when an overlay covers > 20% of viewport."""
	overlay = OverlayInfo(
		is_blocking=True,
		description='modal dialog',
		close_index=35,
		z_index=9999,
		coverage_pct=0.75,
	)
	state = _make_browser_state(blocking_overlays=[overlay])
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert 'BLOCKING OVERLAY' in banner
	assert 'modal dialog' in banner
	assert '35' in banner  # close button index
	assert 'PRIORITY' in banner


def test_overlay_banner_non_blocking():
	"""Non-blocking overlays (e.g. toasts) should get a mild informational note."""
	overlay = OverlayInfo(
		is_blocking=False,
		description='toast',
		close_index=None,
		z_index=200,
		coverage_pct=0.08,
	)
	state = _make_browser_state(blocking_overlays=[overlay])
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	# Should NOT show the strong blocking warning
	assert 'BLOCKING OVERLAY' not in banner
	# Should mention it exists (informational)
	assert 'toast' in banner


def test_overlay_banner_no_overlays():
	"""Empty overlay list should produce no banner text."""
	state = _make_browser_state(blocking_overlays=[])
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert banner == ''


def test_overlay_banner_multiple_blocking():
	"""Multiple blocking overlays: mention additional ones."""
	o1 = OverlayInfo(is_blocking=True, description='modal dialog', close_index=None, z_index=9999, coverage_pct=0.8)
	o2 = OverlayInfo(is_blocking=True, description='drawer', close_index=5, z_index=500, coverage_pct=0.3)
	state = _make_browser_state(blocking_overlays=[o1, o2])
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert 'modal dialog' in banner
	assert 'drawer' in banner


# ─────────────────────────────────────────────────────────────────────────────
# 5. Loop detection — _compute_action_signature and _check_action_loop
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_action(action_type_name: str = 'ClickElementAction', index: int | None = 1) -> MagicMock:
	"""Create a mock ActionModel that behaves like a typed action."""
	action = MagicMock()
	action.__class__.__name__ = action_type_name
	payload = {'index': index} if index is not None else {}
	action.model_dump.return_value = {action_type_name: payload}
	return action


def _make_agent_with_signatures(sigs: list[str]):
	"""Create a minimal Agent-like namespace with recent_action_signatures pre-populated."""
	from browser_use.agent.views import AgentState
	state = AgentState()
	state.recent_action_signatures = list(sigs)

	# Minimal mock of Agent with just what _check_action_loop needs
	agent = MagicMock()
	agent.state = state
	agent.browser_session = MagicMock()
	agent.browser_session._cached_browser_state_summary = MagicMock()
	agent.browser_session._cached_browser_state_summary.url = 'https://example.com/page'
	agent.browser_session.get_element_by_index.return_value = None  # no label resolution

	# Bind the real methods and class-level constants to the mock agent
	from browser_use.agent.service import Agent
	agent._LOOP_EXCLUDED_ACTION_TYPES = Agent._LOOP_EXCLUDED_ACTION_TYPES
	agent._compute_action_signature = Agent._compute_action_signature.__get__(agent, type(agent))
	agent._check_action_loop = Agent._check_action_loop.__get__(agent, type(agent))
	agent._record_action_signatures = Agent._record_action_signatures.__get__(agent, type(agent))

	return agent


def test_loop_detection_fires():
	"""Loop detection should fire when same signature appears 2+ times in last 6 steps."""
	url = 'https://example.com/page'
	url_bucket = hashlib.md5(url.encode()).hexdigest()[:8]
	# Pre-fill with 2 identical signatures
	sig = f'ClickElementAction::{url_bucket}'
	agent = _make_agent_with_signatures([sig, sig])
	action = _make_mock_action('ClickElementAction', index=None)
	warning = agent._check_action_loop([action])
	assert warning is not None
	assert 'LOOP' in warning.upper() or 'loop' in warning.lower()


def test_loop_detection_no_false_positive_different_actions():
	"""Loop detection should NOT fire for 3 different action signatures."""
	url = 'https://example.com/page'
	url_bucket = hashlib.md5(url.encode()).hexdigest()[:8]
	sigs = [
		f'ClickElementAction:button-a:{url_bucket}',
		f'ClickElementAction:button-b:{url_bucket}',
		f'NavigateAction::{url_bucket}',
	]
	agent = _make_agent_with_signatures(sigs)
	action = _make_mock_action('ScrollAction', index=None)
	warning = agent._check_action_loop([action])
	assert warning is None


def test_loop_detection_no_false_positive_empty_history():
	"""Loop detection should NOT fire with no history."""
	agent = _make_agent_with_signatures([])
	action = _make_mock_action('ClickElementAction', index=None)
	warning = agent._check_action_loop([action])
	assert warning is None


def test_record_action_signatures_caps_at_10():
	"""_record_action_signatures should keep at most 10 entries."""
	agent = _make_agent_with_signatures([f'sig{i}' for i in range(9)])
	actions = [_make_mock_action('ClickElementAction', index=None) for _ in range(5)]
	agent._record_action_signatures(actions, 'https://example.com')
	assert len(agent.state.recent_action_signatures) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# 6. Action grounding validation — _validate_action_grounding
# ─────────────────────────────────────────────────────────────────────────────

async def _make_grounding_agent(
	selector_map: dict,
	current_nav_gen: int = 5,
	cached_dom: bool = True,
):
	"""Build a minimal mock agent for grounding validation testing."""
	from browser_use.agent.views import AgentState
	from browser_use.agent.service import Agent

	agent = MagicMock()
	agent.state = AgentState()
	agent.browser_session = MagicMock()
	agent.browser_session.nav_gen = current_nav_gen

	if cached_dom:
		dom_state = MagicMock()
		dom_state.selector_map = selector_map
		bss = MagicMock()
		bss.dom_state = dom_state
		agent.browser_session._cached_browser_state_summary = bss
	else:
		agent.browser_session._cached_browser_state_summary = None

	# fingerprint reacquisition always fails in these tests
	agent.browser_session.reacquire_element_by_fingerprint = AsyncMock(return_value=None)

	# Bind the real methods (including container specificity helpers called by grounding)
	agent._CONTAINER_TAGS = Agent._CONTAINER_TAGS
	agent._get_interactive_descendants_in_map = Agent._get_interactive_descendants_in_map.__get__(agent, type(agent))
	agent._check_container_specificity = Agent._check_container_specificity.__get__(agent, type(agent))
	agent._validate_action_grounding = Agent._validate_action_grounding.__get__(agent, type(agent))
	return agent


async def test_grounding_validation_success():
	"""Grounding should succeed for an index present in the current selector map."""
	selector_map = {1: MagicMock()}
	agent = await _make_grounding_agent(selector_map, current_nav_gen=5)
	action = _make_mock_action('ClickElementAction', index=1)
	result = await agent._validate_action_grounding(action, captured_nav_gen=5)
	assert result.is_valid is True


async def test_grounding_validation_stale_index():
	"""Grounding should fail if the index is not in the current selector map and fingerprint reacquisition fails."""
	selector_map = {2: MagicMock()}  # index 1 is NOT present
	agent = await _make_grounding_agent(selector_map, current_nav_gen=5)
	action = _make_mock_action('ClickElementAction', index=1)
	result = await agent._validate_action_grounding(action, captured_nav_gen=5)
	assert result.is_valid is False
	assert 'selector map' in result.reason.lower() or 'not in' in result.reason.lower()


async def test_grounding_validation_nav_gen_changed():
	"""Grounding should fail if nav_gen changed since state was captured (navigation occurred)."""
	selector_map = {1: MagicMock()}
	agent = await _make_grounding_agent(selector_map, current_nav_gen=6)  # nav_gen advanced
	action = _make_mock_action('ClickElementAction', index=1)
	result = await agent._validate_action_grounding(action, captured_nav_gen=5)  # captured at 5
	assert result.is_valid is False
	assert 'nav_gen' in result.reason.lower() or 'navigat' in result.reason.lower() or 'changed' in result.reason.lower()


async def test_grounding_validation_no_index():
	"""Actions without an index (navigate, done, etc.) should always pass grounding."""
	selector_map = {}
	agent = await _make_grounding_agent(selector_map, current_nav_gen=5)
	# NavigateAction has no index
	action = MagicMock()
	action.__class__.__name__ = 'NavigateAction'
	action.model_dump.return_value = {'navigate': {'url': 'https://example.com'}}
	result = await agent._validate_action_grounding(action, captured_nav_gen=5)
	assert result.is_valid is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Guardrails — _email_is_otp_capable generalisation
#
# Note: agent_service_lib imports mailbox_service which uses the `crypt` stdlib
# module (unavailable on Windows). We skip these tests on Windows — they pass
# in CI (Linux). The logic being tested is a pure domain-matching function.
# ─────────────────────────────────────────────────────────────────────────────

import sys

_GUARDRAILS_SKIP = pytest.mark.skipif(
	sys.platform == 'win32',
	reason='agent_service_lib.mailbox_service uses crypt module unavailable on Windows',
)


@_GUARDRAILS_SKIP
def test_email_is_otp_capable_default_domain():
	"""Should return True for emails matching the default OTP domain."""
	from agent_service_lib.service_agent_guardrails import _email_is_otp_capable

	assert _email_is_otp_capable('user@pathix.io', ['pathix.io']) is True
	assert _email_is_otp_capable('USER@PATHIX.IO', ['pathix.io']) is True  # case-insensitive


@_GUARDRAILS_SKIP
def test_email_is_otp_capable_custom_domain():
	"""Should work with any configured domain, not just the hardcoded one."""
	from agent_service_lib.service_agent_guardrails import _email_is_otp_capable

	assert _email_is_otp_capable('test@mycompany.com', ['mycompany.com']) is True
	assert _email_is_otp_capable('test@other.io', ['mycompany.com']) is False


@_GUARDRAILS_SKIP
def test_email_is_otp_capable_multiple_domains():
	"""Should return True if any of multiple configured domains match."""
	from agent_service_lib.service_agent_guardrails import _email_is_otp_capable

	domains = ['acme.io', 'beta.com']
	assert _email_is_otp_capable('user@acme.io', domains) is True
	assert _email_is_otp_capable('user@beta.com', domains) is True
	assert _email_is_otp_capable('user@other.com', domains) is False


@_GUARDRAILS_SKIP
def test_email_is_otp_capable_empty_domains():
	"""Empty domain list should always return False (no OTP supported)."""
	from agent_service_lib.service_agent_guardrails import _email_is_otp_capable

	assert _email_is_otp_capable('user@anything.com', []) is False


# ─────────────────────────────────────────────────────────────────────────────
# 8. BrowserStateSummary backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

def test_browser_state_summary_new_fields_default():
	"""BrowserStateSummary should default new fields to None / empty list."""
	state = _make_browser_state()
	assert state.page_readiness is None
	assert state.blocking_overlays == []


def test_browser_state_summary_with_readiness_and_overlay():
	"""BrowserStateSummary should correctly store page_readiness and blocking_overlays."""
	readiness = PageReadinessInfo(is_ready=False, ready_state='loading', stable_for_ms=0)
	overlay = OverlayInfo(is_blocking=True, description='modal', close_index=5, z_index=1000, coverage_pct=0.5)
	state = _make_browser_state(page_readiness=readiness, blocking_overlays=[overlay])
	assert state.page_readiness is readiness
	assert len(state.blocking_overlays) == 1
	assert state.blocking_overlays[0].description == 'modal'


# ─────────────────────────────────────────────────────────────────────────────
# 9. Goal-relevant modal classification
# ─────────────────────────────────────────────────────────────────────────────

def test_goal_relevant_modal_classified_correctly():
	"""An overlay with active inputs + focused element should be classified as goal-relevant."""
	from browser_use.agent.prompts import _is_goal_relevant_overlay

	overlay = OverlayInfo(
		is_blocking=True, description='modal dialog', close_index=None,
		z_index=9999, coverage_pct=0.8,
		has_active_inputs=True, has_focused_element=True, has_submit_cta=False,
	)
	assert _is_goal_relevant_overlay(overlay) is True


def test_goal_relevant_modal_via_submit_cta():
	"""An overlay with active inputs + submit CTA (but no focus) should also be goal-relevant."""
	from browser_use.agent.prompts import _is_goal_relevant_overlay

	overlay = OverlayInfo(
		is_blocking=True, description='modal dialog', close_index=None,
		z_index=9999, coverage_pct=0.8,
		has_active_inputs=True, has_focused_element=False, has_submit_cta=True,
	)
	assert _is_goal_relevant_overlay(overlay) is True


def test_unrelated_modal_classified_as_blocker():
	"""An overlay with no active inputs should NOT be goal-relevant (unrelated blocker)."""
	from browser_use.agent.prompts import _is_goal_relevant_overlay

	overlay = OverlayInfo(
		is_blocking=True, description='backdrop overlay', close_index=3,
		z_index=500, coverage_pct=0.5,
		has_active_inputs=False, has_focused_element=False, has_submit_cta=False,
	)
	assert _is_goal_relevant_overlay(overlay) is False


def test_goal_relevant_overlay_banner_says_active_dialog():
	"""A goal-relevant blocking overlay should produce ACTIVE DIALOG banner, not BLOCKING OVERLAY."""
	overlay = OverlayInfo(
		is_blocking=True, description='modal dialog', close_index=None,
		z_index=9999, coverage_pct=0.8,
		has_active_inputs=True, has_focused_element=True, has_submit_cta=True,
	)
	state = _make_browser_state(blocking_overlays=[overlay])
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert 'ACTIVE DIALOG' in banner
	assert 'BLOCKING OVERLAY' not in banner


def test_unrelated_modal_banner_says_blocking_overlay():
	"""An unrelated blocking overlay should produce BLOCKING OVERLAY banner, not ACTIVE DIALOG."""
	overlay = OverlayInfo(
		is_blocking=True, description='modal dialog', close_index=5,
		z_index=9999, coverage_pct=0.8,
		has_active_inputs=False, has_focused_element=False, has_submit_cta=False,
	)
	state = _make_browser_state(blocking_overlays=[overlay])
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert 'BLOCKING OVERLAY' in banner
	assert 'ACTIVE DIALOG' not in banner
	assert '[5]' in banner  # close button index shown


# ─────────────────────────────────────────────────────────────────────────────
# 10. Loop detection — hard block + auth action exclusion
# ─────────────────────────────────────────────────────────────────────────────

def test_loop_hard_block_after_third_repeat():
	"""Same signature 3x in history should return LOOP_BLOCKED signal."""
	url = 'https://example.com/page'
	url_bucket = hashlib.md5(url.encode()).hexdigest()[:8]
	sig = f'ClickElementAction::{url_bucket}'
	# 3 identical signatures in history
	agent = _make_agent_with_signatures([sig, sig, sig])
	action = _make_mock_action('ClickElementAction', index=None)
	result = agent._check_action_loop([action])
	assert result is not None
	assert result.startswith('LOOP_BLOCKED:')


def test_loop_warning_at_second_repeat_not_block():
	"""Same signature 2x in history should return a warning string, not a LOOP_BLOCKED signal."""
	url = 'https://example.com/page'
	url_bucket = hashlib.md5(url.encode()).hexdigest()[:8]
	sig = f'ClickElementAction::{url_bucket}'
	# 2 identical signatures in history → warning
	agent = _make_agent_with_signatures([sig, sig])
	action = _make_mock_action('ClickElementAction', index=None)
	result = agent._check_action_loop([action])
	assert result is not None
	assert not result.startswith('LOOP_BLOCKED:')
	assert 'LOOP' in result.upper()


def test_auth_actions_excluded_from_loop_signatures():
	"""Auth/infra actions should return None from _compute_action_signature."""
	agent = _make_agent_with_signatures([])
	for excluded_type in ('fetch_mailbox_otp', 'fetch_mailbox_verification_link', 'wait'):
		action = _make_mock_action(excluded_type, index=None)
		sig = agent._compute_action_signature(action, 'https://example.com')
		assert sig is None, f'Expected None for {excluded_type}, got {sig!r}'


def test_record_action_signatures_skips_excluded_types():
	"""_record_action_signatures should not append entries for excluded action types."""
	agent = _make_agent_with_signatures([])
	actions = [_make_mock_action('fetch_mailbox_otp', index=None)]
	agent._record_action_signatures(actions, 'https://example.com')
	assert len(agent.state.recent_action_signatures) == 0, (
		'Excluded action types should not be recorded in loop history'
	)


def test_screenshot_done_excluded_from_loop_signatures():
	"""screenshot and done action types should also be excluded from loop signature tracking."""
	agent = _make_agent_with_signatures([])
	for excluded_type in ('screenshot', 'take_screenshot', 'capture_screenshot', 'done'):
		action = _make_mock_action(excluded_type, index=None)
		sig = agent._compute_action_signature(action, 'https://example.com')
		assert sig is None, f'Expected None for {excluded_type}, got {sig!r}'


# ─────────────────────────────────────────────────────────────────────────────
# 11. Phase 3: is_dialog_type improves goal-relevant overlay classification
# ─────────────────────────────────────────────────────────────────────────────

def test_dialog_type_with_active_inputs_is_goal_relevant():
	"""role=dialog overlay with active inputs should be goal-relevant even without focus or CTA."""
	from browser_use.agent.prompts import _is_goal_relevant_overlay

	overlay = OverlayInfo(
		is_blocking=True, description='Add Category', close_index=None,
		z_index=9999, coverage_pct=0.6,
		has_active_inputs=True, has_focused_element=False, has_submit_cta=False,
		is_dialog_type=True,
	)
	assert _is_goal_relevant_overlay(overlay) is True


def test_dialog_type_without_active_inputs_is_not_goal_relevant():
	"""role=dialog overlay with NO active inputs should NOT be goal-relevant (e.g. confirmation dialog with only text)."""
	from browser_use.agent.prompts import _is_goal_relevant_overlay

	overlay = OverlayInfo(
		is_blocking=True, description='info dialog', close_index=3,
		z_index=9999, coverage_pct=0.4,
		has_active_inputs=False, has_focused_element=False, has_submit_cta=False,
		is_dialog_type=True,
	)
	assert _is_goal_relevant_overlay(overlay) is False


def test_dialog_type_banner_says_active_dialog_not_blocking():
	"""role=dialog overlay with inputs should show ACTIVE DIALOG banner."""
	overlay = OverlayInfo(
		is_blocking=True, description='Add Category', close_index=None,
		z_index=9999, coverage_pct=0.6,
		has_active_inputs=True, has_focused_element=False, has_submit_cta=False,
		is_dialog_type=True,
	)
	state = _make_browser_state(blocking_overlays=[overlay])
	prompt = _make_prompt(state)
	banner = prompt._get_page_readiness_and_overlay_banner()
	assert 'ACTIVE DIALOG' in banner
	assert 'BLOCKING OVERLAY' not in banner


# ─────────────────────────────────────────────────────────────────────────────
# 12. Phase 3: Container div click rejection via _check_container_specificity
# ─────────────────────────────────────────────────────────────────────────────

def _make_container_node(tag: str, child_backend_ids: list[int], role: str = '') -> MagicMock:
	"""Build a mock DOM node that looks like a container with interactive children."""
	node = MagicMock()
	node.tag_name = tag
	node.node_name = tag
	node.attributes = {'role': role} if role else {}
	# Build children_nodes that each have a backend_node_id
	children = []
	for bid in child_backend_ids:
		child = MagicMock()
		child.backend_node_id = bid
		child.children_nodes = []
		children.append(child)
	node.children_nodes = children
	return node


def _make_container_agent():
	"""Minimal agent mock for container specificity tests."""
	from browser_use.agent.service import Agent
	agent = MagicMock()
	agent._CONTAINER_TAGS = Agent._CONTAINER_TAGS
	agent._get_interactive_descendants_in_map = Agent._get_interactive_descendants_in_map.__get__(agent, type(agent))
	agent._check_container_specificity = Agent._check_container_specificity.__get__(agent, type(agent))
	return agent


def test_container_div_with_descendants_rejected():
	"""Clicking a container div that has interactive descendants in the selector map should be rejected."""
	agent = _make_container_agent()
	child_node = MagicMock()
	child_node.backend_node_id = 42
	child_node.children_nodes = []
	container = _make_container_node('div', [])
	container.children_nodes = [child_node]

	selector_map = {42: child_node}  # child is in the map
	action = _make_mock_action('ClickElementAction', index=1)
	result = agent._check_container_specificity(container, selector_map, action)
	assert result is not None
	assert result.is_valid is False
	assert '42' in result.reason  # descendant index mentioned


def test_container_div_no_descendants_allowed():
	"""Clicking a container div with NO interactive descendants in selector map should pass (only option)."""
	agent = _make_container_agent()
	container = _make_container_node('div', [])
	selector_map = {}  # no descendants in map
	action = _make_mock_action('ClickElementAction', index=1)
	result = agent._check_container_specificity(container, selector_map, action)
	assert result is None  # no rejection, container is the only option


def test_button_tag_not_rejected_by_container_check():
	"""Interactive tags like <button> should never be rejected by container specificity."""
	agent = _make_container_agent()
	child_node = MagicMock()
	child_node.backend_node_id = 10
	child_node.children_nodes = []
	# Even a <button> containing children is fine
	node = MagicMock()
	node.tag_name = 'button'
	node.node_name = 'button'
	node.attributes = {}
	node.children_nodes = [child_node]
	selector_map = {10: child_node}
	action = _make_mock_action('ClickElementAction', index=5)
	result = agent._check_container_specificity(node, selector_map, action)
	assert result is None  # button is never a container


def test_div_with_role_button_not_rejected():
	"""A <div role='button'> should not be rejected even if it contains descendants."""
	agent = _make_container_agent()
	child_node = MagicMock()
	child_node.backend_node_id = 7
	child_node.children_nodes = []
	container = _make_container_node('div', [], role='button')
	container.children_nodes = [child_node]
	selector_map = {7: child_node}
	action = _make_mock_action('ClickElementAction', index=2)
	result = agent._check_container_specificity(container, selector_map, action)
	assert result is None  # explicit role overrides container rejection


def test_non_click_action_not_rejected():
	"""Container specificity check should only apply to click actions, not input/scroll/etc."""
	agent = _make_container_agent()
	child_node = MagicMock()
	child_node.backend_node_id = 99
	child_node.children_nodes = []
	container = _make_container_node('div', [])
	container.children_nodes = [child_node]
	selector_map = {99: child_node}
	action = _make_mock_action('InputTextAction', index=1)
	result = agent._check_container_specificity(container, selector_map, action)
	assert result is None  # input actions not rejected
