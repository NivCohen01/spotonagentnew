from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import logging
import time
from typing import Any, Optional

from browser_use.agent.views import ActionResult
from browser_use.tools.service import Tools
from browser_use.tools.views import InputTextAction

from .mailbox_service import (
    DEFAULT_DOMAIN,
    compute_password,
    derive_base_from_url,
    ensure_mailbox_exists,
    fetch_latest_otp_imap,
    get_next_available_email,
)
from .service_agent_auth_utils import (
    _candidate_mailbox_passwords,
    _classify_field,
    _detect_otp_indices,
    _page_has_password_or_otp,
    _save_workspace_domain_if_applicable,
    _redact_url,
    detect_email_link_verification_heuristic,
    detect_otp_verification_heuristic,
    detect_login_error_heuristic,
    FetchVerifyLinkParams,
    PageStateClassification,
)

_COMMON_DUMMY = {
    "user@example.com",
    "test@test.com",
    "example@example.com",
    "dummyPassword123",
    "dummyPassword",
    "password",
    "test123",
}
_OTP_SUBMISSION_PENDING_SECONDS = 20.0
_OTP_POST_SUBMIT_STABILIZATION_SECONDS = 3.0


def _record_auth_event(sess: Any, event: dict[str, Any]) -> None:
    events = getattr(sess, "auth_events", None)
    if events is None:
        return
    seen = getattr(sess, "_auth_event_keys", None)
    if seen is None:
        seen = set()
        setattr(sess, "_auth_event_keys", seen)
    key_parts = [
        str(event.get("action") or event.get("type") or ""),
        str(event.get("step") or ""),
        str(event.get("page_url") or ""),
        str(event.get("params", {}).get("email") or ""),
    ]
    key = "|".join(key_parts)
    if key in seen:
        return
    events.append(event)
    seen.add(key)


async def ensure_signup_mailbox(sess: Any, log: Optional[logging.Logger] = None) -> dict[str, str]:
    """
    Ensure sess.generated_credentials exists. Creates a deterministic pathix mailbox if needed.
    Mirrors the previous inline ensure_signup_mailbox() behavior.
    """
    log = log or logging.getLogger("service")

    if getattr(sess, "generated_credentials", None):
        return sess.generated_credentials

    base_source = getattr(sess, "start_url", None) or getattr(sess, "task", None) or "pathix"
    email_addr = await get_next_available_email(derive_base_from_url(base_source))
    password_val = compute_password(email_addr)
    await ensure_mailbox_exists(email_addr, password_val)

    sess.generated_credentials = {"email": email_addr, "password": password_val}
    sess.generated_credentials_created_at = dt.datetime.utcnow()

    log.info("Generated mailbox %s for session %s", email_addr, getattr(sess, "id", "<unknown>"))
    await _save_workspace_domain_if_applicable(sess, email_addr, password_val)
    return sess.generated_credentials


def _email_is_otp_capable(email: str, allowed_otp_domains: list[str]) -> bool:
    """Return True if the email domain is in the list of OTP-capable domains.

    General, site-agnostic check. Does NOT hardcode any specific domain name.
    Callers configure which domains support automated OTP retrieval.
    """
    email_lower = email.lower()
    return any(email_lower.endswith(f'@{d.lower().lstrip("@")}') for d in allowed_otp_domains)


def _get_otp_runtime_state(sess: Any) -> dict[str, Any]:
    state = getattr(sess, "otp_runtime_state", None)
    if isinstance(state, dict):
        return state
    state = {
        "generation": 0,
        "last_page_signature": None,
        "last_fields_signature": None,
        "last_fill_fingerprint": None,
        "last_fill_time": 0.0,
        "last_email": None,
        "submission_pending": False,
        "submission_started_at": 0.0,
        "submission_page_signature": None,
        "submission_fields_signature": None,
        "submission_fill_fingerprint": None,
    }
    setattr(sess, "otp_runtime_state", state)
    return state


def _stable_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def _read_attr(node: Any, key: str) -> str:
    attrs = getattr(node, "attributes", None) or {}
    raw = attrs.get(key)
    return str(raw).strip() if raw is not None else ""


async def _get_current_url_safe(browser_session) -> str:
    if not browser_session:
        return ""
    with contextlib.suppress(Exception):
        return await browser_session.get_current_page_url()
    return ""


def _compute_otp_fields_signature(selector_map: dict[int, Any], otp_indices: list[int]) -> str:
    parts: list[str] = []
    for idx in sorted(otp_indices):
        node = selector_map.get(idx)
        if not node:
            parts.append(f"{idx}:missing")
            continue
        tag = (getattr(node, "tag_name", None) or getattr(node, "node_name", None) or "").lower()
        attrs = [
            _read_attr(node, "type"),
            _read_attr(node, "autocomplete"),
            _read_attr(node, "name"),
            _read_attr(node, "id"),
            _read_attr(node, "maxlength"),
            _read_attr(node, "inputmode"),
            _read_attr(node, "pattern"),
        ]
        parts.append(f"{idx}:{tag}:{'|'.join(attrs)}")
    return _stable_hash("||".join(parts))


async def _read_otp_field_values(
    browser_session,
    selector_map: dict[int, Any],
    otp_indices: list[int],
) -> dict[int, str]:
    values: dict[int, str] = {}
    for idx in otp_indices:
        node = selector_map.get(idx)
        if not node:
            continue
        attrs = getattr(node, "attributes", None) or {}
        raw = attrs.get("value")
        if isinstance(raw, str):
            values[idx] = raw.strip()

    page = None
    with contextlib.suppress(Exception):
        page = await browser_session.get_current_page()
    if not page:
        return values

    try:
        js_indices = json.dumps([int(i) for i in otp_indices])
        script = (
            "() => {"
            f"const ids = {js_indices};"
            "const out = {};"
            "for (const idx of ids) {"
            "  const selector = `[data-highlight-index=\"${idx}\"]`;"
            "  const el = document.querySelector(selector);"
            "  if (!el) continue;"
            "  let raw = '';"
            "  if ('value' in el && typeof el.value === 'string') raw = el.value;"
            "  else if (el.textContent) raw = el.textContent;"
            "  out[String(idx)] = (raw || '').trim();"
            "}"
            "return out;"
            "}"
        )
        js_values = await page.evaluate(script)
        if isinstance(js_values, dict):
            for key, raw_val in js_values.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                if idx not in otp_indices:
                    continue
                if isinstance(raw_val, str):
                    values[idx] = raw_val.strip()
    except Exception:
        pass

    return values


def _is_probable_otp_submit_click(action_data: dict[str, Any], selector_map: dict[int, Any]) -> bool:
    click_data = action_data.get("click")
    if not isinstance(click_data, dict):
        return False
    idx = click_data.get("index")
    if idx is None:
        return False
    node = selector_map.get(idx)
    if not node:
        return False
    tag = (getattr(node, "tag_name", None) or getattr(node, "node_name", None) or "").lower()
    attrs = getattr(node, "attributes", None) or {}
    role = str(attrs.get("role") or "").strip().lower()
    input_type = str(attrs.get("type") or "").strip().lower()
    if tag == "button":
        return True
    if tag == "input" and input_type in {"submit", "button", "image"}:
        return True
    return role in {"button", "menuitem", "option"}


def install_auth_guardrails(tools: Tools, sess: Any, allowed_otp_domains: list[str] | None = None) -> None:
    """Install runtime guardrails on tools.act.

    Args:
        tools: The Tools instance to wrap.
        sess: The session object (arbitrary; accessed via getattr).
        allowed_otp_domains: Domains for which automated OTP retrieval is supported.
            Defaults to [DEFAULT_DOMAIN] for backward compatibility.
            Pass a custom list to enable OTP for other mailbox domains.

    Guardrails applied:
    - OTP auto-fill when OTP fields are detected and email is in allowed_otp_domains
    - Prevent calling done while still unauthenticated (password/otp fields visible)
    - Block dummy credentials; enforce allowed credentials only
    - Auto-generate credentials during signup intent (via ensure_signup_mailbox)
    """
    if allowed_otp_domains is None:
        allowed_otp_domains = [DEFAULT_DOMAIN]

    log = logging.getLogger("service")

    allowed_creds_holder = {
        "email": (getattr(sess, "user_credentials", None) or {}).get("email") if getattr(sess, "user_credentials", None) else None,
        "password": (getattr(sess, "user_credentials", None) or {}).get("password") if getattr(sess, "user_credentials", None) else None,
    }
    if getattr(sess, "generated_credentials", None):
        if not allowed_creds_holder["email"]:
            allowed_creds_holder["email"] = sess.generated_credentials.get("email")
        if not allowed_creds_holder["password"]:
            allowed_creds_holder["password"] = sess.generated_credentials.get("password")

    orig_act = tools.act

    async def guarded_act(
        action,
        browser_session=None,
        page_extraction_llm=None,
        sensitive_data=None,
        available_file_paths=None,
        file_system=None,
        action_screenshot_recorder=None,
        step_number=0,
    ):
        allowed_email = allowed_creds_holder.get("email") or (getattr(sess, "generated_credentials", None) or {}).get("email")
        allowed_password = allowed_creds_holder.get("password") or (getattr(sess, "generated_credentials", None) or {}).get("password")
        action_data = action.model_dump(exclude_unset=True)
        requires_auth = bool(
            getattr(sess, "signup_intent", False)
            or (getattr(sess, "intent", None) and getattr(sess.intent, "needs_auth", False))
        )

        otp_indices: list[int] = []
        selector_map: dict[int, Any] = {}
        otp_runtime_state = _get_otp_runtime_state(sess)
        otp_fill_fingerprint: str | None = None
        otp_submission_click_candidate = False
        if browser_session:
            with contextlib.suppress(Exception):
                selector_map = await browser_session.get_selector_map() or {}
            otp_indices = await _detect_otp_indices(browser_session)
            if otp_indices:
                target_email = ""
                if allowed_email and _email_is_otp_capable(allowed_email, allowed_otp_domains):
                    target_email = allowed_email.lower()
                elif _email_is_otp_capable((getattr(sess, "user_credentials", None) or {}).get("email", ""), allowed_otp_domains):
                    target_email = (getattr(sess, "user_credentials", None) or {}).get("email", "").lower()

                current_url = await _get_current_url_safe(browser_session)
                redacted_url = _redact_url(current_url)
                page_signature = _stable_hash(redacted_url)
                fields_signature = _compute_otp_fields_signature(selector_map, otp_indices)
                otp_fill_fingerprint = _stable_hash(f"{target_email}|{page_signature}|{fields_signature}")
                otp_values = await _read_otp_field_values(browser_session, selector_map, otp_indices)
                filled_count = sum(1 for idx in otp_indices if (otp_values.get(idx) or "").strip())
                otp_is_filled = bool(otp_indices) and filled_count >= len(otp_indices)
                now = time.time()

                material_change = (
                    otp_runtime_state.get("last_page_signature") != page_signature
                    or otp_runtime_state.get("last_fields_signature") != fields_signature
                    or otp_runtime_state.get("last_email") != target_email
                )
                if material_change:
                    otp_runtime_state["generation"] = int(otp_runtime_state.get("generation") or 0) + 1
                    otp_runtime_state["submission_pending"] = False
                    otp_runtime_state["submission_started_at"] = 0.0
                    otp_runtime_state["submission_page_signature"] = None
                    otp_runtime_state["submission_fields_signature"] = None
                    otp_runtime_state["submission_fill_fingerprint"] = None
                    otp_runtime_state["last_page_signature"] = page_signature
                    otp_runtime_state["last_fields_signature"] = fields_signature
                    otp_runtime_state["last_email"] = target_email

                submission_pending = bool(otp_runtime_state.get("submission_pending"))
                if submission_pending:
                    pending_started = float(otp_runtime_state.get("submission_started_at") or 0.0)
                    pending_page_sig = otp_runtime_state.get("submission_page_signature")
                    pending_fields_sig = otp_runtime_state.get("submission_fields_signature")
                    pending_timed_out = now - pending_started > _OTP_SUBMISSION_PENDING_SECONDS
                    pending_changed = (
                        not otp_indices
                        or pending_page_sig != page_signature
                        or pending_fields_sig != fields_signature
                        or not otp_is_filled
                    )
                    if pending_timed_out or pending_changed:
                        otp_runtime_state["submission_pending"] = False
                        otp_runtime_state["submission_started_at"] = 0.0
                        otp_runtime_state["submission_page_signature"] = None
                        otp_runtime_state["submission_fields_signature"] = None
                        otp_runtime_state["submission_fill_fingerprint"] = None
                    else:
                        # OTP was just submitted and state has not materially changed yet.
                        # Keep waiting for success/failure before attempting another fill.
                        pass

                if otp_is_filled and otp_fill_fingerprint:
                    otp_runtime_state["last_fill_fingerprint"] = otp_fill_fingerprint
                    otp_runtime_state["last_fill_time"] = now

                already_satisfied = bool(
                    otp_fill_fingerprint
                    and otp_runtime_state.get("last_fill_fingerprint") == otp_fill_fingerprint
                    and otp_is_filled
                )
                waiting_on_submit = bool(
                    otp_runtime_state.get("submission_pending")
                    and (now - float(otp_runtime_state.get("submission_started_at") or 0.0)) <= _OTP_SUBMISSION_PENDING_SECONDS
                )

                if not already_satisfied and not waiting_on_submit:
                    if not target_email:
                        return ActionResult(error="OTP required; cannot continue automatically.")
                    if not _email_is_otp_capable(target_email, allowed_otp_domains):
                        return ActionResult(error="OTP required; cannot continue automatically.")
                    password_candidates = _candidate_mailbox_passwords(sess, target_email, allowed_password)
                    mailbox_password = password_candidates[0] if password_candidates else None
                    if not mailbox_password:
                        return ActionResult(error="OTP required; IMAP unavailable.")

                    since_ts = getattr(sess, "generated_credentials_created_at", None) or (dt.datetime.utcnow() - dt.timedelta(minutes=10))
                    try:
                        code = await fetch_latest_otp_imap(target_email, mailbox_password, since_ts, attempts=18, interval=10)
                    except Exception as exc:
                        log.info("OTP auto-fill aborted: IMAP fetch failed for %s | %r", target_email, exc)
                        return ActionResult(error="OTP required; IMAP unavailable.")
                    if not code:
                        return ActionResult(error="OTP not received")

                    async def _send_otp(idx: int, txt: str):
                        payload = InputTextAction(index=idx, text=txt, clear=True)
                        otp_action = action.__class__(**{"input": payload})
                        return await orig_act(
                            action=otp_action,
                            browser_session=browser_session,
                            page_extraction_llm=page_extraction_llm,
                            sensitive_data=sensitive_data,
                            available_file_paths=available_file_paths,
                            file_system=file_system,
                            action_screenshot_recorder=action_screenshot_recorder,
                            step_number=step_number,
                        )

                    if len(otp_indices) == 1:
                        await _send_otp(otp_indices[0], code)
                    else:
                        for pos, digit in enumerate(code):
                            if pos >= len(otp_indices):
                                break
                            await _send_otp(otp_indices[pos], digit)

                    otp_values = await _read_otp_field_values(browser_session, selector_map, otp_indices)
                    filled_count = sum(1 for idx in otp_indices if (otp_values.get(idx) or "").strip())
                    if filled_count >= len(otp_indices):
                        otp_runtime_state["last_fill_fingerprint"] = otp_fill_fingerprint
                        otp_runtime_state["last_fill_time"] = time.time()

                    log.info("OTP auto-fill completed for %s on %s", target_email, current_url or "<unknown>")
                    if not hasattr(sess, "otp_filled_cache"):
                        sess.otp_filled_cache = {}
                    sess.otp_filled_cache[current_url or ""] = code
                    sess.otp_filled_cache[f"email:{target_email}"] = code
                    if otp_fill_fingerprint:
                        sess.otp_filled_cache[f"fingerprint:{otp_fill_fingerprint}"] = code
                    _record_auth_event(
                        sess,
                        {
                            "action": "otp",
                            "step": getattr(sess, "current_step", None),
                            "page_url": current_url,
                            "params": {
                                "email": target_email,
                                "since_ts": since_ts.isoformat() if isinstance(since_ts, dt.datetime) else None,
                                "digits": len(code),
                                "generation": otp_runtime_state.get("generation"),
                            },
                            "ts": int(dt.datetime.utcnow().timestamp() * 1000),
                        },
                    )

                otp_submission_click_candidate = bool(
                    otp_is_filled
                    and _is_probable_otp_submit_click(action_data, selector_map)
                )

        if "fetch_mailbox_verification_link" in action_data and browser_session:
            current_url = ""
            with contextlib.suppress(Exception):
                current_url = await browser_session.get_current_page_url()
            redacted_url = _redact_url(current_url)

            attempted = getattr(sess, "verification_attempted_urls", None)
            if attempted is None:
                sess.verification_attempted_urls = set()
                attempted = sess.verification_attempted_urls
            if redacted_url and redacted_url in attempted:
                return ActionResult(error="Verification link not received")

            result = await orig_act(
                action=action,
                browser_session=browser_session,
                page_extraction_llm=page_extraction_llm,
                sensitive_data=sensitive_data,
                available_file_paths=available_file_paths,
                file_system=file_system,
                action_screenshot_recorder=action_screenshot_recorder,
                step_number=step_number,
            )
            if getattr(result, "error", None):
                return result

            pending_url = getattr(sess, "pending_verification_url", None)
            if not pending_url:
                return ActionResult(error="Verification link not received")

            attempted.add(redacted_url or "")
            try:
                await browser_session.navigate_to(pending_url, new_tab=False)
            except Exception as exc:
                log.info("Verification link navigation failed for %s | %r", allowed_email or "<unknown>", exc)
                return ActionResult(error="Verification link not received")
            finally:
                with contextlib.suppress(Exception):
                    setattr(sess, "pending_verification_url", None)

            redacted_link = _redact_url(pending_url)
            log.info("Opened verification link: %s", redacted_link)
            _record_auth_event(
                sess,
                {
                    "action": "verification_link",
                    "step": getattr(sess, "current_step", None),
                    "page_url": current_url,
                    "params": {
                        "email": allowed_email,
                        "since_ts": (getattr(sess, "generated_credentials_created_at", None) or dt.datetime.utcnow()).isoformat(),
                    },
                    "ts": int(dt.datetime.utcnow().timestamp() * 1000),
                },
            )
            return ActionResult(
                extracted_content="Opened verification link.",
                long_term_memory="Opened verification link from mailbox.",
                metadata={"url": redacted_link},
            )

        # Block done if still unauthenticated (signup/auth flows)
        if "done" in action_data and requires_auth and browser_session:
            if otp_indices or await _page_has_password_or_otp(browser_session):
                return ActionResult(error="Not authenticated yet; continue.")
            current_url = ""
            with contextlib.suppress(Exception):
                current_url = await browser_session.get_current_page_url()
            redacted_url = _redact_url(current_url)

            # Check page_state from session first (extracted from agent's memory)
            page_state: PageStateClassification | None = getattr(sess, "page_state", None)
            if page_state:
                if page_state.needs_email_link:
                    return ActionResult(error="Not authenticated yet; continue.")
                if page_state.needs_otp:
                    return ActionResult(error="Not authenticated yet; continue.")
            else:
                # Fallback to LLM classification if page_state not available
                verification_state = await detect_email_link_verification_heuristic(
                    browser_session,
                    page_extraction_llm,
                    current_url=redacted_url,
                )
                if verification_state.needs_email_link_verification:
                    return ActionResult(error="Not authenticated yet; continue.")
                otp_state = await detect_otp_verification_heuristic(
                    browser_session,
                    page_extraction_llm,
                    current_url=redacted_url,
                )
                if otp_state.needs_otp_verification:
                    return ActionResult(error="Not authenticated yet; continue.")

        # Auto-handle email link verification when required (non-OTP flows only).
        if browser_session and requires_auth and not otp_indices and "done" not in action_data:
            current_url = ""
            with contextlib.suppress(Exception):
                current_url = await browser_session.get_current_page_url()
            redacted_url = _redact_url(current_url)

            attempted = getattr(sess, "verification_attempted_urls", None)
            if attempted is None:
                sess.verification_attempted_urls = set()
                attempted = sess.verification_attempted_urls

            if redacted_url and redacted_url in attempted:
                pass
            else:
                # Check page_state from session first (extracted from agent's memory)
                page_state = getattr(sess, "page_state", None)
                needs_email_link = False
                if page_state:
                    needs_email_link = page_state.needs_email_link
                else:
                    # Fallback to LLM classification if page_state not available
                    verification_state = await detect_email_link_verification_heuristic(
                        browser_session,
                        page_extraction_llm,
                        current_url=redacted_url,
                    )
                    needs_email_link = verification_state.needs_email_link_verification

                if needs_email_link:
                    target_email = None
                    if allowed_email and _email_is_otp_capable(allowed_email, allowed_otp_domains):
                        target_email = allowed_email.lower()
                    elif _email_is_otp_capable((getattr(sess, "user_credentials", None) or {}).get("email", ""), allowed_otp_domains):
                        target_email = (getattr(sess, "user_credentials", None) or {}).get("email", "").lower()

                    if not target_email:
                        return ActionResult(error="Verification email required; cannot continue automatically.")

                    attempted.add(redacted_url or "")
                    fetch_link = getattr(sess, "fetch_mailbox_verification_link", None)
                    if not fetch_link:
                        return ActionResult(error="Verification link not received")

                    params = FetchVerifyLinkParams(email=target_email)
                    result = await fetch_link(
                        params=params,
                        browser_session=browser_session,
                        page_extraction_llm=page_extraction_llm,
                    )
                    if getattr(result, "error", None):
                        return result

                    pending_url = getattr(sess, "pending_verification_url", None)
                    if not pending_url:
                        return ActionResult(error="Verification link not received")

                    try:
                        await browser_session.navigate_to(pending_url, new_tab=False)
                    except Exception as exc:
                        log.info("Verification link navigation failed for %s | %r", target_email, exc)
                        return ActionResult(error="Verification link not received")
                    finally:
                        with contextlib.suppress(Exception):
                            setattr(sess, "pending_verification_url", None)

                    redacted_link = _redact_url(pending_url)
                    log.info("Opened verification link for %s: %s", target_email, redacted_link)
                    _record_auth_event(
                        sess,
                        {
                            "action": "verification_link",
                            "step": getattr(sess, "current_step", None),
                            "page_url": current_url,
                            "params": {
                                "email": target_email,
                                "since_ts": (getattr(sess, "generated_credentials_created_at", None) or dt.datetime.utcnow()).isoformat(),
                            },
                            "ts": int(dt.datetime.utcnow().timestamp() * 1000),
                        },
                    )
                    return ActionResult(
                        extracted_content="Opened verification link.",
                        long_term_memory="Opened verification link from mailbox.",
                        metadata={"url": redacted_link},
                    )

        # If a pending verification URL exists, prefer it over placeholder navigation.
        if "navigate" in action_data and browser_session:
            pending_url = getattr(sess, "pending_verification_url", None)
            if pending_url:
                nav_params = action_data.get("navigate") or {}
                requested_url = nav_params.get("url") if isinstance(nav_params, dict) else None
                if requested_url and _redact_url(requested_url) == _redact_url(pending_url):

                    def _rebuild_action_with_url(new_url: str):
                        action_dict = action.model_dump(exclude_unset=True)
                        nav_data = action_dict.get("navigate")
                        if isinstance(nav_data, dict):
                            nav_data = dict(nav_data)
                        elif hasattr(nav_data, "model_dump"):
                            nav_data = nav_data.model_dump()
                        else:
                            nav_data = {}
                        nav_data["url"] = new_url
                        action_dict["navigate"] = nav_data
                        return action.__class__(**action_dict)

                    action = _rebuild_action_with_url(pending_url)
                    with contextlib.suppress(Exception):
                        setattr(sess, "pending_verification_url", None)

        # Credential enforcement on input actions
        if "input" in action_data and action_data["input"] is not None:
            params = action_data["input"]
            if isinstance(params, dict):
                text_val = params.get("text")
                idx_val = params.get("index")
            else:
                text_val = getattr(params, "text", None)
                idx_val = getattr(params, "index", None)

            if text_val is not None:
                field_kind = "other"
                if browser_session and idx_val is not None:
                    with contextlib.suppress(Exception):
                        node = await browser_session.get_element_by_index(idx_val)
                        field_kind = _classify_field(node)

                dummy_pw_terms = {"examplepassword", "examplepassword123!", "dummypassword", "password123"}
                dummy_password = text_val in _COMMON_DUMMY or text_val.lower() in dummy_pw_terms
                placeholder_text = str(text_val).strip()
                placeholder_lower = placeholder_text.lower()
                looks_like_email_value = (
                    "@" in placeholder_text
                    and "." in placeholder_text
                    and " " not in placeholder_text
                    and not placeholder_text.startswith("@")
                )
                looks_like_email_placeholder = bool(
                    ("<" in placeholder_text and ">" in placeholder_text and "email" in placeholder_lower)
                    or placeholder_lower in {"<generated_email>", "<your email>", "<email>", "generated_email"}
                )
                if field_kind != "email" and (looks_like_email_value or looks_like_email_placeholder):
                    field_kind = "email"

                def _rebuild_action_with_text(new_text: str):
                    action_dict = action.model_dump(exclude_unset=True)
                    input_data = action_dict.get("input")
                    if isinstance(input_data, dict):
                        input_data = dict(input_data)
                    elif hasattr(input_data, "model_dump"):
                        input_data = input_data.model_dump()
                    else:
                        input_data = {}
                    input_data["text"] = new_text
                    action_dict["input"] = input_data
                    return action.__class__(**action_dict)

                agent_profile = getattr(sess, "agent_profile", None) or {}
                placeholder_is_tagged = "<" in placeholder_text and ">" in placeholder_text
                profile_value: Optional[str] = None
                if placeholder_is_tagged and field_kind not in {"email", "password"}:
                    if "first name" in placeholder_lower:
                        profile_value = agent_profile.get("first_name")
                    elif "last name" in placeholder_lower:
                        profile_value = agent_profile.get("last_name")
                    elif "full name" in placeholder_lower or placeholder_lower in {"<your name>", "<name>"}:
                        profile_value = agent_profile.get("full_name") or agent_profile.get("display_name")
                    elif "username" in placeholder_lower or "user name" in placeholder_lower:
                        profile_value = agent_profile.get("username")

                if profile_value:
                    action = _rebuild_action_with_text(profile_value)
                elif field_kind == "email":
                    if allowed_email:
                        # Use existing credentials
                        action = _rebuild_action_with_text(allowed_email)
                    elif getattr(sess, "signup_intent", False):
                        # Explicit signup intent - auto-generate credentials
                        log.info("Auto-generating credentials for email field (signup_intent=True)")
                        creds = await ensure_signup_mailbox(sess, log=log)
                        allowed_creds_holder["email"] = creds.get("email")
                        allowed_creds_holder["password"] = creds.get("password")
                        action = _rebuild_action_with_text(creds.get("email") or "")
                    # else: No credentials and no signup_intent - let agent decide (allow placeholder for demos)

                elif field_kind == "password":
                    if allowed_password:
                        action = _rebuild_action_with_text(allowed_password)
                    elif allowed_creds_holder.get("password"):
                        # Use password from recently generated credentials
                        action = _rebuild_action_with_text(allowed_creds_holder.get("password") or "")
                    elif getattr(sess, "signup_intent", False):
                        # Explicit signup intent - auto-generate credentials
                        log.info("Auto-generating credentials for password field (signup_intent=True)")
                        creds = await ensure_signup_mailbox(sess, log=log)
                        allowed_creds_holder["email"] = creds.get("email")
                        allowed_creds_holder["password"] = creds.get("password")
                        action = _rebuild_action_with_text(creds.get("password") or "")
                    # else: No credentials and no signup_intent - let agent decide (allow placeholder for demos)
                else:
                    # For otp/other fields, allow without credential enforcement.
                    pass

        # When OTP fields are already filled and a submit-like button is clicked,
        # enter a short pending state so helper logic waits for verification outcome.
        if otp_submission_click_candidate and otp_fill_fingerprint:
            otp_runtime_state["submission_pending"] = True
            otp_runtime_state["submission_started_at"] = time.time()
            otp_runtime_state["submission_page_signature"] = otp_runtime_state.get("last_page_signature")
            otp_runtime_state["submission_fields_signature"] = otp_runtime_state.get("last_fields_signature")
            otp_runtime_state["submission_fill_fingerprint"] = otp_fill_fingerprint

        # Execute the action
        result = await orig_act(
            action=action,
            browser_session=browser_session,
            page_extraction_llm=page_extraction_llm,
            sensitive_data=sensitive_data,
            available_file_paths=available_file_paths,
            file_system=file_system,
            action_screenshot_recorder=action_screenshot_recorder,
            step_number=step_number,
        )

        if browser_session and otp_runtime_state.get("submission_pending"):
            # Observe immediate post-submit outcome before allowing another OTP fetch/fill cycle.
            with contextlib.suppress(Exception):
                await browser_session.wait_for_stable_ui(
                    timeout=_OTP_POST_SUBMIT_STABILIZATION_SECONDS,
                    quiet_ms=250,
                )
            post_indices = await _detect_otp_indices(browser_session)
            if not post_indices:
                otp_runtime_state["submission_pending"] = False
                otp_runtime_state["submission_started_at"] = 0.0
                otp_runtime_state["submission_page_signature"] = None
                otp_runtime_state["submission_fields_signature"] = None
                otp_runtime_state["submission_fill_fingerprint"] = None

        # After click actions on auth pages, wait for navigation/page stability and check for login errors
        if "click" in action_data and browser_session and requires_auth:
            # Wait for page to stabilize after clicking (especially for form submissions)
            # This gives the page time to: submit the form, navigate, or show error messages
            try:
                await asyncio.sleep(1.5)  # Initial wait for network request to start
                await browser_session.wait_for_stable_ui(timeout=3.0, quiet_ms=300)
            except Exception as stability_exc:
                log.debug("UI stability wait after click: %r", stability_exc)

            current_url = ""
            with contextlib.suppress(Exception):
                current_url = await browser_session.get_current_page_url()

            # Check page_state from session first (extracted from agent's memory)
            page_state = getattr(sess, "page_state", None)
            has_error = False
            error_type: str | None = None
            error_msg: str | None = None

            if page_state:
                has_error = page_state.has_login_error or page_state.has_captcha
                if page_state.has_login_error:
                    error_type = page_state.error_type or "unknown"
                    error_msg = page_state.error_message or "Authentication error detected"
                elif page_state.has_captcha:
                    error_type = "captcha_required"
                    error_msg = "CAPTCHA detected"
            else:
                # Fallback to LLM classification if page_state not available
                error_state = await detect_login_error_heuristic(
                    browser_session,
                    page_extraction_llm,
                    current_url=current_url,
                )
                has_error = error_state.has_error
                error_type = error_state.error_type
                error_msg = error_state.error_message

            if has_error:
                otp_runtime_state["submission_pending"] = False
                otp_runtime_state["submission_started_at"] = 0.0
                otp_runtime_state["submission_page_signature"] = None
                otp_runtime_state["submission_fields_signature"] = None
                otp_runtime_state["submission_fill_fingerprint"] = None
                error_type = error_type or "unknown"
                error_msg = error_msg or "Authentication error detected"
                log.info("Login error detected: type=%s, message=%s", error_type, error_msg)

                # Record the auth event
                _record_auth_event(
                    sess,
                    {
                        "action": "login_error",
                        "step": getattr(sess, "current_step", None),
                        "page_url": current_url,
                        "params": {
                            "error_type": error_type,
                            "error_message": error_msg,
                        },
                        "ts": int(dt.datetime.utcnow().timestamp() * 1000),
                    },
                )

                # For CAPTCHA errors, inform the agent to use solve_captcha
                if error_type == "captcha_required":
                    return ActionResult(
                        error="CAPTCHA required. Use detect_captcha and solve_captcha actions to proceed.",
                        extracted_content=f"Login blocked by CAPTCHA: {error_msg}",
                    )

                # For other errors, return informative error
                return ActionResult(
                    error=f"Login failed: {error_msg}",
                    extracted_content=f"Authentication error ({error_type}): {error_msg}",
                )

        return result

    tools.act = guarded_act  # type: ignore
