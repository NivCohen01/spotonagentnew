from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
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


def install_auth_guardrails(tools: Tools, sess: Any) -> None:
    """
    Install runtime guardrails on tools.act:
    - OTP auto-fill for @pathix.io when OTP fields are detected
    - Prevent calling done while still unauthenticated (password/otp fields visible)
    - Block dummy credentials; enforce allowed credentials only
    - Auto-generate pathix credentials during signup intent
    """
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

        # OTP auto-fill before executing any action
        otp_indices: list[int] = []
        if browser_session:
            otp_indices = await _detect_otp_indices(browser_session)
            if otp_indices:
                current_url = None
                with contextlib.suppress(Exception):
                    current_url = await browser_session.get_current_page_url()

                attempted = getattr(sess, "otp_attempted_urls", None)
                if attempted is None:
                    sess.otp_attempted_urls = set()
                    attempted = sess.otp_attempted_urls

                if current_url and current_url in attempted:
                    pass
                else:
                    attempted.add(current_url or "")
                    if not allowed_email:
                        return ActionResult(error="OTP required; cannot continue automatically.")
                    if not allowed_email.lower().endswith(f"@{DEFAULT_DOMAIN}"):
                        return ActionResult(error="OTP required; cannot continue automatically.")

                    password_candidates = _candidate_mailbox_passwords(sess, allowed_email, allowed_password)
                    mailbox_password = password_candidates[0] if password_candidates else None
                    if not mailbox_password:
                        return ActionResult(error="OTP required; IMAP unavailable.")

                    since_ts = getattr(sess, "generated_credentials_created_at", None) or (dt.datetime.utcnow() - dt.timedelta(minutes=10))
                    try:
                        code = await fetch_latest_otp_imap(allowed_email, mailbox_password, since_ts, attempts=18, interval=10)
                    except Exception as exc:
                        log.info("OTP auto-fill aborted: IMAP fetch failed for %s | %r", allowed_email, exc)
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

                    log.info("OTP auto-fill completed for %s on %s", allowed_email, current_url or "<unknown>")
                    _record_auth_event(
                        sess,
                        {
                            "action": "otp",
                            "step": getattr(sess, "current_step", None),
                            "page_url": current_url,
                            "params": {
                                "email": allowed_email,
                                "since_ts": since_ts.isoformat() if isinstance(since_ts, dt.datetime) else None,
                                "digits": len(code),
                            },
                            "ts": int(dt.datetime.utcnow().timestamp() * 1000),
                        },
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
                    if allowed_email and allowed_email.lower().endswith(f"@{DEFAULT_DOMAIN}"):
                        target_email = allowed_email.lower()
                    elif (getattr(sess, "user_credentials", None) or {}).get("email", "").lower().endswith(f"@{DEFAULT_DOMAIN}"):
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

        # Auto-handle OTP if we detect OTP fields and have a pathix.io mailbox we can read.
        if "click" in action_data and browser_session:
            otp_indices = await _detect_otp_indices(browser_session)
            if otp_indices:
                target_email = None
                if allowed_email and allowed_email.lower().endswith(f"@{DEFAULT_DOMAIN}"):
                    target_email = allowed_email.lower()
                elif (getattr(sess, "user_credentials", None) or {}).get("email", "").lower().endswith(f"@{DEFAULT_DOMAIN}"):
                    target_email = (getattr(sess, "user_credentials", None) or {}).get("email", "").lower()

                if target_email and target_email.endswith(f"@{DEFAULT_DOMAIN}"):
                    log.info("OTP fields detected %s; attempting auto-fill using %s", otp_indices, target_email)
                    current_url = ""
                    with contextlib.suppress(Exception):
                        current_url = await browser_session.get_current_page_url()
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
                        log.info("OTP auto-fill exhausted polling for %s", target_email)
                        return ActionResult(error="OTP not received")

                    async def _send_otp_text(idx: int, txt: str):
                        new_input = InputTextAction(index=idx, text=txt, clear=True)
                        input_action = action.__class__(**{"input": new_input})
                        return await orig_act(
                            action=input_action,
                            browser_session=browser_session,
                            page_extraction_llm=page_extraction_llm,
                            sensitive_data=sensitive_data,
                            available_file_paths=available_file_paths,
                            file_system=file_system,
                            action_screenshot_recorder=action_screenshot_recorder,
                            step_number=step_number,
                        )

                    if len(otp_indices) == 1:
                        await _send_otp_text(otp_indices[0], code)
                    else:
                        for pos, digit in enumerate(code):
                            if pos >= len(otp_indices):
                                break
                            await _send_otp_text(otp_indices[pos], digit)
                    log.info("OTP auto-fill completed for %s", target_email)
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
                            },
                            "ts": int(dt.datetime.utcnow().timestamp() * 1000),
                        },
                    )
                else:
                    return ActionResult(error="OTP required; cannot continue automatically.")

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
