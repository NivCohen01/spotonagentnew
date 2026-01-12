from __future__ import annotations

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
    ):
        allowed_email = allowed_creds_holder.get("email") or (getattr(sess, "generated_credentials", None) or {}).get("email")
        allowed_password = allowed_creds_holder.get("password") or (getattr(sess, "generated_credentials", None) or {}).get("password")
        action_data = action.model_dump(exclude_unset=True)

        # OTP auto-fill before executing any action
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
                        )

                    if len(otp_indices) == 1:
                        await _send_otp(otp_indices[0], code)
                    else:
                        for pos, digit in enumerate(code):
                            if pos >= len(otp_indices):
                                break
                            await _send_otp(otp_indices[pos], digit)

                    log.info("OTP auto-fill completed for %s on %s", allowed_email, current_url or "<unknown>")

        # Block done if still unauthenticated (signup/auth flows)
        if "done" in action_data and (getattr(sess, "signup_intent", False) or (getattr(sess, "intent", None) and getattr(sess.intent, "needs_auth", False))) and browser_session:
            otp_indices = await _detect_otp_indices(browser_session)
            if otp_indices or await _page_has_password_or_otp(browser_session):
                return ActionResult(error="Not authenticated yet; continue.")

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
                        )

                    if len(otp_indices) == 1:
                        await _send_otp_text(otp_indices[0], code)
                    else:
                        for pos, digit in enumerate(code):
                            if pos >= len(otp_indices):
                                break
                            await _send_otp_text(otp_indices[pos], digit)
                    log.info("OTP auto-fill completed for %s", target_email)
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

                if field_kind == "email":
                    if (
                        "<" in text_val
                        or ">" in text_val
                        or text_val.endswith(("example.com", "test.com"))
                        or "@@" in text_val
                        or "email@" in text_val.lower()
                    ):
                        return ActionResult(error="Credentials required; cannot proceed.")

                    if allowed_email:
                        action = _rebuild_action_with_text(allowed_email)
                    else:
                        if getattr(sess, "signup_intent", False):
                            creds = await ensure_signup_mailbox(sess, log=log)
                            allowed_creds_holder["email"] = creds.get("email")
                            action = _rebuild_action_with_text(creds.get("email"))
                        else:
                            return ActionResult(error="Credentials required; cannot continue.")

                elif field_kind == "password":
                    if "<" in text_val or ">" in text_val:
                        return ActionResult(error="Credentials required; cannot proceed.")

                    if allowed_password:
                        action = _rebuild_action_with_text(allowed_password)
                    else:
                        if getattr(sess, "signup_intent", False):
                            creds = await ensure_signup_mailbox(sess, log=log)
                            allowed_creds_holder["password"] = creds.get("password")
                            action = _rebuild_action_with_text(creds.get("password"))
                        elif dummy_password:
                            return ActionResult(error="Credentials required; cannot proceed.")
                        else:
                            return ActionResult(error="Credentials required; cannot continue.")
                else:
                    # For otp/other fields, allow without credential enforcement.
                    pass

        return await orig_act(
            action=action,
            browser_session=browser_session,
            page_extraction_llm=page_extraction_llm,
            sensitive_data=sensitive_data,
            available_file_paths=available_file_paths,
            file_system=file_system,
        )

    tools.act = guarded_act  # type: ignore
