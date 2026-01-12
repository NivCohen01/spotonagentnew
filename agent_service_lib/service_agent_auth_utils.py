from __future__ import annotations

import contextlib
import datetime as dt
import logging
from typing import Any, Optional

from browser_use.agent.views import ActionResult
from browser_use.tools.service import Tools
from pydantic import BaseModel, Field

from .mailbox_service import (
    DEFAULT_DOMAIN,
    compute_password,
    derive_base_from_url,
    ensure_mailbox_exists,
    fetch_latest_otp_imap,
    get_next_available_email,
    normalize_domain,
    save_workspace_domain_credentials,
)

_MICRO_VERBS: tuple[str, ...] = ()
_SUBMIT_TERMS: tuple[str, ...] = ()


class TaskIntent(BaseModel):
    needs_auth: bool
    needs_account_creation: bool
    include_auth_in_final_guide: bool


class TaskCredentials(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None


def _classify_field(node) -> str:
    """Best-effort classification of an input field using standard HTML semantics (no task/keyword heuristics)."""
    try:
        attrs = getattr(node, "attributes", None) or {}
        input_type = (attrs.get("type") or "").lower()
        autocomplete = (attrs.get("autocomplete") or "").lower()
        inputmode = (attrs.get("inputmode") or "").lower()
        pattern = (attrs.get("pattern") or "").lower()
        maxlength = None
        with contextlib.suppress(Exception):
            maxlength = int(attrs.get("maxlength")) if attrs.get("maxlength") is not None else None

        if input_type == "password" or autocomplete in {"current-password", "new-password"}:
            return "password"
        if input_type == "email" or autocomplete in {"username", "email"} or inputmode == "email":
            return "email"
        # OTP-like: one-time-code, numeric short inputs, or pattern digits
        if autocomplete == "one-time-code":
            return "otp"
        otp_pattern_hint = False
        if pattern:
            pat = pattern.lower()
            if any(token in pat for token in ["\\d", "[0-9", "^[0-9", "^[\\d", "{0-9", "d{"]):
                otp_pattern_hint = True
        if inputmode == "numeric" or input_type in {"tel", "text"}:
            if maxlength is not None and maxlength <= 2:
                return "otp"
            if otp_pattern_hint:
                return "otp"
        if otp_pattern_hint:
            return "otp"
    except Exception:
        return "other"
    return "other"


async def _page_has_password_or_otp(browser_session) -> bool:
    selector_map: dict[int, Any] = {}
    with contextlib.suppress(Exception):
        selector_map = await browser_session.get_selector_map()
        if not selector_map:
            await browser_session.get_element_by_index(1)
            selector_map = await browser_session.get_selector_map()
    for node in (selector_map or {}).values():
        if _classify_field(node) in {"password", "otp"}:
            return True
    return False


async def _page_has_email_field(browser_session) -> bool:
    selector_map: dict[int, Any] = {}
    with contextlib.suppress(Exception):
        selector_map = await browser_session.get_selector_map()
        if not selector_map:
            await browser_session.get_element_by_index(1)
            selector_map = await browser_session.get_selector_map()
    for node in (selector_map or {}).values():
        if _classify_field(node) == "email":
            return True
    return False


async def _detect_otp_indices(browser_session) -> list[int]:
    """Detect OTP-like input fields from the current selector map (HTML semantics only)."""
    selector_map: dict[int, Any] = {}
    with contextlib.suppress(Exception):
        selector_map = await browser_session.get_selector_map()
        if not selector_map:
            # Force DOM build once
            await browser_session.get_element_by_index(1)
            selector_map = await browser_session.get_selector_map()
    otp_indices: list[int] = []
    for idx, node in (selector_map or {}).items():
        if _classify_field(node) == "otp":
            otp_indices.append(idx)
    return sorted(otp_indices)


class CreateMailboxParams(BaseModel):
    target_url: Optional[str] = Field(default=None, description="URL for deriving mailbox base.")
    base_hint: Optional[str] = Field(default=None, description="Explicit base to use instead of the derived domain label.")


class FetchOTPParams(BaseModel):
    email: Optional[str] = Field(default=None, description="pathix.io mailbox to read; defaults to the generated mailbox.")
    since_ts: Optional[dt.datetime] = Field(default=None, description="Only accept OTP codes newer than this timestamp.")


def _sanitize_credentials_payload(payload: Any, creds: dict[str, str]) -> Any:
    """Replace generated credentials with placeholders in strings/dicts/lists."""
    if not payload or not creds:
        return payload
    replacements = {}
    email = creds.get("email")
    password = creds.get("password")
    if email:
        replacements[email] = "<your email>"
    if password:
        replacements[password] = "<your password>"

    def _replace(val: Any) -> Any:
        if isinstance(val, str):
            out = val
            for old, new in replacements.items():
                out = out.replace(old, new)
            return out
        if isinstance(val, list):
            return [_replace(v) for v in val]
        if isinstance(val, dict):
            return {k: _replace(v) for k, v in val.items()}
        return val

    return _replace(payload)


def _is_micro_action(desc: str) -> bool:
    # Let the agent decide regrouping without predefined verb lists.
    return False


async def _save_workspace_domain_if_applicable(sess: Any, email_addr: str, password: str) -> None:
    """
    Persist generated signup credentials into workspace_domains when running without provided user creds.
    """
    if not getattr(sess, "workspace_id", None):
        return
    if not email_addr or not password:
        return
    if getattr(sess, "user_credentials", None):
        return

    domain_source = getattr(sess, "start_url", None) or ""
    if not domain_source:
        domain_source = normalize_domain(domain_source)
        if not domain_source:
            return

    try:
        await save_workspace_domain_credentials(
            workspace_id=sess.workspace_id,
            domain_or_url=domain_source,
            username=email_addr,
            password=password,
            is_primary=False,
        )
    except Exception as exc:
        logging.getLogger("service").warning(
            "Failed to save workspace domain creds for workspace_id=%s domain=%s email=%s | %r",
            sess.workspace_id,
            domain_source,
            email_addr,
            exc,
        )


def _candidate_mailbox_passwords(sess: Any, email_addr: str, preferred_password: Optional[str] = None) -> list[str]:
    """
    Prefer explicitly provided credentials for IMAP access, then generated creds, with deterministic fallback last.
    """
    log = logging.getLogger("service")
    email_clean = (email_addr or "").strip().lower()
    candidates: list[str] = []

    for pwd in (
        preferred_password,
        (getattr(sess, "user_credentials", None) or {}).get("password"),
        (getattr(sess, "generated_credentials", None) or {}).get("password"),
    ):
        if pwd:
            candidates.append(pwd)

    if email_clean.endswith(f"@{DEFAULT_DOMAIN}"):
        try:
            candidates.append(compute_password(email_clean))
        except Exception as exc:
            log.info("compute_password failed for %s | %r", email_clean, exc)

    seen: set[str] = set()
    ordered: list[str] = []
    for pwd in candidates:
        if pwd and pwd not in seen:
            ordered.append(pwd)
            seen.add(pwd)
    return ordered


def _register_mailbox_actions(tools: Tools, sess: Any) -> None:
    log = logging.getLogger("service")

    @tools.action(
        "Generate a signup-ready pathix.io mailbox and return email + password for the target site.",
        param_model=CreateMailboxParams,
    )
    async def create_signup_mailbox(params: CreateMailboxParams):
        base_source = params.base_hint or params.target_url or getattr(sess, "start_url", None) or getattr(sess, "task", None) or ""
        base = derive_base_from_url(base_source)

        if getattr(sess, "generated_credentials", None):
            existing_email = sess.generated_credentials.get("email")
            existing_password = sess.generated_credentials.get("password")
            if existing_email and existing_password:
                log.info("Reusing generated mailbox %s for session %s", existing_email, getattr(sess, "id", "<unknown>"))
                return ActionResult(
                    extracted_content=f"Reuse generated credentials: {existing_email} / {existing_password}",
                    long_term_memory=f"Generated mailbox already available: {existing_email}",
                    metadata={"email": existing_email, "password": existing_password},
                )

        email_addr = await get_next_available_email(base)
        password = compute_password(email_addr)
        await ensure_mailbox_exists(email_addr, password)
        sess.generated_credentials = {"email": email_addr, "password": password}
        sess.generated_credentials_created_at = dt.datetime.utcnow()
        await _save_workspace_domain_if_applicable(sess, email_addr, password)
        if not getattr(sess, "signup_intent", False):
            log.info(
                "Mailbox generated outside signup-intent task; agent decided account creation was required | session=%s",
                getattr(sess, "id", "<unknown>"),
            )
        detail = f"Use these credentials for signup: {email_addr} / {password}"
        return ActionResult(
            extracted_content=detail,
            long_term_memory=f"Generated pathix.io mailbox {email_addr}",
            metadata={"email": email_addr, "password": password},
        )

    @tools.action(
        "Fetch OTP from a pathix.io mailbox (polls up to ~2 minutes, 4-8 digit codes).",
        param_model=FetchOTPParams,
    )
    async def fetch_mailbox_otp(params: FetchOTPParams):
        chosen_email = params.email or (getattr(sess, "user_credentials", None) or {}).get("email") or (getattr(sess, "generated_credentials", None) or {}).get("email")
        target_email = (chosen_email or "").strip().lower()
        if not target_email:
            msg = "OTP required; no mailbox available."
            return ActionResult(error=msg, extracted_content=msg)
        if not target_email.endswith(f"@{DEFAULT_DOMAIN}"):
            msg = "OTP required; cannot continue automatically."
            return ActionResult(error=msg, extracted_content=msg)

        password_candidates = _candidate_mailbox_passwords(sess, target_email)
        mailbox_password = password_candidates[0] if password_candidates else None
        if not mailbox_password:
            return ActionResult(error="OTP required; cannot continue automatically.")

        since_ts = params.since_ts or getattr(sess, "generated_credentials_created_at", None) or (dt.datetime.utcnow() - dt.timedelta(minutes=10))
        code = await fetch_latest_otp_imap(target_email, mailbox_password, since_ts, attempts=18, interval=10)
        if code:
            message = f"OTP for {target_email}: {code}"
            log.info("OTP retrieved for %s", target_email)
            return ActionResult(
                extracted_content=message,
                long_term_memory="OTP retrieved from mailbox",
                metadata={"otp": code, "email": target_email},
            )

        msg = "OTP not received"
        log.info("OTP polling exhausted for %s", target_email)
        return ActionResult(error=msg, extracted_content=msg)
