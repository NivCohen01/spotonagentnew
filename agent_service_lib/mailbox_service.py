from __future__ import annotations
import asyncio
import contextlib
import datetime as dt
import email
import imaplib
import inspect
import logging
import re
import crypt
from dataclasses import dataclass
from email.message import Message
from typing import Any, Optional
from urllib.parse import urlparse

from sqlalchemy import text

from .service_db import SessionLocal

LOGGER = logging.getLogger("service")

DEFAULT_DOMAIN = "pathix.io"
DEFAULT_IMAP_HOST = "127.0.0.1"
DEFAULT_IMAP_PORT = 993

# 4–8 digit OTP anywhere in subject/body
_OTP_PATTERN = re.compile(r"\b(\d{4,8})\b")


# -----------------------------
# Small utilities
# -----------------------------
def _utc_now() -> dt.datetime:
    return dt.datetime.utcnow()


def _clean_local_base(base: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "", (base or "").lower()) or "pathix"
    return base[:64]


def normalize_domain(value: str) -> str:
    """
    Normalize a URL or hostname into a bare domain string.
    Examples:
      - https://login.monday.com -> monday.com
      - monday.com -> monday.com
    """
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
    host = (parsed.netloc or parsed.path or "").split("@")[-1]
    host = host.split(":")[0].lower()
    host = host.lstrip(".").rstrip(".")
    return host


def derive_base_from_url(url_or_domain: str) -> str:
    """
    Return a clean mailbox base from a URL or domain.
    Examples:
      - monday.com -> monday
      - login.amazon.co.uk -> amazon
    """
    host = normalize_domain(url_or_domain)
    if not host:
        return "pathix"

    parts = [p for p in host.split(".") if p and p != "www"]
    if not parts:
        return "pathix"

    # Minimal multipart-TLD handling without external deps.
    multipart_tlds = {("co", "uk"), ("com", "au"), ("co", "jp"), ("com", "br"), ("co", "in")}
    if len(parts) >= 3 and tuple(parts[-2:]) in multipart_tlds:
        base = parts[-3]
    elif len(parts) >= 2:
        base = parts[-2]
    else:
        base = parts[0]

    base = _clean_local_base(base) or "pathix"
    return base


def compute_password(email_addr: str) -> str:
    """
    Password rule (deterministic, used for automation consistency):
      base@pathix.io   => base12345pathix!
      baseN@pathix.io  => baseN12345pathixN!
    """
    local_part = (email_addr or "").split("@", 1)[0].strip().lower()
    m = re.match(r"^(?P<root>[a-z0-9]+?)(?P<num>\d+)?$", local_part)
    root = (m.group("root") if m else local_part) or "pathix"
    num = m.group("num") if (m and m.group("num")) else ""
    if num:
        return f"{root}{num}12345pathix{num}!"
    return f"{root}12345pathix!"


def _hash_password_for_mailserver(password: str) -> str:
    """Hash password using SHA512-CRYPT with {SHA512-CRYPT} prefix (Dovecot-compatible)."""
    salt = crypt.mksalt(crypt.METHOD_SHA512)
    hashed = crypt.crypt(password, salt)
    return f"{{SHA512-CRYPT}}{hashed}"


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _db_execute(db: Any, stmt: Any, params: Optional[dict] = None) -> Any:
    """
    Execute a SQLAlchemy text() statement on either AsyncSession or Session.
    """
    params = params or {}
    execute_fn = getattr(db, "execute", None)
    if execute_fn is None:
        raise RuntimeError("DB session has no execute()")

    if inspect.iscoroutinefunction(execute_fn):
        return await db.execute(stmt, params)
    # sync session: run in a thread to avoid blocking the event loop
    return await asyncio.to_thread(db.execute, stmt, params)


async def _db_commit(db: Any) -> None:
    commit_fn = getattr(db, "commit", None)
    if commit_fn is None:
        return
    if inspect.iscoroutinefunction(commit_fn):
        await db.commit()
    else:
        await asyncio.to_thread(db.commit)


@contextlib.asynccontextmanager
async def _session():
    """
    Provide a session that works for both AsyncSession and sync SessionLocal.
    """
    sess = SessionLocal()
    # AsyncSessionLocal often supports `async with SessionLocal() as db:`
    if hasattr(sess, "__aenter__") and hasattr(sess, "__aexit__"):
        async with sess as db:
            yield db
        return

    # Sync SessionLocal
    try:
        yield sess
    finally:
        with contextlib.suppress(Exception):
            close_fn = getattr(sess, "close", None)
            if close_fn:
                close_fn()


# -----------------------------
# Mailbox allocation / creation
# -----------------------------

async def get_next_available_email(base: str, domain: str = DEFAULT_DOMAIN) -> str:
    """
    Efficiently find the next available email for the given base using (at most) ONE DB query.

    Rules:
    - Prefer `base@domain` if it's free.
    - Otherwise use max numeric suffix + 1:
        base@domain exists and max(baseN) is 27 -> return base28@domain

    This avoids N queries for N existing mailboxes.
    """
    cleaned_base = _clean_local_base(base)
    domain = (domain or "").strip().lower()
    if not domain:
        raise ValueError("domain is required")

    try:
        async with _session() as db:
            # MySQL-friendly:
            # - base_exists: whether exact base@domain exists
            # - max_suffix: maximum numeric suffix N for baseN@domain
            #   (only matches base + digits + @ + domain)
            res = await _db_execute(
                db,
                text(
                    r"""
                    SELECT
                        SUM(email = CONCAT(:base, '@', :domain)) AS base_exists,
                        MAX(
                            CASE
                                WHEN email REGEXP CONCAT(
                                    '^', :base, '[0-9]+@', REPLACE(:domain, '.', '\\.'), '$'
                                )
                                THEN CAST(
                                    SUBSTRING(
                                        email,
                                        LENGTH(:base) + 1,
                                        LOCATE('@', email) - LENGTH(:base) - 1
                                    ) AS UNSIGNED
                                )
                                ELSE NULL
                            END
                        ) AS max_suffix
                    FROM mailserver.users
                    WHERE email LIKE CONCAT(:base, '%@', :domain)
                    """
                ),
                {"base": cleaned_base, "domain": domain},
            )
            row = res.first()
            base_exists = bool((row[0] if row else 0) or 0)
            max_suffix = row[1] if row else None

        if not base_exists:
            return f"{cleaned_base}@{domain}"

        next_suffix = (int(max_suffix) + 1) if max_suffix is not None else 1
        return f"{cleaned_base}{next_suffix}@{domain}"

    except Exception as exc:
        # Safe fallback (still correct, just less efficient) if DB regex/substr behaves differently.
        LOGGER.warning("Efficient mailbox allocation failed; falling back to loop. base=%s domain=%s err=%r", cleaned_base, domain, exc)

        suffix = 0
        while True:
            local_part = f"{cleaned_base}{suffix or ''}"
            candidate = f"{local_part}@{domain}"
            async with _session() as db:
                r = await _db_execute(
                    db,
                    text("SELECT 1 FROM mailserver.users WHERE email=:email LIMIT 1"),
                    {"email": candidate},
                )
                if r.scalar() is None:
                    return candidate

            suffix += 1
            if suffix > 10_000:
                raise RuntimeError("Could not find available mailbox name after 10k attempts")



@dataclass(frozen=True)
class _ColumnInfo:
    name: str
    is_nullable: bool


async def _get_table_columns(db: Any, schema: str, table_name: str) -> dict[str, _ColumnInfo]:
    """
    Return columns for schema.table_name => {col_name: ColumnInfo(is_nullable=...)}
    """
    res = await _db_execute(
        db,
        text(
            """
            SELECT COLUMN_NAME, IS_NULLABLE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA=:schema AND TABLE_NAME=:table
            """
        ),
        {"schema": schema, "table": table_name},
    )
    out: dict[str, _ColumnInfo] = {}
    for col_name, is_nullable in res.fetchall():
        out[str(col_name)] = _ColumnInfo(name=str(col_name), is_nullable=(str(is_nullable).upper() == "YES"))
    return out


async def _resolve_or_create_domain_id(db: Any, domain: str) -> Optional[int]:
    """
    Resolve domain_id generically without hardcoding “1”.
    Strategy:
    - Find a table in schema mailserver that looks like a domain table (has id + (domain|name))
    - SELECT id WHERE (domain|name)=:domain
    - If missing, INSERT and then SELECT again
    If no suitable table exists, returns None.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return None

    # Find candidate tables with columns: id AND (domain OR name)
    res = await _db_execute(
        db,
        text(
            """
            SELECT t.TABLE_NAME
            FROM information_schema.TABLES t
            WHERE t.TABLE_SCHEMA='mailserver'
            """
        ),
    )
    table_names = [r[0] for r in res.fetchall()]

    candidates: list[tuple[str, str]] = []  # (table_name, value_column)
    for tname in sorted(table_names):
        try:
            cols = await _get_table_columns(db, "mailserver", tname)
        except Exception:
            continue
        if "id" not in cols:
            continue
        if "domain" in cols:
            candidates.append((tname, "domain"))
        elif "name" in cols:
            candidates.append((tname, "name"))

    if not candidates:
        return None

    for table_name, value_col in candidates:
        # 1) Try select
        try:
            sel = await _db_execute(
                db,
                text(f"SELECT id FROM mailserver.{table_name} WHERE {value_col}=:d LIMIT 1"),
                {"d": domain},
            )
            row = sel.first()
            if row and row[0] is not None:
                return int(row[0])
        except Exception:
            continue

        # 2) Try insert (best-effort)
        try:
            await _db_execute(
                db,
                text(f"INSERT INTO mailserver.{table_name} ({value_col}) VALUES (:d)"),
                {"d": domain},
            )
            await _db_commit(db)

            sel2 = await _db_execute(
                db,
                text(f"SELECT id FROM mailserver.{table_name} WHERE {value_col}=:d LIMIT 1"),
                {"d": domain},
            )
            row2 = sel2.first()
            if row2 and row2[0] is not None:
                return int(row2[0])
        except Exception:
            # Keep trying other candidates
            continue

    return None


async def ensure_mailbox_exists(email_addr: str, password: str) -> None:
    """
    Ensure a mailbox row exists in mailserver.users.
    - Creates user row if missing
    - If users.domain_id exists and is NOT NULL: resolves/creates matching domain row and uses its id
    """
    email_addr = (email_addr or "").strip().lower()
    if not email_addr or "@" not in email_addr:
        return

    local_part, _, domain = email_addr.partition("@")
    now = _utc_now()

    async with _session() as db:
        # If already exists, no-op.
        existing = await _db_execute(
            db,
            text("SELECT 1 FROM mailserver.users WHERE email=:email LIMIT 1"),
            {"email": email_addr},
        )
        if existing.scalar():
            return

        columns = await _get_table_columns(db, "mailserver", "users")

        data: dict[str, Any] = {"email": email_addr}

        # Password column naming differs between setups; keep codex’s assumption but make it guarded.
        if "password" in columns:
            data["password"] = _hash_password_for_mailserver(password)

        if "name" in columns:
            data["name"] = local_part

        # Domain-id handling, without hardcoding.
        if "domain_id" in columns:
            domain_id = await _resolve_or_create_domain_id(db, domain)
            if domain_id is None and not columns["domain_id"].is_nullable:
                raise RuntimeError(
                    f"mailserver.users.domain_id is NOT NULL, but domain_id could not be resolved/created for domain='{domain}'. "
                    "Create the domain row in the mailserver domain table, or adjust schema."
                )
            if domain_id is not None:
                data["domain_id"] = domain_id

        # Optional convenience columns (only set if they exist)
        if "maildir" in columns:
            data["maildir"] = f"{domain}/{local_part}/"
        if "home" in columns:
            data["home"] = f"/var/mail/vhosts/{domain}/{local_part}"
        if "local_part" in columns:
            data["local_part"] = local_part
        if "domain" in columns:
            data["domain"] = domain
        if "quota" in columns:
            data["quota"] = 0
        if "created" in columns:
            data["created"] = now
        if "modified" in columns:
            data["modified"] = now
        if "active" in columns:
            data["active"] = 1

        cols = ", ".join(data.keys())
        params = ", ".join(f":{k}" for k in data.keys())

        await _db_execute(db, text(f"INSERT INTO mailserver.users ({cols}) VALUES ({params})"), data)
        await _db_commit(db)
        LOGGER.info("Created mailserver.users entry for %s", email_addr)


# -----------------------------
# OTP extraction (IMAP)
# -----------------------------

def _message_text(msg: Message) -> str:
    """Extract concatenated subject/body text for OTP search."""
    parts: list[str] = []
    subject = msg.get("subject") or ""
    if subject:
        parts.append(str(subject))

    def _add_payload(payload: bytes | None, charset: str, is_html: bool) -> None:
        if not payload:
            return
        text_val = payload.decode(charset or "utf-8", errors="ignore")
        if is_html:
            text_val = re.sub(r"<[^>]+>", " ", text_val)
        parts.append(text_val)

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_disposition() == "attachment":
                continue
            with contextlib.suppress(Exception):
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                _add_payload(payload, charset, part.get_content_subtype() == "html")
    else:
        with contextlib.suppress(Exception):
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            _add_payload(payload, charset, msg.get_content_subtype() == "html")

    return "\n".join(parts)


def _extract_otp_from_message(msg: Message, since_ts: Optional[dt.datetime]) -> Optional[str]:
    # Date filter (best-effort)
    if since_ts:
        with contextlib.suppress(Exception):
            from email.utils import parsedate_to_datetime

            date_hdr = msg.get("Date")
            if date_hdr:
                msg_dt = parsedate_to_datetime(date_hdr)
                if msg_dt:
                    # normalize naive vs aware comparisons
                    if msg_dt.tzinfo is not None and since_ts.tzinfo is None:
                        since_ts = since_ts.replace(tzinfo=dt.timezone.utc)
                    if msg_dt.tzinfo is None and since_ts.tzinfo is not None:
                        msg_dt = msg_dt.replace(tzinfo=dt.timezone.utc)
                    if msg_dt < since_ts:
                        return None

    blob = _message_text(msg)
    m = _OTP_PATTERN.search(blob)
    return m.group(1) if m else None


def _imap_fetch_latest_otp_sync(
    email_addr: str,
    password: str,
    since_ts: Optional[dt.datetime],
    host: str,
    port: int,
) -> Optional[str]:
    """
    Blocking IMAP fetch (run via asyncio.to_thread).
    """
    with imaplib.IMAP4_SSL(host, port) as imap_conn:
        imap_conn.login(email_addr, password)
        imap_conn.select("INBOX")
        typ, data = imap_conn.search(None, "ALL")
        if typ != "OK":
            return None

        ids = data[0].split() if data and data[0] else []
        # newest-first, cap to last 50 messages
        for msg_id in reversed(ids[-50:]):
            typ2, msg_data = imap_conn.fetch(msg_id, "(RFC822)")
            if typ2 != "OK" or not msg_data:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)
            code = _extract_otp_from_message(msg, since_ts)
            if code:
                return code

    return None


async def fetch_latest_otp_imap(
    email_addr: str,
    password: str,
    since_ts: Optional[dt.datetime],
    attempts: int = 12,
    interval: int = 10,
    host: str = DEFAULT_IMAP_HOST,
    port: int = DEFAULT_IMAP_PORT,
) -> Optional[str]:
    """
    Poll IMAP for an OTP (4–8 digit code) without touching filesystem.
    - Uses asyncio.to_thread to avoid blocking the event loop.
    """
    email_addr = (email_addr or "").strip().lower()
    if not email_addr or not password:
        return None

    for attempt in range(attempts):
        LOGGER.info("IMAP OTP check attempt %s/%s for %s", attempt + 1, attempts, email_addr)
        try:
            code = await asyncio.to_thread(
                _imap_fetch_latest_otp_sync,
                email_addr,
                password,
                since_ts,
                host,
                port,
            )
            if code:
                LOGGER.info("OTP found for %s", email_addr)
                return code
        except Exception as exc:
            LOGGER.warning("IMAP OTP fetch failed for %s: %r", email_addr, exc)

        if attempt < attempts - 1:
            await asyncio.sleep(interval)

    LOGGER.info("OTP not received for %s after %s attempts", email_addr, attempts)
    return None


# -----------------------------
# Persist “used credentials” into workspace_domains
# -----------------------------

async def save_workspace_domain_credentials(
    workspace_id: int,
    domain_or_url: str,
    username: str,
    password: str,
    is_primary: bool = False,
) -> None:
    if not domain_or_url:
        raise ValueError("target_url_or_domain must include a valid domain or url")
    if not workspace_id:
        raise ValueError("workspace_id is required")
    if not username:
        raise ValueError("username is required")
    if password is None:
        raise ValueError("password is required")

    now = _utc_now()

    async with _session() as db:
        if is_primary:
            # Only one primary per workspace + target key
            with contextlib.suppress(Exception):
                await _db_execute(
                    db,
                    text(
                        """
                        UPDATE workspace_domains
                        SET is_primary=0
                        WHERE workspace_id=:ws AND domain=:d
                        """
                    ),
                    {"ws": workspace_id, "d": domain_or_url},
                )

        # Upsert by (workspace_id, domain_key, username)
        sel = await _db_execute(
            db,
            text(
                """
                SELECT id
                FROM workspace_domains
                WHERE workspace_id=:ws AND domain=:d AND username=:u
                LIMIT 1
                """
            ),
            {"ws": workspace_id, "d": domain_or_url, "u": username},
        )
        row = sel.first()

        if row and row[0] is not None:
            await _db_execute(
                db,
                text(
                    """
                    UPDATE workspace_domains
                    SET password=:p, is_primary=:ip
                    WHERE id=:id
                    """
                ),
                {"p": password, "ip": 1 if is_primary else 0, "id": int(row[0])},
            )
        else:
            await _db_execute(
                db,
                text(
                    """
                    INSERT INTO workspace_domains (workspace_id, domain, username, password, is_primary, created_at)
                    VALUES (:ws, :d, :u, :p, :ip, :ts)
                    """
                ),
                {
                    "ws": workspace_id,
                    "d": domain_or_url,
                    "u": username,
                    "p": password,  # plain text for now (per your current system)
                    "ip": 1 if is_primary else 0,
                    "ts": now,
                },
            )

        await _db_commit(db)
        LOGGER.info(
            "Saved workspace_domains credentials for workspace=%s target=%s username=%s primary=%s",
            workspace_id,
            domain_or_url,
            username,
            bool(is_primary),
        )