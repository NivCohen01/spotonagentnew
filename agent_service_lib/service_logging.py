from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import json
import logging
import re
from typing import Any, Callable

from sqlalchemy import text

from .service_config import ANSI_RE, COLLAPSE_INTERNAL_SPACES, CTRL_ZW_RE, STRIP_ANSI

run_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
_STD_LOG_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


def _clean_message(msg: str) -> str:
    if STRIP_ANSI:
        msg = ANSI_RE.sub("", msg)
    msg = CTRL_ZW_RE.sub("", msg)
    if COLLAPSE_INTERNAL_SPACES:
        msg = re.sub(r"[ \t\u00A0]{2,}", " ", msg)
    return msg.strip()


def _hijack_library_loggers():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("OpenAI").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        if not getattr(record, "run_id", None):
            setattr(record, "run_id", run_id_ctx.get())
        return True


class PerRunDBAndMemoryHandler(logging.Handler):
    def __init__(self, session_lookup: Callable[[str], Any], session_factory):
        super().__init__()
        self._session_lookup = session_lookup
        self._session_factory = session_factory

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging
        try:
            run_id = getattr(record, "run_id", None) or run_id_ctx.get()
            if not run_id:
                return

            try:
                text_line = _clean_message(record.getMessage())
            except Exception:
                text_line = _clean_message(str(record.msg))

            if not text_line:
                return

            extras = {k: v for k, v in record.__dict__.items() if k not in _STD_LOG_FIELDS}
            payload = None
            if extras:
                try:
                    payload = json.loads(json.dumps(extras, default=str))
                except Exception:
                    payload = {k: str(v) for k, v in extras.items()}

            sess = self._session_lookup(run_id)
            if not sess:
                return
            sess.log_seq += 1
            seq = sess.log_seq
            sess.log_lines.append(text_line)

            if self._session_factory:
                async def _save():
                    ts = dt.datetime.utcnow()
                    async with self._session_factory() as db:
                        await db.execute(
                            text(
                                """
                                INSERT INTO run_logs(run_id, ts, seq, line, level, logger, payload)
                                VALUES (:run_id,:ts,:seq,:line,:level,:logger,:payload)
                                """
                            ),
                            {
                                "run_id": run_id,
                                "ts": ts,
                                "seq": seq,
                                "line": text_line,
                                "level": record.levelname,
                                "logger": record.name,
                                "payload": json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                            },
                        )
                        await db.commit()
                asyncio.create_task(_save())
        except Exception:
            pass
