# agent_service.py
from __future__ import annotations
"""
Agent Service — FastAPI + FIFO queue + Browser Use agent + async MySQL

Key changes in this version:
- Creates an EMPTY placeholder guide as soon as a run starts (linked via guides.run_id = runs.id).
- On successful finish, updates that guide with the final title/description/slug and sets status back to 'draft'.
- Includes migrations to add guides.run_id (+ index + FK) and to allow a 'generating' status in guides.status.

Environment (examples):
  PORT=9000
  BIND=0.0.0.0
  ALLOWED_ORIGINS=*
  API_KEY=                           (optional)
  OPENAI_MODEL=gpt-4.1-mini
  DB_URL=mysql+aiomysql://user:pass@127.0.0.1:3306/spoton
  DEFAULT_AUTHOR_ID=9                (optional; for guides.created_by/updated_by)
  MAX_CONCURRENCY=4
  BROWSER_USE_CHROME_PATH=/usr/bin/chromium
  STRIP_ANSI_IN_DB=1
  COLLAPSE_INTERNAL_SPACES=0
  SCREENSHOTS_BASE=./screenshots
"""

import asyncio
import contextlib
import contextvars
import datetime as dt
import inspect
import json
import logging
import os
import re
import socket
import time
import uuid
from collections import deque
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator

# --- browser-use (agent) ---
from browser_use import Agent, Browser
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.agent.task_optimizer import AgentTaskOptimizer, TaskOptimizationRequest
from browser_use.screenshots.models import ActionScreenshotSettings

# --- DB (async MySQL via SQLAlchemy) ---
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "Agent Service"
APP_VERSION = "3.2.0"

PORT = int(os.getenv("PORT", "9000"))
BIND = os.getenv("BIND", "0.0.0.0")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
API_KEY = os.getenv("API_KEY", "").strip()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_CONCURRENCY = max(1, int(os.getenv("MAX_CONCURRENCY", "4")))

STRIP_ANSI = bool(int(os.getenv("STRIP_ANSI_IN_DB", "1")))
COLLAPSE_INTERNAL_SPACES = bool(int(os.getenv("COLLAPSE_INTERNAL_SPACES", "0")))
DEFAULT_AUTHOR_ID = os.getenv("DEFAULT_AUTHOR_ID", "").strip()
DEFAULT_AUTHOR_ID = int(DEFAULT_AUTHOR_ID) if DEFAULT_AUTHOR_ID.isdigit() else None

DB_URL = os.getenv("DB_URL", "").strip()
engine = create_async_engine(DB_URL, pool_pre_ping=True) if DB_URL else None
SessionLocal = async_sessionmaker(engine, expire_on_commit=False) if engine else None

SCREENSHOTS_BASE = Path(os.getenv("SCREENSHOTS_BASE", "./screenshots")).resolve()
SCREENSHOTS_BASE.mkdir(parents=True, exist_ok=True)

# Logging cleanup regex
_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_CTRL_ZW_RE = re.compile("[" + "\u200B\u200C\u200D\u200E\u200F" + "\u2060" + "\uFEFF" + "]")
_STEP_IMG_RE = re.compile(r"step_(\d{1,3})_.*\.(png|jpg|jpeg)$", re.IGNORECASE)

# =============================================================================
# Chrome discovery
# =============================================================================

def find_chrome_binary() -> str:
    env = os.getenv("BROWSER_USE_CHROME_PATH")
    if env and Path(env).exists():
        return env
    for c in (
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ):
        if Path(c).exists():
            return c
    raise RuntimeError("Chrome/Chromium not found. Set BROWSER_USE_CHROME_PATH or install Chromium.")

CHROME_BIN = find_chrome_binary()
BASE_PROFILE_DIR = Path("/tmp/agent-profiles")
BASE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FastAPI app
# =============================================================================

_allow_credentials = ALLOWED_ORIGINS != ["*"]
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Models
# =============================================================================

class GuideOutput(BaseModel):
    title: str
    steps: list[str]
    links: list[str] = []
    notes: Optional[str] = None
    success: bool

VisionLevel = Literal["auto", "low", "high"]
SessionState = Literal["queued", "starting", "running", "done", "error", "stopped"]
DeviceType = Literal["mobile", "desktop", "custom"]


class ActionScreenshotOptions(BaseModel):
    """Client-facing configuration for annotated screenshots."""

    enabled: bool = False
    annotate: bool = True
    spotlight: bool = False
    include_in_available_files: bool = False
    session_subdirectories: bool = False


class StartReq(BaseModel):
    task: str
    start_url: Optional[str] = None
    headless: bool = True
    workspace_id: Optional[int] = None
    ws_id: Optional[int] = Field(None, alias="ws_id")  # alias accepted

    use_vision: bool = True
    max_failures: int = 3
    extend_system_message: Optional[str] = None
    generate_gif: bool = False
    max_actions_per_step: int = 10
    use_thinking: bool = False
    flash_mode: bool = False
    calculate_cost: bool = True
    vision_detail_level: VisionLevel = "auto"
    step_timeout: int = 120
    directly_open_url: bool = True
    optimize_task: bool = True

    action_screenshots_enabled: bool = False  # legacy fields (prefer action_screenshots payload)
    action_screenshots_annotate: bool = True
    screenshot_spotlight: bool = False
    action_screenshots: ActionScreenshotOptions | None = None

    device_type: Optional[DeviceType] = "desktop"
    viewport_width: Optional[int] = 1920
    viewport_height: Optional[int] = 1080

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "task": "Compare 3 official passkey guides and summarize.",
            "start_url": "https://www.google.com",
            "workspace_id": 42,
            "headless": True,
            "optimize_task": True,
            "action_screenshots": {"enabled": True, "spotlight": False}
        }
    })

    @model_validator(mode="after")
    def _sync_workspace_id(self):
        if self.workspace_id is None and self.ws_id is not None:
            self.workspace_id = self.ws_id
        if self.action_screenshots is None:
            self.action_screenshots = ActionScreenshotOptions(
                enabled=self.action_screenshots_enabled,
                annotate=self.action_screenshots_annotate,
                spotlight=self.screenshot_spotlight,
            )
        return self

    @field_validator("max_failures")
    @classmethod
    def _vf_max_failures(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_failures must be >= 1")
        return v

    @field_validator("step_timeout")
    @classmethod
    def _vf_step_timeout(cls, v: int) -> int:
        if v < 5:
            raise ValueError("step_timeout must be >= 5")
        return v

    @field_validator("max_actions_per_step")
    @classmethod
    def _vf_actions(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_actions_per_step must be >= 1")
        return v

class Status(BaseModel):
    session_id: str
    state: SessionState
    task: str
    start_url: Optional[str] = None
    final_response: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    screenshots_dir: Optional[str] = None
    queue_position: Optional[int] = None

class ResultPayload(BaseModel):
    state: SessionState
    final_response: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    screenshots_dir: Optional[str] = None

class OptimizeReq(BaseModel):
    task: str
    mode: Literal["regular", "aggressive", "short", "verbose"] = "regular"
    ptype: Literal['optimize', 'feature_compiler'] = 'optimize'
    llm_model: Optional[str] = None

class OptimizeResp(BaseModel):
    original: str
    optimized: str

# =============================================================================
# Optional API key
# =============================================================================

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# =============================================================================
# Utility: logging & cleaning
# =============================================================================

def _clean_message(msg: str) -> str:
    if not msg:
        return msg
    msg = msg.replace("\r", "").replace("\t", " ")
    if STRIP_ANSI:
        msg = _ANSI_RE.sub("", msg)
    msg = _CTRL_ZW_RE.sub("", msg).strip()
    if COLLAPSE_INTERNAL_SPACES:
        msg = re.sub(r" {2,}", " ", msg)
    return msg

def _hijack_library_loggers():
    names = [
        "Agent", "tools", "service", "BrowserSession",
        "cdp_use.client", "browser_use", "browser_use.agent",
        "browser_use.browser", "browser_use.tools",
    ]
    for name in names:
        lg = logging.getLogger(name)
        lg.propagate = True
        lg.setLevel(logging.INFO)
        for h in list(lg.handlers):
            lg.removeHandler(h)

def _to_public_image_path(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(SCREENSHOTS_BASE.resolve())
        return "/" + rel.as_posix()
    except Exception:
        return "/" + p.name

def _index_screens_by_step(screens_dir: Path) -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}
    if not screens_dir.exists():
        return mapping
    for p in sorted([q for q in screens_dir.glob("*") if q.is_file()], key=lambda q: q.name):
        m = _STEP_IMG_RE.search(p.name)
        if not m:
            continue
        step_no = int(m.group(1))
        mapping.setdefault(step_no, []).append(_to_public_image_path(p))
    return mapping

def _attach_screenshots_to_steps(enriched: dict, screens_dir: Path) -> dict:
    if not screens_dir:
        return enriched
    by_step = _index_screens_by_step(screens_dir)
    steps = enriched.get("steps") or []
    if not isinstance(steps, list):
        return enriched
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        n = step.get("number") or i
        imgs = by_step.get(int(n), [])
        if imgs:
            old = step.get("images")
            if isinstance(old, list) and old:
                step["images"] = list(dict.fromkeys([*old, *imgs]))
            else:
                step["images"] = imgs
    return enriched

def _maybe_to_dict(obj: Any) -> Any:
    for m in ("model_dump", "dict", "to_dict"):
        f = getattr(obj, m, None)
        if callable(f):
            with contextlib.suppress(Exception):
                return f()
    if is_dataclass(obj):
        with contextlib.suppress(Exception):
            return asdict(obj)
    return obj

def _ensure_json_text(value: Any) -> str:
    if value is None:
        return "null"
    value = _maybe_to_dict(value)
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except Exception:
            return json.dumps(value, ensure_ascii=False)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)

def _extract_json_snippet_from_text(text_: str) -> Optional[dict]:
    candidate = text_.replace("\\\\", "\\").replace('\\"', '"')
    for m in re.finditer(r'\{[^{}]*"title"[^{}]*"steps"[^{}]*\}', candidate, flags=re.DOTALL):
        blob = m.group(0)
        with contextlib.suppress(Exception):
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
    candidate2 = candidate.replace("'", '"')
    for m in re.finditer(r'\{[^{}]*"title"[^{}]*"steps"[^{}]*\}', candidate2, flags=re.DOTALL):
        blob = m.group(0)
        with contextlib.suppress(Exception):
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
    return None

def _coerce_guide_output_dict(obj: Any) -> Optional[dict]:
    required = {"title", "steps", "links", "success"}

    if isinstance(obj, BaseModel):
        data = obj.model_dump()
        if required.issubset(data.keys()):
            return {k: (list(data[k]) if k in ("steps", "links") else data[k])
                    for k in ["title", "steps", "links", "notes", "success"] if k in data}

    obj = _maybe_to_dict(obj)

    if isinstance(obj, dict):
        if required.issubset(obj.keys()):
            return {k: (list(obj[k]) if k in ("steps", "links") else obj.get(k))
                    for k in ["title", "steps", "links", "notes", "success"]}
        history = obj.get("history")
        if isinstance(history, list):
            for step in reversed(history):
                with contextlib.suppress(Exception):
                    results = step.get("result") or []
                    for r in results:
                        if r.get("is_done") is True and r.get("extracted_content"):
                            inner = json.loads(r["extracted_content"])
                            if isinstance(inner, dict) and required.issubset(inner.keys()):
                                return {k: (list(inner[k]) if k in ("steps", "links") else inner.get(k))
                                        for k in ["title", "steps", "links", "notes", "success"]}
        amo = obj.get("all_model_outputs")
        if isinstance(amo, list):
            for entry in reversed(amo):
                ed = entry if isinstance(entry, dict) else _maybe_to_dict(entry)
                done = ed.get("done") if isinstance(ed, dict) else None
                if done is None:
                    done_attr = getattr(entry, "done", None)
                    if isinstance(done_attr, BaseModel):
                        done = done_attr.model_dump()
                    elif isinstance(done_attr, dict):
                        done = done_attr
                if not isinstance(done, dict):
                    continue
                data = done.get("data") or done.get("output") or done.get("result")
                if isinstance(data, BaseModel):
                    data = data.model_dump()
                if isinstance(data, dict) and required.issubset(data.keys()):
                    return {k: (list(data[k]) if k in ("steps", "links") else data.get(k))
                            for k in ["title", "steps", "links", "notes", "success"]}

    if isinstance(obj, str):
        with contextlib.suppress(Exception):
            return _coerce_guide_output_dict(json.loads(obj))
        inner = _extract_json_snippet_from_text(obj)
        if inner and required.issubset(inner.keys()):
            return {k: (list(inner[k]) if k in ("steps", "links") else inner.get(k))
                    for k in ["title", "steps", "links", "notes", "success"]}

    amo_attr = getattr(obj, "all_model_outputs", None)
    if isinstance(amo_attr, list):
        for entry in reversed(amo_attr):
            ed = entry if isinstance(entry, dict) else _maybe_to_dict(entry)
            done = ed.get("done") if isinstance(ed, dict) else None
            if done is None:
                da = getattr(entry, "done", None)
                if isinstance(da, BaseModel):
                    done = da.model_dump()
                elif isinstance(da, dict):
                    done = da
            if not isinstance(done, dict):
                continue
            data = done.get("data") or done.get("output") or done.get("result")
            if isinstance(data, BaseModel):
                data = data.model_dump()
            if isinstance(data, dict) and required.issubset(data.keys()):
                return {k: (list(data[k]) if k in ("steps", "links") else data.get(k))
                        for k in ["title", "steps", "links", "notes", "success"]}

    return None

def _shape_steps_with_placeholders(extracted: dict) -> dict:
    raw = list(extracted.get("steps") or [])
    shaped = []
    for i, s in enumerate(raw, 1):
        if isinstance(s, dict):
            description = s.get("description") or s.get("text") or s.get("title") or ""
            page_url   = s.get("pageUrl") or s.get("page_url") or None
            images     = s.get("images") if isinstance(s.get("images"), list) else []
            shaped.append({"number": s.get("number", i), "description": description,
                           "pageUrl": page_url, "images": images})
        else:
            shaped.append({"number": i, "description": str(s), "pageUrl": None, "images": []})
    enriched = dict(extracted)
    enriched["steps"] = shaped
    return enriched

def slugify_title_for_guide(title: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    base = re.sub(r"-{2,}", "-", base)
    rnd  = uuid.uuid4().hex[:4]
    return f"how-to-{base}-{rnd}" if base else f"how-to-guide-{rnd}"

# =============================================================================
# In-memory session + logging handler
# =============================================================================

class Session:
    def __init__(self, session_id: str, req: StartReq):
        self.id = session_id
        self.task = req.task
        self.start_url = req.start_url
        self.headless = req.headless
        self.workspace_id = req.workspace_id

        self.use_vision = req.use_vision
        self.max_failures = req.max_failures
        self.extend_system_message = req.extend_system_message
        self.generate_gif = req.generate_gif
        self.max_actions_per_step = req.max_actions_per_step
        self.use_thinking = req.use_thinking
        self.flash_mode = req.flash_mode
        self.calculate_cost = req.calculate_cost
        self.vision_detail_level = req.vision_detail_level
        self.step_timeout = req.step_timeout
        self.directly_open_url = req.directly_open_url
        self.optimize_task = req.optimize_task

        opts = req.action_screenshots or ActionScreenshotOptions()
        self.action_screenshot_options = opts
        self.action_screenshots_enabled = opts.enabled
        self.action_screenshots_annotate = opts.annotate
        self.screenshot_spotlight = opts.spotlight
        self.action_screenshots_include_files = opts.include_in_available_files
        self.action_screenshots_session_dirs = opts.session_subdirectories

        self.device_type = req.device_type
        self.viewport_width = req.viewport_width
        self.viewport_height = req.viewport_height

        self.state: SessionState = "queued"
        self.final_response: Optional[str] = None
        self.result_only: Optional[str] = None
        self.error: Optional[str] = None

        self.screenshots_dir: Path = (SCREENSHOTS_BASE / self.id / "images")
        self.chrome_proc: Optional[asyncio.subprocess.Process] = None
        self.cdp_port: Optional[int] = None
        self.cdp_ws_url: Optional[str] = None

        self.log_lines: list[str] = []
        self.log_seq: int = 0
        self.finished_at: Optional[dt.datetime] = None

        # Guide linkage: created at start, filled at finish
        self.guide_id: Optional[int] = None
        self.guide_slug: Optional[str] = None

    def status(self, queue_position: Optional[int] = None) -> Status:
        return Status(
            session_id=self.id,
            state=self.state,
            task=self.task,
            start_url=self.start_url,
            final_response=self.final_response,
            result=self.result_only,
            error=self.error,
            screenshots_dir=str(self.screenshots_dir) if self.action_screenshots_enabled else None,
            queue_position=queue_position
        )

sessions: Dict[str, Session] = {}

# Per-run ContextVar for logging attribution
run_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
_STD_LOG_FIELDS = {
    'name','msg','args','levelname','levelno','pathname','filename','module',
    'exc_info','exc_text','stack_info','lineno','funcName','created','msecs',
    'relativeCreated','thread','threadName','processName','process'
}

class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "run_id", None):
            setattr(record, "run_id", run_id_ctx.get())
        return True

class PerRunDBAndMemoryHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
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

            sess = sessions.get(run_id)
            if not sess:
                return
            sess.log_seq += 1
            seq = sess.log_seq
            sess.log_lines.append(text_line)

            if SessionLocal:
                async def _save():
                    ts = dt.datetime.utcnow()
                    async with SessionLocal() as db:
                        await db.execute(
                            text("""INSERT INTO run_logs(run_id, ts, seq, line, level, logger, payload)
                                    VALUES (:run_id,:ts,:seq,:line,:level,:logger,:payload)"""),
                            {
                                "run_id": run_id,
                                "ts": ts,
                                "seq": seq,
                                "line": text_line,
                                "level": record.levelname,
                                "logger": record.name,
                                "payload": json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                            }
                        )
                        await db.commit()
                asyncio.create_task(_save())
        except Exception:
            pass

# =============================================================================
# DB schema & helpers
# =============================================================================

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  id             VARCHAR(16) PRIMARY KEY,
  ws_id          BIGINT NULL,
  task           TEXT NOT NULL,
  start_url      TEXT NULL,
  headless       TINYINT(1) NOT NULL,
  state          VARCHAR(16) NOT NULL,
  started_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at    DATETIME NULL,
  final_response LONGTEXT NULL,
  result         LONGTEXT NULL,
  error          LONGTEXT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS run_logs (
  id      BIGINT AUTO_INCREMENT PRIMARY KEY,
  run_id  VARCHAR(16) NOT NULL,
  ts      DATETIME NOT NULL,
  seq     INT NOT NULL,
  line    TEXT NOT NULL,
  level   VARCHAR(16) NULL,
  logger  VARCHAR(32) NULL,
  payload JSON NULL,
  INDEX idx_run_seq (run_id, seq),
  INDEX idx_run_ts (run_id, ts),
  CONSTRAINT fk_run_logs_runs
    FOREIGN KEY (run_id) REFERENCES runs(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

MIGRATIONS_SQL = [
    # runs tweaks
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS ws_id BIGINT NULL AFTER id",
    "CREATE INDEX IF NOT EXISTS idx_runs_ws_id ON runs (ws_id)",
    "ALTER TABLE runs MODIFY COLUMN started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
    "ALTER TABLE run_logs ADD COLUMN IF NOT EXISTS level  VARCHAR(16) NULL",
    "ALTER TABLE run_logs ADD COLUMN IF NOT EXISTS logger VARCHAR(32) NULL",
    "ALTER TABLE run_logs ADD COLUMN IF NOT EXISTS payload JSON NULL",
    "CREATE INDEX IF NOT EXISTS idx_run_ts ON run_logs (run_id, ts)",

    # guides linkage (best-effort; safe to re-run under suppress)
    "ALTER TABLE guides ADD COLUMN IF NOT EXISTS run_id VARCHAR(16) NULL AFTER workspace_id",
    "CREATE INDEX IF NOT EXISTS idx_guides_run_id ON guides (run_id)",
    # MySQL doesn't support IF NOT EXISTS for constraints universally; safe under suppress:
    "ALTER TABLE guides ADD CONSTRAINT fk_guides_runs FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL",

    # Optional: enable 'generating' status value (keep default 'draft')
    "ALTER TABLE guides MODIFY COLUMN status ENUM('generating','draft','published','archived') NOT NULL DEFAULT 'draft'",
]

async def db_insert_run(sess: Session):
    if not SessionLocal:
        return
    async with SessionLocal() as db:
        await db.execute(
            text("""INSERT INTO runs (id, ws_id, task, start_url, headless, state)
                    VALUES (:id, :ws_id, :task, :start_url, :headless, :state)"""),
            {
                "id": sess.id,
                "ws_id": sess.workspace_id,
                "task": sess.task,
                "start_url": sess.start_url,
                "headless": 1 if sess.headless else 0,
                "state": "starting",
            },
        )
        await db.commit()

async def db_finish_run(sess: Session):
    if not SessionLocal:
        return
    sess.finished_at = dt.datetime.utcnow()
    async with SessionLocal() as db:
        await db.execute(
            text("""UPDATE runs
                    SET state=:state, finished_at=:finished_at,
                        final_response=:final_response, result=:result, error=:error
                    WHERE id=:id"""),
            {
                "id": sess.id,
                "state": sess.state,
                "finished_at": sess.finished_at,
                "final_response": sess.final_response,
                "result": sess.result_only,
                "error": sess.error,
            },
        )
        await db.commit()

async def db_create_empty_guide(sess: Session) -> None:
    """
    Create an empty/placeholder guide row right after we create the run.
    Links it via guides.run_id = sess.id so we can update later.
    """
    if not SessionLocal or sess.workspace_id is None:
        return

    now = dt.datetime.utcnow()
    placeholder_slug = f"run-{sess.id}"  # minimal, guaranteed unique by run id
    title = (sess.task or "Generating guide").strip()
    if len(title) > 255:
        title = title[:255]

    data: Dict[str, Any] = {
        "workspace_id": sess.workspace_id,
        "run_id": sess.id,
        "slug": placeholder_slug,
        "title": title or "Generating guide",
        "status": "generating",  # if enum not updated, migration suppress will ignore but insert will still work in most setups
        "visibility": "private",
        "created_at": now,
        "updated_at": now,
    }
    if DEFAULT_AUTHOR_ID is not None:
        data["created_by"] = DEFAULT_AUTHOR_ID
        data["updated_by"] = DEFAULT_AUTHOR_ID

    cols = ", ".join(data.keys())
    params = ", ".join(f":{k}" for k in data.keys())
    sql = f"INSERT INTO guides ({cols}) VALUES ({params})"

    async with SessionLocal() as db:
        await db.execute(text(sql), data)
        gid_row = await db.execute(text("SELECT LAST_INSERT_ID()"))
        sess.guide_id = gid_row.scalar()
        sess.guide_slug = placeholder_slug
        await db.commit()

async def db_update_guide_from_result(sess: Session, enriched: dict):
    """
    Fill the placeholder guide (created at start) with final title/description/slug/status.
    If a placeholder doesn't exist (legacy), insert a new row (backward compatible).
    """
    if not SessionLocal or not enriched:
        return

    title = str(enriched.get("title") or "Guide")
    new_slug = slugify_title_for_guide(title)
    description_json = json.dumps(enriched, ensure_ascii=False)
    now = dt.datetime.utcnow()

    update_sql = text("""
        UPDATE guides
        SET title=:title,
            slug=:slug,
            description=:description,
            status=:status,
            visibility=:visibility,
            updated_at=:updated_at,
            updated_by=:updated_by
        WHERE run_id=:run_id
        LIMIT 1
    """)

    params_update = {
        "title": title,
        "slug": new_slug,
        "description": description_json,
        "status": "draft",          # keep as 'draft'; publishing is handled by your app
        "visibility": "private",
        "updated_at": now,
        "updated_by": DEFAULT_AUTHOR_ID,
        "run_id": sess.id,
    }

    async with SessionLocal() as db:
        res = await db.execute(update_sql, params_update)
        if res.rowcount and res.rowcount > 0:
            await db.commit()
            return

        # Fallback: no placeholder exists, insert like legacy flow
        if sess.workspace_id is None:
            return

        insert_sql = text("""
            INSERT INTO guides
              (workspace_id, run_id, slug, title, description, status, visibility,
               created_by, updated_by, created_at, updated_at)
            VALUES
              (:workspace_id, :run_id, :slug, :title, :description, :status, :visibility,
               :created_by, :updated_by, :created_at, :updated_at)
        """)
        await db.execute(insert_sql, {
            "workspace_id": sess.workspace_id,
            "run_id": sess.id,
            "slug": new_slug,
            "title": title,
            "description": description_json,
            "status": "draft",
            "visibility": "private",
            "created_by": DEFAULT_AUTHOR_ID,
            "updated_by": DEFAULT_AUTHOR_ID,
            "created_at": now,
            "updated_at": now,
        })
        await db.commit()

# =============================================================================
# Queue Manager (strict FIFO with deque)
# =============================================================================

class QueueManager:
    """
    FIFO queue backed by a deque + N worker tasks.
    - start(): spawn N workers
    - stop():  cancel workers
    - enqueue(sid): append to queue; returns 1-based position
    - position(sid): get 1-based position or None
    """
    def __init__(self, workers_count: int):
        self.workers_count = workers_count
        self._queue: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._workers: list[asyncio.Task] = []
        self._stopping = False

    async def start(self):
        for i in range(self.workers_count):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def stop(self):
        self._stopping = True
        async with self._not_empty:
            self._not_empty.notify_all()
        for t in self._workers:
            t.cancel()
        for t in self._workers:
            with contextlib.suppress(Exception):
                await t

    async def enqueue(self, sid: str) -> int:
        async with self._not_empty:
            self._queue.append(sid)
            pos = len(self._queue)
            self._not_empty.notify()
            return pos

    async def remove_if_present(self, sid: str) -> bool:
        async with self._lock:
            try:
                self._queue.remove(sid)
                return True
            except ValueError:
                return False

    async def position(self, sid: str) -> Optional[int]:
        async with self._lock:
            try:
                return list(self._queue).index(sid) + 1
            except ValueError:
                return None

    async def _pop(self) -> Optional[str]:
        async with self._not_empty:
            while not self._queue and not self._stopping:
                await self._not_empty.wait()
            if self._stopping:
                return None
            return self._queue.popleft()

    async def _worker_loop(self, worker_id: int):
        log = logging.getLogger("service")
        while True:
            sid = await self._pop()
            if sid is None:
                return
            sess = sessions.get(sid)
            if not sess:
                continue
            try:
                await run_session(sess)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("Worker %s crashed running %s: %r", worker_id, sid, e)

queue_mgr = QueueManager(MAX_CONCURRENCY)

# =============================================================================
# Agent execution
# =============================================================================

def pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

async def wait_for_cdp_json_version(port: int, timeout_s: float = 25.0) -> dict:
    url = f"http://127.0.0.1:{port}/json/version"
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout_s:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"CDP /json/version not ready on port {port}")

async def launch_chrome(sess: Session) -> str:
    port = pick_free_port()
    profile = BASE_PROFILE_DIR / f"profile-{sess.id}"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        CHROME_BIN,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-dev-shm-usage",
        "--disable-gpu",
    ]
    if sess.headless:
        cmd.append("--headless=new")
    sess.cdp_port = port
    sess.chrome_proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    ver = await wait_for_cdp_json_version(port, timeout_s=25.0)
    ws = ver.get("webSocketDebuggerUrl")
    if not ws:
        raise RuntimeError(f"webSocketDebuggerUrl missing: {json.dumps(ver)}")
    sess.cdp_ws_url = ws
    return ws

async def _safe_browser_stop(browser: Any) -> None:
    try:
        for name in ("stop", "kill", "close"):
            method = getattr(browser, name, None)
            if not callable(method):
                continue
            res = method()
            if inspect.isawaitable(res):
                await res
            return
    except Exception:
        pass

async def run_session(sess: Session):
    token = run_id_ctx.set(sess.id)
    log = logging.getLogger("service")
    browser: Optional[Browser] = None

    # queued -> starting
    sess.state = "starting"

    # Persist start
    with contextlib.suppress(Exception):
        await db_insert_run(sess)

    # Create placeholder guide linked to this run (best-effort)
    with contextlib.suppress(Exception):
        await db_create_empty_guide(sess)

    try:
        ws = await launch_chrome(sess)
        browser = Browser(cdp_url=ws)

        if not (sess.task and sess.task.strip()):
            raise ValueError("Empty task provided")
        if sess.start_url is not None and not sess.start_url.strip():
            raise ValueError("Empty start_url provided")

        optimized_task = f"{sess.task} Start url: {sess.start_url}" if sess.start_url else sess.task
        if sess.optimize_task:
            try:
                optimizer = AgentTaskOptimizer(llm=ChatOpenAI(model=OPENAI_MODEL))
                optimized_resp = await optimizer.optimize_async(
                    TaskOptimizationRequest(task=optimized_task, mode="regular")
                )
                if optimized_resp.optimized_task.strip():
                    optimized_task = optimized_resp.optimized_task.strip()
            except Exception as e:
                log.info("optimizer failed; using original task | %r", e)

        prev = sess.extend_system_message or ""
        sess.extend_system_message = (
            "When you finish, call the `done` action. "
            "The `done` payload MUST put the final answer under `data` with fields: "
            "{title: string, steps: string[], links: string[], success: boolean, notes: string|null}. "
            "Return each step as its own item in `steps`. "
            "Notes may be null or omitted. "
            f"{prev}"
        )

        screenshots_dir_str: Optional[str] = None
        action_screenshot_settings: ActionScreenshotSettings | None = None
        if sess.action_screenshots_enabled:
            sess.screenshots_dir.mkdir(parents=True, exist_ok=True)
            screenshots_dir_str = str(sess.screenshots_dir)
            action_screenshot_settings = ActionScreenshotSettings(
                enabled=True,
                output_dir=screenshots_dir_str,
                annotate=sess.action_screenshots_annotate,
                spotlight=sess.screenshot_spotlight,
                include_in_available_files=sess.action_screenshots_include_files,
                session_subdirectories=sess.action_screenshots_session_dirs,
            )

        agent = Agent(
            llm=ChatOpenAI(model=OPENAI_MODEL),
            task=optimized_task,
            start_url=sess.start_url,
            headless=sess.headless,
            workspace_id=sess.workspace_id,
            use_vision=sess.use_vision,
            max_failures=sess.max_failures,
            step_timeout=sess.step_timeout,
            browser=browser,
            max_actions_per_step=sess.max_actions_per_step,
            final_response_after_failure=True,
            output_model_schema=GuideOutput,
            extend_system_message=sess.extend_system_message,
            generate_gif=sess.generate_gif,
            use_thinking=sess.use_thinking,
            flash_mode=sess.flash_mode,
            calculate_cost=sess.calculate_cost,
            vision_detail_level=sess.vision_detail_level,
            directly_open_url=sess.directly_open_url,
            action_screenshots=action_screenshot_settings,
        )

        # starting -> running
        sess.state = "running"
        log.info("agent running...")

        result = await agent.run()

        # store full final response
        sess.final_response = _ensure_json_text(result)

        # extract / enrich GuideOutput
        extracted = _coerce_guide_output_dict(result)
        if extracted is None:
            with contextlib.suppress(Exception):
                extracted = _coerce_guide_output_dict(json.loads(sess.final_response))

        if extracted is not None:
            enriched = _shape_steps_with_placeholders(extracted)
            if sess.action_screenshots_enabled:
                with contextlib.suppress(Exception):
                    enriched = _attach_screenshots_to_steps(enriched, sess.screenshots_dir)
            sess.result_only = _ensure_json_text(enriched)

            # Fill the placeholder guide (best-effort)
            with contextlib.suppress(Exception):
                await db_update_guide_from_result(sess, enriched)

        sess.state = "done"
        log.info("agent done")

    except asyncio.CancelledError:
        sess.state = "stopped"
        logging.getLogger("service").warning("agent stopped")
        # Optional: mark placeholder as draft (no description) if you want
        with contextlib.suppress(Exception):
            if SessionLocal:
                async with SessionLocal() as db:
                    await db.execute(text("""
                        UPDATE guides
                        SET status='draft', updated_at=:now, updated_by=:uid
                        WHERE run_id=:run_id
                    """), {"now": dt.datetime.utcnow(), "uid": DEFAULT_AUTHOR_ID, "run_id": sess.id})
                    await db.commit()
        raise
    except Exception:
        import traceback
        sess.error = traceback.format_exc()
        sess.state = "error"
        logging.getLogger("service").error("ERROR:\n%s", sess.error)
        # Optional: mark placeholder as draft on error too
        with contextlib.suppress(Exception):
            if SessionLocal:
                async with SessionLocal() as db:
                    await db.execute(text("""
                        UPDATE guides
                        SET status='draft', updated_at=:now, updated_by=:uid
                        WHERE run_id=:run_id
                    """), {"now": dt.datetime.utcnow(), "uid": DEFAULT_AUTHOR_ID, "run_id": sess.id})
                    await db.commit()
    finally:
        if browser is not None:
            with contextlib.suppress(Exception):
                await _safe_browser_stop(browser)
        with contextlib.suppress(Exception):
            if sess.chrome_proc and sess.chrome_proc.returncode is None:
                sess.chrome_proc.terminate()
                try:
                    await asyncio.wait_for(sess.chrome_proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    sess.chrome_proc.kill()
        with contextlib.suppress(Exception):
            await db_finish_run(sess)
        run_id_ctx.reset(token)

# =============================================================================
# Startup / Shutdown
# =============================================================================

@app.on_event("startup")
async def on_startup():
    # DB bootstrap
    if engine:
        async with engine.begin() as conn:
            # create tables
            for stmt in CREATE_TABLES_SQL.strip().split(";\n\n"):
                s = stmt.strip().rstrip(";")
                if s:
                    await conn.execute(text(s))
            # best-effort migrations
            for stmt in MIGRATIONS_SQL:
                with contextlib.suppress(Exception):
                    await conn.execute(text(stmt))

    # logging to console + per-run handler
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(logging.StreamHandler())
    if not any(isinstance(f, RunIdFilter) for f in root.filters):
        root.addFilter(RunIdFilter())
    if not any(isinstance(h, PerRunDBAndMemoryHandler) for h in root.handlers):
        h = PerRunDBAndMemoryHandler()
        h.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(h)

    _hijack_library_loggers()

    # start FIFO workers
    await queue_mgr.start()

@app.on_event("shutdown")
async def on_shutdown():
    with contextlib.suppress(Exception):
        await queue_mgr.stop()
    for sess in list(sessions.values()):
        with contextlib.suppress(Exception):
            if sess.chrome_proc and sess.chrome_proc.returncode is None:
                sess.chrome_proc.kill()

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", dependencies=[Depends(require_api_key)])
def root():
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "chrome": CHROME_BIN,
        "db": bool(DB_URL),
        "max_concurrency": MAX_CONCURRENCY,
        "origins": ALLOWED_ORIGINS,
        "allow_credentials": _allow_credentials,
        "model": OPENAI_MODEL,
    }

@app.post("/sessions", response_model=Status, dependencies=[Depends(require_api_key)])
async def start_session(req: StartReq):
    # Require workspace_id (or ws_id alias)
    if req.workspace_id is None:
        raise HTTPException(status_code=400, detail="workspace_id (or ws_id) is required")
    sid = uuid.uuid4().hex[:12]
    sess = Session(session_id=sid, req=req)
    sessions[sid] = sess
    pos = await queue_mgr.enqueue(sid)
    return sess.status(queue_position=pos)

@app.get("/sessions/{sid}", response_model=Status, dependencies=[Depends(require_api_key)])
async def get_status(sid: str):
    sess = sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    pos = await queue_mgr.position(sid) if sess.state == "queued" else None
    return sess.status(queue_position=pos)

@app.get("/sessions/{sid}/queue", dependencies=[Depends(require_api_key)])
async def get_queue_position(sid: str):
    sess = sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    pos = await queue_mgr.position(sid)
    return {"session_id": sid, "position": pos}

@app.get("/sessions/{sid}/logs", dependencies=[Depends(require_api_key)])
async def get_logs(sid: str, offset: int = 0, plain: bool = False):
    sess = sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    offset = max(0, int(offset))
    if plain:
        lines = sess.log_lines[offset:]
        text_blob = "\n".join(lines)
        return PlainTextResponse(text_blob if text_blob else "", headers={"X-Log-Size": str(len(sess.log_lines))})
    slice_ = sess.log_lines[offset:]
    return {"count": len(slice_), "lines": slice_}

@app.get("/sessions/{sid}/result", response_model=ResultPayload, dependencies=[Depends(require_api_key)])
async def get_result(sid: str):
    sess = sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    return ResultPayload(
        state=sess.state,
        final_response=sess.final_response,
        result=sess.result_only,
        error=sess.error,
        screenshots_dir=str(sess.screenshots_dir) if sess.action_screenshots_enabled else None,
    )

@app.get("/sessions/{sid}/screenshots", dependencies=[Depends(require_api_key)])
async def list_screenshots(sid: str):
    sess = sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    if not sess.action_screenshots_enabled:
        return {"enabled": False, "images": []}
    d = sess.screenshots_dir
    if not d.exists():
        return {"enabled": True, "images": []}
    files = sorted(_to_public_image_path(p) for p in d.glob("*") if p.is_file())
    return {"enabled": True, "images": files}

@app.post("/sessions/{sid}/stop", response_model=Status, dependencies=[Depends(require_api_key)])
async def stop_session(sid: str):
    """
    Best effort:
    - If queued: remove from queue and mark stopped.
    - If running: terminate Chrome and mark stopped.
    """
    sess = sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")

    if sess.state == "queued":
        removed = await queue_mgr.remove_if_present(sid)
        if removed:
            sess.state = "stopped"
            # Optional: transition placeholder to draft
            with contextlib.suppress(Exception):
                if SessionLocal:
                    async with SessionLocal() as db:
                        await db.execute(text("""
                            UPDATE guides
                            SET status='draft', updated_at=:now, updated_by=:uid
                            WHERE run_id=:run_id
                        """), {"now": dt.datetime.utcnow(), "uid": DEFAULT_AUTHOR_ID, "run_id": sess.id})
                        await db.commit()
            return sess.status(queue_position=None)

    with contextlib.suppress(Exception):
        if sess.chrome_proc and sess.chrome_proc.returncode is None:
            sess.chrome_proc.terminate()
    sess.state = "stopped"
    # Optional: transition placeholder to draft
    with contextlib.suppress(Exception):
        if SessionLocal:
            async with SessionLocal() as db:
                await db.execute(text("""
                    UPDATE guides
                    SET status='draft', updated_at=:now, updated_by=:uid
                    WHERE run_id=:run_id
                """), {"now": dt.datetime.utcnow(), "uid": DEFAULT_AUTHOR_ID, "run_id": sess.id})
                await db.commit()
    return sess.status(queue_position=None)

@app.post("/optimize", response_model=OptimizeResp, dependencies=[Depends(require_api_key)])
async def optimize_prompt(req: OptimizeReq):
    model = req.llm_model or OPENAI_MODEL
    opt = OptimizeAgentPrompt(
        task=req.task,
        mode=req.mode,
        ptype=req.ptype,
        llm=ChatOpenAI(model=model),
    )
    out = await opt.optimize_async()
    if not isinstance(out, str) or not out.strip():
        out = req.task
    return OptimizeResp(original=req.task, optimized=out)
