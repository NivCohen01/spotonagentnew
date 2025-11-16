from __future__ import annotations

import contextlib
import datetime as dt
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .service_config import DB_URL, DEFAULT_AUTHOR_ID
from .service_models import ActionTraceEntry
from .service_trace import _flatten_action_trace_summary, slugify_title_for_guide

if TYPE_CHECKING:
    from .service_agent import Session

engine = create_async_engine(DB_URL, pool_pre_ping=True) if DB_URL else None
SessionLocal = async_sessionmaker(engine, expire_on_commit=False) if engine else None

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
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS ws_id BIGINT NULL AFTER id",
    "CREATE INDEX IF NOT EXISTS idx_run_state ON runs (state)",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS final_response LONGTEXT NULL AFTER finished_at",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS result LONGTEXT NULL AFTER final_response",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS error LONGTEXT NULL AFTER result",
    "CREATE INDEX IF NOT EXISTS idx_run_ts ON run_logs (run_id, ts)",
    "ALTER TABLE guides ADD COLUMN IF NOT EXISTS run_id VARCHAR(16) NULL AFTER workspace_id",
    "CREATE INDEX IF NOT EXISTS idx_guides_run_id ON guides (run_id)",
    "ALTER TABLE guides ADD CONSTRAINT fk_guides_runs FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL",
    "ALTER TABLE guides MODIFY COLUMN status ENUM('generating','draft','published','archived') NOT NULL DEFAULT 'draft'",
]


async def db_insert_run(sess: "Session"):
    if not SessionLocal:
        return
    async with SessionLocal() as db:
        await db.execute(
            text(
                """INSERT INTO runs (id, ws_id, task, start_url, headless, state)
                    VALUES (:id, :ws_id, :task, :start_url, :headless, :state)"""
            ),
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


async def db_finish_run(sess: "Session"):
    if not SessionLocal:
        return
    sess.finished_at = dt.datetime.utcnow()
    async with SessionLocal() as db:
        await db.execute(
            text(
                """UPDATE runs
                    SET state=:state, finished_at=:finished_at,
                        final_response=:final_response, result=:result, error=:error
                    WHERE id=:id"""
            ),
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


async def db_fetch_action_trace(session_id: str) -> list[ActionTraceEntry]:
    """Fetch and parse the stored action_trace summary from guides.description for a run."""
    if not SessionLocal:
        return []

    async with SessionLocal() as db:
        row = await db.execute(
            text("SELECT description FROM guides WHERE run_id=:run_id ORDER BY id DESC LIMIT 1"),
            {"run_id": session_id},
        )
        description = row.scalar()

    if not description:
        return []

    if isinstance(description, (bytes, bytearray)):
        with contextlib.suppress(Exception):
            description = description.decode("utf-8", errors="ignore")

    desc_obj = None
    if isinstance(description, str):
        with contextlib.suppress(Exception):
            desc_obj = json.loads(description)
    elif isinstance(description, dict):
        desc_obj = description

    if not isinstance(desc_obj, dict):
        return []

    trace = desc_obj.get("action_trace") or desc_obj.get("actionTrace")
    return _flatten_action_trace_summary(trace)


async def db_create_empty_guide(sess: "Session") -> None:
    """
    Create an empty/placeholder guide row right after we create the run.
    Links it via guides.run_id = sess.id so we can update later.
    """
    if not SessionLocal or sess.workspace_id is None:
        return

    now = dt.datetime.utcnow()
    placeholder_slug = f"run-{sess.id}"
    title = (sess.task or "Generating guide").strip()
    if len(title) > 255:
        title = title[:255]

    data: Dict[str, Any] = {
        "workspace_id": sess.workspace_id,
        "run_id": sess.id,
        "slug": placeholder_slug,
        "title": title or "Generating guide",
        "status": "generating",
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


async def db_update_guide_from_result(sess: "Session", enriched: dict):
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

    update_sql = text(
        """
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
    """
    )

    params_update = {
        "title": title,
        "slug": new_slug,
        "description": description_json,
        "status": "draft",
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

        if sess.workspace_id is None:
            return

        insert_sql = text(
            """
            INSERT INTO guides
              (workspace_id, run_id, slug, title, description, status, visibility,
               created_by, updated_by, created_at, updated_at)
            VALUES
              (:workspace_id, :run_id, :slug, :title, :description, :status, :visibility,
               :created_by, :updated_by, :created_at, :updated_at)
        """
        )
        await db.execute(
            insert_sql,
            {
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
            },
        )
        await db.commit()
