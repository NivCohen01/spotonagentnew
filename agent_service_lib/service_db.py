from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio import AsyncConnection

from .service_config import DB_URL, DEFAULT_AUTHOR_ID
from .service_models import ActionTraceEntry
from .service_trace import _flatten_action_trace_summary, slugify_title_for_guide

if TYPE_CHECKING:
    from .service_agent import Session

GUIDE_SCHEMA_VERSION = "v2"

logger = logging.getLogger("service")

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

CREATE TABLE IF NOT EXISTS guide_families (
  id          BIGINT AUTO_INCREMENT PRIMARY KEY,
  workspace_id BIGINT NOT NULL,
  family_key  VARCHAR(191) NULL,
  created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  deleted_at  DATETIME NULL,
  UNIQUE KEY uq_guide_families_ws_key (workspace_id, family_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS guide_family_variants (
  guide_family_id BIGINT NOT NULL,
  guide_id        BIGINT NOT NULL,
  device_type     ENUM('desktop','mobile') NOT NULL,
  version         INT NOT NULL DEFAULT 1,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (guide_family_id, device_type, version),
  UNIQUE KEY uq_guide_family_variants_guide_id (guide_id),
  CONSTRAINT fk_guide_family_variants_family
    FOREIGN KEY (guide_family_id) REFERENCES guide_families(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_guide_family_variants_guide
    FOREIGN KEY (guide_id) REFERENCES guides(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

MIGRATIONS_SQL = [
    "ALTER TABLE runs ADD COLUMN ws_id BIGINT NULL AFTER id",
    "CREATE INDEX idx_run_state ON runs (state)",
    "ALTER TABLE runs ADD COLUMN final_response LONGTEXT NULL AFTER finished_at",
    "ALTER TABLE runs ADD COLUMN result LONGTEXT NULL AFTER final_response",
    "ALTER TABLE runs ADD COLUMN error LONGTEXT NULL AFTER result",
    "CREATE INDEX idx_run_ts ON run_logs (run_id, ts)",
    "ALTER TABLE guides ADD COLUMN run_id VARCHAR(16) NULL AFTER workspace_id",
    "ALTER TABLE guides ADD COLUMN video_filename VARCHAR(255) NULL AFTER description",
    "CREATE INDEX idx_guides_run_id ON guides (run_id)",
    "ALTER TABLE guides ADD CONSTRAINT fk_guides_runs FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL",
    "ALTER TABLE guides MODIFY COLUMN status ENUM('generating','draft','published','archived') NOT NULL DEFAULT 'draft'",
    "ALTER TABLE guide_families ADD COLUMN family_key VARCHAR(191) NULL AFTER workspace_id",
    "ALTER TABLE guide_families ADD COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
    "ALTER TABLE guide_families ADD COLUMN updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
    "ALTER TABLE guide_families ADD COLUMN deleted_at DATETIME NULL AFTER updated_at",
    "CREATE UNIQUE INDEX uq_guide_families_ws_key ON guide_families (workspace_id, family_key)",
    "ALTER TABLE guide_family_variants ADD COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
    "ALTER TABLE guide_family_variants ADD COLUMN updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
    "ALTER TABLE guide_family_variants MODIFY COLUMN device_type ENUM('desktop','mobile') NOT NULL",
    "ALTER TABLE guide_family_variants ADD COLUMN version INT NOT NULL DEFAULT 1",
    "ALTER TABLE guide_family_variants DROP PRIMARY KEY",
    "ALTER TABLE guide_family_variants ADD PRIMARY KEY (guide_family_id, device_type, version)",
    "CREATE UNIQUE INDEX uq_guide_family_variants_guide_id ON guide_family_variants (guide_id)",
    "ALTER TABLE guide_family_variants ADD CONSTRAINT fk_guide_family_variants_family FOREIGN KEY (guide_family_id) REFERENCES guide_families(id) ON DELETE CASCADE",
    "ALTER TABLE guide_family_variants ADD CONSTRAINT fk_guide_family_variants_guide FOREIGN KEY (guide_id) REFERENCES guides(id) ON DELETE CASCADE",
]


def _normalize_device_type(device_type: str | None) -> str:
    if not device_type:
        return "desktop"
    lowered = device_type.lower()
    return lowered if lowered in ("desktop", "mobile") else "desktop"


def derive_guide_family_key(workspace_id: Optional[int], task: Optional[str], start_url: Optional[str]) -> str:
    base_parts: list[str] = [f"schema:{GUIDE_SCHEMA_VERSION}"]
    if workspace_id is not None:
        base_parts.append(str(workspace_id))
    if start_url:
        base_parts.append(start_url.strip())
    if task:
        base_parts.append(task.strip())
    raw = " | ".join(part for part in base_parts if part) or "guide"
    cleaned = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    cleaned = cleaned[:160] if cleaned else "guide"
    key = f"{cleaned}-{digest}"
    return key[:191]


async def _column_exists(conn: AsyncConnection, table: str, column: str) -> bool:
    res = await conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table AND COLUMN_NAME = :column
            LIMIT 1
        """
        ),
        {"table": table, "column": column},
    )
    return res.scalar() is not None


async def _index_exists(conn: AsyncConnection, table: str, index: str) -> bool:
    res = await conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table AND INDEX_NAME = :index
            LIMIT 1
        """
        ),
        {"table": table, "index": index},
    )
    return res.scalar() is not None


async def _constraint_exists(conn: AsyncConnection, table: str, constraint: str) -> bool:
    res = await conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.TABLE_CONSTRAINTS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table AND CONSTRAINT_NAME = :constraint
            LIMIT 1
        """
        ),
        {"table": table, "constraint": constraint},
    )
    return res.scalar() is not None


async def _primary_key_columns(conn: AsyncConnection, table: str) -> list[str]:
    res = await conn.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table AND CONSTRAINT_NAME = 'PRIMARY'
            ORDER BY ORDINAL_POSITION
        """
        ),
        {"table": table},
    )
    return [row[0] for row in res.fetchall()]


def _dialect_name() -> str:
    try:
        if engine and getattr(engine, "dialect", None):
            return (engine.dialect.name or "").lower()
    except Exception:
        return ""
    return ""


def _for_update_clause() -> str:
    return " FOR UPDATE" if _dialect_name() not in ("sqlite", "") else ""


def _last_insert_id_sql() -> str:
    return "SELECT last_insert_rowid()" if _dialect_name() == "sqlite" else "SELECT LAST_INSERT_ID()"


def _is_already_exists_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("already exists", "duplicate", "errno 1060", "errno 1061", "errno 1062"))


async def _execute_ignoring_exists(conn: AsyncConnection, stmt: str, params: Optional[dict] = None) -> None:
    try:
        await conn.execute(text(stmt), params or {})
    except Exception as exc:  # pragma: no cover - defensive
        if _is_already_exists_error(exc):
            logging.getLogger("service").info("Migration skipped (already exists): %s", stmt)
            return
        raise


async def run_migrations(conn: AsyncConnection) -> None:
    log = logging.getLogger("service")

    if not await _column_exists(conn, "runs", "ws_id"):
        await _execute_ignoring_exists(conn, "ALTER TABLE runs ADD COLUMN ws_id BIGINT NULL AFTER id")
    if not await _index_exists(conn, "runs", "idx_run_state"):
        await _execute_ignoring_exists(conn, "CREATE INDEX idx_run_state ON runs (state)")
    if not await _column_exists(conn, "runs", "final_response"):
        await _execute_ignoring_exists(conn, "ALTER TABLE runs ADD COLUMN final_response LONGTEXT NULL AFTER finished_at")
    if not await _column_exists(conn, "runs", "result"):
        await _execute_ignoring_exists(conn, "ALTER TABLE runs ADD COLUMN result LONGTEXT NULL AFTER final_response")
    if not await _column_exists(conn, "runs", "error"):
        await _execute_ignoring_exists(conn, "ALTER TABLE runs ADD COLUMN error LONGTEXT NULL AFTER result")
    if not await _index_exists(conn, "run_logs", "idx_run_ts"):
        await _execute_ignoring_exists(conn, "CREATE INDEX idx_run_ts ON run_logs (run_id, ts)")

    if not await _column_exists(conn, "guides", "run_id"):
        await _execute_ignoring_exists(conn, "ALTER TABLE guides ADD COLUMN run_id VARCHAR(16) NULL AFTER workspace_id")
    if not await _column_exists(conn, "guides", "video_filename"):
        await _execute_ignoring_exists(conn, "ALTER TABLE guides ADD COLUMN video_filename VARCHAR(255) NULL AFTER description")
    if not await _index_exists(conn, "guides", "idx_guides_run_id"):
        await _execute_ignoring_exists(conn, "CREATE INDEX idx_guides_run_id ON guides (run_id)")
    if not await _constraint_exists(conn, "guides", "fk_guides_runs"):
        await _execute_ignoring_exists(
            conn, "ALTER TABLE guides ADD CONSTRAINT fk_guides_runs FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL"
        )
    with contextlib.suppress(Exception):
        await conn.execute(text("ALTER TABLE guides MODIFY COLUMN status ENUM('generating','draft','published','archived') NOT NULL DEFAULT 'draft'"))

    # guide_families table columns/indexes for pre-existing tables
    if not await _column_exists(conn, "guide_families", "family_key"):
        await _execute_ignoring_exists(conn, "ALTER TABLE guide_families ADD COLUMN family_key VARCHAR(191) NULL AFTER workspace_id")
    if not await _column_exists(conn, "guide_families", "created_at"):
        await _execute_ignoring_exists(conn, "ALTER TABLE guide_families ADD COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")
    if not await _column_exists(conn, "guide_families", "updated_at"):
        await _execute_ignoring_exists(
            conn, "ALTER TABLE guide_families ADD COLUMN updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        )
    if not await _column_exists(conn, "guide_families", "deleted_at"):
        await _execute_ignoring_exists(conn, "ALTER TABLE guide_families ADD COLUMN deleted_at DATETIME NULL AFTER updated_at")
    if not await _index_exists(conn, "guide_families", "uq_guide_families_ws_key"):
        await _execute_ignoring_exists(conn, "CREATE UNIQUE INDEX uq_guide_families_ws_key ON guide_families (workspace_id, family_key)")

    if not await _column_exists(conn, "guide_family_variants", "created_at"):
        await _execute_ignoring_exists(conn, "ALTER TABLE guide_family_variants ADD COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")
    if not await _column_exists(conn, "guide_family_variants", "updated_at"):
        await _execute_ignoring_exists(
            conn, "ALTER TABLE guide_family_variants ADD COLUMN updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        )
    with contextlib.suppress(Exception):
        await conn.execute(text("ALTER TABLE guide_family_variants MODIFY COLUMN device_type ENUM('desktop','mobile') NOT NULL"))
    if not await _column_exists(conn, "guide_family_variants", "version"):
        await _execute_ignoring_exists(
            conn, "ALTER TABLE guide_family_variants ADD COLUMN version INT NOT NULL DEFAULT 1 AFTER device_type"
        )
    try:
        pk_cols = [col.lower() for col in await _primary_key_columns(conn, "guide_family_variants")]
    except Exception:  # pragma: no cover - defensive for non-MySQL engines
        pk_cols = []
    expected_pk = ["guide_family_id", "device_type", "version"]
    if pk_cols and pk_cols != expected_pk:
        await _execute_ignoring_exists(conn, "ALTER TABLE guide_family_variants DROP PRIMARY KEY")
        await _execute_ignoring_exists(
            conn, "ALTER TABLE guide_family_variants ADD PRIMARY KEY (guide_family_id, device_type, version)"
        )
    elif not pk_cols:
        await _execute_ignoring_exists(
            conn, "ALTER TABLE guide_family_variants ADD PRIMARY KEY (guide_family_id, device_type, version)"
        )
    if not await _index_exists(conn, "guide_family_variants", "uq_guide_family_variants_guide_id"):
        await _execute_ignoring_exists(conn, "CREATE UNIQUE INDEX uq_guide_family_variants_guide_id ON guide_family_variants (guide_id)")
    if not await _constraint_exists(conn, "guide_family_variants", "fk_guide_family_variants_family"):
        await _execute_ignoring_exists(
            conn,
            "ALTER TABLE guide_family_variants ADD CONSTRAINT fk_guide_family_variants_family FOREIGN KEY (guide_family_id) REFERENCES guide_families(id) ON DELETE CASCADE",
        )
    if not await _constraint_exists(conn, "guide_family_variants", "fk_guide_family_variants_guide"):
        await _execute_ignoring_exists(
            conn,
            "ALTER TABLE guide_family_variants ADD CONSTRAINT fk_guide_family_variants_guide FOREIGN KEY (guide_id) REFERENCES guides(id) ON DELETE CASCADE",
        )

    log.info("Migrations applied (idempotent check complete)")


async def ensure_family_and_variant_guide(
    workspace_id: Optional[int],
    family_key: Optional[str],
    device_type: str,
    run_id: str,
    title: str,
    visibility: str = "private",
    status: str = "generating",
    guide_family_id: Optional[int] = None,
    existing_guide_id: Optional[int] = None,
) -> Optional[tuple[int, int, str]]:
    """
    Ensure there is a guide_family row and create a new versioned variant per (family, device_type).
    Returns (family_id, guide_id, slug) when successful.
    """
    if not SessionLocal or workspace_id is None:
        return None

    log = logging.getLogger("service")
    now = dt.datetime.utcnow()
    normalized_device = _normalize_device_type(device_type)

    cleaned_title = (title or "Generating guide").strip()
    if len(cleaned_title) > 255:
        cleaned_title = cleaned_title[:255]

    family_key_val = (family_key or "").strip() or derive_guide_family_key(workspace_id, cleaned_title, None)
    family_key_val = family_key_val[:191]

    slug_value = ""
    async with SessionLocal() as db:
        async with db.begin():
            fam_id = guide_family_id
            if fam_id:
                fam_row = await db.execute(
                    text(f"SELECT id FROM guide_families WHERE id=:fid{_for_update_clause()}"), {"fid": fam_id}
                )
                fam_id = fam_row.scalar()

            if not fam_id:
                fam_row = await db.execute(
                    text(
                        f"SELECT id FROM guide_families WHERE workspace_id=:ws_id AND family_key=:family_key LIMIT 1{_for_update_clause()}"
                    ),
                    {"ws_id": workspace_id, "family_key": family_key_val},
                )
                fam_id = fam_row.scalar()

            created_family = False
            if not fam_id:
                await db.execute(
                    text(
                        """
                        INSERT INTO guide_families (workspace_id, family_key, created_at, updated_at)
                        VALUES (:ws_id, :family_key, :created_at, :updated_at)
                    """
                    ),
                    {"ws_id": workspace_id, "family_key": family_key_val, "created_at": now, "updated_at": now},
                )
                fam_row = await db.execute(text(_last_insert_id_sql()))
                fam_id = fam_row.scalar()
                created_family = True

            await db.execute(text("UPDATE guide_families SET updated_at=:updated_at WHERE id=:fid"), {"updated_at": now, "fid": fam_id})

            latest_variant_row = await db.execute(
                text(
                    f"""
                    SELECT guide_id, version
                    FROM guide_family_variants
                    WHERE guide_family_id=:family_id AND device_type=:device_type
                    ORDER BY version DESC
                    LIMIT 1{_for_update_clause()}
                """
                ),
                {"family_id": fam_id, "device_type": normalized_device},
            )
            latest_variant = latest_variant_row.first()
            previous_guide_id = latest_variant[0] if latest_variant else None
            latest_version = int(latest_variant[1]) if latest_variant and latest_variant[1] is not None else 0
            existing_variant_version = None
            if existing_guide_id is not None:
                existing_variant_row = await db.execute(
                    text(
                        f"""
                        SELECT version
                        FROM guide_family_variants
                        WHERE guide_family_id=:family_id AND device_type=:device_type AND guide_id=:guide_id
                        LIMIT 1{_for_update_clause()}
                    """
                    ),
                    {"family_id": fam_id, "device_type": normalized_device, "guide_id": existing_guide_id},
                )
                existing_variant_version = existing_variant_row.scalar()
            reuse_version = existing_variant_version
            if reuse_version is None and latest_variant and existing_guide_id is not None and latest_variant[0] == existing_guide_id:
                reuse_version = latest_version

            guide_id = existing_guide_id
            created_variant = False

            if guide_id:
                update_params = {
                    "guide_id": guide_id,
                    "run_id": run_id,
                    "status": status,
                    "visibility": visibility,
                    "updated_at": now,
                    "updated_by": DEFAULT_AUTHOR_ID,
                    "title": cleaned_title,
                }
                await db.execute(
                    text(
                        """
                        UPDATE guides
                        SET run_id=:run_id,
                            title=:title,
                            status=:status,
                            visibility=:visibility,
                            updated_at=:updated_at,
                            updated_by=:updated_by
                        WHERE id=:guide_id
                        LIMIT 1
                    """
                    ),
                    update_params,
                )

            if not guide_id:
                slug = slugify_title_for_guide(cleaned_title)
                data: Dict[str, Any] = {
                    "workspace_id": workspace_id,
                    "run_id": run_id,
                    "slug": slug,
                    "title": cleaned_title,
                    "status": status,
                    "visibility": visibility,
                    "created_at": now,
                    "updated_at": now,
                }
                if DEFAULT_AUTHOR_ID is not None:
                    data["created_by"] = DEFAULT_AUTHOR_ID
                    data["updated_by"] = DEFAULT_AUTHOR_ID

                cols = ", ".join(data.keys())
                params = ", ".join(f":{k}" for k in data.keys())
                insert_sql = f"INSERT INTO guides ({cols}) VALUES ({params})"
                await db.execute(text(insert_sql), data)
                gid_row = await db.execute(text(_last_insert_id_sql()))
                guide_id = gid_row.scalar()
                slug_value = slug
            else:
                slug_value_row = await db.execute(text("SELECT slug FROM guides WHERE id=:gid"), {"gid": guide_id})
                slug_value = slug_value_row.scalar() or ""

            version_to_use = reuse_version
            if guide_id is not None and version_to_use is None:
                next_version = (latest_version or 0) + 1
                insert_variant_sql = text(
                    """
                    INSERT INTO guide_family_variants (guide_family_id, guide_id, device_type, version, created_at, updated_at)
                    VALUES (:family_id, :guide_id, :device_type, :version, :created_at, :updated_at)
                """
                )
                attempt_version = next_version
                for _ in range(2):
                    try:
                        await db.execute(
                            insert_variant_sql,
                            {
                                "family_id": fam_id,
                                "guide_id": guide_id,
                                "device_type": normalized_device,
                                "version": attempt_version,
                                "created_at": now,
                                "updated_at": now,
                            },
                        )
                        version_to_use = attempt_version
                        created_variant = True
                        break
                    except Exception:
                        retry_row = await db.execute(
                            text(
                                f"""
                                SELECT guide_id, version
                                FROM guide_family_variants
                                WHERE guide_family_id=:family_id AND device_type=:device_type
                                ORDER BY version DESC
                                LIMIT 1{_for_update_clause()}
                            """
                            ),
                            {"family_id": fam_id, "device_type": normalized_device},
                        )
                        retry_latest = retry_row.first()
                        if not retry_latest:
                            raise
                        attempt_version = (int(retry_latest[1]) if retry_latest[1] is not None else 0) + 1
                        previous_guide_id = previous_guide_id or retry_latest[0]

            if guide_id:
                slug_row = await db.execute(text("SELECT slug FROM guides WHERE id=:gid"), {"gid": guide_id})
                slug_value = slug_row.scalar() or slug_value

            if created_family:
                log.info("Created guide family %s (workspace=%s key=%s)", fam_id, workspace_id, family_key_val)
            if created_variant and guide_id:
                log.info(
                    "Created guide variant %s v%s for family %s device=%s (previous latest was %s)",
                    guide_id,
                    version_to_use,
                    fam_id,
                    normalized_device,
                    previous_guide_id or "none",
                )
            elif guide_id and version_to_use is not None and version_to_use == reuse_version:
                log.info(
                    "Using existing guide variant %s v%s for family %s device=%s",
                    guide_id,
                    version_to_use,
                    fam_id,
                    normalized_device,
                )

    return (fam_id, guide_id, slug_value)


async def db_insert_run(sess: "Session"):
    if not SessionLocal:
        return
    log = logging.getLogger("service")

    if sess.run_preexists:
        # NodeJS already created the runs record — just update state.
        async with SessionLocal() as db:
            await db.execute(
                text("UPDATE runs SET state=:state WHERE id=:id"),
                {"state": "starting", "id": sess.id},
            )
            await db.commit()
        log.info("Run %s pre-exists (NodeJS job queue); updated state to 'starting'", sess.id)
    else:
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

    family_key = getattr(sess, "guide_family_key", None) or derive_guide_family_key(sess.workspace_id, sess.task, sess.start_url)
    title = (sess.task or "Generating guide").strip()
    ensured = await ensure_family_and_variant_guide(
        workspace_id=sess.workspace_id,
        family_key=family_key,
        device_type=sess.device_type,
        run_id=sess.id,
        title=title or "Generating guide",
        visibility="private",
        status="generating",
        guide_family_id=getattr(sess, "guide_family_id", None),
        existing_guide_id=getattr(sess, "guide_id", None),
    )
    if ensured:
        sess.guide_family_id, sess.guide_id, sess.guide_slug = ensured
        sess.guide_family_key = family_key
        log.info(
            "Ensured guide family/link for run %s | family_id=%s guide_id=%s device=%s",
            sess.id,
            sess.guide_family_id,
            sess.guide_id,
            sess.device_type,
        )


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
        logger.warning("db_fetch_action_trace: SessionLocal is None; DB_URL missing?")
        return []

    async with SessionLocal() as db:
        row = await db.execute(
            text("SELECT description FROM guides WHERE run_id=:run_id ORDER BY id DESC LIMIT 1"),
            {"run_id": session_id},
        )
        description = row.scalar()

    desc_obj = None
    if description:
        if isinstance(description, (bytes, bytearray)):
            with contextlib.suppress(Exception):
                description = description.decode("utf-8", errors="ignore")

        if isinstance(description, str):
            try:
                desc_obj = json.loads(description)
            except Exception as exc:
                logging.getLogger("service").warning("db_fetch_action_trace: json parse failed for session %s: %r", session_id, exc)
                desc_obj = None
        elif isinstance(description, dict):
            desc_obj = description
    else:
        logger.warning("db_fetch_action_trace: no guide description found for run_id=%s", session_id)

    def _entries_from_obj(obj: Any) -> list[ActionTraceEntry]:
        if not isinstance(obj, dict):
            return []
        full = obj.get("action_trace_full") or obj.get("actionTraceFull")
        if isinstance(full, list):
            entries: list[ActionTraceEntry] = []
            for item in full:
                if not isinstance(item, dict):
                    continue
                with contextlib.suppress(Exception):
                    entries.append(ActionTraceEntry.model_validate(item))
            if entries:
                return entries
        trace = obj.get("action_trace") or obj.get("actionTrace")
        return _flatten_action_trace_summary(trace)

    entries = _entries_from_obj(desc_obj)
    if entries:
        return entries

    # Fallback: try runs.result or runs.final_response
    async with SessionLocal() as db:
        row = await db.execute(
            text("SELECT result, final_response FROM runs WHERE id=:run_id LIMIT 1"),
            {"run_id": session_id},
        )
        run_row = row.fetchone()

    if run_row:
        mapping = run_row._mapping if hasattr(run_row, "_mapping") else {}
        for col in ("result", "final_response"):
            payload = mapping.get(col)
            if not payload:
                continue
            if isinstance(payload, (bytes, bytearray)):
                with contextlib.suppress(Exception):
                    payload = payload.decode("utf-8", errors="ignore")
            obj = None
            if isinstance(payload, str):
                with contextlib.suppress(Exception):
                    obj = json.loads(payload)
            elif isinstance(payload, dict):
                obj = payload
            entries = _entries_from_obj(obj)
            if entries:
                logger.info("db_fetch_action_trace: loaded trace from runs.%s for run_id=%s", col, session_id)
                return entries

    logger.warning("db_fetch_action_trace: no trace found for run_id=%s", session_id)
    return []


async def db_create_empty_guide(sess: "Session") -> None:
    """
    Create an empty/placeholder guide row right after we create the run.
    Links it via guides.run_id = sess.id so we can update later.
    """
    if not SessionLocal or sess.workspace_id is None:
        return

    family_key = getattr(sess, "guide_family_key", None) or derive_guide_family_key(sess.workspace_id, sess.task, sess.start_url)
    title = (sess.task or "Generating guide").strip()

    ensured = await ensure_family_and_variant_guide(
        workspace_id=sess.workspace_id,
        family_key=family_key,
        device_type=sess.device_type,
        run_id=sess.id,
        title=title or "Generating guide",
        visibility="private",
        status="generating",
        guide_family_id=getattr(sess, "guide_family_id", None),
        existing_guide_id=getattr(sess, "guide_id", None),
    )
    if ensured:
        sess.guide_family_id, sess.guide_id, sess.guide_slug = ensured
        sess.guide_family_key = family_key


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

    if sess.workspace_id is not None:
        family_key = getattr(sess, "guide_family_key", None) or derive_guide_family_key(sess.workspace_id, sess.task, sess.start_url)
        ensured = await ensure_family_and_variant_guide(
            workspace_id=sess.workspace_id,
            family_key=family_key,
            device_type=sess.device_type,
            run_id=sess.id,
            title=title,
            visibility="private",
            status="draft",
            guide_family_id=getattr(sess, "guide_family_id", None),
            existing_guide_id=getattr(sess, "guide_id", None),
        )
        if ensured:
            sess.guide_family_id, sess.guide_id, sess.guide_slug = ensured
            sess.guide_family_key = family_key

    condition_sql = "id=:guide_id" if sess.guide_id else "run_id=:run_id"
    params_update = {
        "title": title,
        "slug": new_slug,
        "description": description_json,
        "status": "draft",
        "visibility": "private",
        "updated_at": now,
        "updated_by": DEFAULT_AUTHOR_ID,
        "guide_id": sess.guide_id,
        "run_id": sess.id,
    }

    update_sql = text(
        f"""
        UPDATE guides
        SET title=:title,
            slug=:slug,
            description=:description,
            status=:status,
            visibility=:visibility,
            updated_at=:updated_at,
            updated_by=:updated_by,
            run_id=:run_id
        WHERE {condition_sql}
        LIMIT 1
    """
    )

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
        gid_row = await db.execute(text(_last_insert_id_sql()))
        maybe_new_id = gid_row.scalar()
        if sess.guide_family_id and maybe_new_id:
            with contextlib.suppress(Exception):
                normalized_device = _normalize_device_type(sess.device_type)
                latest_variant_row = await db.execute(
                    text(
                        f"""
                        SELECT version
                        FROM guide_family_variants
                        WHERE guide_family_id=:family_id AND device_type=:device_type
                        ORDER BY version DESC
                        LIMIT 1{_for_update_clause()}
                    """
                    ),
                    {"family_id": sess.guide_family_id, "device_type": normalized_device},
                )
                latest_version = latest_variant_row.scalar()
                next_version = (int(latest_version) if latest_version is not None else 0) + 1
                await db.execute(
                    text(
                        """
                        INSERT INTO guide_family_variants (guide_family_id, guide_id, device_type, version, created_at, updated_at)
                        VALUES (:family_id, :guide_id, :device_type, :version, :created_at, :updated_at)
                    """
                    ),
                    {
                        "family_id": sess.guide_family_id,
                        "guide_id": maybe_new_id,
                        "device_type": normalized_device,
                        "version": next_version,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
            sess.guide_id = maybe_new_id
            sess.guide_slug = new_slug
        await db.commit()


async def db_update_video_filename(run_id: str, filename: str) -> None:
    """Persist the generated video filename (or 'failed') for a run's guide."""
    if not SessionLocal:
        return

    now = dt.datetime.utcnow()
    async with SessionLocal() as db:
        res = await db.execute(
            text(
                """
                UPDATE guides
                SET video_filename=:filename,
                    updated_at=:updated_at,
                    updated_by=:updated_by
                WHERE run_id=:run_id
                """
            ),
            {
                "filename": filename,
                "updated_at": now,
                "updated_by": DEFAULT_AUTHOR_ID,
                "run_id": run_id,
            },
        )
        await db.commit()
        if not res.rowcount or res.rowcount <= 0:
            logging.getLogger("service").info("Video filename update skipped; no guide linked to run_id=%s", run_id)


async def db_fetch_video_info(*, run_id: Optional[str] = None, guide_id: Optional[int] = None) -> tuple[Optional[str], Optional[str]]:
    """
    Fetch video filename and run_id for a guide.
    Prefers direct run_id lookup; falls back to guide_id when provided.
    """
    if not SessionLocal or (run_id is None and guide_id is None):
        return None, run_id

    query = ""
    params: dict[str, Any] = {}
    if run_id is not None:
        query = "SELECT video_filename, run_id FROM guides WHERE run_id=:run_id ORDER BY id DESC LIMIT 1"
        params["run_id"] = run_id
    else:
        query = "SELECT video_filename, run_id FROM guides WHERE id=:guide_id LIMIT 1"
        params["guide_id"] = guide_id

    async with SessionLocal() as db:
        res = await db.execute(text(query), params)
        row = res.first()

    if not row:
        return None, run_id

    filename = row[0]
    resolved_run_id = row[1] or run_id
    return filename, resolved_run_id


async def _fetch_org_id_for_workspace(workspace_id: Optional[int]) -> Optional[int]:
    if not SessionLocal or workspace_id is None:
        return None

    async with SessionLocal() as db:
        res = await db.execute(text("SELECT organization_id FROM workspaces WHERE id=:ws_id LIMIT 1"), {"ws_id": workspace_id})
        return res.scalar()


async def db_insert_llm_call(
    workspace_id: Optional[int],
    guide_id: Optional[int],
    purpose: str,
    model: str,
    prompt_text: str,
    response_text: str,
    tokens_prompt: int,
    tokens_completion: int,
    cost_cents: float,
    latency_ms: int,
    success: bool,
    step_id: Optional[int] = None,
    run_id: Optional[str] = None,
) -> None:
    if not SessionLocal:
        return

    now = dt.datetime.utcnow()
    org_id = await _fetch_org_id_for_workspace(workspace_id)

    # Use run_id as guide_id fallback if guide_id is not provided
    # (guide_id column stores the run identifier)
    effective_guide_id = guide_id

    async with SessionLocal() as db:
        await db.execute(
            text(
                """
                INSERT INTO llm_calls (
                    organization_id,
                    workspace_id,
                    guide_id,
                    step_id,
                    purpose,
                    model,
                    prompt_text,
                    response_text,
                    tokens_prompt,
                    tokens_completion,
                    cost_cents,
                    latency_ms,
                    success,
                    created_at
                )
                VALUES (
                    :organization_id,
                    :workspace_id,
                    :guide_id,
                    :step_id,
                    :purpose,
                    :model,
                    :prompt_text,
                    :response_text,
                    :tokens_prompt,
                    :tokens_completion,
                    :cost_cents,
                    :latency_ms,
                    :success,
                    :created_at
                )
                """
            ),
            {
                "organization_id": org_id,
                "workspace_id": workspace_id,
                "guide_id": effective_guide_id,
                "step_id": step_id,
                "purpose": purpose,
                "model": model,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "cost_cents": cost_cents,
                "latency_ms": latency_ms,
                "success": 1 if success else 0,
                "created_at": now,
            },
        )
        await db.commit()
