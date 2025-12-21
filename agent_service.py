from __future__ import annotations
"""
FastAPI entrypoint that wires together queue management, agent execution, and database helpers.

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
import datetime as dt
import logging
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from browser_use.llm.openai.chat import ChatOpenAI
from agent_service_lib.service_agent import Session, sessions, run_session
from agent_service_lib.service_config import (
    ALLOWED_ORIGINS,
    API_KEY,
    APP_NAME,
    APP_VERSION,
    BIND,
    CHROME_BIN,
    DB_URL,
    DEFAULT_AUTHOR_ID,
    MAX_CONCURRENCY,
    OPENAI_MODEL,
    PORT,
)
from agent_service_lib.service_db import (
    CREATE_TABLES_SQL,
    SessionLocal,
    db_fetch_action_trace,
    derive_guide_family_key,
    ensure_family_and_variant_guide,
    run_migrations,
    engine,
)
from agent_service_lib.service_logging import PerRunDBAndMemoryHandler, RunIdFilter, _hijack_library_loggers
from agent_service_lib.service_models import GenerateVideoResponse, OptimizeReq, OptimizeResp, ResultPayload, StartReq, Status
from agent_service_lib.service_optimize import OptimizeAgentPrompt
from agent_service_lib.service_queue import QueueManager
from agent_service_lib.service_replay import replay_action_trace_to_video
from agent_service_lib.service_trace import _to_public_image_path

_allow_credentials = ALLOWED_ORIGINS != ["*"]

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

queue_mgr = QueueManager(MAX_CONCURRENCY, run_session, lambda sid: sessions.get(sid))


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.on_event("startup")
async def on_startup():
    if engine:
        async with engine.begin() as conn:
            for stmt in CREATE_TABLES_SQL.strip().split(";\n\n"):
                s = stmt.strip().rstrip(";")
                if s:
                    await conn.execute(text(s))
            await run_migrations(conn)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(logging.StreamHandler())
    if not any(isinstance(f, RunIdFilter) for f in root_logger.filters):
        root_logger.addFilter(RunIdFilter())
    if not any(isinstance(h, PerRunDBAndMemoryHandler) for h in root_logger.handlers):
        handler = PerRunDBAndMemoryHandler(session_lookup=lambda rid: sessions.get(rid), session_factory=SessionLocal)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

    _hijack_library_loggers()
    await queue_mgr.start()


@app.on_event("shutdown")
async def on_shutdown():
    with contextlib.suppress(Exception):
        await queue_mgr.stop()
    for sess in list(sessions.values()):
        with contextlib.suppress(Exception):
            if sess.chrome_proc and sess.chrome_proc.returncode is None:
                sess.chrome_proc.kill()


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
    if req.workspace_id is None:
        raise HTTPException(status_code=400, detail="workspace_id (or ws_id) is required")
    session_id = uuid.uuid4().hex[:12]
    sess = Session(session_id=session_id, req=req)

    family_key = req.guide_family_key or derive_guide_family_key(req.workspace_id, req.task, req.start_url)
    sess.guide_family_key = family_key
    try:
        ensured = await ensure_family_and_variant_guide(
            workspace_id=req.workspace_id,
            family_key=family_key,
            device_type=req.device_type,
            run_id=session_id,
            title=req.task or "Generating guide",
            visibility="private",
            status="generating",
            guide_family_id=req.guide_family_id,
            existing_guide_id=req.guide_id,
        )
    except Exception as exc:
        logging.getLogger("service").error("Failed to ensure guide family/variant at start | %r", exc)
        raise HTTPException(status_code=500, detail="failed to create guide family/variant")
    if ensured:
        sess.guide_family_id, sess.guide_id, sess.guide_slug = ensured
        sess.guide_family_key = family_key
        logging.getLogger("service").info(
            "Session %s linked to guide_family_id=%s guide_id=%s device=%s",
            session_id,
            sess.guide_family_id,
            sess.guide_id,
            req.device_type,
        )

    sessions[session_id] = sess
    pos = await queue_mgr.enqueue(session_id)
    return Status(**sess.to_status(queue_position=pos))


@app.get("/sessions/{sid}", response_model=Status, dependencies=[Depends(require_api_key)])
async def get_status(sid: str):
    sess = sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    pos = await queue_mgr.position(sid) if sess.state == "queued" else None
    return Status(**sess.to_status(queue_position=pos))


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
    return ResultPayload(**sess.to_result())


@app.post("/sessions/{sid}/generate-video", response_model=GenerateVideoResponse, dependencies=[Depends(require_api_key)])
async def generate_video(sid: str):
    sess = sessions.get(sid)
    if sess and sess.action_trace:
        trace_entries = sess.action_trace
    else:
        trace_entries = await db_fetch_action_trace(sid)

    if not trace_entries:
        raise HTTPException(status_code=404, detail="action_trace not found for session")

    video_path, applied, skipped = await replay_action_trace_to_video(
        sid,
        trace_entries,
        device_type=sess.device_type if sess else "desktop",
        viewport_width=sess.viewport_width if sess else None,
        viewport_height=sess.viewport_height if sess else None,
    )
    return GenerateVideoResponse(
        session_id=sid,
        video_path=str(video_path) if video_path else None,
        actions_replayed=applied,
        skipped_actions=skipped,
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
    sess = sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")

    if sess.state == "queued":
        removed = await queue_mgr.remove_if_present(sid)
        if removed:
            sess.state = "stopped"
            with contextlib.suppress(Exception):
                if SessionLocal:
                    async with SessionLocal() as db:
                        condition_sql = "id=:guide_id" if sess.guide_id else "run_id=:run_id"
                        await db.execute(
                            text(
                                f"""
                            UPDATE guides
                            SET status='draft', updated_at=:now, updated_by=:uid
                            WHERE {condition_sql}
                        """
                            ),
                            {
                                "now": dt.datetime.utcnow(),
                                "uid": DEFAULT_AUTHOR_ID,
                                "run_id": sess.id,
                                "guide_id": sess.guide_id,
                            },
                        )
                        await db.commit()
            return Status(**sess.to_status(queue_position=None))

    with contextlib.suppress(Exception):
        if sess.chrome_proc and sess.chrome_proc.returncode is None:
            sess.chrome_proc.terminate()
    sess.state = "stopped"
    with contextlib.suppress(Exception):
        if SessionLocal:
            async with SessionLocal() as db:
                condition_sql = "id=:guide_id" if sess.guide_id else "run_id=:run_id"
                await db.execute(
                    text(
                        f"""
                    UPDATE guides
                    SET status='draft', updated_at=:now, updated_by=:uid
                    WHERE {condition_sql}
                """
                    ),
                    {
                        "now": dt.datetime.utcnow(),
                        "uid": DEFAULT_AUTHOR_ID,
                        "run_id": sess.id,
                        "guide_id": sess.guide_id,
                    },
                )
                await db.commit()
    return Status(**sess.to_status(queue_position=None))


@app.post("/optimize", response_model=OptimizeResp, dependencies=[Depends(require_api_key)])
async def optimize_prompt(req: OptimizeReq):
    model_name = req.llm_model or OPENAI_MODEL
    optimizer = OptimizeAgentPrompt(
        task=req.task,
        mode=req.mode,
        ptype=req.ptype,
        llm=ChatOpenAI(model=model_name),
        custom_instructions=req.custom_instructions,
    )
    optimized = await optimizer.optimize_async()
    out = optimized if isinstance(optimized, str) and optimized.strip() else req.task
    return OptimizeResp(original=req.task, optimized=out)
