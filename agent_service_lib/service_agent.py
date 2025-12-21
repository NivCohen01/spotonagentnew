from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from browser_use import Agent, Browser
from browser_use.agent.task_optimizer import AgentTaskOptimizer, TaskOptimizationRequest
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.messages import ContentPartTextParam, SystemMessage, UserMessage
from browser_use.screenshots.models import ActionScreenshotSettings
from sqlalchemy import text

from .service_browser import _safe_browser_stop, launch_chrome
from .service_config import DEFAULT_AUTHOR_ID, OPENAI_MODEL, SCREENSHOTS_BASE
from .service_db import SessionLocal, db_finish_run, db_insert_run, db_update_guide_from_result
from .service_logging import run_id_ctx
from .service_models import (
    ActionScreenshotOptions,
    ActionTraceEntry,
    GuideOutput,
    GuideOutputWithEvidence,
    GuideStepWithEvidence,
    EvidenceEvent,
    SessionState,
    StartReq,
)
from .service_trace import (
    _attach_images_by_evidence,
    _build_evidence_table,
    _build_action_trace,
    _coerce_guide_output_dict,
    _ensure_json_text,
    _format_evidence_table,
    _discover_screenshots,
    _shape_steps_with_placeholders,
    _summarize_action_trace,
)


def _draft_steps_to_text(draft_steps: list[Any]) -> str:
    lines: list[str] = []
    for idx, step in enumerate(draft_steps, 1):
        if isinstance(step, dict):
            desc = step.get("description") or step.get("text") or step.get("title") or json.dumps(step)
        else:
            desc = str(step)
        lines.append(f"{idx}. {desc}")
        if idx >= 25:
            lines.append("... truncated ...")
            break
    return "\n".join(lines)


def _fallback_guide_from_evidence(task: str, draft: dict | None, evidence_table: list[EvidenceEvent]) -> GuideOutputWithEvidence:
    title = (draft or {}).get("title") or task
    links = (draft or {}).get("links") or []
    notes = (draft or {}).get("notes")
    success = bool((draft or {}).get("success", True))
    steps: list[GuideStepWithEvidence] = []

    for i, ev in enumerate(evidence_table, 1):
        desc = ev.label or ev.element_text or ev.element_tag or (ev.action_types[0] if ev.action_types else "Step")
        steps.append(
            GuideStepWithEvidence(
                number=i,
                description=str(desc),
                pageUrl=ev.page_url,
                evidence_ids=[ev.evidence_id],
                primary_evidence_id=ev.evidence_id,
                images=[],
            )
        )

    if not steps and draft:
        for i, step in enumerate(draft.get("steps") or [], 1):
            if isinstance(step, dict):
                desc = step.get("description") or step.get("text") or step.get("title") or ""
                page_url = step.get("pageUrl") or step.get("page_url")
            else:
                desc = str(step)
                page_url = None
            steps.append(
                GuideStepWithEvidence(
                    number=i,
                    description=desc,
                    pageUrl=page_url,
                    evidence_ids=[],
                    primary_evidence_id=None,
                    images=[],
                )
            )

    return GuideOutputWithEvidence(title=title, steps=steps, links=links, notes=notes, success=success)


async def _generate_final_guide_with_evidence(
    task: str,
    draft: dict | None,
    evidence_table: list[EvidenceEvent],
    model_name: str,
) -> GuideOutputWithEvidence:
    llm = ChatOpenAI(model=model_name)
    draft = draft or {}
    evidence_table = evidence_table or []

    system_text = (
        "You are preparing the final user-facing guide from a browser automation run.\n"
        "Return strict JSON only.\n"
        "Constraints:\n"
        "- Use only evidence_ids from the provided evidence table.\n"
        "- Each evidence_id may appear in at most one guide step; merge steps if they would reuse the same evidence.\n"
        "- primary_evidence_id must be null or one of evidence_ids (prefer click evidence).\n"
        "- Combine related micro-actions from the same screen/evidence into a single clear instruction when it improves readability.\n"
        "- Keep step numbers sequential starting from 1 and use concise imperative descriptions."
    )

    draft_steps_text = _draft_steps_to_text(draft.get("steps") or [])
    evidence_text = _format_evidence_table(evidence_table) or "No evidence captured. Leave evidence_ids empty."

    user_payload = (
        f"User task: {task}\n"
        f"Draft title: {draft.get('title') or task}\n"
        f"Draft success flag: {draft.get('success', True)}\n"
        f"Draft steps:\n{draft_steps_text or 'None'}\n\n"
        f"Evidence table:\n{evidence_text}\n\n"
        "Requirements:\n"
        "- Every step MUST list evidence_ids (empty only when evidence table is empty).\n"
        "- Never repeat an evidence_id across steps; merge content instead.\n"
        "- primary_evidence_id must be either null or one of evidence_ids.\n"
        "- Prefer click evidence when picking primary_evidence_id because those have screenshots."
    )

    try:
        result = await llm.ainvoke(
            [
                SystemMessage(content=[ContentPartTextParam(text=system_text)]),
                UserMessage(content=[ContentPartTextParam(text=user_payload)]),
            ],
            output_format=GuideOutputWithEvidence,
        )

        completion = result.completion
        return GuideOutputWithEvidence(
            title=completion.title,
            steps=list(completion.steps),
            links=completion.links,
            notes=completion.notes,
            success=completion.success,
        )
    except Exception as exc:  # pragma: no cover - safeguard
        logging.getLogger("service").warning("Guide generation with evidence failed, using fallback | %r", exc)
        return _fallback_guide_from_evidence(task, draft, evidence_table)


class Session:
    def __init__(self, session_id: str, req: StartReq):
        self.id = session_id
        self.task = req.task
        self.start_url = req.start_url
        self.headless = req.headless
        self.workspace_id = req.workspace_id
        self.guide_family_id: Optional[int] = req.guide_family_id
        self.guide_id: Optional[int] = req.guide_id
        self.guide_family_key: Optional[str] = req.guide_family_key

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
        self.action_trace: Optional[list[ActionTraceEntry]] = None
        self.action_trace_summary: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None

        self.screenshots_dir: Path = SCREENSHOTS_BASE / self.id / "images"
        self.chrome_proc: Optional[asyncio.subprocess.Process] = None
        self.cdp_port: Optional[int] = None
        self.cdp_ws_url: Optional[str] = None

        self.log_lines: list[str] = []
        self.log_seq: int = 0
        self.finished_at: Optional[dt.datetime] = None

        self.guide_slug: Optional[str] = None

    def to_status(self, queue_position: Optional[int] = None) -> Dict[str, Any]:
        return {
            "session_id": self.id,
            "state": self.state,
            "task": self.task,
            "start_url": self.start_url,
            "final_response": self.final_response,
            "result": self.result_only,
            "error": self.error,
            "screenshots_dir": str(self.screenshots_dir) if self.screenshots_dir else None,
            "queue_position": queue_position,
            "action_trace": self.action_trace,
            "action_trace_summary": self.action_trace_summary,
        }

    def to_result(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "final_response": self.final_response,
            "result": self.result_only,
            "error": self.error,
            "screenshots_dir": str(self.screenshots_dir) if self.screenshots_dir else None,
            "action_trace": self.action_trace,
            "action_trace_summary": self.action_trace_summary,
        }


sessions: Dict[str, Session] = {}


async def run_session(sess: Session):
    token = run_id_ctx.set(sess.id)
    log = logging.getLogger("service")
    browser: Optional[Browser] = None

    sess.state = "starting"

    with contextlib.suppress(Exception):
        await db_insert_run(sess)

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
                optimized_resp = await optimizer.optimize_async(TaskOptimizationRequest(task=optimized_task, mode="regular"))
                if optimized_resp.optimized_task.strip():
                    optimized_task = optimized_resp.optimized_task.strip()
            except Exception as exc:
                log.info("optimizer failed; using original task | %r", exc)

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
            device_type=sess.device_type,
            viewport_width=sess.viewport_width,
            viewport_height=sess.viewport_height,
        )

        sess.state = "running"
        log.info("agent running...")

        result = await agent.run()
        sess.action_trace = _build_action_trace(result)
        sess.action_trace_summary = _summarize_action_trace(sess.action_trace) if sess.action_trace else None

        sess.final_response = _ensure_json_text(result)

        evidence_images = []
        evidence_table: list[EvidenceEvent] = []
        if sess.action_screenshots_enabled:
            with contextlib.suppress(Exception):
                evidence_images = _discover_screenshots(sess.screenshots_dir)
        with contextlib.suppress(Exception):
            evidence_table = _build_evidence_table(sess.action_trace or [], evidence_images)

        if evidence_table:
            log.info("Evidence table:\n%s", _format_evidence_table(evidence_table))
        else:
            log.info("Evidence table empty (no captured screenshots or trace-derived evidence).")

        if evidence_table:
            for ev in evidence_table:
                log.info("Evidence item | id=%s actions=%s best_image=%s", ev.evidence_id, ev.action_types, ev.best_image)

        extracted = _coerce_guide_output_dict(result)
        if extracted is None:
            with contextlib.suppress(Exception):
                extracted = _coerce_guide_output_dict(json.loads(sess.final_response))

        draft = _shape_steps_with_placeholders(extracted) if extracted is not None else None

        final_guide = await _generate_final_guide_with_evidence(
            task=sess.task, draft=draft, evidence_table=evidence_table, model_name=OPENAI_MODEL
        )

        final_with_images = _attach_images_by_evidence(final_guide, evidence_table)
        if not final_with_images and isinstance(final_guide, GuideOutputWithEvidence):
            final_with_images = final_guide.model_dump()
        elif not final_with_images:
            final_with_images = draft or {}
        if sess.action_trace_summary:
            final_with_images = dict(final_with_images)
            final_with_images["action_trace"] = sess.action_trace_summary
        if evidence_table:
            final_with_images = dict(final_with_images)
            final_with_images["evidence"] = [ev.model_dump() for ev in evidence_table]

        sess.result_only = _ensure_json_text(final_with_images)

        if final_with_images.get("steps"):
            lines = [
                f"step {step.get('number')}: evidence_ids={step.get('evidence_ids')} primary={step.get('primary_evidence_id')} image={step.get('images') or []}"
                for step in final_with_images.get("steps", [])
            ]
            log.info("Guide step evidence mapping:\n%s", "\n".join(lines))

        with contextlib.suppress(Exception):
            await db_update_guide_from_result(sess, final_with_images)

        sess.state = "done"
        log.info("agent done")

    except asyncio.CancelledError:
        sess.state = "stopped"
        logging.getLogger("service").warning("agent stopped")
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
        raise
    except Exception:
        import traceback

        sess.error = traceback.format_exc()
        sess.state = "error"
        logging.getLogger("service").error("ERROR:\n%s", sess.error)
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
