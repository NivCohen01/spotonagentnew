from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import time
from textwrap import dedent
from pathlib import Path
from typing import Any, Dict, Optional
import re

from browser_use import Agent, Browser
from browser_use.agent.task_optimizer import AgentTaskOptimizer, TaskOptimizationRequest
from browser_use.agent.views import ActionResult
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.messages import ContentPartTextParam, SystemMessage, UserMessage
from browser_use.screenshots.models import ActionScreenshotSettings
from browser_use.tools.service import Tools
from pydantic import BaseModel, Field
from sqlalchemy import text
from browser_use.tokens.service import TokenCost

from .mailbox_service import DEFAULT_DOMAIN
from .service_agent_auth_utils import (
    TaskIntent,
    TaskCredentials,
    PageStateClassification,
    _sanitize_credentials_payload,
    _register_mailbox_actions,
    _is_micro_action,
)
from .service_agent_guardrails import install_auth_guardrails
from .service_browser import _safe_browser_stop, launch_chrome
from .service_config import DEFAULT_AUTHOR_ID, OPENAI_MODEL, SCREENSHOTS_BASE
from .service_db import SessionLocal, db_finish_run, db_insert_llm_call, db_insert_run, db_update_guide_from_result
from .service_logging import run_id_ctx
from .service_models import (
    ActionScreenshotOptions,
    ActionTraceEntry,
    GuideOutput,
    GuideOutputWithEvidence,
    GuideStepWithEvidence,
    GuideSubStepWithEvidence,
    EvidenceEvent,
    SessionState,
    StartReq,
)
from .service_trace import (
    _apply_relevance_labels,
    _attach_images_by_evidence,
    _build_evidence_table,
    _build_action_trace,
    _coerce_guide_output_dict,
    _ensure_json_text,
    _filter_relevant_actions,
    _format_evidence_table,
    _discover_screenshots,
    _inject_auth_events,
    _shape_steps_with_placeholders,
    _summarize_action_trace,
)
import json


_SUBMIT_TERMS: tuple[str, ...] = ()
DEFAULT_AGENT_PROFILE = {
    "first_name": "Pathix",
    "last_name": "Agent",
    "full_name": "Pathix Agent",
    "username": "pathix-agent",
    "display_name": "Pathix Agent",
}


def _regroup_micro_steps(guide: GuideOutputWithEvidence) -> GuideOutputWithEvidence:
    """
    Fallback regrouping to collapse 3+ consecutive micro-actions on the same page into one parent with sub_steps.
    Example: Enter First Name / Enter Last Name / Enter Email / Click Submit -> one parent "Complete the form and submit it" with sub_steps.
    """
    steps = list(guide.steps or [])
    if len(steps) < 3:
        return guide

    def same_page(a: GuideStepWithEvidence, b: GuideStepWithEvidence) -> bool:
        return (a.pageUrl or None) == (b.pageUrl or None)

    new_steps: list[GuideStepWithEvidence] = []
    idx = 0
    while idx < len(steps):
        step = steps[idx]
        if not _is_micro_action(step.description):
            new_steps.append(step)
            idx += 1
            continue

        run = [step]
        j = idx + 1
        while j < len(steps) and _is_micro_action(steps[j].description) and same_page(step, steps[j]):
            run.append(steps[j])
            j += 1

        if len(run) < 3:
            new_steps.extend(run)
            idx = j
            continue

        run_page = step.pageUrl
        sub_steps: list[GuideSubStepWithEvidence] = []
        all_evidence: list[int] = []
        primary_candidate: Optional[int] = None
        has_submit = any(any(term in (s.description or "").lower() for term in _SUBMIT_TERMS) for s in run)

        for k, s in enumerate(run, 1):
            sub = GuideSubStepWithEvidence(
                number=k,
                description=s.description,
                pageUrl=s.pageUrl,
                evidence_ids=list(s.evidence_ids or []),
                primary_evidence_id=s.primary_evidence_id,
                images=[],
            )
            sub_steps.append(sub)
            all_evidence.extend(list(s.evidence_ids or []))
            if primary_candidate is None and s.primary_evidence_id is not None:
                primary_candidate = s.primary_evidence_id

        # Preserve submit action inside the grouped sub_steps
        summary = "Complete the form and submit it." if has_submit else "Fill in the required details."
        parent = GuideStepWithEvidence(
            number=0,
            description=summary,
            pageUrl=run_page,
            evidence_ids=list(dict.fromkeys(all_evidence)),
            primary_evidence_id=primary_candidate,
            images=[],
            sub_steps=sub_steps,
        )
        new_steps.append(parent)
        idx = j

    for n, st in enumerate(new_steps, 1):
        st.number = n
        if st.sub_steps:
            for m, sub in enumerate(st.sub_steps, 1):
                sub.number = m
    return GuideOutputWithEvidence(
        title=guide.title,
        steps=new_steps,
        links=guide.links,
        notes=guide.notes,
        success=guide.success,
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


class TraceRelevanceLabels(BaseModel):
    labels: list[int]


def _flatten_guide_steps(guide: GuideOutputWithEvidence | dict | None) -> list[str]:
    if not guide:
        return []
    try:
        guide_dict = guide.model_dump() if isinstance(guide, GuideOutputWithEvidence) else dict(guide)
    except Exception:
        return []
    steps = guide_dict.get("steps") or []
    flat: list[str] = []
    for step in steps:
        if isinstance(step, dict):
            desc = step.get("description") or step.get("text") or step.get("title")
        else:
            desc = str(step)
        if desc:
            flat.append(str(desc))
    return flat


def _summarize_action_for_rank(entry: ActionTraceEntry) -> dict[str, Any]:
    payload = {
        "order": entry.order,
        "step": entry.step,
        "action": entry.action,
        "value": entry.value,
        "page_url": entry.page_url,
        "element_text": entry.element_text,
        "element_tag": entry.element_tag,
    }
    if entry.params:
        # keep params small and JSON-safe
        compact = {}
        for key, value in entry.params.items():
            if value is None:
                continue
            text = str(value)
            compact[key] = text[:120]
        if compact:
            payload["params"] = compact
    return payload


async def _rank_action_trace(
    entries: list[ActionTraceEntry],
    *,
    task: str,
    guide: GuideOutputWithEvidence | dict | None,
    llm: ChatOpenAI,
    batch_size: int = 35,
) -> list[int]:
    if not entries:
        return []

    guide_steps = _flatten_guide_steps(guide)
    labels: list[int] = []
    system_text = dedent(
        """
        You label whether each action was relevant to reaching the final goal.

        Rules:
        - Return STRICT JSON only.
        - "labels" must be a list of 0/1 with the same length as the input actions.
        - If unsure, use 1 (relevant) to avoid removing required steps.
        - Consider the task and final guide steps to decide which actions were actually needed.
        """
    ).strip()

    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        actions_payload = [_summarize_action_for_rank(e) for e in batch]
        user_text = json.dumps(
            {
                "task": task,
                "final_guide_steps": guide_steps,
                "actions": actions_payload,
            },
            ensure_ascii=False,
        )
        try:
            result = await llm.ainvoke(
                [
                    SystemMessage(content=[ContentPartTextParam(text=system_text)]),
                    UserMessage(content=[ContentPartTextParam(text=user_text)]),
                ],
                output_format=TraceRelevanceLabels,
            )
            batch_labels = list(result.completion.labels or [])
        except Exception as exc:
            logging.getLogger("service").warning("Trace relevance ranking failed; defaulting to 1s | %r", exc)
            batch_labels = [1] * len(batch)

        if len(batch_labels) != len(batch):
            logging.getLogger("service").warning(
                "Trace relevance label count mismatch (got %s, expected %s). Defaulting to 1s.",
                len(batch_labels),
                len(batch),
            )
            batch_labels = [1] * len(batch)

        labels.extend([1 if int(val or 0) else 0 for val in batch_labels])

    return labels


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
    llm: ChatOpenAI,
    include_auth: bool = False,
) -> GuideOutputWithEvidence:
    draft = draft or {}
    evidence_table = evidence_table or []

    base_scope = dedent("""
        You are a senior technical writer producing the final, user-facing guide from a browser automation run.

        OUTPUT FORMAT (HARD RULES)
        - Return STRICT JSON ONLY. No markdown. No commentary.
        - The JSON MUST match the required schema exactly.
        - `steps` is an array of step objects. Each step can include `sub_steps` (one level only). Each sub_step uses the same fields as a step but does not nest further.
        - Always set `images: []` for every step and sub_step. Images are attached server-side via evidence_ids.

        PRIMARY GOAL
        Create a polished, minimal, top-tier support article that helps a real user complete the task quickly.

        CRITICAL SCOPE POLICY (HARD RULES)
        - If authentication steps are out of scope, do NOT include login/MFA/signup/password reset/credential entry steps. Treat any "Start url: ..." text as tooling context, not a user step.
        - If authentication steps are in scope, you may include them when relevant but never reveal secrets (use placeholders).
        - Default starting point is post-authenticated only when auth steps are out of scope; otherwise follow the run flow.
        - Credentials included in the task (username/password/OTP hints) are EXECUTION CONTEXT ONLY.
        They do NOT mean the user asked to document login.
        Only include authentication steps if the user explicitly asked "how to log in/sign up/reset password/verify OTP".
        - If authentication is out of scope, assume the user is already signed in and start from the first in-app page
        (e.g., /dashboard), even if the run started at /auth and the draft includes login/OTP steps.
        - When auth is out of scope, you MUST ignore any draft steps and any evidence items whose pageUrl is an auth page
        (/auth, /login, /signin, /signup, /mfa, /otp, /verify, /password-reset).

        STEP WRITING QUALITY BAR
        - Keep it short: usually 3-8 top-level steps.
        - Form fields and multiple inputs MUST NOT be separate parent steps; they MUST be sub_steps under one parent summary (e.g., "Fill in the form and submit it").
        - Use sub_steps when 3+ micro-actions happen on the same screen (form fills, multiple toggles, multi-click sequences). The parent step description should summarize the cluster; each sub_step is one concrete action.
        - One action per step or sub_step. Do not combine independent actions in a single step.
        - Never create a parent step with only ONE sub_step. If there is only one action, keep it as a parent step and omit sub_steps.
        - Do not repeat the same sentence in a parent step and its sub_step. The parent must summarize; the sub_step must be a concrete action.
        - If the evidence table shows 2+ distinct evidence_ids with screenshots, produce at least 2 top-level steps unless they are truly the same action on the same screen.
        - Be direct: avoid filler ("navigate to", "access"). Prefer: "In the left sidebar, click 'Chat'."
        - Use exact visible UI text in quotes ("Chat", "New message", "Send").
        - For icon-only controls, name the function and add a short hint: "Send (paper-plane icon)".
        - Do NOT include secrets or credentials. For text entry, use placeholders like <your message> or <your email>.
        - Do NOT add "Step X" or similar inside descriptions.
        - sub_steps is OPTIONAL and may be empty.

        GROUNDING REQUIREMENTS (HARD RULES)
        - Steps must be supported by the evidence table or draft. Do not invent UI labels or controls.
        - Banned hedging words/phrases in user-facing text: "usually", "typically", "might", "may", "often", "likely",
          "around", "approximately", "roughly", "or a similar option", "or equivalent", "if needed".
        - Do not use vague placeholders like "appropriate button", "checkmark", "save icon", or "some menu".
        - If a required control or label is not identifiable from evidence AND is not present in the draft, set success=false and add a short note explaining what could not be verified.
        - If evidence labels are generic or low-information (e.g., "button button", "div div[4]"), prefer the draft step text and leave evidence_ids empty rather than dropping the step.
        - Do not collapse the guide to a single step when the draft contains multiple distinct actions; keep separate steps for distinct actions even if evidence is sparse.

        EVIDENCE CONSTRAINTS (HARD RULES)
        - Use ONLY evidence_ids from the provided evidence table. Do not invent IDs.
        - If a step has pageUrl, ONLY assign evidence_ids whose page_url matches that pageUrl.
        - primary_evidence_id must be null or one of the step's evidence_ids.
        - Top-level steps should prefer unique evidence_ids across different parent steps to preserve screenshot alignment.
        - Sub-steps may reuse evidence_ids (including the parent's primary_evidence_id) when needed; reuse is expected for forms on the same screen.
    """).strip()

    if include_auth:
        system_text = base_scope
    else:
        system_text = base_scope + dedent("""

        EVIDENCE + SCOPE INTERACTION (HARD RULES)
        - Treat authentication evidence as OUT OF SCOPE unless the task explicitly asked for it.
        * Do NOT write login/signup/password reset/MFA steps.
        * Do NOT assign authentication-related evidence_ids to any step.
        - HOWEVER: for any in-scope step, you MUST attach relevant evidence_ids when such evidence exists.
        * evidence_ids may be empty ONLY if there is no matching in-scope evidence for that step.
        * If the evidence table contains at least one IN-SCOPE item with best_image, at least one step MUST reference it.

        CONSISTENCY CHECK (MANDATORY)
        - When authentication is out of scope:
          * `steps` MUST NOT mention login, credentials, passwords, MFA, or the login page.
          * Any authentication-related evidence may be ignored and must not force login steps.
        - If any step violates this policy, regenerate a compliant guide.
        """).strip()

    draft_steps_text = _draft_steps_to_text(draft.get("steps") or [])
    evidence_text = _format_evidence_table(evidence_table) or "No evidence captured. Leave evidence_ids empty."

    user_payload = dedent(f"""
        User task: {task}

        Draft title: {draft.get("title") or task}
        Draft success flag: {draft.get("success", True)}
        Draft steps:
        {draft_steps_text or "None"}

        Evidence table:
        {evidence_text}

        Reminder (must follow):
        - Only use evidence_ids from the evidence table.
        - 3+ micro-actions (enter/type/select/check/toggle/click submit/save/send) on the SAME pageUrl => one parent summary with sub_steps; do NOT leave them as separate parent steps.
        - Each PARENT step should include at least one evidence_id when evidence exists; avoid reusing the same evidence_id across different parent steps when possible.
        - sub_steps may reuse evidence_ids (including the parent's primary_evidence_id). Set them when you know the mapping; they may be empty if no specific sub-step evidence.
        - primary_evidence_id must be either null or one of evidence_ids.
        - Always set images: [] for steps and sub_steps.
        - Do not invent UI labels or controls; avoid vague placeholders like "appropriate button" or "checkmark".
        - Do not use hedging words like "usually", "typically", "might", or "may".
    """).strip()

    try:
        result = await llm.ainvoke(
            [
                SystemMessage(content=[ContentPartTextParam(text=system_text)]),
                UserMessage(content=[ContentPartTextParam(text=user_payload)]),
            ],
            output_format=GuideOutputWithEvidence,
        )

        completion = result.completion
        regrouped = _regroup_micro_steps(
            GuideOutputWithEvidence(
                title=completion.title,
                steps=list(completion.steps),
                links=completion.links,
                notes=completion.notes,
                success=completion.success,
            )
        )
        return regrouped
    except Exception as exc:  # pragma: no cover - safeguard
        logging.getLogger("service").warning("Guide generation with evidence failed, using fallback | %r", exc)
        return _fallback_guide_from_evidence(task, draft, evidence_table)


def _messages_to_prompt_text(messages: list[Any]) -> str:
    try:
        parts: list[str] = []
        for m in messages or []:
            content = getattr(m, "content", None)
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        txt = c.get("text")
                        if txt:
                            parts.append(str(txt))
                    elif hasattr(c, "text"):
                        parts.append(str(getattr(c, "text")))
            elif isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)[:500000]
    except Exception:
        return ""


def _response_to_text(response: Any) -> str:
    if response is None:
        return ""
    with contextlib.suppress(Exception):
        comp = getattr(response, "completion", None)
        if comp is not None:
            if hasattr(comp, "model_dump"):
                return json.dumps(comp.model_dump(), ensure_ascii=False)[:5000]
            return str(comp)[:5000]
    return str(response)[:5000]


def _wrap_llm_with_logging(
    llm: ChatOpenAI,
    sess: "Session",
    purpose: str,
    token_cost: TokenCost,
    guide_id: Optional[int],
) -> ChatOpenAI:
    original = llm.ainvoke

    async def wrapped(messages, *args, **kwargs):
        start = time.perf_counter()
        usage = None
        response = None
        success = True
        try:
            response = await original(messages, *args, **kwargs)
            usage = getattr(response, "usage", None)
            return response
        except Exception:
            success = False
            raise
        finally:
            try:
                latency_ms = int((time.perf_counter() - start) * 1000)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                cost_cents = 0.0
                if usage:
                    with contextlib.suppress(Exception):
                        cost = await token_cost.calculate_cost(llm.model, usage)
                        if cost:
                            cost_cents = float((cost.prompt_cost or 0) + (cost.completion_cost or 0)) * 100

                prompt_text = _messages_to_prompt_text(messages)
                response_text = _response_to_text(response)

                # Get step_id for generation calls, None for non-step calls (planning, summarization, etc.)
                step_id = None
                if purpose == "generation":
                    step_id = getattr(sess, "current_step", None)

                await db_insert_llm_call(
                    workspace_id=sess.workspace_id,
                    guide_id=guide_id,
                    purpose=purpose,
                    model=getattr(llm, "model", ""),
                    prompt_text=prompt_text,
                    response_text=response_text,
                    tokens_prompt=prompt_tokens,
                    tokens_completion=completion_tokens,
                    cost_cents=cost_cents,
                    latency_ms=latency_ms,
                    success=success,
                    step_id=step_id,
                    run_id=sess.id,
                )
            except Exception:
                pass

    llm.ainvoke = wrapped  # type: ignore
    return llm


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

        self.generated_credentials: Optional[dict[str, str]] = None
        self.generated_credentials_created_at: Optional[dt.datetime] = None
        self.signup_intent: bool = False
        self.user_credentials: Optional[dict[str, str]] = None
        self.intent: Optional[TaskIntent] = None
        self.otp_attempted_urls: set[str] = set()
        self.verification_attempted_urls: set[str] = set()
        self.pending_verification_url: Optional[str] = None
        self.agent_profile: dict[str, str] = dict(DEFAULT_AGENT_PROFILE)

        self.state: SessionState = "queued"
        self.final_response: Optional[str] = None
        self.result_only: Optional[str] = None
        self.action_trace: Optional[list[ActionTraceEntry]] = None
        self.action_trace_summary: Optional[dict[str, Any]] = None
        self.action_trace_full: Optional[list[ActionTraceEntry]] = None
        self.auth_events: list[dict[str, Any]] = []
        self._auth_event_keys: set[str] = set()
        self.current_step: Optional[int] = None
        self.current_url: Optional[str] = None
        self.error: Optional[str] = None
        self.page_state: Optional[PageStateClassification] = None  # Page state from agent output

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
    token_cost = TokenCost(include_cost=True)
    with contextlib.suppress(Exception):
        await token_cost.initialize()

    sess.state = "starting"

    with contextlib.suppress(Exception):
        await db_insert_run(sess)
    guide_id_for_run = sess.guide_id

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
                opt_llm = _wrap_llm_with_logging(ChatOpenAI(model=OPENAI_MODEL), sess, "planning", token_cost, guide_id_for_run)
                optimizer = AgentTaskOptimizer(llm=opt_llm)
                optimized_resp = await optimizer.optimize_async(TaskOptimizationRequest(task=optimized_task, mode="regular"))
                if optimized_resp.optimized_task.strip():
                    optimized_task = optimized_resp.optimized_task.strip()
            except Exception as exc:
                log.info("optimizer failed; using original task | %r", exc)

        # Intent classification (language-agnostic, no keyword heuristics)
        try:
            intent_llm = _wrap_llm_with_logging(ChatOpenAI(model=OPENAI_MODEL), sess, "intent", token_cost, guide_id_for_run)
            intent_system = SystemMessage(
                content=[
                    ContentPartTextParam(
                        text=dedent("""
                            You classify a task for authentication needs.

                            Return STRICT JSON for TaskIntent:
                            - needs_auth: boolean (will the agent likely need to be signed in to complete the task?)
                            - needs_account_creation: boolean (does the task require creating a new account?)
                            - include_auth_in_final_guide: boolean

                            IMPORTANT POLICY:
                            - include_auth_in_final_guide MUST be false by default.
                            - include_auth_in_final_guide can be true ONLY if the user explicitly asked for login/signup/password reset/MFA/verification
                            (in any language), e.g.:
                            - "how do I log in", "sign in", "create an account", "verify my email", "enter OTP", "reset password",
                            - "from the login page", "I can't sign in", "my OTP doesn't arrive"
                            - If the task is about an in-app workflow (e.g., create article, send message, create user),
                            include_auth_in_final_guide MUST be false even if authentication is required to perform the task.
                            - Do NOT treat 'Start url: ...' as a user request. It's tooling context.

                            Output JSON only.
                            """).strip()
                    )
                ]
            )

            intent_user = UserMessage(content=[ContentPartTextParam(text=f"Task: {optimized_task}")])
            intent_resp = await intent_llm.ainvoke([intent_system, intent_user], output_format=TaskIntent)
            sess.intent = intent_resp.completion
            sess.signup_intent = bool(sess.intent.needs_account_creation)
        except Exception as exc:
            log.info("Intent classification failed, defaulting to no-auth | %r", exc)
            sess.intent = TaskIntent(needs_auth=False, needs_account_creation=False, include_auth_in_final_guide=False)
            sess.signup_intent = False

        # Credential extraction (language-agnostic, LLM-based; email may be inferred directly if obvious)
        try:
            cred_system = SystemMessage(
                content=[
                    ContentPartTextParam(
                        text=dedent(
                            """
                            Extract credentials from the task text only if they are explicitly present (any language).
                            Return STRICT JSON matching TaskCredentials with fields:
                            - email: string or null
                            - password: string or null
                            Do NOT invent values; use null when absent.
                            """
                        ).strip()
                    )
                ]
            )
            cred_user = UserMessage(content=[ContentPartTextParam(text=f"Task: {optimized_task}")])
            cred_resp = await intent_llm.ainvoke([cred_system, cred_user], output_format=TaskCredentials)
            creds = cred_resp.completion
            sess.user_credentials = {"email": creds.email, "password": creds.password} if (creds.email and creds.password) else None
        except Exception as exc:
            log.info("Credential extraction failed; none captured | %r", exc)
            sess.user_credentials = None

        prev = sess.extend_system_message or ""
        agent_profile = getattr(sess, "agent_profile", None) or DEFAULT_AGENT_PROFILE
        agent_profile_block = f"""

            PROFILE IDENTITY (EXECUTION ONLY)
            - When a form asks for first name, last name, or username, use:
              first name "{agent_profile.get("first_name", "Pathix")}",
              last name "{agent_profile.get("last_name", "Agent")}",
              username "{agent_profile.get("username", "pathix-agent")}".
            - Do not include these values in the final guide; keep placeholders there.
            """
        signup_execution_block = ""
        if sess.user_credentials and sess.user_credentials.get("email"):
            log.info("User credentials detected for session %s (email=%s)", sess.id, sess.user_credentials.get("email"))

        if sess.signup_intent:
            creds_block = ""
            if sess.generated_credentials:
                email_hint = sess.generated_credentials.get("email", "")
                password_hint = sess.generated_credentials.get("password", "")
                creds_block = f"""
            Generated signup credentials (for browser actions only; never show in final guide):
            EMAIL={email_hint}
            PASSWORD={password_hint}
            Use placeholders like <your email> / <your password> in outputs."""
            signup_execution_block = f"""

            EXECUTION BEFORE DONE (SIGNUP MODE - HARD RULES)
            - You MUST create an account and end in an authenticated state before calling `done`.
            - Do NOT stop at the signup page; fill the form, submit it, and only call done when no password/otp fields are visible.
            - After signup, sign in with the same credentials if not clearly signed in (log out/in if needed).
            - If OTP is requested and the email is not @{DEFAULT_DOMAIN}, stop with "OTP required; cannot continue automatically."
            - If OTP is requested and the email is @{DEFAULT_DOMAIN}, call `fetch_mailbox_otp` (polls up to ~2 minutes). If none arrives, stop with "OTP not received."
            - Never expose real credentials in the guide output; always use placeholders.
            {creds_block}
            """

        sess.extend_system_message = dedent(f"""
            You are a senior technical writer creating a concise "how to" guide (like Slack / Microsoft / Amazon support articles).

            EXECUTION RULES (HARD)
            - Follow page state: credential fields (email/password) or OTP fields mean you are not authenticated; complete them before proceeding.
            - NEVER type dummy/placeholder credentials like "<your email>". Only use user-provided credentials or generated pathix.io credentials.
            - If credential fields appear and no usable credentials exist, call `create_signup_mailbox` to get a pathix.io email/password and use them to sign up/login. PROACTIVELY create accounts rather than stopping with an incomplete guide.
            - OTP/verification: When page state shows OTP fields (numeric short inputs, autocomplete one-time-code, etc.), call `fetch_mailbox_otp` for @pathix.io accounts you can access, enter the code, and continue. If the email is not @pathix.io or no mailbox creds exist, stop and say "OTP required; cannot continue automatically."

            DECISION LOOP (HARD)
            - Observe -> Plan -> Act -> Verify each step.
            - Observe: summarize only current, visible facts from browser_state/browser_vision.
            - Plan: propose up to 2-3 candidate actions with rationale tied to the next goal.
            - Act: choose ONE action that best matches the next goal.
            - Verify: confirm progress using concrete signals (URL change, modal opened/closed, new fields visible, confirmation text). If no clear progress or any error occurred, do not mark success.
            - If the same action type fails twice, change strategy (open menu, search if present, scroll, or go back).
            - Prefer elements with clear semantic purpose (roles/aria/labels). Avoid clicking large containers unless they clearly match the goal.
            - Close unrelated modals/overlays before continuing.

            SCOPE / DEFAULT START STATE (HARD RULES)
            - Include authentication steps in the final guide only if the classified intent says to include them; otherwise omit them (you may still perform them during execution).
            - Ignore any "Start url: ..." text; it is tooling context and must not appear in steps or notes.
            - Do NOT add notes like "You're signed in..."; just start from the first in-app action.

            STEP WRITING RULES (QUALITY BAR)
            - Usually 3-8 steps.
            - Each step is one concrete user action, imperative voice: "Click...", "Select...", "Type...".
            - One action per step. Do not combine independent actions in a single step.
            - Use exact visible UI labels in quotes: "Chat", "Send", "Settings".
            - No "Step X" text inside descriptions.
            - No secrets/credentials; use placeholders like <your email>, <your message>.
            - Do not invent UI labels or use vague placeholders like "appropriate button" or "checkmark".
            - Avoid hedging words like "usually", "typically", "might", or "may".

            AUTH / MAILBOX RULES (HARD GUARDRAILS)
            - Account creation IS APPROPRIATE when ALL of these are true: (1) you need to authenticate to complete the task, (2) no user credentials were provided to you, (3) the site offers a signup/create-account option. In this case, call `create_signup_mailbox` to get a pathix.io email/password and use them to sign up.
            - Account creation is NOT appropriate only when: the task explicitly requires using a specific existing account, OR the site has no signup option (invite-only/enterprise-only).
            - When you encounter a login/signup page and have no credentials, DO NOT stop and generate a speculative guide. Instead, call `create_signup_mailbox`, sign up, then continue with the actual task.
            - When an OTP/verification code is requested for a pathix.io email, call `fetch_mailbox_otp` (it polls for ~2 minutes). If the email is not @pathix.io, stop and say: "OTP required; cannot continue automatically."
            {signup_execution_block}
            {agent_profile_block}

            PAGE STATE CLASSIFICATION (REQUIRED IN EVERY RESPONSE)
            As part of your response, you MUST classify the current page state. Include in your `memory` field a JSON block with key "page_state" containing:
            - needs_otp: boolean - Does the page show OTP/verification code input fields?
            - needs_email_link: boolean - Does the page ask user to click a verification link in their email (not enter a code)?
            - has_login_error: boolean - Does the page show ANY error message preventing login success? This includes CAPTCHA errors, validation errors, network errors, or any text indicating the login attempt failed or was blocked.
            - has_captcha: boolean - Is there a CAPTCHA widget OR CAPTCHA-related error message visible? Set TRUE if you see any text mentioning reCAPTCHA, hCaptcha, Turnstile, "verify you're human", or error messages like "An error with reCAPTCHA occurred", "CAPTCHA validation failed", etc. A CAPTCHA error message means CAPTCHA is blocking login even if no widget is visible.
            - error_type: string or null - One of: "invalid_password", "account_not_found", "account_locked", "captcha_error", "other"
            - error_message: string or null - Copy the exact error text shown on the page (e.g. "An error with reCAPTCHA occured. Please try again.")
            - reason: string or null - Brief explanation of why you classified the page this way

            Example memory format: "Observed login page with email field. page_state: {{"needs_otp": false, "needs_email_link": false, "has_login_error": false, "has_captcha": false, "error_type": null, "error_message": null, "reason": "Clean login form, no errors visible"}}"
            Example with CAPTCHA error: "Login form shows reCAPTCHA error. page_state: {{"needs_otp": false, "needs_email_link": false, "has_login_error": true, "has_captcha": true, "error_type": "captcha_error", "error_message": "An error with reCAPTCHA occured. Please try again.", "reason": "reCAPTCHA error text visible on page, blocking login"}}"

            CAPTCHA HANDLING (CRITICAL)
            - If you see a reCAPTCHA/CAPTCHA error message on the page, use `detect_captcha` to identify the type.
            - If `detect_captcha` returns "recaptcha_enterprise" with "unsolvable: true", OR if you see a CAPTCHA error but `detect_captcha` finds no widget, this means the site uses INVISIBLE reCAPTCHA Enterprise which CANNOT be solved programmatically.
            - In this case, DO NOT retry login in a loop. Report failure immediately with: "Site uses invisible reCAPTCHA Enterprise that blocks automated logins. Cannot proceed automatically."
            - Retrying login when blocked by reCAPTCHA will only worsen the block. Stop after seeing CAPTCHA errors.
            - Only solvable CAPTCHA types (recaptcha_v2, hcaptcha, turnstile, funcaptcha) can be addressed with `solve_captcha`.

            OUTPUT (HARD RULES)
            - When finished, call `done`.
            - `steps` must be plain strings (no numbering).
            - `title` should be short and task-focused.
            - `notes` optional; use only for one important caveat (not prerequisites).
            - Only call `done` when the goal is verified from the CURRENT page state. If you cannot verify the outcome, keep working or return failure with a concrete reason.
            - Login success must be verified by absence of login form AND presence of authenticated UI signals (account/avatar/menu/dashboard).
            - Navigation success must be verified by a relevant URL/content change.

            When you finish, call the `done` action.
            The `done` payload MUST put the final answer under `data` with fields:
            {{title: string, steps: string[], links: string[], success: boolean, notes: string|null}}.

            {prev}
            """).strip()

        exclude_actions = ["screenshot"] if sess.use_vision != "auto" else []
        tools = Tools(exclude_actions=exclude_actions)

        # Register tool actions (create_signup_mailbox / fetch_mailbox_otp)
        _register_mailbox_actions(tools, sess)

        # Install runtime guardrails (OTP autofill + credential enforcement + done-blocking)
        install_auth_guardrails(tools, sess)

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

        agent_llm = _wrap_llm_with_logging(ChatOpenAI(model=OPENAI_MODEL), sess, "generation", token_cost, guide_id_for_run)

        def _extract_page_state_from_memory(memory: str | None) -> PageStateClassification | None:
            """Extract page_state JSON from agent's memory field."""
            if not memory:
                return None
            try:
                # Look for page_state: {...} pattern in memory - use greedy match to capture full JSON
                match = re.search(r'page_state:\s*(\{.*?\})\s*(?:$|["\']|[A-Z])', memory, re.IGNORECASE | re.DOTALL)
                if not match:
                    # Try alternate patterns the LLM might use
                    match = re.search(r'"page_state":\s*(\{.*?\})', memory, re.IGNORECASE | re.DOTALL)
                if match:
                    state_json = match.group(1)
                    # Try parsing; if it fails, try extending to find more closing braces
                    state_dict = None
                    for _ in range(3):  # Try up to 3 times to find valid JSON
                        try:
                            state_dict = json.loads(state_json)
                            break
                        except json.JSONDecodeError:
                            # Might have cut off too early, try finding next }
                            rest = memory[match.end():]
                            next_brace = rest.find('}')
                            if next_brace >= 0:
                                state_json = state_json + rest[:next_brace + 1]
                            else:
                                break
                    if state_dict:
                        return PageStateClassification(
                            needs_otp=bool(state_dict.get("needs_otp", False)),
                            needs_email_link=bool(state_dict.get("needs_email_link", False)),
                            has_login_error=bool(state_dict.get("has_login_error", False)),
                            has_captcha=bool(state_dict.get("has_captcha", False)),
                            error_type=state_dict.get("error_type"),
                            error_message=state_dict.get("error_message"),
                            reason=state_dict.get("reason", "Extracted from agent memory"),
                        )
            except Exception as exc:
                log.debug("Failed to extract page state from memory: %r", exc)
            return None

        def _update_step_context(browser_state_summary, model_output, step_number: int):
            sess.current_step = int(step_number)
            sess.current_url = getattr(browser_state_summary, "url", None) if browser_state_summary else None
            # Extract and store page state from agent's memory field
            if model_output:
                memory = getattr(model_output, "memory", None)
                page_state = _extract_page_state_from_memory(memory)
                if page_state:
                    sess.page_state = page_state
                    log.debug("Page state extracted: otp=%s, email_link=%s, error=%s (type=%s, msg=%r), captcha=%s, reason=%r",
                              page_state.needs_otp, page_state.needs_email_link,
                              page_state.has_login_error, page_state.error_type, page_state.error_message,
                              page_state.has_captcha, page_state.reason)

        agent = Agent(
            llm=agent_llm,
            tools=tools,
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
            register_new_step_callback=_update_step_context,
        )

        sess.state = "running"
        log.info("agent running...")

        result = await agent.run()
        usage_summary: dict[str, Any] | None = None
        with contextlib.suppress(Exception):
            usage_summary = result.usage.model_dump() if getattr(result, "usage", None) else None

        history_payload = result.model_dump()
        if usage_summary is not None:
            history_payload["usage"] = usage_summary

        full_trace = _build_action_trace(result)
        full_trace = _inject_auth_events(full_trace, sess.auth_events)
        sess.action_trace_full = full_trace

        sess.final_response = _ensure_json_text(history_payload)

        evidence_images = []
        evidence_table: list[EvidenceEvent] = []
        if sess.action_screenshots_enabled:
            with contextlib.suppress(Exception):
                evidence_images = _discover_screenshots(sess.screenshots_dir)
        with contextlib.suppress(Exception):
            evidence_table = _build_evidence_table(full_trace or [], evidence_images)

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

        guide_llm = _wrap_llm_with_logging(ChatOpenAI(model=OPENAI_MODEL, temperature=0), sess, "summarization", token_cost, guide_id_for_run)

        final_guide = await _generate_final_guide_with_evidence(
            task=sess.task,
            draft=draft,
            evidence_table=evidence_table,
            llm=guide_llm,
            include_auth=bool(sess.intent.include_auth_in_final_guide) if sess.intent else False,
        )

        rank_llm = _wrap_llm_with_logging(
            ChatOpenAI(model=OPENAI_MODEL, temperature=0), sess, "trace_ranking", token_cost, guide_id_for_run
        )
        relevance_labels = await _rank_action_trace(full_trace, task=sess.task, guide=final_guide, llm=rank_llm)
        _apply_relevance_labels(full_trace, relevance_labels)
        sess.action_trace = _filter_relevant_actions(full_trace)
        sess.action_trace_summary = _summarize_action_trace(sess.action_trace) if sess.action_trace else None

        final_with_images = _attach_images_by_evidence(final_guide, evidence_table)
        if not final_with_images and isinstance(final_guide, GuideOutputWithEvidence):
            final_with_images = final_guide.model_dump()
        elif not final_with_images:
            final_with_images = draft or {}
        if sess.action_trace_summary:
            final_with_images = dict(final_with_images)
            final_with_images["action_trace"] = sess.action_trace_summary
        if sess.action_trace_full:
            final_with_images = dict(final_with_images)
            final_with_images["action_trace_full"] = [entry.model_dump() for entry in sess.action_trace_full]
        if evidence_table:
            final_with_images = dict(final_with_images)
            final_with_images["evidence"] = [ev.model_dump() for ev in evidence_table]
        if usage_summary is not None:
            final_with_images = dict(final_with_images)
            final_with_images["token_usage"] = usage_summary

        if sess.generated_credentials:
            final_with_images = _sanitize_credentials_payload(final_with_images, sess.generated_credentials)

        if sess.user_credentials:
            final_with_images = _sanitize_credentials_payload(final_with_images, sess.user_credentials)
        if sess.generated_credentials:
            final_with_images = _sanitize_credentials_payload(final_with_images, sess.generated_credentials)

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
