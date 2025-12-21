from __future__ import annotations

import contextlib
import json
import logging
import re
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel

from .service_config import SCREENSHOTS_BASE, STEP_IMG_RE
from .service_models import ActionTraceEntry, EvidenceEvent, EvidenceImage, GuideOutputWithEvidence, GuideStepWithEvidence

logger = logging.getLogger(__name__)

_FILENAME_IMG_RE = re.compile(
    r"step_(?P<step>\d+?)_(?P<action>[a-zA-Z0-9]+)_(?P<action_index>\d+?)_(?P<click_index>[a-zA-Z0-9]+)_(?P<ts>\d+)\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)


def _to_public_image_path(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(SCREENSHOTS_BASE.resolve())
        return "/" + rel.as_posix()
    except Exception:
        return "/" + p.name


def _maybe_to_dict(obj: Any) -> Any:
    for method in ("model_dump", "dict", "to_dict"):
        func = getattr(obj, method, None)
        if callable(func):
            with contextlib.suppress(Exception):
                return func()
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
    for m in re.finditer(r"\{[^{}]*\"title\"[^{}]*\"steps\"[^{}]*\}", candidate, flags=re.DOTALL):
        blob = m.group(0)
        with contextlib.suppress(Exception):
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
    candidate2 = candidate.replace("'", '"')
    for m in re.finditer(r"\{[^{}]*\"title\"[^{}]*\"steps\"[^{}]*\}", candidate2, flags=re.DOTALL):
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
            return {
                k: (list(data[k]) if k in ("steps", "links") else data[k])
                for k in ["title", "steps", "links", "notes", "success"]
                if k in data
            }

    obj = _maybe_to_dict(obj)

    if isinstance(obj, dict):
        if required.issubset(obj.keys()):
            return {k: (list(obj[k]) if k in ("steps", "links") else obj.get(k)) for k in ["title", "steps", "links", "notes", "success"]}
        history = obj.get("history")
        if isinstance(history, list):
            for step in reversed(history):
                with contextlib.suppress(Exception):
                    results = step.get("result") or []
                    for r in results:
                        if r.get("is_done") is True and r.get("extracted_content"):
                            inner = json.loads(r["extracted_content"])
                            if isinstance(inner, dict) and required.issubset(inner.keys()):
                                return {
                                    k: (list(inner[k]) if k in ("steps", "links") else inner.get(k))
                                    for k in ["title", "steps", "links", "notes", "success"]
                                }
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
                    return {
                        k: (list(data[k]) if k in ("steps", "links") else data.get(k))
                        for k in ["title", "steps", "links", "notes", "success"]
                    }

    if isinstance(obj, str):
        with contextlib.suppress(Exception):
            return _coerce_guide_output_dict(json.loads(obj))
        inner = _extract_json_snippet_from_text(obj)
        if inner and required.issubset(inner.keys()):
            return {k: (list(inner[k]) if k in ("steps", "links") else inner.get(k)) for k in ["title", "steps", "links", "notes", "success"]}

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
                return {k: (list(data[k]) if k in ("steps", "links") else data.get(k)) for k in ["title", "steps", "links", "notes", "success"]}

    return None


def _shape_steps_with_placeholders(extracted: dict) -> dict:
    raw = list(extracted.get("steps") or [])
    shaped = []
    for i, step in enumerate(raw, 1):
        if isinstance(step, dict):
            description = step.get("description") or step.get("text") or step.get("title") or ""
            page_url = step.get("pageUrl") or step.get("page_url") or None
            images = step.get("images") if isinstance(step.get("images"), list) else []
            evidence_ids = step.get("evidence_ids") or step.get("evidenceIds") or []
            if isinstance(evidence_ids, int):
                evidence_ids = [evidence_ids]
            evidence_ids = [int(eid) for eid in evidence_ids if isinstance(eid, (int, str)) and str(eid).isdigit()]
            primary_id = step.get("primary_evidence_id") or step.get("primaryEvidenceId")
            try:
                primary_id = int(primary_id) if primary_id is not None else None
            except Exception:
                primary_id = None
            shaped.append(
                {
                    "number": step.get("number", i),
                    "description": description,
                    "pageUrl": page_url,
                    "images": images,
                    "evidence_ids": evidence_ids,
                    "primary_evidence_id": primary_id,
                }
            )
        else:
            shaped.append({"number": i, "description": str(step), "pageUrl": None, "images": [], "evidence_ids": []})
    enriched = dict(extracted)
    enriched["steps"] = shaped
    return enriched


def _normalize_element_dict(element: Any) -> Optional[dict]:
    if element is None:
        return None
    if isinstance(element, dict):
        return element
    normalized = _maybe_to_dict(element)
    if isinstance(normalized, dict):
        return normalized
    return None


def _stringify_action_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value if v not in (None, "")]
        return ", ".join(parts)
    return str(value)


def _extract_action_value(action_name: str, params: dict[str, Any]) -> Optional[str]:
    if not isinstance(params, dict):
        return None
    preferred_keys = ("value", "text", "input_text", "keys", "content", "query", "prompt", "message")
    for key in preferred_keys:
        if key in params and params[key] not in (None, ""):
            return _stringify_action_value(params[key])
    for fallback_key in ("url", "selector", "xpath"):
        if fallback_key in params and params[fallback_key]:
            return _stringify_action_value(params[fallback_key])
    return None


def _history_entries_from_result(agent_result: Any) -> list[dict[str, Any]]:
    with contextlib.suppress(Exception):
        if hasattr(agent_result, "model_dump"):
            data = agent_result.model_dump()
            if isinstance(data, dict) and isinstance(data.get("history"), list):
                return data["history"]

    history_attr = getattr(agent_result, "history", None)
    if isinstance(history_attr, list) and history_attr:
        entries: list[dict[str, Any]] = []
        for item in history_attr:
            item_dict = None
            if hasattr(item, "model_dump"):
                with contextlib.suppress(Exception):
                    item_dict = item.model_dump()
            if item_dict is None:
                item_dict = _maybe_to_dict(item)
            if isinstance(item_dict, dict):
                entries.append(item_dict)
        if entries:
            return entries

    maybe_dict = _maybe_to_dict(agent_result)
    if isinstance(maybe_dict, dict) and isinstance(maybe_dict.get("history"), list):
        return maybe_dict["history"]

    return []


def _build_action_trace(agent_result: Any) -> list[ActionTraceEntry]:
    entries: list[ActionTraceEntry] = []
    history_items = _history_entries_from_result(agent_result)
    if not history_items:
        return entries

    order = 1
    for step_index, history_item in enumerate(history_items, 1):
        model_output = history_item.get("model_output") or {}
        state = history_item.get("state") or {}
        actions = model_output.get("action") or []
        results = history_item.get("result") or []
        if not isinstance(actions, list) or not actions:
            continue
        page_url = state.get("url")
        interacted_elements = state.get("interacted_element") or []
        if not isinstance(interacted_elements, list):
            interacted_elements = [interacted_elements]
        if not isinstance(results, list):
            results = []

        step_meta = history_item.get("metadata") or {}
        metadata_step = step_meta.get("step_number")
        try:
            run_step_id = int(metadata_step)
        except Exception:
            run_step_id = step_index

        for action_pos, action_dump in enumerate(actions):
            if not isinstance(action_dump, dict) or not action_dump:
                continue
            action_name, action_params = next(iter(action_dump.items()))
            if action_params is None:
                params_dict: dict[str, Any] = {}
            elif isinstance(action_params, dict):
                params_dict = dict(action_params)
            else:
                maybe_dict = _maybe_to_dict(action_params)
                params_dict = maybe_dict if isinstance(maybe_dict, dict) else {"value": str(maybe_dict)}

            interacted_element = interacted_elements[action_pos] if action_pos < len(interacted_elements) else None
            element_dict = _normalize_element_dict(interacted_element) or {}
            action_result = results[action_pos] if action_pos < len(results) else None
            if isinstance(action_result, dict) and action_result.get("error"):
                continue

            entries.append(
                ActionTraceEntry(
                    step=run_step_id,
                    order=order,
                    action=action_name,
                    value=_extract_action_value(action_name, params_dict),
                    page_url=page_url,
                    xpath=element_dict.get("x_path") or element_dict.get("xpath"),
                    element_text=element_dict.get("node_value"),
                    element_tag=element_dict.get("node_name"),
                    element_attributes=element_dict.get("attributes"),
                    params=params_dict,
                )
            )
            order += 1

    return entries


def _summarize_action_trace(entries: list[ActionTraceEntry]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        packed = {
            "step": entry.step,
            "order": entry.order,
            "value": entry.value,
            "xpath": entry.xpath,
            "page_url": entry.page_url,
            "element_text": entry.element_text,
            "element_tag": entry.element_tag,
        }
        if entry.params:
            packed["params"] = entry.params
        grouped.setdefault(entry.action, []).append({k: v for k, v in packed.items() if v not in (None, "", [], {})})

    summary: dict[str, Any] = {}
    for action, items in grouped.items():
        summary[action] = items[0] if len(items) == 1 else items
    return summary


def _flatten_action_trace_summary(trace: Any) -> list[ActionTraceEntry]:
    if isinstance(trace, str):
        with contextlib.suppress(Exception):
            trace = json.loads(trace)
    if not isinstance(trace, dict):
        return []

    entries: list[ActionTraceEntry] = []
    for action_name, payload in trace.items():
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue

            params = item.get("params") or {}
            if not isinstance(params, dict):
                params = {}

            try:
                entries.append(
                    ActionTraceEntry(
                        step=int(item.get("step") or 0),
                        order=int(item.get("order") or 0),
                        action=str(action_name),
                        value=item.get("value"),
                        page_url=item.get("page_url") or item.get("pageUrl"),
                        xpath=item.get("xpath"),
                        element_text=item.get("element_text"),
                        element_tag=item.get("element_tag"),
                        params=params,
                    )
                )
            except Exception:
                continue

    entries.sort(key=lambda e: (e.step, e.order))
    return entries


def _parse_image_from_filename(p: Path) -> EvidenceImage | None:
    m = _FILENAME_IMG_RE.search(p.name)
    if not m:
        return None
    try:
        run_step_id = int(m.group("step"))
        action_index = int(m.group("action_index"))
        ts = int(m.group("ts"))
    except Exception:
        return None
    click_index_raw = m.group("click_index")
    try:
        click_index = int(click_index_raw) if click_index_raw.lower() != "na" else None
    except Exception:
        click_index = None
    return EvidenceImage(
        file=_to_public_image_path(p),
        phase=None,
        run_step_id=run_step_id,
        action_index=action_index,
        click_index=click_index,
        page_url=None,
        ts=ts,
    )


def _load_screenshot_manifest(screens_dir: Path) -> list[EvidenceImage]:
    manifest_path = screens_dir / "manifest.jsonl"
    entries: list[EvidenceImage] = []
    if not manifest_path.exists():
        return entries
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        with contextlib.suppress(Exception):
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                continue
            file_path = obj.get("file")
            if file_path:
                obj["file"] = _to_public_image_path(Path(file_path))
            entry = EvidenceImage.model_validate(obj)
            entries.append(entry)
    return entries


def _discover_screenshots(screens_dir: Path) -> list[EvidenceImage]:
    if not screens_dir or not screens_dir.exists():
        return []
    manifest_entries = _load_screenshot_manifest(screens_dir)
    by_file: dict[str, EvidenceImage] = {e.file: e for e in manifest_entries if e.file}

    for e in manifest_entries:
        if e.run_step_id is None:
            logger.warning("Manifest entry missing run_step_id for file %s", e.file)

    for p in screens_dir.glob("*"):
        if not p.is_file():
            continue
        e = _parse_image_from_filename(p)
        if e is None:
            logger.warning("Screenshot file could not be matched to a run_step_id: %s", p)
            continue
        if e.file in by_file:
            existing = by_file[e.file]
            if existing.run_step_id is None and e.run_step_id is not None:
                existing.run_step_id = e.run_step_id
            if existing.ts is None and e.ts is not None:
                existing.ts = e.ts
            continue
        by_file[e.file] = e

    return list(by_file.values())


def _compact_url(url: str | None) -> str:
    if not url:
        return "-"
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or "/"
    if len(path) > 60:
        path = path[:57] + "..."
    return f"{host}{path}" if host else path


def _first_nonempty(values: list[str | None]) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _parse_screenshot_name(name: str) -> dict[str, Any] | None:
    m = _FILENAME_IMG_RE.search(name)
    if not m:
        m = STEP_IMG_RE.search(name)
    if not m:
        return None
    data: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        data["run_step_id"] = int(m.group("step"))
    with contextlib.suppress(Exception):
        data["action"] = m.group("action")
    with contextlib.suppress(Exception):
        data["action_index"] = int(m.group("action_index"))
    with contextlib.suppress(Exception):
        click = m.group("click_index")
        data["click_index"] = int(click) if str(click).lower() != "na" else None
    with contextlib.suppress(Exception):
        data["ts"] = int(m.group("ts"))
    return data


def _matches_run_step(path: str, run_step_id: int | None) -> bool:
    if run_step_id is None:
        return False
    name = Path(path).name
    info = _parse_screenshot_name(name)
    return bool(info and info.get("run_step_id") == int(run_step_id))


def _pick_primary_action(actions: list[str]) -> str | None:
    priority = {"click": 0, "input": 1, "navigate": 2, "extract": 3}
    if not actions:
        return None
    sorted_actions = sorted(actions, key=lambda a: priority.get(a, 99))
    return sorted_actions[0] if sorted_actions else None


def _pick_primary_entry(entries: list[ActionTraceEntry]) -> ActionTraceEntry | None:
    if not entries:
        return None
    priority = {"click": 0, "input": 1, "navigate": 2, "extract": 3}
    return sorted(entries, key=lambda e: (priority.get(e.action, 99), e.order))[0]


def _derive_label(entry: ActionTraceEntry | None) -> str | None:
    if entry is None:
        return None
    if entry.element_text:
        text = str(entry.element_text).strip()
        if text:
            return text[:120]
    attrs = entry.element_attributes or {}
    if isinstance(attrs, dict):
        for key in ("aria-label", "aria_label", "label", "name", "title", "placeholder"):
            val = attrs.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:120]
    if entry.value:
        val = str(entry.value).strip()
        if val:
            if entry.action in ("input", "type", "send_keys"):
                return f"Typed {val[:60]}"
            return val[:120]
    if entry.element_tag and entry.xpath:
        return f'{entry.element_tag.lower()} {entry.xpath.rsplit("/", 1)[-1]}'
    if entry.element_tag:
        return entry.element_tag.lower()
    return entry.action


def _select_best_images(images: list[EvidenceImage]) -> tuple[str | None, str | None, str | None]:
    if not images:
        return None, None, None
    sorted_imgs = sorted(images, key=lambda img: img.ts or 0)

    before_imgs = [img for img in sorted_imgs if img.phase and img.phase.lower() == "before"]
    after_imgs = [img for img in sorted_imgs if img.phase and img.phase.lower() == "after"]

    best = before_imgs[0].file if before_imgs else sorted_imgs[0].file
    before = before_imgs[0].file if before_imgs else None
    after = after_imgs[0].file if after_imgs else None
    return best, before, after


def _build_evidence_table(action_trace: list[ActionTraceEntry], images: list[EvidenceImage]) -> list[EvidenceEvent]:
    manifest_by_step: dict[int, list[EvidenceImage]] = {}
    unmatched_images: list[EvidenceImage] = []
    for entry in images or []:
        if entry.run_step_id is None:
            unmatched_images.append(entry)
            continue
        manifest_by_step.setdefault(int(entry.run_step_id), []).append(entry)

    for img in unmatched_images:
        logger.warning("Image without run_step_id: %s", img.file)

    actions_by_step: dict[int, list[ActionTraceEntry]] = {}
    for entry in action_trace or []:
        actions_by_step.setdefault(int(entry.step), []).append(entry)

    evidence: list[EvidenceEvent] = []
    for step_id in sorted(actions_by_step.keys()):
        step_actions = actions_by_step[step_id]
        primary_entry = _pick_primary_entry(step_actions)
        images_for_step = manifest_by_step.get(step_id, [])
        best_image, before_image, after_image = _select_best_images(images_for_step)

        page_url = _first_nonempty([a.page_url for a in step_actions])
        if page_url and images_for_step:
            for img in images_for_step:
                if img.page_url and img.page_url != page_url:
                    logger.warning("Evidence %s page_url mismatch; using image page_url %s", step_id, img.page_url)
                    page_url = img.page_url
                    break

        evidence.append(
            EvidenceEvent(
                evidence_id=step_id,
                action_types=[a.action for a in step_actions],
                page_url=page_url,
                element_tag=primary_entry.element_tag if primary_entry else None,
                element_text=primary_entry.element_text if primary_entry else None,
                label=_derive_label(primary_entry),
                best_image=best_image,
                before_image=before_image,
                after_image=after_image,
                images=images_for_step,
                params=primary_entry.params if primary_entry else {},
            )
        )

    evidence_ids = {ev.evidence_id for ev in evidence}
    for step_id, imgs in manifest_by_step.items():
        if step_id not in evidence_ids:
            logger.warning("Images found for run_step_id with no evidence: %s (%s)", step_id, [i.file for i in imgs])

    return evidence


def _format_evidence_table(evidence: list[EvidenceEvent]) -> str:
    lines: list[str] = []
    for ev in evidence:
        primary_action = _pick_primary_action(ev.action_types) or "action"
        img_flag = ev.best_image or ev.after_image or ev.before_image
        label = ev.label or ev.element_text or ev.element_tag or primary_action
        url = _compact_url(ev.page_url)
        lines.append(
            f"[{ev.evidence_id}] {primary_action} url={url} tag={ev.element_tag or '-'} "
            f'label="{label}" image={"yes" if img_flag else "no"}'
        )
    return "\n".join(lines)


def _candidate_lines(evidence: list[EvidenceEvent]) -> list[str]:
    lines: list[str] = []
    for ev in sorted(evidence, key=lambda e: e.evidence_id):
        actions_desc = "; ".join([a for a in ev.action_types]) or "action"
        label = ev.label or ev.element_text or ev.element_tag or actions_desc
        url = _compact_url(ev.page_url)
        lines.append(
            f"[{ev.evidence_id}] actions: [{actions_desc}] label=\"{label}\" "
            f"url={url} screenshot_after={bool(ev.after_image)} screenshot_before={bool(ev.before_image)}"
        )
    return lines


def _intent_from_description(description: str) -> str:
    desc = (description or "").lower()
    if any(k in desc for k in ("click", "select", "open", "press")):
        return "click"
    if any(k in desc for k in ("type", "enter", "fill")):
        return "input"
    if any(k in desc for k in ("wait", "after login", "you will be directed", "page loads", "ensure", "loaded")):
        return "wait"
    if any(k in desc for k in ("extract", "verify", "confirm", "check")):
        return "extract"
    return "unknown"


def _evidence_matches_intent(ev: EvidenceEvent, intent: str) -> bool:
    if intent == "click":
        return "click" in ev.action_types
    if intent == "input":
        return any(a in ev.action_types for a in ("input", "type", "send_keys"))
    if intent == "extract":
        return any(a in ev.action_types for a in ("extract", "verify"))
    if intent == "wait":
        return True
    return True


def _is_click_only(ev: EvidenceEvent) -> bool:
    return ev.action_types and all(a == "click" for a in ev.action_types)


def _nearest_unused_evidence(
    intent: str,
    page_url: str | None,
    used: set[int],
    evidence_table: list[EvidenceEvent],
    anchor_ids: list[int] | None = None,
) -> EvidenceEvent | None:
    candidates = [ev for ev in evidence_table if ev.evidence_id not in used and _evidence_matches_intent(ev, intent)]
    if not candidates:
        return None

    def _host_path(url: str | None) -> tuple[str | None, str | None]:
        if not url:
            return None, None
        parsed = urlparse(url)
        return parsed.netloc, parsed.path

    anchor_ids = anchor_ids or []
    anchor_set = {int(a) for a in anchor_ids if isinstance(a, int)}
    anchor_min = min(anchor_set) if anchor_set else None
    anchor_max = max(anchor_set) if anchor_set else None

    req_host, req_path = _host_path(page_url)

    def distance(ev: EvidenceEvent) -> int:
        if not anchor_set:
            return 0
        return min(abs(ev.evidence_id - a) for a in anchor_set)

    def score(ev: EvidenceEvent) -> tuple[int, int, int]:
        host, path = _host_path(ev.page_url)
        url_penalty = 0
        if req_host and host:
            if req_host != host:
                url_penalty = 5
            elif req_path and path and req_path.split("/")[1:2] != path.split("/")[1:2]:
                url_penalty = 2
        elif req_host and not host:
            url_penalty = 3
        dist = distance(ev)
        image_penalty = 0 if ev.best_image else 1
        return (url_penalty, dist, image_penalty, abs(ev.evidence_id))

    candidates.sort(key=score)
    best = candidates[0]

    if anchor_set:
        if distance(best) > 3:
            return None
    if req_host and best.page_url:
        host, _ = _host_path(best.page_url)
        if host and req_host != host:
            return None

    return best


def _post_process_guide_steps(steps: list[Any], evidence_table: list[EvidenceEvent]) -> list[dict[str, Any]]:
    ev_lookup = {ev.evidence_id: ev for ev in evidence_table}
    used: set[int] = set()
    normalized_steps: list[dict[str, Any]] = []

    for step in steps or []:
        step_obj = step if isinstance(step, GuideStepWithEvidence) else GuideStepWithEvidence.model_validate(step)
        intent = _intent_from_description(step_obj.description or "")
        original_ids = list(step_obj.evidence_ids)

        # Drop evidence not in lookup or already used
        step_obj.evidence_ids = [eid for eid in step_obj.evidence_ids if eid in ev_lookup and eid not in used]

        # Wait/state steps shouldn't steal click-only evidence when others exist
        if intent == "wait":
            click_only = [eid for eid in step_obj.evidence_ids if _is_click_only(ev_lookup[eid])]
            non_click = [eid for eid in step_obj.evidence_ids if not _is_click_only(ev_lookup[eid])]
            if non_click:
                step_obj.evidence_ids = non_click

        # For click/input prefer matching intent evidence if present
        if intent in ("click", "input", "extract"):
            matching = [eid for eid in step_obj.evidence_ids if _evidence_matches_intent(ev_lookup[eid], intent)]
            if matching:
                step_obj.evidence_ids = matching

        # Repair: assign nearest unused matching evidence if none remain for click/input
        if intent in ("click", "input") and not step_obj.evidence_ids:
            candidate = _nearest_unused_evidence(intent, step_obj.pageUrl, used, evidence_table, anchor_ids=original_ids)
            if candidate:
                step_obj.evidence_ids.append(candidate.evidence_id)

        # Enforce uniqueness across steps
        step_obj.evidence_ids = [eid for eid in step_obj.evidence_ids if eid not in used]
        for eid in step_obj.evidence_ids:
            used.add(eid)

        # Recompute primary
        step_obj.primary_evidence_id = None
        for eid in step_obj.evidence_ids:
            ev = ev_lookup.get(eid)
            if ev and ev.best_image and _evidence_matches_intent(ev, intent):
                step_obj.primary_evidence_id = eid
                break
        if step_obj.primary_evidence_id is None and step_obj.evidence_ids:
            step_obj.primary_evidence_id = step_obj.evidence_ids[0]

        # Images from primary evidence only
        if step_obj.primary_evidence_id is not None:
            ev = ev_lookup.get(step_obj.primary_evidence_id)
            step_obj.images = [ev.best_image] if ev and ev.best_image else []
        else:
            step_obj.images = []

        normalized_steps.append(step_obj.model_dump())

    # Renumber sequentially
    for i, st in enumerate(normalized_steps, 1):
        st["number"] = i
    return normalized_steps


def _attach_images_by_evidence(
    guide: GuideOutputWithEvidence | dict, evidence_table: list[EvidenceEvent]
) -> dict[str, Any]:
    guide_dict: dict[str, Any]
    if isinstance(guide, GuideOutputWithEvidence):
        guide_dict = guide.model_dump()
    elif isinstance(guide, dict):
        guide_dict = dict(guide)
    else:
        return {}

    processed_steps = _post_process_guide_steps(guide_dict.get("steps") or [], evidence_table)
    ev_lookup = {ev.evidence_id: ev for ev in evidence_table}

    for step in processed_steps:
        candidate_ids: list[int] = []
        primary_id = step.get("primary_evidence_id")
        if primary_id is not None:
            try:
                candidate_ids.append(int(primary_id))
            except Exception:
                pass
        for eid in step.get("evidence_ids") or []:
            try:
                ival = int(eid)
                if ival not in candidate_ids:
                    candidate_ids.append(ival)
            except Exception:
                continue

        chosen_path: str | None = None
        chosen_id: int | None = None
        for cid in candidate_ids:
            ev = ev_lookup.get(cid)
            if not ev:
                continue
            for candidate in (ev.best_image, ev.after_image, ev.before_image):
                if candidate and _matches_run_step(candidate, cid):
                    chosen_path = candidate
                    chosen_id = cid
                    break
            if chosen_path:
                break

        step["primary_screenshot_step_id"] = chosen_id if chosen_id is not None else primary_id
        if chosen_path and chosen_id is not None:
            step["images"] = [chosen_path]
        else:
            step["images"] = []

        if step.get("images"):
            img_path = step["images"][0]
            info = _parse_screenshot_name(Path(img_path).name)
            if not info or info.get("run_step_id") != step.get("primary_screenshot_step_id"):
                logger.warning(
                    "Image mismatch for step %s: image step %s vs primary %s; clearing images",
                    step.get("number"),
                    info.get("run_step_id") if info else None,
                    step.get("primary_screenshot_step_id"),
                )
                step["images"] = []
            else:
                logger.debug(
                    "Step %s mapped image %s to run_step_id=%s",
                    step.get("number"),
                    img_path,
                    step.get("primary_screenshot_step_id"),
                )

    guide_dict["steps"] = processed_steps
    return guide_dict


def slugify_title_for_guide(title: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    base = re.sub(r"-{2,}", "-", base)
    rnd = uuid.uuid4().hex[:4]
    return f"how-to-{base}-{rnd}" if base else f"how-to-guide-{rnd}"
