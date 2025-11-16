from __future__ import annotations

import contextlib
import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from .service_config import SCREENSHOTS_BASE, STEP_IMG_RE
from .service_models import ActionTraceEntry


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
        m = STEP_IMG_RE.search(p.name)
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
        number = step.get("number") or i
        imgs = by_step.get(int(number), [])
        if imgs:
            old = step.get("images")
            if isinstance(old, list) and old:
                step["images"] = list(dict.fromkeys([*old, *imgs]))
            else:
                step["images"] = imgs
    return enriched


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
            shaped.append({"number": step.get("number", i), "description": description, "pageUrl": page_url, "images": images})
        else:
            shaped.append({"number": i, "description": str(step), "pageUrl": None, "images": []})
    enriched = dict(extracted)
    enriched["steps"] = shaped
    return enriched


def _normalize_element_dict(element: Any) -> Optional[dict]:
    """Convert DOMInteractedElement/dataclass instances into plain dictionaries."""
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
    """Best-effort extraction of the primary user-provided value for an action."""
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
    """Return history entries as dictionaries regardless of raw object type."""
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
    """Serialize every action from the agent history into ActionTraceEntry records."""
    entries: list[ActionTraceEntry] = []
    history_items = _history_entries_from_result(agent_result)
    if not history_items:
        return entries

    order = 1
    for step_index, history_item in enumerate(history_items, 1):
        model_output = history_item.get("model_output") or {}
        state = history_item.get("state") or {}
        actions = model_output.get("action") or []
        if not isinstance(actions, list) or not actions:
            continue
        page_url = state.get("url")
        interacted_elements = state.get("interacted_element") or []
        if not isinstance(interacted_elements, list):
            interacted_elements = [interacted_elements]

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

            entries.append(
                ActionTraceEntry(
                    step=step_index,
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
    """Group action trace entries by action name for a compact summary."""
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
    """Flatten the stored action_trace summary (grouped by action) into ordered ActionTraceEntry items."""
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


def slugify_title_for_guide(title: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    base = re.sub(r"-{2,}", "-", base)
    rnd = uuid.uuid4().hex[:4]
    return f"how-to-{base}-{rnd}" if base else f"how-to-guide-{rnd}"
