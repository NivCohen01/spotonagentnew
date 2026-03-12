from __future__ import annotations

import contextlib
import datetime as dt
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
from .service_models import (
    ActionTraceEntry,
    ElementSignature,
    EvidenceEvent,
    EvidenceImage,
    GuideOutputWithEvidence,
    GuideStepWithEvidence,
    LocatorCandidate,
    TargetBBox,
)

try:  # Optional image hashing dependencies
    from PIL import Image
except Exception:  # pragma: no cover - best-effort import
    Image = None

try:  # Optional, used for pHash
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - best-effort import
    np = None

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

    def _normalize_evidence_ids(raw_ids: Any) -> list[int]:
        evidence_ids = raw_ids or []
        if isinstance(evidence_ids, int):
            evidence_ids = [evidence_ids]
        normalized: list[int] = []
        for eid in evidence_ids:
            try:
                ival = int(eid)
            except Exception:
                continue
            normalized.append(ival)
        return normalized

    def _shape_sub_steps(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        shaped_sub: list[dict[str, Any]] = []
        for j, sub in enumerate(items, 1):
            if not isinstance(sub, dict):
                shaped_sub.append(
                    {
                        "number": j,
                        "description": str(sub),
                        "pageUrl": None,
                        "images": [],
                        "evidence_ids": [],
                        "primary_evidence_id": None,
                    }
                )
                continue
            page_url = sub.get("pageUrl") or sub.get("page_url") or None
            evidence_ids = _normalize_evidence_ids(sub.get("evidence_ids") or sub.get("evidenceIds"))
            primary_id = sub.get("primary_evidence_id") or sub.get("primaryEvidenceId")
            try:
                primary_id = int(primary_id) if primary_id is not None else None
            except Exception:
                primary_id = None
            shaped_sub.append(
                {
                    "number": sub.get("number", j),
                    "description": sub.get("description") or sub.get("text") or sub.get("title") or "",
                    "pageUrl": page_url,
                    "images": sub.get("images") if isinstance(sub.get("images"), list) else [],
                    "evidence_ids": evidence_ids,
                    "primary_evidence_id": primary_id,
                }
            )
        return shaped_sub

    for i, step in enumerate(raw, 1):
        if isinstance(step, dict):
            description = step.get("description") or step.get("text") or step.get("title") or ""
            page_url = step.get("pageUrl") or step.get("page_url") or None
            images = step.get("images") if isinstance(step.get("images"), list) else []
            evidence_ids = _normalize_evidence_ids(step.get("evidence_ids") or step.get("evidenceIds"))
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
                    "sub_steps": _shape_sub_steps(step.get("sub_steps") or step.get("subSteps")),
                }
            )
        else:
            shaped.append({"number": i, "description": str(step), "pageUrl": None, "images": [], "evidence_ids": [], "sub_steps": []})
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


_ELEMENT_ACTIONS = {
    "click",
    "type",
    "input",
    "fill",
    "select",
    "send_keys",
    "press",
    "tap",
    "choose",
    "check",
    "uncheck",
}

_STABLE_ATTR_KEYS = ("id", "name", "type", "href", "data-testid", "aria-label", "placeholder")

_DCT_CACHE: dict[int, Any] = {}


def _is_element_action(action_name: str | None) -> bool:
    return bool(action_name and action_name in _ELEMENT_ACTIONS)


def _normalize_text_value(text: Any, max_len: int = 120) -> Optional[str]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    cleaned = " ".join(raw.split())
    cleaned = re.sub(r"\b[a-f0-9]{8,}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{6,}\b", "", cleaned)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None
    return cleaned[:max_len]


def _derive_role(tag: str | None, attrs: dict[str, Any]) -> Optional[str]:
    if isinstance(attrs, dict):
        role = attrs.get("role")
        if isinstance(role, str) and role.strip():
            return role.strip().lower()
    if not tag:
        return None
    tag = tag.lower()
    if tag == "a":
        return "link"
    if tag == "button":
        return "button"
    if tag in ("select", "textarea"):
        return "textbox" if tag == "textarea" else "combobox"
    if tag == "input":
        input_type = (attrs or {}).get("type") if isinstance(attrs, dict) else None
        if isinstance(input_type, str):
            it = input_type.lower()
            if it in ("checkbox", "radio", "submit", "button"):
                return it if it in ("checkbox", "radio") else "button"
        return "textbox"
    return None


def _extract_element_name(attrs: dict[str, Any], element_text: str | None) -> Optional[str]:
    if not isinstance(attrs, dict):
        attrs = {}
    for key in ("display_name", "aria-label", "aria_label", "name", "title", "placeholder"):
        val = attrs.get(key)
        name = _normalize_text_value(val)
        if name:
            return name
    return _normalize_text_value(element_text)


def _extract_element_signature(
    action_name: str | None,
    element_tag: str | None,
    element_text: str | None,
    element_attributes: dict[str, Any] | None,
) -> ElementSignature | None:
    if not _is_element_action(action_name):
        return None
    attrs = element_attributes or {}
    tag = element_tag.upper() if isinstance(element_tag, str) and element_tag else None
    role = _derive_role(element_tag, attrs)
    name = _extract_element_name(attrs, element_text)
    text_snippet = _normalize_text_value(element_text or attrs.get("display_name") if isinstance(attrs, dict) else element_text)
    stable_attrs: dict[str, Optional[str]] = {}
    if isinstance(attrs, dict):
        for key in _STABLE_ATTR_KEYS:
            val = attrs.get(key)
            if isinstance(val, str) and val.strip():
                stable_attrs[key] = val.strip()
            elif val is not None and key in ("type",):
                stable_attrs[key] = str(val)
    return ElementSignature(tag=tag, role=role, name=name, text_snippet=text_snippet, attrs=stable_attrs)


def _escape_css_value(value: str) -> str:
    return value.replace('"', '\\"').strip()


def _append_candidate(
    candidates: list[LocatorCandidate],
    seen: set[tuple[str, str | None, str | None, str | None, str | None]],
    candidate: LocatorCandidate,
) -> None:
    key = (candidate.type, candidate.role, candidate.name, candidate.value, candidate.match)
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def _build_locator_candidates(
    action_name: str | None,
    *,
    element_tag: str | None,
    element_text: str | None,
    element_attributes: dict[str, Any] | None,
    xpath: str | None,
) -> list[LocatorCandidate] | None:
    if not _is_element_action(action_name):
        return None
    attrs = element_attributes or {}
    tag = element_tag.lower() if isinstance(element_tag, str) else None
    candidates: list[LocatorCandidate] = []
    seen: set[tuple[str, str | None, str | None, str | None, str | None]] = set()

    role = _derive_role(tag, attrs)
    name = _extract_element_name(attrs, element_text)
    if role and name:
        _append_candidate(
            candidates,
            seen,
            LocatorCandidate(type="role", role=role, name=name, confidence=0.9),
        )

    if isinstance(attrs, dict):
        data_testid = attrs.get("data-testid")
        if isinstance(data_testid, str) and data_testid.strip():
            value = f'[data-testid="{_escape_css_value(data_testid)}"]'
            _append_candidate(
                candidates,
                seen,
                LocatorCandidate(type="css", value=value, confidence=0.9),
            )

        element_id = attrs.get("id")
        if isinstance(element_id, str) and element_id.strip():
            value = f"#{_escape_css_value(element_id)}"
            _append_candidate(
                candidates,
                seen,
                LocatorCandidate(type="css", value=value, confidence=0.85),
            )

        name_attr = attrs.get("name")
        if isinstance(name_attr, str) and name_attr.strip():
            name_value = _escape_css_value(name_attr)
            value = f'{tag}[name="{name_value}"]' if tag else f'[name="{name_value}"]'
            _append_candidate(
                candidates,
                seen,
                LocatorCandidate(type="css", value=value, confidence=0.7),
            )

        href = attrs.get("href")
        if isinstance(href, str) and href.strip():
            href_value = _escape_css_value(href)
            value = f'a[href="{href_value}"]' if tag == "a" else f'[href="{href_value}"]'
            _append_candidate(
                candidates,
                seen,
                LocatorCandidate(type="css", value=value, confidence=0.65),
            )

        input_type = attrs.get("type")
        if isinstance(input_type, str) and input_type.strip():
            type_value = _escape_css_value(input_type)
            value = f'{tag}[type="{type_value}"]' if tag else f'[type="{type_value}"]'
            _append_candidate(
                candidates,
                seen,
                LocatorCandidate(type="css", value=value, confidence=0.55),
            )

    text_value = _normalize_text_value(element_text or name)
    if text_value:
        match = "exact" if len(text_value) <= 80 else "contains"
        confidence = 0.6 if match == "exact" else 0.45
        _append_candidate(
            candidates,
            seen,
            LocatorCandidate(type="text", value=text_value, match=match, confidence=confidence),
        )

    if xpath:
        _append_candidate(
            candidates,
            seen,
            LocatorCandidate(type="xpath", value=str(xpath), confidence=0.2),
        )

    return candidates


def _extract_target_bbox(element_dict: dict[str, Any]) -> TargetBBox | None:
    if not isinstance(element_dict, dict):
        return None
    bounds = element_dict.get("bounds") or element_dict.get("bounding_box")
    if not isinstance(bounds, dict):
        return None
    try:
        x = float(bounds.get("x"))
        y = float(bounds.get("y"))
        w = float(bounds.get("width", bounds.get("w")))
        h = float(bounds.get("height", bounds.get("h")))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return TargetBBox(x=x, y=y, w=w, h=h)


def _resolve_image_path(path: str | None) -> Optional[Path]:
    if not path:
        return None
    raw = str(path)
    candidate = Path(raw)
    if candidate.exists():
        return candidate
    rel = raw.lstrip("/")
    fallback = (SCREENSHOTS_BASE / rel).resolve()
    if fallback.exists():
        return fallback
    return None


def _dct_matrix(n: int) -> Any:
    if np is None:
        return None
    if n in _DCT_CACHE:
        return _DCT_CACHE[n]
    x = np.arange(n, dtype=float)
    k = x.reshape((n, 1))
    mat = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    mat[0, :] *= 1.0 / np.sqrt(2)
    mat *= np.sqrt(2 / n)
    _DCT_CACHE[n] = mat
    return mat


def _compute_phash(image: Any) -> Optional[str]:
    if Image is None or np is None or image is None:
        return None
    try:
        img = image.convert("L").resize((32, 32), Image.LANCZOS)
        pixels = np.asarray(img, dtype=float)
        mat = _dct_matrix(32)
        if mat is None:
            return None
        dct = mat @ pixels @ mat.T
        low = dct[:8, :8].flatten()
        if low.size < 2:
            return None
        median = float(np.median(low[1:]))
        bits = [1 if v > median else 0 for v in low]
        hex_str = ""
        for i in range(0, 64, 4):
            nibble = (bits[i] << 3) | (bits[i + 1] << 2) | (bits[i + 2] << 1) | bits[i + 3]
            hex_str += f"{nibble:x}"
        return f"phash:{hex_str}"
    except Exception:
        return None


def _crop_bbox(image: Any, bbox: TargetBBox, padding: int = 16) -> Any:
    if image is None or bbox is None:
        return None
    try:
        x1 = int(bbox.x) - padding
        y1 = int(bbox.y) - padding
        x2 = int(bbox.x + bbox.w) + padding
        y2 = int(bbox.y + bbox.h) + padding
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))
    except Exception:
        return None


def _phash_for_image_path(path: str | None, bbox: TargetBBox | None) -> Optional[str]:
    if not path:
        return None
    if Image is None or np is None:
        return None
    resolved = _resolve_image_path(path)
    if not resolved:
        return None
    try:
        with Image.open(resolved) as img:
            img = img.convert("RGB")
            if bbox:
                cropped = _crop_bbox(img, bbox)
                if cropped is None:
                    return None
                return _compute_phash(cropped)
            return _compute_phash(img)
    except Exception:
        return None


def _attach_target_phashes(action_trace: list[ActionTraceEntry], evidence: list[EvidenceEvent]) -> None:
    if not action_trace or not evidence:
        return
    evidence_by_step = {ev.evidence_id: ev for ev in evidence}
    for ev in evidence:
        if ev.target_bbox and ev.best_image:
            ev.target_crop_phash = _phash_for_image_path(ev.best_image, ev.target_bbox)
        elif ev.target_bbox and (ev.before_image or ev.after_image):
            img_path = ev.before_image or ev.after_image
            ev.target_crop_phash = _phash_for_image_path(img_path, ev.target_bbox)
        else:
            ev.target_crop_phash = None
    for entry in action_trace:
        if not entry.target_bbox:
            entry.target_crop_phash = None
            continue
        ev = evidence_by_step.get(entry.step)
        if not ev:
            entry.target_crop_phash = None
            continue
        img_path = ev.best_image or ev.before_image or ev.after_image
        entry.target_crop_phash = _phash_for_image_path(img_path, entry.target_bbox)


def _iso_from_ts(ts: int | None) -> str:
    if ts is None:
        return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    try:
        return dt.datetime.utcfromtimestamp(ts / 1000.0).replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _pick_baseline_ts(ev: EvidenceEvent | None) -> int | None:
    if not ev or not ev.images:
        return None
    ts_values = [img.ts for img in ev.images if img.ts is not None]
    if not ts_values:
        return None
    return min(ts_values)


def _build_page_baselines(page_urls: list[str], evidence: list[EvidenceEvent]) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    if not page_urls:
        return baselines
    for url in page_urls:
        if not url or url in baselines:
            continue
        candidates = [ev for ev in evidence if ev.page_url == url]
        chosen = None
        for ev in candidates:
            if ev.best_image:
                chosen = ev
                break
        if chosen is None and candidates:
            chosen = candidates[0]
        img_path = None
        if chosen:
            img_path = chosen.best_image or chosen.before_image or chosen.after_image
        ts = _pick_baseline_ts(chosen)
        baselines[url] = {
            "page_url": url,
            "dom_hash": None,
            "screenshot_phash": _phash_for_image_path(img_path, None) if img_path else None,
            "viewport": None,
            "captured_at": _iso_from_ts(ts),
        }
    return baselines


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
            element_tag = element_dict.get("node_name")
            element_text = element_dict.get("node_value")
            element_attrs = element_dict.get("attributes")
            xpath = element_dict.get("x_path") or element_dict.get("xpath")

            element_signature = _extract_element_signature(
                action_name,
                element_tag=element_tag,
                element_text=element_text,
                element_attributes=element_attrs if isinstance(element_attrs, dict) else None,
            )
            locator_candidates = _build_locator_candidates(
                action_name,
                element_tag=element_tag,
                element_text=element_text,
                element_attributes=element_attrs if isinstance(element_attrs, dict) else None,
                xpath=xpath,
            )
            target_bbox = _extract_target_bbox(element_dict)
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
                    xpath=xpath,
                    element_text=element_text,
                    element_tag=element_tag,
                    element_attributes=element_attrs,
                    params=params_dict,
                    locator_candidates=locator_candidates,
                    target_bbox=target_bbox,
                    element_signature=element_signature,
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
            "relevance": entry.relevance,
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
                        relevance=int(item["relevance"]) if "relevance" in item and item["relevance"] is not None else 1,
                    )
                )
            except Exception:
                continue

    entries.sort(key=lambda e: (e.step, e.order))
    return entries


def _apply_relevance_labels(entries: list[ActionTraceEntry], labels: list[int]) -> None:
    """Apply relevance labels (0/1) to action trace entries in order."""
    if not entries:
        return
    if len(labels) != len(entries):
        logger.warning("Relevance label count mismatch: %s labels for %s entries", len(labels), len(entries))
        return
    for entry, label in zip(entries, labels):
        entry.relevance = 1 if int(label or 0) else 0
        if entry.action in ("otp", "verification_link"):
            entry.relevance = 1


def _filter_relevant_actions(entries: list[ActionTraceEntry]) -> list[ActionTraceEntry]:
    """Return only relevance=1 entries in original order."""
    return [entry for entry in entries if int(getattr(entry, "relevance", 1)) == 1]


def _inject_auth_events(entries: list[ActionTraceEntry], auth_events: list[dict[str, Any]] | None) -> list[ActionTraceEntry]:
    """Insert OTP / verification events into the action trace near their step."""
    if not auth_events:
        return entries

    events_by_step: dict[int, list[dict[str, Any]]] = {}
    for ev in auth_events:
        try:
            step_val = int(ev.get("step") or 0)
        except Exception:
            step_val = 0
        events_by_step.setdefault(step_val, []).append(ev)

    entries_by_step: dict[int, list[ActionTraceEntry]] = {}
    for entry in entries:
        entries_by_step.setdefault(int(entry.step), []).append(entry)

    all_steps = sorted(set(entries_by_step.keys()) | set(events_by_step.keys()))
    new_entries: list[ActionTraceEntry] = []
    order = 1

    for step_id in all_steps:
        step_events = sorted(events_by_step.get(step_id, []), key=lambda e: e.get("ts") or 0)
        for ev in step_events:
            action = str(ev.get("action") or ev.get("type") or "otp")
            params = ev.get("params") if isinstance(ev.get("params"), dict) else {}
            new_entries.append(
                ActionTraceEntry(
                    step=step_id,
                    order=order,
                    action=action,
                    value=ev.get("value"),
                    page_url=ev.get("page_url"),
                    xpath=ev.get("xpath"),
                    element_text=ev.get("element_text"),
                    element_tag=ev.get("element_tag"),
                    params=params,
                    relevance=1,
                )
            )
            order += 1

        step_actions = sorted(entries_by_step.get(step_id, []), key=lambda e: e.order)
        for entry in step_actions:
            entry.order = order
            new_entries.append(entry)
            order += 1

    return new_entries


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


def _normalize_page_url(url: str | None) -> tuple[str | None, str | None]:
    """
    Normalize URLs for matching:
    - add https:// if scheme missing (without affecting path)
    - lowercase host
    - strip leading www. only (keep other subdomains)
    - normalize trailing slash on path
    - ignore query/fragment
    """
    if not url:
        return None, None
    raw = str(url).strip()
    if not raw:
        return None, None

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc or ""
    path = parsed.path or "/"

    if not host and parsed.path:
        parts = parsed.path.split("/", 1)
        host = parts[0]
        path = f"/{parts[1]}" if len(parts) > 1 else "/"

    host = host.lower()
    if host.startswith("www."):
        host = host[4:]

    path = "/" + path.lstrip("/")
    path = path.rstrip("/") or "/"

    return (host or None), path


def _page_url_matches(step_url: str | None, evidence_url: str | None) -> bool:
    if step_url is None:
        return True
    step_host, step_path = _normalize_page_url(step_url)
    ev_host, ev_path = _normalize_page_url(evidence_url)
    if ev_path is None:
        return False
    if step_host and ev_host and step_host != ev_host:
        return False
    if step_host and ev_host is None:
        return False
    return step_path == ev_path


def _self_test_page_url_matches() -> None:
    cases = [
        ("https://bringoz.com", "www.bringoz.com/", True),
        ("bringoz.com/book-a-demo", "https://www.bringoz.com/book-a-demo/", True),
        ("https://app.bringoz.com", "https://www.bringoz.com", False),
        ("https://example.com/a", "https://example.com/b", False),
    ]
    for step_url, ev_url, expected in cases:
        result = _page_url_matches(step_url, ev_url)
        assert result == expected, f"Mismatch for {step_url} vs {ev_url}: got {result}, expected {expected}"


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
        display_name = attrs.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            return display_name.strip()[:120]
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
                locator_candidates=primary_entry.locator_candidates if primary_entry else None,
                target_bbox=primary_entry.target_bbox if primary_entry else None,
                element_signature=primary_entry.element_signature if primary_entry else None,
                target_crop_phash=None,
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


_SCREENSHOT_VERBS = ("click", "select", "open", "choose", "press", "tap", "type", "enter", "fill", "send", "submit", "login", "log in")


def _needs_screenshot(description: str | None) -> bool:
    desc = (description or "").lower()
    return any(k in desc for k in _SCREENSHOT_VERBS)


def _preferred_action_kind(description: str | None) -> str:
    desc = (description or "").lower()
    if any(k in desc for k in ("click", "select", "open", "choose", "press", "tap", "submit", "send")):
        return "click"
    if any(k in desc for k in ("type", "enter", "fill", "login", "log in")):
        return "input"
    return "any"


def _extract_quoted_labels(description: str | None) -> list[str]:
    if not description:
        return []
    matches = re.findall(r"['\"]([^'\"]{1,80})['\"]", description)
    labels = [m.strip().lower() for m in matches if m.strip()]
    return labels


def _evidence_label_text(ev: EvidenceEvent) -> str:
    parts = [
        ev.label,
        ev.element_text,
        ev.element_tag,
    ]
    return " ".join([str(p) for p in parts if p]).lower()


def _label_matches_tokens(ev: EvidenceEvent, tokens: list[str]) -> bool:
    if not tokens:
        return True
    label_text = _evidence_label_text(ev)
    return any(tok in label_text for tok in tokens)


def _first_screenshot_evidence_id(step: GuideStepWithEvidence, ev_lookup: dict[int, EvidenceEvent]) -> int | None:
    for eid in step.evidence_ids:
        ev = ev_lookup.get(eid)
        if ev and ev.best_image:
            return eid
    return None


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


def _select_gap_fill_candidate(
    step: GuideStepWithEvidence,
    evidence_table: list[EvidenceEvent],
    used: set[int],
    last_screenshot_id: int | None,
    label_tokens: list[str] | None = None,
) -> EvidenceEvent | None:
    # Prefer same-page evidence when a pageUrl is present; otherwise fall back to nearest unused evidence with a screenshot.
    label_tokens = label_tokens or []
    if not step.pageUrl:
        anchor_ids = [last_screenshot_id] if last_screenshot_id is not None else None
        candidate = _nearest_unused_evidence(
            intent=_preferred_action_kind(step.description),
            page_url=None,
            used=used,
            evidence_table=evidence_table,
            anchor_ids=anchor_ids,
        )
        if candidate and candidate.best_image and _label_matches_tokens(candidate, label_tokens):
            return candidate
        # If label didn't match, try any matching candidate with a screenshot.
        matching = [
            ev
            for ev in evidence_table
            if ev.evidence_id not in used and ev.best_image and _label_matches_tokens(ev, label_tokens)
        ]
        if matching:
            return min(matching, key=lambda ev: ev.evidence_id)
        # If the best nearby evidence lacks a dedicated screenshot, allow it as a last resort.
        return candidate

    candidates = [
        ev for ev in evidence_table if ev.evidence_id not in used and ev.best_image and _page_url_matches(step.pageUrl, ev.page_url)
    ]
    if not candidates:
        return None

    preferred_kind = _preferred_action_kind(step.description)
    if preferred_kind == "click":
        preferred = [ev for ev in candidates if "click" in ev.action_types and _label_matches_tokens(ev, label_tokens)]
    elif preferred_kind == "input":
        preferred = [
            ev
            for ev in candidates
            if any(a in ev.action_types for a in ("input", "type", "send_keys")) and _label_matches_tokens(ev, label_tokens)
        ]
    else:
        preferred = [ev for ev in candidates if _label_matches_tokens(ev, label_tokens)]
    pool = preferred or candidates

    if last_screenshot_id is None:
        return min(pool, key=lambda ev: ev.evidence_id)

    greater = [ev for ev in pool if ev.evidence_id > last_screenshot_id]
    if greater:
        return min(greater, key=lambda ev: ev.evidence_id)

    return min(pool, key=lambda ev: (abs(ev.evidence_id - last_screenshot_id), ev.evidence_id))


def _post_process_guide_steps(steps: list[Any], evidence_table: list[EvidenceEvent]) -> list[dict[str, Any]]:
    ev_lookup = {ev.evidence_id: ev for ev in evidence_table}
    if not steps:
        return []

    step_objs: list[GuideStepWithEvidence] = []
    original_ids_by_idx: dict[int, list[int]] = {}

    for idx, step in enumerate(steps or [], 1):
        step_obj = step if isinstance(step, GuideStepWithEvidence) else GuideStepWithEvidence.model_validate(step)
        normalized_ids: list[int] = []
        for raw_id in step_obj.evidence_ids or []:
            try:
                eid = int(raw_id)
            except Exception:
                continue
            if eid in ev_lookup:
                normalized_ids.append(eid)
        step_obj.evidence_ids = list(dict.fromkeys(normalized_ids))

        for sub in step_obj.sub_steps:
            normalized_sub_ids: list[int] = []
            for raw_id in sub.evidence_ids or []:
                try:
                    eid = int(raw_id)
                except Exception:
                    continue
                if eid in ev_lookup:
                    normalized_sub_ids.append(eid)
            sub.evidence_ids = list(dict.fromkeys(normalized_sub_ids))

        original_ids_by_idx[idx] = list(step_obj.evidence_ids)
        step_objs.append(step_obj)

    # A) Label-based filter + page-url coherence filter
    for step_obj in step_objs:
        label_tokens = _extract_quoted_labels(step_obj.description)
        needs_screenshot = _needs_screenshot(step_obj.description)
        filtered_by_label: list[int] = []
        for eid in step_obj.evidence_ids:
            ev = ev_lookup.get(eid)
            if not ev:
                continue
            if "done" in ev.action_types:
                continue
            if needs_screenshot and not ev.best_image:
                continue
            if label_tokens and not _label_matches_tokens(ev, label_tokens):
                continue
            filtered_by_label.append(eid)
        if filtered_by_label or label_tokens:
            step_obj.evidence_ids = filtered_by_label

        if not step_obj.pageUrl:
            continue
        before = list(step_obj.evidence_ids)
        filtered: list[int] = []
        removed: list[int] = []
        for eid in step_obj.evidence_ids:
            ev = ev_lookup.get(eid)
            if ev and _page_url_matches(step_obj.pageUrl, ev.page_url):
                filtered.append(eid)
            else:
                removed.append(eid)
        if removed:
            logger.debug(
                "Step %s pageUrl=%s removed evidence_ids due to mismatch: %s (ev_urls=%s)",
                step_obj.number,
                step_obj.pageUrl,
                removed,
                [ev_lookup.get(rid).page_url if ev_lookup.get(rid) else None for rid in removed],
            )
        step_obj.evidence_ids = filtered

        for sub in step_obj.sub_steps:
            if not sub.pageUrl:
                continue
            sub_filtered: list[int] = []
            sub_removed: list[int] = []
            for eid in sub.evidence_ids:
                ev = ev_lookup.get(eid)
                if ev and _page_url_matches(sub.pageUrl, ev.page_url):
                    sub_filtered.append(eid)
                else:
                    sub_removed.append(eid)
            if sub_removed:
                logger.debug(
                    "Sub-step %s.%s pageUrl=%s removed evidence_ids due to mismatch: %s",
                    step_obj.number,
                    sub.number,
                    sub.pageUrl,
                    sub_removed,
                )
            sub.evidence_ids = sub_filtered

    # B) Uniqueness constraint across steps
    evidence_to_steps: dict[int, list[int]] = {}
    for idx, step_obj in enumerate(step_objs):
        for eid in step_obj.evidence_ids:
            evidence_to_steps.setdefault(eid, []).append(idx)

    for eid, idxs in evidence_to_steps.items():
        if len(idxs) <= 1:
            continue
        ev = ev_lookup.get(eid)
        matching_idxs = []
        for idx in idxs:
            st = step_objs[idx]
            if ev and _page_url_matches(st.pageUrl, ev.page_url):
                matching_idxs.append(idx)
            elif st.pageUrl is None and (ev is None or ev.page_url is None):
                matching_idxs.append(idx)
        if matching_idxs:
            keep_idx = min(matching_idxs, key=lambda i: step_objs[i].number or (i + 1))
        else:
            keep_idx = min(idxs, key=lambda i: step_objs[i].number or (i + 1))
        drop_idxs = [i for i in idxs if i != keep_idx]
        if drop_idxs:
            logger.debug(
                "Evidence %s duplicated across steps %s; keeping step %s, removing from %s",
                eid,
                [step_objs[i].number for i in idxs],
                step_objs[keep_idx].number,
                [step_objs[i].number for i in drop_idxs],
            )
        for idx in drop_idxs:
            st = step_objs[idx]
            st.evidence_ids = [val for val in st.evidence_ids if val != eid]

    used: set[int] = {eid for st in step_objs for eid in st.evidence_ids}

    # C) Fill missing screenshot evidence
    last_screenshot_id: int | None = None
    for step_obj in step_objs:
        label_tokens = _extract_quoted_labels(step_obj.description)
        existing_screenshot_id = _first_screenshot_evidence_id(step_obj, ev_lookup)
        if existing_screenshot_id is not None:
            last_screenshot_id = existing_screenshot_id
            continue

        if not _needs_screenshot(step_obj.description):
            continue

        candidate = _select_gap_fill_candidate(step_obj, evidence_table, used, last_screenshot_id, label_tokens=label_tokens)
        if candidate:
            step_obj.evidence_ids.append(candidate.evidence_id)
            used.add(candidate.evidence_id)
            if not (
                step_obj.primary_evidence_id
                and step_obj.primary_evidence_id in ev_lookup
                and ev_lookup[step_obj.primary_evidence_id].best_image
            ):
                step_obj.primary_evidence_id = candidate.evidence_id
            last_screenshot_id = candidate.evidence_id
            logger.debug(
                "Gap fill assigned evidence_id=%s to step %s (pageUrl=%s)", candidate.evidence_id, step_obj.number, step_obj.pageUrl
            )
        else:
            logger.debug(
                "Gap fill: step %s needs screenshot but no unused evidence found on pageUrl=%s",
                step_obj.number,
                step_obj.pageUrl,
            )

    # D) Primary evidence choice and final normalization
    normalized_steps: list[dict[str, Any]] = []
    for step_obj in step_objs:
        step_obj.evidence_ids = list(dict.fromkeys(step_obj.evidence_ids))
        if not step_obj.evidence_ids:
            step_obj.primary_evidence_id = None
            step_obj.images = []
        else:
            if step_obj.primary_evidence_id not in step_obj.evidence_ids:
                step_obj.primary_evidence_id = None
            has_primary_image = bool(
                step_obj.primary_evidence_id
                and step_obj.primary_evidence_id in ev_lookup
                and ev_lookup[step_obj.primary_evidence_id].best_image
            )
            if not has_primary_image:
                image_candidates = [eid for eid in step_obj.evidence_ids if ev_lookup.get(eid) and ev_lookup[eid].best_image]
                if image_candidates:
                    old_primary = step_obj.primary_evidence_id
                    step_obj.primary_evidence_id = image_candidates[0]
                    if old_primary and old_primary != step_obj.primary_evidence_id:
                        logger.debug(
                            "Primary evidence switched for step %s from %s to %s to use screenshot",
                            step_obj.number,
                            old_primary,
                            step_obj.primary_evidence_id,
                        )
            if step_obj.primary_evidence_id is None:
                step_obj.primary_evidence_id = step_obj.evidence_ids[0]
            step_obj.images = []

        for sub in step_obj.sub_steps:
            sub.evidence_ids = list(dict.fromkeys(sub.evidence_ids))
            if not sub.evidence_ids:
                sub.primary_evidence_id = None
                sub.images = []
            else:
                if sub.primary_evidence_id not in sub.evidence_ids:
                    sub.primary_evidence_id = None
                has_sub_primary_image = bool(
                    sub.primary_evidence_id
                    and sub.primary_evidence_id in ev_lookup
                    and ev_lookup[sub.primary_evidence_id].best_image
                )
                if not has_sub_primary_image:
                    sub_image_candidates = [eid for eid in sub.evidence_ids if ev_lookup.get(eid) and ev_lookup[eid].best_image]
                    if sub_image_candidates:
                        sub.primary_evidence_id = sub_image_candidates[0]
                if sub.primary_evidence_id is None:
                    sub.primary_evidence_id = sub.evidence_ids[0]
                sub.images = []
        normalized_steps.append(step_obj.model_dump())

    for idx, st in enumerate(normalized_steps, 1):
        before = original_ids_by_idx.get(idx, [])
        after = st.get("evidence_ids") or []
        if before != after:
            logger.debug("Step %s evidence_ids updated %s -> %s", st.get("number"), before, after)
        else:
            logger.debug("Step %s evidence_ids unchanged %s", st.get("number"), after)
        st["number"] = idx
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

    def _attach_for_step(step_dict: dict[str, Any], label: str = "step") -> None:
        candidate_ids: list[int] = []
        primary_id = step_dict.get("primary_evidence_id")
        if primary_id is not None:
            try:
                candidate_ids.append(int(primary_id))
            except Exception:
                pass
        for eid in step_dict.get("evidence_ids") or []:
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

        step_dict["primary_screenshot_step_id"] = chosen_id if chosen_id is not None else primary_id
        if chosen_path and chosen_id is not None:
            step_dict["images"] = [chosen_path]
        else:
            step_dict["images"] = []

        if step_dict.get("images"):
            img_path = step_dict["images"][0]
            info = _parse_screenshot_name(Path(img_path).name)
            if not info or info.get("run_step_id") != step_dict.get("primary_screenshot_step_id"):
                logger.warning(
                    "Image mismatch for %s %s: image step %s vs primary %s; clearing images",
                    label,
                    step_dict.get("number"),
                    info.get("run_step_id") if info else None,
                    step_dict.get("primary_screenshot_step_id"),
                )
                step_dict["images"] = []
            else:
                logger.debug(
                    "%s %s mapped image %s to run_step_id=%s",
                    label,
                    step_dict.get("number"),
                    img_path,
                    step_dict.get("primary_screenshot_step_id"),
                )

    for step in processed_steps:
        _attach_for_step(step, label="step")
        sub_steps = step.get("sub_steps") if isinstance(step, dict) else None
        parent_fallback: list[int] = []
        try:
            if step.get("primary_evidence_id") is not None:
                parent_fallback.append(int(step.get("primary_evidence_id")))
        except Exception:
            parent_fallback = []
        if not parent_fallback:
            for eid in step.get("evidence_ids") or []:
                try:
                    parent_fallback.append(int(eid))
                    break
                except Exception:
                    continue

        if isinstance(sub_steps, list) and sub_steps:
            for sub in sub_steps:
                if not isinstance(sub, dict):
                    continue
                if not sub.get("evidence_ids"):
                    fallback_ids = [pid for pid in parent_fallback if pid is not None]
                    sub["evidence_ids"] = fallback_ids
                    if fallback_ids and sub.get("primary_evidence_id") is None:
                        sub["primary_evidence_id"] = fallback_ids[0]
                _attach_for_step(sub, label="sub-step")

    # ------------------------------------------------------------------
    # Deduplication pass: each unique image should appear at most once
    # across the entire guide. If a parent step and its sub-steps share
    # the same image, keep it only on the parent. If the same image
    # appears on multiple unrelated steps, keep it on the first one.
    # ------------------------------------------------------------------
    used_images: set[str] = set()

    for step in processed_steps:
        parent_imgs = step.get("images") or []
        parent_img = parent_imgs[0] if parent_imgs else None

        sub_steps = step.get("sub_steps") if isinstance(step, dict) else None
        if isinstance(sub_steps, list):
            for sub in sub_steps:
                if not isinstance(sub, dict):
                    continue
                sub_imgs = sub.get("images") or []
                if not sub_imgs:
                    continue
                sub_img = sub_imgs[0]
                # Remove if same as parent or already used elsewhere
                if sub_img == parent_img or sub_img in used_images:
                    sub["images"] = []
                    logger.debug(
                        "Dedup: removed duplicate image %s from sub-step %s (parent has same or already used)",
                        sub_img,
                        sub.get("number"),
                    )
                else:
                    used_images.add(sub_img)

        # Now handle the parent image
        if parent_img:
            if parent_img in used_images:
                step["images"] = []
                logger.debug(
                    "Dedup: removed duplicate image %s from step %s (already used)",
                    parent_img,
                    step.get("number"),
                )
            else:
                used_images.add(parent_img)

    guide_dict["steps"] = processed_steps
    return guide_dict


def slugify_title_for_guide(title: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    base = re.sub(r"-{2,}", "-", base)
    rnd = uuid.uuid4().hex[:4]
    return f"how-to-{base}-{rnd}" if base else f"how-to-guide-{rnd}"
