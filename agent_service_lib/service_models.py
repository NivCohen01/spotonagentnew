from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VisionLevel = Literal["auto", "low", "high"]
SessionState = Literal["queued", "starting", "running", "done", "error", "stopped"]
DeviceType = Literal["mobile", "desktop"]


class GuideOutput(BaseModel):
    title: str
    steps: list[str]
    links: list[str] = []
    notes: Optional[str] = None
    success: bool


class GuideSubStepWithEvidence(BaseModel):
    """Nested sub-step with evidence identifiers and image slots."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    number: int
    description: str
    pageUrl: Optional[str] = Field(default=None, alias="page_url")
    evidence_ids: list[int] = Field(default_factory=list)
    primary_evidence_id: Optional[int] = None
    images: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ensure_primary_is_listed(self) -> "GuideSubStepWithEvidence":
        if self.primary_evidence_id is not None and self.primary_evidence_id not in self.evidence_ids:
            self.evidence_ids.append(self.primary_evidence_id)
        return self


class GuideStepWithEvidence(BaseModel):
    """Guide step enriched with evidence identifiers and image slots."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    number: int
    description: str
    pageUrl: Optional[str] = Field(default=None, alias="page_url")
    evidence_ids: list[int] = Field(default_factory=list)
    primary_evidence_id: Optional[int] = None
    images: list[str] = Field(default_factory=list)
    sub_steps: list[GuideSubStepWithEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ensure_primary_is_listed(self) -> "GuideStepWithEvidence":
        if self.primary_evidence_id is not None and self.primary_evidence_id not in self.evidence_ids:
            self.evidence_ids.append(self.primary_evidence_id)
        return self


class GuideOutputWithEvidence(BaseModel):
    """Final guide schema with deterministic evidence mapping."""

    model_config = ConfigDict(extra="ignore")

    title: str
    steps: list[GuideStepWithEvidence]
    links: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    success: bool


class EvidenceImage(BaseModel):
    """Screenshot manifest entry associated with a run step."""

    file: str
    phase: Optional[str] = None
    run_step_id: int
    action_index: Optional[int] = None
    click_index: Optional[int] = None
    page_url: Optional[str] = None
    ts: Optional[int] = None


class EvidenceEvent(BaseModel):
    """Compact, LLM-friendly evidence descriptor derived from trace + screenshots."""

    evidence_id: int
    action_types: list[str] = Field(default_factory=list)
    page_url: Optional[str] = None
    element_tag: Optional[str] = None
    element_text: Optional[str] = None
    label: Optional[str] = None
    best_image: Optional[str] = None
    before_image: Optional[str] = None
    after_image: Optional[str] = None
    images: list[EvidenceImage] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class ActionTraceEntry(BaseModel):
    """Serialized representation of a single agent action and its DOM context."""

    step: int
    order: int
    action: str
    value: Optional[str] = None
    page_url: Optional[str] = None
    xpath: Optional[str] = None
    element_text: Optional[str] = None
    element_tag: Optional[str] = None
    element_attributes: Optional[dict[str, Any]] = None
    params: dict[str, Any] = Field(default_factory=dict)


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
    guide_family_id: Optional[int] = None
    guide_id: Optional[int] = None
    guide_family_key: Optional[str] = Field(None, alias="family_key")
    guide_family_id: Optional[int] = None
    guide_id: Optional[int] = None

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

    # legacy screenshot flags
    action_screenshots_enabled: bool = False
    action_screenshots_annotate: bool = True
    action_screenshots_include_files: bool = True
    action_screenshots_session_dirs: bool = False
    action_screenshots_spotlight: bool = False

    action_screenshots: Optional[ActionScreenshotOptions] = None

    device_type: DeviceType = "desktop"
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("task", mode="before")
    @classmethod
    def ensure_task_not_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("task must not be empty")
        return v

    @model_validator(mode="after")
    def sync_workspace_id(self) -> "StartReq":
        # accept either workspace_id or ws_id
        if self.workspace_id is None and self.ws_id is not None:
            self.workspace_id = self.ws_id
        return self



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
    action_trace: Optional[list[ActionTraceEntry]] = None
    action_trace_summary: Optional[dict[str, Any]] = None


class ResultPayload(BaseModel):
    state: SessionState
    final_response: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    screenshots_dir: Optional[str] = None
    action_trace: Optional[list[ActionTraceEntry]] = None
    action_trace_summary: Optional[dict[str, Any]] = None


class GenerateVideoResponse(BaseModel):
    session_id: str
    accepted: bool = False
    reason: Optional[str] = None
    video_path: Optional[str] = None
    video_filename: Optional[str] = None
    actions_replayed: int = 0
    skipped_actions: list[str] = Field(default_factory=list)


class OptimizeReq(BaseModel):
    task: str
    mode: Literal["regular", "aggressive", "short", "verbose"] = "regular"
    ptype: Literal["optimize", "feature_compiler"] = "optimize"
    llm_model: Optional[str] = None
    custom_instructions: Optional[str] = None


class OptimizeResp(BaseModel):
    original: str
    optimized: str
