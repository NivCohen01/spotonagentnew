from __future__ import annotations

from pathlib import Path
from typing import Any

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import ContentPartTextParam, SystemMessage, UserMessage
from browser_use.llm.openai.chat import ChatOpenAI

from .service_config import OPENAI_MODEL

_OPTIMIZE_MODE_HINTS: dict[str, str] = {
    "regular": "Rewrite the task so it is clear, concise, and actionable while preserving intent.",
    "aggressive": "Aggressively clarify ambiguity, add safety rails, and spell out required confirmations.",
    "short": "Rewrite in a compact form (around 5 bullet points or less) without dropping critical details.",
    "verbose": "Rewrite with explicit numbered steps, validations, and completion criteria.",
}

_PROMPT_TEMPLATES: dict[str, str] = {
    "optimize": "optimize_prompt_template.md",
    "feature_compiler": "feature_compiler_prompt_template.md",
}


class OptimizeAgentPrompt:
    """LLM-backed helper that rewrites user tasks into clearer Browser-Use prompts."""

    def __init__(
        self,
        task: str,
        mode: str = "regular",
        ptype: str = "optimize",
        llm: BaseChatModel | None = None,
        custom_instructions: str | None = None,
        templates_dir: Path | None = None,
    ) -> None:
        self.task = (task or "").strip()
        self.mode = mode if mode in _OPTIMIZE_MODE_HINTS else "regular"
        self.ptype = ptype if ptype in _PROMPT_TEMPLATES else "optimize"
        self.llm = llm or ChatOpenAI(model=OPENAI_MODEL)
        self.custom_instructions = (custom_instructions or "").strip()
        self._templates_dir = templates_dir or Path(__file__).resolve().parent

    def _normalize_llm_text(self, result: Any) -> str:
        completion = getattr(result, "completion", None)
        if isinstance(completion, str):
            return completion.strip()

        output_text = getattr(result, "output_text", None)
        if isinstance(output_text, str):
            return output_text.strip()

        content = getattr(result, "content", None)
        if isinstance(content, str):
            return content.strip()

        if isinstance(result, str):
            return result.strip()

        choices = getattr(result, "choices", None)
        if choices and isinstance(choices, list):
            message = getattr(choices[0], "message", None)
            if message and hasattr(message, "content") and isinstance(message.content, str):
                return message.content.strip()

        return str(result).strip()

    def _load_template(self) -> str:
        template_name = _PROMPT_TEMPLATES.get(self.ptype, _PROMPT_TEMPLATES["optimize"])
        template_path = self._templates_dir / template_name
        fallback = (
            "You are a prompt optimizer for a web-browsing automation agent.\n\n"
            "Rewrite the task using mode: {mode_hint}.\n\n"
            "{custom_instructions_section}User task:\n{task}"
        )
        try:
            return template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return fallback

    def _build_prompt(self) -> str:
        template = self._load_template()
        custom_section = ""
        if self.custom_instructions:
            sanitized_custom = self.custom_instructions.replace("{", "{{").replace("}", "}}")
            custom_section = f"Custom instructions (apply before rewriting):\n{sanitized_custom}\n\n"
        mode_hint = _OPTIMIZE_MODE_HINTS.get(self.mode, _OPTIMIZE_MODE_HINTS["regular"])
        return template.format(
            task=self.task.replace("{", "{{").replace("}", "}}"),
            mode_hint=mode_hint,
            custom_instructions_section=custom_section,
        )

    def _build_messages(self) -> list:
        prompt_text = self._build_prompt()
        return [
            SystemMessage(
                content=[ContentPartTextParam(text="You optimize tasks for a Browser-Use agent. Return ONLY the improved task text.")]
            ),
            UserMessage(content=[ContentPartTextParam(text=prompt_text)]),
        ]

    def optimize(self) -> str:
        messages = self._build_messages()
        result = self.llm.invoke(messages)  # type: ignore[attr-defined]
        return self._normalize_llm_text(result)

    async def optimize_async(self) -> str:
        messages = self._build_messages()
        result = await self.llm.ainvoke(messages)
        return self._normalize_llm_text(result)
