from __future__ import annotations

import importlib.resources
from typing import Any, Literal

from pydantic import BaseModel, Field

from browser_use.llm.base import BaseChatModel
from browser_use.llm.browser_use.chat import ChatBrowserUse
from browser_use.llm.messages import (
	BaseMessage,
	ContentPartTextParam,
	SystemMessage,
	UserMessage,
)

# Modes map to tone/length expectations for the optimized task.
_MODE_HINTS: dict[str, str] = {
	'short': 'Rewrite the task in <=5 bullet points focusing on key actions only.',
	'regular': 'Rewrite the task with clear numbered steps, actionable verbs, and fallback guidance.',
	'detailed': (
		'Rewrite the task with exhaustive sub-steps, data validation requirements, and safety precautions. '
		'Call out when to capture screenshots or structured outputs.'
	),
	'flash': (
		'Rewrite the task for flash mode (no thinking field). Keep steps ultra-concise but still actionable, '
		'favoring single-sentence imperatives.'
	),
}

_DEFAULT_TEMPLATE_NAME = 'task_optimization_prompt.md'


class TaskOptimizationRequest(BaseModel):
	"""LLM-friendly request payload describing the optimization goal."""

	task: str = Field(..., min_length=1, description='Raw task to optimize')
	mode: Literal['short', 'regular', 'detailed', 'flash'] = Field(
		default='regular',
		description='Controls verbosity of the optimized task instructions',
	)


class TaskOptimizationResponse(BaseModel):
	"""Structured response for optimized tasks."""

	optimized_task: str = Field(..., description='Improved Browser-Use task instructions')


def _normalize_llm_text(result: Any) -> str:
	"""Normalize various LLM client return types into a clean string."""

	completion = getattr(result, 'completion', None)
	if isinstance(completion, str):
		return completion.strip()

	output_text = getattr(result, 'output_text', None)
	if isinstance(output_text, str):
		return output_text.strip()

	content = getattr(result, 'content', None)
	if isinstance(content, str):
		return content.strip()

	if isinstance(result, str):
		return result.strip()

	choices = getattr(result, 'choices', None)
	if choices and isinstance(choices, list):
		message = getattr(choices[0], 'message', None)
		if message and hasattr(message, 'content') and isinstance(message.content, str):
			return message.content.strip()

	return str(result).strip()


class AgentTaskOptimizer:
	"""Utility that rewrites user tasks into Browser-Use friendly prompts."""

	def __init__(
		self,
		llm: BaseChatModel | None = None,
		template_name: str = _DEFAULT_TEMPLATE_NAME,
	):
		self.llm = llm or ChatBrowserUse()
		self._template_name = template_name
		self._prompt_template = self._load_template(template_name)

	def _load_template(self, template_name: str) -> str:
		try:
			with importlib.resources.files('browser_use.agent').joinpath(template_name).open('r', encoding='utf-8') as fh:
				return fh.read()
		except FileNotFoundError as exc:
			raise RuntimeError(f'Prompt template {template_name} is missing from browser_use.agent') from exc

	def _build_messages(self, request: TaskOptimizationRequest) -> list[BaseMessage]:
		mode_hint = _MODE_HINTS.get(request.mode, _MODE_HINTS['regular'])
		formatted_prompt = self._prompt_template.format(task=request.task.strip(), mode_hint=mode_hint)

		return [
			SystemMessage(
				content=[
					ContentPartTextParam(
						text='You are a senior Browser-Use prompt engineer. Return ONLY the improved task text—no commentary.'
					)
				]
			),
			UserMessage(content=[ContentPartTextParam(text=formatted_prompt)]),
		]

	def optimize(
		self,
		task: str | TaskOptimizationRequest,
		mode: Literal['short', 'regular', 'detailed', 'flash'] | None = None,
	) -> TaskOptimizationResponse:
		"""Synchronous helper that returns optimized Browser-Use instructions."""

		request = self._coerce_request(task, mode)
		messages = self._build_messages(request)
		result = self.llm.invoke(messages)  # type: ignore[attr-defined]
		return TaskOptimizationResponse(optimized_task=_normalize_llm_text(result))

	async def optimize_async(
		self,
		task: str | TaskOptimizationRequest,
		mode: Literal['short', 'regular', 'detailed', 'flash'] | None = None,
	) -> TaskOptimizationResponse:
		"""Async helper that returns optimized Browser-Use instructions."""

		request = self._coerce_request(task, mode)
		messages = self._build_messages(request)
		result = await self.llm.ainvoke(messages)
		return TaskOptimizationResponse(optimized_task=_normalize_llm_text(result))

	def _coerce_request(
		self,
		task: str | TaskOptimizationRequest,
		mode: Literal['short', 'regular', 'detailed', 'flash'] | None,
	) -> TaskOptimizationRequest:
		if isinstance(task, TaskOptimizationRequest):
			return task if mode is None else task.model_copy(update={'mode': mode})
		return TaskOptimizationRequest(task=task, mode=mode or 'regular')
