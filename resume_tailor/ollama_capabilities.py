"""Explicit, configurable local model capabilities and deterministic budgeting.

The Qwen writer previously hardcoded ``num_ctx=8192`` and ``num_predict=4096``
inside the chat request. A prompt of 7,590 tokens plus 1,145 generated tokens
exceeds an 8,192-token window, so generation can silently overflow the context
and lose the structured-output framing long before the output ceiling is
reached. This module makes those limits explicit, configurable, and checked
before any provider request is launched.

Nothing here relaxes a schema, an immutable fact, an approved-edit rule, an
evidence rule, or a content budget. The budget is an additional local gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .utilities import OllamaBudgetError


#: Conservative bytes-per-token divisor for budget estimation only.
#:
#: This deliberately under-estimates tokens per byte (i.e. over-estimates token
#: count) so the gate errs toward refusing a marginal request rather than
#: launching one that can overflow mid-generation. It is never used to trim,
#: summarize, or truncate authoritative content.
BUDGET_BYTES_PER_TOKEN = 3.0

#: Tokens reserved for the system message, chat template, and role scaffolding.
BUDGET_OVERHEAD_TOKENS = 320

DEFAULT_CONTEXT_WINDOW = 32_768
DEFAULT_MAX_OUTPUT_TOKENS = 8_192
DEFAULT_MIN_OUTPUT_TOKENS = 2_048

_MIN_CONTEXT_WINDOW = 2_048
_MAX_CONTEXT_WINDOW = 1_048_576


@dataclass(frozen=True)
class OllamaModelCapabilities:
    """Declared capabilities of one local writer model.

    Attributes:
        context_window: Total token window shared by prompt and generation.
        max_output_tokens: Upper bound requested via ``num_predict``.
        min_output_tokens: Refuse to launch when fewer than this many tokens
            remain for generation after the prompt is accounted for.
        supports_json_schema: Whether the model accepts a JSON Schema in
            ``format``. A model without this cannot be used for tailoring.
    """

    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    min_output_tokens: int = DEFAULT_MIN_OUTPUT_TOKENS
    supports_json_schema: bool = True

    def __post_init__(self) -> None:
        if not _MIN_CONTEXT_WINDOW <= self.context_window <= _MAX_CONTEXT_WINDOW:
            raise OllamaBudgetError(
                "The configured local model context window is outside the "
                "supported range."
            )
        if self.max_output_tokens < 1:
            raise OllamaBudgetError(
                "The configured local model output ceiling must be positive."
            )
        if self.min_output_tokens < 1:
            raise OllamaBudgetError(
                "The configured local model minimum output budget must be "
                "positive."
            )
        if self.min_output_tokens > self.max_output_tokens:
            raise OllamaBudgetError(
                "The configured minimum output budget exceeds the output "
                "ceiling."
            )
        if self.max_output_tokens >= self.context_window:
            raise OllamaBudgetError(
                "The configured output ceiling must leave room for the prompt "
                "inside the context window."
            )

    def sanitized(self) -> dict[str, Any]:
        """Return a content-free description for run metadata."""
        return {
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "min_output_tokens": self.min_output_tokens,
            "supports_json_schema": self.supports_json_schema,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "OllamaModelCapabilities":
        """Build capabilities from a partial mapping, keeping documented defaults."""
        if value is None:
            return cls()
        allowed = {
            "context_window",
            "max_output_tokens",
            "min_output_tokens",
            "supports_json_schema",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise OllamaBudgetError(
                "The local model capability configuration contains unsupported "
                f"keys: {', '.join(unknown)}."
            )
        base = cls()
        updates: dict[str, Any] = {}
        for key in ("context_window", "max_output_tokens", "min_output_tokens"):
            if key in value:
                candidate = value[key]
                if isinstance(candidate, bool) or not isinstance(candidate, int):
                    raise OllamaBudgetError(
                        f"The local model capability {key!r} must be an integer."
                    )
                updates[key] = candidate
        if "supports_json_schema" in value:
            candidate = value["supports_json_schema"]
            if not isinstance(candidate, bool):
                raise OllamaBudgetError(
                    "The local model capability 'supports_json_schema' must be "
                    "a boolean."
                )
            updates["supports_json_schema"] = candidate
        return replace(base, **updates)


#: Declared capabilities per known local model, by exact name then by prefix.
MODEL_CAPABILITIES: dict[str, OllamaModelCapabilities] = {
    "resume-tailor-qwen": OllamaModelCapabilities(
        context_window=32_768,
        max_output_tokens=8_192,
        min_output_tokens=2_048,
    ),
}


def capabilities_for_model(
    model: str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> OllamaModelCapabilities:
    """Resolve declared capabilities for ``model``, applying explicit overrides."""
    base = MODEL_CAPABILITIES.get(model)
    if base is None:
        # Match on the tag-free name so "resume-tailor-qwen:latest" resolves.
        bare = model.split(":", 1)[0]
        base = MODEL_CAPABILITIES.get(bare, OllamaModelCapabilities())
    if overrides is None:
        return base
    merged = {**base.sanitized(), **dict(overrides)}
    return OllamaModelCapabilities.from_mapping(merged)


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` deterministically and conservatively."""
    byte_length = len(text.encode("utf-8"))
    if byte_length == 0:
        return 0
    return max(1, math.ceil(byte_length / BUDGET_BYTES_PER_TOKEN))


@dataclass(frozen=True)
class OllamaBudget:
    """A deterministic, content-free budget decision for one request."""

    context_window: int
    prompt_tokens_estimated: int
    overhead_tokens: int
    available_output_tokens: int
    requested_output_tokens: int
    prompt_bytes: int

    def sanitized(self) -> dict[str, Any]:
        return {
            "context_window": self.context_window,
            "prompt_bytes": self.prompt_bytes,
            "prompt_tokens_estimated": self.prompt_tokens_estimated,
            "overhead_tokens": self.overhead_tokens,
            "available_output_tokens": self.available_output_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "bytes_per_token_divisor": BUDGET_BYTES_PER_TOKEN,
            "estimated": True,
        }


def plan_ollama_budget(
    *,
    prompt: str,
    capabilities: OllamaModelCapabilities,
) -> OllamaBudget:
    """Decide the output budget for ``prompt``, or refuse before any request.

    Raises:
        OllamaBudgetError: When the estimated prompt leaves fewer than
            ``capabilities.min_output_tokens`` for generation. Refusing here is
            strictly safer than launching a request that can overflow the
            context window and lose its structured-output framing.
    """
    if not capabilities.supports_json_schema:
        raise OllamaBudgetError(
            "The configured local model does not declare JSON-Schema "
            "structured-output support, so no tailoring request was launched."
        )
    prompt_bytes = len(prompt.encode("utf-8"))
    prompt_tokens = estimate_tokens(prompt)
    reserved = prompt_tokens + BUDGET_OVERHEAD_TOKENS
    available = capabilities.context_window - reserved
    if available < capabilities.min_output_tokens:
        raise OllamaBudgetError(
            "The approved tailoring prompt does not leave enough of the local "
            f"model context window for a complete response: about {prompt_tokens} "
            f"estimated prompt tokens plus {BUDGET_OVERHEAD_TOKENS} reserved "
            f"tokens against a {capabilities.context_window}-token window leaves "
            f"{max(0, available)}, below the required "
            f"{capabilities.min_output_tokens}. No Ollama request was launched "
            "and no approved content was trimmed."
        )
    return OllamaBudget(
        context_window=capabilities.context_window,
        prompt_tokens_estimated=prompt_tokens,
        overhead_tokens=BUDGET_OVERHEAD_TOKENS,
        available_output_tokens=available,
        requested_output_tokens=min(available, capabilities.max_output_tokens),
        prompt_bytes=prompt_bytes,
    )
