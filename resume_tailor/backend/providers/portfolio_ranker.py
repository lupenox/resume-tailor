"""Structured-output provider adapters for GitHub portfolio ranking.

The application layer owns prompts, evidence-request validation, score
calculation, and approval.  This module only invokes an explicitly selected
existing analysis provider and returns a schema-validated object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from resume_tailor.backend.engine.analysis import normalize_analysis_provider
from resume_tailor.backend.providers.grok_analysis import (
    parse_grok_inner_analysis,
    parse_grok_transport_envelope,
    resolve_grok_executable,
)
from resume_tailor.backend.providers.ollama_transport import run_ollama_request
from resume_tailor.backend.providers.ollama_writer import DEFAULT_OLLAMA_MODEL
from resume_tailor.backend.providers.subprocess_isolation import (
    enforce_tool_free_capability,
    external_provider_environment,
    isolated_provider_workspace,
)
from resume_tailor.backend.utils.schemas import parse_json_text
from resume_tailor.backend.utils.utilities import (
    ModelError,
    run_command,
)


PORTFOLIO_ANALYSIS_PROVIDERS = ("gemma_local", "grok_cli")


def portfolio_provider_is_external(provider: str) -> bool:
    """Return whether repository content leaves the local machine."""

    return normalize_analysis_provider(provider) != "gemma_local"


def _validate_output(
    value: Any,
    schema: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    try:
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(dict(schema))
        jsonschema.validate(instance=value, schema=dict(schema))
    except jsonschema.SchemaError as exc:
        raise ModelError(f"{label} local output schema is invalid.") from exc
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ModelError(
            f"{label} failed local schema validation at {location}."
        ) from exc
    if not isinstance(value, dict):
        raise ModelError(f"{label} returned a non-object response.")
    return value


@dataclass(frozen=True)
class ProviderPortfolioRanker:
    provider: str
    timeout_seconds: int
    model: str | None = None
    model_strength: str | None = None

    def __post_init__(self) -> None:
        provider = normalize_analysis_provider(self.provider)
        enforce_tool_free_capability(
            capability="portfolio_ranking",
            provider=provider,
            restrict_external_tools=True,
        )
        if provider not in PORTFOLIO_ANALYSIS_PROVIDERS:
            raise ModelError(
                "GitHub portfolio ranking supports only Gemma Local and the "
                "locked Grok CLI adapter."
            )
        if (
            not isinstance(self.timeout_seconds, int)
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds < 1
        ):
            raise ModelError("Portfolio ranking timeout must be positive.")

    def request_evidence(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        run_directory: Path,
        round_index: int,
    ) -> Mapping[str, Any]:
        if round_index not in {1, 2}:
            raise ModelError("Portfolio evidence-request round is invalid.")
        return self._invoke(
            prompt=prompt,
            schema=schema,
            run_directory=run_directory,
            phase=f"evidence-request-{round_index}",
        )

    def rank(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        run_directory: Path,
    ) -> Mapping[str, Any]:
        return self._invoke(
            prompt=prompt,
            schema=schema,
            run_directory=run_directory,
            phase="ranking",
        )

    def _invoke(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        run_directory: Path,
        phase: str,
    ) -> dict[str, Any]:
        provider = normalize_analysis_provider(self.provider)
        label = f"{provider} GitHub portfolio {phase}"
        if provider == "grok_cli":
            value = self._invoke_grok(
                prompt=prompt,
                schema=schema,
                run_directory=run_directory,
            )
        elif provider == "gemma_local":
            value = self._invoke_gemma(
                prompt=prompt,
                schema=schema,
                run_directory=run_directory,
            )
        else:  # normalize_analysis_provider is authoritative; defensive only.
            raise ModelError(f"Unsupported portfolio analysis provider: {provider}.")
        return _validate_output(value, schema, label=label)

    def _invoke_grok(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        run_directory: Path,
    ) -> Any:
        executable = resolve_grok_executable()
        encoded_schema = json.dumps(
            schema,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with isolated_provider_workspace(
            run_directory,
            prefix="resume-tailor-github-portfolio-",
        ) as workspace:
            args = [
                executable,
                "--no-auto-update",
                "--json-schema",
                encoded_schema,
                "--disable-web-search",
                "--no-subagents",
                "--no-memory",
                "--no-plan",
                "--max-turns",
                "1",
                "--permission-mode",
                "dontAsk",
                "--deny",
                "*",
                "--sandbox",
                "strict",
                "--cwd",
                str(workspace),
                "-p",
                prompt,
                "--output-format",
                "json",
            ]
            if self.model:
                args.extend(["--model", self.model])
            if self.model_strength:
                args.extend(["--reasoning-effort", self.model_strength])
            result = run_command(
                args,
                cwd=workspace,
                timeout_seconds=self.timeout_seconds,
                env=external_provider_environment(),
            )
        if result.returncode != 0:
            raise ModelError(
                "Grok GitHub portfolio ranking failed. Provider output was omitted."
            )
        envelope = parse_grok_transport_envelope(result.stdout)
        return parse_grok_inner_analysis(envelope["text"])

    def _invoke_gemma(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        run_directory: Path,
    ) -> Any:
        response = run_ollama_request(
            path="/api/chat",
            body={
                "model": self.model or DEFAULT_OLLAMA_MODEL,
                "stream": False,
                "format": dict(schema),
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0},
            },
            cwd=run_directory,
            timeout_seconds=self.timeout_seconds,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, Mapping):
            return dict(content)
        if not isinstance(content, str):
            raise ModelError(
                "Gemma Local portfolio response omitted structured content."
            )
        return parse_json_text(content, label="Gemma Local portfolio response")



__all__ = [
    "PORTFOLIO_ANALYSIS_PROVIDERS",
    "ProviderPortfolioRanker",
    "portfolio_provider_is_external",
]
