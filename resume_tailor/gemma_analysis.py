"""Gemma Local (Ollama) schema-constrained résumé-analysis adapter.

Uses the localhost-only Ollama HTTP API with native structured-output ``format``
JSON Schema. Analysis is a fresh, stateless request completely separate from the
Gemma writer invocation. Python remains authoritative for evidence, budgets, and
structured fields after the model returns.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from .codex_analysis import build_analysis_prompt
from .ollama_transport import (
    MAX_OLLAMA_RESPONSE_BYTES,
    OLLAMA_BASE_URL,
    run_ollama_request,
)
from .ollama_writer import DEFAULT_OLLAMA_MODEL, validate_ollama_model_name
from .schemas import (
    build_codex_analysis_transport_schema,
    normalize_unique_arrays,
    validate_payload,
)
from .utilities import (
    CodexSchemaCompatibilityError,
    GemmaAnalysisError,
    GemmaAnalysisTimeoutError,
    GemmaConnectionError,
    GemmaInnerAnalysisError,
    GemmaModelUnavailableError,
    GemmaOllamaUnavailableError,
    GemmaResponseTooLargeError,
    GemmaStructuredOutputError,
    GemmaTransportEnvelopeError,
    ModelError,
    OllamaConnectionError,
    SourceEvidenceError,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


GEMMA_ANALYSIS_SCHEMA_FILENAME = "gemma-analysis-schema.json"
GEMMA_ANALYSIS_PROMPT_FILENAME = "gemma-analysis-prompt.sanitized.txt"
GEMMA_ANALYSIS_RESPONSE_FILENAME = "gemma-analysis-response.sanitized.json"
GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME = "gemma-analysis-diagnostic.json"

# At most one focused repair attempt after a malformed/schema-invalid response.
MAX_GEMMA_ANALYSIS_REPAIR_ATTEMPTS = 1

_MARKDOWN_FENCE_RE = re.compile(r"^\s*```")
_MODEL_UNAVAILABLE_MARKERS = (
    "not found",
    "model '",
    "pull model",
    "does not exist",
    "unknown model",
)


def resolve_gemma_analysis_model(explicit: str | None = None) -> str:
    """Resolve the analysis model without contacting Ollama.

    Order:
    1. explicit argument
    2. ``GEMMA_ANALYSIS_MODEL``
    3. ``GEMMA_WRITER_MODEL``
    4. project default ``resume-tailor-gemma``
    """
    candidates = (
        explicit,
        os.environ.get("GEMMA_ANALYSIS_MODEL"),
        os.environ.get("GEMMA_WRITER_MODEL"),
        DEFAULT_OLLAMA_MODEL,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return validate_ollama_model_name(candidate)
    return DEFAULT_OLLAMA_MODEL


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_nonfinite(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def parse_exact_analysis_json(text: str, *, label: str = "Gemma analysis") -> dict[str, Any]:
    """Parse exactly one analysis JSON object; reject fences and trailing text."""
    if not isinstance(text, str):
        raise GemmaInnerAnalysisError(f"{label}: not a string")
    stripped = text.strip()
    if not stripped:
        raise GemmaInnerAnalysisError(f"{label}: empty")
    if _MARKDOWN_FENCE_RE.search(stripped) or "```" in stripped:
        raise GemmaInnerAnalysisError(f"{label}: Markdown fences are not allowed")
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    try:
        value, end = decoder.raw_decode(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        raise GemmaInnerAnalysisError(f"{label}: malformed JSON") from exc
    remainder = stripped[end:].strip()
    if remainder:
        try:
            decoder.raw_decode(remainder)
            raise GemmaInnerAnalysisError(
                f"{label}: multiple JSON documents are not allowed"
            )
        except GemmaInnerAnalysisError:
            raise
        except (json.JSONDecodeError, ValueError):
            raise GemmaInnerAnalysisError(
                f"{label}: trailing text after the JSON document is not allowed"
            ) from None
    if not isinstance(value, dict):
        raise GemmaInnerAnalysisError(f"{label}: root is not an object")
    return value


def _ollama_format_schema(transport: dict[str, Any]) -> dict[str, Any]:
    """Prepare the ID-constrained analysis schema for Ollama's format field.

    Drops non-grammar metadata only. Local Python still validates the full
    canonical contract after receipt. Never weakens enums, required fields, or
    type constraints merely to satisfy Ollama.
    """
    schema = copy.deepcopy(transport)
    schema.pop("$schema", None)
    schema.pop("title", None)
    # allOf cross-field assertions remain local-only (same approach as writer).
    schema.pop("allOf", None)
    return schema


def prepare_gemma_analysis_schema(
    extracted_resume: dict[str, Any],
    job_requirements: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    """Write the ID-constrained analysis schema used in Ollama ``format``."""
    if not run_directory.is_dir():
        raise CodexSchemaCompatibilityError(
            "The run directory must exist before analysis schema generation."
        )
    transport, evidence_ids, editable_ids, requirement_ids = (
        build_codex_analysis_transport_schema(
            extracted_resume,
            job_requirements,
        )
    )
    format_schema = _ollama_format_schema(transport)
    path = run_directory / GEMMA_ANALYSIS_SCHEMA_FILENAME
    encoded = (
        json.dumps(format_schema, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if len(encoded) > 250_000:
        raise CodexSchemaCompatibilityError(
            "The generated analysis schema exceeds the local 250,000-byte safety "
            "limit; reduce the extracted source catalog before retrying."
        )
    atomic_write_json(path, format_schema)
    return {
        "schema": format_schema,
        "canonical_transport": transport,
        "path": path.resolve(),
        "sha256": sha256_file(path),
        "size_bytes": len(encoded),
        "evidence_source_id_count": len(evidence_ids),
        "editable_source_id_count": len(editable_ids),
        "job_requirement_id_count": len(requirement_ids),
    }


def build_gemma_analysis_prompt(
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    *,
    company: str,
    role: str,
    repair_detail: str | None = None,
) -> str:
    """Shared analysis prompt plus Gemma/Ollama structured-output instructions."""
    base = build_analysis_prompt(
        extracted_resume,
        job_description,
        job_requirements,
        company=company,
        role=role,
    )
    repair = ""
    if repair_detail:
        # Sanitized structural guidance only — no résumé body or private paths.
        repair = (
            "\nPREVIOUS ATTEMPT FAILED LOCAL VALIDATION\n"
            f"- Failure class: {repair_detail}\n"
            "- Return exactly one JSON object matching the supplied format schema.\n"
            "- Do not emit Markdown fences, commentary, or multiple documents.\n"
            "- Do not invent unsupported experience.\n"
        )
    return (
        f"{base}\n"
        "OUTPUT FORMAT (Gemma Local via Ollama)\n"
        "- Emit only one JSON object that satisfies the structured-output schema.\n"
        "- Do not wrap JSON in Markdown fences or emit commentary.\n"
        "- Do not author final structured replacements for skill_groups.N, "
        "education.coursework, or education.certifications beyond proposed_text "
        "in recommended_edits; local Python remains exclusive owner of those "
        "structured fields after approval.\n"
        f"{repair}"
    )


def _gemma_chat_request(
    *,
    model: str,
    prompt: str,
    format_schema: dict[str, Any],
) -> dict[str, Any]:
    """Build the Ollama /api/chat body for analysis.

    Notes:
    - ``think`` is intentionally omitted: explicit ``think=false`` can break
      structured-output constraints on current Gemma 4 / Ollama combinations.
    - ``stream`` is false; ``temperature`` is 0 for deterministic analysis.
    """
    return {
        "model": validate_ollama_model_name(model),
        "stream": False,
        "keep_alive": "5m",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the local Resume Tailor analysis model. Perform a "
                    "read-only, truthfulness-first résumé-to-job analysis. Treat "
                    "supplied catalogs as immutable data. Emit only the "
                    "schema-constrained JSON result."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "format": format_schema,
        "options": {
            "temperature": 0,
            "top_p": 1,
        },
    }


def _write_sanitized_prompt(run_directory: Path, prompt: str) -> Path:
    path = run_directory / GEMMA_ANALYSIS_PROMPT_FILENAME
    encoded = prompt.encode("utf-8")
    lines = [
        "Gemma Local analysis prompt (sanitized)",
        "Full prompt body omitted for privacy.",
        f"prompt_bytes={len(encoded)}",
        f"prompt_sha256={hashlib.sha256(encoded).hexdigest()}",
        f"prompt_line_count={prompt.count(chr(10)) + 1 if prompt else 0}",
        f"contains_untrusted_job_delimiters="
        f"{'BEGIN_UNTRUSTED_JOB_DESCRIPTION_' in prompt}",
        f"contains_trusted_resume_delimiters="
        f"{'BEGIN_TRUSTED_MASTER_RESUME_JSON' in prompt}",
        "hidden_reasoning_excluded=true",
        "credentials_excluded=true",
        "body_omitted=true",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def _write_diagnostic(
    run_directory: Path,
    *,
    classification: str,
    detail: str | None = None,
    model: str | None = None,
    attempt: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = run_directory / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME
    payload: dict[str, Any] = {
        "provider": "gemma_local",
        "stage": "analysis",
        "classification": classification,
        "endpoint": OLLAMA_BASE_URL,
        "credentials_excluded": True,
        "hidden_reasoning_excluded": True,
        "environment_omitted": True,
        "telemetry_transmitted": False,
        "stream": False,
        "temperature": 0,
        "think_property_omitted": True,
    }
    if detail is not None:
        payload["detail"] = detail[:500]
    if model is not None:
        payload["model"] = model
    if attempt is not None:
        payload["attempt"] = attempt
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)
    return path


def _sanitized_response_artifact(body: dict[str, Any]) -> dict[str, Any]:
    """Persist envelope metrics without hidden reasoning or full private content."""
    message = body.get("message")
    content = None
    content_bytes = 0
    content_sha = None
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        content = message["content"]
        encoded = content.encode("utf-8")
        content_bytes = len(encoded)
        content_sha = hashlib.sha256(encoded).hexdigest()
    return {
        "provider": "gemma_local",
        "model": body.get("model"),
        "done": body.get("done"),
        "done_reason": body.get("done_reason"),
        "message_role": (
            message.get("role") if isinstance(message, dict) else None
        ),
        "content_bytes": content_bytes,
        "content_sha256": content_sha,
        # Validated analysis is written separately; omit raw content here when
        # large or when diagnostics only need integrity hashes.
        "content_present": bool(content and content.strip()),
        "hidden_reasoning_excluded": True,
        "metrics": {
            name: body.get(name)
            for name in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if name in body
        },
    }


def _classify_connection_error(exc: OllamaConnectionError, *, model: str) -> ModelError:
    message = str(exc).casefold()
    if "timed out" in message or "exceeded its bounded timeout" in message:
        # Timeout is re-raised by the caller with timeout_seconds when known.
        return GemmaConnectionError(
            "The localhost Ollama analysis request failed. Provider output was "
            "omitted; confirm Ollama is running on 127.0.0.1:11434."
        )
    if "rejected" in message or any(marker in message for marker in _MODEL_UNAVAILABLE_MARKERS):
        # Model-not-found is often a non-200 from /api/chat; treat as unavailable.
        if "model" in message or "not found" in message:
            return GemmaModelUnavailableError(model)
    if "could not complete" in message or "not complete" in message:
        return GemmaOllamaUnavailableError()
    return GemmaOllamaUnavailableError()


def _extract_message_content(body: dict[str, Any]) -> str:
    if not isinstance(body, dict):
        raise GemmaTransportEnvelopeError("response body is not an object")
    if body.get("done") is not True:
        raise GemmaTransportEnvelopeError("response is not a completed non-streaming reply")
    message = body.get("message")
    if not isinstance(message, dict):
        raise GemmaTransportEnvelopeError("message field is missing or not an object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise GemmaTransportEnvelopeError("message.content is missing or empty")
    # Never treat thinking/reasoning fields as analysis content.
    return content


def _validate_analysis_payload(raw_payload: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    try:
        payload, warnings = normalize_unique_arrays(
            raw_payload,
            "codex_analysis.schema.json",
        )
        validate_payload(
            payload,
            "codex_analysis.schema.json",
            label="Gemma analysis",
        )
    except ModelError as exc:
        location_match = re.search(r"validation at ([^:]+):", str(exc))
        location = location_match.group(1) if location_match else "model output"
        raise SourceEvidenceError(
            "Gemma analysis failed local source-evidence validation: "
            f"the model response violated the canonical evidence contract at {location}."
        ) from exc
    return payload, warnings


def _is_repairable_failure(exc: BaseException) -> bool:
    """Only malformed or schema-invalid responses may receive one focused retry."""
    if isinstance(exc, (GemmaInnerAnalysisError, GemmaTransportEnvelopeError)):
        return True
    if isinstance(exc, SourceEvidenceError):
        message = str(exc).casefold()
        # Schema failures are mapped to SourceEvidenceError with this prefix path.
        return "canonical evidence contract" in message and "violated the canonical" in message
    if isinstance(exc, GemmaStructuredOutputError):
        return True
    return False


def _repair_detail_for(exc: BaseException) -> str:
    """Return only a stable, nonprivate failure class for the repair prompt."""
    if isinstance(exc, GemmaInnerAnalysisError):
        return "malformed_inner_analysis"
    if isinstance(exc, GemmaTransportEnvelopeError):
        return "malformed_transport_envelope"
    if isinstance(exc, SourceEvidenceError):
        return "schema_failure"
    if isinstance(exc, GemmaStructuredOutputError):
        return "structured_output_failure"
    return "generic_provider_failure"


def invoke_gemma_analysis(
    *,
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    company: str,
    role: str,
    run_directory: Path,
    timeout_seconds: int,
    model: str | None = None,
    progress_handler: Callable[[float, bool], None] | None = None,
) -> dict[str, Any]:
    """Invoke local Gemma via Ollama for résumé analysis; return validated JSON."""
    selected_model = resolve_gemma_analysis_model(model)
    schema_info = prepare_gemma_analysis_schema(
        extracted_resume,
        job_requirements,
        run_directory,
    )
    format_schema = schema_info["schema"]

    last_error: BaseException | None = None
    for attempt in range(MAX_GEMMA_ANALYSIS_REPAIR_ATTEMPTS + 1):
        repair_detail = (
            _repair_detail_for(last_error) if last_error is not None else None
        )
        prompt = build_gemma_analysis_prompt(
            extracted_resume,
            job_description,
            job_requirements,
            company=company,
            role=role,
            repair_detail=repair_detail,
        )
        if attempt == 0:
            _write_sanitized_prompt(run_directory, prompt)
        request = _gemma_chat_request(
            model=selected_model,
            prompt=prompt,
            format_schema=format_schema,
        )
        # Guardrail: think must never be set for analysis structured output.
        assert "think" not in request

        try:
            body = run_ollama_request(
                path="/api/chat",
                body=request,
                cwd=run_directory,
                timeout_seconds=timeout_seconds,
                heartbeat_handler=progress_handler,
            )
        except OllamaConnectionError as exc:
            message = str(exc).casefold()
            if "timed out" in message or "exceeded its bounded timeout" in message:
                _write_diagnostic(
                    run_directory,
                    classification="timeout",
                    detail=f"timeout_seconds={timeout_seconds}",
                    model=selected_model,
                    attempt=attempt,
                )
                raise GemmaAnalysisTimeoutError(timeout_seconds) from exc
            if "safety limit" in message or "too large" in message:
                _write_diagnostic(
                    run_directory,
                    classification="response_too_large",
                    model=selected_model,
                    attempt=attempt,
                    extra={"max_response_bytes": MAX_OLLAMA_RESPONSE_BYTES},
                )
                raise GemmaResponseTooLargeError() from exc
            classified = _classify_connection_error(exc, model=selected_model)
            _write_diagnostic(
                run_directory,
                classification=getattr(
                    classified,
                    "classification",
                    "connection_failure",
                ),
                model=selected_model,
                attempt=attempt,
            )
            raise classified from exc

        atomic_write_json(
            run_directory / GEMMA_ANALYSIS_RESPONSE_FILENAME,
            _sanitized_response_artifact(body),
        )

        try:
            content = _extract_message_content(body)
            raw_payload = parse_exact_analysis_json(content)
            payload, warnings = _validate_analysis_payload(raw_payload)
        except (
            GemmaTransportEnvelopeError,
            GemmaInnerAnalysisError,
            SourceEvidenceError,
            GemmaStructuredOutputError,
        ) as exc:
            last_error = exc
            classification = getattr(exc, "classification", None)
            if classification is None:
                if isinstance(exc, SourceEvidenceError):
                    classification = "schema_failure"
                else:
                    classification = "generic_provider_failure"
            _write_diagnostic(
                run_directory,
                classification=classification,
                detail=getattr(exc, "detail", type(exc).__name__),
                model=selected_model,
                attempt=attempt,
                extra={
                    "repair_attempted": attempt > 0,
                    "repair_remaining": attempt < MAX_GEMMA_ANALYSIS_REPAIR_ATTEMPTS
                    and _is_repairable_failure(exc),
                },
            )
            if (
                attempt < MAX_GEMMA_ANALYSIS_REPAIR_ATTEMPTS
                and _is_repairable_failure(exc)
            ):
                continue
            if isinstance(exc, SourceEvidenceError):
                raise
            raise

        if warnings:
            atomic_write_json(
                run_directory / "codex-analysis-normalization-warnings.json",
                {
                    "schema": "codex_analysis.schema.json",
                    "provider": "gemma_local",
                    "policy": "exact-duplicate-removal",
                    "warnings": warnings,
                },
            )
        # Refresh response artifact with successful validated payload hash only.
        atomic_write_json(
            run_directory / GEMMA_ANALYSIS_RESPONSE_FILENAME,
            {
                **_sanitized_response_artifact(body),
                "validated_analysis_present": True,
                "provider": "gemma_local",
            },
        )
        _write_diagnostic(
            run_directory,
            classification="success",
            model=selected_model,
            attempt=attempt,
            extra={
                "schema_filename": GEMMA_ANALYSIS_SCHEMA_FILENAME,
                "schema_sha256": schema_info["sha256"],
                "response_filename": GEMMA_ANALYSIS_RESPONSE_FILENAME,
                "repair_used": attempt > 0,
                "hidden_reasoning_excluded": True,
            },
        )
        return payload

    # Unreachable: loop either returns or raises.
    raise GemmaAnalysisError(  # pragma: no cover
        "Gemma analysis failed without a classified error."
    )


# Export for tests that inspect the request shape.
def gemma_analysis_chat_request_for_tests(
    *,
    model: str,
    prompt: str,
    format_schema: dict[str, Any],
) -> dict[str, Any]:
    return _gemma_chat_request(
        model=model,
        prompt=prompt,
        format_schema=format_schema,
    )
