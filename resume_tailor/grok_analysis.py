"""Grok Build CLI adapter for the résumé-analysis stage.

Grok is an explicitly selected alternative to Codex. It receives the same
analysis prompt contract and is validated against the same canonical local
schema. The CLI returns a transport envelope; only the ``text`` field is parsed
as the inner analysis document. The ``thought`` field is never treated as
analysis, displayed as reasoning, or persisted in normal artifacts.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from .codex_analysis import build_analysis_prompt
from .schemas import (
    build_codex_analysis_transport_schema,
    normalize_unique_arrays,
    validate_payload,
)
from .utilities import (
    CodexSchemaCompatibilityError,
    DependencyError,
    GrokAuthenticationError,
    GrokExecutableError,
    GrokInnerAnalysisError,
    GrokProcessError,
    GrokPromptTooLargeError,
    GrokTimeoutError,
    GrokTransportEnvelopeError,
    GrokUsageLimitError,
    ModelError,
    SourceEvidenceError,
    atomic_write_json,
    atomic_write_text,
    run_command,
    sha256_file,
)


DEFAULT_GROK_EXECUTABLE = Path.home() / ".grok" / "bin" / "grok"
GROK_ANALYSIS_SCHEMA_FILENAME = "grok-analysis-schema.json"
GROK_ANALYSIS_PROMPT_FILENAME = "grok-analysis-prompt.sanitized.txt"
GROK_ANALYSIS_TRANSPORT_FILENAME = "grok-analysis-transport.json"
GROK_ANALYSIS_RESPONSE_FILENAME = "grok-analysis-response.sanitized.json"
GROK_ANALYSIS_DIAGNOSTIC_FILENAME = "grok-analysis-diagnostic.json"

ACCEPTABLE_STOP_REASONS = frozenset({"end_turn", "stop", "completed"})
_MARKDOWN_FENCE_RE = re.compile(r"^\s*```")
_AUTH_FAILURE_MARKERS = (
    "not logged in",
    "not authenticated",
    "authentication failed",
    "authentication required",
    "login required",
    "unauthorized",
    "unauthenticated",
    "please log in",
    "please login",
    "auth required",
)
_USAGE_LIMIT_MARKERS = (
    "you've hit your usage limit",
    "you have hit your usage limit",
    "usage limit",
    "rate limit",
    "quota exceeded",
    "quota limit",
)


def resolve_grok_executable(explicit: str | None = None) -> str:
    """Resolve the Grok Build CLI without shell expansion.

    Prefer an explicit path, then the verified install location
    ``~/.grok/bin/grok``, then ``PATH``.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise GrokExecutableError(
            f"Configured Grok executable is missing or not executable: {candidate}"
        )
    preferred = DEFAULT_GROK_EXECUTABLE.expanduser()
    if preferred.is_file() and os.access(preferred, os.X_OK):
        return str(preferred.resolve())
    from shutil import which

    resolved = which("grok")
    if resolved:
        return resolved
    raise GrokExecutableError(
        "Required executable 'grok' was not found. Install Grok Build CLI and "
        "ensure it is available at ~/.grok/bin/grok or on PATH. See README.md."
    )


def grok_analysis_args(*, executable: str, prompt: str) -> list[str]:
    """Return the verified headless Grok argv (never passed through a shell)."""
    return [
        executable,
        "--no-auto-update",
        "-p",
        prompt,
        "--output-format",
        "json",
    ]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_nonfinite(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _parse_exact_json_value(
    text: str,
    *,
    label: str,
    reject_markdown_fences: bool = False,
) -> Any:
    """Parse exactly one JSON value with no leading/trailing non-JSON text."""
    if not isinstance(text, str):
        raise GrokInnerAnalysisError(f"{label}: not a string")
    stripped = text.strip()
    if not stripped:
        raise GrokInnerAnalysisError(f"{label}: empty")
    # Fence rejection applies only to the inner analysis document. The transport
    # envelope may legally contain fence characters inside the string-valued
    # ``text`` field when the model misbehaves; that is rejected after the
    # envelope is decoded.
    if reject_markdown_fences and (
        _MARKDOWN_FENCE_RE.search(stripped) or "```" in stripped
    ):
        raise GrokInnerAnalysisError(f"{label}: Markdown fences are not allowed")
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    try:
        value, end = decoder.raw_decode(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        raise GrokInnerAnalysisError(f"{label}: malformed JSON") from exc
    remainder = stripped[end:].strip()
    if remainder:
        # A second document or prose after the first object is always rejected.
        try:
            decoder.raw_decode(remainder)
            raise GrokInnerAnalysisError(
                f"{label}: multiple JSON documents are not allowed"
            )
        except GrokInnerAnalysisError:
            raise
        except (json.JSONDecodeError, ValueError):
            raise GrokInnerAnalysisError(
                f"{label}: trailing text after the JSON document is not allowed"
            ) from None
    return value


def parse_grok_transport_envelope(stdout: str) -> dict[str, Any]:
    """Parse the Grok CLI transport envelope from stdout.

    Requirements:
    - exactly one JSON document
    - top-level object
    - nonempty string field named ``text``
    - acceptable ``stopReason``
    - no reliance on thought, usage, sessionId, requestId, or modelUsage
    """
    try:
        envelope = _parse_exact_json_value(
            stdout,
            label="Grok transport envelope",
            reject_markdown_fences=False,
        )
    except GrokInnerAnalysisError as exc:
        raise GrokTransportEnvelopeError(exc.detail) from exc
    if not isinstance(envelope, dict):
        raise GrokTransportEnvelopeError("transport root is not an object")
    text = envelope.get("text")
    if not isinstance(text, str) or not text.strip():
        raise GrokTransportEnvelopeError(
            "missing or non-string nonempty text field"
        )
    stop_reason = envelope.get("stopReason")
    if not isinstance(stop_reason, str) or stop_reason not in ACCEPTABLE_STOP_REASONS:
        raise GrokTransportEnvelopeError(
            f"unacceptable stopReason: {stop_reason!r}"
            if isinstance(stop_reason, str)
            else "missing or non-string stopReason"
        )
    return envelope


def parse_grok_inner_analysis(text: str) -> dict[str, Any]:
    """Parse the envelope ``text`` field as exactly one analysis object."""
    value = _parse_exact_json_value(
        text,
        label="Grok inner analysis",
        reject_markdown_fences=True,
    )
    if not isinstance(value, dict):
        raise GrokInnerAnalysisError("inner analysis root is not an object")
    return value


def build_grok_analysis_prompt(
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    *,
    company: str,
    role: str,
    output_schema: dict[str, Any],
) -> str:
    """Build the shared analysis prompt plus the exact output-schema contract."""
    base = build_analysis_prompt(
        extracted_resume,
        job_description,
        job_requirements,
        company=company,
        role=role,
    )
    schema_json = json.dumps(output_schema, ensure_ascii=False, indent=2)
    return (
        f"{base}\n"
        "OUTPUT FORMAT (Grok Build)\n"
        "- Return only one JSON object that matches the schema below.\n"
        "- Do not wrap the JSON in Markdown fences.\n"
        "- Do not emit commentary, planning text, or multiple JSON documents.\n"
        "- Do not author final structured replacements for skill_groups.N, "
        "education.coursework, or education.certifications beyond proposed_text "
        "in recommended_edits; local Python remains exclusive owner of those "
        "structured fields after approval.\n"
        "BEGIN_ANALYSIS_OUTPUT_SCHEMA\n"
        f"{schema_json}\n"
        "END_ANALYSIS_OUTPUT_SCHEMA\n"
    )


def _content_hashes(prompt: str) -> dict[str, Any]:
    encoded = prompt.encode("utf-8")
    return {
        "prompt_bytes": len(encoded),
        "prompt_sha256": hashlib.sha256(encoded).hexdigest(),
        "prompt_line_count": prompt.count("\n") + 1 if prompt else 0,
        "contains_untrusted_job_delimiters": (
            "BEGIN_UNTRUSTED_JOB_DESCRIPTION_" in prompt
            and "END_UNTRUSTED_JOB_DESCRIPTION_" in prompt
        ),
        "contains_trusted_resume_delimiters": (
            "BEGIN_TRUSTED_MASTER_RESUME_JSON" in prompt
            and "END_TRUSTED_MASTER_RESUME_JSON" in prompt
        ),
        "contains_output_schema_delimiters": (
            "BEGIN_ANALYSIS_OUTPUT_SCHEMA" in prompt
            and "END_ANALYSIS_OUTPUT_SCHEMA" in prompt
        ),
        "body_omitted": True,
    }


def _write_sanitized_prompt(run_directory: Path, prompt: str) -> Path:
    """Persist structure-only prompt diagnostics without résumé or job body text."""
    path = run_directory / GROK_ANALYSIS_PROMPT_FILENAME
    hashes = _content_hashes(prompt)
    lines = [
        "Grok analysis prompt (sanitized)",
        "Full prompt body omitted for privacy.",
        f"prompt_bytes={hashes['prompt_bytes']}",
        f"prompt_sha256={hashes['prompt_sha256']}",
        f"prompt_line_count={hashes['prompt_line_count']}",
        f"contains_untrusted_job_delimiters={hashes['contains_untrusted_job_delimiters']}",
        f"contains_trusted_resume_delimiters={hashes['contains_trusted_resume_delimiters']}",
        f"contains_output_schema_delimiters={hashes['contains_output_schema_delimiters']}",
        "thought_field_excluded=true",
        "credentials_excluded=true",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def _sanitized_transport_artifact(envelope: dict[str, Any]) -> dict[str, Any]:
    """Persist transport metadata without thought, usage, or full account data."""
    text = envelope.get("text")
    text_bytes = text.encode("utf-8") if isinstance(text, str) else b""
    artifact: dict[str, Any] = {
        "provider": "grok",
        "stopReason": envelope.get("stopReason"),
        "text_bytes": len(text_bytes),
        "text_sha256": hashlib.sha256(text_bytes).hexdigest() if text_bytes else None,
        "thought_present": "thought" in envelope
        and envelope.get("thought") not in (None, ""),
        "thought_excluded": True,
        "sessionId_present": "sessionId" in envelope,
        "requestId_present": "requestId" in envelope,
        "usage_present": "usage" in envelope or "modelUsage" in envelope,
        "modelUsage_omitted": True,
        "usage_omitted": True,
        "sessionId_omitted": True,
        "requestId_omitted": True,
        "credentials_excluded": True,
    }
    # Optional non-sensitive model name for diagnostics only; never required.
    model_usage = envelope.get("modelUsage")
    if isinstance(model_usage, dict):
        for key, value in model_usage.items():
            if isinstance(key, str) and "grok" in key.casefold() and isinstance(
                value, dict
            ):
                artifact["observed_model_key"] = key[:80]
                break
    return artifact


def _write_diagnostic(
    run_directory: Path,
    *,
    classification: str,
    detail: str | None = None,
    returncode: int | None = None,
    executable: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = run_directory / GROK_ANALYSIS_DIAGNOSTIC_FILENAME
    payload: dict[str, Any] = {
        "provider": "grok",
        "stage": "analysis",
        "classification": classification,
        "credentials_excluded": True,
        "thought_excluded": True,
        "environment_omitted": True,
        "telemetry_transmitted": False,
    }
    if detail is not None:
        payload["detail"] = detail[:500]
    if returncode is not None:
        payload["returncode"] = returncode
    if executable is not None:
        # Basename only — never full account paths that may leak home directories
        # with usernames is acceptable as path basename of the binary.
        payload["executable_basename"] = Path(executable).name
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)
    return path


def _combined_process_detail(stdout: str, stderr: str) -> str:
    return f"{stderr}\n{stdout}".casefold()


def _classify_nonzero_exit(
    *,
    stdout: str,
    stderr: str,
    returncode: int,
) -> ModelError:
    detail = _combined_process_detail(stdout, stderr)
    if any(marker in detail for marker in _USAGE_LIMIT_MARKERS):
        return GrokUsageLimitError(
            "Grok reported a usage or rate limit. Wait for the limit to reset "
            "or start a new run after capacity is available. No automatic "
            "provider fallback was attempted."
        )
    if any(marker in detail for marker in _AUTH_FAILURE_MARKERS):
        return GrokAuthenticationError()
    return GrokProcessError(
        f"Grok analysis exited with status {returncode}. Provider stdout and "
        "stderr content were omitted from the exception."
    )


def prepare_grok_analysis_schema(
    extracted_resume: dict[str, Any],
    job_requirements: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    """Write the same ID-constrained analysis schema used by Codex, for Grok."""
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
    path = run_directory / GROK_ANALYSIS_SCHEMA_FILENAME
    encoded = (
        json.dumps(transport, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(encoded) > 250_000:
        raise CodexSchemaCompatibilityError(
            "The generated analysis schema exceeds the local 250,000-byte safety "
            "limit; reduce the extracted source catalog before retrying."
        )
    atomic_write_json(path, transport)
    return {
        "schema": transport,
        "path": path.resolve(),
        "sha256": sha256_file(path),
        "size_bytes": len(encoded),
        "evidence_source_id_count": len(evidence_ids),
        "editable_source_id_count": len(editable_ids),
        "job_requirement_id_count": len(requirement_ids),
    }


def invoke_grok_analysis(
    *,
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    company: str,
    role: str,
    run_directory: Path,
    timeout_seconds: int,
    executable: str | None = None,
    progress_handler: Callable[[float, bool], None] | None = None,
) -> dict[str, Any]:
    """Invoke Grok Build for résumé analysis and return validated analysis JSON."""
    try:
        grok = resolve_grok_executable(executable)
    except GrokExecutableError as exc:
        _write_diagnostic(
            run_directory,
            classification=exc.classification,
            detail=str(exc),
        )
        raise

    schema_info = prepare_grok_analysis_schema(
        extracted_resume,
        job_requirements,
        run_directory,
    )
    prompt = build_grok_analysis_prompt(
        extracted_resume,
        job_description,
        job_requirements,
        company=company,
        role=role,
        output_schema=schema_info["schema"],
    )
    _write_sanitized_prompt(run_directory, prompt)
    args = grok_analysis_args(executable=grok, prompt=prompt)

    try:
        result = run_command(
            args,
            cwd=run_directory,
            timeout_seconds=timeout_seconds,
            heartbeat_handler=progress_handler,
        )
    except ModelError as exc:
        message = str(exc).casefold()
        if "timed out" in message:
            _write_diagnostic(
                run_directory,
                classification="timeout",
                detail=f"timeout_seconds={timeout_seconds}",
                executable=grok,
            )
            raise GrokTimeoutError(timeout_seconds) from exc
        if "could not run" in message:
            _write_diagnostic(
                run_directory,
                classification="executable_unavailable",
                detail="process start failed",
                executable=grok,
            )
            raise GrokExecutableError(
                f"Could not run Grok executable: {Path(grok).name}"
            ) from exc
        _write_diagnostic(
            run_directory,
            classification="generic_provider_failure",
            detail=type(exc).__name__,
            executable=grok,
        )
        raise
    except DependencyError as exc:
        cause = exc.__cause__
        if isinstance(cause, OSError) and cause.errno == errno.E2BIG:
            _write_diagnostic(
                run_directory,
                classification="prompt_too_large",
                detail="os_errno=E2BIG",
                executable=grok,
                extra={"prompt_contents_omitted": True},
            )
            raise GrokPromptTooLargeError() from exc
        _write_diagnostic(
            run_directory,
            classification="executable_unavailable",
            detail=type(exc).__name__,
            executable=grok,
        )
        raise GrokExecutableError(str(exc)) from exc

    if result.returncode != 0:
        error = _classify_nonzero_exit(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
        classification = getattr(error, "classification", "nonzero_exit")
        _write_diagnostic(
            run_directory,
            classification=classification,
            returncode=result.returncode,
            executable=grok,
            extra={
                "stdout_bytes": len(result.stdout.encode("utf-8")),
                "stderr_bytes": len(result.stderr.encode("utf-8")),
                "stdout_sha256": hashlib.sha256(
                    result.stdout.encode("utf-8")
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    result.stderr.encode("utf-8")
                ).hexdigest(),
                "provider_output_omitted": True,
            },
        )
        raise error

    try:
        envelope = parse_grok_transport_envelope(result.stdout)
    except GrokTransportEnvelopeError as exc:
        _write_diagnostic(
            run_directory,
            classification=exc.classification,
            detail=exc.detail,
            returncode=result.returncode,
            executable=grok,
            extra={
                "stdout_bytes": len(result.stdout.encode("utf-8")),
                "stdout_sha256": hashlib.sha256(
                    result.stdout.encode("utf-8")
                ).hexdigest(),
                "provider_output_omitted": True,
            },
        )
        raise

    atomic_write_json(
        run_directory / GROK_ANALYSIS_TRANSPORT_FILENAME,
        _sanitized_transport_artifact(envelope),
    )

    try:
        raw_payload = parse_grok_inner_analysis(envelope["text"])
    except GrokInnerAnalysisError as exc:
        _write_diagnostic(
            run_directory,
            classification=exc.classification,
            detail=exc.detail,
            returncode=result.returncode,
            executable=grok,
        )
        raise

    try:
        payload, warnings = normalize_unique_arrays(
            raw_payload,
            "codex_analysis.schema.json",
        )
        validate_payload(
            payload,
            "codex_analysis.schema.json",
            label="Grok analysis",
        )
    except ModelError as exc:
        location_match = re.search(r"validation at ([^:]+):", str(exc))
        location = location_match.group(1) if location_match else "model output"
        _write_diagnostic(
            run_directory,
            classification="schema_failure",
            detail=f"location={location}",
            returncode=result.returncode,
            executable=grok,
        )
        # Preserve the rejected payload under the Grok response name only after
        # stripping any non-analysis envelope fields (thought never present here).
        atomic_write_json(
            run_directory / GROK_ANALYSIS_RESPONSE_FILENAME,
            raw_payload if isinstance(raw_payload, dict) else {"invalid": True},
        )
        raise SourceEvidenceError(
            "Grok analysis failed local source-evidence validation: "
            f"the model response violated the canonical evidence contract at {location}."
        ) from exc

    atomic_write_json(run_directory / GROK_ANALYSIS_RESPONSE_FILENAME, payload)
    if warnings:
        atomic_write_json(
            run_directory / "codex-analysis-normalization-warnings.json",
            {
                "schema": "codex_analysis.schema.json",
                "provider": "grok",
                "policy": "exact-duplicate-removal",
                "warnings": warnings,
            },
        )
    _write_diagnostic(
        run_directory,
        classification="success",
        returncode=result.returncode,
        executable=grok,
        extra={
            "schema_filename": GROK_ANALYSIS_SCHEMA_FILENAME,
            "schema_sha256": schema_info["sha256"],
            "response_filename": GROK_ANALYSIS_RESPONSE_FILENAME,
            "thought_excluded": True,
        },
    )
    return payload
