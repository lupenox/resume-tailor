"""Antigravity CLI adapter for the résumé-analysis stage.

Antigravity uses the stream-json format similar to the tailoring stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resume_tailor.backend.providers.codex_analysis import build_analysis_prompt
from resume_tailor.backend.providers.antigravity_response import (
    AntigravityResponseCandidate,
    locate_stream_json_terminal,
    locate_json_tailoring_candidate,
    parse_stream_json_events,
)
from resume_tailor.backend.providers.antigravity_transport import (
    antigravity_parse_diagnostic,
    antigravity_process_failure,
    run_antigravity_prompt,
)
from resume_tailor.backend.utils.schemas import (
    build_codex_analysis_transport_schema,
    normalize_unique_arrays,
    validate_payload,
)
from resume_tailor.backend.utils.utilities import (
    AntigravityResponseEnvelopeError,
    CodexSchemaCompatibilityError,
    ModelError,
    SourceEvidenceError,
    atomic_write_json,
    atomic_write_text,
    require_executable,
    sha256_file,
)


ANTIGRAVITY_ANALYSIS_SCHEMA_FILENAME = "antigravity-analysis-schema.json"
ANTIGRAVITY_ANALYSIS_RESPONSE_FILENAME = "antigravity-analysis-response.json"
ANTIGRAVITY_ANALYSIS_METADATA_FILENAME = "antigravity-analysis-metadata.json"


def prepare_antigravity_analysis_schema(
    extracted_resume: dict[str, Any],
    job_requirements: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    """Write the ID-constrained analysis schema used by Codex, for Antigravity."""
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
    path = run_directory / ANTIGRAVITY_ANALYSIS_SCHEMA_FILENAME
    encoded = (
        json.dumps(transport, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write_json(path, transport)
    return {
        "schema": transport,
        "path": path.resolve(),
        "sha256": sha256_file(path),
        "size_bytes": len(encoded),
    }


def invoke_antigravity_analysis(
    *,
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    company: str,
    role: str,
    run_directory: Path,
    timeout_seconds: int,
    executable: str | None = None,
    model: str | None = None,
    model_strength: str | None = None,
) -> dict[str, Any]:
    """Invoke Antigravity CLI for résumé analysis and return validated analysis JSON."""
    agy = executable or require_executable("agy")

    schema_info = prepare_antigravity_analysis_schema(
        extracted_resume,
        job_requirements,
        run_directory,
    )
    prompt = build_analysis_prompt(
        extracted_resume,
        job_description,
        job_requirements,
        company=company,
        role=role,
    )
    
    result = run_antigravity_prompt(
        executable=agy,
        prompt=prompt,
        prompt_label="Antigravity analysis prompt",
        schema=schema_info["path"],
        print_timeout="60s",
        cwd=run_directory,
        timeout_seconds=timeout_seconds,
        model=model,
        model_strength=model_strength,
    )
    
    response_path = run_directory / ANTIGRAVITY_ANALYSIS_RESPONSE_FILENAME
    
    try:
        events = parse_stream_json_events(result.stdout)
    except AntigravityResponseEnvelopeError as exc:
        atomic_write_json(
            run_directory / "antigravity-analysis-diagnostic.json",
            antigravity_parse_diagnostic(result),
        )
        if result.returncode != 0:
            raise antigravity_process_failure(result, label="Antigravity")
        raise
        
    atomic_write_text(run_directory / "antigravity-analysis-raw.txt", result.stdout)

    if result.returncode != 0:
        raise antigravity_process_failure(result, label="Antigravity")

    try:
        envelope, stream_type = locate_stream_json_terminal(events)
        candidate = locate_json_tailoring_candidate(
            envelope,
            expected_schema=schema_info["schema"],
        )
    except AntigravityResponseEnvelopeError:
        raise

    raw_payload = candidate.payload
    try:
        payload, warnings = normalize_unique_arrays(
            raw_payload,
            "codex_analysis.schema.json",
        )
        validate_payload(
            payload,
            "codex_analysis.schema.json",
            label="Antigravity analysis",
        )
    except ModelError as exc:
        raise SourceEvidenceError(
            "Antigravity analysis failed local source-evidence validation."
        ) from exc

    atomic_write_json(response_path, payload)
    return payload
