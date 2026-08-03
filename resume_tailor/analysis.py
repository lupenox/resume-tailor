"""Provider-neutral résumé-analysis entrypoints.

Analysis providers may be ``gemma_local`` (default for new runs), ``codex``, or
``grok_cli``. Selection is always explicit at run time: the pipeline never
silently switches providers and never reuses a failed payload from one provider
as input to another.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .codex_analysis import build_analysis_prompt, invoke_codex_analysis, readable_analysis
from .utilities import InputError, atomic_write_json, utc_now_iso

# Public provider identifiers (CLI/UI values).
ANALYSIS_PROVIDERS = ("gemma_local", "codex", "grok_cli")
# Historical alias retained for recovery metadata and older tests.
_PROVIDER_ALIASES = {
    "grok": "grok_cli",
    "gemma": "gemma_local",
    "ollama": "gemma_local",
}

# Default for *new* runs. Historical recovery uses recorded metadata.provider and
# does not depend on this default, so switching the default does not break
# authenticated recovery of Codex/Grok runs.
DEFAULT_ANALYSIS_PROVIDER = "gemma_local"

# Canonical provider-neutral resolved analysis artifact.
ANALYSIS_RESOLVED_FILENAME = "analysis-resolved.json"
# Codex-only legacy compatibility alias (never written for Gemma or Grok runs).
CODEX_ANALYSIS_RESOLVED_FILENAME = "codex-analysis-resolved.json"
ANALYSIS_RESOLVED_DOCUMENT_VERSION = 1
ANALYSIS_CANONICAL_SCHEMA_NAME = "codex_analysis.schema.json"

_PROVIDER_LABELS = {
    "gemma_local": "Gemma Local",
    "codex": "Codex",
    "grok_cli": "Grok CLI",
}

_PROVIDER_WORKFLOW_LABELS = {
    "gemma_local": "Gemma Local analysis",
    "codex": "Codex analysis",
    "grok_cli": "Grok CLI analysis",
}


def normalize_analysis_provider(value: str | None) -> str:
    """Return a validated analysis-provider name; default is Gemma Local."""
    if value is None or value == "":
        return DEFAULT_ANALYSIS_PROVIDER
    provider = str(value).strip().casefold()
    provider = _PROVIDER_ALIASES.get(provider, provider)
    if provider not in ANALYSIS_PROVIDERS:
        raise InputError(
            f"Unsupported analysis provider: {value!r}. "
            f"Allowed values: {', '.join(ANALYSIS_PROVIDERS)}."
        )
    return provider


def analysis_provider_label(provider: str) -> str:
    """Human-readable short label for progress, approval, and error copy."""
    normalized = normalize_analysis_provider(provider)
    return _PROVIDER_LABELS[normalized]


def analysis_workflow_label(provider: str | None) -> str:
    """Workflow-stepper label for the analysis stage."""
    if provider is None or provider == "":
        # Unknown historical runs without metadata default to Codex wording so
        # pre-provider-metadata artifacts remain recognizable.
        return _PROVIDER_WORKFLOW_LABELS["codex"]
    try:
        normalized = normalize_analysis_provider(provider)
    except InputError:
        return _PROVIDER_WORKFLOW_LABELS["codex"]
    return _PROVIDER_WORKFLOW_LABELS[normalized]


def workflow_stages_for_provider(
    analysis_provider: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return workflow stage labels with the recorded analysis provider name."""
    analysis_label = analysis_workflow_label(analysis_provider)
    return (
        ("validating_input", "Validating input"),
        ("fetching_job", "Apify job retrieval"),
        ("confirming_posting", "Confirming posting"),
        ("codex_analysis", analysis_label),
        ("reviewing_changes", "Reviewing proposed changes"),
        ("antigravity_tailoring", "Local résumé tailoring"),
        ("evidence_validation", "Evidence validation"),
        ("rendering", "Rendering DOCX/PDF"),
        ("final_qa", "Initial Codex QA"),
        ("revision_phase", "Optional one-shot revision"),
        ("complete", "Complete"),
    )


def build_resolved_analysis_document(
    analysis: dict[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """Wrap a resolved analysis with explicit provider provenance metadata."""
    selected = normalize_analysis_provider(provider)
    return {
        "version": ANALYSIS_RESOLVED_DOCUMENT_VERSION,
        "provider": selected,
        "provider_label": analysis_provider_label(selected),
        "schema": ANALYSIS_CANONICAL_SCHEMA_NAME,
        "resolved_at": utc_now_iso(),
        "analysis": analysis,
    }


def unwrap_resolved_analysis_document(payload: Any) -> dict[str, Any]:
    """Return the bare analysis from a wrapper or a historical bare document.

    Historical ``codex-analysis-resolved.json`` files store the analysis object
    at the root. New ``analysis-resolved.json`` files wrap it under ``analysis``
    with explicit provider metadata.
    """
    if not isinstance(payload, dict):
        raise InputError("Resolved analysis artifact is not a JSON object.")
    nested = payload.get("analysis")
    if (
        payload.get("version") == ANALYSIS_RESOLVED_DOCUMENT_VERSION
        and isinstance(nested, dict)
        and isinstance(payload.get("provider"), str)
    ):
        return nested
    # Historical bare Codex analysis document (no provider wrapper).
    if "role_summary" in payload and "recommended_edits" in payload:
        return payload
    raise InputError(
        "Resolved analysis artifact is missing a valid analysis payload."
    )


def resolved_analysis_provider(payload: Any) -> str | None:
    """Return the recorded provider when the document is the neutral wrapper."""
    if not isinstance(payload, dict):
        return None
    provider = payload.get("provider")
    if not isinstance(provider, str):
        return None
    try:
        return normalize_analysis_provider(provider)
    except InputError:
        return None


def write_resolved_analysis_artifact(
    run_directory: Path,
    analysis: dict[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """Persist the provider-neutral resolved analysis artifact.

    Always writes ``analysis-resolved.json`` with provider metadata. For Codex
    runs only, also writes the legacy bare ``codex-analysis-resolved.json``
    alias so historical recovery tooling keeps working. Gemma Local and Grok
    results are never written under a Codex-named resolved artifact.
    """
    selected = normalize_analysis_provider(provider)
    document = build_resolved_analysis_document(analysis, provider=selected)
    neutral_path = run_directory / ANALYSIS_RESOLVED_FILENAME
    atomic_write_json(neutral_path, document)
    result: dict[str, Any] = {
        "filename": ANALYSIS_RESOLVED_FILENAME,
        "provider": selected,
        "provider_label": document["provider_label"],
        "legacy_codex_alias_written": False,
    }
    if selected == "codex":
        atomic_write_json(
            run_directory / CODEX_ANALYSIS_RESOLVED_FILENAME,
            analysis,
        )
        result["legacy_codex_alias_written"] = True
        result["legacy_codex_alias_filename"] = CODEX_ANALYSIS_RESOLVED_FILENAME
    return result


def invoke_analysis(
    *,
    provider: str = DEFAULT_ANALYSIS_PROVIDER,
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    company: str,
    role: str,
    run_directory: Path,
    timeout_seconds: int,
    executable: str | None = None,
    transport_artifact: Any | None = None,
    progress_handler: Callable[[float, bool], None] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the selected analysis provider and return the validated raw payload.

    Local schema validation, evidence resolution, character-budget checks, and
    structured-field authority remain Python-owned after this call returns.
    """
    selected = normalize_analysis_provider(provider)
    if selected == "codex":
        return invoke_codex_analysis(
            extracted_resume=extracted_resume,
            job_description=job_description,
            job_requirements=job_requirements,
            company=company,
            role=role,
            run_directory=run_directory,
            timeout_seconds=timeout_seconds,
            executable=executable,
            transport_artifact=transport_artifact,
            progress_handler=progress_handler,
        )
    if selected == "grok_cli":
        from .grok_analysis import invoke_grok_analysis

        return invoke_grok_analysis(
            extracted_resume=extracted_resume,
            job_description=job_description,
            job_requirements=job_requirements,
            company=company,
            role=role,
            run_directory=run_directory,
            timeout_seconds=timeout_seconds,
            executable=executable,
            progress_handler=progress_handler,
        )
    if selected == "gemma_local":
        from .gemma_analysis import invoke_gemma_analysis

        return invoke_gemma_analysis(
            extracted_resume=extracted_resume,
            job_description=job_description,
            job_requirements=job_requirements,
            company=company,
            role=role,
            run_directory=run_directory,
            timeout_seconds=timeout_seconds,
            model=model,
            progress_handler=progress_handler,
            status_handler=None,
        )
    raise InputError(f"Unsupported analysis provider: {provider!r}.")


__all__ = [
    "ANALYSIS_CANONICAL_SCHEMA_NAME",
    "ANALYSIS_PROVIDERS",
    "ANALYSIS_RESOLVED_DOCUMENT_VERSION",
    "ANALYSIS_RESOLVED_FILENAME",
    "CODEX_ANALYSIS_RESOLVED_FILENAME",
    "DEFAULT_ANALYSIS_PROVIDER",
    "analysis_provider_label",
    "analysis_workflow_label",
    "build_analysis_prompt",
    "build_resolved_analysis_document",
    "invoke_analysis",
    "normalize_analysis_provider",
    "readable_analysis",
    "resolved_analysis_provider",
    "unwrap_resolved_analysis_document",
    "workflow_stages_for_provider",
    "write_resolved_analysis_artifact",
]
