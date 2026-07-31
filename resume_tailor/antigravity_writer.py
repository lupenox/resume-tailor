from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .antigravity_response import (
    AntigravityResponseCandidate,
    locate_json_tailoring_candidate,
)
from .antigravity_transport import (
    antigravity_parse_diagnostic,
    antigravity_process_failure,
    run_antigravity_prompt,
)
from .schemas import load_schema, parse_json_text, schema_path, validate_payload
from .utilities import (
    AntigravityCannotApplyError,
    AntigravityResponseEnvelopeError,
    AntigravityTailoringContractError,
    AntigravityTailoringPreflightError,
    AntigravityTechnicalFailureError,
    ModelError,
    atomic_write_json,
    atomic_write_text,
    require_executable,
    sha256_file,
)

ANTIGRAVITY_RESPONSE_METADATA_FILENAME = "antigravity-response-envelope.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _preflight_failure(reason: str) -> AntigravityTailoringPreflightError:
    return AntigravityTailoringPreflightError(
        "Local Antigravity tailoring preflight failed: "
        f"{reason} No provider request was launched."
    )


def approved_edit_catalog(approved_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    edits = approved_analysis.get("recommended_edits")
    if not isinstance(edits, list):
        raise _preflight_failure("the approved edit collection is missing.")
    catalog: list[dict[str, Any]] = []
    for index, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            raise _preflight_failure("an approved edit is not a structured object.")
        catalog.append(
            {
                **edit,
                "edit_id": f"edit.{index:03d}",
            }
        )
    return catalog


def preflight_tailoring_inputs(
    *,
    master_content: dict[str, Any],
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    approved_analysis: dict[str, Any],
    company: str,
    role: str,
) -> list[dict[str, Any]]:
    """Validate the immutable post-approval tailoring inputs without a provider."""
    if not isinstance(company, str) or not company.strip():
        raise _preflight_failure("the confirmed company is missing.")
    if not isinstance(role, str) or not role.strip():
        raise _preflight_failure("the confirmed role is missing.")
    if not isinstance(job_description, str) or not job_description.strip():
        raise _preflight_failure("the confirmed job description is missing.")
    if not isinstance(master_content, dict) or not master_content:
        raise _preflight_failure("the extracted master résumé content is missing.")
    extracted_content = extracted_resume.get("content")
    if (
        not isinstance(extracted_content, dict)
        or _canonical_json(master_content) != _canonical_json(extracted_content)
    ):
        raise _preflight_failure(
            "the supplied master content does not match the authenticated extraction."
        )
    source_blocks = extracted_resume.get("source_blocks")
    if not isinstance(source_blocks, list) or not source_blocks:
        raise _preflight_failure("the immutable résumé source catalog is missing.")
    if approved_analysis.get("questions_for_user"):
        raise _preflight_failure(
            "the approved analysis still contains an unanswered factual question."
        )

    try:
        from .job_requirements import validate_job_requirement_catalog

        validate_job_requirement_catalog(
            job_requirements,
            job_description=job_description.rstrip("\n"),
        )
    except Exception as exc:
        raise _preflight_failure(
            "the confirmed job-requirement catalog is invalid."
        ) from exc

    try:
        from .evidence import resolve_analysis_evidence

        resolved, issues = resolve_analysis_evidence(
            approved_analysis,
            extracted_resume,
            job_requirements,
        )
    except Exception as exc:
        raise _preflight_failure(
            "the approved analysis could not be resolved against local catalogs."
        ) from exc
    if issues or _canonical_json(resolved) != _canonical_json(approved_analysis):
        raise _preflight_failure(
            "the approved analysis no longer resolves exactly against local catalogs."
        )

    blocks_by_id: dict[str, dict[str, Any]] = {}
    for block in source_blocks:
        source_id = block.get("source_id") if isinstance(block, dict) else None
        if not isinstance(source_id, str) or not source_id or source_id in blocks_by_id:
            raise _preflight_failure("the résumé source catalog contains invalid IDs.")
        blocks_by_id[source_id] = block

    catalog = approved_edit_catalog(approved_analysis)
    for edit in catalog:
        target_id = edit.get("target_source_id")
        target = blocks_by_id.get(target_id)
        if target is None or target.get("editable") is not True:
            raise _preflight_failure(
                "an approved edit target is missing or not locally editable."
            )
        if edit.get("existing_text") != target.get("exact_text"):
            raise _preflight_failure(
                "an approved edit no longer matches its exact local target text."
            )
        evidence_ids = edit.get("evidence_source_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) != len(set(evidence_ids))
        ):
            raise _preflight_failure(
                "an approved edit has missing or duplicate evidence references."
            )
        if any(
            source_id not in blocks_by_id
            or blocks_by_id[source_id].get("evidence_allowed") is not True
            for source_id in evidence_ids
        ):
            raise _preflight_failure(
                "an approved edit references unavailable résumé evidence."
            )
    return catalog


def build_tailoring_prompt(
    *,
    master_content: dict[str, Any],
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    approved_analysis: dict[str, Any],
    company: str,
    role: str,
) -> str:
    source_blocks = extracted_resume["source_blocks"]
    edits = preflight_tailoring_inputs(
        master_content=master_content,
        extracted_resume=extracted_resume,
        job_description=job_description,
        job_requirements=job_requirements,
        approved_analysis=approved_analysis,
        company=company,
        role=role,
    )
    budgets = [
        {
            "content_id": paragraph["content_id"],
            **paragraph["content_budget"],
        }
        for paragraph in extracted_resume["paragraphs"]
    ]
    return f"""Execute this already-approved resume transformation now.
This is a bounded content-transformation task, not a planning or factual-
discovery task. Return only the strict structured result required by the
supplied JSON schema.
Do not use Markdown. Do not edit or write any file. Do not execute commands,
call tools, invoke agents, or initiate applications.

TARGET
Company: {company}
Role: {role}

MASTER RESUME CONTENT (TRUSTED FACTUAL SOURCE)
BEGIN_TRUSTED_MASTER_RESUME_CONTENT
{json.dumps(master_content, ensure_ascii=False, indent=2)}
END_TRUSTED_MASTER_RESUME_CONTENT

IMMUTABLE SOURCE CATALOG (TRUSTED EXACT TEXT)
BEGIN_TRUSTED_SOURCE_CATALOG
{json.dumps(source_blocks, ensure_ascii=False, indent=2)}
END_TRUSTED_SOURCE_CATALOG

APPROVED EDIT CATALOG (IMMUTABLE PLAN)
BEGIN_APPROVED_EDIT_CATALOG
{json.dumps(edits, ensure_ascii=False, indent=2)}
END_APPROVED_EDIT_CATALOG

IMMUTABLE FACTS
{json.dumps(approved_analysis["immutable_facts"], ensure_ascii=False, indent=2)}

FORBIDDEN CLAIMS
{json.dumps(approved_analysis["forbidden_claims"], ensure_ascii=False, indent=2)}

FORMATTING AND LENGTH CONSTRAINTS
{json.dumps(budgets, ensure_ascii=False, indent=2)}

NON-NEGOTIABLE RULES
- Apply every approved edit and no unapproved edit. Preserve all other content.
- The confirmed job has already been analyzed and approved. Do not reopen
  requirement discovery, infer new edits, or ask the user any factual question.
- Unsupported requirements were intentionally omitted. Never request missing
  skills, experience, metrics, credentials, or accomplishments and never add them.
- The master resume is the sole factual authority. Analysis cannot create evidence.
- Existing text and evidence in the approved analysis were resolved locally from
  the immutable source catalog. Never replace them with model-generated wording.
- Never combine separate source blocks into a quotation or treat proposed_text as
  source evidence.
- Tailor wording and emphasis only when existing source evidence supports it.
- Preserve institution, degree details, certification status, employment facts and
  dates, project names, open-source identity/link context, and all numeric claims.
- Preserve exactly three skill groups, three projects, each project's existing
  bullet count, one open-source contribution, and one employment entry.
- Preserve the current order and labels so content can be inserted deterministically.
- Each logical paragraph must stay within its corresponding maximum-character budget.
- Technologies in skill groups and project technology lines must already occur in
  the master resume. Reordering is allowed; invention is not.
- Never claim RAG, GraphQL, observability, distributed production scale, IVR
  platforms, or another absent technology or capability.
- Use only ATS terms whose locally derived status is present_verbatim or
  supported_by_source. Never insert an unsupported ATS term as a factual claim.
- Avoid first-person pronouns. Use concise, accomplishment-oriented language.
- Never invent metrics, certifications, availability, seniority, leadership,
  employment, or customer impact.
- Return status complete with the full tailored content when all approved edits
  can be applied.
- If one approved edit cannot be applied safely, return status cannot_apply with
  that local edit_id and one bounded reason code. Do not ask for more information.
- Use technical_failure only for a genuine execution or structured-output problem.
- Never return WAITING, needs_input, a plan, a readiness statement, a question,
  or a request for another task. Never guess.
"""


def _write_response_metadata(
    *,
    run_directory: Path,
    response_path: Path,
    envelope_type: str,
    validation_result: str,
) -> dict[str, Any]:
    schema = schema_path("tailored_resume.schema.json")
    metadata = {
        "version": 1,
        "provider": "antigravity",
        "execution_mode": "print",
        "agent_mode": "default",
        "output_format": "json",
        "sandboxed": True,
        "response_envelope_type": envelope_type,
        "validation_result": validation_result,
        "schema": {
            "filename": schema.name,
            "sha256": sha256_file(schema),
        },
        "response": {
            "filename": response_path.name,
            "sha256": sha256_file(response_path),
        },
    }
    atomic_write_json(
        run_directory / ANTIGRAVITY_RESPONSE_METADATA_FILENAME,
        metadata,
    )
    return metadata


def load_antigravity_response_metadata(
    run_directory: Path,
) -> dict[str, Any] | None:
    path = run_directory / ANTIGRAVITY_RESPONSE_METADATA_FILENAME
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 50_000:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_tailoring_candidate(
    candidate: AntigravityResponseCandidate,
    *,
    approved_analysis: dict[str, Any],
) -> dict[str, Any]:
    payload = candidate.payload


    status = payload.get("status") if isinstance(payload, dict) else None
    if status == "WAITING":
        raise AntigravityTailoringContractError(
            "Antigravity did not execute the approved tailoring task and returned "
            "a generic request for more information. The provider message was "
            "omitted; no factual input is requested."
        )
    try:
        validate_payload(
            payload,
            "tailored_resume.schema.json",
            label="Antigravity output",
        )
    except ModelError as exc:
        raise AntigravityTailoringContractError(
            "Antigravity violated the post-approval tailoring response contract. "
            "Provider content was omitted from the exception."
        ) from exc

    if payload["status"] == "cannot_apply":
        detail = payload["cannot_apply"]
        allowed_ids = {
            edit["edit_id"] for edit in approved_edit_catalog(approved_analysis)
        }
        if detail["edit_id"] not in allowed_ids:
            raise AntigravityTailoringContractError(
                "Antigravity returned an unknown approved edit ID in cannot_apply."
            )
        raise AntigravityCannotApplyError(
            "Antigravity could not apply approved "
            f"{detail['edit_id']} ({detail['reason_code']}). No factual input is "
            "requested; the approved artifacts were preserved for review."
        )
    if payload["status"] == "technical_failure":
        detail = payload["technical_failure"]
        raise AntigravityTechnicalFailureError(
            "Antigravity reported technical failure "
            f"{detail['reason_code']}. Provider text was omitted from the exception."
        )
    return payload["tailored_resume"]


def resolve_tailoring_response_with_envelope(
    raw_payload: Any,
    *,
    approved_analysis: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Resolve one documented stored response without launching a provider."""
    candidate = locate_json_tailoring_candidate(
        raw_payload,
        expected_schema=load_schema("tailored_resume.schema.json"),
    )
    return (
        _resolve_tailoring_candidate(
            candidate,
            approved_analysis=approved_analysis,
        ),
        candidate.envelope_type,
    )


def resolve_tailoring_response(
    raw_payload: Any,
    *,
    approved_analysis: dict[str, Any],
) -> dict[str, Any]:
    content, _ = resolve_tailoring_response_with_envelope(
        raw_payload,
        approved_analysis=approved_analysis,
    )
    return content


def invoke_antigravity(
    *,
    master_content: dict[str, Any],
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    approved_analysis: dict[str, Any],
    company: str,
    role: str,
    run_directory: Path,
    timeout_seconds: int,
    antigravity_duration: str,
    executable: str | None = None,
) -> dict[str, Any]:
    agy = executable or require_executable("agy")
    prompt = build_tailoring_prompt(
        master_content=master_content,
        extracted_resume=extracted_resume,
        job_description=job_description,
        job_requirements=job_requirements,
        approved_analysis=approved_analysis,
        company=company,
        role=role,
    )
    result = run_antigravity_prompt(
        executable=agy,
        prompt=prompt,
        prompt_label="Antigravity tailoring prompt",
        schema=schema_path("tailored_resume.schema.json"),
        print_timeout=antigravity_duration,
        cwd=run_directory,
        timeout_seconds=timeout_seconds + 10,
    )
    response_path = run_directory / "antigravity-response.json"
    try:
        raw_payload = parse_json_text(result.stdout, label="Antigravity")
    except ModelError as exc:
        atomic_write_json(
            response_path,
            antigravity_parse_diagnostic(result),
        )
        _write_response_metadata(
            run_directory=run_directory,
            response_path=response_path,
            envelope_type="malformed-json-output",
            validation_result="REJECTED",
        )
        if result.returncode != 0:
            raise antigravity_process_failure(result, label="Antigravity")
        raise AntigravityResponseEnvelopeError(
            "Antigravity returned malformed JSON output.",
            envelope_type="malformed-json-output",
        ) from exc
    # Preserve the exact valid UTF-8 JSON bytes returned by print mode so its
    # recorded hash authenticates the provider response used by local parsing.
    atomic_write_text(response_path, result.stdout)

    if result.returncode != 0:
        raise antigravity_process_failure(result, label="Antigravity")

    try:
        candidate = locate_json_tailoring_candidate(
            raw_payload,
            expected_schema=load_schema("tailored_resume.schema.json"),
        )
    except AntigravityResponseEnvelopeError as exc:
        _write_response_metadata(
            run_directory=run_directory,
            response_path=response_path,
            envelope_type=exc.envelope_type,
            validation_result="REJECTED",
        )
        raise
    try:
        resolved = _resolve_tailoring_candidate(
            candidate,
            approved_analysis=approved_analysis,
        )
    except ModelError:
        _write_response_metadata(
            run_directory=run_directory,
            response_path=response_path,
            envelope_type=candidate.envelope_type,
            validation_result="REJECTED",
        )
        raise
    _write_response_metadata(
        run_directory=run_directory,
        response_path=response_path,
        envelope_type=candidate.envelope_type,
        validation_result="PASS",
    )
    return resolved
