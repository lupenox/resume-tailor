"""Two-phase Gemma Local analysis tests (mocked Ollama only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.backend.engine.analysis import (
    ANALYSIS_RESOLVED_FILENAME,
    CODEX_ANALYSIS_RESOLVED_FILENAME,
    DEFAULT_ANALYSIS_PROVIDER,
    write_resolved_analysis_artifact,
)
from resume_tailor.backend.providers.codex_analysis import build_analysis_prompt
from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.evidence import resolve_analysis_evidence
from resume_tailor.backend.providers.gemma_analysis import (
    DEFAULT_COVERAGE_MAX_OUTPUT_TOKENS,
    resolve_coverage_batch_max_output_tokens,
    resolve_coverage_batch_size,
    COVERAGE_DIAGNOSTIC_FILENAME,
    COVERAGE_SCHEMA_FILENAME,
    DEFAULT_COVERAGE_BATCH_MAX_OUTPUT_TOKENS,
    DEFAULT_EDIT_MAX_OUTPUT_TOKENS,
    EDITS_DIAGNOSTIC_FILENAME,
    MAX_GEMMA_ANALYSIS_EDITS,
    allowed_evidence_source_ids_for_target,
    assemble_canonical_analysis,
    build_coverage_prompt,
    build_coverage_schema,
    build_edits_prompt,
    build_edits_schema,
    build_target_requirement_evidence_matrix,
    estimate_prompt_tokens,
    evidence_compatible_with_target,
    gemma_analysis_chat_request_for_tests,
    invoke_gemma_analysis,
    parse_exact_analysis_json,
    resolve_coverage_max_output_tokens,
    resolve_edit_max_output_tokens,
    validate_coverage_payload,
    validate_edits_payload,
)
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.utils.utilities import (
    GemmaAnalysisTimeoutError,
    GemmaOutputLimitError,
    OllamaRequestError,
    SourceEvidenceError,
)


def _job_catalog() -> dict:
    return build_job_requirement_catalog(
        "Skills: Python and RAG.",
        structured_job={"technologies_and_skills": ["Python", "RAG"]},
    )


def _requirement_ids(catalog: dict) -> list[str]:
    return [item["requirement_id"] for item in catalog["requirements"]]


def _coverage_body(catalog: dict, *, support_first: bool = True) -> dict[str, Any]:
    ids = _requirement_ids(catalog)
    requirements = []
    for index, requirement_id in enumerate(ids):
        if support_first and index == 0:
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "status": "supported",
                    "evidence_source_ids": ["skill_groups.0"],
                }
            )
        else:
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "status": "unsupported",
                    "evidence_source_ids": [],
                }
            )
    return {"requirements": requirements}


def _edits_body() -> dict[str, Any]:
    return {
        "edits": [
            {
                "target_source_id": "professional_summary",
                "requirement_ids": [],  # filled by test when known
                "evidence_source_ids": ["professional_summary"],
                "proposed_text": "Python-focused engineer with agentic workflow experience.",
            }
        ]
    }


def _ollama_body(
    content: str | dict[str, Any],
    *,
    done_reason: str = "stop",
    eval_count: int | None = 20,
) -> dict[str, Any]:
    if isinstance(content, dict):
        text = json.dumps(content, ensure_ascii=False)
    else:
        text = content
    body: dict[str, Any] = {
        "model": "resume-tailor-gemma",
        "done": True,
        "done_reason": done_reason,
        "message": {"role": "assistant", "content": text},
        "prompt_eval_count": 10,
    }
    if eval_count is not None:
        body["eval_count"] = eval_count
    return body


@pytest.fixture
def mock_ollama(monkeypatch: pytest.MonkeyPatch):
    state: dict[str, Any] = {"bodies": [], "requests": [], "calls": 0, "errors": []}

    def _set_bodies(*bodies: dict[str, Any]) -> None:
        state["bodies"] = list(bodies)
        state["errors"] = []

    def _set_errors(*errors: BaseException) -> None:
        state["errors"] = list(errors)
        state["bodies"] = []

    def fake_run_ollama_request(**kwargs: Any) -> dict[str, Any]:
        state["calls"] += 1
        state["requests"].append(kwargs)
        if state["errors"]:
            raise state["errors"].pop(0)
        if not state["bodies"]:
            raise OllamaRequestError(
                "refused",
                classification="connection_refused",
            )
        return state["bodies"].pop(0)

    monkeypatch.setattr(
        "resume_tailor.backend.providers.gemma_analysis.run_ollama_request",
        fake_run_ollama_request,
    )
    state["set_bodies"] = _set_bodies
    state["set_errors"] = _set_errors
    return state


def test_defaults_and_token_ceilings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DEFAULT_ANALYSIS_PROVIDER == "gemma_local"
    monkeypatch.delenv("GEMMA_ANALYSIS_COVERAGE_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GEMMA_ANALYSIS_EDIT_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", raising=False)
    assert resolve_coverage_max_output_tokens(None) == DEFAULT_COVERAGE_MAX_OUTPUT_TOKENS
    assert resolve_edit_max_output_tokens(None) == DEFAULT_EDIT_MAX_OUTPUT_TOKENS
    assert MAX_GEMMA_ANALYSIS_EDITS == 8
    # Blank / invalid phase-specific values fall through to defaults.
    monkeypatch.setenv("GEMMA_ANALYSIS_COVERAGE_MAX_OUTPUT_TOKENS", "   ")
    assert resolve_coverage_max_output_tokens(None) == DEFAULT_COVERAGE_MAX_OUTPUT_TOKENS
    monkeypatch.setenv("GEMMA_ANALYSIS_EDIT_MAX_OUTPUT_TOKENS", "not-a-number")
    assert resolve_edit_max_output_tokens(None) == DEFAULT_EDIT_MAX_OUTPUT_TOKENS


def test_legacy_and_phase_token_env_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS must not inflate both phases."""
    for name in (
        "GEMMA_ANALYSIS_COVERAGE_MAX_OUTPUT_TOKENS",
        "GEMMA_ANALYSIS_EDIT_MAX_OUTPUT_TOKENS",
        "GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    # Only new phase-specific variables.
    monkeypatch.setenv("GEMMA_ANALYSIS_COVERAGE_MAX_OUTPUT_TOKENS", "900")
    monkeypatch.setenv("GEMMA_ANALYSIS_EDIT_MAX_OUTPUT_TOKENS", "1200")
    assert resolve_coverage_max_output_tokens(None) == 900
    assert resolve_edit_max_output_tokens(None) == 1200

    # Only legacy: cap defaults downward; 4096 must not raise either phase.
    monkeypatch.delenv("GEMMA_ANALYSIS_COVERAGE_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GEMMA_ANALYSIS_EDIT_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", "4096")
    assert resolve_coverage_max_output_tokens(None) == DEFAULT_COVERAGE_MAX_OUTPUT_TOKENS
    assert resolve_edit_max_output_tokens(None) == DEFAULT_EDIT_MAX_OUTPUT_TOKENS
    monkeypatch.setenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", "512")
    assert resolve_coverage_max_output_tokens(None) == 512
    assert resolve_edit_max_output_tokens(None) == 512

    # Both legacy and new: phase-specific wins; legacy is ignored for those phases.
    monkeypatch.setenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("GEMMA_ANALYSIS_COVERAGE_MAX_OUTPUT_TOKENS", "800")
    monkeypatch.setenv("GEMMA_ANALYSIS_EDIT_MAX_OUTPUT_TOKENS", "1100")
    assert resolve_coverage_max_output_tokens(None) == 800
    assert resolve_edit_max_output_tokens(None) == 1100

    # Blank / invalid legacy ignored.
    monkeypatch.delenv("GEMMA_ANALYSIS_COVERAGE_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GEMMA_ANALYSIS_EDIT_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", "   ")
    assert resolve_coverage_max_output_tokens(None) == DEFAULT_COVERAGE_MAX_OUTPUT_TOKENS
    monkeypatch.setenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", "abc")
    assert resolve_edit_max_output_tokens(None) == DEFAULT_EDIT_MAX_OUTPUT_TOKENS


def test_phase_schemas_and_chat_request_shape() -> None:
    coverage = build_coverage_schema(
        requirement_ids=["skill.001", "skill.002"],
        evidence_ids=["skill_groups.0", "professional_summary"],
    )
    assert coverage["properties"]["requirements"]["minItems"] == 2
    assert coverage["properties"]["requirements"]["maxItems"] == 2
    edits = build_edits_schema(
        editable_ids=["professional_summary"],
        evidence_ids=["skill_groups.0"],
        eligible_requirement_ids=["skill.001"],
        max_edits=8,
    )
    assert edits["properties"]["edits"]["maxItems"] == 8
    request = gemma_analysis_chat_request_for_tests(
        model="resume-tailor-gemma",
        prompt="x",
        format_schema=coverage,
        max_output_tokens=1536,
    )
    assert request["stream"] is False
    assert request["options"]["temperature"] == 0
    assert request["options"]["num_predict"] == 1536
    assert request.get("think") is False


def test_prompt_compaction_two_phase_vs_legacy(
    master_resume: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    job = "Skills: Python and RAG. Build agentic systems."
    legacy = build_analysis_prompt(
        extracted,
        job,
        catalog,
        company="Example",
        role="Developer",
    )
    coverage_prompt = build_coverage_prompt(
            extracted_resume=extracted,
            batch_requirements=catalog["requirements"],
            company="Example",
            role="Developer",
        )
    coverage_payload = _coverage_body(catalog)
    edits_prompt = build_edits_prompt(
        extracted_resume=extracted,
        coverage=coverage_payload,
        company="Example",
        role="Developer",
    )
    assert len(coverage_prompt.encode()) < len(legacy.encode())
    assert len(edits_prompt.encode()) < len(legacy.encode())
    assert "SOURCE_CATALOG" in coverage_prompt
    assert "JOB_REQUIREMENTS" in coverage_prompt
    assert "max_edits" in edits_prompt
    assert "skill_groups.N" in edits_prompt
    assert estimate_prompt_tokens(coverage_prompt) < estimate_prompt_tokens(legacy)


def test_successful_two_phase_analysis_and_assembly(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    coverage = _coverage_body(catalog)
    supported_id = coverage["requirements"][0]["requirement_id"]
    edits = {
        "edits": [
            {
                "target_source_id": "professional_summary",
                "requirement_ids": [supported_id],
                "evidence_source_ids": ["skill_groups.0"],
                "proposed_text": "Python engineer with production agent workflows.",
            }
        ]
    }
    mock_ollama["set_bodies"](_ollama_body(coverage), _ollama_body(edits))
    statuses: list[str] = []
    payload = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=catalog,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=60,
        status_handler=statuses.append,
    )
    assert payload["supported_requirement_mappings"]
    assert payload["unsupported_requirement_ids"]
    assert payload["recommended_edits"]
    assert "role_summary" in payload
    assert mock_ollama["calls"] == 2
    assert mock_ollama["requests"][0]["body"]["options"]["num_predict"] == (
        DEFAULT_COVERAGE_BATCH_MAX_OUTPUT_TOKENS
    )
    assert mock_ollama["requests"][1]["body"]["options"]["num_predict"] == (
        DEFAULT_EDIT_MAX_OUTPUT_TOKENS
    )
    assert any("Mapping job requirements" in s for s in statuses)
    assert "Planning résumé edits" in statuses
    assert (tmp_path / "gemma-analysis-coverage-batch-001-schema.json").is_file()
    assert (tmp_path / COVERAGE_DIAGNOSTIC_FILENAME).is_file()
    assert (tmp_path / EDITS_DIAGNOSTIC_FILENAME).is_file()

    resolved, issues = resolve_analysis_evidence(payload, extracted, catalog)
    assert issues == []
    meta = write_resolved_analysis_artifact(
        tmp_path, resolved, provider="gemma_local"
    )
    assert meta["provider"] == "gemma_local"
    assert not (tmp_path / CODEX_ANALYSIS_RESOLVED_FILENAME).exists()
    document = json.loads(
        (tmp_path / ANALYSIS_RESOLVED_FILENAME).read_text(encoding="utf-8")
    )
    assert document["provider"] == "gemma_local"


def test_missing_and_duplicate_requirement_ids_rejected() -> None:
    requirement_ids = ["skill.001", "skill.002"]
    evidence = {"skill_groups.0"}
    with pytest.raises(SourceEvidenceError, match="missing requirement"):
        validate_coverage_payload(
            {
                "requirements": [
                    {
                        "requirement_id": "skill.001",
                        "status": "unsupported",
                        "evidence_source_ids": [],
                    }
                ]
            },
            requirement_ids=requirement_ids,
            evidence_ids=evidence,
        )
    with pytest.raises(SourceEvidenceError, match="duplicate"):
        validate_coverage_payload(
            {
                "requirements": [
                    {
                        "requirement_id": "skill.001",
                        "status": "unsupported",
                        "evidence_source_ids": [],
                    },
                    {
                        "requirement_id": "skill.001",
                        "status": "supported",
                        "evidence_source_ids": ["skill_groups.0"],
                    },
                ]
            },
            requirement_ids=requirement_ids,
            evidence_ids=evidence,
        )


def test_invalid_evidence_ids_rejected_in_coverage() -> None:
    with pytest.raises(SourceEvidenceError, match="invalid evidence"):
        validate_coverage_payload(
            {
                "requirements": [
                    {
                        "requirement_id": "skill.001",
                        "status": "supported",
                        "evidence_source_ids": ["not.a.source"],
                    }
                ]
            },
            requirement_ids=["skill.001"],
            evidence_ids={"skill_groups.0"},
        )


def test_unsupported_must_not_cite_evidence() -> None:
    with pytest.raises(SourceEvidenceError, match="must not cite evidence"):
        validate_coverage_payload(
            {
                "requirements": [
                    {
                        "requirement_id": "skill.001",
                        "status": "unsupported",
                        "evidence_source_ids": ["skill_groups.0"],
                    }
                ]
            },
            requirement_ids=["skill.001"],
            evidence_ids={"skill_groups.0"},
        )


def test_edit_count_maximum_enforced() -> None:
    edits = {
        "edits": [
            {
                "target_source_id": f"professional_summary",
                "requirement_ids": ["skill.001"],
                "evidence_source_ids": ["skill_groups.0"],
                "proposed_text": "text",
            }
        ]
        * 9
    }
    # Force distinct targets for max check first
    edits = {
        "edits": [
            {
                "target_source_id": f"t{i}",
                "requirement_ids": ["skill.001"],
                "evidence_source_ids": ["skill_groups.0"],
                "proposed_text": "text",
            }
            for i in range(9)
        ]
    }
    with pytest.raises(SourceEvidenceError, match="more than"):
        validate_edits_payload(
            edits,
            editable_ids={f"t{i}" for i in range(9)},
            evidence_ids={"skill_groups.0"},
            eligible_requirement_ids={"skill.001"},
            max_edits=8,
        )


def test_only_supported_requirements_enter_edit_eligibility(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    coverage = _coverage_body(catalog, support_first=True)
    unsupported_id = coverage["requirements"][1]["requirement_id"]
    supported_id = coverage["requirements"][0]["requirement_id"]
    # Attempt to attach unsupported requirement in edit → rejected, no Phase B repair for evidence
    bad_edits = {
        "edits": [
            {
                "target_source_id": "professional_summary",
                "requirement_ids": [unsupported_id],
                "evidence_source_ids": ["skill_groups.0"],
                "proposed_text": "text",
            }
        ]
    }
    mock_ollama["set_bodies"](_ollama_body(coverage), _ollama_body(bad_edits))
    with pytest.raises(SourceEvidenceError, match="not eligible"):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=catalog,
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=60,
        )
    # coverage + one edits attempt (no repair for evidence)
    assert mock_ollama["calls"] == 2
    assert supported_id != unsupported_id


def test_phase_a_timeout_prevents_phase_b(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    mock_ollama["set_errors"](
        OllamaRequestError("timeout", classification="timeout")
    )
    with pytest.raises(GemmaAnalysisTimeoutError):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    assert mock_ollama["calls"] == 1
    diagnostic = json.loads(
        (tmp_path / COVERAGE_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["phase"] in ("coverage", "coverage_summary")
    assert diagnostic["classification"] == "analysis_timeout"
    assert not (tmp_path / EDITS_DIAGNOSTIC_FILENAME).exists()


def test_phase_a_output_limit_prevents_phase_b(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    mock_ollama["set_bodies"](
        _ollama_body('{"requirements":', done_reason="length", eval_count=1536)
    )
    with pytest.raises(GemmaOutputLimitError):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            coverage_max_output_tokens=1536,
        )
    assert mock_ollama["calls"] == 1
    diagnostic = json.loads(
        (tmp_path / COVERAGE_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "output_limit_reached"


def test_phase_b_timeout_reports_edits_phase(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    coverage = _coverage_body(catalog)
    mock_ollama["set_bodies"](_ollama_body(coverage))
    mock_ollama["errors"] = [
        OllamaRequestError("timeout", classification="timeout")
    ]
    # After first body, next call uses errors — set after coverage body consumed
    def fake(**kwargs: Any) -> dict[str, Any]:
        mock_ollama["calls"] += 1
        mock_ollama["requests"].append(kwargs)
        if mock_ollama["calls"] == 1:
            return _ollama_body(coverage)
        raise OllamaRequestError("timeout", classification="timeout")

    import resume_tailor.backend.providers.gemma_analysis as ga

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ga, "run_ollama_request", fake)
    try:
        with pytest.raises(GemmaAnalysisTimeoutError):
            invoke_gemma_analysis(
                extracted_resume=extracted,
                job_description="Python role",
                job_requirements=catalog,
                company="Example",
                role="Developer",
                run_directory=tmp_path,
                timeout_seconds=30,
            )
        diagnostic = json.loads(
            (tmp_path / EDITS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
        )
        assert diagnostic["phase"] == "edits"
        assert diagnostic["classification"] == "analysis_timeout"
    finally:
        monkeypatch.undo()


def test_malformed_coverage_gets_one_repair(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    coverage = _coverage_body(catalog)
    supported_id = coverage["requirements"][0]["requirement_id"]
    edits = {
        "edits": [
            {
                "target_source_id": "professional_summary",
                "requirement_ids": [supported_id],
                "evidence_source_ids": ["skill_groups.0"],
                "proposed_text": "Python engineer.",
            }
        ]
    }
    mock_ollama["set_bodies"](
        _ollama_body("```json\n{}\n```"),
        _ollama_body(coverage),
        _ollama_body(edits),
    )
    payload = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=catalog,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=60,
    )
    assert payload["recommended_edits"]
    # coverage fail + repair + edits = 3
    assert mock_ollama["calls"] == 3
    repair_prompt = mock_ollama["requests"][1]["body"]["messages"][1]["content"]
    assert "failure_class=malformed_inner_analysis" in repair_prompt
    assert "```" not in repair_prompt or "failure_class" in repair_prompt


def test_no_retry_for_invalid_evidence(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    bad = _coverage_body(catalog)
    bad["requirements"][0]["evidence_source_ids"] = ["not.real"]
    bad["requirements"][0]["status"] = "supported"
    mock_ollama["set_bodies"](_ollama_body(bad))
    with pytest.raises(SourceEvidenceError, match="invalid evidence"):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=catalog,
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    assert mock_ollama["calls"] == 1


def test_assemble_canonical_passes_schema_and_evidence(
    master_resume: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    coverage = validate_coverage_payload(
        _coverage_body(catalog),
        requirement_ids=_requirement_ids(catalog),
        evidence_ids={
            block["source_id"]
            for block in extracted["source_blocks"]
            if block.get("evidence_allowed")
        },
    )
    supported_id = coverage["requirements"][0]["requirement_id"]
    edits = validate_edits_payload(
        {
            "edits": [
                {
                    "target_source_id": "professional_summary",
                    "requirement_ids": [supported_id],
                    "evidence_source_ids": ["skill_groups.0"],
                    "proposed_text": "Python-focused summary.",
                }
            ]
        },
        editable_ids={"professional_summary"},
        evidence_ids={"skill_groups.0", "professional_summary"},
        eligible_requirement_ids={supported_id},
    )
    analysis = assemble_canonical_analysis(
        coverage=coverage,
        edits=edits,
        company="Example",
        role="Developer",
        job_requirements=catalog,
    )
    resolved, issues = resolve_analysis_evidence(analysis, extracted, catalog)
    assert issues == []
    assert resolved["recommended_edits"][0]["existing_text"]


def test_parse_rejects_fences() -> None:
    with pytest.raises(Exception):
        parse_exact_analysis_json("```json\n{}\n```")


def test_unknown_requirement_id_rejected() -> None:
    with pytest.raises(SourceEvidenceError, match="unknown requirement_id"):
        validate_coverage_payload(
            {
                "requirements": [
                    {
                        "requirement_id": "skill.999",
                        "status": "unsupported",
                        "evidence_source_ids": [],
                    }
                ]
            },
            requirement_ids=["skill.001"],
            evidence_ids={"skill_groups.0"},
        )


def test_supported_without_evidence_rejected() -> None:
    with pytest.raises(SourceEvidenceError, match="needs evidence"):
        validate_coverage_payload(
            {
                "requirements": [
                    {
                        "requirement_id": "skill.001",
                        "status": "supported",
                        "evidence_source_ids": [],
                    }
                ]
            },
            requirement_ids=["skill.001"],
            evidence_ids={"skill_groups.0"},
        )
    with pytest.raises(SourceEvidenceError, match="needs evidence"):
        validate_coverage_payload(
            {
                "requirements": [
                    {
                        "requirement_id": "skill.001",
                        "status": "partially_supported",
                        "evidence_source_ids": [],
                    }
                ]
            },
            requirement_ids=["skill.001"],
            evidence_ids={"skill_groups.0"},
        )


def test_every_requirement_exactly_once_accepted() -> None:
    payload = validate_coverage_payload(
        {
            "requirements": [
                {
                    "requirement_id": "skill.001",
                    "status": "supported",
                    "evidence_source_ids": ["skill_groups.0"],
                },
                {
                    "requirement_id": "skill.002",
                    "status": "unsupported",
                    "evidence_source_ids": [],
                },
            ]
        },
        requirement_ids=["skill.001", "skill.002"],
        evidence_ids={"skill_groups.0"},
    )
    assert [item["requirement_id"] for item in payload["requirements"]] == [
        "skill.001",
        "skill.002",
    ]


def test_duplicate_phase_b_target_rejected() -> None:
    with pytest.raises(SourceEvidenceError, match="duplicate target_source_id"):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "professional_summary",
                        "requirement_ids": ["skill.001"],
                        "evidence_source_ids": ["skill_groups.0"],
                        "proposed_text": "First proposal.",
                    },
                    {
                        "target_source_id": "professional_summary",
                        "requirement_ids": ["skill.001"],
                        "evidence_source_ids": ["skill_groups.0"],
                        "proposed_text": "Competing proposal.",
                    },
                ]
            },
            editable_ids={"professional_summary"},
            evidence_ids={"skill_groups.0"},
            eligible_requirement_ids={"skill.001"},
            requirement_evidence={"skill.001": {"skill_groups.0"}},
        )


def test_unrelated_but_globally_valid_evidence_rejected() -> None:
    """A catalog-valid source ID that Phase A did not map must not pass Phase B.

    The target itself remains allowlisted (self-evidence). A different block that
    Phase A never cited for the requirement is still rejected.
    """
    with pytest.raises(SourceEvidenceError, match="unrelated evidence"):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "professional_summary",
                        "requirement_ids": ["skill.001"],
                        # experience bullet is catalog-eligible but not Phase A evidence
                        # and is not the target.
                        "evidence_source_ids": ["experience.0.bullets.0"],
                        "proposed_text": "Python-focused summary.",
                    }
                ]
            },
            editable_ids={"professional_summary"},
            evidence_ids={
                "skill_groups.0",
                "professional_summary",
                "experience.0.bullets.0",
            },
            eligible_requirement_ids={"skill.001"},
            requirement_evidence={"skill.001": {"skill_groups.0"}},
        )


def test_sibling_bullet_evidence_rejected_even_when_phase_a_cited_it() -> None:
    """Replay structure of the Jobright failure: .bullets.1 must not launder into .0."""
    with pytest.raises(
        SourceEvidenceError,
        match=r"unrelated evidence source_id 'projects\.2\.bullets\.1' for target "
        r"'projects\.2\.bullets\.0'",
    ):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "projects.2.bullets.0",
                        "requirement_ids": ["required.002"],
                        "evidence_source_ids": ["projects.2.bullets.1"],
                        "proposed_text": (
                            "Developed a local-first chat app with modular service "
                            "layer loading FLAN-T5 and TinyLlama via Ollama."
                        ),
                    }
                ]
            },
            editable_ids={
                "projects.2.bullets.0",
                "projects.2.bullets.1",
                "projects.2.bullets.2",
            },
            evidence_ids={
                "projects.2.bullets.0",
                "projects.2.bullets.1",
                "projects.2.bullets.2",
                "skill_groups.0",
            },
            eligible_requirement_ids={"required.002"},
            # Even if Phase A mapped the requirement to the sibling, laundering
            # sibling claims into another bullet is contractually forbidden.
            requirement_evidence={
                "required.002": {"projects.2.bullets.1", "skill_groups.0"}
            },
        )


def test_same_source_and_cross_section_evidence_still_pass() -> None:
    """Self-evidence and Phase-A skill→summary evidence remain valid."""
    same_source = validate_edits_payload(
        {
            "edits": [
                {
                    "target_source_id": "projects.2.bullets.0",
                    "requirement_ids": ["required.002"],
                    "evidence_source_ids": ["projects.2.bullets.0"],
                    "proposed_text": "Local-first desktop chat using Flask and Ollama.",
                }
            ]
        },
        editable_ids={"projects.2.bullets.0"},
        evidence_ids={"projects.2.bullets.0", "skill_groups.0"},
        eligible_requirement_ids={"required.002"},
        requirement_evidence={"required.002": {"projects.2.bullets.0"}},
    )
    assert same_source["edits"][0]["evidence_source_ids"] == [
        "projects.2.bullets.0"
    ]

    cross_section = validate_edits_payload(
        {
            "edits": [
                {
                    "target_source_id": "professional_summary",
                    "requirement_ids": ["skill.001"],
                    "evidence_source_ids": ["skill_groups.0"],
                    "proposed_text": "Python engineer with agentic workflow experience.",
                }
            ]
        },
        editable_ids={"professional_summary"},
        evidence_ids={"professional_summary", "skill_groups.0"},
        eligible_requirement_ids={"skill.001"},
        requirement_evidence={"skill.001": {"skill_groups.0"}},
    )
    assert cross_section["edits"][0]["evidence_source_ids"] == ["skill_groups.0"]


def test_legal_plus_illegal_sibling_does_not_become_valid_by_dropping() -> None:
    """Any illegal evidence ID rejects the edit; proposed_text is not retained.

    Silently stripping projects.2.bullets.1 while keeping proposed_text would
    launder sibling claims into a metadata-valid edit.
    """
    with pytest.raises(
        SourceEvidenceError,
        match=r"unrelated evidence source_id 'projects\.2\.bullets\.1'",
    ):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "projects.2.bullets.0",
                        "requirement_ids": ["required.002"],
                        "evidence_source_ids": [
                            "projects.2.bullets.0",  # legal
                            "projects.2.bullets.1",  # illegal sibling
                            "skill_groups.0",  # legal
                        ],
                        "proposed_text": (
                            "Local-first chat app with modular FLAN-T5 loading "
                            "from the sibling service layer."
                        ),
                    }
                ]
            },
            editable_ids={"projects.2.bullets.0", "projects.2.bullets.1"},
            evidence_ids={
                "projects.2.bullets.0",
                "projects.2.bullets.1",
                "skill_groups.0",
            },
            eligible_requirement_ids={"required.002"},
            requirement_evidence={
                "required.002": {"projects.2.bullets.0", "skill_groups.0"}
            },
        )


def test_illegal_evidence_fails_closed_without_model_regeneration(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    """Evidence-contract failures do not strip IDs or silently retry into validity."""
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    coverage = _coverage_body(catalog, support_first=True)
    supported_id = coverage["requirements"][0]["requirement_id"]
    bad_edits = {
        "edits": [
            {
                "target_source_id": "projects.0.bullets.0",
                "requirement_ids": [supported_id],
                "evidence_source_ids": [
                    "projects.0.bullets.0",
                    "projects.0.bullets.1",
                ],
                "proposed_text": "Rewritten with sibling claims.",
            }
        ]
    }
    # No second edits body: a silent repair would consume another Ollama call.
    mock_ollama["set_bodies"](_ollama_body(coverage), _ollama_body(bad_edits))
    with pytest.raises(SourceEvidenceError, match="unrelated evidence"):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=catalog,
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=60,
        )
    # coverage + one edits attempt; no regeneration, no provider fallback
    assert mock_ollama["calls"] == 2
    assert len(mock_ollama["requests"]) == 2


def test_edits_prompt_exposes_target_requirement_evidence_matrix(
    master_resume: Path,
) -> None:
    """Prompt must not over-union Phase A evidence across all requirements."""
    extracted, _ = extract_resume(master_resume)
    coverage = {
        "requirements": [
            {
                "requirement_id": "preferred.001",
                "status": "supported",
                "evidence_source_ids": ["skill_groups.2"],
            },
            {
                "requirement_id": "required.002",
                "status": "supported",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
    }
    prompt = build_edits_prompt(
        extracted_resume=extracted,
        coverage=coverage,
        company="Example",
        role="Developer",
    )
    assert "target_requirement_evidence" in prompt
    assert "target_evidence_allowlists" not in prompt
    assert "Do not invent requirement/evidence links" in prompt
    assert "skills→summary" in prompt or "skill" in prompt.casefold()
    assert "Sibling bullets" in prompt or "sibling bullet" in prompt.casefold()
    assert "ONLY facts contained in the selected legal" in prompt

    # Matrix: skill_groups.2 legal for summary only under preferred.001, not required.002
    evidence_ids = {
        "professional_summary",
        "skill_groups.0",
        "skill_groups.2",
        "projects.0.bullets.0",
        "projects.0.bullets.1",
    }
    requirement_evidence = {
        "preferred.001": {"skill_groups.2"},
        "required.002": {"skill_groups.0"},
    }
    matrix = build_target_requirement_evidence_matrix(
        editable_targets=["professional_summary", "projects.0.bullets.0"],
        requirement_evidence=requirement_evidence,
        evidence_ids=evidence_ids,
    )
    by_target = {row["target_source_id"]: row for row in matrix}
    summary_rows = {
        r["requirement_id"]: r["allowed_evidence_source_ids"]
        for r in by_target["professional_summary"]["requirement_evidence"]
    }
    assert "skill_groups.2" in summary_rows["preferred.001"]
    assert "skill_groups.2" not in summary_rows["required.002"]
    assert "skill_groups.0" in summary_rows["required.002"]

    # Cited-req scoping: without requirement_ids, no Phase-A evidence (no silent union).
    assert allowed_evidence_source_ids_for_target(
        "professional_summary",
        evidence_ids=evidence_ids,
        requirement_evidence=requirement_evidence,
    ) == []
    assert "skill_groups.2" in allowed_evidence_source_ids_for_target(
        "professional_summary",
        evidence_ids=evidence_ids,
        requirement_ids=["preferred.001"],
        requirement_evidence=requirement_evidence,
    )
    assert "skill_groups.2" not in allowed_evidence_source_ids_for_target(
        "professional_summary",
        evidence_ids=evidence_ids,
        requirement_ids=["required.002"],
        requirement_evidence=requirement_evidence,
    )
    # Self-evidence only when Phase A linked the target itself.
    assert "professional_summary" not in allowed_evidence_source_ids_for_target(
        "professional_summary",
        evidence_ids=evidence_ids,
        requirement_ids=["preferred.001"],
        requirement_evidence=requirement_evidence,
    )
    assert not evidence_compatible_with_target(
        "projects.0.bullets.1", "projects.0.bullets.0"
    )


def test_multi_requirement_evidence_must_cover_each_cited_requirement() -> None:
    """Union of legal evidence is necessary but not sufficient for multi-req edits."""
    with pytest.raises(
        SourceEvidenceError,
        match=r"does not cover cited requirement_id 'req\.b'",
    ):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "professional_summary",
                        "requirement_ids": ["req.a", "req.b"],
                        "evidence_source_ids": ["skill_groups.0"],  # only req.a
                        "proposed_text": "Summary claiming both A and B.",
                    }
                ]
            },
            editable_ids={"professional_summary"},
            evidence_ids={
                "professional_summary",
                "skill_groups.0",
                "skill_groups.1",
            },
            eligible_requirement_ids={"req.a", "req.b"},
            requirement_evidence={
                "req.a": {"skill_groups.0"},
                "req.b": {"skill_groups.1"},
            },
        )

    covered = validate_edits_payload(
        {
            "edits": [
                {
                    "target_source_id": "professional_summary",
                    "requirement_ids": ["req.a", "req.b"],
                    "evidence_source_ids": ["skill_groups.0", "skill_groups.1"],
                    "proposed_text": "Summary covering A and B with both skill groups.",
                }
            ]
        },
        editable_ids={"professional_summary"},
        evidence_ids={
            "professional_summary",
            "skill_groups.0",
            "skill_groups.1",
        },
        eligible_requirement_ids={"req.a", "req.b"},
        requirement_evidence={
            "req.a": {"skill_groups.0"},
            "req.b": {"skill_groups.1"},
        },
    )
    assert covered["edits"][0]["evidence_source_ids"] == [
        "skill_groups.0",
        "skill_groups.1",
    ]


def test_target_self_evidence_cannot_bypass_phase_a() -> None:
    """Self-evidence is not free: Phase A must link the target to the requirement."""
    with pytest.raises(
        SourceEvidenceError,
        match=r"unrelated evidence source_id 'professional_summary'",
    ):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "professional_summary",
                        "requirement_ids": ["req.a"],
                        "evidence_source_ids": ["professional_summary"],
                        "proposed_text": "Summary rewrite without Phase A self-link.",
                    }
                ]
            },
            editable_ids={"professional_summary"},
            evidence_ids={"professional_summary", "skill_groups.0"},
            eligible_requirement_ids={"req.a"},
            requirement_evidence={"req.a": {"skill_groups.0"}},
        )

    # Self is Phase-A-linked to req.a only: legal in the union but cannot cover req.b.
    with pytest.raises(
        SourceEvidenceError,
        match=r"does not cover cited requirement_id 'req\.b'",
    ):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "professional_summary",
                        "requirement_ids": ["req.a", "req.b"],
                        "evidence_source_ids": [
                            "skill_groups.0",
                            "professional_summary",
                        ],
                        "proposed_text": "A via skills/self; B left unsupported.",
                    }
                ]
            },
            editable_ids={"professional_summary"},
            evidence_ids={
                "professional_summary",
                "skill_groups.0",
                "skill_groups.1",
            },
            eligible_requirement_ids={"req.a", "req.b"},
            requirement_evidence={
                "req.a": {"skill_groups.0", "professional_summary"},
                "req.b": {"skill_groups.1"},
            },
        )

    # Legitimate same-source when Phase A linked the target itself.
    linked = validate_edits_payload(
        {
            "edits": [
                {
                    "target_source_id": "professional_summary",
                    "requirement_ids": ["req.a"],
                    "evidence_source_ids": ["professional_summary"],
                    "proposed_text": "Summary rephrase backed by Phase A self-link.",
                }
            ]
        },
        editable_ids={"professional_summary"},
        evidence_ids={"professional_summary", "skill_groups.0"},
        eligible_requirement_ids={"req.a"},
        requirement_evidence={"req.a": {"professional_summary"}},
    )
    assert linked["edits"][0]["evidence_source_ids"] == ["professional_summary"]


def test_skill_group_cross_section_requires_phase_a_linked_requirement() -> None:
    """Replay …-075631 shape: skill_groups.2 → summary only if Phase A linked."""
    # Linked requirement: succeeds
    ok = validate_edits_payload(
        {
            "edits": [
                {
                    "target_source_id": "professional_summary",
                    "requirement_ids": ["preferred.001"],
                    "evidence_source_ids": ["skill_groups.2"],
                    "proposed_text": "Engineer with infrastructure and Python delivery.",
                }
            ]
        },
        editable_ids={"professional_summary"},
        evidence_ids={"professional_summary", "skill_groups.2", "skill_groups.0"},
        eligible_requirement_ids={"preferred.001", "required.002"},
        requirement_evidence={
            "preferred.001": {"skill_groups.2"},
            "required.002": {"skill_groups.0"},
        },
    )
    assert ok["edits"][0]["evidence_source_ids"] == ["skill_groups.2"]

    # Unrelated requirement (Jobright failure shape): fails closed
    with pytest.raises(
        SourceEvidenceError,
        match=r"unrelated evidence source_id 'skill_groups\.2' for target "
        r"'professional_summary'",
    ):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "professional_summary",
                        "requirement_ids": ["required.002"],
                        "evidence_source_ids": ["skill_groups.2"],
                        "proposed_text": "Engineer with infrastructure delivery.",
                    }
                ]
            },
            editable_ids={"professional_summary"},
            evidence_ids={
                "professional_summary",
                "skill_groups.2",
                "skill_groups.0",
            },
            eligible_requirement_ids={"preferred.001", "required.002"},
            requirement_evidence={
                "preferred.001": {"skill_groups.2"},
                "required.002": {"skill_groups.0"},
            },
        )


def test_phase_b_cannot_invent_requirement_evidence_relationship() -> None:
    """Evidence legal for one requirement cannot ride along with another alone."""
    with pytest.raises(SourceEvidenceError, match="unrelated evidence"):
        validate_edits_payload(
            {
                "edits": [
                    {
                        "target_source_id": "professional_summary",
                        "requirement_ids": ["req.a"],
                        # skill_groups.1 is Phase-A only for req.b
                        "evidence_source_ids": ["skill_groups.1"],
                        "proposed_text": "Invented cross-requirement linkage.",
                    }
                ]
            },
            editable_ids={"professional_summary"},
            evidence_ids={
                "professional_summary",
                "skill_groups.0",
                "skill_groups.1",
            },
            eligible_requirement_ids={"req.a", "req.b"},
            requirement_evidence={
                "req.a": {"skill_groups.0"},
                "req.b": {"skill_groups.1"},
            },
        )


def test_no_supported_requirements_skips_phase_b(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    coverage = _coverage_body(catalog, support_first=False)
    mock_ollama["set_bodies"](_ollama_body(coverage))
    payload = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=catalog,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    assert mock_ollama["calls"] == 1  # Phase B skipped
    assert payload["recommended_edits"] == []
    assert payload["unsupported_requirement_ids"]
    assert payload["supported_requirement_mappings"] == []
    diagnostic = json.loads(
        (tmp_path / EDITS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["phase"] == "edits"
    assert diagnostic.get("skipped") is True
    assert diagnostic.get("skip_reason") == "no_supported_or_editable_targets"


def test_canonical_fields_are_deterministic_and_honest() -> None:
    catalog = _job_catalog()
    ids = _requirement_ids(catalog)
    coverage = {
        "requirements": [
            {
                "requirement_id": ids[0],
                "status": "supported",
                "evidence_source_ids": ["skill_groups.0"],
            },
            *[
                {
                    "requirement_id": rid,
                    "status": "unsupported",
                    "evidence_source_ids": [],
                }
                for rid in ids[1:]
            ],
        ]
    }
    analysis = assemble_canonical_analysis(
        coverage=coverage,
        edits={"edits": []},
        company="Acme",
        role="Engineer",
        job_requirements=catalog,
    )
    assert analysis["role_summary"] == "Engineer opportunity at Acme."
    assert "Acme" in analysis["fit_assessment"]["overall"]
    assert analysis["questions_for_user"] == []
    assert analysis["content_budget_guidance"] == []
    assert analysis["immutable_facts"] == []
    # Unsupported requirements preserved; no invented employer/metric claims.
    assert set(analysis["unsupported_requirement_ids"]) == set(ids[1:])
    catalog_texts = {
        item["exact_text"]
        for item in catalog["requirements"]
        if isinstance(item.get("exact_text"), str)
    }
    for gap in analysis["fit_assessment"]["gaps"]:
        assert gap in catalog_texts or gap.startswith("Some posting")
        assert "led a team of" not in gap.casefold()
        assert "increased revenue" not in gap.casefold()
    for claim in analysis["forbidden_claims"]:
        assert claim in catalog_texts
    # Re-assembly is deterministic.
    again = assemble_canonical_analysis(
        coverage=coverage,
        edits={"edits": []},
        company="Acme",
        role="Engineer",
        job_requirements=catalog,
    )
    assert again == analysis


def test_structured_targets_remain_compiler_owned() -> None:
    """Phase B may name structured targets; they must still be compiler-classified."""
    from resume_tailor.backend.engine.structured_patch_compiler import (
        is_deterministic_structured_target,
        partition_edit_catalog,
    )

    for target in (
        "skill_groups.0",
        "education.coursework",
        "education.certifications",
    ):
        assert is_deterministic_structured_target(target)

    edits = [
        {
            "target_source_id": "skill_groups.0",
            "operation": "replace",
            "proposed_text": "Python, RAG",
            "alignment_rationale": "Supports: Python",
            "evidence_source_ids": ["skill_groups.0"],
        },
        {
            "target_source_id": "professional_summary",
            "operation": "replace",
            "proposed_text": "Python engineer.",
            "alignment_rationale": "Supports: Python",
            "evidence_source_ids": ["skill_groups.0"],
        },
    ]
    structured, prose = partition_edit_catalog(edits)
    assert [e["target_source_id"] for e in structured] == ["skill_groups.0"]
    assert [e["target_source_id"] for e in prose] == ["professional_summary"]


def test_success_diagnostics_include_phase_metadata(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    coverage = _coverage_body(catalog)
    supported_id = coverage["requirements"][0]["requirement_id"]
    edits = {
        "edits": [
            {
                "target_source_id": "professional_summary",
                "requirement_ids": [supported_id],
                "evidence_source_ids": ["skill_groups.0"],
                "proposed_text": "Python engineer.",
            }
        ]
    }
    mock_ollama["set_bodies"](_ollama_body(coverage), _ollama_body(edits))
    invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=catalog,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=60,
    )
    for path, phase, tokens in (
        (COVERAGE_DIAGNOSTIC_FILENAME, "coverage_summary", DEFAULT_COVERAGE_BATCH_MAX_OUTPUT_TOKENS),
        (EDITS_DIAGNOSTIC_FILENAME, "edits", DEFAULT_EDIT_MAX_OUTPUT_TOKENS),
    ):
        diagnostic = json.loads((tmp_path / path).read_text(encoding="utf-8"))
        assert diagnostic["phase"] == phase
        assert diagnostic["classification"] == "success"
        assert diagnostic["attempt"] == 0
        assert diagnostic["model"]
        if phase == "coverage_summary":
            assert diagnostic["effective_output_ceiling"] == tokens
        else:
            assert diagnostic["effective_num_predict"] == tokens
        if phase != "coverage_summary":
            assert diagnostic["max_output_tokens"] == tokens
            assert "prompt_bytes" in diagnostic
            assert "schema_bytes" in diagnostic
        assert "elapsed_seconds" in diagnostic or "total_elapsed_seconds" in diagnostic
        if phase != "coverage_summary":
            assert diagnostic.get("done_reason") == "stop"
        assert diagnostic["hidden_reasoning_excluded"] is True
        assert diagnostic["credentials_excluded"] is True
        # No raw prompt or model content.
        assert "prompt" not in diagnostic
        assert "content" not in diagnostic
        assert "messages" not in diagnostic


def test_ui_timeout_message_not_unavailable() -> None:
    from resume_tailor.ui.ui import _safe_error_message

    message = _safe_error_message(GemmaAnalysisTimeoutError(900))
    assert "generation time limit" in message.casefold()
    assert "unavailable" not in message.casefold()
