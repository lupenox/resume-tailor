#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema

from resume_tailor.backend.engine.evidence import diagnose_legacy_analysis_evidence
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.utils.schemas import load_schema


EXPECTED_FILES = (
    "codex-analysis.json",
    "codex-analysis-transport.schema.json",
    "extracted-master-resume.json",
    "job-source.json",
    "job-description.txt",
    "run-metadata.json",
)


def _safe_child(directory: Path, name: str) -> Path:
    path = directory / name
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve().parent != directory.resolve()
    ):
        raise SystemExit(f"Unsafe or missing diagnostic artifact: {name}")
    return path


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Diagnostic artifact is not an object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a preserved analysis contract offline and print only sanitized "
            "codes, locations, counts, and hashes."
        )
    )
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    if directory.is_symlink() or not directory.is_dir():
        raise SystemExit("The diagnostic run directory is not safe.")
    paths = {name: _safe_child(directory, name) for name in EXPECTED_FILES}
    analysis = _json_object(paths["codex-analysis.json"])
    old_transport = _json_object(paths["codex-analysis-transport.schema.json"])
    extraction = _json_object(paths["extracted-master-resume.json"])
    job_source = _json_object(paths["job-source.json"])
    metadata = _json_object(paths["run-metadata.json"])
    job_description = paths["job-description.txt"].read_text(encoding="utf-8").rstrip(
        "\n"
    )

    old_transport_valid = True
    try:
        jsonschema.validate(analysis, old_transport)
    except jsonschema.ValidationError:
        old_transport_valid = False
    current_contract_valid = True
    try:
        jsonschema.validate(analysis, load_schema("codex_analysis.schema.json"))
    except jsonschema.ValidationError:
        current_contract_valid = False

    structured = (
        job_source
        if job_source.get("normalized_job_description") == job_description
        else None
    )
    requirements = build_job_requirement_catalog(
        job_description,
        structured_job=structured,
    )
    issues = diagnose_legacy_analysis_evidence(analysis, extraction)
    report = {
        "mode": "offline-read-only",
        "provider_invoked": False,
        "old_transport_accepted_response": old_transport_valid,
        "current_requirement_id_contract_accepted_legacy_response": (
            current_contract_valid
        ),
        "legacy_issue_count": len(issues),
        "legacy_issues": [
            {"code": issue.code, "location": issue.location} for issue in issues
        ],
        "generated_requirement_count": len(requirements["requirements"]),
        "run_status": metadata.get("status"),
        "run_stage": metadata.get("stage"),
        "artifact_hashes": {
            name: _sha256(path) for name, path in sorted(paths.items())
        },
        "content_printed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
