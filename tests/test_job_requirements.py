from __future__ import annotations

import copy

import pytest

from resume_tailor.job_requirements import (
    build_job_requirement_catalog,
    job_requirement_index,
    validate_job_requirement_catalog,
)
from resume_tailor.utilities import InputError


def test_structured_catalog_is_deterministic_and_uses_every_supported_field() -> None:
    posting = {
        "responsibilities": ["Build safe services."],
        "required_qualifications": ["Python"],
        "preferred_qualifications": ["Testing experience"],
        "technologies_and_skills": ["JSON Schema"],
        "ai_focus_areas": ["Model evaluation"],
    }

    first = build_job_requirement_catalog("Confirmed synthetic posting.", structured_job=posting)
    second = build_job_requirement_catalog("Confirmed synthetic posting.", structured_job=posting)

    assert first == second
    assert first["source_kind"] == "confirmed_structured_posting"
    assert [item["requirement_id"] for item in first["requirements"]] == [
        "responsibility.001",
        "required.001",
        "preferred.001",
        "skill.001",
        "ai_focus.001",
    ]
    assert [item["category"] for item in first["requirements"]] == [
        "responsibility",
        "required_qualification",
        "preferred_qualification",
        "technology_and_skill",
        "ai_focus_area",
    ]


def test_unstructured_catalog_preserves_local_text_and_heading_categories() -> None:
    description = """Responsibilities:
- Build Python—based validation.
Skills:
- Python
- JSON Schema
"""

    catalog = build_job_requirement_catalog(description)
    index = job_requirement_index(catalog, job_description=description)

    assert index["responsibility.001"]["exact_text"] == (
        "Build Python—based validation."
    )
    assert index["skill.001"]["exact_text"] == "Python"
    assert index["skill.002"]["exact_text"] == "JSON Schema"


def test_catalog_hash_mismatch_and_duplicate_ids_fail_closed() -> None:
    catalog = build_job_requirement_catalog("Synthetic requirement.")
    duplicate = copy.deepcopy(catalog)
    duplicate["requirements"].append(copy.deepcopy(duplicate["requirements"][0]))

    with pytest.raises(InputError, match="duplicate IDs"):
        validate_job_requirement_catalog(duplicate)
    with pytest.raises(InputError, match="does not match"):
        validate_job_requirement_catalog(
            catalog,
            job_description="Changed synthetic requirement.",
        )
