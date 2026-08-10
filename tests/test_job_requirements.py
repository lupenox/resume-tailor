import json

import pytest
from pathlib import Path
from resume_tailor.backend.utils.utilities import InputError, RequirementExtractionError
from resume_tailor.backend.jobs.job_requirements import (
    build_job_requirement_catalog,
    validate_job_requirement_catalog,
    job_requirement_index,
)
from resume_tailor.backend.jobs.job_text import MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS

def test_job_requirement_catalog_versioning() -> None:
    description = "Synthetic requirement."
    catalog = build_job_requirement_catalog(description)
    assert catalog["version"] == 1
    assert catalog["source_kind"] == "confirmed_job_text"

def test_duplicate_ids_rejected() -> None:
    description = "Synthetic requirement."
    catalog = build_job_requirement_catalog(description)
    duplicate = dict(catalog)
    duplicate["requirements"] = list(catalog["requirements"]) * 2
    with pytest.raises(InputError, match="duplicate IDs"):
        validate_job_requirement_catalog(duplicate)

def test_confirmed_job_description_over_limit_reports_actual_and_permitted() -> None:
    actual = MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS + 1
    with pytest.raises(InputError) as raised:
        build_job_requirement_catalog("x" * actual)
    message = str(raised.value)
    assert f"{actual:,}" in message
    assert f"{MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS:,}" in message

# --- New tests required by prompt ---

def test_standard_bullet_requirements() -> None:
    doc = """Requirements:
- Must have Python
* Must have Java
• Knows AWS"""
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) == 3
    assert reqs[0]["exact_text"] == "Must have Python"
    assert reqs[0]["category"] == "required_qualification"
    assert reqs[1]["exact_text"] == "Must have Java"
    assert reqs[2]["exact_text"] == "Knows AWS"

def test_numbered_qualifications() -> None:
    doc = """Qualifications:
1) Degree in CS
2. Five years experience"""
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) == 2
    assert reqs[0]["exact_text"] == "Degree in CS"
    assert reqs[1]["exact_text"] == "Five years experience"

def test_required_and_preferred_sections() -> None:
    doc = """Required Qualifications:
- A
- B
Preferred Qualifications:
- C"""
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) == 3
    assert reqs[0]["category"] == "required_qualification"
    assert reqs[1]["category"] == "required_qualification"
    assert reqs[2]["category"] == "preferred_qualification"

def test_responsibilities_plus_qualifications() -> None:
    doc = """What You'll Do:
- Build things
Skills:
- React"""
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) == 2
    assert reqs[0]["category"] == "responsibility"
    assert reqs[1]["category"] == "technology_and_skill"

def test_wrapped_multiline_bullets_and_html_whitespace() -> None:
    doc = "Requirements:<ul><li>Line<br>wrapped</li><li>  Extra   spaces  </li></ul>"
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) == 2
    assert reqs[0]["exact_text"] == "Line wrapped"
    assert reqs[1]["exact_text"] == "Extra spaces"

def test_unicode_bullets() -> None:
    doc = """Qualifications:
◦ Thing 1
‣ Thing 2
▪ Thing 3"""
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) == 3
    assert reqs[0]["exact_text"] == "Thing 1"

def test_semicolon_separated_qualifications() -> None:
    doc = """Requirements:
Ability to code in Python; Knows React and Node; Five years experience;"""
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) == 3
    assert reqs[0]["exact_text"] == "Ability to code in Python"
    assert reqs[1]["exact_text"] == "Knows React and Node"
    assert reqs[2]["exact_text"] == "Five years experience"

def test_duplicate_bullets_removed() -> None:
    doc = """Skills:
- Python
- Python
- python """
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) == 1
    assert reqs[0]["exact_text"] == "Python"

def test_short_unstructured_posting_fallback() -> None:
    doc = "We are looking for a great developer to join us."
    cat = build_job_requirement_catalog(doc)
    assert len(cat["requirements"]) == 1
    assert cat["requirements"][0]["category"] == "unstructured_requirement"

def test_long_unstructured_paragraphs_split_safely() -> None:
    # Paragraph > 500 chars with sentences
    sentences = ["This is sentence number one."] * 20
    doc = "Requirements:\
" + " ".join(sentences)
    cat = build_job_requirement_catalog(doc)
    reqs = cat["requirements"]
    assert len(reqs) > 1 # It should have split the sentences


def test_short_two_block_posting_secondary_segments_oversized_sentences(
    tmp_path: Path,
) -> None:
    def sentence(prefix: str, length: int) -> str:
        assert len(prefix) + 1 < length
        return prefix + ("x" * (length - len(prefix) - 1)) + "."

    intro_prefix = "A concise synthetic overview of the engineering role "
    intro = intro_prefix + ("z" * (390 - len(intro_prefix)))
    oversized = " ".join(
        (
            sentence("you will design validated Python services ", 326),
            sentence("you will implement deterministic safety checks ", 326),
            sentence("you will collaborate on evidence-backed delivery ", 326),
            sentence("you will maintain reliable automated tests ", 325),
        )
    )
    posting = intro + "\n" + oversized
    assert len(posting) == 1_697
    assert len(oversized) == 1_306

    catalog = build_job_requirement_catalog(posting, run_directory=tmp_path)

    requirements = catalog["requirements"]
    assert len(requirements) == 5
    assert requirements[0]["exact_text"] == intro
    assert all(len(item["exact_text"]) <= 500 for item in requirements)
    diagnostic = json.loads(
        (tmp_path / "requirement-extraction-diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["initial_block_count"] == 2
    assert diagnostic["detected_headings"] == []
    assert diagnostic["largest_pre_segmentation_item_length"] == 1_306
    assert diagnostic["secondary_segmentation_count"] == 3
    assert diagnostic["secondary_segmentation_strategies"] == [
        "sentence_boundary"
    ]


def test_oversized_block_uses_semicolon_clause_segmentation(
    tmp_path: Path,
) -> None:
    def clause(prefix: str) -> str:
        return prefix + ("x" * (430 - len(prefix)))

    posting = "Synthetic role overview.\n" + "; ".join(
        (
            clause("Experience with deterministic Python services "),
            clause("Proficiency in structured validation systems "),
            clause("Knowledge of reliable automated test design "),
        )
    )

    catalog = build_job_requirement_catalog(posting, run_directory=tmp_path)

    assert len(catalog["requirements"]) == 4
    diagnostic = json.loads(
        (tmp_path / "requirement-extraction-diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["secondary_segmentation_strategies"] == [
        "semicolon_clause"
    ]


def test_oversized_block_uses_requirement_cue_segmentation(
    tmp_path: Path,
) -> None:
    def clause(prefix: str) -> str:
        return prefix + ("x" * (430 - len(prefix)))

    posting = "Synthetic role overview.\n" + " ".join(
        (
            clause("Must have deterministic Python experience "),
            clause("You will maintain evidence-backed services "),
            clause("Ability to create reliable automated tests "),
        )
    )

    catalog = build_job_requirement_catalog(posting, run_directory=tmp_path)

    assert len(catalog["requirements"]) == 4
    diagnostic = json.loads(
        (tmp_path / "requirement-extraction-diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["secondary_segmentation_strategies"] == [
        "requirement_cue"
    ]

def test_pathological_item_source_ratio_rejected() -> None:
    # 2000 chars of no punctuation. Should reject.
    doc = "x" * 2000
    with pytest.raises(RequirementExtractionError, match="exceeds maximum character threshold"):
        build_job_requirement_catalog(doc)

def test_whole_document_fallback_rejected() -> None:
    # 1500 chars of no punctuation but with a short other requirement
    doc = "Requirements:\
- Small req\
" + "x" * 1600
    with pytest.raises(RequirementExtractionError, match="exceeds maximum character threshold"):
        build_job_requirement_catalog(doc)

def test_baker_tilly_shaped_synthetic_posting(tmp_path: Path) -> None:
    doc = """Job DescriptionAI Solutions EngineerAre you passionate about business?
What You Will DoTeam with colleagues. Develop utilities.
Successful Candidates Will HaveBachelor's degree requiredFive years experience.
"""
    cat = build_job_requirement_catalog(doc, run_directory=tmp_path)
    reqs = cat["requirements"]
    # We should have multiple atomic requirements
    assert len(reqs) >= 4
    diag_file = tmp_path / "requirement-extraction-diagnostic.json"
    assert diag_file.is_file()

def test_stable_ordering_and_ids() -> None:
    doc = "Requirements:\
- A\
- B\
- C"
    cat1 = build_job_requirement_catalog(doc)
    cat2 = build_job_requirement_catalog(doc)
    assert cat1["requirements"] == cat2["requirements"]
