from __future__ import annotations

import copy
import json

import pytest

from resume_tailor.character_budget import (
    CHARACTER_COUNTING_CONTRACT,
    calculate_content_budget,
    canonicalize_budget_text,
    compose_rendered_text,
    count_budget_characters,
    mutable_character_capacity,
    rendered_prefix,
)
from resume_tailor.codex_analysis import build_analysis_prompt
from resume_tailor.docx_extract import _content_budget, extract_resume
from resume_tailor.evidence import validate_tailored_content
from resume_tailor.patch_engine import (
    PatchCharacterBudgetError,
    TargetDescriptor,
    _validate_replacement_text,
)
from resume_tailor.utilities import OllamaTailoringContractError


def test_canonical_text_normalizes_unicode_and_line_endings_only() -> None:
    original = "Cafe\u0301\r\n  “quoted”\tvalue  \rnext"

    canonical = canonicalize_budget_text(original)

    assert canonical == "Café\n  “quoted”\tvalue  \nnext"
    assert canonical.startswith("Café\n  ")
    assert "value  \n" in canonical


def test_decoded_json_characters_are_counted_without_escape_syntax() -> None:
    decoded = json.loads(r'"Line\n\"quoted\" \\ path"')

    assert decoded == 'Line\n"quoted" \\ path'
    assert count_budget_characters(decoded) == len(decoded)
    assert count_budget_characters("Café") == count_budget_characters("Cafe\u0301")
    assert count_budget_characters("“curly”") == len("“curly”")


def test_patch_validator_applies_the_same_canonical_representation_it_counts() -> None:
    raw = "Cafe\u0301\r\n  “quoted”\tvalue"
    canonical = "Café\n  “quoted”\tvalue"
    descriptor = TargetDescriptor(
        edit_id="edit.001",
        target_source_id="professional_summary",
        operation="replace",
        kind="plain",
        label=None,
        current_mutable_text="Original synthetic summary.",
        exact_rendered_existing_text="Original synthetic summary.",
        maximum_rendered_characters=count_budget_characters(canonical),
        proposed_text=raw,
        alignment_rationale="Synthetic canonical counting test.",
        evidence_source_ids=["professional_summary"],
    )

    validated = _validate_replacement_text(
        edit_id="edit.001",
        descriptor=descriptor,
        replacement_text=raw,
        evidence_texts=["Original synthetic summary."],
        forbidden_claims=[],
    )

    assert validated == canonical
    assert count_budget_characters(validated) == (
        descriptor.maximum_rendered_characters
    )
    with pytest.raises(PatchCharacterBudgetError):
        _validate_replacement_text(
            edit_id="edit.001",
            descriptor=descriptor,
            replacement_text=raw + "!",
            evidence_texts=["Original synthetic summary."],
            forbidden_claims=[],
        )


@pytest.mark.parametrize("replacement", [" Leading text", "Trailing text ", "Text\n"])
def test_patch_validator_rejects_edge_whitespace_instead_of_trimming(
    replacement: str,
) -> None:
    descriptor = TargetDescriptor(
        edit_id="edit.001",
        target_source_id="professional_summary",
        operation="replace",
        kind="plain",
        label=None,
        current_mutable_text="Original synthetic summary.",
        exact_rendered_existing_text="Original synthetic summary.",
        maximum_rendered_characters=100,
        proposed_text=replacement,
        alignment_rationale="Synthetic whitespace test.",
        evidence_source_ids=["professional_summary"],
    )

    with pytest.raises(
        OllamaTailoringContractError,
        match="leading or trailing whitespace.*silently stripped",
    ):
        _validate_replacement_text(
            edit_id="edit.001",
            descriptor=descriptor,
            replacement_text=replacement,
            evidence_texts=["Original synthetic summary."],
            forbidden_claims=[],
        )


def test_structured_grounding_accepts_canonically_equivalent_unicode_evidence() -> None:
    descriptor = TargetDescriptor(
        edit_id="edit.001",
        target_source_id="skill_groups.0",
        operation="replace",
        kind="composite_labelled",
        label="Languages",
        current_mutable_text="Cafe\u0301, SQL",
        exact_rendered_existing_text="Languages: Cafe\u0301, SQL",
        maximum_rendered_characters=100,
        proposed_text="SQL, Café",
        alignment_rationale="Synthetic Unicode reorder.",
        evidence_source_ids=["skill_groups.0"],
    )

    validated = _validate_replacement_text(
        edit_id="edit.001",
        descriptor=descriptor,
        replacement_text="SQL, Café",
        evidence_texts=["Languages: Cafe\u0301, SQL"],
        forbidden_claims=[],
    )

    assert validated == "SQL, Café"


def test_rendered_prefix_and_mutable_capacity_share_the_canonical_counter() -> None:
    label = "Cafe\u0301 Skills"
    body = "Python\r\nFastAPI"
    prefix = rendered_prefix(label)

    assert prefix == "Café Skills: "
    assert compose_rendered_text(body, immutable_label=label) == (
        "Café Skills: Python\nFastAPI"
    )
    assert mutable_character_capacity(40, immutable_label=label) == (
        40 - count_budget_characters(prefix)
    )
    with pytest.raises(ValueError, match="immutable rendered prefix exceeds"):
        mutable_character_capacity(5, immutable_label=label)
    assert mutable_character_capacity(40) == 40


@pytest.mark.parametrize(
    ("text", "expected_original", "expected_maximum"),
    [
        ("x" * 10, 10, 14),
        ("x" * 187, 187, 193),
        ("x" * 1_000, 1_000, 1_020),
        ("Cafe\u0301", 4, 8),
    ],
)
def test_content_budget_retains_central_bounded_allowance(
    text: str,
    expected_original: int,
    expected_maximum: int,
) -> None:
    expected = {
        "original_characters": expected_original,
        "maximum_characters": expected_maximum,
        "original_words": 1,
    }
    assert calculate_content_budget(text) == expected
    assert _content_budget(text) == expected


def test_codex_prompt_exposes_exact_mutable_capacity_for_composite_labels(
    master_resume,
) -> None:
    extracted, _ = extract_resume(master_resume)

    prompt = build_analysis_prompt(
        extracted,
        "Synthetic offline role requiring Python validation.",
        {"requirements": []},
        company="Synthetic Systems",
        role="Validation Engineer",
    )
    trusted_json = prompt.split(
        "BEGIN_TRUSTED_MASTER_RESUME_JSON\n",
        1,
    )[1].split("\nEND_TRUSTED_MASTER_RESUME_JSON", 1)[0]
    trusted = json.loads(trusted_json)
    descriptors = {
        item["source_id"]: item for item in trusted["content_budgets"]
    }
    composite = descriptors["skill_groups.0"]
    label = extracted["content"]["skill_groups"][0]["label"]
    prefix = rendered_prefix(label)

    assert trusted["character_counting_contract"] == CHARACTER_COUNTING_CONTRACT
    assert composite["immutable_rendered_prefix"] == prefix
    assert composite["immutable_prefix_characters"] == count_budget_characters(
        prefix
    )
    assert composite["maximum_mutable_characters"] == (
        composite["maximum_rendered_characters"]
        - composite["immutable_prefix_characters"]
    )
    plain = descriptors["professional_summary"]
    assert plain["immutable_rendered_prefix"] == ""
    assert plain["maximum_mutable_characters"] == plain[
        "maximum_rendered_characters"
    ]
    assert "JSON escape syntax" in prompt
    assert "exact remaining capacity" in prompt


def test_final_evidence_budget_uses_canonical_character_count(master_resume) -> None:
    extracted, _ = extract_resume(master_resume)
    original = extracted["content"]
    tailored = copy.deepcopy(original)
    tailored["professional_summary"] = "Cafe\u0301"
    for paragraph in extracted["paragraphs"]:
        if paragraph["content_id"] == "professional_summary":
            paragraph["content_budget"]["maximum_characters"] = 4
            break
    analysis = {
        "recommended_edits": [{"target_source_id": "professional_summary"}],
        "forbidden_claims": [],
        "supported_ats_keywords": [],
    }

    at_limit = validate_tailored_content(
        original=original,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="",
    )
    assert not any("professional_summary is" in issue for issue in at_limit.issues)

    tailored["professional_summary"] = "Cafe\u0301!"
    over_limit = validate_tailored_content(
        original=original,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="",
    )
    assert any(
        "professional_summary is 5 characters; its template-derived budget is 4"
        in issue
        for issue in over_limit.issues
    )


def test_final_evidence_compares_structured_items_in_canonical_unicode(
    master_resume,
) -> None:
    extracted, _ = extract_resume(master_resume)
    original = copy.deepcopy(extracted["content"])
    original["skill_groups"][0]["text"] = "Cafe\u0301, JavaScript, SQL"
    tailored = copy.deepcopy(original)
    tailored["skill_groups"][0]["text"] = "Café, JavaScript, SQL"
    analysis = {
        "recommended_edits": [{"target_source_id": "skill_groups.0"}],
        "forbidden_claims": [],
        "supported_ats_keywords": [],
    }

    report = validate_tailored_content(
        original=original,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="",
    )

    assert report.introduced_technologies == []
    assert not any("lacks verbatim source evidence" in issue for issue in report.issues)
