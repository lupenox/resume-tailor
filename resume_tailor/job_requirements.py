from __future__ import annotations

import hashlib
import re
from typing import Any

from .job_text import (
    MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS,
    validate_confirmed_job_description,
)
from .utilities import InputError, RequirementExtractionError, atomic_write_json
from pathlib import Path
import json


CATALOG_VERSION = 1
MAX_JOB_REQUIREMENTS = 999
MAX_REQUIREMENT_CHARACTERS = 5_000

_STRUCTURED_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("responsibilities", "responsibility", "responsibility"),
    ("required_qualifications", "required_qualification", "required"),
    ("preferred_qualifications", "preferred_qualification", "preferred"),
    ("technologies_and_skills", "technology_and_skill", "skill"),
    ("ai_focus_areas", "ai_focus_area", "ai_focus"),
)
_ALLOWED_CATEGORIES = frozenset(
    {category for _, category, _ in _STRUCTURED_FIELDS}
    | {"unstructured_requirement"}
)
_ID_RE = re.compile(
    r"^(?:responsibility|required|preferred|skill|ai_focus|text)\.[0-9]{3}$"
)
_BULLET_PREFIX_RE = re.compile(r"^(?:[-*•‣▪◦]|[0-9]{1,3}[.)])\s+")
_HEADING_RE = re.compile(
    r"^(?:responsibilities|what you(?:'|’)ll do|required qualifications?|"
    r"requirements?|preferred qualifications?|nice to have|skills?|"
    r"technologies|ai focus(?: areas?)?)\s*:?$",
    re.I,
)


def _description_bytes(job_description: str) -> bytes:
    return (job_description.rstrip() + "\n").encode("utf-8")


def job_description_sha256(job_description: str) -> str:
    return hashlib.sha256(_description_bytes(job_description)).hexdigest()


def _safe_requirement_text(
    value: Any,
    *,
    location: str,
    maximum_characters: int = MAX_REQUIREMENT_CHARACTERS,
) -> str:
    if not isinstance(value, str):
        raise InputError(f"Job requirement {location} must be text.")
    normalized = " ".join(
        value.replace("\r\n", "\n").replace("\r", "\n").split()
    )
    if not normalized:
        raise InputError(f"Job requirement {location} is empty.")
    if len(normalized) > maximum_characters:
        raise InputError(
            f"Job requirement {location} is {len(normalized):,} characters; the "
            f"maximum permitted length is {maximum_characters:,} characters."
        )
    if any(
        (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
        and character not in {"\t", "\n"}
        for character in normalized
    ):
        raise InputError(f"Job requirement {location} contains unsafe control text.")
    return normalized


def _structured_requirements(
    structured_job: dict[str, Any] | None,
) -> list[tuple[str, str, str]]:
    if structured_job is None:
        return []
    requirements: list[tuple[str, str, str]] = []
    counters: dict[str, int] = {}
    for field, category, prefix in _STRUCTURED_FIELDS:
        values = structured_job.get(field, [])
        if not isinstance(values, list):
            raise InputError(f"Confirmed structured job field {field!r} is invalid.")
        for position, value in enumerate(values):
            text = _safe_requirement_text(value, location=f"{field}[{position}]")
            counters[prefix] = counters.get(prefix, 0) + 1
            requirement_id = f"{prefix}.{counters[prefix]:03d}"
            requirements.append((requirement_id, category, text))
    return requirements



def _category_for_heading(heading: str | None) -> tuple[str, str]:
    normalized = (heading or "").casefold()
    if "respons" in normalized or "what you'll do" in normalized or "what you’ll do" in normalized or "what you will do" in normalized:
        return "responsibility", "responsibility"
    if "preferred" in normalized or "nice to have" in normalized:
        return "preferred_qualification", "preferred"
    if "skill" in normalized or "technolog" in normalized:
        return "technology_and_skill", "skill"
    if "ai focus" in normalized:
        return "ai_focus_area", "ai_focus"
    if "required" in normalized or "qualification" in normalized or "requirement" in normalized or "what we're looking" in normalized or "what we’re looking" in normalized or "successful candidates" in normalized or "what you'll need" in normalized or "what you’ll need" in normalized:
        return "required_qualification", "required"
    if "education" in normalized or "experience" in normalized:
        return "required_qualification", "required"
    return "unstructured_requirement", "text"


def _unstructured_requirements(
    job_description: str,
    diagnostic: dict[str, Any],
) -> list[tuple[str, str, str]]:
    text = re.sub(r'<(p|li|hr|div|h[1-6])[^>]*>', r'\n\n', job_description, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)

    headings = [
        "Requirements", "Qualifications", "Required Qualifications", "Minimum Qualifications",
        "Preferred Qualifications", "What You'll Need", r"What You[’']ll Need",
        "What We're Looking For", r"What We[’']re Looking For", "Responsibilities", "What You'll Do",
        r"What You[’']ll Do", "Skills", "Education", "Experience", "Overview", "Job Description",
        "Successful Candidates Will Have", "What You Will Do"
    ]
    for h in headings:
        pattern = r"([a-z.?!]|\b)(" + h + r")([A-Z])"
        text = re.sub(pattern, r"\1\n\2\n\3", text, flags=re.IGNORECASE)

    text = re.sub(r"([a-z]{2,}[.?!])([A-Z])", r"\1\n\2", text)

    blocks = [b.strip() for b in text.split('\n') if b.strip()]
    diagnostic["initial_block_count"] = len(blocks)
    diagnostic["detected_headings"] = []

    items = []
    heading = None

    local_heading_re = re.compile(
        r"^(?:responsibilities|what you(?:'|’)ll do|what you will do|what you(?:'|’)ll need|required qualifications?|qualifications?|"
        r"requirements?|preferred qualifications?|nice to have|skills?|"
        r"technologies|ai focus(?: areas?)?|education|experience|"
        r"what we(?:'|’)re looking for|successful candidates will have)\s*:?$",
        re.I,
    )
    local_bullet_re = re.compile(r"^(?:[-*•‣▪◦○]|[0-9]{1,3}[.)])\s+")

    for block in blocks:
        if local_heading_re.fullmatch(block):
            heading = block.rstrip(":")
            if heading not in diagnostic["detected_headings"]:
                diagnostic["detected_headings"].append(heading)
            continue

        block = re.sub(r"(?<!^)(\s+[-*•‣▪◦○]\s+)", r"\n\1", block)
        sub_blocks = [s.strip() for s in block.split('\n') if s.strip()]

        for sub in sub_blocks:
            category, prefix = _category_for_heading(heading)
            candidate = local_bullet_re.sub("", sub).strip()
            if not candidate:
                continue

            if category != "unstructured_requirement" and ";" in candidate and len(candidate.split(";")) > 2:
                parts = [p.strip() for p in candidate.split(";") if p.strip()]
                for p in parts:
                    items.append((p, category, prefix))
                continue

            if len(candidate) > 500:
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', candidate)
                for s in sentences:
                    if s.strip():
                        items.append((s.strip(), category, prefix))
            else:
                items.append((candidate, category, prefix))

    final_requirements = []
    counters = {}
    seen_texts = set()
    total_len = len(job_description)

    diagnostic["largest_item_length"] = 0
    diagnostic["largest-item/source_ratio"] = 0.0
    diagnostic["fallback_use"] = False
    diagnostic["deduplication_count"] = 0

    for item_text, cat, pref in items:
        try:
            norm = _safe_requirement_text(
                item_text,
                location="extraction",
                maximum_characters=MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS
            )
        except InputError:
            continue
        if not norm:
            continue

        if norm.casefold() in seen_texts:
            diagnostic["deduplication_count"] += 1
            continue
        seen_texts.add(norm.casefold())

        if len(norm) > diagnostic["largest_item_length"]:
            diagnostic["largest_item_length"] = len(norm)
            diagnostic["largest-item/source_ratio"] = round(len(norm) / max(total_len, 1), 3)

        if len(norm) > 1500:
            diagnostic["failure_classification"] = "item_exceeds_character_threshold"
            raise RequirementExtractionError("Requirement exceeds maximum character threshold for an atomic item.", diagnostic)

        if len(norm) > (total_len * 0.3) and len(norm) > 500:
            diagnostic["failure_classification"] = "item_disproportionate_percentage"
            raise RequirementExtractionError("Requirement contains disproportionate percentage of source text.", diagnostic)

        counters[pref] = counters.get(pref, 0) + 1
        req_id = f"{pref}.{counters[pref]:03d}"

        final_requirements.append((req_id, cat, norm))

    diagnostic["final_requirement_count"] = len(final_requirements)

    if len(final_requirements) == 1 and diagnostic["largest-item/source_ratio"] > 0.8 and total_len > 1000:
        diagnostic["failure_classification"] = "entire_posting_single_requirement"
        raise RequirementExtractionError("The entire job posting was extracted as a single requirement.", diagnostic)

    if not final_requirements:
        diagnostic["failure_classification"] = "no_valid_requirements"
        raise RequirementExtractionError("No deterministic job requirements could be extracted from the confirmed input.", diagnostic)

    # Mark fallback if all unstructured and we have headings
    if all(cat == "unstructured_requirement" for _, cat, _ in final_requirements):
        diagnostic["fallback_use"] = True

    return final_requirements


def build_job_requirement_catalog(
    job_description: str,
    *,
    structured_job: dict[str, Any] | None = None,
    run_directory: Path | None = None,
) -> dict[str, Any]:
    """Build an immutable, deterministic catalog from the confirmed job input."""
    validate_confirmed_job_description(job_description)
    requirements = _structured_requirements(structured_job)
    source_kind = "confirmed_structured_posting"
    diagnostic = {
        "source_job_text_length": len(job_description),
        "normalized_length": len(job_description.strip()),
    }
    try:
        if not requirements:
            requirements = _unstructured_requirements(job_description, diagnostic)
            source_kind = "confirmed_job_text"
        else:
            diagnostic["fallback_use"] = False
            diagnostic["final_requirement_count"] = len(requirements)

        if not requirements:
            diagnostic["failure_classification"] = "no_valid_requirements"
            raise RequirementExtractionError(
                "No deterministic job requirements could be extracted from the confirmed input.",
                diagnostic
            )
        if len(requirements) > MAX_JOB_REQUIREMENTS:
            diagnostic["failure_classification"] = "too_many_requirements"
            raise RequirementExtractionError(
                f"The confirmed posting contains {len(requirements):,} requirement blocks; "
                f"the local safety limit is {MAX_JOB_REQUIREMENTS:,}.",
                diagnostic
            )

        if run_directory:
            atomic_write_json(run_directory / "requirement-extraction-diagnostic.json", diagnostic)

    except RequirementExtractionError as e:
        if run_directory:
            atomic_write_json(run_directory / "requirement-extraction-diagnostic.json", e.diagnostic)
        raise e

    catalog = {
        "version": CATALOG_VERSION,
        "job_description_sha256": job_description_sha256(job_description),
        "source_kind": source_kind,
        "requirements": [
            {
                "requirement_id": requirement_id,
                "category": category,
                "exact_text": text,
            }
            for requirement_id, category, text in requirements
        ],
    }
    validate_job_requirement_catalog(catalog, job_description=job_description)
    return catalog


def validate_job_requirement_catalog(
    catalog: dict[str, Any],
    *,
    job_description: str | None = None,
) -> list[dict[str, str]]:
    if not isinstance(catalog, dict) or catalog.get("version") != CATALOG_VERSION:
        raise InputError("The job-requirement catalog has an unsupported version.")
    if job_description is not None:
        validate_confirmed_job_description(job_description)
    expected_hash = catalog.get("job_description_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise InputError("The job-requirement catalog is missing its input hash.")
    if job_description is not None and expected_hash != job_description_sha256(job_description):
        raise InputError(
            "The job-requirement catalog does not match the confirmed job description."
        )
    source_kind = catalog.get("source_kind")
    if source_kind not in {
        "confirmed_structured_posting",
        "confirmed_job_text",
    }:
        raise InputError("The job-requirement catalog has an invalid source kind.")
    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise InputError("The job-requirement catalog has no requirements.")
    if len(requirements) > MAX_JOB_REQUIREMENTS:
        raise InputError("The job-requirement catalog exceeds its local item limit.")
    seen: set[str] = set()
    validated: list[dict[str, str]] = []
    for position, item in enumerate(requirements):
        if not isinstance(item, dict) or set(item) != {
            "requirement_id",
            "category",
            "exact_text",
        }:
            raise InputError(
                f"Job-requirement catalog entry {position} has an invalid shape."
            )
        requirement_id = item.get("requirement_id")
        category = item.get("category")
        if not isinstance(requirement_id, str) or not _ID_RE.fullmatch(requirement_id):
            raise InputError(
                f"Job-requirement catalog entry {position} has an invalid stable ID."
            )
        if requirement_id in seen:
            raise InputError("The job-requirement catalog contains duplicate IDs.")
        seen.add(requirement_id)
        if category not in _ALLOWED_CATEGORIES:
            raise InputError(
                f"Job-requirement catalog entry {position} has an invalid category."
            )
        text = _safe_requirement_text(
            item.get("exact_text"),
            location=f"catalog[{position}]",
            maximum_characters=(
                MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS
                if source_kind == "confirmed_job_text"
                else MAX_REQUIREMENT_CHARACTERS
            ),
        )
        if text != item.get("exact_text"):
            raise InputError(
                f"Job-requirement catalog entry {position} is not canonically normalized."
            )
        validated.append(
            {
                "requirement_id": requirement_id,
                "category": str(category),
                "exact_text": text,
            }
        )
    return validated


def job_requirement_index(
    catalog: dict[str, Any],
    *,
    job_description: str | None = None,
) -> dict[str, dict[str, str]]:
    return {
        item["requirement_id"]: item
        for item in validate_job_requirement_catalog(
            catalog,
            job_description=job_description,
        )
    }
