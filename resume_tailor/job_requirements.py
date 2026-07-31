from __future__ import annotations

import hashlib
import re
from typing import Any

from .utilities import InputError


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


def _safe_requirement_text(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise InputError(f"Job requirement {location} must be text.")
    normalized = " ".join(
        value.replace("\r\n", "\n").replace("\r", "\n").split()
    )
    if not normalized:
        raise InputError(f"Job requirement {location} is empty.")
    if len(normalized) > MAX_REQUIREMENT_CHARACTERS:
        raise InputError(
            f"Job requirement {location} exceeds the {MAX_REQUIREMENT_CHARACTERS:,}-"
            "character safety limit."
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
    if "respons" in normalized or "what you" in normalized:
        return "responsibility", "responsibility"
    if "preferred" in normalized or "nice to have" in normalized:
        return "preferred_qualification", "preferred"
    if "skill" in normalized or "technolog" in normalized:
        return "technology_and_skill", "skill"
    if "ai focus" in normalized:
        return "ai_focus_area", "ai_focus"
    if "required" in normalized or "qualification" in normalized:
        return "required_qualification", "required"
    return "unstructured_requirement", "text"


def _unstructured_requirements(job_description: str) -> list[tuple[str, str, str]]:
    requirements: list[tuple[str, str, str]] = []
    counters: dict[str, int] = {}
    heading: str | None = None
    for raw_line in job_description.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        if _HEADING_RE.fullmatch(line):
            heading = line.rstrip(":")
            continue
        candidate = _BULLET_PREFIX_RE.sub("", line).strip()
        if not candidate:
            continue
        category, prefix = _category_for_heading(heading)
        text = _safe_requirement_text(candidate, location="confirmed job text")
        counters[prefix] = counters.get(prefix, 0) + 1
        requirements.append((f"{prefix}.{counters[prefix]:03d}", category, text))
    return requirements


def build_job_requirement_catalog(
    job_description: str,
    *,
    structured_job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an immutable, deterministic catalog from the confirmed job input."""
    if not isinstance(job_description, str) or not job_description.strip():
        raise InputError("The confirmed job description is empty.")
    requirements = _structured_requirements(structured_job)
    source_kind = "confirmed_structured_posting"
    if not requirements:
        requirements = _unstructured_requirements(job_description)
        source_kind = "confirmed_job_text"
    if not requirements:
        raise InputError(
            "No deterministic job requirements could be extracted from the confirmed input."
        )
    if len(requirements) > MAX_JOB_REQUIREMENTS:
        raise InputError(
            f"The confirmed posting contains {len(requirements):,} requirement blocks; "
            f"the local safety limit is {MAX_JOB_REQUIREMENTS:,}."
        )
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
    expected_hash = catalog.get("job_description_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise InputError("The job-requirement catalog is missing its input hash.")
    if job_description is not None and expected_hash != job_description_sha256(job_description):
        raise InputError(
            "The job-requirement catalog does not match the confirmed job description."
        )
    if catalog.get("source_kind") not in {
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
            item.get("exact_text"), location=f"catalog[{position}]"
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
