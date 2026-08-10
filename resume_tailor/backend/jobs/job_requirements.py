from __future__ import annotations

import hashlib
import re
from typing import Any

from resume_tailor.backend.jobs.job_text import (
    MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS,
    validate_confirmed_job_description,
)
from resume_tailor.backend.utils.utilities import InputError, RequirementExtractionError, atomic_write_json
from pathlib import Path
import json


CATALOG_VERSION = 1
MAX_JOB_REQUIREMENTS = 999
MAX_REQUIREMENT_CHARACTERS = 5_000
MAX_SECONDARY_SEGMENTS_PER_BLOCK = MAX_JOB_REQUIREMENTS

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
_BULLET_PREFIX_RE = re.compile(r"^(?:[-*•‣▪◦○]|[0-9]{1,3}[.)])\s+")
_LIST_MARKER_CHARS = "•‣▪◦○"
# Apify/LinkedIn HTML collapse often glues unicode list markers to the previous
# token with no leading whitespace: "users• High ownership".
_GLUED_LIST_MARKER_RE = re.compile(rf"(?<=\S)([{_LIST_MARKER_CHARS}])\s*")
_HEADING_LINE_RE = re.compile(
    r"^(?:"
    r"responsibilities|what you(?:'|’)ll do|what you will do|"
    r"what you(?:'|’)ll need|required qualifications?|minimum qualifications?|"
    r"qualifications?|qualification(?:\s*required)?|requirements?|"
    r"preferred qualifications?|preferred|required|nice to have|skills?|"
    r"technologies|ai focus(?: areas?)?|education|experience|"
    r"what we(?:'|’)re looking for|successful candidates will have|"
    r"why join us|overview|job description"
    r")\s*:?$",
    re.I,
)
# Keep the historical name used by callers/tests that import the private symbol.
_HEADING_RE = _HEADING_LINE_RE
# Longer phrases first so "Required Qualifications" wins over bare "Required".
_FUSED_SECTION_HEADINGS: tuple[str, ...] = tuple(
    sorted(
        (
            "Required Qualifications",
            "Minimum Qualifications",
            "Preferred Qualifications",
            "What You'll Need",
            r"What You[’']ll Need",
            "What We're Looking For",
            r"What We[’']re Looking For",
            "What You'll Do",
            r"What You[’']ll Do",
            "What You Will Do",
            "Successful Candidates Will Have",
            "Qualification Required",
            "QualificationRequired",
            "Job Description",
            "Responsibilities",
            "Qualifications",
            "Qualification",
            "Requirements",
            "Why Join Us",
            "Technologies",
            "Education",
            "Experience",
            "Overview",
            "Preferred",
            "Required",
            "Skills",
        ),
        key=lambda value: len(re.sub(r"\\.", "", value)),
        reverse=True,
    )
)
_SECONDARY_CUE_RE = re.compile(
    r"(?i)(?<!\S)(?:"
    r"must\s+(?:have|be|demonstrate)|"
    r"you(?:\s+(?:will|must|should)|(?:'|’)ll)|"
    r"the\s+(?:successful\s+)?candidate\s+(?:will|must|should)|"
    r"successful\s+candidates\s+(?:will|must|should)|"
    r"responsibilities\s+include|qualifications\s+include|"
    r"minimum\s+qualifications|required\s+qualifications|"
    r"preferred\s+qualifications|experience\s+with|"
    r"proficiency\s+in|knowledge\s+of|ability\s+to"
    r")\b"
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


def _requires_secondary_segmentation(value: str, source_length: int) -> bool:
    normalized = " ".join(value.split())
    return len(normalized) > 1500 or (
        len(normalized) > 500
        and len(normalized) > (source_length * 0.3)
    )


def _bounded_parts(parts: list[str], original: str) -> list[str]:
    cleaned = [part.strip() for part in parts if part.strip()]
    if len(cleaned) <= 1:
        return [original]
    return cleaned


def _defuse_glued_list_markers(value: str) -> str:
    """Insert boundaries before unicode list markers glued to prior text.

    Collapsed HTML list markup often yields ``users• High ownership`` instead of
    a newline-delimited bullet list. ASCII ``-``/``*`` markers still require
    surrounding whitespace so hyphenated prose is not split.
    """

    return _GLUED_LIST_MARKER_RE.sub(r"\n\1 ", value)


def _defuse_fused_section_headings(value: str) -> str:
    """Separate section headings fused into adjacent prose or bullet runs.

    Handles classic camel-case fusion (``searchResponsibilitiesImplement``),
    heading-then-bullet fusion (``searchResponsibilities• Implement``), and
    heading stuck on the previous bullet after list markers are split
    (``...job searchResponsibilities`` on its own line).

    Uppercase look-aheads stay case-sensitive so shorter headings such as
    ``Qualification`` do not split the plural ``Qualifications``.
    """

    text = value
    # (?-i:[A-Z]) keeps the continuation check case-sensitive even when the
    # heading alternative itself is matched case-insensitively.
    upper_or_bullet = rf"(?=(?-i:[A-Z])|[{_LIST_MARKER_CHARS}])"
    for heading in _FUSED_SECTION_HEADINGS:
        # lowercase/punct + Heading + (bullet or Uppercase continuation)
        text = re.sub(
            rf"(?<=[a-z.?!])({heading}){upper_or_bullet}",
            r"\n\1\n",
            text,
            flags=re.IGNORECASE,
        )
        # Heading glued at the end of a line after list-marker defusion.
        text = re.sub(
            rf"(?m)(?<=[a-z.?!])({heading})\s*$",
            r"\n\1",
            text,
            flags=re.IGNORECASE,
        )
        # Heading glued directly to a following list marker on the same line.
        text = re.sub(
            rf"(?m)^({heading})(?=[{_LIST_MARKER_CHARS}])",
            r"\1\n",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _split_existing_boundaries(value: str) -> list[str]:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _defuse_glued_list_markers(normalized)
    normalized = re.sub(
        r"(?<!^)[ \t]+(?=(?:[-*•‣▪◦○]|[0-9]{1,3}[.)])\s+)",
        "\n",
        normalized,
    )
    return _bounded_parts(
        [
            _BULLET_PREFIX_RE.sub("", part.strip()).strip()
            for part in normalized.split("\n")
        ],
        value,
    )


def _split_sentence_boundaries(value: str) -> list[str]:
    # The first-pass parser historically required an uppercase character after
    # punctuation. Real postings often continue with lower-case list prose, so
    # the bounded fallback accepts any whitespace-delimited sentence boundary.
    return _bounded_parts(re.split(r"(?<=[.!?])\s+", value), value)


def _split_semicolon_clauses(value: str) -> list[str]:
    return _bounded_parts(re.split(r"\s*;\s*", value), value)


def _split_requirement_cues(value: str) -> list[str]:
    starts = [match.start() for match in _SECONDARY_CUE_RE.finditer(value)]
    starts = [start for start in starts if start > 0]
    if not starts:
        return [value]
    parts: list[str] = []
    cursor = 0
    for start in starts:
        parts.append(value[cursor:start])
        cursor = start
    parts.append(value[cursor:])
    return _bounded_parts(parts, value)


def _secondary_segment_oversized_item(
    value: str,
    *,
    source_length: int,
) -> tuple[list[str], tuple[str, ...]]:
    """Atomize one oversized candidate through bounded deterministic stages."""

    segments = [value]
    used: list[str] = []
    strategies = (
        ("existing_boundary", _split_existing_boundaries),
        ("sentence_boundary", _split_sentence_boundaries),
        ("semicolon_clause", _split_semicolon_clauses),
        ("requirement_cue", _split_requirement_cues),
    )
    for name, splitter in strategies:
        expanded: list[str] = []
        split_used = False
        for segment in segments:
            if not _requires_secondary_segmentation(segment, source_length):
                expanded.append(segment)
                continue
            parts = splitter(segment)
            if len(parts) > 1:
                split_used = True
            expanded.extend(parts)
            if len(expanded) > MAX_SECONDARY_SEGMENTS_PER_BLOCK:
                return [value], ()
        segments = expanded
        if split_used:
            used.append(name)
    return segments, tuple(used)


def _unstructured_requirements(
    job_description: str,
    diagnostic: dict[str, Any],
) -> list[tuple[str, str, str]]:
    text = re.sub(r'<(p|li|hr|div|h[1-6])[^>]*>', r'\n\n', job_description, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Collapse HTML/list formatting artifacts before heading detection so a
    # single giant glued run does not swallow every section.
    text = _defuse_glued_list_markers(text)
    text = _defuse_fused_section_headings(text)
    # Classic camel-case fusion still used by some postings after HTML strip.
    # Keep the uppercase continuation case-sensitive (see defuse helper).
    for heading in _FUSED_SECTION_HEADINGS:
        text = re.sub(
            rf"([a-z.?!]|\b)({heading})(?=(?-i:[A-Z]))",
            r"\1\n\2\n",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"([a-z]{2,}[.?!])([A-Z])", r"\1\n\2", text)

    blocks = [b.strip() for b in text.split('\n') if b.strip()]
    diagnostic["initial_block_count"] = len(blocks)
    diagnostic["detected_headings"] = []

    items = []
    heading = None

    local_bullet_re = re.compile(r"^(?:[-*•‣▪◦○]|[0-9]{1,3}[.)])\s+")

    for block in blocks:
        if _HEADING_LINE_RE.fullmatch(block):
            heading = block.rstrip(":")
            if heading not in diagnostic["detected_headings"]:
                diagnostic["detected_headings"].append(heading)
            continue

        block = _defuse_glued_list_markers(block)
        block = re.sub(r"(?<!^)(\s+[-*•‣▪◦○]\s+)", r"\n\1", block)
        sub_blocks = [s.strip() for s in block.split('\n') if s.strip()]

        for sub in sub_blocks:
            if _HEADING_LINE_RE.fullmatch(sub):
                heading = sub.rstrip(":")
                if heading not in diagnostic["detected_headings"]:
                    diagnostic["detected_headings"].append(heading)
                continue
            category, prefix = _category_for_heading(heading)
            candidate = local_bullet_re.sub("", sub).strip()
            if not candidate:
                continue

            if (
                not _requires_secondary_segmentation(
                    candidate,
                    len(job_description),
                )
                and category != "unstructured_requirement"
                and ";" in candidate
                and len(candidate.split(";")) > 2
            ):
                parts = [p.strip() for p in candidate.split(";") if p.strip()]
                for p in parts:
                    items.append((p, category, prefix))
                continue

            items.append((candidate, category, prefix))

    final_requirements = []
    counters = {}
    seen_texts = set()
    total_len = len(job_description)

    diagnostic["largest_item_length"] = 0
    diagnostic["largest-item/source_ratio"] = 0.0
    diagnostic["fallback_use"] = False
    diagnostic["deduplication_count"] = 0
    diagnostic["secondary_segmentation_count"] = 0
    diagnostic["secondary_segmentation_strategies"] = []
    diagnostic["largest_pre_segmentation_item_length"] = max(
        (len(" ".join(item_text.split())) for item_text, _, _ in items),
        default=0,
    )

    atomic_items: list[tuple[str, str, str]] = []
    for item_text, cat, pref in items:
        segments, strategies = _secondary_segment_oversized_item(
            item_text,
            source_length=total_len,
        )
        if len(segments) > 1:
            diagnostic["secondary_segmentation_count"] += len(segments) - 1
        for strategy in strategies:
            if strategy not in diagnostic["secondary_segmentation_strategies"]:
                diagnostic["secondary_segmentation_strategies"].append(strategy)
        atomic_items.extend((segment, cat, pref) for segment in segments)

    for item_text, cat, pref in atomic_items:
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
