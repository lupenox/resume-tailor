from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

from .utilities import ApifyLinkedInRetrievalError, InputError


FETCH_STATUSES = frozenset({"success"})
ALLOWED_LINKEDIN_HOSTS = frozenset({"linkedin.com", "www.linkedin.com"})
_JOB_PATH_RE = re.compile(r"^/jobs/view/[^/]+/?$")
_JOB_ID_RE = re.compile(r"(?:^|-)([0-9]{5,20})$")
_MINIMUM_DESCRIPTION_CHARACTERS = 200
_MINIMUM_DESCRIPTION_WORDS = 30


@dataclass(frozen=True)
class ValidatedLinkedInURL:
    original: str
    normalized: str
    hostname: str
    path: str
    job_id: str | None


def _has_unsafe_control(value: str, *, allow_layout: bool = False) -> bool:
    allowed = {"\n", "\t"} if allow_layout else set()
    for character in value:
        if character in allowed:
            continue
        codepoint = ord(character)
        if (
            codepoint < 32
            or 0x7F <= codepoint <= 0x9F
            or unicodedata.category(character) == "Cf"
        ):
            return True
    return False


def validate_linkedin_url(
    value: str,
    *,
    require_job_path: bool = True,
    require_job_id: bool | None = None,
) -> ValidatedLinkedInURL:
    """Validate a public LinkedIn URL and extract its stable numeric job ID locally."""

    if not isinstance(value, str):
        raise InputError("LinkedIn job URL must be text.")
    candidate = value.strip()
    if not candidate or len(candidate) > 2048:
        raise InputError("LinkedIn job URL is empty or exceeds 2,048 characters.")
    if _has_unsafe_control(candidate) or any(
        character.isspace() for character in candidate
    ):
        raise InputError("LinkedIn job URL contains whitespace or control characters.")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise InputError(f"Malformed LinkedIn job URL: {exc}") from exc
    if parsed.scheme.casefold() != "https":
        raise InputError("LinkedIn job URLs must use HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise InputError("LinkedIn job URLs must not contain embedded credentials.")
    hostname = (parsed.hostname or "").casefold()
    if hostname not in ALLOWED_LINKEDIN_HOSTS:
        raise InputError(
            "Unsupported job URL hostname. Only https://www.linkedin.com/jobs/view/… "
            "and https://linkedin.com/jobs/view/… are accepted."
        )
    if port not in (None, 443):
        raise InputError("LinkedIn job URLs must not use a non-HTTPS port.")

    decoded_path = unquote(parsed.path)
    if (
        _has_unsafe_control(decoded_path)
        or "\\" in decoded_path
        or "/../" in f"{decoded_path}/"
    ):
        raise InputError("LinkedIn job URL contains a suspicious encoded path.")
    if require_job_path and not _JOB_PATH_RE.fullmatch(decoded_path):
        raise InputError(
            "URL is not a supported LinkedIn job posting path; expected "
            "https://www.linkedin.com/jobs/view/…"
        )

    job_id: str | None = None
    if _JOB_PATH_RE.fullmatch(decoded_path):
        final_segment = decoded_path.rstrip("/").rsplit("/", 1)[-1]
        match = _JOB_ID_RE.search(final_segment)
        if match:
            job_id = match.group(1)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query_ids = query.get("currentJobId", [])
    if len(query_ids) > 1 and len(set(query_ids)) > 1:
        raise InputError("LinkedIn URL contains conflicting currentJobId values.")
    if query_ids:
        query_id = query_ids[0]
        if not re.fullmatch(r"[0-9]{5,20}", query_id):
            raise InputError("LinkedIn currentJobId query parameter is malformed.")
        if job_id is not None and query_id != job_id:
            raise InputError("LinkedIn URL contains conflicting job IDs.")
        job_id = query_id

    must_have_id = require_job_path if require_job_id is None else require_job_id
    if must_have_id and job_id is None:
        raise InputError(
            "LinkedIn job URL must contain a stable numeric job ID in the posting "
            "path or currentJobId query parameter."
        )

    normalized = urlunsplit(
        (
            "https",
            hostname,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )
    return ValidatedLinkedInURL(
        original=candidate,
        normalized=normalized,
        hostname=hostname,
        path=decoded_path,
        job_id=job_id,
    )


def normalize_job_description(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = blank
    return "\n".join(normalized_lines).strip()


def _raise_malformed() -> None:
    raise ApifyLinkedInRetrievalError("malformed_output")


def _validate_safe_web_text(payload: dict[str, Any]) -> None:
    scalar_fields = (
        "job_title",
        "company",
        "location",
        "employment_type",
        "salary",
        "seniority_level",
        "date_posted",
    )
    for field in scalar_fields:
        value = payload.get(field)
        if value is not None and _has_unsafe_control(value):
            _raise_malformed()
    applicant_count = payload.get("applicant_count")
    if isinstance(applicant_count, str) and _has_unsafe_control(applicant_count):
        _raise_malformed()
    description = payload["normalized_job_description"]
    if _has_unsafe_control(description, allow_layout=True):
        _raise_malformed()
    for field in (
        "responsibilities",
        "required_qualifications",
        "preferred_qualifications",
        "technologies_and_skills",
        "ai_focus_areas",
        "warnings",
    ):
        for item in payload[field]:
            if not item.strip() or _has_unsafe_control(item, allow_layout=False):
                _raise_malformed()


def validate_job_source(
    payload: dict[str, Any],
    *,
    requested: ValidatedLinkedInURL,
) -> dict[str, Any]:
    """Authenticate one schema-valid result against the locally trusted request."""

    if requested.job_id is None:
        raise InputError("The requested LinkedIn URL has no locally verified job ID.")
    _validate_safe_web_text(payload)

    try:
        returned_request = validate_linkedin_url(payload["requested_url"])
    except InputError as exc:
        raise ApifyLinkedInRetrievalError("no_matching_result") from exc
    if returned_request.normalized != requested.normalized:
        raise ApifyLinkedInRetrievalError("no_matching_result")

    status = payload["fetch_status"]
    if status not in FETCH_STATUSES:
        _raise_malformed()
    if status != "success":
        _raise_malformed()

    final_url = payload["final_resolved_url"]
    if not isinstance(final_url, str):
        raise ApifyLinkedInRetrievalError("no_matching_result")
    try:
        validated_final = validate_linkedin_url(final_url)
    except InputError as exc:
        raise ApifyLinkedInRetrievalError("no_matching_result") from exc

    extracted_id = payload["linkedin_job_id"]
    if (
        extracted_id != requested.job_id
        or validated_final.job_id != requested.job_id
        or returned_request.job_id != requested.job_id
    ):
        raise ApifyLinkedInRetrievalError("no_matching_result")

    company = payload["company"]
    title = payload["job_title"]
    if not isinstance(company, str) or not company.strip():
        raise ApifyLinkedInRetrievalError("insufficient_content")
    if not isinstance(title, str) or not title.strip():
        raise ApifyLinkedInRetrievalError("insufficient_content")
    description = normalize_job_description(payload["normalized_job_description"])
    if (
        len(description) < _MINIMUM_DESCRIPTION_CHARACTERS
        or len(description.split()) < _MINIMUM_DESCRIPTION_WORDS
    ):
        raise ApifyLinkedInRetrievalError("insufficient_content")

    normalized_payload = dict(payload)
    normalized_payload["requested_url"] = requested.normalized
    normalized_payload["final_resolved_url"] = validated_final.normalized
    normalized_payload["linkedin_job_id"] = requested.job_id
    normalized_payload["company"] = company.strip()
    normalized_payload["job_title"] = title.strip()
    normalized_payload["normalized_job_description"] = description
    return normalized_payload


def posting_confirmation_text(job_source: dict[str, Any]) -> str:
    preview = " ".join(job_source["normalized_job_description"].split())
    if len(preview) > 500:
        preview = preview[:497].rstrip() + "..."
    warnings = job_source["warnings"]
    lines = [
        "",
        "LinkedIn posting confirmation",
        f"Company: {job_source['company']}",
        f"Job title: {job_source['job_title']}",
        f"Location: {job_source['location'] or 'Not specified'}",
        f"Requested URL: {job_source['requested_url']}",
        f"Final resolved URL: {job_source['final_resolved_url']}",
        "Description preview:",
        f"  {preview}",
        "Retrieval warnings:",
    ]
    lines.extend(f"  - {warning}" for warning in warnings)
    if not warnings:
        lines.append("  - None")
    return "\n".join(lines)
