from __future__ import annotations

from typing import Any

from resume_tailor.backend.utils.schemas import load_schema
from resume_tailor.backend.utils.utilities import DependencyError, InputError


def _configured_job_description_limit() -> int:
    # Keep the canonical schema as the single numeric source for every input mode.
    schema = load_schema("linkedin_job.schema.json")
    try:
        maximum = schema["properties"]["normalized_job_description"]["maxLength"]
    except (KeyError, TypeError) as exc:
        raise DependencyError(
            "The canonical LinkedIn job schema is missing the confirmed "
            "job-description length limit."
        ) from exc
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise DependencyError(
            "The canonical LinkedIn job schema has an invalid confirmed "
            "job-description length limit."
        )
    return maximum


MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS = _configured_job_description_limit()


def validate_confirmed_job_description(
    value: Any,
    *,
    label: str = "The confirmed job description",
) -> str:
    """Validate the shared bounded policy without altering confirmed text."""
    if not isinstance(value, str):
        raise InputError(f"{label} must be text.")
    countable = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not countable:
        raise InputError(f"{label} is empty.")
    actual = len(countable)
    if actual > MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS:
        raise InputError(
            f"{label} is {actual:,} characters; the maximum permitted length is "
            f"{MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS:,} characters. Reduce the "
            "posting to the permitted length and try again."
        )
    return value
