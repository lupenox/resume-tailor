"""Canonical character accounting for rendered resume content.

Layout budgets count decoded Unicode text, not its JSON serialization or UTF-8
encoding.  Canonically equivalent Unicode and newline spellings are normalized
before counting, while every other character (including spaces, tabs, and line
breaks) is preserved.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any


CONTENT_BUDGET_ALLOWANCE_RATE = 0.03
MINIMUM_CONTENT_BUDGET_ALLOWANCE = 4
MAXIMUM_CONTENT_BUDGET_ALLOWANCE = 20

CHARACTER_COUNTING_CONTRACT = (
    "Count decoded Unicode code points after NFC normalization and after "
    "normalizing CRLF or CR line endings to LF. Preserve and count all other "
    "spaces, tabs, line breaks, punctuation, and content. JSON escape syntax "
    "does not add characters to the decoded text."
)


def canonicalize_budget_text(value: str) -> str:
    """Return the exact representation used for character-budget decisions."""
    if not isinstance(value, str):
        raise TypeError("Character-budget text must be a string.")
    normalized_newlines = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines)


def count_budget_characters(value: str) -> int:
    """Count decoded Unicode code points in canonical rendered text."""
    return len(canonicalize_budget_text(value))


def rendered_prefix(immutable_label: str | None) -> str:
    """Return the rendered prefix owned by Python for a labelled paragraph."""
    if immutable_label is None:
        return ""
    if not isinstance(immutable_label, str):
        raise TypeError("An immutable rendered label must be text or None.")
    return f"{canonicalize_budget_text(immutable_label)}: "


def composite_label_for_source_id(
    content: dict[str, Any],
    source_id: str,
) -> str | None:
    """Resolve the immutable label for a supported composite list target."""
    if source_id in {"education.coursework", "education.certifications"}:
        education = content.get("education")
        name = source_id.rsplit(".", 1)[-1]
        item = education.get(name) if isinstance(education, dict) else None
        label = item.get("label") if isinstance(item, dict) else None
        return label if isinstance(label, str) else None
    if source_id.startswith("skill_groups."):
        suffix = source_id.removeprefix("skill_groups.")
        groups = content.get("skill_groups")
        if suffix.isdigit() and isinstance(groups, list):
            index = int(suffix)
            item = groups[index] if 0 <= index < len(groups) else None
            label = item.get("label") if isinstance(item, dict) else None
            return label if isinstance(label, str) else None
    return None


def mutable_text_from_composite_proposal(
    proposed_text: str,
    *,
    immutable_label: str,
) -> str:
    """Remove only an authenticated composite label wrapper from a proposal.

    This preserves the established normalization of outer wrapper whitespace
    while requiring the label itself to be canonically identical. Colons in a
    body-only proposal and mismatched labels remain untouched.
    """
    if not isinstance(proposed_text, str):
        raise TypeError("A composite proposal must be text.")
    if not isinstance(immutable_label, str) or not immutable_label.strip():
        raise ValueError("A composite proposal requires an immutable label.")
    stripped = proposed_text.strip()
    colon_index = stripped.find(":")
    if colon_index == -1:
        return proposed_text
    candidate_prefix = stripped[:colon_index]
    if canonicalize_budget_text(candidate_prefix) == canonicalize_budget_text(
        immutable_label
    ):
        return stripped[colon_index + 1 :].lstrip()
    return proposed_text


def compose_rendered_text(
    mutable_text: str,
    *,
    immutable_label: str | None = None,
) -> str:
    """Compose and canonicalize the complete text subject to a hard budget."""
    return rendered_prefix(immutable_label) + canonicalize_budget_text(mutable_text)


def mutable_character_capacity(
    maximum_rendered_characters: int,
    *,
    immutable_label: str | None = None,
) -> int:
    """Return the exact mutable capacity after an immutable prefix is counted."""
    if (
        isinstance(maximum_rendered_characters, bool)
        or not isinstance(maximum_rendered_characters, int)
        or maximum_rendered_characters < 0
    ):
        raise ValueError("The maximum rendered character count must be nonnegative.")
    prefix_characters = count_budget_characters(rendered_prefix(immutable_label))
    if prefix_characters > maximum_rendered_characters:
        raise ValueError(
            "The immutable rendered prefix exceeds the hard character budget."
        )
    return maximum_rendered_characters - prefix_characters


def calculate_content_budget(text: str) -> dict[str, int]:
    """Calculate the existing bounded 3% allowance using canonical counting."""
    canonical_text = canonicalize_budget_text(text)
    length = count_budget_characters(canonical_text)
    allowance = min(
        MAXIMUM_CONTENT_BUDGET_ALLOWANCE,
        max(
            MINIMUM_CONTENT_BUDGET_ALLOWANCE,
            math.ceil(length * CONTENT_BUDGET_ALLOWANCE_RATE),
        ),
    )
    return {
        "original_characters": length,
        "maximum_characters": length + allowance,
        "original_words": len(canonical_text.split()),
    }


def character_budget_descriptor(
    *,
    source_id: str,
    maximum_rendered_characters: Any,
    immutable_label: str | None = None,
) -> dict[str, Any]:
    """Build one prompt-safe descriptor from an authenticated hard limit."""
    if (
        isinstance(maximum_rendered_characters, bool)
        or not isinstance(maximum_rendered_characters, int)
        or maximum_rendered_characters < 0
    ):
        raise ValueError("The authenticated maximum character count is invalid.")
    prefix = rendered_prefix(immutable_label)
    return {
        "source_id": source_id,
        # Preserve the established field while defining that it is rendered.
        "maximum_characters": maximum_rendered_characters,
        "maximum_rendered_characters": maximum_rendered_characters,
        "immutable_rendered_prefix": prefix,
        "immutable_prefix_characters": count_budget_characters(prefix),
        "maximum_mutable_characters": mutable_character_capacity(
            maximum_rendered_characters,
            immutable_label=immutable_label,
        ),
    }
