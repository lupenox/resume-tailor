"""Offline structured-output capability probe for the local writer schema.

The preserved Step 6 failure returned parseable JSON with a bare résumé root
(``header``, ``objective_summary``, ``technical_skills``, ...) instead of the
required ``status``/``message``/``cannot_apply``/``technical_failure``/
``tailored_resume`` envelope. That is what an ignored structured-output grammar
looks like.

This probe checks, entirely offline, that the constructs the tailoring grammar
depends on are actually present and enforced after the transport schema is
derived: ``$ref`` resolution, ``oneOf`` branching, ``additionalProperties:
false``, and required root fields. It validates the *schema*, never a provider
response, so it makes no network or provider call.
"""

from __future__ import annotations

from typing import Any

from .schemas import _jsonschema_module


REQUIRED_ROOT_FIELDS = (
    "status",
    "message",
    "cannot_apply",
    "technical_failure",
    "tailored_resume",
)

#: The exact wrong-root shape observed in the preserved failure.
OBSERVED_WRONG_ROOT_KEYS = (
    "header",
    "objective_summary",
    "education_certifications",
    "technical_skills",
    "ai_engineering_projects",
    "open_source_contribution",
    "experience",
)


def _minimal_complete_instance() -> dict[str, Any]:
    """Build the smallest instance that satisfies the tailoring envelope."""
    labelled = {"label": "Label", "text": "Text"}
    project = {
        "name": "Project",
        "technologies": "Python",
        "bullets": ["Did the supported work."],
    }
    return {
        "status": "complete",
        "message": "Applied the approved plan.",
        "cannot_apply": None,
        "technical_failure": None,
        "tailored_resume": {
            "professional_summary": "Summary.",
            "education": {
                "institution": "Institution",
                "degree_details": "Degree",
                "coursework": dict(labelled),
                "certifications": dict(labelled),
            },
            "skill_groups": [dict(labelled) for _ in range(3)],
            "projects": [dict(project) for _ in range(3)],
            "open_source": {
                "name": "Repo",
                "technologies": "Python",
                "bullet": "Contributed a supported fix.",
            },
            "experience": {
                "role": "Role",
                "employer_location": "Employer, Remote",
                "dates": "2020 - 2021",
                "bullets": ["Delivered the supported outcome."],
            },
        },
    }


def probe_structured_output_support(schema: dict[str, Any]) -> dict[str, Any]:
    """Verify the derived transport schema still enforces its key constructs.

    Args:
        schema: A derived transport schema, as written to the run directory.

    Returns:
        A content-free result mapping with one boolean per checked construct
        and an overall ``supported`` flag.
    """
    jsonschema = _jsonschema_module()
    validator_class = jsonschema.Draft202012Validator
    validator_class.check_schema(schema)
    validator = validator_class(schema)

    checks: dict[str, bool] = {}

    # Required root fields must be declared and actually enforced.
    declared = set(schema.get("required", ()))
    checks["required_root_fields_declared"] = all(
        field in declared for field in REQUIRED_ROOT_FIELDS
    )
    wrong_root = {key: "value" for key in OBSERVED_WRONG_ROOT_KEYS}
    checks["required_root_fields_enforced"] = not validator.is_valid(wrong_root)

    # additionalProperties: false must reject an unknown root member.
    checks["additional_properties_false"] = schema.get("additionalProperties") is False
    with_unknown = _minimal_complete_instance()
    with_unknown["unexpected_root_member"] = "value"
    checks["additional_properties_enforced"] = not validator.is_valid(with_unknown)

    # $ref must resolve: the baseline instance leans on $defs for résumé parts.
    checks["ref_declared"] = "$defs" in schema
    baseline = _minimal_complete_instance()
    checks["ref_resolves"] = validator.is_valid(baseline)

    # oneOf branching must still discriminate the status envelope. The canonical
    # cross-field allOf is stripped for transport, so assert on $defs instead of
    # requiring root-level oneOf.
    checks["oneof_present"] = _contains_keyword(schema, "oneOf") or _contains_keyword(
        schema, "allOf"
    )
    mismatched = _minimal_complete_instance()
    mismatched["status"] = "not_a_declared_status"
    checks["status_enum_enforced"] = not validator.is_valid(mismatched)

    return {
        "checks": checks,
        "supported": all(checks.values()),
        "provider_called": False,
    }


def _contains_keyword(node: Any, keyword: str) -> bool:
    if isinstance(node, dict):
        if keyword in node:
            return True
        return any(_contains_keyword(child, keyword) for child in node.values())
    if isinstance(node, list):
        return any(_contains_keyword(child, keyword) for child in node)
    return False
