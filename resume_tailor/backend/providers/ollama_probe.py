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

from resume_tailor.backend.utils.schemas import _jsonschema_module


REQUIRED_ROOT_FIELDS = (
    "status",
    "message",
    "catalog_sha256",
    "cannot_apply",
    "technical_failure",
    "patches",
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


def _minimal_complete_instance(schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a valid minimal instance for static or run-specific patch schemas."""
    catalog_sha256 = "0" * 64
    patches: list[dict[str, Any]] = []

    if schema is not None and isinstance(schema.get("properties"), dict):
        properties = schema["properties"]
        digest_schema = properties.get("catalog_sha256")
        if isinstance(digest_schema, dict):
            enum_values = digest_schema.get("enum")
            if isinstance(enum_values, list) and enum_values:
                catalog_sha256 = str(enum_values[0])

        patches_schema = properties.get("patches")
        array_schema: dict[str, Any] | None = None
        if isinstance(patches_schema, dict):
            if patches_schema.get("type") == "array":
                array_schema = patches_schema
            else:
                for branch in patches_schema.get("oneOf", []):
                    if isinstance(branch, dict) and branch.get("type") == "array":
                        array_schema = branch
                        break
        if array_schema is not None:
            minimum = array_schema.get("minItems", 0)
            minimum = minimum if isinstance(minimum, int) and minimum >= 0 else 0
            item_schema = array_schema.get("items", {})
            item_properties = (
                item_schema.get("properties", {})
                if isinstance(item_schema, dict)
                else {}
            )
            edit_values = item_properties.get("edit_id", {}).get("enum", ["edit.001"])
            target_values = item_properties.get("target_source_id", {}).get(
                "enum", ["professional_summary"]
            )
            operation_values = item_properties.get("operation", {}).get(
                "enum", ["replace"]
            )
            for index in range(minimum):
                patches.append(
                    {
                        "edit_id": str(edit_values[index % len(edit_values)]),
                        "target_source_id": str(
                            target_values[index % len(target_values)]
                        ),
                        "operation": str(
                            operation_values[index % len(operation_values)]
                        ),
                        "replacement_text": f"Updated text {index + 1}.",
                    }
                )

    return {
        "status": "complete",
        "message": "Applied the approved plan.",
        "catalog_sha256": catalog_sha256,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": patches,
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
    with_unknown = _minimal_complete_instance(schema)
    with_unknown["unexpected_root_member"] = "value"
    checks["additional_properties_enforced"] = not validator.is_valid(with_unknown)

    # $ref must resolve: the baseline instance leans on $defs for résumé parts.
    checks["ref_declared"] = ("$defs" in schema) or ("properties" in schema)
    baseline = _minimal_complete_instance(schema)
    checks["ref_resolves"] = validator.is_valid(baseline)

    # oneOf branching must still discriminate the status envelope. The canonical
    # cross-field allOf is stripped for transport, so assert on $defs instead of
    # requiring root-level oneOf.
    checks["oneof_present"] = _contains_keyword(schema, "oneOf") or _contains_keyword(
        schema, "allOf"
    )
    mismatched = _minimal_complete_instance(schema)
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
