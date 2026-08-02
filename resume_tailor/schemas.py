from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utilities import (
    CodexSchemaCompatibilityError,
    DependencyError,
    ModelError,
    atomic_write_json,
    sha256_file,
)


SCHEMA_DIRECTORY = Path(__file__).resolve().parent.parent / "schemas"

CODEX_TRANSPORT_SCHEMAS = {
    "codex_analysis.schema.json": "codex_analysis.openai.schema.json",
    "final_qa_provider.schema.json": "final_qa_provider.openai.schema.json",
}
CODEX_ANALYSIS_RUN_SCHEMA_NAME = "codex-analysis-transport.schema.json"

_MAX_CODEX_SCHEMA_BYTES = 250_000
_MAX_CODEX_SCHEMA_NODES = 10_000
_MAX_CODEX_ENUM_VALUES = 2_048
_MAX_CODEX_ENUM_STRING_BYTES = 512

# OpenAI Structured Outputs supports a documented JSON Schema subset. These
# canonical-only assertion keywords are deliberately enforced after receipt.
_CODEX_LOCAL_ONLY_ASSERTIONS = {
    "uniqueItems",
    "minLength",
    "maxLength",
}
_CODEX_OMITTED_ANNOTATIONS = {
    "$schema",
    "title",
}
_CODEX_SUPPORTED_KEYWORDS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "const",
    "description",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "items",
    "maximum",
    "maxItems",
    "minimum",
    "minItems",
    "multipleOf",
    "pattern",
    "properties",
    "required",
    "type",
}
_CODEX_SUPPORTED_TYPES = {
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
}
_CODEX_SUPPORTED_FORMATS = {
    "date",
    "date-time",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "time",
    "uuid",
}


@dataclass(frozen=True)
class CodexAnalysisTransportArtifact:
    path: Path
    sha256: str
    size_bytes: int
    evidence_source_id_count: int
    editable_source_id_count: int
    job_requirement_id_count: int

    def metadata(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "evidence_source_id_count": self.evidence_source_id_count,
            "editable_source_id_count": self.editable_source_id_count,
            "job_requirement_id_count": self.job_requirement_id_count,
            "generated_from_source_and_requirement_catalogs": True,
        }


def schema_path(name: str) -> Path:
    path = (SCHEMA_DIRECTORY / name).resolve()
    if not path.is_file():
        raise DependencyError(f"Bundled JSON schema is missing: {path}")
    return path


def load_schema(name: str) -> dict[str, Any]:
    try:
        payload = json.loads(schema_path(name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyError(f"Bundled JSON schema is invalid: {name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DependencyError(f"Bundled JSON schema must be an object: {name}")
    return payload


def _jsonschema_module() -> Any:
    try:
        import jsonschema
    except ImportError as exc:
        raise DependencyError(
            "Python package 'jsonschema' is required. Install project dependencies "
            "as documented in README.md."
        ) from exc
    return jsonschema


def _check_canonical_schema(schema: dict[str, Any], *, name: str) -> None:
    jsonschema = _jsonschema_module()
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise DependencyError(
            f"Bundled schema {name} is invalid: {exc.message}"
        ) from exc


def validate_payload(payload: Any, schema_name: str, *, label: str) -> None:
    jsonschema = _jsonschema_module()
    schema = load_schema(schema_name)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.SchemaError as exc:
        raise DependencyError(
            f"Bundled schema {schema_name} is invalid: {exc.message}"
        ) from exc
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ModelError(
            f"{label} failed local schema validation at {location}: {exc.message}"
        ) from exc


def validate_resume_content_payload(payload: Any, *, label: str) -> None:
    jsonschema = _jsonschema_module()
    root_schema = load_schema("tailored_resume.schema.json")
    wrapper_schema = {
        "$schema": root_schema.get("$schema"),
        "$defs": root_schema.get("$defs", {}),
        "$ref": "#/$defs/resume",
    }
    try:
        jsonschema.Draft202012Validator.check_schema(wrapper_schema)
        jsonschema.validate(instance=payload, schema=wrapper_schema)
    except jsonschema.SchemaError as exc:
        raise DependencyError(
            f"Bundled schema tailored_resume.schema.json is invalid: {exc.message}"
        ) from exc
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ModelError(
            f"{label} failed local schema validation at {location}: {exc.message}"
        ) from exc


def parse_json_text(text: str, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = item
        return value

    def reject_nonfinite(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        line = exc.lineno if isinstance(exc, json.JSONDecodeError) else 1
        column = exc.colno if isinstance(exc, json.JSONDecodeError) else 1
        raise ModelError(
            f"{label} did not return valid JSON (line {line}, column {column})."
        ) from exc


def _schema_location(path: tuple[str, ...]) -> str:
    return "/" + "/".join(path) if path else "<root>"


def _compatibility_error(
    *,
    label: str,
    path: tuple[str, ...],
    message: str,
) -> CodexSchemaCompatibilityError:
    return CodexSchemaCompatibilityError(
        f"{label} is incompatible with OpenAI Structured Outputs at "
        f"{_schema_location(path)}: {message}"
    )


def _transform_schema_node(
    node: Any,
    *,
    label: str,
    path: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise _compatibility_error(
            label=label,
            path=path,
            message="a schema node must be an object",
        )
    transformed: dict[str, Any] = {}
    for keyword, value in node.items():
        keyword_path = (*path, keyword)
        if keyword in _CODEX_OMITTED_ANNOTATIONS:
            continue
        if keyword in _CODEX_LOCAL_ONLY_ASSERTIONS:
            continue
        if keyword not in _CODEX_SUPPORTED_KEYWORDS:
            raise _compatibility_error(
                label=label,
                path=keyword_path,
                message=f"unsupported keyword {keyword!r}",
            )
        if keyword in {"properties", "$defs"}:
            if not isinstance(value, dict):
                raise _compatibility_error(
                    label=label,
                    path=keyword_path,
                    message=f"{keyword!r} must be an object",
                )
            transformed[keyword] = {
                name: _transform_schema_node(
                    child,
                    label=label,
                    path=(*keyword_path, name),
                )
                for name, child in value.items()
            }
        elif keyword == "items":
            transformed[keyword] = _transform_schema_node(
                value,
                label=label,
                path=keyword_path,
            )
        elif keyword == "anyOf":
            if not isinstance(value, list):
                raise _compatibility_error(
                    label=label,
                    path=keyword_path,
                    message="'anyOf' must be an array",
                )
            transformed[keyword] = [
                _transform_schema_node(
                    child,
                    label=label,
                    path=(*keyword_path, str(index)),
                )
                for index, child in enumerate(value)
            ]
        else:
            transformed[keyword] = copy.deepcopy(value)
    return transformed


def derive_codex_transport_schema(
    canonical_schema: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Derive a provider schema while retaining canonical-only local assertions."""

    _check_canonical_schema(canonical_schema, name=label)
    transport = _transform_schema_node(
        canonical_schema,
        label=label,
        path=(),
    )
    audit_codex_transport_schema(transport, label=label)
    return transport


def _audit_schema_node(
    node: Any,
    *,
    label: str,
    path: tuple[str, ...],
    depth: int,
    counters: dict[str, int],
) -> None:
    if not isinstance(node, dict):
        raise _compatibility_error(
            label=label,
            path=path,
            message="a schema node must be an object",
        )
    if depth > 10:
        raise _compatibility_error(
            label=label,
            path=path,
            message="schema nesting exceeds the supported depth of 10",
        )
    counters["nodes"] += 1
    if counters["nodes"] > _MAX_CODEX_SCHEMA_NODES:
        raise _compatibility_error(
            label=label,
            path=path,
            message=(
                "schema complexity exceeds the local 10,000-node safety limit; "
                "reduce the extracted source catalog before retrying"
            ),
        )
    for keyword in node:
        if keyword not in _CODEX_SUPPORTED_KEYWORDS:
            raise _compatibility_error(
                label=label,
                path=(*path, keyword),
                message=f"unsupported keyword {keyword!r}",
            )

    schema_type = node.get("type")
    if isinstance(schema_type, str):
        declared_types = {schema_type}
    elif isinstance(schema_type, list) and schema_type:
        declared_types = set(schema_type)
    elif "$ref" in node or "anyOf" in node:
        declared_types = set()
    else:
        raise _compatibility_error(
            label=label,
            path=(*path, "type"),
            message="every non-reference schema node must declare a type",
        )
    if not declared_types.issubset(_CODEX_SUPPORTED_TYPES):
        unsupported = sorted(declared_types - _CODEX_SUPPORTED_TYPES)
        raise _compatibility_error(
            label=label,
            path=(*path, "type"),
            message=f"unsupported type values: {unsupported}",
        )

    if "format" in node and node["format"] not in _CODEX_SUPPORTED_FORMATS:
        raise _compatibility_error(
            label=label,
            path=(*path, "format"),
            message=f"unsupported string format {node['format']!r}",
        )

    if "enum" in node:
        values = node["enum"]
        if not isinstance(values, list) or not values:
            raise _compatibility_error(
                label=label,
                path=(*path, "enum"),
                message="enum must contain at least one value",
            )
        tokens = [json.dumps(value, sort_keys=True) for value in values]
        if len(tokens) != len(set(tokens)):
            raise _compatibility_error(
                label=label,
                path=(*path, "enum"),
                message="enum values must be unique",
            )
        counters["enum_values"] += len(values)
        if counters["enum_values"] > _MAX_CODEX_ENUM_VALUES:
            raise _compatibility_error(
                label=label,
                path=(*path, "enum"),
                message=(
                    "schema enums exceed the local 2,048-value safety limit; "
                    "reduce the extracted source catalog before retrying"
                ),
            )
        if any(
            isinstance(value, str)
            and len(value.encode("utf-8")) > _MAX_CODEX_ENUM_STRING_BYTES
            for value in values
        ):
            raise _compatibility_error(
                label=label,
                path=(*path, "enum"),
                message="an enum string exceeds the local 512-byte safety limit",
            )

    if "object" in declared_types:
        properties = node.get("properties")
        required = node.get("required")
        if not isinstance(properties, dict):
            raise _compatibility_error(
                label=label,
                path=(*path, "properties"),
                message="object schemas must define properties",
            )
        if node.get("additionalProperties") is not False:
            raise _compatibility_error(
                label=label,
                path=(*path, "additionalProperties"),
                message="object schemas must set additionalProperties to false",
            )
        if (
            not isinstance(required, list)
            or len(required) != len(set(required))
            or set(required) != set(properties)
        ):
            raise _compatibility_error(
                label=label,
                path=(*path, "required"),
                message="every object property must be required exactly once",
            )
        counters["properties"] += len(properties)
        if counters["properties"] > 5_000:
            raise _compatibility_error(
                label=label,
                path=(*path, "properties"),
                message="schema exceeds the supported total of 5,000 properties",
            )
        for name, child in properties.items():
            _audit_schema_node(
                child,
                label=label,
                path=(*path, "properties", name),
                depth=depth + 1,
                counters=counters,
            )

    if "array" in declared_types:
        if "items" not in node:
            raise _compatibility_error(
                label=label,
                path=(*path, "items"),
                message="array schemas must define an item schema",
            )
        _audit_schema_node(
            node["items"],
            label=label,
            path=(*path, "items"),
            depth=depth + 1,
            counters=counters,
        )

    for index, child in enumerate(node.get("anyOf", [])):
        _audit_schema_node(
            child,
            label=label,
            path=(*path, "anyOf", str(index)),
            depth=depth + 1,
            counters=counters,
        )
    for name, child in node.get("$defs", {}).items():
        _audit_schema_node(
            child,
            label=label,
            path=(*path, "$defs", name),
            depth=depth + 1,
            counters=counters,
        )


def audit_codex_transport_schema(
    schema: dict[str, Any],
    *,
    label: str,
) -> None:
    """Fail locally before Codex receives an unsupported transport schema."""

    if schema.get("type") != "object" or "anyOf" in schema:
        raise _compatibility_error(
            label=label,
            path=(),
            message="the root must be an object and must not use anyOf",
        )
    encoded_size = len(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    if encoded_size > _MAX_CODEX_SCHEMA_BYTES:
        raise _compatibility_error(
            label=label,
            path=(),
            message=(
                "schema exceeds the local 250,000-byte safety limit; reduce the "
                "extracted source catalog before retrying"
            ),
        )
    _audit_schema_node(
        schema,
        label=label,
        path=(),
        depth=1,
        counters={"properties": 0, "nodes": 0, "enum_values": 0},
    )


def codex_transport_schema_path(canonical_name: str) -> Path:
    try:
        transport_name = CODEX_TRANSPORT_SCHEMAS[canonical_name]
    except KeyError as exc:
        raise CodexSchemaCompatibilityError(
            f"No Codex transport schema is registered for {canonical_name}."
        ) from exc
    canonical = load_schema(canonical_name)
    expected = derive_codex_transport_schema(canonical, label=canonical_name)
    actual = load_schema(transport_name)
    audit_codex_transport_schema(actual, label=transport_name)
    if actual != expected:
        raise CodexSchemaCompatibilityError(
            f"Bundled Codex transport schema {transport_name} does not match the "
            f"provider-safe derivation of {canonical_name}."
        )
    return schema_path(transport_name)


def _analysis_source_id_sets(
    extracted_resume: dict[str, Any],
) -> tuple[list[str], list[str]]:
    from .docx_extract import source_blocks_from_paragraphs

    blocks = extracted_resume.get("source_blocks")
    if not isinstance(blocks, list):
        paragraphs = extracted_resume.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise CodexSchemaCompatibilityError(
                "Cannot generate the analysis schema without a local source catalog."
            )
        blocks = source_blocks_from_paragraphs(paragraphs)
    evidence_ids: list[str] = []
    editable_ids: list[str] = []
    seen: set[str] = set()
    for position, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise CodexSchemaCompatibilityError(
                f"Source catalog entry {position} is not an object."
            )
        source_id = block.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise CodexSchemaCompatibilityError(
                f"Source catalog entry {position} has no deterministic source ID."
            )
        if source_id in seen:
            raise CodexSchemaCompatibilityError(
                "The source catalog contains duplicate source IDs."
            )
        seen.add(source_id)
        if block.get("evidence_allowed") is True:
            evidence_ids.append(source_id)
        if block.get("editable") is True:
            editable_ids.append(source_id)
    if not evidence_ids:
        raise CodexSchemaCompatibilityError(
            "The source catalog has no evidence-eligible blocks."
        )
    if not editable_ids:
        raise CodexSchemaCompatibilityError(
            "The source catalog has no editable blocks."
        )
    if not set(editable_ids).issubset(evidence_ids):
        raise CodexSchemaCompatibilityError(
            "Editable source IDs must also be evidence eligible."
        )
    return evidence_ids, editable_ids


def build_codex_analysis_transport_schema(
    extracted_resume: dict[str, Any],
    job_requirements: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    """Create a provider-safe schema constrained to both immutable ID catalogs."""
    from .job_requirements import validate_job_requirement_catalog

    # Retain the checked-in transport as a compatibility sentinel, then derive a
    # fresh run schema from the complete canonical contract.
    codex_transport_schema_path("codex_analysis.schema.json")
    transport = derive_codex_transport_schema(
        load_schema("codex_analysis.schema.json"),
        label="run-specific codex analysis schema",
    )
    evidence_ids, editable_ids = _analysis_source_id_sets(extracted_resume)
    requirement_ids = [
        item["requirement_id"]
        for item in validate_job_requirement_catalog(job_requirements)
    ]
    if len(requirement_ids) > _MAX_CODEX_ENUM_VALUES:
        raise CodexSchemaCompatibilityError(
            "The job-requirement catalog exceeds the local Structured Outputs "
            "enum limit; reduce the confirmed posting before retrying."
        )
    properties = transport["properties"]

    evidence_arrays = (
        properties["supported_requirement_mappings"]["items"]["properties"][
            "evidence_source_ids"
        ],
        properties["recommended_edits"]["items"]["properties"][
            "evidence_source_ids"
        ],
    )
    for array_schema in evidence_arrays:
        array_schema["minItems"] = 1
        array_schema["items"]["enum"] = list(evidence_ids)

    properties["supported_requirement_mappings"]["items"]["properties"][
        "requirement_id"
    ]["enum"] = list(requirement_ids)
    properties["unsupported_requirement_ids"]["items"]["enum"] = list(
        requirement_ids
    )

    for target_schema in (
        properties["recommended_edits"]["items"]["properties"][
            "target_source_id"
        ],
        properties["content_budget_guidance"]["items"]["properties"][
            "target_source_id"
        ],
    ):
        target_schema["enum"] = list(editable_ids)

    _check_canonical_schema(transport, name="run-specific codex analysis schema")
    audit_codex_transport_schema(
        transport,
        label="run-specific codex analysis schema",
    )
    return transport, evidence_ids, editable_ids, requirement_ids


def prepare_codex_analysis_transport_schema(
    extracted_resume: dict[str, Any],
    job_requirements: dict[str, Any],
    run_directory: Path,
) -> CodexAnalysisTransportArtifact:
    if not run_directory.is_dir():
        raise CodexSchemaCompatibilityError(
            "The run directory must exist before analysis schema generation."
        )
    transport, evidence_ids, editable_ids, requirement_ids = (
        build_codex_analysis_transport_schema(
            extracted_resume,
            job_requirements,
        )
    )
    path = run_directory / CODEX_ANALYSIS_RUN_SCHEMA_NAME
    encoded = (
        json.dumps(transport, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    size_bytes = len(encoded)
    if size_bytes > _MAX_CODEX_SCHEMA_BYTES:
        raise CodexSchemaCompatibilityError(
            "The generated analysis schema exceeds the local 250,000-byte safety "
            "limit; reduce the extracted source catalog before retrying."
        )
    atomic_write_json(path, transport)
    return CodexAnalysisTransportArtifact(
        path=path.resolve(),
        sha256=sha256_file(path),
        size_bytes=size_bytes,
        evidence_source_id_count=len(evidence_ids),
        editable_source_id_count=len(editable_ids),
        job_requirement_id_count=len(requirement_ids),
    )


def validate_codex_analysis_transport_artifact(
    artifact: CodexAnalysisTransportArtifact,
    extracted_resume: dict[str, Any],
    job_requirements: dict[str, Any],
    run_directory: Path,
) -> None:
    if (
        artifact.path.is_symlink()
        or not artifact.path.is_file()
        or artifact.path.resolve().parent != run_directory.resolve()
        or artifact.path.name != CODEX_ANALYSIS_RUN_SCHEMA_NAME
    ):
        raise CodexSchemaCompatibilityError(
            "The generated analysis schema is not a safe run artifact."
        )
    if sha256_file(artifact.path) != artifact.sha256:
        raise CodexSchemaCompatibilityError(
            "The generated analysis schema changed before Codex launch."
        )
    try:
        actual = json.loads(artifact.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CodexSchemaCompatibilityError(
            "The generated analysis schema could not be revalidated."
        ) from exc
    expected, evidence_ids, editable_ids, requirement_ids = (
        build_codex_analysis_transport_schema(
            extracted_resume,
            job_requirements,
        )
    )
    if actual != expected:
        raise CodexSchemaCompatibilityError(
            "The generated analysis schema does not match the current source or "
            "job-requirement catalog."
        )
    if (
        artifact.evidence_source_id_count != len(evidence_ids)
        or artifact.editable_source_id_count != len(editable_ids)
        or artifact.job_requirement_id_count != len(requirement_ids)
        or artifact.size_bytes != artifact.path.stat().st_size
    ):
        raise CodexSchemaCompatibilityError(
            "The generated analysis schema metadata is inconsistent."
        )
    _check_canonical_schema(actual, name=artifact.path.name)
    audit_codex_transport_schema(actual, label=artifact.path.name)


def _resolve_local_reference(
    reference: str,
    *,
    root_schema: dict[str, Any],
) -> dict[str, Any] | None:
    if reference == "#":
        return root_schema
    if not reference.startswith("#/"):
        return None
    value: Any = root_schema
    for encoded in reference[2:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, dict) else None


def _duplicate_token(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_unique_node(
    value: Any,
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    path: tuple[str, ...],
    warnings: list[str],
) -> Any:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        resolved = _resolve_local_reference(reference, root_schema=root_schema)
        if resolved is not None:
            return _normalize_unique_node(
                value,
                resolved,
                root_schema=root_schema,
                path=path,
                warnings=warnings,
            )

    if isinstance(value, list):
        normalized_items = value
        if schema.get("uniqueItems") is True:
            seen: set[str] = set()
            deduplicated: list[Any] = []
            removed = 0
            for item in normalized_items:
                token = _duplicate_token(item)
                if token in seen:
                    removed += 1
                    continue
                seen.add(token)
                deduplicated.append(item)
            normalized_items = deduplicated
            if removed:
                location = ".".join(path) or "<root>"
                noun = "value" if removed == 1 else "values"
                warnings.append(
                    f"{location}: removed {removed} exact duplicate {noun} "
                    "before canonical validation."
                )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            normalized_items = [
                _normalize_unique_node(
                    item,
                    item_schema,
                    root_schema=root_schema,
                    path=(*path, str(index)),
                    warnings=warnings,
                )
                for index, item in enumerate(normalized_items)
            ]
        value = normalized_items

    if isinstance(value, dict):
        for name, child_schema in schema.get("properties", {}).items():
            if name in value and isinstance(child_schema, dict):
                value[name] = _normalize_unique_node(
                    value[name],
                    child_schema,
                    root_schema=root_schema,
                    path=(*path, name),
                    warnings=warnings,
                )

    for child_schema in schema.get("anyOf", []):
        if isinstance(child_schema, dict):
            value = _normalize_unique_node(
                value,
                child_schema,
                root_schema=root_schema,
                path=path,
                warnings=warnings,
            )
    return value


def normalize_unique_arrays(
    payload: Any,
    schema_name: str,
) -> tuple[Any, list[str]]:
    """Remove exact duplicates only where the canonical schema requires uniqueness."""

    canonical = load_schema(schema_name)
    normalized = copy.deepcopy(payload)
    warnings: list[str] = []
    normalized = _normalize_unique_node(
        normalized,
        canonical,
        root_schema=canonical,
        path=(),
        warnings=warnings,
    )
    return normalized, warnings
