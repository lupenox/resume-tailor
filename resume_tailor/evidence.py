from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .utilities import flatten_strings, normalized_text


_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d+)?%?|\d+\s*[–—-]\s*\d+(?:\.\d+)?%?)(?![\w.])"
)
_FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine|myself|we|our|ours|ourselves)\b", re.I)
_SENIORITY_RE = re.compile(
    r"\b(?:senior|staff|principal|lead|manager|director|architect|owner|founder)\b",
    re.I,
)
_AVAILABILITY_RE = re.compile(
    r"\b(?:available immediately|immediate availability|available to start|"
    r"start date|open to relocation|willing to relocate)\b",
    re.I,
)
_HIGH_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("RAG", re.compile(r"\bRAG\b|\bretrieval[- ]augmented generation\b", re.I)),
    ("GraphQL", re.compile(r"\bGraphQL\b", re.I)),
    ("observability", re.compile(r"\bobservability\b", re.I)),
    (
        "distributed production scale",
        re.compile(r"\bdistributed\b.{0,40}\bproduction\b|\bproduction[- ]scale\b", re.I),
    ),
    ("IVR platforms", re.compile(r"\bIVR\b|\binteractive voice response\b", re.I)),
    ("Kubernetes", re.compile(r"\bKubernetes\b|\bk8s\b", re.I)),
    ("vector databases", re.compile(r"\bvector database\b|\bPinecone\b|\bWeaviate\b|\bChromaDB?\b", re.I)),
    ("LangChain/LlamaIndex", re.compile(r"\bLangChain\b|\bLlamaIndex\b", re.I)),
    ("Kafka", re.compile(r"\bApache Kafka\b|\bKafka\b", re.I)),
    ("Redis", re.compile(r"\bRedis\b", re.I)),
    ("gRPC", re.compile(r"\bgRPC\b", re.I)),
    ("OpenTelemetry", re.compile(r"\bOpenTelemetry\b", re.I)),
    ("Prometheus/Grafana", re.compile(r"\bPrometheus\b|\bGrafana\b", re.I)),
)


@dataclass
class EvidenceReport:
    issues: list[str] = field(default_factory=list)
    introduced_technologies: list[str] = field(default_factory=list)
    introduced_metrics: list[str] = field(default_factory=list)
    introduced_role_labels: list[str] = field(default_factory=list)
    introduced_availability: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


def validate_analysis_evidence(
    analysis: dict[str, Any],
    extracted_resume: dict[str, Any],
) -> list[str]:
    """Fail closed when Codex labels absent text as supported source evidence."""
    source_text = _resume_text(extracted_resume["content"])
    normalized_source = normalized_text(source_text)
    issues: list[str] = []
    for keyword in analysis.get("supported_ats_keywords", []):
        if normalized_text(keyword) not in normalized_source:
            issues.append(
                f"Codex marked ATS keyword {keyword!r} as supported, but it does "
                "not occur in the master resume."
            )
    for index, edit in enumerate(analysis.get("recommended_edits", []), start=1):
        evidence = edit["exact_supporting_evidence"]
        existing = edit["existing_claim"]
        if normalized_text(evidence) not in normalized_source:
            issues.append(
                f"Recommended edit {index} cites evidence not found verbatim in "
                f"the master resume: {evidence!r}."
            )
        if normalized_text(existing) not in normalized_source:
            issues.append(
                f"Recommended edit {index} identifies existing text not found "
                f"verbatim in the master resume: {existing!r}."
            )
    valid_budget_ids = {
        paragraph["content_id"]: paragraph["content_budget"]["maximum_characters"]
        for paragraph in extracted_resume["paragraphs"]
    }
    for guidance in analysis.get("content_budget_guidance", []):
        content_id = guidance["content_id"]
        maximum = guidance["maximum_characters"]
        if content_id not in valid_budget_ids:
            issues.append(
                f"Codex supplied content guidance for unknown paragraph {content_id!r}."
            )
        elif maximum > valid_budget_ids[content_id]:
            issues.append(
                f"Codex expanded the content budget for {content_id!r} from "
                f"{valid_budget_ids[content_id]} to {maximum} characters."
            )
    return list(dict.fromkeys(issues))


def _resume_text(content: dict[str, Any]) -> str:
    return "\n".join(flatten_strings(content))


def _technology_items(content: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for index, group in enumerate(content["skill_groups"]):
        for item in re.split(r"\s*[,•]\s*", group["text"]):
            if item.strip():
                values.append((f"skill_groups.{index}", item.strip().rstrip(".")))
    for index, project in enumerate(content["projects"]):
        for item in re.split(r"\s*[,•]\s*", project["technologies"]):
            if item.strip():
                values.append((f"projects.{index}.technologies", item.strip().rstrip(".")))
    return values


def _paragraph_values(content: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {
        "professional_summary": content["professional_summary"],
        "education.degree": (
            f"{content['education']['institution']} | "
            f"{content['education']['degree_details']}"
        ),
        "education.coursework": (
            f"{content['education']['coursework']['label']}: "
            f"{content['education']['coursework']['text']}"
        ),
        "education.certifications": (
            f"{content['education']['certifications']['label']}: "
            f"{content['education']['certifications']['text']}"
        ),
        "open_source.heading": (
            f"{content['open_source']['name']} | "
            f"{content['open_source']['technologies']}"
        ),
        "open_source.bullet": content["open_source"]["bullet"],
        "experience.heading": (
            f"{content['experience']['role']} | "
            f"{content['experience']['employer_location']} "
            f"({content['experience']['dates']})"
        ),
    }
    for index, group in enumerate(content["skill_groups"]):
        values[f"skill_groups.{index}"] = f"{group['label']}: {group['text']}"
    for project_index, project in enumerate(content["projects"]):
        values[f"projects.{project_index}.heading"] = (
            f"{project['name']} | {project['technologies']}"
        )
        for bullet_index, bullet in enumerate(project["bullets"]):
            values[f"projects.{project_index}.bullets.{bullet_index}"] = bullet
    for bullet_index, bullet in enumerate(content["experience"]["bullets"]):
        values[f"experience.bullets.{bullet_index}"] = bullet
    return values


def _assert_exact(
    report: EvidenceReport,
    *,
    field_name: str,
    original: str,
    tailored: str,
) -> None:
    if tailored != original:
        report.issues.append(
            f"Immutable field changed at {field_name}: {original!r} → {tailored!r}."
        )


def validate_tailored_content(
    *,
    original: dict[str, Any],
    tailored: dict[str, Any],
    extracted_resume: dict[str, Any],
    analysis: dict[str, Any],
    target_role: str,
) -> EvidenceReport:
    report = EvidenceReport()

    education_fields = ("institution", "degree_details")
    for name in education_fields:
        _assert_exact(
            report,
            field_name=f"education.{name}",
            original=original["education"][name],
            tailored=tailored["education"][name],
        )
    for name in ("coursework", "certifications"):
        _assert_exact(
            report,
            field_name=f"education.{name}.label",
            original=original["education"][name]["label"],
            tailored=tailored["education"][name]["label"],
        )

    if len(tailored["skill_groups"]) != len(original["skill_groups"]):
        report.issues.append("The number of technical-skill groups changed.")
    else:
        for index, (before, after) in enumerate(
            zip(original["skill_groups"], tailored["skill_groups"], strict=True)
        ):
            _assert_exact(
                report,
                field_name=f"skill_groups.{index}.label",
                original=before["label"],
                tailored=after["label"],
            )

    if len(tailored["projects"]) != len(original["projects"]):
        report.issues.append("The number of projects changed.")
    else:
        for index, (before, after) in enumerate(
            zip(original["projects"], tailored["projects"], strict=True)
        ):
            _assert_exact(
                report,
                field_name=f"projects.{index}.name",
                original=before["name"],
                tailored=after["name"],
            )
            if len(before["bullets"]) != len(after["bullets"]):
                report.issues.append(
                    f"Project {before['name']!r} changed from "
                    f"{len(before['bullets'])} to {len(after['bullets'])} bullets."
                )

    for name in ("name", "technologies"):
        _assert_exact(
            report,
            field_name=f"open_source.{name}",
            original=original["open_source"][name],
            tailored=tailored["open_source"][name],
        )
    for name in ("role", "employer_location", "dates"):
        _assert_exact(
            report,
            field_name=f"experience.{name}",
            original=original["experience"][name],
            tailored=tailored["experience"][name],
        )
    if len(tailored["experience"]["bullets"]) != len(original["experience"]["bullets"]):
        report.issues.append("The number of employment bullets changed.")

    budgets = {
        paragraph["content_id"]: paragraph["content_budget"]["maximum_characters"]
        for paragraph in extracted_resume["paragraphs"]
    }
    for content_id, value in _paragraph_values(tailored).items():
        maximum = budgets.get(content_id)
        if maximum is not None and len(value) > maximum:
            report.issues.append(
                f"{content_id} is {len(value)} characters; its template-derived "
                f"budget is {maximum}."
            )

    source_text = _resume_text(original)
    tailored_text = _resume_text(tailored)
    normalized_source = normalized_text(source_text)

    original_tech_by_location = {
        (location, normalized_text(item))
        for location, item in _technology_items(original)
    }
    for location, item in _technology_items(tailored):
        normalized_item = normalized_text(item)
        if (location, normalized_item) not in original_tech_by_location:
            report.introduced_technologies.append(f"{location}: {item}")
        if normalized_item not in normalized_source:
            report.issues.append(
                f"Technology/skill item lacks verbatim source evidence at "
                f"{location}: {item!r}."
            )

    original_metrics = set(_NUMBER_RE.findall(source_text))
    for metric in _NUMBER_RE.findall(tailored_text):
        if metric not in original_metrics and metric not in report.introduced_metrics:
            report.introduced_metrics.append(metric)
            report.issues.append(
                f"New numeric or metric claim {metric!r} is not present in the master resume."
            )

    for label, pattern in _HIGH_RISK_PATTERNS:
        if pattern.search(tailored_text) and not pattern.search(source_text):
            report.issues.append(
                f"Forbidden unsupported capability introduced: {label}."
            )

    for forbidden in analysis.get("forbidden_claims", []):
        normalized_forbidden = normalized_text(forbidden)
        if (
            len(normalized_forbidden) >= 8
            and normalized_forbidden in normalized_text(tailored_text)
            and normalized_forbidden not in normalized_source
        ):
            report.issues.append(
                f"Codex-marked forbidden claim appears in tailored content: {forbidden!r}."
            )

    if _FIRST_PERSON_RE.search(tailored_text):
        report.issues.append("First-person language was introduced.")

    for match in _SENIORITY_RE.finditer(tailored_text):
        term = match.group(0)
        if not re.search(rf"\b{re.escape(term)}\b", source_text, re.I):
            if term.casefold() not in {item.casefold() for item in report.introduced_role_labels}:
                report.introduced_role_labels.append(term)
            report.issues.append(
                f"Unsupported seniority/leadership role label introduced: {term!r}."
            )

    if target_role and normalized_text(target_role) not in normalized_source:
        if normalized_text(target_role) in normalized_text(tailored_text):
            report.introduced_role_labels.append(target_role)
            report.issues.append(
                f"Target role label {target_role!r} was introduced as a resume claim "
                "without appearing in the source."
            )

    for match in _AVAILABILITY_RE.finditer(tailored_text):
        phrase = match.group(0)
        if not _AVAILABILITY_RE.search(source_text):
            report.introduced_availability.append(phrase)
            report.issues.append(
                f"New availability statement lacks source evidence: {phrase!r}."
            )

    for keyword in analysis.get("supported_ats_keywords", []):
        if not keyword.strip():
            continue
        count = normalized_text(tailored_text).count(normalized_text(keyword))
        original_count = normalized_source.count(normalized_text(keyword))
        if count > max(3, original_count + 2):
            report.issues.append(
                f"Possible keyword stuffing: {keyword!r} appears {count} times "
                f"(source: {original_count})."
            )

    report.issues = list(dict.fromkeys(report.issues))
    report.introduced_technologies = list(dict.fromkeys(report.introduced_technologies))
    report.introduced_metrics = list(dict.fromkeys(report.introduced_metrics))
    report.introduced_role_labels = list(dict.fromkeys(report.introduced_role_labels))
    report.introduced_availability = list(dict.fromkeys(report.introduced_availability))
    return report


def _section_lines(content: dict[str, Any]) -> list[tuple[str, list[str]]]:
    education = content["education"]
    sections: list[tuple[str, list[str]]] = [
        ("Professional Summary", [content["professional_summary"]]),
        (
            "Education & Certifications",
            [
                f"{education['institution']} | {education['degree_details']}",
                f"{education['coursework']['label']}: {education['coursework']['text']}",
                f"{education['certifications']['label']}: "
                f"{education['certifications']['text']}",
            ],
        ),
        (
            "Technical Skills",
            [f"{group['label']}: {group['text']}" for group in content["skill_groups"]],
        ),
    ]
    project_lines: list[str] = []
    for project in content["projects"]:
        project_lines.append(f"{project['name']} | {project['technologies']}")
        project_lines.extend(f"• {bullet}" for bullet in project["bullets"])
    sections.append(("AI Engineering Projects", project_lines))
    sections.append(
        (
            "Open Source Contribution",
            [
                f"{content['open_source']['name']} | "
                f"{content['open_source']['technologies']}",
                f"• {content['open_source']['bullet']}",
            ],
        )
    )
    sections.append(
        (
            "Experience",
            [
                f"{content['experience']['role']} | "
                f"{content['experience']['employer_location']} "
                f"({content['experience']['dates']})",
                *(f"• {bullet}" for bullet in content["experience"]["bullets"]),
            ],
        )
    )
    return sections


def _markdown_list(values: Iterable[str]) -> str:
    items = list(values)
    return "\n".join(f"- {item}" for item in items) if items else "- None detected"


def build_content_diff(
    original: dict[str, Any],
    tailored: dict[str, Any],
    report: EvidenceReport,
) -> str:
    before_sections = dict(_section_lines(original))
    after_sections = dict(_section_lines(tailored))
    lines = [
        "# Tailored Resume Content Diff",
        "",
        "## Local evidence-check result",
        "",
        "PASS — no blocking unsupported claims detected."
        if report.passed
        else "BLOCKED — questionable claims require correction; rendering is disabled.",
        "",
        "### Newly introduced technologies or skill placements",
        "",
        _markdown_list(report.introduced_technologies),
        "",
        "### Newly introduced metrics",
        "",
        _markdown_list(report.introduced_metrics),
        "",
        "### Newly introduced role labels",
        "",
        _markdown_list(report.introduced_role_labels),
        "",
        "### Newly introduced availability statements",
        "",
        _markdown_list(report.introduced_availability),
        "",
        "### Blocking evidence issues",
        "",
        _markdown_list(report.issues),
        "",
        "## Section-by-section changes",
        "",
    ]
    for section_name, before in before_sections.items():
        after = after_sections[section_name]
        lines.extend([f"### {section_name}", ""])
        if before == after:
            lines.extend(["No change.", ""])
            continue
        diff = difflib.unified_diff(
            before,
            after,
            fromfile="before",
            tofile="after",
            lineterm="",
        )
        lines.extend(f"    {line}" for line in diff)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
