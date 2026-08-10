from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


PORTFOLIO_DOCUMENT_VERSION = 1


@dataclass(frozen=True)
class RepositoryEvidence:
    """One immutable, locally identified fact source from a repository head."""

    evidence_id: str
    repository_id: str
    category: str
    exact_text: str
    content_sha256: str
    source_url: str
    head_sha: str
    source_path: str | None = None

    def to_document(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "repository_id": self.repository_id,
            "category": self.category,
            "exact_text": self.exact_text,
            "content_sha256": self.content_sha256,
            "source_url": self.source_url,
            "head_sha": self.head_sha,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class RepositoryDossier:
    """Bounded, evidence-backed description of one discovered repository."""

    repository_id: str
    full_name: str
    owner: str
    name: str
    visibility: str
    source_url: str
    description: str | None
    topics: tuple[str, ...]
    languages: Mapping[str, int]
    fork: bool
    archived: bool
    disabled: bool
    empty: bool
    created_at: str | None
    updated_at: str | None
    pushed_at: str | None
    default_branch: str | None
    head_sha: str | None
    readme_excerpt: str | None
    root_manifest: tuple[str, ...]
    detected_manifests: tuple[str, ...]
    detected_frameworks: tuple[str, ...]
    test_indicators: tuple[str, ...]
    ci_indicators: tuple[str, ...]
    deployment_indicators: tuple[str, ...]
    license: str | None
    homepage_url: str | None
    evidence: tuple[RepositoryEvidence, ...]
    warnings: tuple[str, ...] = ()
    eligible: bool = True
    exclusion_reasons: tuple[str, ...] = ()
    private: bool = False
    known_paths: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def to_document(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "full_name": self.full_name,
            "owner": self.owner,
            "name": self.name,
            "visibility": self.visibility,
            "private": self.private,
            "source_url": self.source_url,
            "description": self.description,
            "topics": list(self.topics),
            "languages": dict(self.languages),
            "fork": self.fork,
            "archived": self.archived,
            "disabled": self.disabled,
            "empty": self.empty,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pushed_at": self.pushed_at,
            "default_branch": self.default_branch,
            "head_sha": self.head_sha,
            "readme_excerpt": self.readme_excerpt,
            "root_manifest": list(self.root_manifest),
            "detected_manifests": list(self.detected_manifests),
            "detected_frameworks": list(self.detected_frameworks),
            "test_indicators": list(self.test_indicators),
            "ci_indicators": list(self.ci_indicators),
            "deployment_indicators": list(self.deployment_indicators),
            "license": self.license,
            "homepage_url": self.homepage_url,
            "evidence": [item.to_document() for item in self.evidence],
            "warnings": list(self.warnings),
            "eligible": self.eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
        }


@dataclass(frozen=True)
class RepositoryCatalog:
    version: int
    generated_at: str
    github_username: str | None
    authenticated: bool
    include_private: bool
    repositories: tuple[RepositoryDossier, ...]
    warnings: tuple[str, ...] = ()

    def to_document(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "github_username": self.github_username,
            "authenticated": self.authenticated,
            "include_private": self.include_private,
            "repositories": [item.to_document() for item in self.repositories],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EvidenceRequest:
    repository_id: str
    request_type: str
    path: str
    requirement_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "request_type": self.request_type,
            "path": self.path,
            "requirement_ids": list(self.requirement_ids),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ScoreComponents:
    job_requirement_relevance: int
    technical_depth: int
    completeness_demonstrability: int
    recency_ownership_confidence: int
    distinctiveness: int
    recruiter_clarity: int

    def to_document(self) -> dict[str, int]:
        return {
            "job_requirement_relevance": self.job_requirement_relevance,
            "technical_depth": self.technical_depth,
            "completeness_demonstrability": self.completeness_demonstrability,
            "recency_ownership_confidence": self.recency_ownership_confidence,
            "distinctiveness": self.distinctiveness,
            "recruiter_clarity": self.recruiter_clarity,
        }


@dataclass(frozen=True)
class RankedRepository:
    repository_id: str
    component_scores: ScoreComponents
    total_score: float
    matched_requirement_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    inclusion_rationale: str
    recommended_resume_angle: str
    risks: tuple[str, ...]
    diversity_category: str

    def to_document(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "component_scores": self.component_scores.to_document(),
            "total_score": self.total_score,
            "matched_requirement_ids": list(self.matched_requirement_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "inclusion_rationale": self.inclusion_rationale,
            "recommended_resume_angle": self.recommended_resume_angle,
            "risks": list(self.risks),
            "diversity_category": self.diversity_category,
        }


@dataclass(frozen=True)
class PortfolioRanking:
    version: int
    generated_at: str
    provider: str
    catalog_sha256: str
    job_requirements_sha256: str
    ranked_repositories: tuple[RankedRepository, ...]
    recommended_repository_ids: tuple[str, ...]
    evidence_request_rounds: int

    def to_document(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "provider": self.provider,
            "catalog_sha256": self.catalog_sha256,
            "job_requirements_sha256": self.job_requirements_sha256,
            "ranked_repositories": [
                item.to_document() for item in self.ranked_repositories
            ],
            "recommended_repository_ids": list(
                self.recommended_repository_ids
            ),
            "evidence_request_rounds": self.evidence_request_rounds,
        }


@dataclass(frozen=True)
class SelectedRepository:
    repository_id: str
    full_name: str
    visibility: str
    head_sha: str
    source_url: str
    approved_resume_angle: str
    approved_evidence: tuple[RepositoryEvidence, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "full_name": self.full_name,
            "visibility": self.visibility,
            "head_sha": self.head_sha,
            "source_url": self.source_url,
            "approved_resume_angle": self.approved_resume_angle,
            "approved_evidence": [
                item.to_document() for item in self.approved_evidence
            ],
        }


@dataclass(frozen=True)
class PortfolioSelection:
    version: int
    decision: str
    approved_at: str
    job_requirements_sha256: str
    ranking_sha256: str
    selected_repository_ids: tuple[str, ...]
    repositories: tuple[SelectedRepository, ...]
    private_provider_transmission_approved: bool

    def to_document(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "decision": self.decision,
            "approved_at": self.approved_at,
            "job_requirements_sha256": self.job_requirements_sha256,
            "ranking_sha256": self.ranking_sha256,
            "selected_repository_ids": list(self.selected_repository_ids),
            "repositories": [item.to_document() for item in self.repositories],
            "private_provider_transmission_approved": (
                self.private_provider_transmission_approved
            ),
        }


@dataclass(frozen=True)
class PortfolioSelectionSettings:
    username: str | None
    include_private: bool
    allow_private_provider: bool
    analysis_provider: str
    timeout_seconds: int
    model: str | None = None
    model_strength: str | None = None
    assume_yes: bool = False
    explicit_repository_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioSelectionResult:
    catalog: RepositoryCatalog
    ranking: PortfolioRanking
    selection: PortfolioSelection
    skipped: bool


class PortfolioCataloger(Protocol):
    def build_catalog(
        self,
        *,
        username: str | None,
        include_private: bool,
        provider_is_external: bool,
        allow_private_provider: bool,
        run_directory: Path,
    ) -> RepositoryCatalog | Mapping[str, Any]: ...

    def fetch_evidence(
        self,
        *,
        repository: RepositoryDossier,
        requests: Sequence[EvidenceRequest],
        round_index: int,
        remaining_bytes: int,
    ) -> Sequence[RepositoryEvidence | Mapping[str, Any]]: ...


class PortfolioRanker(Protocol):
    def request_evidence(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        run_directory: Path,
        round_index: int,
    ) -> Mapping[str, Any]: ...

    def rank(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        run_directory: Path,
    ) -> Mapping[str, Any]: ...


__all__ = [
    "EvidenceRequest",
    "PORTFOLIO_DOCUMENT_VERSION",
    "PortfolioCataloger",
    "PortfolioRanker",
    "PortfolioRanking",
    "PortfolioSelection",
    "PortfolioSelectionResult",
    "PortfolioSelectionSettings",
    "RankedRepository",
    "RepositoryCatalog",
    "RepositoryDossier",
    "RepositoryEvidence",
    "ScoreComponents",
    "SelectedRepository",
]
