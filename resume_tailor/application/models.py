from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resume_tailor.backend.engine.retry import (
        AntigravityReprocessContext,
        AntigravityRetryContext,
        RetryContext,
    )


@dataclass(frozen=True)
class PipelineRequest:
    """Complete, adapter-neutral input for one pipeline execution."""

    resume: Path
    clipboard: bool = False
    job_file: Path | None = None
    job_url: str | None = None
    company: str | None = None
    role: str | None = None
    output_dir: Path = field(
        default_factory=lambda: Path("~/Documents/Resumes/Tailored")
    )
    analytics_db: Path | None = None
    yes: bool = False
    keep_workdir: bool = False
    timeout: tuple[int, str] = (900, "15m")
    writer_provider: str = "antigravity"
    analysis_provider: str = "gemma_local"
    ollama_model: str = "resume-tailor-gemma"
    antigravity_model: str | None = None
    antigravity_strength: str | None = None
    grok_model: str | None = None
    grok_strength: str | None = None
    codex_model: str | None = None
    codex_strength: str | None = None
    initial_qa_provider: str | None = None
    job_source_override: str = "job-file"
    retry_context: RetryContext | None = None
    antigravity_retry_context: AntigravityRetryContext | None = None
    antigravity_reprocess_context: AntigravityReprocessContext | None = None
    github_portfolio: bool = False
    github_username: str | None = None
    github_include_private: bool = False
    github_allow_private_provider: bool = False
    github_analysis_provider: str | None = None
    github_project_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    """Stable result returned after a pipeline run completes."""

    run_directory: Path
