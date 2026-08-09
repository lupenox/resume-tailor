"""Typed application boundary shared by the command-line and web adapters."""

from resume_tailor.application.models import PipelineRequest, PipelineResult
from resume_tailor.application.pipeline import run_pipeline

__all__ = ["PipelineRequest", "PipelineResult", "run_pipeline"]
