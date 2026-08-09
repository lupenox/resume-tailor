from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resume_tailor import __version__
from resume_tailor.application import pipeline as application_pipeline
from resume_tailor.application.models import PipelineRequest
from resume_tailor.backend.engine.analysis import (
    ANALYSIS_PROVIDERS,
    DEFAULT_ANALYSIS_PROVIDER,
)
from resume_tailor.backend.engine.orchestration import PipelineHooks
from resume_tailor.backend.providers.ollama_writer import DEFAULT_OLLAMA_MODEL
from resume_tailor.backend.utils.analytics import default_analytics_database_path
from resume_tailor.backend.utils.utilities import (
    ExitCode,
    ResumeTailorError,
    parse_duration,
)
from resume_tailor.ui.terminal_hooks import TerminalPipelineHooks


def _duration_argument(value: str) -> tuple[int, str]:
    try:
        return parse_duration(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tailor-resume",
        description=(
            "Create a truthful, human-gated tailored DOCX/PDF from a structured "
            "master resume."
        ),
    )
    parser.add_argument("--resume", required=True, type=Path, help="master .docx path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--clipboard",
        action="store_true",
        help="read the job description from the Linux clipboard",
    )
    source.add_argument("--job-file", type=Path, help="UTF-8 job-description file")
    source.add_argument(
        "--job-url",
        help="public HTTPS LinkedIn /jobs/view/ URL to retrieve and validate",
    )
    parser.add_argument(
        "--company",
        help="target company (required with --clipboard or --job-file)",
    )
    parser.add_argument(
        "--role",
        help="target role (required with --clipboard or --job-file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/Documents/Resumes/Tailored"),
        help="artifact parent directory (default: ~/Documents/Resumes/Tailored)",
    )
    parser.add_argument(
        "--analytics-db",
        type=Path,
        default=default_analytics_database_path(),
        help="private local SQLite analytics database (default: XDG application data)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip human prompts (truthfulness and safety checks remain enforced)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep internal LibreOffice/model QA working files",
    )
    parser.add_argument(
        "--timeout",
        type=_duration_argument,
        default=_duration_argument("15m"),
        metavar="DURATION",
        help="model timeout such as 90s, 15m, or 1h (default: 15m)",
    )
    parser.add_argument(
        "--writer-provider",
        choices=("ollama", "antigravity"),
        default="ollama",
        help=(
            "résumé-writing provider (default: ollama; antigravity remains a "
            "compatibility option)"
        ),
    )
    parser.add_argument(
        "--analysis-provider",
        choices=ANALYSIS_PROVIDERS,
        default=DEFAULT_ANALYSIS_PROVIDER,
        help=(
            "résumé-analysis provider (default: gemma_local; codex and grok_cli "
            "are explicit alternatives and are never selected automatically)"
        ),
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"local Ollama model/profile (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--antigravity-model",
        default=None,
        help="Antigravity model name (e.g. flash, pro) to use instead of default",
    )
    parser.add_argument(
        "--antigravity-strength",
        default=None,
        help="Antigravity model strength to use instead of default",
    )
    parser.add_argument(
        "--grok-model",
        default=None,
        help="Grok model name to use instead of default",
    )
    parser.add_argument(
        "--grok-strength",
        default=None,
        help="Grok model strength to use instead of default",
    )
    parser.add_argument(
        "--codex-model",
        default=None,
        help="Codex model name to use instead of default",
    )
    parser.add_argument(
        "--codex-strength",
        default=None,
        help="Codex model strength to use instead of default",
    )
    parser.add_argument(
        "--initial-qa-provider",
        choices=("gemma_local", "codex", "grok", "antigravity"),
        default=None,
        help=(
            "optional preselection for Initial QA after rendering (never auto-launches; "
            "default preselection may also come from INITIAL_QA_PROVIDER)"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _validate_mode_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.job_url is not None:
        if args.company is not None or args.role is not None:
            parser.error(
                "--company and --role must be omitted with --job-url; both are "
                "derived from the fetched posting"
            )
        return
    if args.company is None or args.role is None:
        parser.error(
            "--company and --role are required with --clipboard or --job-file"
        )


def pipeline_request_from_namespace(args: argparse.Namespace) -> PipelineRequest:
    """Translate the legacy CLI/UI namespace into the typed application request."""

    return PipelineRequest(
        resume=args.resume,
        clipboard=args.clipboard,
        job_file=args.job_file,
        job_url=args.job_url,
        company=args.company,
        role=args.role,
        output_dir=args.output_dir,
        analytics_db=getattr(args, "analytics_db", None),
        yes=args.yes,
        keep_workdir=args.keep_workdir,
        timeout=args.timeout,
        writer_provider=getattr(args, "writer_provider", "antigravity"),
        analysis_provider=getattr(
            args,
            "analysis_provider",
            DEFAULT_ANALYSIS_PROVIDER,
        ),
        ollama_model=getattr(args, "ollama_model", DEFAULT_OLLAMA_MODEL),
        antigravity_model=getattr(args, "antigravity_model", None),
        antigravity_strength=getattr(args, "antigravity_strength", None),
        grok_model=getattr(args, "grok_model", None),
        grok_strength=getattr(args, "grok_strength", None),
        codex_model=getattr(args, "codex_model", None),
        codex_strength=getattr(args, "codex_strength", None),
        initial_qa_provider=getattr(args, "initial_qa_provider", None),
        job_source_override=getattr(args, "job_source_override", "job-file"),
        retry_context=getattr(args, "retry_context", None),
        antigravity_retry_context=getattr(
            args,
            "antigravity_retry_context",
            None,
        ),
        antigravity_reprocess_context=getattr(
            args,
            "antigravity_reprocess_context",
            None,
        ),
    )


def run_pipeline(
    args: argparse.Namespace,
    *,
    hooks: PipelineHooks | None = None,
) -> Path:
    """Compatibility adapter retaining the historical Namespace/Path contract."""

    request = pipeline_request_from_namespace(args)
    active_hooks = hooks if hooks is not None else TerminalPipelineHooks()
    return application_pipeline.run_pipeline(
        request,
        hooks=active_hooks,
    ).run_directory


# Temporary compatibility re-exports for existing callers of former CLI internals.
# The implementations live exclusively in the application/backend layers.
_validate_label = application_pipeline.validate_label
_initial_generation_metadata = application_pipeline._initial_generation_metadata
invoke_codex_analysis = application_pipeline.invoke_codex_analysis
invoke_final_qa = application_pipeline.invoke_final_qa
historical_initial_qa_provider = application_pipeline.historical_initial_qa_provider


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_mode_arguments(parser, args)
    try:
        run_directory = run_pipeline(args)
    except ResumeTailorError as exc:
        print(f"tailor-resume: {exc}", file=sys.stderr)
        return int(exc.exit_code)
    print(f"\nCompleted successfully. Artifacts: {run_directory}")
    return int(ExitCode.OK)


if __name__ == "__main__":
    raise SystemExit(main())
