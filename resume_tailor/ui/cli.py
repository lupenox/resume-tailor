from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from resume_tailor import __version__
from resume_tailor.application import pipeline as application_pipeline
from resume_tailor.application.models import PipelineRequest
from resume_tailor.backend.engine.analysis import (
    ANALYSIS_PROVIDERS,
    DEFAULT_ANALYSIS_PROVIDER,
    normalize_analysis_provider,
)
from resume_tailor.backend.engine.orchestration import PipelineHooks
from resume_tailor.backend.providers.ollama_writer import DEFAULT_OLLAMA_MODEL
from resume_tailor.backend.providers.portfolio_ranker import (
    PORTFOLIO_ANALYSIS_PROVIDERS,
)
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
    parser.add_argument(
        "--github-portfolio",
        action="store_true",
        help=(
            "optionally rank GitHub repositories against the confirmed job and "
            "require a separate project-selection approval"
        ),
    )
    parser.add_argument(
        "--github-username",
        help=(
            "GitHub username for public discovery; omit to use the authenticated "
            "user from GITHUB_TOKEN"
        ),
    )
    parser.add_argument(
        "--github-include-private",
        action="store_true",
        help=(
            "allow private repositories to be considered (requires GITHUB_TOKEN; "
            "private content remains local by default)"
        ),
    )
    parser.add_argument(
        "--github-allow-private-provider",
        action="store_true",
        help=(
            "explicitly allow selected private-repository evidence to be sent to "
            "the chosen analysis provider"
        ),
    )
    parser.add_argument(
        "--github-analysis-provider",
        choices=PORTFOLIO_ANALYSIS_PROVIDERS,
        default=None,
        help=(
            "bounded provider for GitHub portfolio ranking (default: gemma_local; "
            "grok_cli uses a locked one-turn deny-all adapter; no fallback)"
        ),
    )
    parser.add_argument(
        "--github-project",
        dest="github_project_ids",
        action="append",
        default=None,
        metavar="REPOSITORY_ID",
        help=(
            "explicit repository ID to approve for non-interactive --yes runs; "
            "repeat exactly two or three times"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _validate_mode_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    github_enabled = bool(getattr(args, "github_portfolio", False))
    github_username = getattr(args, "github_username", None)
    github_include_private = bool(
        getattr(args, "github_include_private", False)
    )
    github_allow_private_provider = bool(
        getattr(args, "github_allow_private_provider", False)
    )
    github_analysis_provider = getattr(args, "github_analysis_provider", None)
    github_project_ids = tuple(getattr(args, "github_project_ids", ()) or ())
    if (
        github_enabled
        and github_analysis_provider is not None
        and github_analysis_provider not in PORTFOLIO_ANALYSIS_PROVIDERS
    ):
        parser.error(
            "--github-analysis-provider supports only gemma_local or grok_cli"
        )
    if github_enabled:
        from resume_tailor.backend.github.client import (
            GitHubConfigurationError,
            validate_github_username,
        )

        if github_username is not None:
            try:
                args.github_username = validate_github_username(github_username)
                github_username = args.github_username
            except GitHubConfigurationError as exc:
                parser.error(str(exc))
        token_present = bool(os.environ.get("GITHUB_TOKEN", "").strip())
        if github_username is None and not token_present:
            parser.error(str(GitHubConfigurationError("missing_identity")))
        if github_include_private and not token_present:
            parser.error(str(GitHubConfigurationError("private_requires_token")))
        analysis_provider = normalize_analysis_provider(
            getattr(args, "analysis_provider", DEFAULT_ANALYSIS_PROVIDER)
        )
        if analysis_provider not in PORTFOLIO_ANALYSIS_PROVIDERS:
            parser.error(
                "--github-portfolio requires --analysis-provider gemma_local "
                "or grok_cli"
            )
        if getattr(args, "writer_provider", "ollama") != "ollama":
            parser.error(
                "--github-portfolio requires --writer-provider ollama"
            )
        initial_qa_provider = getattr(args, "initial_qa_provider", None)
        if initial_qa_provider not in {None, "gemma_local", "grok"}:
            parser.error(
                "--github-portfolio permits only gemma_local or grok for "
                "--initial-qa-provider"
            )
        private_stays_local = (
            github_include_private and not github_allow_private_provider
        )
        effective_portfolio_provider = github_analysis_provider or "gemma_local"
        if private_stays_local and (
            effective_portfolio_provider != "gemma_local"
            or analysis_provider != "gemma_local"
            or initial_qa_provider == "grok"
        ):
            parser.error(
                "private GitHub evidence without "
                "--github-allow-private-provider requires Gemma Local for "
                "portfolio ranking, résumé analysis, and any preselected QA"
            )
    if not github_enabled and any(
        (
            github_username,
            github_include_private,
            github_allow_private_provider,
            github_analysis_provider,
            github_project_ids,
        )
    ):
        parser.error(
            "GitHub portfolio options require --github-portfolio"
        )
    if github_allow_private_provider and not github_include_private:
        parser.error(
            "--github-allow-private-provider requires --github-include-private"
        )
    if github_project_ids:
        normalized_ids = [value.strip() for value in github_project_ids]
        if any(not value for value in normalized_ids):
            parser.error("--github-project values must not be empty")
        if len(normalized_ids) not in {2, 3}:
            parser.error("repeat --github-project exactly two or three times")
        if len(set(normalized_ids)) != len(normalized_ids):
            parser.error("--github-project values must be unique")
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
        github_portfolio=bool(getattr(args, "github_portfolio", False)),
        github_username=getattr(args, "github_username", None),
        github_include_private=bool(
            getattr(args, "github_include_private", False)
        ),
        github_allow_private_provider=bool(
            getattr(args, "github_allow_private_provider", False)
        ),
        github_analysis_provider=(
            getattr(args, "github_analysis_provider", None) or "gemma_local"
            if bool(getattr(args, "github_portfolio", False))
            else None
        ),
        github_project_ids=tuple(
            getattr(args, "github_project_ids", ()) or ()
        ),
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
