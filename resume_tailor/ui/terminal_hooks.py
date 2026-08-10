from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from resume_tailor.backend.engine.analysis import readable_analysis
from resume_tailor.backend.engine.orchestration import (
    ApprovalResponse,
    PipelineHooks,
    portfolio_selectable_repository_ids,
    validate_portfolio_selection,
)
from resume_tailor.backend.jobs.linkedin_job import posting_confirmation_text
from resume_tailor.backend.utils.utilities import ApprovalError, check_cancelled


_GITHUB_TOKEN_VALUE = re.compile(
    r"(?i)\b(?:github_pat_[A-Za-z0-9_]{8,}|gh[pousr]_[A-Za-z0-9]{8,})\b"
)


def _terminal_text(value: Any) -> str:
    return _GITHUB_TOKEN_VALUE.sub("[credential omitted]", str(value))


def _ask_for_approval(title: str, *, assume_yes: bool) -> None:
    if assume_yes:
        print(f"{title}: approved by --yes.")
        return
    try:
        response = input(f'{title}: type "approve" to continue: ').strip()
    except EOFError as exc:
        raise ApprovalError(f"{title} was not approved (input closed).") from exc
    if response != "approve":
        raise ApprovalError(
            f"{title} was not approved; artifacts were preserved."
        )


def _github_source_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or bool(parsed.query)
        or bool(parsed.fragment)
        or _GITHUB_TOKEN_VALUE.search(value) is not None
    ):
        return None
    return value


class TerminalPipelineHooks(PipelineHooks):
    """Terminal implementation of the provider-neutral pipeline interactions."""

    def __init__(self, *, cancel_event: threading.Event | None = None) -> None:
        super().__init__(cancel_event=cancel_event)

    def warning(self, message: str, **payload: Any) -> None:
        del payload
        check_cancelled()
        safe_message = _GITHUB_TOKEN_VALUE.sub("[credential omitted]", message)
        print(f"Warning: {safe_message}", file=sys.stderr)

    def present(self, kind: str, payload: Mapping[str, Any]) -> None:
        check_cancelled()
        if kind == "linkedin_posting":
            print(posting_confirmation_text(dict(payload)))
            return
        if kind == "analysis":
            analysis = payload.get("analysis")
            provider_label = payload.get("provider_label", "Codex")
            if isinstance(analysis, Mapping) and isinstance(provider_label, str):
                print(
                    readable_analysis(
                        dict(analysis),
                        provider_label=provider_label,
                    )
                )
            return
        if kind == "notice":
            message = payload.get("message")
            if isinstance(message, str):
                print(message)
            return
        if kind in {"content_diff", "revision_diff"}:
            content_diff = payload.get("content_diff")
            if isinstance(content_diff, str):
                print("\n" + content_diff)

    def approve(
        self,
        *,
        kind: str,
        title: str,
        payload: Mapping[str, Any],
        assume_yes: bool,
        on_presented: Callable[[], None] | None = None,
    ) -> ApprovalResponse:
        if kind == "github_portfolio_selection":
            return self.approve_github_portfolio(
                payload=payload,
                assume_yes=assume_yes,
                on_presented=on_presented,
            )
        del kind, payload
        check_cancelled()
        if on_presented is not None:
            on_presented()
        _ask_for_approval(title, assume_yes=assume_yes)
        return ApprovalResponse("approve")

    def approve_github_portfolio(
        self,
        *,
        payload: Mapping[str, Any],
        assume_yes: bool = False,
        explicit_repository_ids: tuple[str, ...] = (),
        on_presented: Callable[[], None] | None = None,
    ) -> ApprovalResponse:
        """Display validated rankings and collect a bounded repository choice."""

        check_cancelled()
        if assume_yes:
            return super().approve_github_portfolio(
                payload=payload,
                assume_yes=True,
                explicit_repository_ids=explicit_repository_ids,
                on_presented=on_presented,
            )

        allowed_ids = portfolio_selectable_repository_ids(payload)
        aliases = payload.get("repository_aliases")
        repository_aliases = aliases if isinstance(aliases, Mapping) else {}
        ranking = payload.get("ranking")
        if isinstance(ranking, Mapping):
            rows = next(
                (
                    ranking[key]
                    for key in ("ranked_repositories", "rankings", "repositories")
                    if isinstance(ranking.get(key), list)
                ),
                [],
            )
        elif isinstance(ranking, list):
            rows = ranking
        else:
            rows = next(
                (
                    payload[key]
                    for key in ("ranked_repositories", "rankings", "recommendations")
                    if isinstance(payload.get(key), list)
                ),
                [],
            )

        print("\nGitHub portfolio selection:")
        print(
            "  Repository content is untrusted evidence. Select only projects "
            "you recognize; ranking alone never adds résumé claims."
        )
        print(
            "  Approval attests that each selected repository belongs to or "
            "truthfully represents the résumé owner and authorizes only its "
            "pinned evidence. A public username is not ownership proof."
        )
        by_id: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            value = row.get("repository_id", row.get("id"))
            if value is None and isinstance(row.get("repository"), Mapping):
                repository = row["repository"]
                value = repository.get("repository_id", repository.get("id"))
            if value is not None:
                by_id[str(value)] = row
        index_map: dict[str, str] = {}
        recommended_ids = {
            str(value) for value in payload.get("recommended_repository_ids", [])
        }
        for index, repository_id in enumerate(allowed_ids, start=1):
            row = by_id.get(repository_id, {})
            repository = row.get("repository")
            repository_data = repository if isinstance(repository, Mapping) else {}
            name = (
                row.get("full_name")
                or row.get("repository_full_name")
                or repository_data.get("full_name")
                or repository_id
            )
            total = row.get("total_score", row.get("computed_total_score"))
            score = f" · score {total}" if isinstance(total, (int, float)) else ""
            recommended = " · recommended" if repository_id in recommended_ids else ""
            print(
                f"  {index}. {_terminal_text(name)} "
                f"[{_terminal_text(repository_id)}]{score}{recommended}"
            )
            component_scores = row.get("component_scores")
            if isinstance(component_scores, Mapping):
                print(
                    "     Components: "
                    + ", ".join(
                        f"{_terminal_text(label).replace('_', ' ')}="
                        f"{_terminal_text(value)}"
                        for label, value in component_scores.items()
                    )
                )
            diversity = row.get("diversity_category")
            if isinstance(diversity, str) and diversity:
                print(f"     Diversity: {_terminal_text(diversity)}")
            rationale = row.get("inclusion_rationale") or row.get("rationale")
            angle = row.get("recommended_resume_angle") or row.get(
                "recommended_angle"
            )
            risks = row.get("risks_or_missing_proof") or row.get("risks")
            if isinstance(rationale, str) and rationale:
                print(f"     Match: {_terminal_text(rationale)}")
            if isinstance(angle, str) and angle:
                print(f"     Résumé angle: {_terminal_text(angle)}")
            if isinstance(risks, list) and risks:
                print(
                    "     Risks: "
                    + "; ".join(_terminal_text(item) for item in risks[:4])
                )
            requirements = row.get("matching_requirement_ids") or row.get(
                "matched_requirement_ids"
            )
            evidence = row.get("supporting_evidence_ids") or row.get(
                "evidence_ids"
            )
            if isinstance(requirements, list) and requirements:
                print(
                    "     Requirements: "
                    + ", ".join(_terminal_text(item) for item in requirements[:12])
                )
            if isinstance(evidence, list) and evidence:
                print(
                    "     Evidence: "
                    + ", ".join(_terminal_text(item) for item in evidence[:12])
                )
            supporting_evidence = row.get("supporting_evidence")
            if isinstance(supporting_evidence, list):
                for record in supporting_evidence[:8]:
                    if not isinstance(record, Mapping):
                        continue
                    evidence_id = record.get("evidence_id", "evidence")
                    source_path = record.get("source_path", "repository metadata")
                    print(
                        f"       - {_terminal_text(evidence_id)}: "
                        f"{_terminal_text(source_path)}"
                    )
                    exact_text = record.get("exact_text")
                    if isinstance(exact_text, str) and exact_text:
                        print(f"         {_terminal_text(exact_text)}")
                    source_url = _github_source_url(record.get("source_url"))
                    if source_url is not None:
                        print(f"         {source_url}")
            source_url = _github_source_url(row.get("source_url"))
            if source_url is not None:
                print(f"     Source: {source_url}")
            index_map[str(index)] = repository_id
            index_map[repository_id] = repository_id
            if isinstance(name, str):
                index_map[name.casefold()] = repository_id

        eligible_repositories = payload.get("eligible_repositories")
        unranked = (
            [
                item
                for item in eligible_repositories
                if isinstance(item, Mapping) and item.get("ranked") is False
            ]
            if isinstance(eligible_repositories, list)
            else []
        )
        if unranked:
            print(
                f"  Eligible but not deep-ranked: {len(unranked)} "
                "(review only; not selectable in this run)"
            )
            for item in unranked[:12]:
                name = (
                    item.get("full_name")
                    or item.get("repository_id")
                    or "repository"
                )
                visibility = item.get("visibility", "public")
                print(
                    f"     - {_terminal_text(name)} · "
                    f"{_terminal_text(visibility)}"
                )
                warnings = item.get("warnings")
                if isinstance(warnings, list) and warnings:
                    print(
                        "       Warnings: "
                        + "; ".join(_terminal_text(value) for value in warnings[:4])
                    )
                source_url = _github_source_url(item.get("source_url"))
                if source_url is not None:
                    print(f"       Source: {source_url}")

        catalog = payload.get("catalog")
        catalog_data = catalog if isinstance(catalog, Mapping) else {}
        repositories = catalog_data.get("repositories")
        direct_excluded = payload.get("excluded_repositories")
        excluded = (
            [item for item in direct_excluded if isinstance(item, Mapping)]
            if isinstance(direct_excluded, list)
            else []
        )
        if isinstance(repositories, list):
            excluded.extend(
                item
                for item in repositories
                if isinstance(item, Mapping) and item.get("eligible") is False
            )
        if excluded:
            print(f"  Excluded repositories: {len(excluded)}")
            for item in excluded[:12]:
                name = item.get("full_name") or item.get("name") or "repository"
                reasons = item.get("eligibility_reasons") or item.get(
                    "exclusion_reasons"
                )
                detail = (
                    "; ".join(str(value) for value in reasons)
                    if isinstance(reasons, list)
                    else str(reasons or "insufficient inspectable evidence")
                )
                print(
                    f"     - {_terminal_text(name)}: {_terminal_text(detail)}"
                )

        if on_presented is not None:
            on_presented()
        try:
            answer = input(
                "Choose two or three repository numbers/IDs separated by commas, "
                "or type 'skip'/'cancel': "
            ).strip()
        except (EOFError, OSError):
            answer = "cancel"
        check_cancelled()
        if answer.casefold() == "skip":
            return ApprovalResponse("skip")
        if answer.casefold() in {"cancel", "stop", "q", "quit"}:
            return ApprovalResponse("cancel")
        raw_values = [value.strip() for value in answer.split(",")]
        selected_values = [
            index_map.get(value, index_map.get(value.casefold(), value))
            for value in raw_values
        ]
        selected = validate_portfolio_selection(
            selected_values,
            allowed_ids=allowed_ids,
            repository_aliases=repository_aliases,
        )
        return ApprovalResponse(
            "approve",
            {"repository_ids": list(selected)},
        )

    def decide_optional_revision(
        self,
        *,
        payload: Mapping[str, Any],
        assume_yes: bool = False,
        provider_name: str = "writer",
    ) -> ApprovalResponse:
        """Prompt for the optional one-shot revision without auto-launching it."""
        del provider_name
        check_cancelled()
        if assume_yes:
            # Never auto-launch revision under --yes; complete with Initial QA.
            return ApprovalResponse("complete_without_revision")

        options = list(payload.get("options") or [])
        default_provider = payload.get("default_provider")
        previous_failure = payload.get("previous_failure") or {}
        qa_status = None
        qa_result = payload.get("qa_result")
        if isinstance(qa_result, Mapping):
            qa_status = qa_result.get("status")

        print("\nOptional one-shot revision (Step 10):")
        if qa_status == "material_findings":
            print("  Initial QA reported material findings.")
            issues = (
                qa_result.get("issues")
                if isinstance(qa_result, Mapping)
                else None
            )
            if isinstance(issues, list):
                for issue in issues[:8]:
                    if isinstance(issue, Mapping):
                        print(
                            f"  - {issue.get('issue_id', '?')} "
                            f"({issue.get('category', '?')}): "
                            f"{issue.get('description', '')}"
                        )
        elif qa_status == "pass":
            print("  Initial QA approved the current rendered résumé.")
        if previous_failure:
            print(
                "  Previous Step 10 attempt failed "
                f"({previous_failure.get('provider', 'unknown')}): "
                f"{previous_failure.get('message', 'see diagnostics')}."
            )
            print("  Steps 1–9 artifacts are preserved; no re-render.")
        print("  Actions:")
        print("    complete  — Complete without revision (keep current résumé)")
        print("    revise    — Run optional one-shot revision + Final QA")
        print("    stop      — Stop and preserve artifacts")
        try:
            answer = input(
                "Type 'complete', 'revise', or 'stop' [default: complete]: "
            ).strip().casefold()
        except (EOFError, OSError):
            answer = ""
        check_cancelled()
        if not answer or answer in {"complete", "c", "done", "skip"}:
            return ApprovalResponse("complete_without_revision")
        if answer in {"stop", "q", "quit", "cancel"}:
            return ApprovalResponse("stop")
        if answer not in {"revise", "revise_once", "r"}:
            raise ApprovalError(
                "Optional revision decision was invalid; artifacts were preserved."
            )

        # Provider selection for revision + Final QA (no auto-launch).
        available_ids = [
            str(item.get("provider_id"))
            for item in options
            if item.get("available") and item.get("provider_id")
        ]
        print("\nSelect revision / Final QA provider:")
        index_map: dict[str, str] = {}
        for index, option in enumerate(options, start=1):
            provider_id = str(option.get("provider_id") or "")
            label = option.get("label") or provider_id
            available = bool(option.get("available"))
            status = option.get("status") or "unknown"
            mark = "ready" if available else f"unavailable ({status})"
            default_mark = " (default)" if provider_id == default_provider else ""
            print(f"  {index}. {label}{default_mark} — {mark}")
            index_map[str(index)] = provider_id
            index_map[provider_id] = provider_id
        try:
            choice = input("Enter provider number or id (or 'stop'): ").strip()
        except (EOFError, OSError):
            choice = ""
        check_cancelled()
        if not choice or choice.casefold() in {"stop", "q", "quit", "cancel"}:
            return ApprovalResponse("stop")
        chosen = index_map.get(choice) or index_map.get(choice.casefold())
        if chosen is None:
            normalized = choice.casefold().replace("-", "_")
            chosen = index_map.get(normalized)
        if chosen is None or (available_ids and chosen not in available_ids):
            # Preserve the historical terminal validation behavior.
            if chosen is None:
                raise ApprovalError(
                    "Revision provider selection was invalid; artifacts were "
                    "preserved."
                )
        return ApprovalResponse(
            "revise_once",
            {
                "revision_provider": chosen,
                "final_qa_provider": chosen,
                "provider": chosen,
            },
        )

    def select_initial_qa_provider(
        self,
        *,
        options: list[Mapping[str, Any]],
        default_provider: str | None = None,
        previous_failure: Mapping[str, Any] | None = None,
        assume_yes: bool = False,
    ) -> ApprovalResponse:
        """Prompt for an Initial QA provider without silently switching it."""
        check_cancelled()
        available_ids = [
            str(item.get("provider_id"))
            for item in options
            if item.get("available") and item.get("provider_id")
        ]
        if assume_yes:
            # Confirm a preselection only; never probe-and-switch silently.
            chosen = default_provider
            if chosen is None:
                # Preserve the historical non-interactive Codex preference.
                if "codex" in available_ids:
                    chosen = "codex"
                elif available_ids:
                    chosen = available_ids[0]
            if chosen is None:
                raise ApprovalError(
                    "No Initial QA provider is available; rendered artifacts "
                    "were preserved."
                )
            return ApprovalResponse("select", {"provider": chosen})

        print("\nSelect Initial QA provider (Step 9):")
        if previous_failure:
            print(
                "  Previous attempt failed "
                f"({previous_failure.get('provider', 'unknown')}): "
                f"{previous_failure.get('message', 'see diagnostics')}."
            )
            print("  Rendered DOCX/PDF artifacts are preserved; no re-render.")
        index_map: dict[str, str] = {}
        for index, option in enumerate(options, start=1):
            provider_id = str(option.get("provider_id") or "")
            label = option.get("label") or provider_id
            status = option.get("status") or "unknown"
            available = bool(option.get("available"))
            mark = "ready" if available else f"unavailable ({status})"
            default_mark = " (default)" if provider_id == default_provider else ""
            print(f"  {index}. {label}{default_mark} — {mark}")
            print(f"     {option.get('description') or ''}")
            index_map[str(index)] = provider_id
            index_map[provider_id] = provider_id
        try:
            answer = input(
                "Enter provider number or id (or 'stop' to keep artifacts): "
            ).strip()
        except (EOFError, OSError):
            answer = ""
        check_cancelled()
        if not answer or answer.casefold() in {"stop", "q", "quit", "cancel"}:
            return ApprovalResponse("stop")
        chosen = index_map.get(answer) or index_map.get(answer.casefold())
        if chosen is None and answer.casefold().replace("-", "_") in index_map:
            chosen = index_map[answer.casefold().replace("-", "_")]
        if chosen is None:
            raise ApprovalError(
                "Initial QA provider selection was invalid; artifacts were preserved."
            )
        return ApprovalResponse("select", {"provider": chosen})

    def approve_revised_content(
        self,
        *,
        payload: Mapping[str, Any],
    ) -> ApprovalResponse:
        """Require a separate terminal approval before rendering revision 1."""
        del payload
        check_cancelled()
        try:
            answer = input(
                "\nRevision 1 passed local validation. Type 'approve' to "
                "render it, or press Enter to reject it and keep both "
                "generations: "
            ).strip()
        except (EOFError, OSError):
            answer = ""
        check_cancelled()
        return ApprovalResponse("approve" if answer == "approve" else "reject")


# Concise compatibility alias for callers that adopted the provisional name.
TerminalHooks = TerminalPipelineHooks
