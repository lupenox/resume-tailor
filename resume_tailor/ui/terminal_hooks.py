from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Mapping
from typing import Any

from resume_tailor.backend.engine.analysis import readable_analysis
from resume_tailor.backend.engine.orchestration import ApprovalResponse, PipelineHooks
from resume_tailor.backend.jobs.linkedin_job import posting_confirmation_text
from resume_tailor.backend.utils.utilities import ApprovalError, check_cancelled


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


class TerminalPipelineHooks(PipelineHooks):
    """Terminal implementation of the provider-neutral pipeline interactions."""

    def __init__(self, *, cancel_event: threading.Event | None = None) -> None:
        super().__init__(cancel_event=cancel_event)

    def warning(self, message: str, **payload: Any) -> None:
        del payload
        check_cancelled()
        print(f"Warning: {message}", file=sys.stderr)

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
        del kind, payload
        check_cancelled()
        if on_presented is not None:
            on_presented()
        _ask_for_approval(title, assume_yes=assume_yes)
        return ApprovalResponse("approve")

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
