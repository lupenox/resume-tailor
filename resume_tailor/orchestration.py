from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .utilities import ApprovalError, ask_for_approval, check_cancelled


ProgressHandler = Callable[[str, str, Mapping[str, Any]], None]
ApprovalHandler = Callable[["ApprovalRequest"], "ApprovalResponse"]
WarningHandler = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class ApprovalRequest:
    kind: str
    title: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    on_presented: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ApprovalResponse:
    action: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PipelineHooks:
    """Interaction boundary shared by the terminal and localhost UI."""

    progress_handler: ProgressHandler | None = None
    approval_handler: ApprovalHandler | None = None
    approval_handler_presents: bool = False
    warning_handler: WarningHandler | None = None
    cancel_event: threading.Event | None = None

    def progress(
        self,
        stage: str,
        message: str,
        **payload: Any,
    ) -> None:
        check_cancelled()
        if self.progress_handler is not None:
            self.progress_handler(stage, message, payload)

    def warning(self, message: str, **payload: Any) -> None:
        check_cancelled()
        if self.warning_handler is not None:
            self.warning_handler(message, payload)
        else:
            print(f"Warning: {message}", file=sys.stderr)

    def approve(
        self,
        *,
        kind: str,
        title: str,
        payload: Mapping[str, Any],
        assume_yes: bool,
        on_presented: Callable[[], None] | None = None,
    ) -> ApprovalResponse:
        check_cancelled()
        request = ApprovalRequest(
            kind=kind,
            title=title,
            payload=payload,
            on_presented=on_presented,
        )
        if assume_yes:
            if request.on_presented is not None:
                request.on_presented()
            ask_for_approval(title, assume_yes=True)
            return ApprovalResponse("approve")
        if self.approval_handler is None:
            if request.on_presented is not None:
                request.on_presented()
            ask_for_approval(title, assume_yes=False)
            return ApprovalResponse("approve")
        if not self.approval_handler_presents and request.on_presented is not None:
            request.on_presented()
        response = self.approval_handler(request)
        check_cancelled()
        if response.action not in {"approve", "use_pasted"}:
            raise ApprovalError(f"{title} was not approved; artifacts were preserved.")
        return response

    def authorize_revision(
        self,
        *,
        payload: Mapping[str, Any],
        provider_name: str = "Antigravity",
    ) -> ApprovalResponse:
        """Request the one optional extra provider call; --yes never bypasses it."""
        check_cancelled()
        title = f"One optional {provider_name} revision"
        if self.approval_handler is None:
            try:
                answer = input(
                    f"\nInitial QA found material issues. Exactly one {provider_name} "
                    "revision is available. Type 'revise' to authorize that "
                    "provider call, or press Enter to stop and keep artifacts: "
                ).strip()
            except (EOFError, OSError):
                answer = ""
            check_cancelled()
            return ApprovalResponse(
                "revise_once" if answer == "revise" else "stop"
            )
        response = self.approval_handler(
            ApprovalRequest(kind="qa_revision", title=title, payload=payload)
        )
        check_cancelled()
        if response.action not in {"revise_once", "stop"}:
            raise ApprovalError(
                "The revision authorization response was invalid; artifacts were "
                "preserved."
            )
        return response

    def select_initial_qa_provider(
        self,
        *,
        options: list[Mapping[str, Any]],
        default_provider: str | None = None,
        previous_failure: Mapping[str, Any] | None = None,
        assume_yes: bool = False,
    ) -> ApprovalResponse:
        """Pause before Step 9 and require an explicit Initial QA provider choice.

        Environment defaults may preselect a radio option but never auto-launch
        QA. ``assume_yes`` confirms the preselected/default provider for CLI
        automation only after Steps 1–8 have already completed.
        """
        check_cancelled()
        title = "Select Initial QA provider"
        payload: dict[str, Any] = {
            "options": list(options),
            "default_provider": default_provider,
            "previous_failure": dict(previous_failure or {}),
        }
        available_ids = [
            str(item.get("provider_id"))
            for item in options
            if item.get("available") and item.get("provider_id")
        ]
        if self.approval_handler is None:
            if assume_yes:
                # Confirm a preselection only; never probe-and-switch silently.
                chosen = default_provider
                if chosen is None:
                    # Non-interactive automation without an explicit preselection
                    # prefers Codex when present (historical Step 9 default).
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
                    f"  Previous attempt failed "
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

        response = self.approval_handler(
            ApprovalRequest(kind="initial_qa_provider", title=title, payload=payload)
        )
        check_cancelled()
        if response.action not in {"select", "stop"}:
            raise ApprovalError(
                "Initial QA provider selection was invalid; artifacts were preserved."
            )
        return response

    def approve_revised_content(
        self,
        *,
        payload: Mapping[str, Any],
    ) -> ApprovalResponse:
        """Require a distinct human decision before rendering revision 1."""
        check_cancelled()
        title = "Revision 1 content diff"
        if self.approval_handler is None:
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
        response = self.approval_handler(
            ApprovalRequest(kind="revised_content", title=title, payload=payload)
        )
        check_cancelled()
        if response.action not in {"approve", "reject"}:
            raise ApprovalError(
                "The revision-content approval response was invalid; artifacts "
                "were preserved."
            )
        return response
