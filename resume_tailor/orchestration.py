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
                    f"\nCodex found material issues. Exactly one {provider_name} "
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
