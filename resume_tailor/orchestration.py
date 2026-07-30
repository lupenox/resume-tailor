from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .utilities import ApprovalError, ask_for_approval, check_cancelled


ProgressHandler = Callable[[str, str, Mapping[str, Any]], None]
ApprovalHandler = Callable[["ApprovalRequest"], "ApprovalResponse"]


@dataclass(frozen=True)
class ApprovalRequest:
    kind: str
    title: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalResponse:
    action: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PipelineHooks:
    """Interaction boundary shared by the terminal and localhost UI."""

    progress_handler: ProgressHandler | None = None
    approval_handler: ApprovalHandler | None = None
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

    def approve(
        self,
        *,
        kind: str,
        title: str,
        payload: Mapping[str, Any],
        assume_yes: bool,
    ) -> ApprovalResponse:
        check_cancelled()
        if assume_yes:
            ask_for_approval(title, assume_yes=True)
            return ApprovalResponse("approve")
        if self.approval_handler is None:
            ask_for_approval(title, assume_yes=False)
            return ApprovalResponse("approve")
        response = self.approval_handler(
            ApprovalRequest(kind=kind, title=title, payload=payload)
        )
        check_cancelled()
        if response.action not in {"approve", "use_pasted"}:
            raise ApprovalError(f"{title} was not approved; artifacts were preserved.")
        return response
