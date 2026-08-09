from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from resume_tailor.backend.utils.utilities import ApprovalError, check_cancelled


ProgressHandler = Callable[[str, str, Mapping[str, Any]], None]
ApprovalHandler = Callable[["ApprovalRequest"], "ApprovalResponse"]
WarningHandler = Callable[[str, Mapping[str, Any]], None]
PresentationHandler = Callable[[str, Mapping[str, Any]], None]


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
    presentation_handler: PresentationHandler | None = None

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

    def present(self, kind: str, payload: Mapping[str, Any]) -> None:
        """Present adapter-specific detail without coupling orchestration to I/O."""
        check_cancelled()
        if self.presentation_handler is not None:
            self.presentation_handler(kind, payload)

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
            return ApprovalResponse("approve")
        if self.approval_handler is None:
            raise ApprovalError(
                f"{title} requires an interaction adapter; artifacts were preserved."
            )
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
        """Compatibility wrapper for the optional Step 10 revision decision.

        Prefer :meth:`decide_optional_revision`. ``--yes`` never auto-launches a
        revision provider.
        """
        return self.decide_optional_revision(
            payload=payload,
            assume_yes=False,
            provider_name=provider_name,
        )

    def decide_optional_revision(
        self,
        *,
        payload: Mapping[str, Any],
        assume_yes: bool = False,
        provider_name: str = "writer",
    ) -> ApprovalResponse:
        """Step 10 gate: complete without revision, revise once, or stop.

        No revision or final-QA provider launches until the user explicitly
        chooses ``revise_once`` with a provider id.
        """
        check_cancelled()
        title = "Optional one-shot revision"
        if assume_yes:
            # Never auto-launch revision under --yes; complete with Initial QA.
            return ApprovalResponse("complete_without_revision")
        if self.approval_handler is None:
            raise ApprovalError(
                f"{title} requires an interaction adapter; artifacts were preserved."
            )

        response = self.approval_handler(
            ApprovalRequest(kind="optional_revision", title=title, payload=payload)
        )
        check_cancelled()
        if response.action not in {
            "complete_without_revision",
            "revise_once",
            "stop",
        }:
            raise ApprovalError(
                "Optional revision decision was invalid; artifacts were preserved."
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
        if self.approval_handler is None:
            raise ApprovalError(
                f"{title} requires an interaction adapter; artifacts were preserved."
            )

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
            raise ApprovalError(
                f"{title} requires an interaction adapter; artifacts were preserved."
            )
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
