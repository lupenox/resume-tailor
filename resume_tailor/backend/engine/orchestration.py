from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from resume_tailor.backend.utils.utilities import ApprovalError, check_cancelled


ProgressHandler = Callable[[str, str, Mapping[str, Any]], None]
ApprovalHandler = Callable[["ApprovalRequest"], "ApprovalResponse"]
WarningHandler = Callable[[str, Mapping[str, Any]], None]
PresentationHandler = Callable[[str, Mapping[str, Any]], None]


def portfolio_selectable_repository_ids(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the stable repository IDs exposed by a validated ranking payload."""

    candidates: Any = payload.get("allowed_repository_ids")
    if not isinstance(candidates, (list, tuple)):
        ranking: Any = payload.get("ranking")
        if isinstance(ranking, Mapping):
            for key in ("ranked_repositories", "rankings", "repositories"):
                value = ranking.get(key)
                if isinstance(value, list):
                    candidates = value
                    break
        elif isinstance(ranking, list):
            candidates = ranking
    if not isinstance(candidates, (list, tuple)):
        for key in ("ranked_repositories", "rankings", "recommendations"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break
    if not isinstance(candidates, (list, tuple)):
        return ()

    identifiers: list[str] = []
    for candidate in candidates:
        value: Any = candidate
        if isinstance(candidate, Mapping):
            value = candidate.get("repository_id", candidate.get("id"))
            if value is None and isinstance(candidate.get("repository"), Mapping):
                repository = candidate["repository"]
                value = repository.get("repository_id", repository.get("id"))
        if isinstance(value, (str, int)):
            identifier = str(value).strip()
            if identifier and identifier not in identifiers:
                identifiers.append(identifier)
    return tuple(identifiers)


def validate_portfolio_selection(
    values: Any,
    *,
    allowed_ids: tuple[str, ...],
    repository_aliases: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or isinstance(values, (str, bytes)):
        raise ApprovalError(
            "GitHub portfolio approval requires two or three repository IDs; "
            "artifacts were preserved."
        )
    aliases = {
        str(alias).strip().casefold(): str(repository_id).strip()
        for alias, repository_id in (repository_aliases or {}).items()
        if str(alias).strip() and str(repository_id).strip()
    }
    supplied = tuple(str(value).strip() for value in values)
    selected = tuple(aliases.get(value.casefold(), value) for value in supplied)
    if len(selected) not in {2, 3} or any(not value for value in selected):
        raise ApprovalError(
            "GitHub portfolio approval requires two or three repository IDs; "
            "artifacts were preserved."
        )
    if len(set(selected)) != len(selected):
        raise ApprovalError(
            "GitHub portfolio approval contained duplicate repositories; "
            "artifacts were preserved."
        )
    allowed = set(allowed_ids)
    unknown = [value for value in selected if value not in allowed]
    if unknown:
        raise ApprovalError(
            "GitHub portfolio approval referenced a repository outside the "
            "validated ranking; artifacts were preserved."
        )
    return selected


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
        if kind == "github_portfolio_selection":
            return self.approve_github_portfolio(
                payload=payload,
                assume_yes=assume_yes,
                on_presented=on_presented,
            )
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

    def approve_github_portfolio(
        self,
        *,
        payload: Mapping[str, Any],
        assume_yes: bool = False,
        explicit_repository_ids: tuple[str, ...] = (),
        on_presented: Callable[[], None] | None = None,
    ) -> ApprovalResponse:
        """Approve two or three ranked repositories, or explicitly skip them.

        ``--yes`` never turns the ranker's recommendation into an approval. It
        approves only repository IDs supplied explicitly by the caller; an
        interactive adapter is required to record a skip.
        """

        check_cancelled()
        title = "GitHub portfolio selection"
        allowed_ids = portfolio_selectable_repository_ids(payload)
        aliases = payload.get("repository_aliases")
        repository_aliases = aliases if isinstance(aliases, Mapping) else {}
        if assume_yes:
            if on_presented is not None:
                on_presented()
            if not explicit_repository_ids:
                raise ApprovalError(
                    "GitHub portfolio selection under --yes requires two or "
                    "three explicit --github-project values; artifacts were "
                    "preserved."
                )
            selected = validate_portfolio_selection(
                explicit_repository_ids,
                allowed_ids=allowed_ids,
                repository_aliases=repository_aliases,
            )
            return ApprovalResponse(
                "approve",
                {"repository_ids": list(selected)},
            )
        if self.approval_handler is None:
            raise ApprovalError(
                f"{title} requires an interaction adapter; artifacts were preserved."
            )
        request = ApprovalRequest(
            kind="github_portfolio_selection",
            title=title,
            payload=payload,
            on_presented=on_presented,
        )
        if not self.approval_handler_presents and on_presented is not None:
            on_presented()
        response = self.approval_handler(request)
        check_cancelled()
        if response.action == "skip":
            return ApprovalResponse("skip")
        if response.action != "approve":
            raise ApprovalError(
                "GitHub portfolio selection was not approved; artifacts were "
                "preserved."
            )
        selected = validate_portfolio_selection(
            response.data.get("repository_ids"),
            allowed_ids=allowed_ids,
            repository_aliases=repository_aliases,
        )
        return ApprovalResponse(
            "approve",
            {"repository_ids": list(selected)},
        )

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
