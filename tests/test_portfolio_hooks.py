from __future__ import annotations

import builtins

import pytest

from resume_tailor.backend.engine.orchestration import (
    ApprovalRequest,
    ApprovalResponse,
    PipelineHooks,
    portfolio_selectable_repository_ids,
)
from resume_tailor.backend.utils.utilities import ApprovalError
from resume_tailor.ui.terminal_hooks import TerminalPipelineHooks


def _payload() -> dict[str, object]:
    return {
        "allowed_repository_ids": ["101", "202", "303"],
        "repository_aliases": {
            "synthetic/alpha": "101",
            "synthetic/beta": "202",
            "synthetic/gamma": "303",
        },
        "recommended_repository_ids": ["101", "202"],
        "ranked_repositories": [
            {
                "repository_id": "101",
                "full_name": "synthetic/alpha",
                "visibility": "public",
                "source_url": "https://github.com/synthetic/alpha",
                "component_scores": {"job_requirement_relevance": 90},
                "total_score": 88.5,
                "matched_requirement_ids": ["job_requirements.0"],
                "supporting_evidence_ids": ["repo-101:readme"],
                "supporting_evidence": [
                    {
                        "evidence_id": "repo-101:readme",
                        "category": "readme",
                        "source_path": "README.md",
                        "source_url": (
                            "https://github.com/synthetic/alpha/blob/abc/README.md"
                        ),
                        "exact_text": "Synthetic bounded evidence.",
                    }
                ],
                "inclusion_rationale": "Matches the confirmed Python requirement.",
                "recommended_resume_angle": "Evidence-backed Python delivery.",
                "risks": ["No deployment proof."],
                "diversity_category": "backend",
            },
            {"repository_id": "202", "full_name": "synthetic/beta"},
            {"repository_id": "303", "full_name": "synthetic/gamma"},
        ],
        "eligible_repositories": [
            {
                "repository_id": "101",
                "full_name": "synthetic/alpha",
                "visibility": "public",
                "ranked": True,
            },
            {
                "repository_id": "404",
                "full_name": "synthetic/eligible-not-ranked",
                "visibility": "public",
                "source_url": (
                    "https://github.com/synthetic/eligible-not-ranked"
                ),
                "warnings": ["public_username_ownership_unverified"],
                "ranked": False,
            },
        ],
        "excluded_repositories": [],
    }


def test_portfolio_selectable_ids_prefer_explicit_allowlist() -> None:
    assert portfolio_selectable_repository_ids(_payload()) == ("101", "202", "303")


def test_assume_yes_requires_explicit_portfolio_ids() -> None:
    with pytest.raises(ApprovalError, match="explicit --github-project"):
        PipelineHooks().approve_github_portfolio(
            payload=_payload(),
            assume_yes=True,
        )


def test_generic_approval_api_cannot_bypass_portfolio_selection_under_yes() -> None:
    with pytest.raises(ApprovalError, match="explicit --github-project"):
        PipelineHooks().approve(
            kind="github_portfolio_selection",
            title="GitHub portfolio selection",
            payload=_payload(),
            assume_yes=True,
        )


def test_assume_yes_normalizes_explicit_repository_aliases() -> None:
    response = PipelineHooks().approve_github_portfolio(
        payload=_payload(),
        assume_yes=True,
        explicit_repository_ids=("synthetic/alpha", "SYNTHETIC/BETA"),
    )

    assert response == ApprovalResponse(
        "approve",
        {"repository_ids": ["101", "202"]},
    )


def test_callback_portfolio_gate_supports_explicit_skip() -> None:
    seen: list[ApprovalRequest] = []

    def respond(request: ApprovalRequest) -> ApprovalResponse:
        seen.append(request)
        return ApprovalResponse("skip")

    response = PipelineHooks(approval_handler=respond).approve_github_portfolio(
        payload=_payload()
    )

    assert response == ApprovalResponse("skip")
    assert seen[0].kind == "github_portfolio_selection"


@pytest.mark.parametrize(
    "repository_ids",
    [
        ["101"],
        ["101", "202", "303", "404"],
        ["101", "101"],
        ["101", "unknown"],
    ],
)
def test_callback_portfolio_gate_rejects_invalid_selections(
    repository_ids: list[str],
) -> None:
    hooks = PipelineHooks(
        approval_handler=lambda _request: ApprovalResponse(
            "approve",
            {"repository_ids": repository_ids},
        )
    )

    with pytest.raises(ApprovalError):
        hooks.approve_github_portfolio(payload=_payload())


def test_terminal_portfolio_gate_uses_ranked_ids_without_printing_other_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _payload()
    payload["github_token"] = "synthetic-secret-token"
    monkeypatch.setattr(builtins, "input", lambda _prompt: "1, synthetic/beta")

    response = TerminalPipelineHooks().approve_github_portfolio(payload=payload)

    output = capsys.readouterr().out
    assert response.data["repository_ids"] == ["101", "202"]
    assert "synthetic/alpha" in output
    assert "repo-101:readme" in output
    assert "Eligible but not deep-ranked: 1" in output
    assert "synthetic/eligible-not-ranked" in output
    assert "review only; not selectable" in output
    assert "https://github.com/synthetic/alpha/blob/abc/README.md" in output
    assert "synthetic-secret-token" not in output


def test_terminal_warning_redacts_github_token_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "github_pat_SYNTHETIC_SECRET_VALUE"

    TerminalPipelineHooks().warning(f"request failed with {secret}")

    assert secret not in capsys.readouterr().err
