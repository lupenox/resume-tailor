"""Private workspaces and environments for external provider CLIs."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from resume_tailor.backend.utils.utilities import ModelError


_GITHUB_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "github_token",
        "gh_token",
        "github_enterprise_token",
        "gh_enterprise_token",
    }
)

_TOOL_FREE_PROVIDERS_BY_CAPABILITY = {
    "portfolio_ranking": frozenset({"gemma_local", "grok_cli"}),
    "analysis": frozenset({"gemma_local", "grok_cli"}),
    "qa": frozenset({"gemma_local", "grok"}),
    "writing": frozenset({"ollama"}),
}


def enforce_tool_free_capability(
    *,
    capability: str,
    provider: str,
    restrict_external_tools: bool,
) -> None:
    """Fail unless the selected adapter can enforce a no-tools contract.

    This check lives at the provider capability boundary so a caller cannot
    bypass portfolio safety by invoking an adapter directly. The allowed Grok
    paths additionally construct a one-turn, deny-all, no-web strict-sandbox
    invocation; Gemma/Ollama use structured local inference without agent tools.
    """

    if not restrict_external_tools:
        return
    allowed = _TOOL_FREE_PROVIDERS_BY_CAPABILITY.get(capability)
    if allowed is None:
        raise ModelError("Unknown restricted provider capability.")
    selected = provider.strip().casefold()
    if selected not in allowed:
        raise ModelError(
            f"The {selected or 'selected'} provider cannot be used for restricted "
            f"{capability.replace('_', ' ')} because its adapter cannot hard-disable "
            "shell and network tools."
        )


def external_provider_environment() -> dict[str, str]:
    """Return the ambient environment without GitHub API credentials."""

    environment = dict(os.environ)
    for key in tuple(environment):
        if key.casefold() in _GITHUB_CREDENTIAL_ENV_NAMES:
            environment.pop(key, None)
    return environment


@contextmanager
def isolated_provider_workspace(
    run_directory: Path,
    *,
    prefix: str,
) -> Iterator[Path]:
    """Yield a mode-0700 temporary directory outside the run directory."""

    with tempfile.TemporaryDirectory(prefix=prefix) as workspace_value:
        workspace = Path(workspace_value).resolve()
        workspace.chmod(0o700)
        resolved_run = run_directory.resolve()
        if workspace == resolved_run or resolved_run in workspace.parents:
            raise ModelError("The external provider workspace could not be isolated.")
        yield workspace


def copy_private_input(
    source: Path,
    workspace: Path,
    *,
    filename: str | None = None,
) -> Path:
    """Copy one provider input into its workspace with owner-only permissions."""

    destination = workspace / (filename or source.name)
    try:
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
    except OSError as exc:
        raise ModelError(
            f"Could not prepare isolated provider input {source.name}."
        ) from exc
    return destination


def publish_provider_output(source: Path, destination: Path) -> bool:
    """Publish an exact provider-output byte stream when the provider created one."""

    if not source.is_file():
        return False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise ModelError(
            f"Could not preserve provider output {destination.name}."
        ) from exc
    return True
