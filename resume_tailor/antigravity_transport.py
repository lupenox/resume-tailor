from __future__ import annotations

import errno
import hashlib
from pathlib import Path
from typing import Any

from .utilities import (
    AntigravityLaunchSizeError,
    CommandResult,
    DependencyError,
    ModelError,
    run_command,
)


MAX_ANTIGRAVITY_PROMPT_BYTES = 750_000


def _validated_prompt(prompt: str, *, label: str) -> bytes:
    encoded = prompt.encode("utf-8")
    if len(encoded) > MAX_ANTIGRAVITY_PROMPT_BYTES:
        raise ModelError(
            f"The {label} is {len(encoded):,} UTF-8 bytes and exceeds the "
            f"{MAX_ANTIGRAVITY_PROMPT_BYTES:,}-byte local resource safety limit. "
            "Reduce the confirmed input size and start a new run; no provider "
            "request was launched."
        )
    return encoded


def antigravity_print_args(
    *,
    executable: str,
    schema: Path,
    print_timeout: str,
    agent_mode: str | None = None,
) -> list[str]:
    """Return prompt-free argv for Antigravity's documented stdin print mode."""
    if agent_mode not in {None, "plan"}:
        raise DependencyError(
            "Unsupported Antigravity execution mode for this adapter."
        )
    args = [
        executable,
        "--prompt",
        "--sandbox",
        "--output-format",
        "json",
        "--json-schema",
        str(schema),
        "--print-timeout",
        print_timeout,
    ]
    if agent_mode is not None:
        args.insert(2, f"--mode={agent_mode}")
    return args


def run_antigravity_prompt(
    *,
    executable: str,
    prompt: str,
    prompt_label: str,
    schema: Path,
    print_timeout: str,
    cwd: Path,
    timeout_seconds: int,
    agent_mode: str | None = None,
) -> CommandResult:
    """Run Antigravity with UTF-8 prompt bytes on stdin, never in argv or env."""
    encoded = _validated_prompt(prompt, label=prompt_label)
    args = antigravity_print_args(
        executable=executable,
        schema=schema,
        print_timeout=print_timeout,
        agent_mode=agent_mode,
    )
    try:
        return run_command(
            args,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            input_text=encoded.decode("utf-8"),
        )
    except DependencyError as exc:
        cause = exc.__cause__
        if isinstance(cause, OSError) and cause.errno == errno.E2BIG:
            raise AntigravityLaunchSizeError(
                "Antigravity could not start because the request exceeded the "
                "operating system's command-line size."
            ) from exc
        raise


def antigravity_process_failure(
    result: CommandResult,
    *,
    label: str,
) -> ModelError:
    return ModelError(
        f"{label} exited with status {result.returncode}. Provider stdout and "
        "stderr were omitted from the exception."
    )


def antigravity_parse_diagnostic(result: CommandResult) -> dict[str, Any]:
    """Describe malformed provider output without retaining its content."""
    stdout = result.stdout.encode("utf-8")
    stderr = result.stderr.encode("utf-8")
    return {
        "parse_error": True,
        "returncode": result.returncode,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "provider_output_omitted": True,
    }
