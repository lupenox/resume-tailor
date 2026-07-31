from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


class ExitCode(IntEnum):
    OK = 0
    DEPENDENCY = 3
    INPUT = 4
    MODEL = 10
    WAITING = 11
    APPROVAL = 12
    TRUTHFULNESS = 13
    TEMPLATE = 14
    RENDER = 15
    QA = 16
    INTEGRITY = 17
    CANCELLED = 18


class ResumeTailorError(Exception):
    exit_code = ExitCode.INPUT


class DependencyError(ResumeTailorError):
    exit_code = ExitCode.DEPENDENCY


class AntigravityLaunchSizeError(DependencyError):
    """Antigravity failed to exec because its argument vector was too large."""


class CodexSchemaCompatibilityError(DependencyError):
    """A bundled Codex transport schema cannot be sent safely."""


class InputError(ResumeTailorError):
    exit_code = ExitCode.INPUT


class ModelError(ResumeTailorError):
    exit_code = ExitCode.MODEL


class AntigravityTailoringContractError(ModelError):
    """Antigravity did not satisfy the post-approval tailoring contract."""


class AntigravityResponseEnvelopeError(ModelError):
    """Antigravity returned no unique documented structured-output envelope."""

    def __init__(self, message: str, *, envelope_type: str) -> None:
        super().__init__(message)
        self.envelope_type = envelope_type


class LinkedInResponseEnvelopeError(AntigravityResponseEnvelopeError):
    """LinkedIn retrieval returned no documented structured-output envelope."""


class AntigravityCannotApplyError(ModelError):
    """Antigravity could not apply one authenticated approved edit."""


class AntigravityTechnicalFailureError(ModelError):
    """Antigravity reported a bounded technical tailoring failure."""


class AntigravityTailoringPreflightError(InputError):
    """Local authenticated tailoring inputs are incomplete or inconsistent."""


class WaitingError(ResumeTailorError):
    exit_code = ExitCode.WAITING


class ApprovalError(ResumeTailorError):
    exit_code = ExitCode.APPROVAL


class TruthfulnessError(ResumeTailorError):
    exit_code = ExitCode.TRUTHFULNESS


class SourceEvidenceError(TruthfulnessError):
    """Codex returned invalid or inappropriate local source references."""


class TemplateError(ResumeTailorError):
    exit_code = ExitCode.TEMPLATE


class RenderError(ResumeTailorError):
    exit_code = ExitCode.RENDER


class QAError(ResumeTailorError):
    exit_code = ExitCode.QA


class IntegrityError(ResumeTailorError):
    exit_code = ExitCode.INTEGRITY


class CancellationError(ResumeTailorError):
    exit_code = ExitCode.CANCELLED


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


_DURATION_RE = re.compile(r"^(?P<number>[1-9][0-9]*)(?P<unit>s|m|h)?$")
_CANCELLATION_EVENT: ContextVar[threading.Event | None] = ContextVar(
    "resume_tailor_cancellation_event",
    default=None,
)


def parse_duration(value: str) -> tuple[int, str]:
    """Return a subprocess timeout in seconds and an Antigravity duration."""
    match = _DURATION_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("use a positive duration such as 90s, 15m, or 1h")
    number = int(match.group("number"))
    unit = match.group("unit") or "s"
    multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    seconds = number * multiplier
    if seconds > 24 * 60 * 60:
        raise ValueError("duration must not exceed 24h")
    return seconds, f"{number}{unit}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_text_file(path: Path, *, label: str, max_bytes: int = 500_000) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InputError(f"Cannot read {label} {path}: {exc}") from exc
    if size > max_bytes:
        raise InputError(
            f"{label.capitalize()} is {size:,} bytes; the safety limit is "
            f"{max_bytes:,} bytes."
        )
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError(f"{label.capitalize()} must be UTF-8 text: {path}") from exc
    except OSError as exc:
        raise InputError(f"Cannot read {label} {path}: {exc}") from exc
    if not text.strip():
        raise InputError(f"{label.capitalize()} is empty: {path}")
    return text.strip()


def slugify(value: str, *, fallback: str = "resume") -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:80].rstrip("-") or fallback).strip(".")


def filename_component(value: str, *, fallback: str = "Resume") -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", ascii_value) if part]
    if not parts:
        parts = [fallback]
    formatted = [
        part if part.isupper() and len(part) <= 3 else part.capitalize()
        for part in parts
    ]
    return "-".join(formatted)[:100].rstrip("-")


def create_unique_run_dir(
    output_dir: Path,
    company: str,
    role: str,
    *,
    now: datetime | None = None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = f"{slugify(company, fallback='company')}-{slugify(role, fallback='role')}-{stamp}"
    for suffix in ("", *(f"-{number}" for number in range(2, 1000))):
        candidate = output_dir / f"{base}{suffix}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise InputError(f"Could not create a unique run directory under {output_dir}")


def rename_run_directory(
    run_directory: Path,
    company: str,
    role: str,
    *,
    now: datetime | None = None,
) -> Path:
    """Atomically rename a provisional run after URL-derived identity is approved."""
    resolved = run_directory.resolve()
    parent = resolved.parent
    if not resolved.is_dir() or resolved == parent:
        raise InputError(f"Cannot rename invalid run directory: {run_directory}")
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = f"{slugify(company, fallback='company')}-{slugify(role, fallback='role')}-{stamp}"
    for suffix in ("", *(f"-{number}" for number in range(2, 1000))):
        candidate = parent / f"{base}{suffix}"
        if candidate.exists():
            continue
        try:
            resolved.rename(candidate)
        except FileExistsError:
            continue
        except OSError as exc:
            raise InputError(
                f"Could not rename URL run directory for the extracted posting: {exc}"
            ) from exc
        return candidate
    raise InputError(f"Could not create a unique run directory under {parent}")


def require_executable(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise DependencyError(
            f"Required executable '{name}' was not found in PATH. See README.md "
            "for dependency installation instructions."
        )
    return resolved


@contextmanager
def cancellable_commands(cancel_event: threading.Event | None) -> Iterator[None]:
    """Make shared subprocess calls cooperatively cancellable in this context."""
    token = _CANCELLATION_EVENT.set(cancel_event)
    try:
        yield
    finally:
        _CANCELLATION_EVENT.reset(token)


def check_cancelled() -> None:
    event = _CANCELLATION_EVENT.get()
    if event is not None and event.is_set():
        raise CancellationError("The run was cancelled; useful artifacts were preserved.")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
    heartbeat_handler: Callable[[float, bool], None] | None = None,
    heartbeat_interval_seconds: float = 15.0,
) -> CommandResult:
    """Run an argument array. This project intentionally never invokes a shell."""
    check_cancelled()
    if heartbeat_interval_seconds <= 0:
        raise InputError("Subprocess heartbeat interval must be positive.")
    try:
        process = subprocess.Popen(
            [str(arg) for arg in args],
            cwd=str(cwd),
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=dict(env) if env is not None else None,
            shell=False,
            start_new_session=True,
        )
    except OSError as exc:
        raise DependencyError(f"Could not run {Path(args[0]).name}: {exc}") from exc

    started = time.monotonic()
    deadline = started + timeout_seconds
    next_heartbeat = started + heartbeat_interval_seconds
    if heartbeat_handler is not None:
        try:
            heartbeat_handler(0.0, True)
        except BaseException:
            _stop_process(process)
            raise
    pending_input = input_text
    while True:
        event = _CANCELLATION_EVENT.get()
        if event is not None and event.is_set():
            _stop_process(process)
            raise CancellationError(
                "The run was cancelled; the active subprocess was stopped and "
                "useful artifacts were preserved."
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_process(process)
            raise ModelError(
                f"Command timed out after {timeout_seconds}s: {Path(args[0]).name}. "
                "The full subprocess group was stopped; retry with a larger bounded "
                "--timeout only after confirming the local provider is responsive."
            )
        now = time.monotonic()
        if heartbeat_handler is not None and now >= next_heartbeat:
            try:
                heartbeat_handler(now - started, process.poll() is None)
            except BaseException:
                _stop_process(process)
                raise
            next_heartbeat = now + heartbeat_interval_seconds
        try:
            poll_timeout = min(0.25, remaining)
            if heartbeat_handler is not None:
                poll_timeout = min(
                    poll_timeout,
                    max(0.001, next_heartbeat - time.monotonic()),
                )
            stdout, stderr = process.communicate(
                input=pending_input,
                timeout=poll_timeout,
            )
            break
        except subprocess.TimeoutExpired:
            pending_input = None

    if heartbeat_handler is not None:
        heartbeat_handler(time.monotonic() - started, False)

    return CommandResult(
        tuple(str(arg) for arg in args),
        stdout,
        stderr,
        process.returncode,
    )


def concise_process_error(result: CommandResult, label: str) -> str:
    detail = (result.stderr or result.stdout).strip()
    if len(detail) > 2_000:
        detail = detail[-2_000:]
    return f"{label} exited with status {result.returncode}" + (
        f":\n{detail}" if detail else "."
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ask_for_approval(stage: str, *, assume_yes: bool) -> None:
    if assume_yes:
        print(f"{stage}: approved by --yes.")
        return
    try:
        response = input(f'{stage}: type "approve" to continue: ').strip()
    except EOFError as exc:
        raise ApprovalError(f"{stage} was not approved (input closed).") from exc
    if response != "approve":
        raise ApprovalError(f"{stage} was not approved; artifacts were preserved.")


def relative_artifacts(run_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    )


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from flatten_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from flatten_strings(nested)
