from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .utilities import InputError


CLIPBOARD_BACKENDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wl-paste", ("wl-paste", "--no-newline")),
    ("xclip", ("xclip", "-selection", "clipboard", "-o")),
    ("xsel", ("xsel", "--clipboard", "--output")),
)


def select_clipboard_backend() -> tuple[str, tuple[str, ...]]:
    for name, command in CLIPBOARD_BACKENDS:
        if shutil.which(name):
            return name, command
    raise InputError(
        "No supported Linux clipboard utility was found. Install wl-clipboard, "
        "xclip, or xsel, or save the posting as UTF-8 text and use --job-file PATH."
    )


def read_clipboard(*, timeout_seconds: int = 10) -> tuple[str, str]:
    backend, command = select_clipboard_backend()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(Path.cwd()),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InputError(f"Could not read the clipboard with {backend}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise InputError(
            f"{backend} could not read the clipboard"
            + (f": {detail[:500]}" if detail else ".")
            + " Use --job-file PATH as an alternative."
        )
    text = completed.stdout.strip()
    if not text:
        raise InputError(
            f"The clipboard returned by {backend} is empty. Copy a job description "
            "or use --job-file PATH."
        )
    if len(text.encode("utf-8")) > 500_000:
        raise InputError("Clipboard content exceeds the 500,000-byte safety limit.")
    return text, backend
