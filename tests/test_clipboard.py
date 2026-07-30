from __future__ import annotations

import subprocess

import pytest

from resume_tailor import clipboard
from resume_tailor.utilities import InputError


def test_clipboard_backend_selection_order(monkeypatch: pytest.MonkeyPatch) -> None:
    available = {"xclip", "xsel"}
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in available else None,
    )
    name, command = clipboard.select_clipboard_backend()
    assert name == "xclip"
    assert command == ("xclip", "-selection", "clipboard", "-o")


def test_empty_clipboard_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clipboard,
        "select_clipboard_backend",
        lambda: ("wl-paste", ("wl-paste", "--no-newline")),
    )
    monkeypatch.setattr(
        clipboard.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )
    with pytest.raises(InputError, match="empty"):
        clipboard.read_clipboard()


def test_no_clipboard_utility_explains_job_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: None)
    with pytest.raises(InputError, match="--job-file"):
        clipboard.select_clipboard_backend()
