from __future__ import annotations

import subprocess

import pytest

from resume_tailor.backend.utils import clipboard
from resume_tailor.backend.jobs.job_text import MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS
from resume_tailor.backend.utils.utilities import InputError


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


def _stub_clipboard(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    monkeypatch.setattr(
        clipboard,
        "select_clipboard_backend",
        lambda: ("wl-paste", ("wl-paste", "--no-newline")),
    )
    monkeypatch.setattr(
        clipboard.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, text, ""),
    )


def test_clipboard_accepts_the_confirmed_character_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "x" * MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS
    _stub_clipboard(monkeypatch, text)

    result, backend = clipboard.read_clipboard()

    assert result == text
    assert backend == "wl-paste"


def test_clipboard_over_limit_reports_actual_and_permitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS + 1
    _stub_clipboard(monkeypatch, "x" * actual)

    with pytest.raises(InputError) as raised:
        clipboard.read_clipboard()

    assert f"{actual:,}" in str(raised.value)
    assert f"{MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS:,}" in str(raised.value)
