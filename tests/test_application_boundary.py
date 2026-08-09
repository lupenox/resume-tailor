from __future__ import annotations

import builtins
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import resume_tailor.application.pipeline as application_pipeline
from resume_tailor.application.models import PipelineRequest, PipelineResult
from resume_tailor.backend.engine.orchestration import PipelineHooks
from resume_tailor.backend.utils.utilities import ApprovalError


def _run_guarded_import(
    *,
    repository_root: Path,
    working_directory: Path,
    blocked: tuple[str, ...],
    target: str,
) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib
        import sys

        blocked = tuple(sys.argv[1].split("|"))

        class BlockedDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == name or fullname.startswith(name + ".")
                    for name in blocked
                ):
                    raise AssertionError(f"blocked dependency imported: {fullname}")
                return None

        for name in tuple(sys.modules):
            if any(name == item or name.startswith(item + ".") for item in blocked):
                del sys.modules[name]
        sys.meta_path.insert(0, BlockedDependency())
        imported = importlib.import_module(sys.argv[2])
        assert imported is not None
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root)
    return subprocess.run(
        [sys.executable, "-c", script, "|".join(blocked), target],
        cwd=working_directory,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_application_imports_without_argparse_or_ui(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    result = _run_guarded_import(
        repository_root=repository_root,
        working_directory=tmp_path,
        blocked=("argparse", "resume_tailor.ui"),
        target="resume_tailor.application.pipeline",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fastapi_adapter_does_not_import_cli_adapter(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    result = _run_guarded_import(
        repository_root=repository_root,
        working_directory=tmp_path,
        blocked=("resume_tailor.ui.cli",),
        target="resume_tailor.ui.ui",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_application_service_returns_typed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "completed-run"
    monkeypatch.setattr(
        application_pipeline,
        "_run_pipeline",
        lambda request, hooks: expected,
    )

    result = application_pipeline.run_pipeline(
        PipelineRequest(resume=tmp_path / "master.docx"),
        hooks=PipelineHooks(),
    )

    assert result == PipelineResult(run_directory=expected)


def test_neutral_hooks_perform_no_terminal_io(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        builtins,
        "input",
        lambda _prompt: pytest.fail("provider-neutral hooks must not read stdin"),
    )
    hooks = PipelineHooks()
    hooks.warning("synthetic warning")
    hooks.present("notice", {"message": "synthetic detail"})
    with pytest.raises(ApprovalError, match="interaction adapter"):
        hooks.approve(
            kind="synthetic",
            title="Synthetic approval",
            payload={},
            assume_yes=False,
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_pipeline_hooks_preserve_positional_cancel_event_contract() -> None:
    cancel_event = threading.Event()

    hooks = PipelineHooks(None, None, False, None, cancel_event)

    assert hooks.cancel_event is cancel_event
    assert hooks.presentation_handler is None
