from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from resume_tailor.backend.utils.utilities import (
    CancellationError,
    ModelError,
    cancellable_commands,
    run_command,
)


def test_cancellation_stops_an_active_subprocess(tmp_path: Path) -> None:
    cancel_event = threading.Event()
    result: list[BaseException] = []

    def invoke() -> None:
        try:
            with cancellable_commands(cancel_event):
                run_command(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    cwd=tmp_path,
                    timeout_seconds=60,
                )
        except BaseException as exc:
            result.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    time.sleep(0.2)
    cancel_event.set()
    worker.join(timeout=4)

    assert not worker.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], CancellationError)


def test_subprocess_liveness_heartbeats_distinguish_running_and_exit(
    tmp_path: Path,
) -> None:
    heartbeats: list[tuple[float, bool]] = []

    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(0.18)"],
        cwd=tmp_path,
        timeout_seconds=2,
        heartbeat_handler=lambda elapsed, alive: heartbeats.append((elapsed, alive)),
        heartbeat_interval_seconds=0.04,
    )

    assert result.returncode == 0
    assert heartbeats[0] == (0.0, True)
    assert any(alive and elapsed > 0 for elapsed, alive in heartbeats)
    assert heartbeats[-1][1] is False
    assert heartbeats[-1][0] >= 0.15


def test_bounded_timeout_stops_process_group_with_actionable_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelError, match="full subprocess group was stopped"):
        run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            timeout_seconds=1,
        )


def test_subprocesses_never_inherit_github_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_synthetic_secret_value")

    inherited = run_command(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('GITHUB_TOKEN', 'absent'))",
        ],
        cwd=tmp_path,
        timeout_seconds=2,
    )
    explicitly_supplied = run_command(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('GITHUB_TOKEN', 'absent'))",
        ],
        cwd=tmp_path,
        timeout_seconds=2,
        env={**os.environ, "GITHUB_TOKEN": "github_pat_other_secret_value"},
    )
    alias_supplied = run_command(
        [
            sys.executable,
            "-c",
            (
                "import os; print(','.join(sorted(k for k in os.environ "
                "if k.casefold() in {'github_token','gh_token',"
                "'github_enterprise_token','gh_enterprise_token'})) or 'absent')"
            ),
        ],
        cwd=tmp_path,
        timeout_seconds=2,
        env={
            **os.environ,
            "github_token": "synthetic-lowercase-secret",
            "GH_TOKEN": "synthetic-gh-secret",
            "GITHUB_ENTERPRISE_TOKEN": "synthetic-enterprise-secret",
            "gh_enterprise_token": "synthetic-lowercase-enterprise-secret",
        },
    )

    assert inherited.stdout.strip() == "absent"
    assert explicitly_supplied.stdout.strip() == "absent"
    assert alias_supplied.stdout.strip() == "absent"
