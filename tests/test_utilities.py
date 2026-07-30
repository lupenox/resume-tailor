from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from resume_tailor.utilities import (
    CancellationError,
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
