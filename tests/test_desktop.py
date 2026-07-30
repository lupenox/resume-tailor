from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.desktop import (
    DesktopInstallError,
    is_managed_desktop_file,
    resolve_desktop_directory,
    write_desktop_launcher,
)
from resume_tailor import ui_cli


def test_chrome_detection_uses_declared_priority() -> None:
    calls: list[str] = []

    def which(name: str) -> str | None:
        calls.append(name)
        return {
            "google-chrome": "/opt/google/chrome/google-chrome",
            "chromium": "/usr/bin/chromium",
        }.get(name)

    assert ui_cli.detect_chrome(which) == "/opt/google/chrome/google-chrome"
    assert calls == ["google-chrome-stable", "google-chrome"]


def test_chrome_launch_preserves_complete_session_url() -> None:
    invocations: list[tuple[list[str], dict[str, Any]]] = []
    url = "http://127.0.0.1:8765/?session=abc%2F123&csrf=a-b_c"

    def popen(arguments: list[str], **kwargs: Any) -> object:
        invocations.append((arguments, kwargs))
        return object()

    selected = ui_cli.open_dashboard(
        url,
        which=lambda name: (
            "/usr/bin/google-chrome-stable"
            if name == "google-chrome-stable"
            else None
        ),
        popen=popen,
        default_open=lambda *_args, **_kwargs: pytest.fail(
            "default browser must not be used"
        ),
    )

    assert selected == "/usr/bin/google-chrome-stable"
    arguments, options = invocations[0]
    assert arguments == [
        "/usr/bin/google-chrome-stable",
        f"--app={url}",
    ]
    assert "shell" not in options


def test_missing_chrome_falls_back_with_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[tuple[str, int]] = []

    def default_open(url: str, *, new: int) -> bool:
        opened.append((url, new))
        return True

    selected = ui_cli.open_dashboard(
        "http://127.0.0.1:8765/",
        which=lambda _name: None,
        popen=lambda *_args, **_kwargs: pytest.fail("Chrome must not launch"),
        default_open=default_open,
    )

    assert selected == "system-default"
    assert opened == [("http://127.0.0.1:8765/", 2)]
    assert "system default browser" in capsys.readouterr().err


def test_desktop_path_prefers_safe_xdg_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    desktop = home / "My KDE Desktop"
    desktop.mkdir(parents=True)
    calls: list[list[str]] = []

    def runner(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, f"{desktop}\n", "")

    assert resolve_desktop_directory(home=home, runner=runner) == desktop.resolve()
    assert calls == [["xdg-user-dir", "DESKTOP"]]


def test_desktop_path_uses_validated_home_fallback(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    def runner(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 0, "/etc\n", "")

    assert resolve_desktop_directory(home=home, runner=runner) == home / "Desktop"


def test_launcher_generation_reuses_icon_and_sets_executable(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "local bin" / "tailor-resume-ui"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o755)
    icon = repository_root / "resume_tailor" / "static" / "favicon.svg"
    destination = tmp_path / "Desktop" / "Resume Tailor.desktop"

    write_desktop_launcher(
        template_path=repository_root / "assets" / "resume-tailor.desktop.in",
        destination=destination,
        launcher=launcher,
        icon=icon,
        executable=True,
    )

    text = destination.read_text(encoding="utf-8")
    assert f'Exec="{launcher}"' in text
    assert f"Icon={icon}" in text
    assert "X-Resume-Tailor-Managed=true" in text
    assert is_managed_desktop_file(destination)
    assert os.access(destination, os.X_OK)


def test_launcher_generation_refuses_unrelated_desktop_file(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "tailor-resume-ui"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o755)
    destination = tmp_path / "Resume Tailor.desktop"
    destination.write_text("[Desktop Entry]\nName=Unrelated\n", encoding="utf-8")

    with pytest.raises(DesktopInstallError, match="unrelated"):
        write_desktop_launcher(
            template_path=repository_root / "assets" / "resume-tailor.desktop.in",
            destination=destination,
            launcher=launcher,
            icon=repository_root / "resume_tailor" / "static" / "favicon.svg",
            executable=True,
            replace_managed=True,
        )

    assert destination.read_text(encoding="utf-8") == (
        "[Desktop Entry]\nName=Unrelated\n"
    )


def test_existing_server_opens_dashboard_without_starting_another(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    url = "http://127.0.0.1:8765/"
    opened: list[str] = []
    monkeypatch.setattr(ui_cli, "_probe_existing_dashboard", lambda _port: url)
    monkeypatch.setattr(
        ui_cli,
        "open_dashboard",
        lambda dashboard_url: opened.append(dashboard_url) or "/usr/bin/chrome",
    )
    monkeypatch.setattr(
        ui_cli,
        "_reserve_listener",
        lambda _port: pytest.fail("a second server must not be reserved"),
    )

    result = ui_cli.main(
        ["--port", "8765", "--output-dir", str(tmp_path / "output")]
    )

    assert result == 0
    assert opened == [url]


def test_bind_race_waits_for_existing_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    url = "http://127.0.0.1:8765/"
    probes = iter((None,))
    opened: list[str] = []
    monkeypatch.setattr(
        ui_cli,
        "_probe_existing_dashboard",
        lambda _port: next(probes),
    )
    monkeypatch.setattr(
        ui_cli,
        "_reserve_listener",
        lambda _port: (_ for _ in ()).throw(OSError("address in use")),
    )
    monkeypatch.setattr(ui_cli, "_wait_for_existing_dashboard", lambda _port: url)
    monkeypatch.setattr(
        ui_cli,
        "open_dashboard",
        lambda dashboard_url: opened.append(dashboard_url) or "/usr/bin/chrome",
    )

    result = ui_cli.main(
        ["--port", "8765", "--output-dir", str(tmp_path / "output")]
    )

    assert result == 0
    assert opened == [url]


def test_keyboard_interrupt_stops_reserved_server_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Listener:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Server:
        def __init__(self, _configuration: object) -> None:
            pass

        def run(self, *, sockets: list[Listener]) -> None:
            assert len(sockets) == 1
            raise KeyboardInterrupt

    listener = Listener()
    monkeypatch.setattr(ui_cli, "_probe_existing_dashboard", lambda _port: None)
    monkeypatch.setattr(ui_cli, "_reserve_listener", lambda _port: listener)
    monkeypatch.setattr(ui_cli, "create_app", lambda **_kwargs: object())
    monkeypatch.setattr(ui_cli.uvicorn, "Server", Server)

    result = ui_cli.main(
        [
            "--no-browser",
            "--port",
            "8765",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result == 0
    assert listener.closed
