from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable

import uvicorn

from resume_tailor import __version__
from resume_tailor.backend.utils.analytics import default_analytics_database_path
from resume_tailor.ui.ui import DEFAULT_HOST, DEFAULT_PORT, create_app, default_output_directory
from resume_tailor.backend.utils.utilities import parse_duration


CHROME_CANDIDATES = (
    "google-chrome-stable",
    "google-chrome",
    "chromium",
)


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1024 and 65535")
    return port


def _duration(value: str) -> tuple[int, str]:
    try:
        return parse_duration(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tailor-resume-ui",
        description=(
            "Launch the local, human-gated resume-tailor web interface. "
            "The server always binds to 127.0.0.1."
        ),
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"localhost TCP port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_directory(),
        help="local run-artifact parent directory",
    )
    parser.add_argument(
        "--analytics-db",
        type=Path,
        default=default_analytics_database_path(),
        help="private local SQLite analytics database (default: XDG application data)",
    )
    parser.add_argument(
        "--timeout",
        type=_duration,
        default=_duration("15m"),
        metavar="DURATION",
        help="model timeout such as 90s or 15m (default: 15m)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start the localhost server without opening a browser",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def detect_chrome(
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    for executable in CHROME_CANDIDATES:
        resolved = which(executable)
        if resolved:
            return resolved
    return None


def open_dashboard(
    url: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    default_open: Callable[..., bool] = webbrowser.open,
) -> str | None:
    """Open an opaque dashboard URL without changing browser preferences."""

    chrome = detect_chrome(which)
    if chrome is not None:
        try:
            popen(
                [chrome, f"--app={url}"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as exc:
            print(
                f"Could not start {chrome}: {exc}. "
                "Falling back to the system default browser.",
                file=sys.stderr,
            )
        else:
            print(f"Opened Resume Tailor with {chrome}.")
            return chrome
    else:
        print(
            "Google Chrome/Chromium was not found "
            "(tried google-chrome-stable, google-chrome, and chromium). "
            "Opening the system default browser instead.",
            file=sys.stderr,
        )

    try:
        opened = default_open(url, new=2)
    except webbrowser.Error as exc:
        print(
            f"Could not open the system default browser: {exc}. "
            f"Open {url} manually.",
            file=sys.stderr,
        )
        return None
    if not opened:
        print(
            f"The system default browser did not accept the request. Open {url} manually.",
            file=sys.stderr,
        )
        return None
    return "system-default"


def _dashboard_url(port: int) -> str:
    return f"http://{DEFAULT_HOST}:{port}/"


def _probe_existing_dashboard(port: int, *, timeout: float = 0.3) -> str | None:
    request = urllib.request.Request(
        f"http://{DEFAULT_HOST}:{port}/health",
        headers={
            "Accept": "application/json",
            "User-Agent": f"resume-tailor-ui/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read(8193))
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        return None
    if (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("application") == "resume-tailor"
        and payload.get("bind_host") == DEFAULT_HOST
    ):
        return _dashboard_url(port)
    return None


def _wait_for_existing_dashboard(port: int) -> str | None:
    for _ in range(50):
        existing = _probe_existing_dashboard(port)
        if existing is not None:
            return existing
        time.sleep(0.1)
    return None


def _open_browser_when_ready(url: str, port: int) -> None:
    for _ in range(50):
        if _probe_existing_dashboard(port) is not None:
            open_dashboard(url)
            return
        time.sleep(0.1)
    print(
        f"Resume Tailor did not become ready. Open {url} manually after checking "
        "the server output.",
        file=sys.stderr,
    )


def _reserve_listener(port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((DEFAULT_HOST, port))
        listener.listen(128)
    except OSError:
        listener.close()
        raise
    return listener


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    url = _dashboard_url(args.port)
    existing = _probe_existing_dashboard(args.port)
    if existing is not None:
        print(f"Resume Tailor is already running at {existing}")
        if not args.no_browser:
            open_dashboard(existing)
        return 0

    try:
        listener = _reserve_listener(args.port)
    except OSError:
        existing = _wait_for_existing_dashboard(args.port)
        if existing is not None:
            print(f"Resume Tailor is already running at {existing}")
            if not args.no_browser:
                open_dashboard(existing)
            return 0
        print(
            f"tailor-resume-ui: localhost port {args.port} is already occupied "
            "by another process. Choose a different --port.",
            file=sys.stderr,
        )
        return 1

    app = create_app(
        output_directory=args.output_dir,
        analytics_database_path=args.analytics_db,
        timeout=args.timeout,
        port=args.port,
    )
    if not args.no_browser:
        opener = threading.Thread(
            target=_open_browser_when_ready,
            args=(url, args.port),
            name="resume-tailor-browser-opener",
            daemon=True,
        )
        opener.start()
    print(f"resume-tailor UI: {url}")
    print("Press Ctrl+C to stop the localhost server.")
    configuration = uvicorn.Config(
        app,
        host=DEFAULT_HOST,
        port=args.port,
        reload=False,
        access_log=True,
    )
    server = uvicorn.Server(configuration)
    try:
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            pass
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
