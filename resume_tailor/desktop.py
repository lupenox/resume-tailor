from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence


MANAGED_MARKER = "X-Resume-Tailor-Managed=true"
DESKTOP_SHORTCUT_NAME = "Resume Tailor.desktop"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class DesktopInstallError(RuntimeError):
    """Raised when a desktop target or launcher is unsafe."""


def _validated_home(home: Path) -> Path:
    try:
        resolved = home.expanduser().resolve(strict=True)
    except OSError as exc:
        raise DesktopInstallError(f"Cannot resolve the home directory: {exc}") from exc
    if resolved == Path("/") or not resolved.is_dir():
        raise DesktopInstallError("The home directory is missing or unsafe.")
    return resolved


def _validated_desktop_candidate(
    raw_path: str,
    *,
    home: Path,
) -> Path | None:
    if (
        not raw_path
        or any(character in raw_path for character in ("\x00", "\r", "\n"))
    ):
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        return None
    try:
        if candidate.exists():
            if not candidate.is_dir():
                return None
            resolved = candidate.resolve(strict=True)
        else:
            resolved = candidate.parent.resolve(strict=True) / candidate.name
        resolved.relative_to(home)
    except (OSError, ValueError):
        return None
    if resolved == home:
        return None
    return resolved


def resolve_desktop_directory(
    *,
    home: Path | None = None,
    runner: CommandRunner = subprocess.run,
) -> Path:
    """Resolve a safe user desktop, preferring xdg-user-dir over ~/Desktop."""

    safe_home = _validated_home(home or Path.home())
    try:
        result = runner(
            ["xdg-user-dir", "DESKTOP"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0:
        output = result.stdout.strip()
        candidate = _validated_desktop_candidate(output, home=safe_home)
        if candidate is not None:
            return candidate

    fallback = _validated_desktop_candidate(
        str(safe_home / "Desktop"),
        home=safe_home,
    )
    if fallback is None:
        raise DesktopInstallError(
            f"The fallback desktop path {safe_home / 'Desktop'} is unsafe."
        )
    return fallback


def is_managed_desktop_file(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        if path.stat().st_size > 64 * 1024:
            return False
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    if MANAGED_MARKER in lines:
        return True
    # Recognize the project's pre-marker application-menu entry during upgrades.
    return (
        "[Desktop Entry]" in lines
        and "Name=resume-tailor" in lines
        and "GenericName=Résumé Tailoring" in lines
        and any(
            line.startswith("Exec=") and "tailor-resume-ui" in line
            for line in lines
        )
    )


def _desktop_exec_value(path: Path) -> str:
    value = str(path)
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise DesktopInstallError("The launcher path contains unsafe characters.")
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
    )
    return f'"{escaped}"'


def _desktop_string_value(path: Path) -> str:
    value = str(path)
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise DesktopInstallError("The icon path contains unsafe characters.")
    return value.replace("\\", "\\\\")


def render_desktop_launcher(
    *,
    template: str,
    launcher: Path,
    icon: Path,
) -> str:
    if template.count("@UI_LAUNCHER@") != 1 or template.count("@ICON_PATH@") != 1:
        raise DesktopInstallError(
            "The desktop template must contain one launcher and one icon placeholder."
        )
    return (
        template.replace("@UI_LAUNCHER@", _desktop_exec_value(launcher))
        .replace("@ICON_PATH@", _desktop_string_value(icon))
        .rstrip()
        + "\n"
    )


def write_desktop_launcher(
    *,
    template_path: Path,
    destination: Path,
    launcher: Path,
    icon: Path,
    executable: bool,
    replace_managed: bool = False,
) -> None:
    destination_existed = destination.exists()
    if destination.is_symlink():
        raise DesktopInstallError(
            f"Refusing to replace symbolic-link desktop target: {destination}"
        )
    if destination.exists():
        if not replace_managed:
            raise DesktopInstallError(
                f"Refusing to overwrite existing desktop file: {destination}"
            )
        if not is_managed_desktop_file(destination):
            raise DesktopInstallError(
                f"Refusing to overwrite unrelated desktop file: {destination}"
            )
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise DesktopInstallError(f"The UI launcher is missing or not executable: {launcher}")
    if not icon.is_file() or icon.is_symlink():
        raise DesktopInstallError(f"The application icon is missing or unsafe: {icon}")

    temporary_path: Path | None = None
    try:
        template = template_path.read_text(encoding="utf-8")
        rendered = render_desktop_launcher(
            template=template,
            launcher=launcher,
            icon=icon,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o755 if executable else 0o644)
        if destination_existed:
            if not is_managed_desktop_file(destination):
                raise DesktopInstallError(
                    f"Refusing to overwrite unrelated desktop file: {destination}"
                )
            os.replace(temporary_path, destination)
        else:
            os.link(temporary_path, destination)
    except OSError as exc:
        raise DesktopInstallError(
            f"Could not install desktop launcher {destination}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resume-tailor-desktop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--home", type=Path, required=True)

    managed = subparsers.add_parser("is-managed")
    managed.add_argument("path", type=Path)

    write = subparsers.add_parser("write")
    write.add_argument("--template", type=Path, required=True)
    write.add_argument("--destination", type=Path, required=True)
    write.add_argument("--launcher", type=Path, required=True)
    write.add_argument("--icon", type=Path, required=True)
    write.add_argument("--executable", action="store_true")
    write.add_argument("--replace-managed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "resolve":
            print(resolve_desktop_directory(home=args.home))
        elif args.command == "is-managed":
            return 0 if is_managed_desktop_file(args.path) else 1
        else:
            write_desktop_launcher(
                template_path=args.template,
                destination=args.destination,
                launcher=args.launcher,
                icon=args.icon,
                executable=args.executable,
                replace_managed=args.replace_managed,
            )
    except DesktopInstallError as exc:
        print(f"resume-tailor desktop: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
