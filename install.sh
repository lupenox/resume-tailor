#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./install.sh [--force] [--desktop]

Copies resume-tailor to ~/.local/share/resume-tailor and installs the CLI and
localhost-UI launchers under ~/.local/bin. With --desktop, it also installs a
KDE/Linux application-menu entry and a literal executable desktop shortcut. It
does not install dependencies or modify shell or desktop configuration. When
the source checkout has a .venv, the installed application reuses it through a
symlink.
EOF
}

FORCE=0
DESKTOP=0
while (($#)); do
  case "$1" in
    --force)
      FORCE=1
      ;;
    --desktop)
      DESKTOP=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "install.sh: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -z "${HOME:-}" || "${HOME}" == "/" ]]; then
  echo "install.sh: HOME is unset or unsafe; refusing to install." >&2
  exit 1
fi

SOURCE_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${HOME}/.local/share"
APP_DIR="${DATA_ROOT}/resume-tailor"
BIN_DIR="${HOME}/.local/bin"
LAUNCHER="${BIN_DIR}/tailor-resume"
UI_LAUNCHER="${BIN_DIR}/tailor-resume-ui"
APPLICATIONS_DIR="${DATA_ROOT}/applications"
DESKTOP_FILE="${APPLICATIONS_DIR}/resume-tailor.desktop"
DESKTOP_DIRECTORY=""
DESKTOP_SHORTCUT=""
if [[ "${DESKTOP}" -eq 1 ]]; then
  if ! DESKTOP_DIRECTORY="$(
    python3 "${SOURCE_DIR}/resume_tailor/ui/desktop.py" resolve --home "${HOME}"
  )"; then
    echo "install.sh: could not resolve a safe desktop directory." >&2
    exit 1
  fi
  DESKTOP_SHORTCUT="${DESKTOP_DIRECTORY}/Resume Tailor.desktop"
fi

if [[ -e "${APP_DIR}" && "${FORCE}" -ne 1 ]]; then
  echo "install.sh: ${APP_DIR} already exists." >&2
  echo "Re-run with --force to replace it while keeping a timestamped backup." >&2
  exit 1
fi
if [[ -e "${LAUNCHER}" && "${FORCE}" -ne 1 ]]; then
  echo "install.sh: ${LAUNCHER} already exists." >&2
  echo "Re-run with --force to replace it while keeping a timestamped backup." >&2
  exit 1
fi
if [[ -e "${UI_LAUNCHER}" && "${FORCE}" -ne 1 ]]; then
  echo "install.sh: ${UI_LAUNCHER} already exists." >&2
  echo "Re-run with --force to replace it while keeping a timestamped backup." >&2
  exit 1
fi
if [[ "${DESKTOP}" -eq 1 \
  && ( -e "${DESKTOP_FILE}" || -L "${DESKTOP_FILE}" ) \
  && "${FORCE}" -ne 1 ]]; then
  echo "install.sh: ${DESKTOP_FILE} already exists." >&2
  echo "Re-run with --force to replace it while keeping a timestamped backup." >&2
  exit 1
fi
if [[ "${DESKTOP}" -eq 1 \
  && ( -e "${DESKTOP_SHORTCUT}" || -L "${DESKTOP_SHORTCUT}" ) \
  && "${FORCE}" -ne 1 ]]; then
  echo "install.sh: ${DESKTOP_SHORTCUT} already exists." >&2
  echo "Refusing to overwrite a desktop file without --force." >&2
  exit 1
fi
if [[ "${DESKTOP}" -eq 1 \
  && ( -e "${DESKTOP_FILE}" || -L "${DESKTOP_FILE}" ) \
  && "${FORCE}" -eq 1 ]]; then
  if ! python3 "${SOURCE_DIR}/resume_tailor/ui/desktop.py" \
    is-managed "${DESKTOP_FILE}"; then
    echo "install.sh: refusing to overwrite unrelated file: ${DESKTOP_FILE}" >&2
    exit 1
  fi
fi
if [[ "${DESKTOP}" -eq 1 \
  && ( -e "${DESKTOP_SHORTCUT}" || -L "${DESKTOP_SHORTCUT}" ) \
  && "${FORCE}" -eq 1 ]]; then
  if ! python3 "${SOURCE_DIR}/resume_tailor/ui/desktop.py" \
    is-managed "${DESKTOP_SHORTCUT}"; then
    echo "install.sh: refusing to overwrite unrelated file: ${DESKTOP_SHORTCUT}" >&2
    exit 1
  fi
fi

mkdir -p "${DATA_ROOT}" "${BIN_DIR}"
STAGING="$(mktemp -d "${DATA_ROOT}/.resume-tailor.install.XXXXXX")"
cleanup() {
  if [[ -n "${STAGING:-}" && -d "${STAGING}" ]]; then
    rm -rf -- "${STAGING}"
  fi
}
trap cleanup EXIT HUP INT TERM

mkdir -p "${STAGING}/resume_tailor" "${STAGING}/schemas" "${STAGING}/template"
cp -a "${SOURCE_DIR}/resume_tailor/"*.py "${STAGING}/resume_tailor/" 2>/dev/null || true
cp -a "${SOURCE_DIR}/resume_tailor/backend" "${STAGING}/resume_tailor/backend"
cp -a "${SOURCE_DIR}/resume_tailor/ui" "${STAGING}/resume_tailor/ui"
cp -a "${SOURCE_DIR}/resume_tailor/static" "${STAGING}/resume_tailor/static"
cp -a "${SOURCE_DIR}/resume_tailor/templates" "${STAGING}/resume_tailor/templates"
cp -a "${SOURCE_DIR}/schemas/"*.json "${STAGING}/schemas/"
cp -a "${SOURCE_DIR}/template/master_resume.docx" "${STAGING}/template/master_resume.docx"
cp -a \
  "${SOURCE_DIR}/pyproject.toml" \
  "${SOURCE_DIR}/README.md" \
  "${SOURCE_DIR}/LICENSE" \
  "${SOURCE_DIR}/tailor-resume" \
  "${SOURCE_DIR}/tailor-resume-ui" \
  "${SOURCE_DIR}/uninstall.sh" \
  "${STAGING}/"
chmod 755 \
  "${STAGING}/tailor-resume" \
  "${STAGING}/tailor-resume-ui" \
  "${STAGING}/uninstall.sh"
if [[ -x "${SOURCE_DIR}/.venv/bin/python" ]]; then
  ln -s -- "${SOURCE_DIR}/.venv" "${STAGING}/.venv"
fi

BACKUP_SUFFIX="$(date +%Y%m%d-%H%M%S)-$$"
if [[ -e "${APP_DIR}" ]]; then
  mv -- "${APP_DIR}" "${APP_DIR}.backup-${BACKUP_SUFFIX}"
  echo "Backed up the previous application to ${APP_DIR}.backup-${BACKUP_SUFFIX}"
fi
if [[ -e "${LAUNCHER}" ]]; then
  mv -- "${LAUNCHER}" "${LAUNCHER}.backup-${BACKUP_SUFFIX}"
  echo "Backed up the previous launcher to ${LAUNCHER}.backup-${BACKUP_SUFFIX}"
fi
if [[ -e "${UI_LAUNCHER}" ]]; then
  mv -- "${UI_LAUNCHER}" "${UI_LAUNCHER}.backup-${BACKUP_SUFFIX}"
  echo "Backed up the previous UI launcher to ${UI_LAUNCHER}.backup-${BACKUP_SUFFIX}"
fi
if [[ "${DESKTOP}" -eq 1 && -e "${DESKTOP_FILE}" ]]; then
  mv -- "${DESKTOP_FILE}" "${DESKTOP_FILE}.backup-${BACKUP_SUFFIX}"
  echo "Backed up the previous desktop entry to ${DESKTOP_FILE}.backup-${BACKUP_SUFFIX}"
fi
if [[ "${DESKTOP}" -eq 1 && -e "${DESKTOP_SHORTCUT}" ]]; then
  mv -- \
    "${DESKTOP_SHORTCUT}" \
    "${DESKTOP_SHORTCUT}.backup-${BACKUP_SUFFIX}"
  echo "Backed up the previous desktop shortcut to ${DESKTOP_SHORTCUT}.backup-${BACKUP_SUFFIX}"
fi

mv -- "${STAGING}" "${APP_DIR}"
STAGING=""
install -m 755 "${APP_DIR}/tailor-resume" "${LAUNCHER}"
install -m 755 "${APP_DIR}/tailor-resume-ui" "${UI_LAUNCHER}"

echo "Installed application: ${APP_DIR}"
echo "Installed CLI:         ${LAUNCHER}"
echo "Installed local UI:    ${UI_LAUNCHER}"
if [[ -L "${APP_DIR}/.venv" ]]; then
  echo "Runtime environment:   ${APP_DIR}/.venv -> ${SOURCE_DIR}/.venv"
fi
if [[ "${DESKTOP}" -eq 1 ]]; then
  mkdir -p "${APPLICATIONS_DIR}" "${DESKTOP_DIRECTORY}"
  APP_ICON="${APP_DIR}/resume_tailor/static/favicon.svg"
  python3 "${SOURCE_DIR}/resume_tailor/ui/desktop.py" write \
    --template "${SOURCE_DIR}/assets/resume-tailor.desktop.in" \
    --destination "${DESKTOP_FILE}" \
    --launcher "${UI_LAUNCHER}" \
    --icon "${APP_ICON}"
  python3 "${SOURCE_DIR}/resume_tailor/ui/desktop.py" write \
    --template "${SOURCE_DIR}/assets/resume-tailor.desktop.in" \
    --destination "${DESKTOP_SHORTCUT}" \
    --launcher "${UI_LAUNCHER}" \
    --icon "${APP_ICON}" \
    --executable
  echo "Installed desktop app: ${DESKTOP_FILE}"
  echo "Installed desktop icon: ${DESKTOP_SHORTCUT}"
else
  echo "Desktop launcher:      not created (use --desktop to opt in)"
fi
if ! "${APP_DIR}/tailor-resume" --version >/dev/null 2>&1; then
  echo
  echo "Python dependencies are not yet available to the launcher."
  echo "Follow the README dependency setup; the installer intentionally installs none."
fi

case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *)
    echo
    echo "Note: ${BIN_DIR} is not in PATH."
    echo "Add it manually for your shell, or invoke ${LAUNCHER} directly."
    echo "No shell configuration was modified."
    ;;
esac
