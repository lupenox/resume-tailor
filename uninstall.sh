#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HOME:-}" || "${HOME}" == "/" ]]; then
  echo "uninstall.sh: HOME is unset or unsafe; refusing to continue." >&2
  exit 1
fi

APP_DIR="${HOME}/.local/share/resume-tailor"
LAUNCHER="${HOME}/.local/bin/tailor-resume"
UI_LAUNCHER="${HOME}/.local/bin/tailor-resume-ui"
DESKTOP_FILE="${HOME}/.local/share/applications/resume-tailor.desktop"
DESKTOP_SHORTCUT=""
DESKTOP_HELPER="${APP_DIR}/resume_tailor/desktop.py"
if [[ -f "${DESKTOP_HELPER}" ]]; then
  if DESKTOP_DIRECTORY="$(
    python3 "${DESKTOP_HELPER}" resolve --home "${HOME}"
  )"; then
    DESKTOP_SHORTCUT="${DESKTOP_DIRECTORY}/Resume Tailor.desktop"
  fi
fi
SHORTCUT_MANAGED=0
if [[ -n "${DESKTOP_SHORTCUT}" && -e "${DESKTOP_SHORTCUT}" ]]; then
  if python3 "${DESKTOP_HELPER}" is-managed "${DESKTOP_SHORTCUT}"; then
    SHORTCUT_MANAGED=1
  else
    echo "Leaving unrelated desktop file untouched: ${DESKTOP_SHORTCUT}" >&2
  fi
fi

if [[ ! -e "${APP_DIR}" \
  && ! -e "${LAUNCHER}" \
  && ! -e "${UI_LAUNCHER}" \
  && ! -e "${DESKTOP_FILE}" \
  && "${SHORTCUT_MANAGED}" -ne 1 ]]; then
  echo "resume-tailor is not installed at the expected locations."
  exit 0
fi

echo "The following paths will be removed:"
[[ -e "${APP_DIR}" ]] && echo "  ${APP_DIR}"
[[ -e "${LAUNCHER}" ]] && echo "  ${LAUNCHER}"
[[ -e "${UI_LAUNCHER}" ]] && echo "  ${UI_LAUNCHER}"
[[ -e "${DESKTOP_FILE}" ]] && echo "  ${DESKTOP_FILE}"
[[ "${SHORTCUT_MANAGED}" -eq 1 ]] && echo "  ${DESKTOP_SHORTCUT}"
echo
echo "Timestamped backups and generated resume output directories are not removed."
read -r -p 'Type "remove" to confirm: ' CONFIRMATION
if [[ "${CONFIRMATION}" != "remove" ]]; then
  echo "Uninstall cancelled."
  exit 1
fi

EXPECTED_APP="${HOME}/.local/share/resume-tailor"
if [[ "${APP_DIR}" != "${EXPECTED_APP}" || "${APP_DIR}" == "/" ]]; then
  echo "uninstall.sh: resolved application target is unsafe." >&2
  exit 1
fi

if [[ -e "${APP_DIR}" ]]; then
  rm -rf -- "${APP_DIR}"
fi
if [[ -e "${LAUNCHER}" ]]; then
  rm -f -- "${LAUNCHER}"
fi
if [[ -e "${UI_LAUNCHER}" ]]; then
  rm -f -- "${UI_LAUNCHER}"
fi
if [[ -e "${DESKTOP_FILE}" ]]; then
  rm -f -- "${DESKTOP_FILE}"
fi
if [[ "${SHORTCUT_MANAGED}" -eq 1 && -e "${DESKTOP_SHORTCUT}" ]]; then
  rm -f -- "${DESKTOP_SHORTCUT}"
fi
echo "resume-tailor was removed."
