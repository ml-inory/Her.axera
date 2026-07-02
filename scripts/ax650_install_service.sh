#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="${SERVICE_NAME:-her-axera-backend}"
SERVICE_PATH="${SERVICE_PATH:-/etc/systemd/system/${SERVICE_NAME}.service}"
PORT="${PORT:-8080}"
ENABLE=0
START=0
PRINT_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/ax650_install_service.sh [options]

Render and install the AX650 backend systemd service.

Options:
  --service-name NAME     systemd service name. Default: her-axera-backend
  --service-path PATH     Destination service file. Default: /etc/systemd/system/<name>.service
  --port PORT             Backend port. Default: 8080
  --enable                Run systemctl enable after install.
  --start                 Run systemctl restart after install.
  --print                 Print rendered service to stdout without installing.
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service-name)
      shift
      [[ $# -gt 0 ]] || { echo "--service-name requires a value" >&2; exit 2; }
      SERVICE_NAME="$1"
      SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
      ;;
    --service-path)
      shift
      [[ $# -gt 0 ]] || { echo "--service-path requires a value" >&2; exit 2; }
      SERVICE_PATH="$1"
      ;;
    --port)
      shift
      [[ $# -gt 0 ]] || { echo "--port requires a value" >&2; exit 2; }
      PORT="$1"
      ;;
    --enable)
      ENABLE=1
      ;;
    --start)
      START=1
      ;;
    --print)
      PRINT_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

TEMPLATE="${REPO_ROOT}/systemd/her-axera-backend.service.in"
[[ -f "${TEMPLATE}" ]] || { echo "Missing template: ${TEMPLATE}" >&2; exit 1; }

render_service() {
  sed \
    -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
    -e "s|@PORT@|${PORT}|g" \
    "${TEMPLATE}"
}

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  render_service
  exit 0
fi

if [[ ! -d "${REPO_ROOT}/backend/.venv" ]]; then
  echo "[check] missing backend/.venv; run scripts/ax650_setup_backend.sh first" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/backend/.env" ]]; then
  echo "[check] missing backend/.env; run scripts/ax650_setup_backend.sh first" >&2
  exit 1
fi

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT
render_service >"${TMP_FILE}"

if [[ -f "${SERVICE_PATH}" ]] && cmp -s "${TMP_FILE}" "${SERVICE_PATH}"; then
  echo "[service] ${SERVICE_PATH} already up to date"
else
  if [[ "${SERVICE_PATH}" == /etc/systemd/system/* ]]; then
    sudo install -m 0644 "${TMP_FILE}" "${SERVICE_PATH}"
  else
    install -m 0644 "${TMP_FILE}" "${SERVICE_PATH}"
  fi
  echo "[service] installed ${SERVICE_PATH}"
fi

sudo systemctl daemon-reload

if [[ "${ENABLE}" -eq 1 ]]; then
  sudo systemctl enable "${SERVICE_NAME}.service"
fi

if [[ "${START}" -eq 1 ]]; then
  sudo systemctl restart "${SERVICE_NAME}.service"
fi

echo "[status] systemctl status ${SERVICE_NAME}.service"
echo "[logs] journalctl -u ${SERVICE_NAME}.service -f"
echo "[start] sudo systemctl restart ${SERVICE_NAME}.service"
echo "[stop] sudo systemctl stop ${SERVICE_NAME}.service"
