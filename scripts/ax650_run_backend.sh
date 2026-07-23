#!/usr/bin/env bash
set -euo pipefail

# Error handler: print last command and line number on failure
error_handler() {
    echo "[ERROR] Failed at line $1: $BASH_COMMAND" >&2
}
trap 'error_handler ${LINENO}' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/backend/.venv}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/backend/.env}"
MODEL_ENV_FILE="${MODEL_ENV_FILE:-${REPO_ROOT}/backend/.env.models}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/backend/data/her-axera-backend.log}"
PID_FILE="${PID_FILE:-${REPO_ROOT}/backend/data/her-axera-backend.pid}"
FOREGROUND=0

usage() {
  cat <<'EOF'
Usage: scripts/ax650_run_backend.sh [options]

Start the Her.axera backend on an AX650 board and run a health check.

Options:
  --host HOST              Bind host. Default: 0.0.0.0
  --port PORT              Bind port. Default: 8080
  --env-file PATH          Backend env file. Default: backend/.env
  --model-env-file PATH    Optional generated model env file. Default: backend/.env.models
  --foreground             Run uvicorn in the foreground.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      shift
      [[ $# -gt 0 ]] || { echo "--host requires a value" >&2; exit 2; }
      HOST="$1"
      ;;
    --port)
      shift
      [[ $# -gt 0 ]] || { echo "--port requires a value" >&2; exit 2; }
      PORT="$1"
      ;;
    --env-file)
      shift
      [[ $# -gt 0 ]] || { echo "--env-file requires a value" >&2; exit 2; }
      ENV_FILE="$1"
      ;;
    --model-env-file)
      shift
      [[ $# -gt 0 ]] || { echo "--model-env-file requires a value" >&2; exit 2; }
      MODEL_ENV_FILE="$1"
      ;;
    --foreground)
      FOREGROUND=1
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

cd "${REPO_ROOT}/backend"
mkdir -p data

if [[ -d "${VENV_DIR}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  cp .env.example "${ENV_FILE}"
  echo "[env] created ${ENV_FILE} from backend/.env.example"
fi

set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
if [[ -f "${MODEL_ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${MODEL_ENV_FILE}"
fi
set +a

health_check() {
  local url="http://127.0.0.1:${PORT}/health"
  python - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=2) as response:
    payload = json.loads(response.read().decode("utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(f"unexpected health response: {payload!r}")
PY
}

if [[ "${FOREGROUND}" -eq 1 ]]; then
  echo "[run] starting backend in foreground on ${HOST}:${PORT}"
  exec uvicorn app.main:app --host "${HOST}" --port "${PORT}"
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "[run] backend already running with pid $(cat "${PID_FILE}")"
else
  echo "[run] starting backend on ${HOST}:${PORT}"
  nohup uvicorn app.main:app --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
  echo "$!" >"${PID_FILE}"
fi

for _ in $(seq 1 30); do
  if health_check >/dev/null 2>&1; then
    echo "[health] ok: http://127.0.0.1:${PORT}/health"
    echo "[ui] open from PC: http://<AX650_IP>:${PORT}/ui/"
    echo "[log] ${LOG_FILE}"
    exit 0
  fi
  sleep 1
done

echo "[health] backend did not become healthy within 30s" >&2
echo "[log] ${LOG_FILE}" >&2
tail -80 "${LOG_FILE}" >&2 || true
exit 1
