#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_URL="${BACKEND_URL:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: scripts/pc_run_frontend.sh --backend-url http://AX650_IP:8080 [options]

Serve the static PC frontend and preconfigure it to connect to the AX650 backend.

Options:
  --backend-url URL   AX650 backend URL, for example http://192.168.1.50:8080
  --host HOST         Frontend bind host. Default: 0.0.0.0
  --port PORT         Frontend port. Default: 7860
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-url)
      shift
      [[ $# -gt 0 ]] || { echo "--backend-url requires a value" >&2; exit 2; }
      BACKEND_URL="${1%/}"
      ;;
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

[[ -n "${BACKEND_URL}" ]] || { usage >&2; exit 2; }

cd "${REPO_ROOT}/frontend"

if "${PYTHON_BIN}" - "${BACKEND_URL}/health" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    raise SystemExit(0 if payload.get("status") == "ok" else 1)
except Exception:
    raise SystemExit(1)
PY
then
  echo "[backend] healthy: ${BACKEND_URL}/health"
else
  echo "[backend] warning: health check failed for ${BACKEND_URL}/health" >&2
fi

QUERY_API="$("${PYTHON_BIN}" - "${BACKEND_URL}" <<'PY'
import sys
import urllib.parse
print(urllib.parse.quote(sys.argv[1], safe=""))
PY
)"

echo "[frontend] serving http://${HOST}:${PORT}"
echo "[open] http://127.0.0.1:${PORT}/?api=${QUERY_API}"
exec "${PYTHON_BIN}" -m http.server "${PORT}" --directory static --bind "${HOST}"
