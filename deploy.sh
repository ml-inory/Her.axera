#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${REPO_ROOT}/backend/.venv"
MODEL_ROOT="${MODEL_ROOT:-/opt/models/her-axera}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
AXENGINE_WHEEL_URL="${AXENGINE_WHEEL_URL:-https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc2/axengine-0.1.3-py3-none-any.whl}"
STATE_FILE="${REPO_ROOT}/.deploy-state"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
VENV_SYSTEM_SITE_PACKAGES="${VENV_SYSTEM_SITE_PACKAGES:-0}"

MODELS=()
WITH_WENET=0
WITH_FIRERED=0
SKIP_MODELS=0
SKIP_SERVICE=0
FOREGROUND=0
FORCE=0

usage() {
  cat <<'EOF'
Usage: ./deploy.sh [options]

一键部署 Her.axera 到 AX650 板端。仅在首次或 --force 时执行耗时步骤（venv 创建、
pip 安装、模型下载），后续运行秒级完成。

Options:
  --models "sensevoice kokoro speaker"   下载指定模型。
  --models all                           下载所有模型。
  --model-root PATH                      模型根目录。默认: /opt/models/her-axera
  --hf-endpoint URL                      Hugging Face 镜像。默认: https://hf-mirror.com
  --host HOST                            绑定地址。默认: 0.0.0.0
  --port PORT                            绑定端口。默认: 8080
  --with-wenet-onnx                      安装 WeNet ONNX 依赖。
  --with-fireredasr-aed                  安装 FireRedASR-AED 依赖。
  --system-site-packages                 venv 暴露板端系统 site-packages。
  --no-models                            跳过模型下载。
  --no-service                           跳过 systemd 服务安装。
  --foreground                           前台运行（不后台化，不装 service）。
  --force                                强制重新执行所有步骤。
  -h, --help                             显示帮助。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift; [[ $# -gt 0 ]] || { echo "--models requires a value" >&2; exit 2; }
      read -r -a MODELS <<<"$1" ;;
    --model-root)    shift; MODEL_ROOT="$1" ;;
    --hf-endpoint)   shift; HF_ENDPOINT="$1" ;;
    --host)          shift; HOST="$1" ;;
    --port)          shift; PORT="$1" ;;
    --with-wenet-onnx)       WITH_WENET=1 ;;
    --with-fireredasr-aed)   WITH_FIRERED=1 ;;
    --system-site-packages)  VENV_SYSTEM_SITE_PACKAGES=1 ;;
    --no-models)     SKIP_MODELS=1 ;;
    --no-service)    SKIP_SERVICE=1 ;;
    --foreground)    FOREGROUND=1 ;;
    --force)         FORCE=1 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log()  { echo "[deploy] $*"; }
skip() { echo "[deploy] ⏭  $* (already done, use --force to redo)"; }
done_() { echo "[deploy] ✓  $*"; }

# ── 0. Preflight ──────────────────────────────────────────────

cd "${REPO_ROOT}"

if [[ ! -f backend/requirements.txt ]]; then
  echo "[deploy] ✗  Run from Her.axera repo root." >&2; exit 1
fi

if [[ "${FORCE}" -eq 1 ]]; then
  rm -f "${STATE_FILE}"
  log "force mode: clearing state file"
fi

touch "${STATE_FILE}"
source_state() { grep -q "^$1=" "${STATE_FILE}" 2>/dev/null; }
mark_state()  { echo "$1=$(date -Iseconds)" >> "${STATE_FILE}"; }

# hash requirements files to detect changes
REQ_HASH="$(cat backend/requirements.txt backend/requirements-model-download.txt 2>/dev/null | sha256sum | cut -d' ' -f1)"

# ── 1. .env ───────────────────────────────────────────────────

if [[ -f backend/.env ]]; then
  skip "backend/.env"
else
  cp backend/.env.example backend/.env
  done_ "backend/.env created from .env.example"
fi

# ── 2. venv + pip ─────────────────────────────────────────────

NEED_PIP=0

if [[ -d "${VENV_DIR}" ]]; then
  skip "venv"
  if source_state "req_hash" && [[ "$(grep '^req_hash=' "${STATE_FILE}" | cut -d= -f2)" == "${REQ_HASH}" ]]; then
    skip "pip install (requirements unchanged)"
  else
    NEED_PIP=1
  fi
else
  venv_args=()
  [[ "${VENV_SYSTEM_SITE_PACKAGES}" -eq 1 ]] && venv_args+=(--system-site-packages)
  "${PYTHON_BIN}" -m venv "${venv_args[@]}" "${VENV_DIR}"
  done_ "venv created"
  NEED_PIP=1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if [[ "${NEED_PIP}" -eq 1 ]]; then
  log "pip install..."
  python -m pip install --upgrade pip -q
  python -m pip install -r backend/requirements.txt -q
  python -m pip install -r backend/requirements-model-download.txt -q
  done_ "pip install complete"
  sed -i '/^req_hash=/d' "${STATE_FILE}"
  mark_state "req_hash=${REQ_HASH}"
fi

# ── 3. axengine ───────────────────────────────────────────────

if [[ -e /soc/lib/libax_engine.so ]]; then
  if source_state "axengine_linked"; then
    skip "axengine linkage"
  elif LD_LIBRARY_PATH="/soc/lib:${LD_LIBRARY_PATH:-}" python -c "import axengine" >/dev/null 2>&1; then
    done_ "axengine available"
  else
    SYSTEM_PYTHON_BIN="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
    if axengine_paths="$(LD_LIBRARY_PATH="/soc/lib:${LD_LIBRARY_PATH:-}" "${SYSTEM_PYTHON_BIN}" - <<'PY' 2>/dev/null
import importlib.metadata as metadata, importlib.util
from pathlib import Path
spec = importlib.util.find_spec("axengine")
if spec is None or spec.origin is None: raise SystemExit(1)
print(Path(spec.origin).parent)
print(Path(metadata.distribution("axengine")._path))
PY
)"; then
      venv_site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
      axengine_pkg="$(echo "${axengine_paths}" | sed -n '1p')"
      axengine_dist="$(echo "${axengine_paths}" | sed -n '2p')"
      ln -sfn "${axengine_pkg}" "${venv_site}/axengine"
      ln -sfn "${axengine_dist}" "${venv_site}/$(basename "${axengine_dist}")"
      done_ "axengine linked from ${axengine_pkg}"
      mark_state "axengine_linked"
    else
      log "installing pyaxengine wheel..."
      python -m pip install "axengine @ ${AXENGINE_WHEEL_URL}" -q
      done_ "axengine wheel installed"
      mark_state "axengine_linked"
    fi
  fi
fi

# ── 4. optional extras ────────────────────────────────────────

if [[ "${WITH_WENET}" -eq 1 ]]; then
  if source_state "wenet_installed"; then
    skip "wenet-onnx"
  else
    python -m pip install -r backend/requirements-wenet-onnx.txt -q
    done_ "wenet-onnx installed"
    mark_state "wenet_installed"
  fi
fi

if [[ "${WITH_FIRERED}" -eq 1 ]]; then
  if source_state "firered_installed"; then
    skip "fireredasr-aed"
  else
    python -m pip install -r backend/requirements-fireredasr-aed.txt -q
    done_ "fireredasr-aed installed"
    mark_state "firered_installed"
  fi
fi

# ── 5. models ─────────────────────────────────────────────────

if [[ "${SKIP_MODELS}" -eq 1 ]]; then
  skip "model download (--no-models)"
elif [[ "${#MODELS[@]}" -gt 0 ]]; then
  if source_state "models_downloaded"; then
    skip "model download"
  else
    export HF_ENDPOINT HER_AXERA_MODEL_ROOT="${MODEL_ROOT}"
    log "downloading models: ${MODELS[*]}"
    python backend/tools/download_ax_models.py "${MODELS[@]}" \
      --root "${MODEL_ROOT}" \
      --endpoint "${HF_ENDPOINT}" \
      --env-file backend/.env.models
    done_ "models downloaded"
    mark_state "models_downloaded"
  fi
else
  log "no models requested; use --models all or --models \"sensevoice kokoro speaker\""
fi

# ── 6. start backend ──────────────────────────────────────────

cd "${REPO_ROOT}/backend"
mkdir -p data
PID_FILE="${REPO_ROOT}/backend/data/her-axera-backend.pid"
LOG_FILE="${REPO_ROOT}/backend/data/her-axera-backend.log"

# load env
set -a
source .env
[[ -f .env.models ]] && source .env.models
set +a

health_check() {
  python - "$1" <<'PY' 2>/dev/null
import json, sys, urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        return 0 if json.loads(r.read()).get("status") == "ok" else 1
except Exception:
    raise SystemExit(1)
PY
}

ALREADY_RUNNING=0
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  if health_check "http://127.0.0.1:${PORT}/health"; then
    skip "backend already running (pid $(cat "${PID_FILE}"))"
    ALREADY_RUNNING=1
  else
    log "stale pid found; restarting..."
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
    sleep 1
  fi
fi

if [[ "${ALREADY_RUNNING}" -eq 0 ]]; then
  log "starting backend on ${HOST}:${PORT}"
  if [[ "${FOREGROUND}" -eq 1 ]]; then
    exec uvicorn app.main:app --host "${HOST}" --port "${PORT}"
  fi
  nohup uvicorn app.main:app --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
  echo "$!" >"${PID_FILE}"

  for _ in $(seq 1 30); do
    if health_check "http://127.0.0.1:${PORT}/health"; then
      done_ "backend healthy: http://127.0.0.1:${PORT}/health"
      break
    fi
    sleep 1
  done

  if ! health_check "http://127.0.0.1:${PORT}/health"; then
    echo "[deploy] ✗  backend failed to become healthy within 30s" >&2
    tail -40 "${LOG_FILE}" >&2
    exit 1
  fi
fi

# ── 7. systemd service ────────────────────────────────────────

if [[ "${SKIP_SERVICE}" -eq 1 ]]; then
  skip "systemd service (--no-service)"
elif [[ "${FOREGROUND}" -eq 1 ]]; then
  skip "systemd service (foreground mode)"
elif source_state "service_installed"; then
  skip "systemd service"
elif [[ -f systemd/her-axera-backend.service.in ]]; then
  TEMPLATE="${REPO_ROOT}/systemd/her-axera-backend.service.in"
  SERVICE_NAME="her-axera-backend"
  SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
  log "installing systemd service..."

  TMPFILE="$(mktemp)"
  sed -e "s|@REPO_ROOT@|${REPO_ROOT}|g" -e "s|@PORT@|${PORT}|g" "${TEMPLATE}" >"${TMPFILE}"
  sudo install -m 0644 "${TMPFILE}" "${SERVICE_PATH}"
  rm -f "${TMPFILE}"
  sudo systemctl daemon-reload
  sudo systemctl enable "${SERVICE_NAME}.service"
  done_ "systemd service installed & enabled"
  mark_state "service_installed"
fi

# ── 8. summary ────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Her.axera deployed ✓"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Backend : http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo "<AX650_IP>"):${PORT}"
echo "  Health  : http://127.0.0.1:${PORT}/health"
echo "  UI      : http://<AX650_IP>:${PORT}/ui/"
echo "  Logs    : ${LOG_FILE}"
if [[ "${SKIP_SERVICE}" -eq 0 && "${FOREGROUND}" -eq 0 ]]; then
  echo "  Service : systemctl status ${SERVICE_NAME}"
fi
echo ""
echo "  PC 端启动前端:"
echo "    scripts/pc_run_frontend.sh --backend-url http://<AX650_IP>:${PORT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
