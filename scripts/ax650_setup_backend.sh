#!/usr/bin/env bash
set -euo pipefail

# Error handler: print last command and line number on failure
error_handler() {
    echo "[ERROR] Failed at line $1: $BASH_COMMAND" >&2
}
trap 'error_handler ${LINENO}' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/backend/.venv}"
VENV_SYSTEM_SITE_PACKAGES="${VENV_SYSTEM_SITE_PACKAGES:-0}"
AXENGINE_WHEEL_URL="${AXENGINE_WHEEL_URL:-https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc2/axengine-0.1.3-py3-none-any.whl}"
MODEL_ROOT="${MODEL_ROOT:-/opt/models/her-axera}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODELS=()
INSTALL_WENET=0
INSTALL_FIRERED=0
SKIP_VENV=0

usage() {
  cat <<'EOF'
Usage: scripts/ax650_setup_backend.sh [options]

Prepare the AX650 backend runtime without overwriting existing configuration.

Options:
  --models "speaker"  Download selected AXERA model repos.
  --models all                   Download all known AXERA model repos.
  --model-root PATH              Model root directory. Default: /opt/models/her-axera
  --hf-endpoint URL              Hugging Face endpoint. Default: https://hf-mirror.com
  --with-wenet-onnx              Install optional WeNet ONNX dependencies.
  
  --no-venv                      Install into the current Python environment.
  --system-site-packages         Expose board system site-packages to the venv.
  -h, --help                     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      [[ $# -gt 0 ]] || { echo "--models requires a value" >&2; exit 2; }
      read -r -a MODELS <<<"$1"
      ;;
    --model-root)
      shift
      [[ $# -gt 0 ]] || { echo "--model-root requires a value" >&2; exit 2; }
      MODEL_ROOT="$1"
      ;;
    --hf-endpoint)
      shift
      [[ $# -gt 0 ]] || { echo "--hf-endpoint requires a value" >&2; exit 2; }
      HF_ENDPOINT="$1"
      ;;
    --with-wenet-onnx)
      INSTALL_WENET=1
      ;;
    --with-fireredasr-aed)
      INSTALL_FIRERED=1
      ;;
    --no-venv)
      SKIP_VENV=1
      ;;
    --system-site-packages)
      VENV_SYSTEM_SITE_PACKAGES=1
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

cd "${REPO_ROOT}"

if [[ ! -f backend/requirements.txt ]]; then
  echo "Run this script from the Her.axera repository, or keep it under scripts/." >&2
  exit 1
fi

SYSTEM_PYTHON_BIN="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"

if [[ ! -f backend/.env ]]; then
  cp backend/.env.example backend/.env
  echo "[env] created backend/.env from backend/.env.example"
else
  echo "[env] backend/.env already exists; leaving it unchanged"
fi

if [[ "${SKIP_VENV}" -eq 0 ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    venv_args=()
    if [[ "${VENV_SYSTEM_SITE_PACKAGES}" -eq 1 ]]; then
      venv_args+=(--system-site-packages)
    fi
    "${PYTHON_BIN}" -m venv "${venv_args[@]}" "${VENV_DIR}"
    echo "[venv] created ${VENV_DIR}"
  else
    echo "[venv] using existing ${VENV_DIR}"
  fi
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
else
  echo "[venv] skipped; installing into current Python environment"
fi

python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install -r backend/requirements-model-download.txt

if [[ -e /soc/lib/libax_engine.so ]]; then
  if LD_LIBRARY_PATH="/soc/lib:${LD_LIBRARY_PATH:-}" python - <<'PY' >/dev/null 2>&1
import axengine
PY
  then
    echo "[axengine] available with /soc/lib"
  else
    if axengine_paths="$(LD_LIBRARY_PATH="/soc/lib:${LD_LIBRARY_PATH:-}" "${SYSTEM_PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("axengine")
if spec is None or spec.origin is None:
    raise SystemExit(1)
package_dir = Path(spec.origin).parent
dist_dir = Path(metadata.distribution("axengine")._path)
print(package_dir)
print(dist_dir)
PY
    )"; then
      venv_site="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
      )"
      axengine_package_dir="$(printf '%s\n' "${axengine_paths}" | sed -n '1p')"
      axengine_dist_dir="$(printf '%s\n' "${axengine_paths}" | sed -n '2p')"
      ln -sfn "${axengine_package_dir}" "${venv_site}/axengine"
      ln -sfn "${axengine_dist_dir}" "${venv_site}/$(basename "${axengine_dist_dir}")"
      echo "[axengine] linked from ${axengine_package_dir}"
    else
      echo "[axengine] not visible in Python; installing AXERA pyaxengine wheel"
      python -m pip install "axengine @ ${AXENGINE_WHEEL_URL}"
    fi
    LD_LIBRARY_PATH="/soc/lib:${LD_LIBRARY_PATH:-}" python - <<'PY' >/dev/null
import axengine
PY
    echo "[axengine] available with /soc/lib"
  fi
fi

if [[ "${INSTALL_WENET}" -eq 1 ]]; then
  python -m pip install -r backend/requirements-wenet-onnx.txt
fi

if [[ "${INSTALL_FIRERED}" -eq 1 ]]; then
fi

if [[ "${#MODELS[@]}" -gt 0 ]]; then
  export HF_ENDPOINT
  export HER_AXERA_MODEL_ROOT="${MODEL_ROOT}"
  python backend/tools/download_ax_models.py "${MODELS[@]}" \
    --root "${MODEL_ROOT}" \
    --endpoint "${HF_ENDPOINT}" \
    --env-file backend/.env.models
  echo "[models] wrote backend/.env.models"
  echo "[models] review it, then merge selected values into backend/.env"
else
  echo "[models] no models requested; pass --models \"speaker\" or --models all to download"
fi

echo "[done] AX650 backend setup completed"
