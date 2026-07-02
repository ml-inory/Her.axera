#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/backend/.venv}"
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
  --models "sensevoice speaker"  Download selected AXERA model repos.
  --models all                   Download all known AXERA model repos.
  --model-root PATH              Model root directory. Default: /opt/models/her-axera
  --hf-endpoint URL              Hugging Face endpoint. Default: https://hf-mirror.com
  --with-wenet-onnx              Install optional WeNet ONNX dependencies.
  --with-fireredasr-aed          Install optional FireRedASR-AED dependencies.
  --no-venv                      Install into the current Python environment.
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

if [[ ! -f backend/.env ]]; then
  cp backend/.env.example backend/.env
  echo "[env] created backend/.env from backend/.env.example"
else
  echo "[env] backend/.env already exists; leaving it unchanged"
fi

if [[ "${SKIP_VENV}" -eq 0 ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
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

if [[ "${INSTALL_WENET}" -eq 1 ]]; then
  python -m pip install -r backend/requirements-wenet-onnx.txt
fi

if [[ "${INSTALL_FIRERED}" -eq 1 ]]; then
  python -m pip install -r backend/requirements-fireredasr-aed.txt
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
  echo "[models] no models requested; pass --models \"sensevoice speaker\" or --models all to download"
fi

echo "[done] AX650 backend setup completed"
