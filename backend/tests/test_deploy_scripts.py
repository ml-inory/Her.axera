import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_deploy_scripts_exist_and_are_non_destructive() -> None:
    scripts = {
        "scripts/ax650_setup_backend.sh": [
            "https://hf-mirror.com",
            "ln -sfn",
            "AXENGINE_WHEEL_URL",
            "backend/requirements-model-download.txt",
            "backend/.env already exists",
        ],
        "scripts/ax650_run_backend.sh": ["uvicorn app.main:app", "/health", "backend/.env.models"],
        "scripts/pc_run_frontend.sh": ["--backend-url", "?api=", "http.server"],
    }
    for relative_path, expected_fragments in scripts.items():
        path = REPO_ROOT / relative_path
        assert path.exists(), relative_path
        content = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in content
        for fragment in expected_fragments:
            assert fragment in content


def test_env_example_is_shell_sourceable_for_ax650_runtime() -> None:
    command = """
set -euo pipefail
set -a
source backend/.env.example
set +a
test "$APP_NAME" = "Her Voice Dialogue API"
case ":${LD_LIBRARY_PATH}:" in
  *:/soc/lib:*) ;;
  *) echo "LD_LIBRARY_PATH must include /soc/lib" >&2; exit 1 ;;
esac
"""
    subprocess.run(["bash", "-c", command], cwd=REPO_ROOT, check=True)
