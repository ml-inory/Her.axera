from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_deploy_scripts_exist_and_are_non_destructive() -> None:
    scripts = {
        "scripts/ax650_setup_backend.sh": ["https://hf-mirror.com", "backend/requirements-model-download.txt", "backend/.env already exists"],
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
