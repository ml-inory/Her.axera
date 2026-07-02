from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_deploy_scripts_exist_and_are_non_destructive() -> None:
    scripts = {
        "scripts/ax650_setup_backend.sh": ["https://hf-mirror.com", "backend/requirements-model-download.txt", "backend/.env already exists"],
        "scripts/ax650_run_backend.sh": ["uvicorn app.main:app", "/health", "backend/.env.models"],
        "scripts/ax650_install_service.sh": ["systemctl daemon-reload", "--enable", "her-axera-backend.service"],
        "scripts/pc_run_frontend.sh": ["--backend-url", "?api=", "http.server"],
    }
    for relative_path, expected_fragments in scripts.items():
        path = REPO_ROOT / relative_path
        assert path.exists(), relative_path
        content = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in content
        for fragment in expected_fragments:
            assert fragment in content


def test_systemd_template_points_at_repo_backend() -> None:
    template = (REPO_ROOT / "systemd/her-axera-backend.service.in").read_text(encoding="utf-8")
    assert "WorkingDirectory=@REPO_ROOT@/backend" in template
    assert "EnvironmentFile=@REPO_ROOT@/backend/.env" in template
    assert "ExecStart=@REPO_ROOT@/backend/.venv/bin/uvicorn app.main:app" in template
