import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "download_ax_models.py"
SPEC = importlib.util.spec_from_file_location("download_ax_models", SCRIPT_PATH)
assert SPEC is not None
download_ax_models = importlib.util.module_from_spec(SPEC)
sys.modules["download_ax_models"] = download_ax_models
assert SPEC.loader is not None
SPEC.loader.exec_module(download_ax_models)


def test_resolve_models_deduplicates_aliases() -> None:
    selected = download_ax_models.resolve_models(["speaker", "3d_speaker", "speaker"])
    assert [spec.key for spec in selected] == ["speaker", "3d_speaker"]


def test_render_env_uses_hf_mirror_and_model_paths() -> None:
    root = Path("/opt/models/her-axera")
    selected = download_ax_models.resolve_models(["speaker", "3d_speaker"])
    env_text = download_ax_models.render_env(selected, root, "https://hf-mirror.com")
    assert "HF_ENDPOINT=https://hf-mirror.com" in env_text
    assert "SPEAKER_REPO_PATH=/opt/models/her-axera/3D-Speaker-MT.Axera" in env_text
    assert "SPEAKER_MODEL_DIR=/opt/models/her-axera/3D-Speaker-MT.Axera/axmodel" in env_text
    assert "SENSEVOICE_REPO_PATH=/opt/models/her-axera/SenseVoice" in env_text
