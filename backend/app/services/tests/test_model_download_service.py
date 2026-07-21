from app.services.model_download_service import (
    ModelDownloadManager,
    ModelDownloadSpec,
    DownloadStatus,
    _build_model_specs,
    get_model_download_manager,
)


class TestModelDownloadSpec:
    def test_minimal_spec(self) -> None:
        spec = ModelDownloadSpec(key="asr_test", display_name="TM", repo_id="org/repo")
        assert spec.key == "asr_test"
        assert spec.repo_id == "org/repo"
        assert spec.model_type == ""
        assert spec.depends_on == []
        assert spec.allow_patterns is None

    def test_full_spec(self) -> None:
        spec = ModelDownloadSpec(
            key="tts_k", display_name="K", repo_id="AXERA-TECH/k",
            allow_patterns=["*.onnx"], required_files=["m.onnx"],
            depends_on=["base"], model_type="tts",
        )
        assert spec.required_files == ["m.onnx"]
        assert spec.depends_on == ["base"]
        assert spec.model_type == "tts"


class TestDownloadStatus:
    def test_enum_values(self) -> None:
        assert DownloadStatus.NOT_STARTED == "not_started"
        assert DownloadStatus.DOWNLOADING == "downloading"
        assert DownloadStatus.DOWNLOADED == "downloaded"
        assert DownloadStatus.FAILED == "failed"
        assert DownloadStatus.NOT_NEEDED == "not_needed"


class TestModelDownloadManager:
    def test_singleton(self) -> None:
        mgr1 = get_model_download_manager()
        mgr2 = get_model_download_manager()
        assert mgr1 is mgr2

    def test_build_model_specs(self) -> None:
        specs = _build_model_specs()
        assert isinstance(specs, dict)
        assert len(specs) > 0
        for v in specs.values():
            assert isinstance(v, ModelDownloadSpec)

    def test_get_state_for_unknown(self) -> None:
        mgr = ModelDownloadManager()
        assert mgr.get_state("nonexistent_key") is None

    def test_get_all_states(self) -> None:
        mgr = ModelDownloadManager()
        states = mgr.get_all_states()
        assert isinstance(states, dict)

    def test_get_states_by_type(self) -> None:
        mgr = ModelDownloadManager()
        asr_states = mgr.get_states_by_type("asr")
        assert isinstance(asr_states, dict)

    def test_is_ready(self) -> None:
        mgr = ModelDownloadManager()
        # Should not raise
        ready = mgr.is_ready("asr")
        assert isinstance(ready, bool)

    def test_on_progress_callback(self) -> None:
        mgr = ModelDownloadManager()
        called = []

        def cb(key: str, state) -> None:
            called.append(key)

        mgr.on_progress(cb)
        assert len(called) == 0  # callback registered, not called yet

    def test_start_download_all_unknown_type(self) -> None:
        mgr = ModelDownloadManager()
        started = mgr.start_download_all(model_type="nonexistent_type")
        assert started == []
