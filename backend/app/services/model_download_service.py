"""Lazy model download manager for AX ASR/TTS providers.

Downloads models from Hugging Face on demand, tracks progress per model,
and exposes status for frontend polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model specs — what to download and where
# ---------------------------------------------------------------------------


@dataclass
class ModelDownloadSpec:
    """Describes a single model download target."""

    key: str  # unique id, e.g. "asr_sensevoice_ax650"
    display_name: str  # human-readable, e.g. "SenseVoice (AX650)"
    repo_id: str  # HuggingFace repo, e.g. "AXERA-TECH/SenseVoice"
    allow_patterns: list[str] | None = None  # glob patterns for hf_hub_download
    ignore_patterns: list[str] | None = None
    post_copy: list[tuple[str, str]] | None = None  # (src_rel, dst_rel) after download
    local_dir: str | None = None  # if set, download to this dir directly
    required_files: list[str] | None = None  # check these to verify download
    depends_on: list[str] = field(default_factory=list)  # keys of prerequisite specs
    model_type: str = ""  # "asr" | "tts" — which provider needs it


class DownloadStatus(str, Enum):
    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    FAILED = "failed"
    NOT_NEEDED = "not_needed"  # model not configured / provider disabled


@dataclass
class ModelDownloadState:
    spec: ModelDownloadSpec
    status: DownloadStatus = DownloadStatus.NOT_STARTED
    progress_pct: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error_message: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def _build_model_specs() -> dict[str, ModelDownloadSpec]:
    """Return the full catalog of downloadable models."""
    settings = get_settings()
    asr_root = Path(settings.ax_asr_model_path or "models-ax650")
    tts_root = Path(settings.ax_tts_model_path or "models-ax650")

    specs: dict[str, ModelDownloadSpec] = {}

    # ---- ASR: SenseVoice ----
    sensevoice_files = [
        "sensevoice_ax650/sensevoice.axmodel",
        "sensevoice_ax650/streaming_sensevoice.axmodel",
    ]
    specs["asr_sensevoice"] = ModelDownloadSpec(
        key="asr_sensevoice",
        display_name="SenseVoice ASR (AX650)",
        repo_id="AXERA-TECH/SenseVoice",
        allow_patterns=["sensevoice_ax650/*"],
        local_dir=str(asr_root / "sensevoice"),
        required_files=[str(asr_root / "sensevoice" / f) for f in sensevoice_files],
        model_type="asr",
    )

    # ---- ASR: Whisper ----
    whisper_files = [
        "whisper_tiny/whisper_encoder_tiny.axmodel",
        "whisper_tiny/whisper_decoder_tiny.axmodel",
    ]
    specs["asr_whisper_tiny"] = ModelDownloadSpec(
        key="asr_whisper_tiny",
        display_name="Whisper Tiny (AX650)",
        repo_id="AXERA-TECH/Whisper",
        allow_patterns=["models-ax650/*"],
        local_dir=str(asr_root / "whisper"),
        post_copy=[("models-ax650", ".")],
        required_files=[str(asr_root / "whisper" / f) for f in whisper_files],
        model_type="asr",
    )

    # ---- TTS: Kokoro model files ----
    kokoro_files = [
        "kokoro_part1_96.axmodel",
        "kokoro_part2_96.axmodel",
        "kokoro_part3_96.axmodel",
        "model4_har_sim.onnx",
        "vocab.txt",
    ]
    specs["tts_kokoro_model"] = ModelDownloadSpec(
        key="tts_kokoro_model",
        display_name="Kokoro TTS Model (AX650)",
        repo_id="AXERA-TECH/kokoro.axera",
        allow_patterns=[
            "models/kokoro_part1_96.axmodel",
            "models/kokoro_part2_96.axmodel",
            "models/kokoro_part3_96.axmodel",
            "models/model4_har_sim.onnx",
        ],
        local_dir=str(tts_root / "kokoro"),
        required_files=[str(tts_root / "kokoro" / f) for f in kokoro_files[:4]],
        model_type="tts",
    )

    # ---- TTS: Kokoro voices ----
    specs["tts_kokoro_voices"] = ModelDownloadSpec(
        key="tts_kokoro_voices",
        display_name="Kokoro TTS Voices",
        repo_id="AXERA-TECH/kokoro.axera",
        allow_patterns=["cpp/voices/*"],
        local_dir=str(tts_root / "kokoro" / "voices"),
        required_files=[str(tts_root / "kokoro" / "voices" / "voices.json")],
        depends_on=["tts_kokoro_model"],
        model_type="tts",
    )

    # ---- TTS: Vocab (bundled with kokoro model but needs a check) ----
    specs["tts_kokoro_vocab"] = ModelDownloadSpec(
        key="tts_kokoro_vocab",
        display_name="Kokoro Vocab",
        repo_id="AXERA-TECH/kokoro.axera",
        allow_patterns=["models/vocab.txt"],
        local_dir=str(tts_root / "kokoro"),
        required_files=[str(tts_root / "kokoro" / "vocab.txt")],
        depends_on=["tts_kokoro_model"],
        model_type="tts",
    )

    return specs


# ---------------------------------------------------------------------------
# Download manager (singleton)
# ---------------------------------------------------------------------------


class ModelDownloadManager:
    """Tracks model download state and runs downloads in background threads."""

    def __init__(self) -> None:
        self.specs = _build_model_specs()
        self._states: dict[str, ModelDownloadState] = {}
        self._lock = threading.Lock()
        self._download_threads: dict[str, threading.Thread] = {}
        self._progress_callbacks: list[Callable[[str, ModelDownloadState], None]] = []

    # ---- state queries ----

    def get_state(self, key: str) -> ModelDownloadState | None:
        with self._lock:
            if key not in self._states:
                spec = self.specs.get(key)
                if spec is None:
                    return None
                self._states[key] = ModelDownloadState(spec=spec)
                # check if already downloaded
                if self._is_downloaded(spec):
                    self._states[key].status = DownloadStatus.DOWNLOADED
            return self._states[key]

    def get_all_states(self) -> dict[str, ModelDownloadState]:
        result: dict[str, ModelDownloadState] = {}
        for key in self.specs:
            state = self.get_state(key)
            if state:
                result[key] = state
        return result

    def get_states_by_type(self, model_type: str) -> dict[str, ModelDownloadState]:
        return {
            key: state
            for key, state in self.get_all_states().items()
            if state.spec.model_type == model_type
        }

    def is_ready(self, model_type: str) -> bool:
        """Check if all models for a given provider type are downloaded."""
        states = self.get_states_by_type(model_type)
        if not states:
            return True  # no models needed = ready
        return all(s.status == DownloadStatus.DOWNLOADED for s in states.values())

    def is_model_present(self, model_type: str, required_files: list[str] | None = None) -> bool:
        """Quick check: are the model files on disk?"""
        if required_files:
            return all(Path(f).exists() for f in required_files)
        states = self.get_states_by_type(model_type)
        if not states:
            return True
        return all(self._is_downloaded(s.spec) for s in states.values())

    # ---- download triggers ----

    def start_download(self, key: str) -> bool:
        """Start downloading a model. Returns False if already done or in progress."""
        state = self.get_state(key)
        if state is None:
            return False
        if state.status in (DownloadStatus.DOWNLOADING, DownloadStatus.DOWNLOADED):
            return False

        with self._lock:
            state.status = DownloadStatus.DOWNLOADING
            state.progress_pct = 0.0
            state.error_message = ""
            import time

            state.started_at = time.time()

        thread = threading.Thread(target=self._download_thread, args=(key,), daemon=True)
        self._download_threads[key] = thread
        thread.start()
        return True

    def start_download_all(self, model_type: str | None = None) -> list[str]:
        """Start downloading all models, optionally filtered by type."""
        started: list[str] = []
        for key, spec in self.specs.items():
            if model_type and spec.model_type != model_type:
                continue
            if self.start_download(key):
                started.append(key)
        return started

    def on_progress(self, callback: Callable[[str, ModelDownloadState], None]) -> None:
        """Register a callback invoked when progress changes."""
        self._progress_callbacks.append(callback)

    # ---- internals ----

    def _is_downloaded(self, spec: ModelDownloadSpec) -> bool:
        """Check if required files exist on disk."""
        if spec.required_files:
            return all(Path(f).exists() for f in spec.required_files)
        if spec.local_dir:
            return Path(spec.local_dir).is_dir() and any(Path(spec.local_dir).iterdir())
        return False

    def _download_thread(self, key: str) -> None:
        """Background thread that downloads a single model."""
        state = self.get_state(key)
        if state is None:
            return
        spec = state.spec

        try:
            self._do_download(spec, key)
            with self._lock:
                state.status = DownloadStatus.DOWNLOADED
                state.progress_pct = 100.0
                import time

                state.finished_at = time.time()
        except Exception as exc:
            logger.exception("Model download failed: %s", key)
            with self._lock:
                state.status = DownloadStatus.FAILED
                state.error_message = str(exc)
                import time

                state.finished_at = time.time()
        finally:
            self._notify_progress(key)

    def _do_download(self, spec: ModelDownloadSpec, key: str) -> None:
        """Execute the actual download via huggingface_hub."""
        from huggingface_hub import snapshot_download

        endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        token = os.environ.get("HF_TOKEN")

        local_dir = spec.local_dir or "."
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Build progress callback
        last_pct = [0.0]

        def _progress_callback(progress: int, total: int) -> None:
            pct = progress / max(total, 1) * 100.0
            if pct - last_pct[0] < 2.0 and pct < 99.9:
                return
            last_pct[0] = pct
            with self._lock:
                state = self._states.get(key)
                if state:
                    state.progress_pct = round(pct, 1)
                    state.downloaded_bytes = progress
                    state.total_bytes = total
            self._notify_progress(key)

        # Download allowed patterns into a temp dir, then copy
        import tempfile
        import shutil

        if spec.allow_patterns:
            # For pattern-based downloads, download into a temp dir then move
            with tempfile.TemporaryDirectory(prefix="her_model_") as tmpdir:
                snapshot_download(
                    repo_id=spec.repo_id,
                    repo_type="model",
                    revision="main",
                    local_dir=tmpdir,
                    endpoint=endpoint,
                    token=token,
                    allow_patterns=spec.allow_patterns,
                    ignore_patterns=spec.ignore_patterns,
                    max_workers=4,
                    tqdm_class=None,
                )
                # Copy from tmpdir to local_dir
                for item in os.listdir(tmpdir):
                    src = os.path.join(tmpdir, item)
                    dst = os.path.join(local_dir, item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst, ignore_errors=True)
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

            # Handle post_copy (move files from subdir to root)
            if spec.post_copy:
                for src_rel, dst_rel in spec.post_copy:
                    src_dir = os.path.join(local_dir, src_rel)
                    dst_dir = os.path.join(local_dir, dst_rel) if dst_rel != "." else local_dir
                    if os.path.isdir(src_dir) and src_dir != dst_dir:
                        for item in os.listdir(src_dir):
                            s = os.path.join(src_dir, item)
                            d = os.path.join(dst_dir, item)
                            if os.path.isdir(s):
                                if os.path.exists(d):
                                    shutil.rmtree(d, ignore_errors=True)
                                shutil.move(s, d)
                            else:
                                shutil.move(s, d)
                        shutil.rmtree(src_dir, ignore_errors=True)
        else:
            # Full repo download into local_dir
            snapshot_download(
                repo_id=spec.repo_id,
                repo_type="model",
                revision="main",
                local_dir=local_dir,
                endpoint=endpoint,
                token=token,
                max_workers=4,
                tqdm_class=None,
            )

        # Verify
        if spec.required_files:
            missing = [f for f in spec.required_files if not Path(f).exists()]
            if missing:
                raise RuntimeError(f"Download completed but missing files: {missing}")

    def _notify_progress(self, key: str) -> None:
        state = self.get_state(key)
        if state is None:
            return
        for cb in self._progress_callbacks:
            try:
                cb(key, state)
            except Exception:
                logger.exception("Progress callback failed for %s", key)


# Singleton
_model_download_manager: ModelDownloadManager | None = None


def get_model_download_manager() -> ModelDownloadManager:
    global _model_download_manager
    if _model_download_manager is None:
        _model_download_manager = ModelDownloadManager()
    return _model_download_manager
