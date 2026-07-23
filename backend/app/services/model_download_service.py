"""Lazy model download manager for AX ASR/TTS providers.

Downloads models from HuggingFace or ModelScope on demand, tracks progress per model,
and exposes status for frontend polling.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELSCOPE_API = "https://modelscope.cn/api/v1"

# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------


@dataclass
class ModelDownloadSpec:
    """Describes a single model download target."""

    key: str
    display_name: str
    repo_id: str
    source: str = "modelscope"  # "modelscope" | "huggingface"
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    post_copy: list[tuple[str, str]] | None = None
    local_dir: str | None = None
    required_files: list[str] | None = None
    depends_on: list[str] = field(default_factory=list)
    model_type: str = ""
    strip_prefix: str | None = None  # strip from downloaded path
    rename_map: dict[str, str] = field(default_factory=dict)


class DownloadStatus(str, Enum):
    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_NEEDED = "not_needed"


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
    model_root = Path(os.environ.get("HER_AXERA_MODEL_ROOT", "/root/models/her-axera"))
    asr_root = model_root / "asr"
    tts_root = model_root / "tts"

    specs: dict[str, ModelDownloadSpec] = {}




    return specs


# ---------------------------------------------------------------------------
# Download Manager
# ---------------------------------------------------------------------------


class ModelDownloadManager:
    """Manages lazy, cancelable model downloads with progress tracking."""

    def __init__(self) -> None:
        self.specs = _build_model_specs()
        self._lock = threading.Lock()
        self._states: dict[str, ModelDownloadState] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._progress_callbacks: list[Callable] = []
        self._init_states()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_model_root(self, root: str) -> None:
        """Change model storage root and rebuild specs/states."""
        os.environ["HER_AXERA_MODEL_ROOT"] = root
        self.specs = _build_model_specs()
        with self._lock:
            new_states: dict[str, ModelDownloadState] = {}
            for key, spec in self.specs.items():
                if key in self._states:
                    old = self._states[key]
                    new_states[key] = ModelDownloadState(
                        spec=spec,
                        status=old.status,
                        progress_pct=old.progress_pct,
                        downloaded_bytes=old.downloaded_bytes,
                        total_bytes=old.total_bytes,
                        error_message=old.error_message,
                    )
                else:
                    new_states[key] = ModelDownloadState(spec=spec)
            self._states = new_states
        new_root = Path(root)
        os.environ["AX_ASR_MODEL_PATH"] = str(new_root / "asr")
        os.environ["AX_TTS_MODEL_PATH"] = str(new_root / "tts")
        logger.info("Model root changed to %s", root)

    def get_state(self, key: str) -> ModelDownloadState | None:
        with self._lock:
            return self._states.get(key)

    def get_all_states(self) -> dict[str, ModelDownloadState]:
        with self._lock:
            return dict(self._states)

    def start_download(self, key: str) -> bool:
        with self._lock:
            if key not in self._states:
                return False
            state = self._states[key]
            if state.status in (DownloadStatus.DOWNLOADING, DownloadStatus.DOWNLOADED):
                return False

        spec = self.specs.get(key)
        if spec and spec.depends_on:
            for dep_key in spec.depends_on:
                dep_state = self.get_state(dep_key)
                if dep_state and dep_state.status != DownloadStatus.DOWNLOADED:
                    logger.warning(
                        "Dependency %s not ready for %s (status=%s)",
                        dep_key, key,
                        dep_state.status.value if dep_state else "unknown",
                    )

        cancel = threading.Event()
        with self._lock:
            self._cancel_events[key] = cancel
            self._states[key].status = DownloadStatus.DOWNLOADING
            self._states[key].progress_pct = 0.0
            self._states[key].downloaded_bytes = 0
            self._states[key].total_bytes = 0
            self._states[key].error_message = ""
            self._states[key].started_at = time.time()

        t = threading.Thread(target=self._download_thread, args=(key, cancel), daemon=True)
        with self._lock:
            self._threads[key] = t
        t.start()
        return True

    def cancel_download(self, key: str) -> bool:
        with self._lock:
            event = self._cancel_events.get(key)
        if event:
            event.set()
            return True
        return False

    def add_progress_callback(self, cb: Callable) -> None:
        self._progress_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_states(self) -> None:
        for key, spec in self.specs.items():
            state = ModelDownloadState(spec=spec)
            if spec.required_files and all(Path(f).exists() for f in spec.required_files):
                state.status = DownloadStatus.DOWNLOADED
                state.progress_pct = 100.0
            self._states[key] = state

    def _download_thread(self, key: str, cancel: threading.Event) -> None:
        spec = self.specs.get(key)
        if spec is None:
            return
        try:
            if spec.source == "modelscope":
                self._download_from_modelscope(key, spec, cancel)
            else:
                self._download_from_huggingface(key, spec, cancel)
        except InterruptedError:
            with self._lock:
                if key in self._states:
                    self._states[key].status = DownloadStatus.CANCELLED
                    self._states[key].finished_at = time.time()
            self._notify_progress(key)
        except Exception as exc:
            logger.exception("Download %s failed", key)
            with self._lock:
                if key in self._states:
                    self._states[key].status = DownloadStatus.FAILED
                    self._states[key].error_message = str(exc)[:500]
                    self._states[key].finished_at = time.time()
            self._notify_progress(key)
        finally:
            with self._lock:
                self._cancel_events.pop(key, None)
                self._threads.pop(key, None)

    # ------------------------------------------------------------------
    # ModelScope download (REST API, no extra dependency)
    # ------------------------------------------------------------------

    def _list_modelscope_files(self, repo_id: str) -> list[dict]:
        url = f"{MODELSCOPE_API}/models/{repo_id}/repo/files"
        resp = requests.get(
            url, params={"Revision": "master", "Recursive": "true"}, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("Code") != 200:
            raise RuntimeError(f"ModelScope API error: {data}")
        return data["Data"]["Files"]

    def _download_from_modelscope(
        self, key: str, spec: ModelDownloadSpec, cancel: threading.Event
    ) -> None:
        local_dir = spec.local_dir or "."
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # 1. List and filter files
        all_files = self._list_modelscope_files(spec.repo_id)
        matching: list[dict] = []
        if spec.allow_patterns:
            for f in all_files:
                if f.get("Type") != "blob":
                    continue
                fname = f["Path"]
                for pat in spec.allow_patterns:
                    if fnmatch.fnmatch(fname, pat):
                        matching.append(f)
                        break
        else:
            matching = [f for f in all_files if f.get("Type") == "blob"]

        if spec.ignore_patterns:
            matching = [
                f for f in matching
                if not any(fnmatch.fnmatch(f["Path"], p) for p in spec.ignore_patterns)
            ]

        total_files = len(matching)
        if total_files == 0:
            raise RuntimeError(f"No files matched patterns {spec.allow_patterns}")

        total_size = sum(f.get("Size", 0) for f in matching)
        downloaded_size = 0

        # 2. Download each file with streaming progress
        for idx, file_info in enumerate(matching):
            if cancel.is_set():
                raise InterruptedError("Download cancelled")

            fname = file_info["Path"]

            # Strip prefix
            target_name = fname
            if spec.strip_prefix:
                prefix = spec.strip_prefix.rstrip("/") + "/"
                if fname.startswith(prefix):
                    target_name = fname[len(prefix):]

            # Apply rename map
            base_name = os.path.basename(target_name)
            if base_name in spec.rename_map:
                parent = os.path.dirname(target_name)
                target_name = (
                    os.path.join(parent, spec.rename_map[base_name])
                    if parent
                    else spec.rename_map[base_name]
                )

            dst_path = os.path.join(local_dir, target_name)
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

            try:
                resp = requests.get(
                    f"{MODELSCOPE_API}/models/{spec.repo_id}/repo",
                    params={"Revision": "master", "FilePath": fname},
                    stream=True,
                    timeout=(30, 600),
                )
                resp.raise_for_status()

                with open(dst_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if cancel.is_set():
                            resp.close()
                            raise InterruptedError("Download cancelled")
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                pct = downloaded_size / total_size * 100.0
                            else:
                                pct = (idx + 1) / total_files * 100.0
                            with self._lock:
                                state = self._states.get(key)
                                if state:
                                    state.progress_pct = round(pct, 1)
                                    state.downloaded_bytes = downloaded_size
                                    state.total_bytes = total_size

                logger.info("Downloaded %s -> %s", fname, dst_path)

            except InterruptedError:
                raise
            except Exception as exc:
                logger.warning("Failed to download %s: %s", fname, exc)

            # File-count progress as floor
            file_pct = (idx + 1) / total_files * 100.0
            with self._lock:
                state = self._states.get(key)
                if state and state.progress_pct < file_pct:
                    state.progress_pct = round(file_pct, 1)
            self._notify_progress(key)

        # 3. post_copy
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

        # 4. Verify
        if spec.required_files:
            missing = [f for f in spec.required_files if not Path(f).exists()]
            if missing:
                raise RuntimeError(f"Download completed but missing files: {missing}")

        # 5. Mark done
        with self._lock:
            if key in self._states:
                self._states[key].status = DownloadStatus.DOWNLOADED
                self._states[key].progress_pct = 100.0
                self._states[key].finished_at = time.time()
        self._notify_progress(key)

    # ------------------------------------------------------------------
    # HuggingFace download (fallback)
    # ------------------------------------------------------------------

    def _download_from_huggingface(
        self, key: str, spec: ModelDownloadSpec, cancel: threading.Event
    ) -> None:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        token = os.environ.get("HF_TOKEN")
        local_dir = spec.local_dir or "."
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(endpoint=endpoint, token=token)
        all_files = list(api.list_repo_files(spec.repo_id, repo_type="model", revision="main"))

        matching: list[str] = []
        if spec.allow_patterns:
            for fname in all_files:
                for pat in spec.allow_patterns:
                    if fnmatch.fnmatch(fname, pat):
                        matching.append(fname)
                        break
        else:
            matching = all_files

        if spec.ignore_patterns:
            matching = [
                f for f in matching
                if not any(fnmatch.fnmatch(f, p) for p in spec.ignore_patterns)
            ]

        total_files = len(matching)
        if total_files == 0:
            raise RuntimeError(f"No files matched patterns {spec.allow_patterns}")

        for idx, fname in enumerate(matching):
            if cancel.is_set():
                raise InterruptedError("Download cancelled")

            target_name = fname
            if spec.allow_patterns:
                for pat in spec.allow_patterns:
                    prefix = pat.rstrip("*").rstrip("/")
                    if prefix and fname.startswith(prefix + "/"):
                        target_name = fname[len(prefix) + 1:]
                        break

            try:
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=fname,
                    repo_type="model",
                    revision="main",
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    endpoint=endpoint,
                    token=token,
                )
                if target_name != fname:
                    src_path = os.path.join(local_dir, fname)
                    dst_path = os.path.join(local_dir, target_name)
                    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)
            except Exception:
                logger.warning("Failed to download %s, skipping", fname)

            pct = (idx + 1) / total_files * 100.0
            with self._lock:
                state = self._states.get(key)
                if state:
                    state.progress_pct = round(pct, 1)
                    state.downloaded_bytes = idx + 1
                    state.total_bytes = total_files
            self._notify_progress(key)

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

        if spec.required_files:
            missing = [f for f in spec.required_files if not Path(f).exists()]
            if missing:
                raise RuntimeError(f"Download completed but missing files: {missing}")

        with self._lock:
            if key in self._states:
                self._states[key].status = DownloadStatus.DOWNLOADED
                self._states[key].progress_pct = 100.0
                self._states[key].finished_at = time.time()
        self._notify_progress(key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
