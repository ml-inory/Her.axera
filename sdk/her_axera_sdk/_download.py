"""Model download manager for the SDK."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger("her_axera_sdk.download")


@dataclass
class ModelSpec:
    key: str
    name: str
    repo_id: str
    allow_patterns: list[str] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    size_hint: str = ""


class ModelDownloader:
    """Downloads ASR/TTS models from Hugging Face on demand."""

    def __init__(
        self,
        asr_root: Path,
        tts_root: Path,
        asr_specs: list[ModelSpec],
        tts_specs: list[ModelSpec],
        espeak_data_path: str = "espeak-ng-data",
        jieba_dict_path: str = "dict",
    ) -> None:
        self._asr_root = asr_root
        self._tts_root = tts_root
        self._asr_specs = asr_specs
        self._tts_specs = tts_specs
        self._espeak_data_path = espeak_data_path
        self._jieba_dict_path = jieba_dict_path
        self._progress_callbacks: list[Callable[[str, str, float], None]] = []

    def on_progress(self, callback: Callable[[str, str, float], None]) -> None:
        """Register a callback (key, status, pct) for progress updates."""
        self._progress_callbacks.append(callback)

    def check_all(self) -> dict[str, bool]:
        """Check which models are present. Returns {key: ready}."""
        result: dict[str, bool] = {}
        for spec in self._asr_specs + self._tts_specs:
            result[spec.key] = self._is_ready(spec)
        return result

    def check_asr(self) -> bool:
        return all(self._is_ready(s) for s in self._asr_specs)

    def check_tts(self) -> bool:
        return all(self._is_ready(s) for s in self._tts_specs)

    def download_all(self) -> dict[str, str]:
        """Download all missing models. Returns {key: status}."""
        results: dict[str, str] = {}
        for spec in self._asr_specs:
            results[spec.key] = self._download_one(spec, self._asr_root)
        for spec in self._tts_specs:
            results[spec.key] = self._download_one(spec, self._tts_root)
        return results

    def download_asr(self) -> None:
        for spec in self._asr_specs:
            self._download_one(spec, self._asr_root)

    def download_tts(self) -> None:
        for spec in self._tts_specs:
            self._download_one(spec, self._tts_root)

    # ---- internals ----

    def _is_ready(self, spec: ModelSpec) -> bool:
        root = self._asr_root if spec in self._asr_specs else self._tts_root
        if spec.required_files:
            return all((root / f).exists() for f in spec.required_files)
        return (root / spec.required_files[0]).parent.is_dir() if spec.required_files else False

    def _download_one(self, spec: ModelSpec, root: Path) -> str:
        if self._is_ready(spec):
            return "already_downloaded"

        self._notify(spec.key, "downloading", 0.0)
        try:
            from huggingface_hub import snapshot_download

            endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            token = os.environ.get("HF_TOKEN")

            root.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix="her_sdk_") as tmpdir:
                snapshot_download(
                    repo_id=spec.repo_id,
                    repo_type="model",
                    revision="main",
                    local_dir=tmpdir,
                    endpoint=endpoint,
                    token=token,
                    allow_patterns=spec.allow_patterns if spec.allow_patterns else None,
                    max_workers=4,
                )
                # Move files from tmpdir to root
                for item in os.listdir(tmpdir):
                    src = os.path.join(tmpdir, item)
                    dst = os.path.join(str(root), item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst, ignore_errors=True)
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

            self._notify(spec.key, "downloaded", 100.0)
            return "downloaded"
        except Exception as exc:
            logger.exception("Download failed: %s", spec.key)
            self._notify(spec.key, "failed", 0.0)
            return f"failed: {exc}"

    def _notify(self, key: str, status: str, pct: float) -> None:
        for cb in self._progress_callbacks:
            try:
                cb(key, status, pct)
            except Exception:
                pass
