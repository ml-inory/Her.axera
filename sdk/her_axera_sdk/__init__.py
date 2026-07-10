"""Her.axera SDK — unified ASR + LLM + TTS pipeline for AX650 boards.

Usage::

    from her_axera_sdk import HerAxeraSDK

    sdk = HerAxeraSDK(
        asr_model_path="/opt/models/asr/models-ax650",
        tts_model_path="/opt/models/tts/models-ax650",
        llm_api_base="https://api.deepseek.com",
        llm_api_key="sk-xxx",
    )

    # Single-step APIs
    text = sdk.transcribe("audio.wav")
    reply = sdk.chat("你好")
    wav = sdk.synthesize("你好世界")

    # Full streaming pipeline
    for event in sdk.dialogue("audio.wav"):
        if event["type"] == "text":
            print(event["content"], end="", flush=True)
        elif event["type"] == "audio":
            play(event["data"])

    with sdk:  # context manager auto-downloads models
        sdk.transcribe("audio.wav")
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Iterator

import numpy as np

from ._download import ModelDownloader, ModelSpec
from ._llm import LLMClient, LLMStreamChunk

logger = logging.getLogger("her_axera_sdk")


# ---------------------------------------------------------------------------
# Model specs (mirrors backend model_download_service specs)
# ---------------------------------------------------------------------------

_ASR_MODELS = [
    ModelSpec(
        key="asr_sensevoice",
        name="SenseVoice ASR",
        repo_id="AXERA-TECH/SenseVoice",
        allow_patterns=["sensevoice_ax650/*"],
        required_files=["sensevoice/sensevoice.axmodel"],
        size_hint="~120 MB",
    ),
    ModelSpec(
        key="asr_whisper_tiny",
        name="Whisper Tiny",
        repo_id="AXERA-TECH/Whisper",
        allow_patterns=["models-ax650/*"],
        required_files=["whisper/whisper_encoder_tiny.axmodel"],
        size_hint="~80 MB",
    ),
]

_TTS_MODELS = [
    ModelSpec(
        key="tts_kokoro_model",
        name="Kokoro TTS Model",
        repo_id="AXERA-TECH/kokoro.axera",
        allow_patterns=[
            "models/kokoro_part1_96.axmodel",
            "models/kokoro_part2_96.axmodel",
            "models/kokoro_part3_96.axmodel",
            "models/model4_har_sim.onnx",
        ],
        required_files=["kokoro/kokoro_part1_96.axmodel"],
        size_hint="~450 MB",
    ),
    ModelSpec(
        key="tts_kokoro_voices",
        name="Kokoro Voices",
        repo_id="AXERA-TECH/kokoro.axera",
        allow_patterns=["cpp/voices/*"],
        required_files=["kokoro/voices/voices.json"],
        size_hint="~5 MB",
    ),
]


# ---------------------------------------------------------------------------
# Main SDK class
# ---------------------------------------------------------------------------


class HerAxeraSDK:
    """Unified voice dialogue SDK.

    Parameters
    ----------
    asr_model_path:
        Root directory for ASR models. Models are stored in ``<path>/sensevoice/`` etc.
    tts_model_path:
        Root directory for TTS models. Models are stored in ``<path>/kokoro/`` etc.
    llm_api_base:
        OpenAI-compatible API base URL, e.g. ``https://api.deepseek.com``.
    llm_api_key:
        API key for the LLM provider.
    llm_model:
        Model name sent to the LLM API. Defaults to ``deepseek-chat``.
    asr_model_type:
        ASR model type passed to ``ax_asr.AX_ASR``. One of ``sensevoice``,
        ``whisper_tiny``, ``whisper_base``, ``whisper_small``, ``whisper_turbo``.
    tts_type:
        TTS engine type passed to ``ax_tts.AX_TTS``. Default ``KOKORO``.
    tts_voice:
        Default TTS voice name. Default ``af_heart``.
    tts_language:
        Default TTS language. Default ``en``.
    tts_sample_rate:
        Output sample rate. Default 24000.
    auto_download:
        If True, attempt to download missing models on first use.
    espeak_data_path:
        Path to espeak-ng-data directory (for TTS frontend).
    jieba_dict_path:
        Path to jieba dictionary directory (for Chinese TTS).
    """

    def __init__(
        self,
        *,
        asr_model_path: str | None = None,
        tts_model_path: str | None = None,
        llm_api_base: str = "https://api.deepseek.com",
        llm_api_key: str | None = None,
        llm_model: str = "deepseek-chat",
        asr_model_type: str = "sensevoice",
        tts_type: str = "KOKORO",
        tts_voice: str = "af_heart",
        tts_language: str = "en",
        tts_sample_rate: int = 24000,
        tts_fade_out: float = 0.3,
        auto_download: bool = True,
        espeak_data_path: str = "espeak-ng-data",
        jieba_dict_path: str = "dict",
    ) -> None:
        self._asr_model_path = asr_model_path or os.environ.get("AX_ASR_MODEL_PATH", "models-ax650")
        self._tts_model_path = tts_model_path or os.environ.get("AX_TTS_MODEL_PATH", "models-ax650")
        self._asr_model_type = asr_model_type
        self._tts_type = tts_type
        self._tts_voice = tts_voice
        self._tts_language = tts_language
        self._tts_sample_rate = tts_sample_rate
        self._tts_fade_out = tts_fade_out
        self._auto_download = auto_download

        # LLM
        self._llm = LLMClient(
            api_base=llm_api_base,
            api_key=llm_api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            model=llm_model,
        )

        # Lazy-initialized engine handles
        self._asr_handle = None
        self._tts_handle = None

        # Model downloader
        self._downloader = ModelDownloader(
            asr_root=Path(self._asr_model_path),
            tts_root=Path(self._tts_model_path),
            asr_specs=_ASR_MODELS,
            tts_specs=_TTS_MODELS,
            espeak_data_path=espeak_data_path,
            jieba_dict_path=jieba_dict_path,
        )

    # ---- Context manager ----

    def __enter__(self) -> HerAxeraSDK:
        if self._auto_download:
            self.download_models()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Release ASR/TTS engine handles."""
        if self._asr_handle is not None:
            try:
                self._asr_handle.close()
            except Exception:
                pass
            self._asr_handle = None
        if self._tts_handle is not None:
            try:
                self._tts_handle.close()
            except Exception:
                pass
            self._tts_handle = None

    def __del__(self) -> None:
        self.close()

    # ---- Model download ----

    def download_models(self) -> dict[str, str]:
        """Download missing ASR/TTS models. Returns {key: status} dict."""
        return self._downloader.download_all()

    def check_models(self) -> dict[str, bool]:
        """Check which models are present on disk. Returns {key: ready} dict."""
        return self._downloader.check_all()

    @property
    def downloader(self) -> ModelDownloader:
        return self._downloader

    # ---- Session Management ----

    def list_sessions(self, user_id: str | None = None) -> list[dict]:
        """List all saved sessions."""
        return self._llm.list_sessions(user_id=user_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages."""
        return self._llm.delete_session(session_id)

    # ---- ASR ----

    def transcribe(self, audio: str | bytes | np.ndarray, language: str = "zh") -> str:
        """Transcribe audio to text.

        Parameters
        ----------
        audio:
            File path (str), raw audio bytes, or numpy float32 PCM array.
        language:
            Language code. ``zh``, ``en``, ``auto``, etc.
        """
        asr = self._get_asr()
        if isinstance(audio, str):
            return asr.transcribe_file(audio, language=language)
        if isinstance(audio, np.ndarray):
            return asr.transcribe_pcm(audio, sample_rate=16000, language=language)
        # bytes → temp file
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(prefix="her_sdk_", suffix=suffix, delete=False) as f:
            f.write(audio)
            tmp_path = f.name
        try:
            return asr.transcribe_file(tmp_path, language=language)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def stream_transcribe_init(self) -> None:
        """Initialize streaming ASR. Call once before feeding chunks."""
        asr = self._get_asr()
        if hasattr(asr, 'stream_init'):
            asr.stream_init()

    def stream_transcribe_feed(self, pcm: np.ndarray, sample_rate: int = 16000) -> str | None:
        """Feed an audio chunk and return partial transcription (if changed)."""
        asr = self._get_asr()
        if not hasattr(asr, 'stream_feed'):
            raise RuntimeError("Streaming not supported by this ASR backend")
        asr.stream_feed(pcm, sample_rate)
        result = asr.stream_result()
        return result if result else None

    def stream_transcribe_result(self) -> str:
        """Get the current partial streaming result."""
        asr = self._get_asr()
        if hasattr(asr, 'stream_result'):
            return asr.stream_result()
        return ""

    def stream_transcribe_reset(self) -> None:
        """Reset streaming state for a new utterance."""
        asr = self._get_asr()
        if hasattr(asr, 'stream_reset'):
            asr.stream_reset()

    # ---- LLM ----

    def chat(self, message: str, *, system_prompt: str | None = None, temperature: float = 0.7) -> str:
        """Send a message to the LLM and get a reply.

        Parameters
        ----------
        message:
            User message text.
        system_prompt:
            Optional system prompt override.
        temperature:
            Sampling temperature (0-2).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return self._llm.chat(messages, temperature=temperature)

    def chat_stream(self, message: str, *, system_prompt: str | None = None, temperature: float = 0.7) -> Iterator[LLMStreamChunk]:
        """Stream LLM tokens for a message."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return self._llm.chat_stream(messages, temperature=temperature)

    # ---- TTS ----

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
        sample_rate: int | None = None,
    ) -> bytes:
        """Synthesize text to WAV audio bytes.

        Returns 16-bit PCM WAV bytes.
        """
        tts = self._get_tts()
        sr_val, audio_np = tts.synthesize(
            text,
            language=language or self._tts_language,
            voice=voice or self._tts_voice,
            speed=speed,
            fade_out=self._tts_fade_out,
            sample_rate=sample_rate or self._tts_sample_rate,
        )
        # Convert float32 numpy → WAV bytes
        pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr_val)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
    ) -> str:
        """Synthesize text to a WAV file. Returns the output path."""
        wav = self.synthesize(text, voice=voice, language=language, speed=speed)
        Path(output_path).write_bytes(wav)
        return output_path

    # ---- Full Pipeline ----

    def dialogue(
        self,
        audio: str | bytes | np.ndarray,
        *,
        language: str = "zh",
        system_prompt: str | None = None,
        tts_voice: str | None = None,
        tts_language: str | None = None,
        temperature: float = 0.7,
    ) -> Iterator[dict]:
        """Run the full ASR → LLM → TTS pipeline.

        Yields event dicts::

            {"type": "asr_text", "text": "..."}
            {"type": "llm_token", "token": "..."}      # per token
            {"type": "llm_sentence", "text": "..."}     # per sentence
            {"type": "tts_audio", "data": b"..."}       # WAV bytes
            {"type": "done", "total_ms": 1234}

        Example::

            for event in sdk.dialogue("audio.wav"):
                if event["type"] == "asr_text":
                    print(f"Heard: {event['text']}")
                elif event["type"] == "llm_token":
                    print(event["token"], end="", flush=True)
                elif event["type"] == "tts_audio":
                    with open("output.wav", "wb") as f:
                        f.write(event["data"])
        """
        import re

        t0 = time.perf_counter()

        # Step 1: ASR
        asr_text = self.transcribe(audio, language=language)
        yield {"type": "asr_text", "text": asr_text}

        # Step 2: LLM (streaming)
        full_response = ""
        buffer = ""
        sentence_re = re.compile(r"[。！？!?；;\n]")

        for chunk in self.chat_stream(asr_text, system_prompt=system_prompt, temperature=temperature):
            full_response += chunk.content
            buffer += chunk.content
            yield {"type": "llm_token", "token": chunk.content}

            # Split into sentences for TTS
            parts = []
            last = 0
            for m in sentence_re.finditer(buffer):
                end = m.end()
                s = buffer[last:end].strip()
                if s:
                    parts.append(s)
                last = end
            buffer = buffer[last:]

            for sentence in parts:
                yield {"type": "llm_sentence", "text": sentence}
                wav = self.synthesize(
                    sentence,
                    voice=tts_voice,
                    language=tts_language or self._tts_language,
                )
                yield {"type": "tts_audio", "data": wav}

        # Flush remaining buffer
        if buffer.strip():
            yield {"type": "llm_sentence", "text": buffer.strip()}
            wav = self.synthesize(
                buffer.strip(),
                voice=tts_voice,
                language=tts_language or self._tts_language,
            )
            yield {"type": "tts_audio", "data": wav}

        total_ms = int((time.perf_counter() - t0) * 1000)
        yield {"type": "done", "total_ms": total_ms, "asr_text": asr_text, "llm_response": full_response}

    # ---- Internals ----

    def _get_asr(self):
        if self._asr_handle is not None:
            return self._asr_handle

        if self._auto_download:
            self._downloader.download_asr()

        from ax_asr import AX_ASR

        self._asr_handle = AX_ASR(self._asr_model_type, self._asr_model_path)
        return self._asr_handle

    def _get_tts(self):
        if self._tts_handle is not None:
            return self._tts_handle

        if self._auto_download:
            self._downloader.download_tts()

        from ax_tts import AX_TTS

        self._tts_handle = AX_TTS(
            model_path=self._tts_model_path,
            espeak_data_path=self._downloader._espeak_data_path,
            jieba_dict_path=self._downloader._jieba_dict_path,
            tts_type=self._tts_type,
        )
        return self._tts_handle
