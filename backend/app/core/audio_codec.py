"""Audio encoding/decoding utilities.

Requires the ``opuslib`` package (``pip install opuslib``) and the system
``libopus`` library.  All functions are optional – callers should check
:func:`opus_available` before attempting to encode/decode.
"""

from __future__ import annotations

import logging
import struct
from io import BytesIO
import wave

logger = logging.getLogger(__name__)

_opus_ok = False
try:
    import opuslib  # type: ignore[import-untyped]

    _opus_ok = True
except Exception:
    opuslib = None  # type: ignore[assignment]


def opus_available() -> bool:
    return _opus_ok


def wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int] | None:
    """Decode a mono/stereo 16-bit PCM WAV into raw PCM16.

    Returns ``(pcm, sample_rate)`` or ``None`` when the payload is not a
    readable 16-bit WAV.  Stereo channels are mixed down to mono.
    """
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError):
        return None
    if sampwidth != 2 or channels not in (1, 2):
        return None
    if channels == 1:
        return frames, sample_rate
    n = len(frames) // 4
    if n == 0:
        return b"", sample_rate
    import array

    stereo = array.array("h")
    stereo.frombytes(frames[: n * 4])
    mono = array.array("h")
    for i in range(0, n, 1):
        left = stereo[2 * i]
        right = stereo[2 * i + 1]
        mono.append((left + right) // 2)
    return mono.tobytes(), sample_rate


def pcm16_to_wav(pcm: bytes, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 into a WAV container."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()

def pcm_to_opus(
    pcm: bytes,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    frame_ms: int = 20,
    bitrate: int = 24000,
) -> bytes:
    """Encode raw PCM (16-bit signed LE) to a sequence of Opus frames.

    Returns a binary blob: each frame is prefixed by a 2-byte big-endian
    length header so the decoder can split them back apart.
    """
    if not _opus_ok:
        raise RuntimeError("opuslib is not installed")

    encoder = opuslib.Encoder(sample_rate, channels, opuslib.APPLICATION_VOIP)
    encoder.bitrate = bitrate

    frame_size = int(sample_rate * frame_ms / 1000)  # samples per frame
    frame_bytes = frame_size * channels * 2  # 16-bit = 2 bytes per sample

    output = BytesIO()
    offset = 0
    while offset + frame_bytes <= len(pcm):
        frame_pcm = pcm[offset : offset + frame_bytes]
        encoded = encoder.encode(frame_pcm, frame_size)
        output.write(struct.pack(">H", len(encoded)))
        output.write(encoded)
        offset += frame_bytes

    # Encode remaining samples if any (pad with silence).
    if offset < len(pcm):
        remaining = pcm[offset:]
        padded = remaining + b"\x00" * (frame_bytes - len(remaining))
        encoded = encoder.encode(padded, frame_size)
        output.write(struct.pack(">H", len(encoded)))
        output.write(encoded)

    return output.getvalue()


def opus_to_pcm(
    data: bytes,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    frame_ms: int = 20,
) -> bytes:
    """Decode a blob of length-prefixed Opus frames back to PCM."""
    if not _opus_ok:
        raise RuntimeError("opuslib is not installed")

    decoder = opuslib.Decoder(sample_rate, channels)
    frame_size = int(sample_rate * frame_ms / 1000)

    output = BytesIO()
    offset = 0
    while offset + 2 <= len(data):
        (frame_len,) = struct.unpack(">H", data[offset : offset + 2])
        offset += 2
        if offset + frame_len > len(data):
            break
        frame_data = data[offset : offset + frame_len]
        pcm_frame = decoder.decode(frame_data, frame_size)
        output.write(pcm_frame)
        offset += frame_len

    return output.getvalue()
