"""Opus audio encoding/decoding utilities.

Requires the ``opuslib`` package (``pip install opuslib``) and the system
``libopus`` library.  All functions are optional – callers should check
:func:`opus_available` before attempting to encode/decode.
"""

from __future__ import annotations

import logging
import struct
from io import BytesIO

logger = logging.getLogger(__name__)

_opus_ok = False
try:
    import opuslib  # type: ignore[import-untyped]

    _opus_ok = True
except Exception:
    opuslib = None  # type: ignore[assignment]


def opus_available() -> bool:
    return _opus_ok


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
