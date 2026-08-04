"""Generation-counter cancellation for dialogue turns.

Adapted from huggingface/speech-to-speech's CancelScope: instead of relying
only on asyncio task cancellation, each turn captures a generation number at
start. Output events are tagged with that generation, and the WebSocket send
path drops any event whose generation no longer matches the connection's
current generation. This prevents stale output (e.g. a detached partial-ASR
task, or an event that was already queued) from being delivered after a
barge-in.

Thread safety: the route runs on a single event loop, so no lock is needed.
"""

from __future__ import annotations


class CancelScope:
    """Monotonic generation counter for turn cancellation."""

    def __init__(self, initial: int = 0) -> None:
        self._gen = initial

    @property
    def generation(self) -> int:
        return self._gen

    def cancel(self) -> int:
        """Invalidate all previously captured generations and return the new one."""
        self._gen += 1
        return self._gen

    def is_stale(self, generation: int) -> bool:
        """True when *generation* belongs to a cancelled (superseded) turn."""
        return generation != self._gen
