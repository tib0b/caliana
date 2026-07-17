"""Time-axis abstraction + event markers.

The current model is frames-only; set ``frame_interval`` (seconds per frame) to
get a real-time axis. Analyses should ask the Timeline for their axis rather than
assuming frames, so a calibrated seconds axis (e.g. for electrode co-analysis)
works without changing them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Event:
    """A stimulus/event marker at a frame index (e.g. wounding, touch)."""
    frame: int
    label: str = ""


class Timeline:
    """Owns the recording's time axis and its event markers."""

    def __init__(self, n_frames: int, frame_interval: Optional[float] = None):
        self.n_frames = n_frames
        # seconds per frame; None => frames-only model (the current default).
        self.frame_interval = frame_interval
        self.events: list[Event] = []

    @property
    def frames(self) -> np.ndarray:
        return np.arange(self.n_frames)

    def seconds(self) -> Optional[np.ndarray]:
        """Seconds for every frame if ``frame_interval`` is set, else ``None``.

        Spans the whole recording; for a (possibly cropped) trace window use
        ``seconds_for`` with the trace's frame indices (``Session.trace_frames``).
        """
        if self.frame_interval is None:
            return None
        return self.frames * self.frame_interval

    def seconds_for(self, frames) -> Optional[np.ndarray]:
        """Seconds for the given (original) frame indices, or ``None`` if uncalibrated.

        Lets a cropped trace window report true recording time: column ``c`` of a
        crop starting at frame ``f0`` is frame ``f0 + c``.
        """
        if self.frame_interval is None:
            return None
        return np.asarray(frames, dtype=float) * self.frame_interval

    def add_event(self, frame: int, label: str = "") -> Event:
        ev = Event(frame=frame, label=label)
        self.events.append(ev)
        return ev
