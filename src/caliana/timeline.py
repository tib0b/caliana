"""Time-axis abstraction + event markers. SPEC.md §3 Stage III.

The time axis is kept isolated here, on purpose: the current model is
frames-only, but a real (seconds) axis will be needed later to co-analyze
electrode recordings (SPEC.md §6, "Forward compatibility: electrode data").
Analyses should ask the Timeline for their axis rather than assuming frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Event:
    """Optional stimulus/event marker (e.g. wounding, touch). SPEC.md §3."""
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
        """Real-time axis for the whole recording if calibrated, else None.

        Frames-only model (SPEC §3) returns None. Note this spans every frame;
        for the (possibly cropped) trace window use ``seconds_for`` with the
        trace's frame indices (see ``Session.trace_frames``).

        Electrode co-analysis (SPEC §6) will require this to be populated and a
        frame<->time mapping defined.
        """
        if self.frame_interval is None:
            return None
        return self.frames * self.frame_interval

    def seconds_for(self, frames) -> Optional[np.ndarray]:
        """Seconds for the given (original) frame indices, or None if uncalibrated.

        Used so a cropped trace window still reports the true recording time:
        column ``c`` of a crop starting at frame ``f0`` is frame ``f0 + c``.
        """
        if self.frame_interval is None:
            return None
        return np.asarray(frames, dtype=float) * self.frame_interval

    def add_event(self, frame: int, label: str = "") -> Event:
        ev = Event(frame=frame, label=label)
        self.events.append(ev)
        return ev
