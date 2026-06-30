"""Shared pyqtgraph helpers for the interactive widgets. SPEC.md §3 time axis."""
from __future__ import annotations

import pyqtgraph as pg


class FrameTimeAxis(pg.AxisItem):
    """Bottom axis that relabels frame ticks as seconds when calibrated.

    Plot data coordinates stay in frames, so ROIs, events, onsets, peaks and the
    baseline region need no rescaling; only the tick *labels* are converted via
    the Timeline's ``frame_interval`` (seconds per frame). ``frame_interval`` of
    None or 0 leaves the axis in frames.
    """
    frame_interval: float | None = None

    def tickStrings(self, values, scale, spacing):
        if self.frame_interval:
            return [f"{v * self.frame_interval:g}" for v in values]
        return super().tickStrings(values, scale, spacing)

    def set_frame_interval(self, interval: float | None) -> None:
        """Switch units and force a tick relabel on the next paint."""
        self.frame_interval = interval
        self.picture = None
        self.update()
