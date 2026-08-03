"""Helpers shared by the interactive widgets: session-to-plot adapters (time axis,
display normalization) and pyqtgraph subclasses. SPEC.md §3 time axis."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg


def frame_interval(session) -> float | None:
    """The session's seconds-per-frame, or None when the axis is frames-only.

    Collapses "no Timeline" and "uncalibrated Timeline" into the single ``None``
    every widget actually branches on.
    """
    tl = session.timeline
    return tl.frame_interval if (tl is not None and tl.frame_interval) else None


def pixel_size(session) -> float | None:
    """The session's µm-per-pixel, or None when distances are pixels-only.

    The spatial counterpart of ``frame_interval``: collapses "no SpatialScale"
    and "uncalibrated" into the single ``None`` the widgets branch on.
    """
    space = getattr(session, "space", None)
    return space.pixel_size if (space is not None and space.pixel_size) else None


def dff0(raw) -> np.ndarray:
    """Display-only normalization of ``[n_roi, T]`` raw F to (F − F[0]) / F[0].

    Makes responses comparable across ROIs regardless of brightness in the live
    previews; rows whose first frame is 0 come back flat rather than inf/NaN. The
    stored and exported traces stay raw mean intensity (SPEC §3).
    """
    raw = np.asarray(raw, dtype=float)
    f0 = raw[:, :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f0 != 0, (raw - f0) / f0, 0.0)


class FrameTimeAxis(pg.AxisItem):
    """Bottom axis that relabels frame ticks as seconds when calibrated.

    Plot data coordinates stay in frames, so ROIs, events, onsets and the
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


class SquarePlotWidget(pg.PlotWidget):
    """A PlotWidget kept square on screen: its height follows its width.

    Each axis still autoscales to its own data (the *data* aspect is not locked),
    so a scatter whose x and y ranges differ by orders of magnitude renders in a
    square box instead of a thin strip.
    """

    def resizeEvent(self, event):
        super().resizeEvent(event)
        side = self.width()
        if side > 0 and self.maximumHeight() != side:
            self.setFixedHeight(side)
