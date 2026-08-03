"""Spatial-scale abstraction — the distance counterpart of ``timeline.py``.

The default model is pixels-only; set ``pixel_size`` (µm per pixel) to get a real
distance axis. Analyses stay in pixels and ask the SpatialScale for their axis
rather than assuming a calibration, so a calibrated recording reports µm (and
propagation µm/s) without changing them.

``pixel_size`` always describes the *loaded* stack: ``Session.load`` multiplies a
file's native pixel size by ``ImportParams.spatial_step``, so an ROI radius in
``Session`` pixels converts with a plain multiplication.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MICRONS = "µm"


@dataclass
class SpatialScale:
    """Owns the recording's spatial scale (µm per pixel)."""

    # µm per pixel of the loaded (downsampled) stack; None => pixels-only model.
    pixel_size: float | None = None

    @property
    def calibrated(self) -> bool:
        """Whether distances can be reported in µm rather than pixels."""
        return bool(self.pixel_size)

    def microns_for(self, pixels) -> np.ndarray | None:
        """Convert a pixel distance (scalar or array) to µm, or ``None`` if
        uncalibrated — mirroring ``Timeline.seconds_for``."""
        if not self.calibrated:
            return None
        return np.asarray(pixels, dtype=float) * self.pixel_size


def distance_units(pixel_size: float | None) -> tuple[float, str]:
    """``(factor, unit)`` taking a pixel distance to the best available units.

    Multiply a px value by ``factor`` to get ``unit`` — ``µm`` when calibrated,
    an identity conversion to ``px`` otherwise. Lets a caller label an axis and
    scale its data from one branch-free call.
    """
    return (pixel_size, MICRONS) if pixel_size else (1.0, "px")


def speed_units(pixel_size: float | None,
                frame_interval: float | None) -> tuple[float, str]:
    """``(factor, unit)`` taking px/frame to the best available speed units.

    The two calibrations are independent, so all four combinations are valid:
    ``µm/s``, ``µm/frame``, ``px/s``, ``px/frame``. Multiply a px/frame value by
    ``factor`` to get ``unit``.
    """
    dist_factor, dist_unit = distance_units(pixel_size)
    time_factor, time_unit = (frame_interval, "s") if frame_interval else (1.0, "frame")
    return dist_factor / time_factor, f"{dist_unit}/{time_unit}"
