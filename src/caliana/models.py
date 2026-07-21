"""Core data structures for Caliana.

Plain dataclasses/enums with no Qt or I/O dependencies, so they can be imported
anywhere (notebook, app, tests) and serialized for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Import / source (Stage I)
# --------------------------------------------------------------------------- #
@dataclass
class ImportParams:
    """Downsample-on-load parameters. Units are pixels and frame indices only.

    Frames ``[start, end)`` are kept, then temporal/spatial downsampling applied.
    """
    start: int = 0
    end: Optional[int] = None              # exclusive; None => until the end
    temporal_step: int = 1                 # average every N frames (1 = no averaging)
    spatial_step: int = 1                  # keep every Nth pixel per axis (1 = full res)
    spatial_window: Optional[tuple[int, int, int, int]] = None  # crop (y0, y1, x0, x1); None = full
    channel: int = 0                       # which channel to keep (single-channel model)


@dataclass
class SourceInfo:
    """Where the data came from and how it was imported (for provenance)."""
    path: Path
    import_params: ImportParams = field(default_factory=ImportParams)
    metadata: dict = field(default_factory=dict)   # raw reader metadata; units ignored


# --------------------------------------------------------------------------- #
# ROIs (Stage II)
# --------------------------------------------------------------------------- #
class ROIShape(str, Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    POLYGON = "polygon"                     # free-hand outline (e.g. a whole leaf)


@dataclass
class ROI:
    """A region of interest: circle, square, or free-hand polygon.

    For CIRCLE/SQUARE the mask is ``center`` + ``size`` (radius / half-side). For
    POLYGON it is ``vertices``; ``size`` is ignored and ``center`` holds the polygon
    centroid. ROIs may overlap.
    """
    center: tuple[float, float]            # (y, x) in pixels
    size: float                            # circle radius / square half-side, px
    shape: ROIShape = ROIShape.CIRCLE
    label: str = ""
    leaf_region: Optional[int] = None      # index into Session.leaf_regions (per-leaf mode)
    vertices: Optional[list[tuple[float, float]]] = None  # (y, x) polygon outline (POLYGON only)


# --------------------------------------------------------------------------- #
# Registration (Stage II)
# --------------------------------------------------------------------------- #
class RegistrationMode(str, Enum):
    NONE = "none"                          # ROIs placed on the raw stack
    WHOLE_FRAME = "whole-frame"            # one rigid transform per frame
    PER_LEAF = "per-leaf"                  # one box per leaf, registered independently


@dataclass
class RigidTransform:
    """Per-frame motion transform (raw → reference).

    ``dy``, ``dx``, ``theta`` (degrees) are the rigid summary: translation plus
    rotation about the region centre. ``matrix``, when set, is the full 3x3
    homogeneous transform (acting on ``(x, y, 1)``) and is authoritative — it
    carries scale/shear from ``scaled_rotation``/``affine`` models through warping
    and ROI motion. ``matrix=None`` means a pure rigid body given by the scalars.
    """
    dy: float = 0.0
    dx: float = 0.0
    theta: float = 0.0
    matrix: Optional[np.ndarray] = None    # full 3x3 (raw→reference); overrides scalars when set


@dataclass
class LeafRegion:
    """A user-drawn leaf box with its own independent registration.

    Draw boxes generously: tissue that drifts outside its box cannot be stabilized
    (such frames are flagged in ``low_confidence_frames``).
    """
    bbox: tuple[int, int, int, int]                       # (y0, y1, x0, x1)
    label: str = ""
    transforms: list[RigidTransform] = field(default_factory=list)   # one per frame
    reference: Optional[np.ndarray] = None                # mean/first of the sub-stack
    low_confidence_frames: list[int] = field(default_factory=list)   # drift-out flag


@dataclass
class RegistrationResult:
    """Whole-session registration state."""
    mode: RegistrationMode = RegistrationMode.NONE
    reference: str = "mean"                               # "mean" | "first" | "previous"
    transforms: list[RigidTransform] = field(default_factory=list)   # whole-frame mode


# --------------------------------------------------------------------------- #
# Traces & analysis (Stage III)
# --------------------------------------------------------------------------- #
# Default ΔF/F baseline window for `Traces.dff`'s auto-computed default value.
DEFAULT_DFF_BASELINE_FRAMES = 10


@dataclass
class Traces:
    """Per-ROI fluorescence traces (raw F and ΔF/F).

    ``dff`` defaults to ΔF/F with the first ``DEFAULT_DFF_BASELINE_FRAMES`` frames
    as baseline, computed automatically from ``raw`` (``None`` only when ``raw`` is
    empty, e.g. no ROIs). Pass ``dff`` explicitly, or call ``analysis.compute_dff``
    afterwards, to use a different baseline.
    """
    raw: np.ndarray                        # [n_roi, T] mean intensity inside each ROI
    dff: Optional[np.ndarray] = None       # [n_roi, T] (F - F0)/F0
    labels: list[str] = field(default_factory=list)
    # Gaussian-smoothed copy of `dff` (see `analysis.smooth_traces`), kept separate
    # so `dff` is never overwritten. None until smoothed.
    smoothed: Optional[np.ndarray] = None            # [n_roi, T]
    smoothed_sigma: Optional[float] = None           # Gaussian std dev used, in frames

    def __post_init__(self):
        if self.dff is None and self.raw.size:
            n = min(DEFAULT_DFF_BASELINE_FRAMES, self.raw.shape[1])
            f0 = self.raw[:, :n].mean(axis=1, keepdims=True)
            self.dff = (self.raw - f0) / f0


class BaselineMethod(str, Enum):
    FIRST_N = "first_n"                    # F0 = mean of first N frames
    REGION = "region"                      # F0 = mean over a user-selected [start, end) window
