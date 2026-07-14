"""Core data structures for Caliana. See SPEC.md §2.1 (the Session model).

These are plain dataclasses/enums with no Qt or I/O dependencies, so they can be
imported anywhere (notebook, app, tests) and serialized for provenance (SPEC §4).
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
    """Downsample-on-load parameters. SPEC.md §3 Stage I.

    Units are pixels and frame indices only — no physical calibration.
    """
    start: int = 0
    end: Optional[int] = None              # exclusive; None => until the end
    temporal_step: int = 1                 # average every N frames (1 = off)
    spatial_step: int = 1                  # spatial stride / binning (1 = off)
    spatial_window: Optional[tuple[int, int, int, int]] = None  # (y0, y1, x0, x1)
    channel: int = 0                       # single-channel model: which channel to keep


@dataclass
class SourceInfo:
    """Where the data came from + how it was imported (for provenance, SPEC §4)."""
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
    """Circle/square or free-hand polygon ROI. SPEC.md §3 Stage II.

    Overlap between ROIs is permitted; no disjointness is enforced.

    For CIRCLE/SQUARE the mask is defined by ``center`` + ``size``. For POLYGON
    the mask is defined by ``vertices`` (a free-hand outline); ``center`` is then
    the polygon centroid (used for leaf assignment and propagation) and ``size``
    is unused.
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
    """Per-frame motion transform (raw → reference). SPEC.md §3 Stage II.

    ``dy``/``dx``/``theta`` (theta in degrees) are the rigid summary — translation
    plus rotation about the region centre — and are what the drift-out heuristic
    and any human-readable readout use. When the registration model estimates more
    than a rigid body (``scaled_rotation``/``affine``), ``matrix`` holds the full
    3x3 homogeneous transform including scale/shear; it is the authoritative form
    used to warp frames and move ROIs, so that scale/shear carries through to the
    stabilized output. When ``matrix`` is ``None`` the transform is exactly the
    rigid body described by the scalar fields.
    """
    dy: float = 0.0
    dx: float = 0.0
    theta: float = 0.0
    matrix: Optional[np.ndarray] = None    # full 3x3 (raw→reference); overrides scalars when set


@dataclass
class LeafRegion:
    """A user-drawn leaf box with its own independent registration. SPEC.md §3.

    Draw boxes generously: tissue that drifts outside its box across the
    recording cannot be stabilized (see SPEC "Drift-out-of-box handling").
    """
    bbox: tuple[int, int, int, int]                       # (y0, y1, x0, x1)
    label: str = ""
    transforms: list[RigidTransform] = field(default_factory=list)   # one per frame
    reference: Optional[np.ndarray] = None                # mean/first of the sub-stack
    low_confidence_frames: list[int] = field(default_factory=list)   # drift-out flag


@dataclass
class RegistrationResult:
    """Whole-session registration state. SPEC.md §2.1 / §3 Stage II."""
    mode: RegistrationMode = RegistrationMode.NONE
    reference: str = "mean"                               # "mean" | "first"
    transforms: list[RigidTransform] = field(default_factory=list)   # whole-frame mode


# --------------------------------------------------------------------------- #
# Traces & analysis (Stage III)
# --------------------------------------------------------------------------- #
@dataclass
class Traces:
    """Per-ROI fluorescence traces. SPEC.md §3 (trace extraction / ΔF/F)."""
    raw: np.ndarray                        # [n_roi, T] mean intensity inside each ROI
    dff: Optional[np.ndarray] = None       # [n_roi, T] (F - F0)/F0
    labels: list[str] = field(default_factory=list)


class BaselineMethod(str, Enum):
    FIRST_N = "first_n"                    # F0 = mean of first N frames
    REGION = "region"                      # F0 = mean over a user-selected [start, end) window
