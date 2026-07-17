"""Caliana — analysis of plant calcium imaging data.

Reusable PyQt widgets around a central ``Session`` object, usable from a Jupyter
notebook or a standalone app.

Quick start (headless):

    import caliana
    s = caliana.Session.from_file("movie.tif", temporal_step=2)
    s.add_roi(center=(64, 80), size=5)
    s.extract_traces()
    s.compute_dff(n=30)            # F0 = mean of first 30 frames
"""
from __future__ import annotations

from .models import (
    BaselineMethod,
    ImportParams,
    LeafRegion,
    RegistrationMode,
    RigidTransform,
    ROI,
    ROIShape,
    Traces,
)
from .session import Session
from .timeline import Event, Timeline

__version__ = "0.1.0"

__all__ = [
    "Session",
    "ImportParams",
    "ROI",
    "ROIShape",
    "LeafRegion",
    "RegistrationMode",
    "RigidTransform",
    "Traces",
    "BaselineMethod",
    "Timeline",
    "Event",
    "__version__",
]
