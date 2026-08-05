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
    ROI,
    BaselineMethod,
    FileScale,
    ImportParams,
    LeafRegion,
    RegistrationMode,
    RigidTransform,
    ROIShape,
    Traces,
)
from .session import Session
from .space import SpatialScale
from .timeline import Event, Timeline

__version__ = "0.3.0"


def open_session(path=None) -> Session:
    """[notebook] Open the file/import widget (blocking) → the loaded ``Session``.

    The interactive counterpart of ``Session.from_file``: pick the recording and
    the downsample-on-load parameters in a window, press Load, close it. The
    Session comes back loaded (or empty, if the window was closed without
    loading). ``path`` pre-fills the file box.

        s = caliana.open_session()          # or open_session("movie.nd2")
    """
    from .widgets._qt import run_widget_blocking
    from .widgets.source_widget import SourceWidget

    def factory():
        widget = SourceWidget(Session())
        if path is not None:
            widget.set_path(path)
        return widget

    return run_widget_blocking(factory)


__all__ = [
    "ROI",
    "BaselineMethod",
    "Event",
    "FileScale",
    "ImportParams",
    "LeafRegion",
    "ROIShape",
    "RegistrationMode",
    "RigidTransform",
    "Session",
    "SpatialScale",
    "Timeline",
    "Traces",
    "__version__",
    "open_session",
]
