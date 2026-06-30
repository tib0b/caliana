"""The Session — single source of truth for one analysis. SPEC.md §2.1.

Widgets and notebook wrappers all read/write this object. The notebook-facing
methods (`preview`, `select_rois`) are thin blocking wrappers around embeddable
Qt widgets (SPEC §2.2); the heavy lifting lives in the io / registration / roi /
analysis / export modules so it is equally callable headless.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import analysis, export, io
from . import registration as registration_mod
from . import roi as roi_mod
from .models import (
    BaselineMethod,
    ImportParams,
    LeafRegion,
    RegistrationMode,
    RegistrationResult,
    ROI,
    ROIShape,
    Traces,
)
from .timeline import Timeline


class Session:
    def __init__(self) -> None:
        self.source = None                              # SourceInfo
        self.data: Optional[np.ndarray] = None          # [T, Y, X] raw downsampled
        self.registered_data: Optional[np.ndarray] = None  # stabilized stack (SPEC §2.1)
        self.timeline: Optional[Timeline] = None
        self.registration = RegistrationResult()
        self.leaf_regions: list[LeafRegion] = []
        self.rois: list[ROI] = []
        self.traces: Optional[Traces] = None
        self.analyses: dict = {}

    # ----------------------------------------------------------------- Stage I
    @classmethod
    def from_file(cls, path, **import_kwargs) -> "Session":
        return cls().load(path, **import_kwargs)

    def load(self, path, **import_kwargs) -> "Session":
        params = ImportParams(**import_kwargs)
        self.data, self.source = io.load_stack(path, params)
        self.timeline = Timeline(n_frames=len(self.data))
        return self

    def preview(self):
        """[notebook] Open the Stage I preview/import widget (blocking). SPEC §2.2."""
        from .widgets._qt import run_widget_blocking
        from .widgets.import_widget import ImportPreviewWidget

        return run_widget_blocking(lambda: ImportPreviewWidget(self))

    def max_projection(self) -> np.ndarray:
        """Normalized per-pixel max-over-time heatmap. SPEC §3 Stage I."""
        self._require_data()
        mip = self._working_stack().max(axis=0).astype(float)
        rng = float(mip.max() - mip.min())
        return (mip - mip.min()) / rng if rng else mip

    # ---------------------------------------------------------------- Stage II
    def add_leaf_region(self, bbox, label: str = "") -> LeafRegion:
        leaf = LeafRegion(bbox=tuple(bbox), label=label)
        self.leaf_regions.append(leaf)
        return leaf

    def register(self, mode=RegistrationMode.WHOLE_FRAME, reference: str = "mean") -> "Session":
        """Run motion correction in the chosen mode. SPEC §3 Stage II."""
        self._require_data()
        mode = RegistrationMode(mode)
        self.registered_data = None
        if mode == RegistrationMode.NONE:
            self.registration = RegistrationResult(mode=mode, reference=reference)
        elif mode == RegistrationMode.WHOLE_FRAME:
            self.registration = registration_mod.register_whole_frame(self.data, reference)
            self.registered_data = registration_mod.apply_transforms(
                self.data, self.registration.transforms
            )
        elif mode == RegistrationMode.PER_LEAF:
            if not self.leaf_regions:
                raise ValueError("per-leaf mode requires leaf_regions; draw boxes first")
            self.leaf_regions = registration_mod.register_per_leaf(
                self.data, self.leaf_regions, reference
            )
            self.registered_data = registration_mod.apply_per_leaf(
                self.data, self.leaf_regions
            )
            self.registration = RegistrationResult(mode=mode, reference=reference)
        self._invalidate_traces()
        return self

    def select_leaves(self):
        """[notebook] Open the Stage II leaf-box widget (blocking). SPEC §2.2.

        Draw one box per leaf for per-leaf registration; ROIs placed later
        auto-assign to the box containing them. Separate from `select_rois` so the
        movable leaf boxes don't interfere with ROI placement.
        """
        from .widgets._qt import run_widget_blocking
        from .widgets.leaf_widget import LeafSelectionWidget

        return run_widget_blocking(lambda: LeafSelectionWidget(self))

    def select_rois(self):
        """[notebook] Open the Stage II ROI-selection widget (blocking). SPEC §2.2."""
        from .widgets._qt import run_widget_blocking
        from .widgets.roi_widget import RoiSelectionWidget

        return run_widget_blocking(lambda: RoiSelectionWidget(self))

    def add_roi(self, center, size, shape=ROIShape.CIRCLE, label: str = "") -> ROI:
        roi = ROI(center=tuple(center), size=size, shape=ROIShape(shape), label=label)
        if self.registration.mode == RegistrationMode.PER_LEAF:
            roi.leaf_region = roi_mod.assign_roi_to_leaf(roi, self.leaf_regions)
        self.rois.append(roi)
        self._invalidate_traces()
        return roi

    def add_polygon_roi(self, vertices, label: str = "") -> ROI:
        """Add a free-hand polygon ROI (e.g. a whole leaf). SPEC §3 Stage II.

        ``vertices`` is a list of (y, x) outline points; the ROI centre is set to
        the polygon centroid (used for leaf assignment and propagation).
        """
        verts = [tuple(v) for v in vertices]
        roi = ROI(
            center=roi_mod.polygon_centroid(verts), size=0.0,
            shape=ROIShape.POLYGON, label=label, vertices=verts,
        )
        if self.registration.mode == RegistrationMode.PER_LEAF:
            roi.leaf_region = roi_mod.assign_roi_to_leaf(roi, self.leaf_regions)
        self.rois.append(roi)
        self._invalidate_traces()
        return roi

    def extract_traces(self) -> Traces:
        """Mean-intensity raw F trace per ROI. SPEC §3 Stage II."""
        self._require_data()
        # _working_stack() is the stabilized stack: whole-frame registers the
        # whole image; per-leaf composites each leaf box's stabilized sub-stack,
        # so ROIs inside a box already sample stabilized tissue (SPEC §3).
        self.traces = roi_mod.extract_all_traces(self._working_stack(), self.rois)
        return self.traces

    # --------------------------------------------------------------- Stage III
    def compute_dff(self, method=BaselineMethod.FIRST_N, n=None, region=None) -> Traces:
        if self.traces is None:
            self.extract_traces()
        return analysis.compute_dff(self.traces, method=BaselineMethod(method), n=n, region=region)

    def detect_peaks(self, use_dff: bool = True, **kwargs) -> list[dict]:
        if self.traces is None:
            self.extract_traces()
        signal = self.traces.dff if (use_dff and self.traces.dff is not None) else self.traces.raw
        results = [analysis.detect_peaks(signal[i], **kwargs) for i in range(len(signal))]
        self.analyses["peaks"] = results
        return results

    def cross_roi_propagation(self, **kwargs):
        if self.traces is None:
            self.extract_traces()
        result = analysis.cross_roi_propagation(self.traces, self.rois, **kwargs)
        self.analyses["propagation"] = result
        return result

    def apply(self, func: Callable):
        """Run a custom callable ``f(traces, data) -> result``. SPEC §3."""
        if self.traces is None:
            self.extract_traces()
        return analysis.apply_custom(func, self.traces, self.data)

    def analyze(self):
        """[notebook] Open the Stage III analysis widget (blocking). SPEC §2.2."""
        from .widgets._qt import run_widget_blocking
        from .widgets.analysis_widget import AnalysisWidget

        return run_widget_blocking(lambda: AnalysisWidget(self))

    # ------------------------------------------------------------------ Export
    def provenance(self) -> dict:
        """Full parameter record for reproducibility. SPEC §4."""
        src = self.source
        events = self.timeline.events if self.timeline else []
        return {
            "source": None if src is None else {
                "path": str(src.path),
                "import_params": vars(src.import_params),
            },
            "registration": {
                "mode": self.registration.mode.value,
                "reference": self.registration.reference,
                "leaf_regions": [
                    {"bbox": list(lr.bbox), "label": lr.label,
                     "low_confidence_frames": lr.low_confidence_frames}
                    for lr in self.leaf_regions
                ],
            },
            "rois": [
                {"center": list(r.center), "size": r.size, "shape": r.shape.value,
                 "label": r.label, "leaf_region": r.leaf_region,
                 "vertices": None if r.vertices is None else [list(v) for v in r.vertices]}
                for r in self.rois
            ],
            "events": [{"frame": e.frame, "label": e.label} for e in events],
            "analyses": sorted(self.analyses.keys()),
        }

    def export_traces(self, path) -> None:
        export.traces_to_csv(self.traces, path, self.timeline)

    def export_stack(self, path) -> None:
        """Export the working (stabilized if registered) stack. SPEC §4."""
        export.stack_to_tiff(self._working_stack(), path)

    def export_provenance(self, path) -> None:
        export.write_provenance(self, path)

    # ---------------------------------------------------------------- Figures
    # Paper-grade static figures (matplotlib). Thin wrappers around figures.py
    # so they're callable from the notebook; all logic lives in the module.
    def figure_traces(self, **kw):
        from . import figures
        return figures.plot_traces(self, **kw)

    def figure_propagation(self, **kw):
        from . import figures
        return figures.plot_propagation(self, **kw)

    def figure_roi_overlay(self, **kw):
        from . import figures
        return figures.plot_roi_overlay(self, **kw)

    def figure_imaging_electrode(self, aux_time, aux_signal, **kw):
        from . import figures
        return figures.plot_imaging_electrode(self, aux_time, aux_signal, **kw)

    # ----------------------------------------------------------------- helpers
    def _require_data(self) -> None:
        if self.data is None:
            raise RuntimeError("No data loaded; call load()/from_file() first.")

    def _working_stack(self) -> np.ndarray:
        """The stack ROIs/heatmaps act on: stabilized if registered, else raw."""
        return self.registered_data if self.registered_data is not None else self.data

    def _invalidate_traces(self) -> None:
        """Recomputation is explicit: upstream changes drop stale traces. SPEC §2.1."""
        self.traces = None
        self.analyses.clear()
