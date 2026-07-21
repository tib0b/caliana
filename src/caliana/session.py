"""The Session — single source of truth for one analysis.

Widgets and notebook wrappers all read/write this object. Methods marked
``[notebook]`` open a blocking Qt widget; every other step is headless and callable
without a GUI. The actual work lives in the io / registration / roi / analysis /
export modules, which ``Session`` orchestrates.

Typical order: ``load`` → ``add_leaf_region`` / ``register`` → ``add_roi`` →
``extract_traces`` → ``compute_dff`` → analysis / ``export_*``.
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
        self.registered_data: Optional[np.ndarray] = None  # stabilized stack; None until register(apply=True)
        self.timeline: Optional[Timeline] = None
        self.registration = RegistrationResult()
        # When True, traces are extracted by moving each ROI with its leaf's
        # per-frame transform over the RAW stack (ROI-follows-tissue), instead of
        # sampling a warped stack. Set by register(apply=False).
        self.track_motion: bool = False
        self.leaf_regions: list[LeafRegion] = []
        self.rois: list[ROI] = []
        # [start, end) frame window traces are cropped to before analysis; None =>
        # the whole recording. In original (uncropped) frame indices.
        self.crop_window: Optional[tuple[int, int]] = None
        self.traces: Optional[Traces] = None
        self.analyses: dict = {}

    # ----------------------------------------------------------------- Stage I
    @classmethod
    def from_file(cls, path, **import_kwargs) -> "Session":
        """New Session with ``path`` loaded. Shorthand for ``Session().load(...)``."""
        return cls().load(path, **import_kwargs)

    def load(self, path, **import_kwargs) -> "Session":
        """Load a ``.tif``/``.tiff``/``.nd2`` stack, applying downsample-on-load.

        ``import_kwargs`` are ``ImportParams`` fields (``start``, ``end``,
        ``temporal_step``, ``spatial_step``, ``spatial_window``, ``channel``).
        """
        params = ImportParams(**import_kwargs)
        self.data, self.source = io.load_stack(path, params)
        self.timeline = Timeline(n_frames=len(self.data))
        return self

    def preview(self):
        """[notebook] Open the preview/import widget (blocking)."""
        from .widgets._qt import run_widget_blocking
        from .widgets.import_widget import ImportPreviewWidget

        return run_widget_blocking(lambda: ImportPreviewWidget(self))

    def max_projection(self) -> np.ndarray:
        """Per-pixel max-over-time image of the working stack, normalized to [0, 1]."""
        self._require_data()
        mip = self._working_stack().max(axis=0).astype(float)
        rng = float(mip.max() - mip.min())
        return (mip - mip.min()) / rng if rng else mip

    # ---------------------------------------------------------------- Stage II
    def add_leaf_region(self, bbox, label: str = "") -> LeafRegion:
        """Register a leaf box ``bbox = (y0, y1, x0, x1)`` for per-leaf mode."""
        leaf = LeafRegion(bbox=tuple(bbox), label=label)
        self.leaf_regions.append(leaf)
        return leaf

    def register(
        self,
        mode=RegistrationMode.WHOLE_FRAME,
        reference: str = "mean",
        mask: bool = False,
        apply: bool = True,
        transformation: str = "affine",
    ) -> "Session":
        """Run motion correction in the chosen mode.

        mode: ``RegistrationMode.NONE`` (ROIs on the raw stack), ``WHOLE_FRAME``
            (one transform per frame), or ``PER_LEAF`` (each leaf box registered
            independently; requires ``add_leaf_region`` first).
        reference: ``"mean"`` (default), ``"first"``, or ``"previous"``.
        transformation: pystackreg model — ``"translation"``, ``"rigid_body"``,
            ``"scaled_rotation"``, or ``"affine"`` (default). Scale/shear is kept
            and carries through to the stabilized stack and ROI tracking.
        mask: estimate on the tissue silhouette, so registration tracks the dim
            leaf rather than the static bright background (recommended here).
        apply: ``True`` warps the stack into ``registered_data`` and samples static
            ROIs on it; ``False`` keeps raw pixels and moves each ROI with its
            tissue at extraction time (``track_motion``), avoiding interpolation
            bias in ΔF/F — preferable on dim, low-SNR data.
        """
        self._require_data()
        mode = RegistrationMode(mode)
        self.registered_data = None
        self.track_motion = False
        if mode == RegistrationMode.NONE:
            self.registration = RegistrationResult(mode=mode, reference=reference)
        elif mode == RegistrationMode.WHOLE_FRAME:
            self.registration = registration_mod.register_whole_frame(
                self.data, reference, mask=mask, transformation=transformation
            )
            if apply:
                self.registered_data = registration_mod.apply_transforms(
                    self.data, self.registration.transforms
                )
            else:
                self.track_motion = True
        elif mode == RegistrationMode.PER_LEAF:
            if not self.leaf_regions:
                raise ValueError("per-leaf mode requires leaf_regions; draw boxes first")
            self.leaf_regions = registration_mod.register_per_leaf(
                self.data, self.leaf_regions, reference, mask=mask,
                transformation=transformation,
            )
            if apply:
                self.registered_data = registration_mod.apply_per_leaf(
                    self.data, self.leaf_regions
                )
            else:
                self.track_motion = True
            self.registration = RegistrationResult(mode=mode, reference=reference)
        self._invalidate_traces()
        return self

    def select_leaves(self):
        """[notebook] Open the leaf-box widget (blocking).

        Draw one box per leaf for per-leaf registration; ROIs added later
        auto-assign to the box containing them. Separate from ``select_rois`` so the
        movable boxes don't interfere with ROI placement.
        """
        from .widgets._qt import run_widget_blocking
        from .widgets.leaf_widget import LeafSelectionWidget

        return run_widget_blocking(lambda: LeafSelectionWidget(self))

    def select_rois(self):
        """[notebook] Open the ROI-selection widget (blocking)."""
        from .widgets._qt import run_widget_blocking
        from .widgets.roi_widget import RoiSelectionWidget

        return run_widget_blocking(lambda: RoiSelectionWidget(self))

    def add_roi(self, center, size, shape=ROIShape.CIRCLE, label: str = "") -> ROI:
        """Add a circle or square ROI.

        center: ``(y, x)`` in pixels. size: radius (circle) or half-side (square).
        shape: ``ROIShape.CIRCLE`` (default) or ``ROIShape.SQUARE`` (or the strings
        ``"circle"``/``"square"``). In per-leaf mode the ROI auto-assigns to its
        containing leaf box.
        """
        roi = ROI(center=tuple(center), size=size, shape=ROIShape(shape), label=label)
        if self.registration.mode == RegistrationMode.PER_LEAF:
            roi.leaf_region = roi_mod.assign_roi_to_leaf(roi, self.leaf_regions)
        self.rois.append(roi)
        self._invalidate_traces()
        return roi

    def add_polygon_roi(self, vertices, label: str = "") -> ROI:
        """Add a free-hand polygon ROI (e.g. a whole leaf).

        ``vertices`` is a list of ``(y, x)`` outline points; the ROI centre is set
        to the polygon centroid. In per-leaf mode it auto-assigns to its containing
        leaf box.
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
        """Extract (and store) the mean-intensity raw F trace per ROI.

        Uses the stabilized stack, or moves each ROI with its tissue when
        registered with ``apply=False`` (``track_motion``). A ``crop_window`` (see
        ``set_crop``/``crop_traces``) restricts the traces to that frame interval.
        """
        self._require_data()
        # Two extraction paths:
        #  - track_motion: keep raw pixels, move each ROI with its tissue per frame
        #    (no resampling of the measured intensities).
        #  - otherwise: _working_stack() is the stabilized stack (whole-frame warp,
        #    or per-leaf composite of stabilized sub-stacks) and ROIs are static.
        stack = self._working_stack()
        start = 0
        if self.crop_window is not None:
            start, end = self.crop_window
            stack = stack[start:end]
        if self.track_motion and self._has_transforms():
            self.traces = self._extract_tracked(stack, start)
        else:
            self.traces = roi_mod.extract_all_traces(stack, self.rois)
        return self.traces

    def _has_transforms(self) -> bool:
        """Whether registration produced per-frame transforms to track ROIs with."""
        reg = self.registration
        if reg.mode == RegistrationMode.WHOLE_FRAME:
            return bool(reg.transforms)
        if reg.mode == RegistrationMode.PER_LEAF:
            return any(leaf.transforms for leaf in self.leaf_regions)
        return False

    def _roi_transform_series(self, roi: ROI):
        """The (per-frame transforms, box-origin) that carry ``roi`` with its tissue.

        Whole-frame ROIs use the global transforms about origin (0, 0); a per-leaf
        ROI uses its assigned leaf's transforms about that box's top-left corner
        (transforms are box-local). Returns ``(None, None)`` when nothing applies
        (e.g. an ROI in no leaf box) so the caller falls back to a static trace.
        """
        reg = self.registration
        if reg.mode == RegistrationMode.WHOLE_FRAME and reg.transforms:
            return reg.transforms, (0.0, 0.0)
        if reg.mode == RegistrationMode.PER_LEAF:
            idx = roi.leaf_region
            if idx is not None and 0 <= idx < len(self.leaf_regions):
                leaf = self.leaf_regions[idx]
                if leaf.transforms:
                    y0, _y1, x0, _x1 = leaf.bbox
                    return leaf.transforms, (float(y0), float(x0))
        return None, None

    def _extract_tracked(self, stack: np.ndarray, start: int) -> Traces:
        """Traces from ROIs that follow the tissue over the raw ``stack``.

        ``start`` is the first frame index (crop offset) so per-frame transforms,
        which are indexed in original recording frames, line up with the (possibly
        cropped) stack. ROIs with no applicable transform fall back to a static mask.
        """
        if not self.rois:
            return Traces(raw=np.empty((0, len(stack))), labels=[])
        rows, labels = [], []
        for i, roi in enumerate(self.rois):
            transforms, origin = self._roi_transform_series(roi)
            if transforms is None:
                rows.append(roi_mod.extract_trace(stack, roi))
            else:
                window = transforms[start:start + len(stack)]
                rows.append(roi_mod.extract_trace_tracked(stack, roi, window, origin))
            labels.append(roi.label or f"roi_{i}")
        return Traces(raw=np.stack(rows), labels=labels)

    def set_crop(self, start: Optional[int], end: Optional[int]) -> Traces:
        """Restrict traces to the ``[start, end)`` frame window, then re-extract.

        ``start``/``end`` are original frame indices (``None`` = open end); a window
        covering the whole recording clears the crop. Honored by every later
        ``extract_traces``. Returns the freshly cropped ``Traces``.
        """
        self._require_data()
        n = len(self._working_stack())
        lo = 0 if start is None else max(0, int(start))
        hi = n if end is None else min(n, int(end))
        if hi <= lo:
            raise ValueError(f"empty crop window: [{lo}, {hi})")
        self.crop_window = None if (lo == 0 and hi == n) else (lo, hi)
        self._invalidate_traces()
        return self.extract_traces()

    def crop_traces(self):
        """[notebook] Open the trace-cropping widget (blocking).

        Drag a window over the full-length traces and validate; returns the cropped
        ``Traces``, also stored so downstream analysis uses the same window.
        """
        from .widgets._qt import run_widget_blocking
        from .widgets.crop_widget import CropTracesWidget

        return run_widget_blocking(lambda: CropTracesWidget(self))

    # --------------------------------------------------------------- Stage III
    def set_frame_interval(self, seconds_per_frame: Optional[float]) -> "Session":
        """Calibrate the time axis (seconds per frame); ``None`` ⇒ frames-only.

        Once set, the analysis plot, CSV export and static figures report seconds
        instead of frames.
        """
        if self.timeline is None:
            raise RuntimeError("No data loaded; call load()/from_file() first.")
        self.timeline.frame_interval = seconds_per_frame
        return self

    def compute_dff(self, method=BaselineMethod.FIRST_N, n=None, region=None) -> Traces:
        """Recompute ΔF/F on the current traces with an explicit baseline,
        overriding the default (extracting traces first if needed). ``traces.dff``
        already holds a first-10-frame-baseline ΔF/F as soon as traces are
        extracted (see ``Traces``); call this to use a different baseline.

        method: ``BaselineMethod.FIRST_N`` — F0 = mean of first ``n`` frames;
            ``BaselineMethod.REGION`` — F0 = mean over ``region`` ``[start, end)``.
        """
        if self.traces is None:
            self.extract_traces()
        return analysis.compute_dff(self.traces, method=BaselineMethod(method), n=n, region=region)

    def smooth_traces(self, sigma: float) -> Traces:
        """Gaussian-smooth the current ΔF/F along time (extracting traces first if
        needed). See ``analysis.smooth_traces``.

        sigma: standard deviation of the Gaussian kernel, in frames. Always
            smooths ``traces.dff`` (which defaults to a first-10-frame baseline —
            see ``Traces``) — never the raw F. The result is stored on
            ``traces.smoothed`` and never overwrites ``dff``.
        """
        if self.traces is None:
            self.extract_traces()
        return analysis.smooth_traces(self.traces, sigma)

    def cross_roi_propagation(self, **kwargs):
        """Estimate signal propagation across ROIs; stores it under ``analyses``.

        Keyword args are forwarded to ``analysis.cross_roi_propagation``
        (``signal``, ``method``, ``frac``, ``k``, ``baseline_frames``,
        ``baseline_region``).
        """
        if self.traces is None:
            self.extract_traces()
        result = analysis.cross_roi_propagation(self.traces, self.rois, **kwargs)
        self.analyses["propagation"] = result
        return result

    def onset_heatmap(
        self,
        method: str = "fraction_of_max",
        frac: float = 0.5,
        k: float = 3.0,
        baseline_region: tuple[int, int] | None = None,
        bin_size: int = 1,
    ) -> np.ndarray:
        """Per-pixel response-onset heatmap over the working stack → 2D ``[Y, X]``.

        Runs the per-ROI detector (``analysis.onset_time_map``) on every pixel,
        honoring ``crop_window`` so it covers the same interval as the traces. See
        ``onset_time`` for ``method``/``frac``/``k`` and ``onset_time_map`` for
        ``bin_size``. ``baseline_region`` is in trace-column (post-crop)
        coordinates. NaN where no rise is detected.
        """
        self._require_data()
        stack = self._working_stack()
        if self.crop_window is not None:
            s, e = self.crop_window
            stack = stack[s:e]
        return analysis.onset_time_map(
            stack, method=method, frac=frac, k=k,
            baseline_region=baseline_region, bin_size=bin_size,
        )

    def apply(self, func: Callable):
        """Run a custom callable ``f(traces, data) -> result`` on the current traces."""
        if self.traces is None:
            self.extract_traces()
        return analysis.apply_custom(func, self.traces, self.data)

    def analyze(self):
        """[notebook] Open the analysis widget (blocking)."""
        from .widgets._qt import run_widget_blocking
        from .widgets.analysis_widget import AnalysisWidget

        return run_widget_blocking(lambda: AnalysisWidget(self))

    # ------------------------------------------------------------------ Export
    def provenance(self) -> dict:
        """Full parameter record (source, registration, ROIs, crop, events) as a dict."""
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
                "motion_tracking": self.track_motion,
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
            "crop_window": None if self.crop_window is None else list(self.crop_window),
            "events": [{"frame": e.frame, "label": e.label} for e in events],
            "analyses": sorted(self.analyses.keys()),
        }

    def trace_frames(self) -> np.ndarray:
        """Original (downsampled) frame index of each current-trace column.

        Honors ``crop_window`` so exports and figures label the true recording
        frames: a crop to ``[f0, f1)`` maps column ``c`` to frame ``f0 + c``.
        """
        if self.traces is not None:
            n = self.traces.raw.shape[1]
        elif self.data is not None:
            n = len(self._working_stack())
        else:
            n = 0
        start = self.crop_window[0] if self.crop_window is not None else 0
        return np.arange(start, start + n)

    def export_traces(self, path) -> None:
        """Write per-ROI raw F and ΔF/F over time to a CSV at ``path``."""
        export.traces_to_csv(self.traces, path, self.timeline, frames=self.trace_frames())

    def export_stack(self, path) -> None:
        """Write the working (stabilized if registered) stack to a TIFF at ``path``."""
        export.stack_to_tiff(self._working_stack(), path)

    def export_provenance(self, path) -> None:
        """Write ``provenance()`` as a JSON sidecar at ``path``."""
        export.write_provenance(self, path)

    # ---------------------------------------------------------------- Figures
    # Paper-grade static matplotlib figures; each returns a Figure and accepts the
    # keyword args of the matching figures.py function (see there for details).
    def figure_traces(self, **kw):
        """Stacked/overlaid ΔF/F traces. See ``figures.plot_traces`` for kwargs."""
        from . import figures
        return figures.plot_traces(self, **kw)

    def figure_propagation(self, **kw):
        """ROI onset map + propagation arrow. See ``figures.plot_propagation``."""
        from . import figures
        return figures.plot_propagation(self, **kw)

    def figure_roi_overlay(self, **kw):
        """Background image with ROIs drawn on top. See ``figures.plot_roi_overlay``."""
        from . import figures
        return figures.plot_roi_overlay(self, **kw)

    def figure_imaging_electrode(self, aux_time, aux_signal, **kw):
        """Calcium trace aligned with an electrode signal. See
        ``figures.plot_imaging_electrode``."""
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
        """Recomputation is explicit: upstream changes drop stale traces."""
        self.traces = None
        self.analyses.clear()
