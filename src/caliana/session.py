"""The Session — single source of truth for one analysis.

Widgets and notebook wrappers all read/write this object. Methods marked
``[notebook]`` open a blocking Qt widget; every other step is headless and callable
without a GUI. The actual work lives in the io / registration / roi / analysis /
export modules, which ``Session`` orchestrates.

Typical order: ``load`` → ``add_leaf_region`` / ``register`` → ``add_roi`` →
``extract_traces`` → ``compute_dff`` → analysis / ``export_*``.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import analysis, export, io
from . import registration as registration_mod
from . import roi as roi_mod
from .models import (
    ROI,
    BaselineMethod,
    ImportParams,
    LeafRegion,
    RegistrationMode,
    RegistrationResult,
    ROIShape,
    Traces,
)
from .space import SpatialScale
from .timeline import Timeline


def _downsampled_scale(native: float | None, step: int) -> float | None:
    """A file's native scale expressed per *loaded* sample.

    Downsampling by ``step`` makes each kept frame/pixel span ``step`` times as
    much time/space, so a 0.5 µm/px file imported with ``spatial_step=2`` has
    1.0 µm pixels. ``None`` (uncalibrated) stays ``None``.
    """
    return None if native is None else native * step


class Session:
    def __init__(self) -> None:
        self.source = None                              # SourceInfo
        self.data: np.ndarray | None = None          # [T, Y, X] raw downsampled
        self.registered_data: np.ndarray | None = None  # stabilized stack; None until register(apply=True)
        self.timeline: Timeline | None = None
        # Space axis, the counterpart of `timeline`: µm/px of the loaded stack,
        # uncalibrated until `load` reads it from the file or `set_pixel_size` sets it.
        self.space = SpatialScale()
        self.registration = RegistrationResult()
        # When True, traces are extracted by moving each ROI with its leaf's
        # per-frame transform over the RAW stack (ROI-follows-tissue), instead of
        # sampling a warped stack. Set by register(apply=False).
        self.track_motion: bool = False
        self.leaf_regions: list[LeafRegion] = []
        self.rois: list[ROI] = []
        # [start, end) frame window traces are cropped to before analysis; None =>
        # the whole recording. In original (uncropped) frame indices.
        self.crop_window: tuple[int, int] | None = None
        self.traces: Traces | None = None
        self.analyses: dict = {}
        # Monotonic change counter (see `_bump`). Panels in the standalone app
        # cache the value they last read and re-read the session when it moves,
        # so a change made on one tab shows up on the next with no cross-widget
        # wiring. Notebook use ignores it.
        self._revision: int = 0

    # ----------------------------------------------------------------- Stage I
    @classmethod
    def from_file(cls, path, **import_kwargs) -> Session:
        """New Session with ``path`` (``.tif``/``.tiff``/``.nd2``) loaded.

        Shorthand for ``Session().load(...)``, and the normal entry point.

        ``import_kwargs`` are ``ImportParams`` fields — the "downsample on load"
        controls, so a multi-GB recording never has to fit in RAM. All are
        optional; the default loads the file as-is:

        start, end: keep frames ``[start, end)`` (``end=None`` ⇒ to the last
            frame). Applied before anything else, so an nd2 only reads the frames
            inside the window.
        temporal_step: average every ``N`` frames into one (``1`` = off). An
            average, not a decimation, so the discarded frames still contribute
            SNR. Yields ``(end - start) // N`` frames, each ``N`` times longer.
        spatial_step: keep every ``N``-th pixel along Y and X (``1`` = full
            resolution). A stride subsample: cheap, and ``N``-times coarser.
        spatial_window: ``(y0, y1, x0, x1)`` crop of the field of view in the
            file's pixel coordinates (``None`` = whole frame).
        channel: which channel to keep from a multi-channel file. Caliana is
            single-channel throughout (no ratiometric math); ignored otherwise.

        They apply in that order — ``channel`` → temporal crop → ``temporal_step``
        → ``spatial_window`` → ``spatial_step`` — which is what makes them
        composable: each acts on the result of the previous one. Note that
        ``spatial_window`` is in *file* pixels but everything afterwards (ROI
        centres, leaf boxes, event frames, ``set_crop``) is in the coordinates of
        the loaded stack, i.e. after cropping and downsampling.

        Physical scale is read from the file's metadata when present (nd2
        acquisition settings; ImageJ/OME/resolution tags in TIFF) and adjusted for
        the downsampling, so ``timeline.frame_interval`` and ``space.pixel_size``
        are calibrated on arrival. Override or supply them with
        ``set_frame_interval`` / ``set_pixel_size``.

        Examples::

            Session.from_file("movie.nd2")                       # as recorded
            Session.from_file("movie.nd2", start=100, end=1600)  # 1500 frames
            Session.from_file("movie.nd2", temporal_step=2, spatial_step=2)
            Session.from_file("movie.tif", spatial_window=(0, 512, 128, 640))
            Session.from_file("two_channel.tif", channel=1)      # GCaMP channel

            ImportParams(**kwargs)  # equivalently, built explicitly

        The parameters used are kept on ``source.import_params`` and written to
        ``provenance()``, so a run can be reproduced from its sidecar. Raises
        ``ValueError`` on an unsupported file extension.
        """
        return cls().load(path, **import_kwargs)

    def load(self, path, **import_kwargs) -> Session:
        """Load a ``.tif``/``.tiff``/``.nd2`` stack, applying downsample-on-load.

        ``import_kwargs`` are ``ImportParams`` fields (``start``, ``end``,
        ``temporal_step``, ``spatial_step``, ``spatial_window``, ``channel``);
        see ``from_file`` for what each one does. Loading resets the time and
        space axes to whatever the file declares, and drops everything derived
        from the previous stack (see ``_reset_derived``) — re-importing the same
        file at a different ``spatial_step`` therefore starts from a clean slate,
        which is the point of the app's Import tab.
        """
        params = ImportParams(**import_kwargs)
        self.data, self.source = io.load_stack(path, params)
        self._reset_derived()
        # Calibrate both axes from the file's metadata where available. The stack
        # is downsampled, so a kept frame/pixel spans `*_step` native ones.
        scale = self.source.scale
        self.timeline = Timeline(
            n_frames=len(self.data),
            frame_interval=_downsampled_scale(scale.frame_interval, params.temporal_step),
        )
        self.space = SpatialScale(
            pixel_size=_downsampled_scale(scale.pixel_size, params.spatial_step)
        )
        self._bump()
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
        self._bump()
        return leaf

    def set_leaf_mask(self, leaf, vertices) -> LeafRegion:
        """Restrict a leaf box's motion estimate to a hand-drawn polygon.

        leaf: a ``LeafRegion`` or its index in ``leaf_regions``.
        vertices: at least 3 ``(y, x)`` outline points, in the loaded stack's
            pixel coordinates (the same ones ``bbox`` uses). The part of the
            outline outside the box is ignored, so drawing it inside the box is
            what makes it mean anything.

        Only the pixels inside the outline take part in the fit; the whole box is
        still warped. Use it when a box unavoidably contains something that does
        not move with the leaf — a neighbouring leaf, a bright static background —
        which would otherwise anchor the estimate. Draw it a couple of pixels
        outside the tissue's edge rather than exactly on it.
        """
        leaf = self._leaf(leaf)
        verts = [tuple(v) for v in vertices]
        if len(verts) < 3:
            raise ValueError(f"a mask polygon needs at least 3 vertices, got {len(verts)}")
        leaf.mask_polygon = verts
        self._bump()
        return leaf

    def clear_leaf_mask(self, leaf) -> LeafRegion:
        """Drop a leaf box's mask polygon, estimating on the whole box again."""
        leaf = self._leaf(leaf)
        leaf.mask_polygon = None
        self._bump()
        return leaf

    def _leaf(self, leaf) -> LeafRegion:
        """A ``LeafRegion`` from either the object itself or its index."""
        if isinstance(leaf, LeafRegion):
            return leaf
        return self.leaf_regions[leaf]

    def register(
        self,
        mode=RegistrationMode.WHOLE_FRAME,
        reference: str = "mean",
        apply: bool = True,
        transformation: str = "affine",
    ) -> Session:
        """Run motion correction in the chosen mode.

        mode: ``RegistrationMode.NONE`` (ROIs on the raw stack), ``WHOLE_FRAME``
            (one transform per frame), or ``PER_LEAF`` (each leaf box registered
            independently; requires ``add_leaf_region`` first, and honours any
            ``set_leaf_mask`` outline drawn inside a box).
        reference: ``"mean"`` (default), ``"first"``, or ``"previous"``.
        transformation: TurboReg model — ``"translation"``, ``"rigid_body"``,
            ``"scaled_rotation"``, or ``"affine"`` (default). Scale/shear is kept
            and carries through to the stabilized stack and ROI tracking.
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
                self.data, reference, transformation=transformation
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
                self.data, self.leaf_regions, reference,
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
        return self._append_roi(
            ROI(center=tuple(center), size=size, shape=ROIShape(shape), label=label)
        )

    def add_polygon_roi(self, vertices, label: str = "") -> ROI:
        """Add a free-hand polygon ROI (e.g. a whole leaf).

        ``vertices`` is a list of ``(y, x)`` outline points; the ROI centre is set
        to the polygon centroid. In per-leaf mode it auto-assigns to its containing
        leaf box.
        """
        verts = [tuple(v) for v in vertices]
        return self._append_roi(ROI(
            center=roi_mod.polygon_centroid(verts), size=0.0,
            shape=ROIShape.POLYGON, label=label, vertices=verts,
        ))

    def _append_roi(self, roi: ROI) -> ROI:
        """Store a freshly built ROI: auto-assign its leaf, then drop stale traces."""
        self.assign_leaf(roi)
        self.rois.append(roi)
        self._invalidate_traces()
        return roi

    def assign_leaf(self, roi: ROI) -> ROI:
        """Point ``roi`` at the leaf box containing its centre (per-leaf mode only).

        Call after moving an ROI so it stays attached to the right box; a no-op in
        any other registration mode.
        """
        if self.registration.mode == RegistrationMode.PER_LEAF:
            roi.leaf_region = roi_mod.assign_roi_to_leaf(roi, self.leaf_regions)
        return roi

    def extract_traces(self) -> Traces:
        """Extract (and store) the mean-intensity raw F trace per ROI.

        Uses the stabilized stack, or moves each ROI with its tissue when
        registered with ``apply=False`` (``track_motion``). A ``crop_window`` (see
        ``set_crop``/``crop_traces``) restricts the traces to that frame interval.
        """
        self._require_data()
        self.traces = self._extract_window(*self._crop_bounds())
        return self.traces

    def _extract_window(self, start: int = 0, end: int | None = None) -> Traces:
        """Traces over frames ``[start, end)`` via the currently active path.

        Two extraction paths:
         - ``track_motion``: keep raw pixels, move each ROI with its tissue per
           frame (no resampling of the measured intensities).
         - otherwise: ``_working_stack()`` is the stabilized stack (whole-frame
           warp, or per-leaf composite of stabilized sub-stacks), ROIs are static.

        Separate from ``extract_traces`` so a caller can preview the *uncropped*
        traces (the crop widget) through the same path the crop will later use.
        """
        stack = self._working_stack()[start:end]
        if self.track_motion and self._has_transforms():
            return self._extract_tracked(stack, start)
        return roi_mod.extract_all_traces(stack, self.rois)

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

    def set_crop(self, start: int | None, end: int | None) -> Traces:
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

        return run_widget_blocking(lambda: CropTracesWidget(self), close_on="applied")

    # --------------------------------------------------------------- Stage III
    def set_frame_interval(self, seconds_per_frame: float | None) -> Session:
        """Calibrate the time axis (seconds per frame); ``None`` ⇒ frames-only.

        Once set, the analysis plot, CSV export and static figures report seconds
        instead of frames.
        """
        if self.timeline is None:
            raise RuntimeError("No data loaded; call load()/from_file() first.")
        self.timeline.frame_interval = seconds_per_frame
        self._bump()
        return self

    def set_pixel_size(self, microns_per_pixel: float | None) -> Session:
        """Calibrate the space axis (µm per pixel); ``None`` ⇒ pixels-only.

        In *loaded* pixels, matching ROI coordinates: a stack imported with
        ``spatial_step=2`` has pixels twice as wide as the file's. ``load`` already
        sets this from file metadata when the recording declares it — call this to
        supply or correct it.

        Once set, propagation speed and the distance-vs-onset graph report µm
        (µm/s when the frame interval is set too), as do the saved figures.
        """
        self._require_data()
        self.space.pixel_size = microns_per_pixel
        self._bump()
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
        (``signal``, ``method``, ``frac``, ``k``, ``d``, ``baseline_region``,
        ``direction_mode``).
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
        d: float = 0.0,
        baseline_region: tuple[int, int] | None = None,
        bin_size: int = 1,
    ) -> np.ndarray:
        """Per-pixel response-onset heatmap over the working stack → 2D ``[Y, X]``.

        Runs the per-ROI detector (``analysis.onset_time_map``) on every pixel,
        honoring ``crop_window`` so it covers the same interval as the traces. See
        ``onset_time`` for ``method``/``frac``/``k``/``d`` and ``onset_time_map`` for
        ``bin_size``. ``baseline_region`` is in trace-column (post-crop)
        coordinates. NaN where no rise is detected.
        """
        self._require_data()
        start, end = self._crop_bounds()
        stack = self._working_stack()[start:end]
        return analysis.onset_time_map(
            stack, method=method, frac=frac, k=k, d=d,
            baseline_region=baseline_region, bin_size=bin_size,
        )

    def kymograph(
        self,
        path,
        width: int = 1,
        step: float = 1.0,
        baseline: tuple[int, int] | None = None,
    ) -> dict:
        """Intensity along a hand-drawn path over time; stores it under ``analyses``.

        Samples the working stack along the polyline ``path`` of ``(y, x)`` pixel
        points, honoring ``crop_window`` so it covers the same interval as the
        traces, and returns the distance × time image (see ``analysis.kymograph``
        for ``width``, ``step`` and ``baseline``, whose frames are trace-column,
        post-crop indices).

        The path is fixed in the stack's coordinates, so with ``track_motion``
        (registered with ``apply=False``, where the stack stays raw and the *ROIs*
        move) the tissue slides under it. Warp the stack — ``register(apply=True)``
        — for a motion-corrected kymograph.
        """
        self._require_data()
        start, end = self._crop_bounds()
        stack = self._working_stack()[start:end]
        result = analysis.kymograph(stack, path, width=width, step=step,
                                    baseline=baseline)
        self.analyses["kymograph"] = result
        return result

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
        """Full parameter record (version, source, registration, ROIs, crop, events)
        as a dict."""
        from . import __version__  # deferred: __init__ imports this module

        src = self.source
        events = self.timeline.events if self.timeline else []
        return {
            "caliana_version": __version__,
            "source": None if src is None else {
                "path": str(src.path),
                "import_params": vars(src.import_params),
                # The file's own µm/px and s/frame, before downsampling; the
                # effective ones are under "scale" below.
                "file_scale": vars(src.scale),
            },
            "scale": {
                "frame_interval": None if self.timeline is None else self.timeline.frame_interval,
                "pixel_size": self.space.pixel_size,
            },
            "registration": {
                "mode": self.registration.mode.value,
                "reference": self.registration.reference,
                "motion_tracking": self.track_motion,
                "leaf_regions": [
                    {"bbox": list(lr.bbox), "label": lr.label,
                     "mask_polygon": (None if lr.mask_polygon is None
                                      else [list(v) for v in lr.mask_polygon]),
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
        start, _end = self._crop_bounds()
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

    # ----------------------------------------------------------------- helpers
    def _require_data(self) -> None:
        if self.data is None:
            raise RuntimeError("No data loaded; call load()/from_file() first.")

    def _working_stack(self) -> np.ndarray:
        """The stack ROIs/heatmaps act on: stabilized if registered, else raw."""
        return self.registered_data if self.registered_data is not None else self.data

    def _crop_bounds(self) -> tuple[int, int | None]:
        """The ``[start, end)`` frame window in play — the crop, else the whole stack.

        ``start`` doubles as the offset from trace columns to original frame
        indices, which is what ``trace_frames`` and tracked extraction need.
        """
        return self.crop_window if self.crop_window is not None else (0, None)

    def _invalidate_traces(self) -> None:
        """Recomputation is explicit: upstream changes drop stale traces."""
        self.traces = None
        self.analyses.clear()
        self._bump()

    def _reset_derived(self) -> None:
        """Drop everything derived from the stack being replaced.

        ROIs, leaf boxes, registration and the crop window are all expressed in
        the coordinates of the stack they were drawn on, so carrying them into a
        newly loaded file (or the same file re-imported at another
        ``spatial_step``) would silently measure the wrong pixels. Called by
        ``load``; the axes themselves are re-read from the file there.
        """
        self.registered_data = None
        self.registration = RegistrationResult()
        self.track_motion = False
        self.leaf_regions = []
        self.rois = []
        self.crop_window = None
        self._invalidate_traces()

    def _bump(self) -> None:
        """Mark the session changed, moving ``_revision`` on.

        Every mutation a panel might need to redraw for goes through here —
        ``load``, ``add_leaf_region``, ``register``, the calibration setters, and
        ``_invalidate_traces`` (which ROI edits and ``set_crop`` funnel through).
        Widgets editing the session in place (deleting an ROI or a leaf box) call
        it themselves.
        """
        self._revision += 1
