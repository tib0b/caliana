"""ROI masks, trace extraction, and leaf assignment."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .models import LeafRegion, ROI, ROIShape, RigidTransform, Traces


def roi_mask(roi: ROI, shape_yx: tuple[int, int]) -> np.ndarray:
    """Boolean mask for an ROI over a frame of shape ``(Y, X)``."""
    h, w = shape_yx
    if roi.shape == ROIShape.POLYGON:
        return polygon_mask(roi.vertices or [], shape_yx)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = roi.center
    if roi.shape == ROIShape.CIRCLE:
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= roi.size ** 2
    # SQUARE: `size` is the half-side.
    return (np.abs(yy - cy) <= roi.size) & (np.abs(xx - cx) <= roi.size)


def polygon_centroid(vertices) -> tuple[float, float]:
    """Area-weighted centroid (cy, cx) of a polygon given (y, x) vertices.

    Falls back to the vertex mean for degenerate (zero-area) polygons.
    """
    v = np.asarray(vertices, dtype=float)
    y, x = v[:, 0], v[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area2 = cross.sum()
    if abs(area2) < 1e-9:
        return float(y.mean()), float(x.mean())
    cx = ((x + x2) * cross).sum() / (3.0 * area2)
    cy = ((y + y2) * cross).sum() / (3.0 * area2)
    return float(cy), float(cx)


def polygon_mask(vertices, shape_yx: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon (free-hand ROI) to a boolean mask over ``(Y, X)``.

    Even-odd ray casting at pixel centres, vectorized over the polygon's bounding
    box (numpy only). Returns all-False for fewer than 3 vertices.
    """
    h, w = shape_yx
    v = np.asarray(vertices, dtype=float)
    mask = np.zeros((h, w), dtype=bool)
    if len(v) < 3:
        return mask
    vy, vx = v[:, 0], v[:, 1]
    y0, y1 = max(int(np.floor(vy.min())), 0), min(int(np.ceil(vy.max())) + 1, h)
    x0, x1 = max(int(np.floor(vx.min())), 0), min(int(np.ceil(vx.max())) + 1, w)
    if y1 <= y0 or x1 <= x0:
        return mask
    gy, gx = np.mgrid[y0:y1, x0:x1]
    inside = _points_in_polygon(gy.ravel().astype(float), gx.ravel().astype(float), vy, vx)
    mask[y0:y1, x0:x1] = inside.reshape(gy.shape)
    return mask


def _points_in_polygon(py, px, vy, vx) -> np.ndarray:
    """Even-odd point-in-polygon test for arrays of points against (vy, vx)."""
    inside = np.zeros(py.shape, dtype=bool)
    n = len(vx)
    j = n - 1
    for i in range(n):
        cond = ((vy[i] > py) != (vy[j] > py)) & (
            px < (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i] + 1e-12) + vx[i]
        )
        inside ^= cond
        j = i
    return inside


def extract_trace(data: np.ndarray, roi: ROI) -> np.ndarray:
    """Mean pixel intensity inside the ROI per frame → raw F ``[T]`` (0 if empty)."""
    mask = roi_mask(roi, data.shape[1:])
    if not mask.any():
        return np.zeros(len(data), dtype=float)
    return data[:, mask].mean(axis=1)


def extract_all_traces(data: np.ndarray, rois: list[ROI]) -> Traces:
    """Raw F traces for every ROI → ``Traces`` with ``raw`` ``[n_roi, T]``."""
    if not rois:
        return Traces(raw=np.empty((0, len(data))), labels=[])
    raw = np.stack([extract_trace(data, r) for r in rois])
    labels = [r.label or f"roi_{i}" for i, r in enumerate(rois)]
    return Traces(raw=raw, labels=labels)


def move_roi(roi: ROI, tf: RigidTransform, origin=(0.0, 0.0)) -> ROI:
    """A copy of ``roi`` re-placed where its tissue sits under transform ``tf``.

    A polygon's vertices are all mapped (so it translates and rotates with the
    leaf); a circle/square keeps its size and only moves its centre (it follows the
    translation but is not itself rotated). ``origin`` is the box-local origin for
    per-leaf transforms (see ``registration.map_point``).
    """
    from .registration import map_point

    if roi.shape == ROIShape.POLYGON and roi.vertices:
        verts = [map_point(v, tf, origin) for v in roi.vertices]
        return replace(roi, vertices=verts, center=polygon_centroid(verts))
    return replace(roi, center=map_point(roi.center, tf, origin))


def extract_trace_tracked(
    data: np.ndarray, roi: ROI, transforms: list[RigidTransform], origin=(0.0, 0.0)
) -> np.ndarray:
    """Trace of an ROI that follows the tissue frame by frame → raw F ``[T]``.

    Moves the ROI by each frame's transform and samples the raw pixels underneath,
    so the measured intensities are never resampled (unlike a static mask on a
    warped stack) — ΔF/F is not biased by interpolation. Frames past the end of
    ``transforms`` use the identity; a frame with the ROI fully out of view is 0.0.
    """
    shape_yx = data.shape[1:]
    out = np.zeros(len(data), dtype=float)
    ident = RigidTransform()
    for t in range(len(data)):
        tf = transforms[t] if t < len(transforms) else ident
        mask = roi_mask(move_roi(roi, tf, origin), shape_yx)
        if mask.any():
            out[t] = data[t][mask].mean()
    return out


def assign_roi_to_leaf(roi: ROI, leaf_regions: list[LeafRegion]) -> int | None:
    """Index of the first leaf box containing the ROI's centre, or None.

    None means the centre falls in no box, so the ROI gets no per-leaf
    stabilization — surface this to the user.
    """
    cy, cx = roi.center
    for i, leaf in enumerate(leaf_regions):
        y0, y1, x0, x1 = leaf.bbox
        if y0 <= cy < y1 and x0 <= cx < x1:
            return i
    return None
