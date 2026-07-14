"""Rigid registration (leaf-motion correction). SPEC.md §3 Stage II.

Three modes — none / whole-frame / per-leaf — all using rigid (translation +
rotation) transforms only. Estimation runs on the downsampled stack; `pystackreg`
is imported lazily. Per-leaf mode registers each leaf box's sub-stack
independently and flags drift-out-of-box frames as low-confidence.
"""
from __future__ import annotations

import warnings

import numpy as np

from .models import LeafRegion, RegistrationResult, RegistrationMode, RigidTransform


def make_reference(stack: np.ndarray, reference: str = "mean") -> np.ndarray:
    """Build the stored registration target. SPEC §3: default mean image, else first frame.

    ``"previous"`` registers each frame to its predecessor inside pystackreg, so there
    is no single target image; the first frame is kept as the representative reference.
    """
    if reference == "mean":
        return stack.mean(axis=0)
    if reference in ("first", "previous"):
        return stack[0]
    raise ValueError(f"Unknown reference {reference!r} (expected 'mean' | 'first' | 'previous')")


def map_point(point_yx, tf: RigidTransform, origin_yx=(0.0, 0.0)) -> tuple[float, float]:
    """Map a (y, x) point through transform ``tf`` about ``origin_yx``.

    This is the single source of truth for "where does this tissue point sit in
    the raw frame". The transform maps raw→reference; applied to a
    reference-frame point it returns the raw position, so an ROI placed on the
    stabilized view follows the tissue as the leaf moves (the ROI-tracking path,
    SPEC §3). Per-leaf transforms are estimated in box-local coordinates, so pass
    the box's top-left ``(y0, x0)`` as the origin; whole-frame uses ``(0, 0)``.
    Both the ROI-selection widget preview and headless trace extraction call this,
    so the picture on screen and the extracted trace can never diverge.
    """
    m = _rigid_to_matrix(tf)
    cy, cx = point_yx
    oy, ox = origin_yx
    x, y = cx - ox, cy - oy                        # matrix acts on (x, y, 1)
    rx = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    ry = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return ry + oy, rx + ox


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold on a 1-D array of intensities (numpy only, no skimage).

    Splits the histogram into two classes maximizing between-class variance; used
    to separate the dark leaf tissue from the bright, high-texture background.
    """
    v = np.asarray(values, dtype=float).ravel()
    hist, edges = np.histogram(v, bins=256)
    hist = hist.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(hist)
    total = w[-1]
    if total == 0:
        return float(v.mean()) if v.size else 0.0
    wsum = np.cumsum(hist * centers)
    wb, wf = w, total - w
    with np.errstate(invalid="ignore", divide="ignore"):
        mb = wsum / wb
        mf = (wsum[-1] - wsum) / wf
        var_between = wb * wf * (mb - mf) ** 2
    var_between[~np.isfinite(var_between)] = -np.inf
    return float(centers[int(np.argmax(var_between))])


def segment_tissue(image: np.ndarray, dark: bool = True, close: int = 2, open_: int = 2) -> np.ndarray:
    """Boolean mask of the leaf tissue in a 2-D image. SPEC §3 Stage II.

    The recordings are dim silhouettes: leaves are dark and low-texture, the
    background (soil/perlite, instruments) is bright, high-texture and static.
    Plain intensity registration therefore locks onto the static background, not
    the leaves. Masking to the tissue (``dark=True``: pixels below Otsu) before
    estimation restores a strong, moving edge for the registration to track.
    Morphological open→close removes speckle and fills small holes.
    """
    from scipy.ndimage import binary_closing, binary_opening

    img = np.asarray(image, dtype=float)
    thr = _otsu_threshold(img)
    mask = img < thr if dark else img > thr
    if open_:
        mask = binary_opening(mask, iterations=open_)
    if close:
        mask = binary_closing(mask, iterations=close)
    return mask


def _silhouette_stack(stack: np.ndarray) -> np.ndarray:
    """Per-frame binary tissue silhouette (0/1), as the registration target.

    For dark-on-bright recordings the leaf *boundary* carries the motion, while
    the raw intensities are dim and low-contrast and the bright background is
    static. Segmenting each frame and registering the silhouettes makes the leaf
    edge the dominant, high-contrast feature (0→1), so the rigid estimate tracks
    the tissue instead of the immobile background. The silhouettes are stable
    frame-to-frame on real data (high IoU); if segmentation flickered it would add
    jitter, which is why this is opt-in (``mask=True``), not the default.
    """
    stack = np.asarray(stack, dtype=float)
    return np.stack([segment_tissue(f).astype(float) for f in stack])


def _matrix_to_rigid(m: np.ndarray) -> RigidTransform:
    """Wrap a pystackreg 3x3 homogeneous transform as a ``RigidTransform``.

    The matrix acts on (x, y, 1). We keep it verbatim in ``matrix`` (the
    authoritative form, so scale/shear from ``scaled_rotation``/``affine`` survive
    to warping and ROI motion) and *also* read off the rigid summary — translation
    from the last column and rotation from ``arctan2(m10, m00)`` — for the drift
    heuristic and display. For a pure rigid body the summary reproduces the matrix
    exactly (see ``_rigid_to_matrix``).
    """
    m = np.asarray(m, dtype=float)
    theta = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    return RigidTransform(dy=float(m[1, 2]), dx=float(m[0, 2]), theta=float(theta), matrix=m)


def _rigid_to_matrix(tf: RigidTransform) -> np.ndarray:
    """The 3x3 homogeneous transform for ``tf`` (acts on (x, y, 1)).

    Uses the full estimated ``matrix`` when present (carrying scale/shear);
    otherwise rebuilds the pure rigid body from the scalar summary.
    """
    if tf.matrix is not None:
        return np.asarray(tf.matrix, dtype=float)
    t = np.radians(tf.theta)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, tf.dx], [s, c, tf.dy], [0.0, 0.0, 1.0]], dtype=float)


# pystackreg transformation models, keyed by friendly name so callers (and the
# Session layer) never have to import pystackreg just to pick one. Restricted to
# the models that produce a plain 3x3 homogeneous matrix (BILINEAR's 4x4 form is
# not a 3x3 affine and cannot round-trip through ``RigidTransform.matrix``).
_STACKREG_TRANSFORMS = {
    "translation": "TRANSLATION",
    "rigid_body": "RIGID_BODY",
    "rigid": "RIGID_BODY",
    "scaled_rotation": "SCALED_ROTATION",
    "affine": "AFFINE",
}


def _resolve_transformation(transformation: str) -> int:
    """Map a friendly transformation name to a ``StackReg`` model constant.

    The full estimated matrix is retained on ``RigidTransform.matrix``, so
    scale/shear from ``scaled_rotation``/``affine`` carries through to warping and
    ROI motion; ``rigid_body`` and ``translation`` are the rigid special cases.
    """
    from pystackreg import StackReg

    try:
        attr = _STACKREG_TRANSFORMS[transformation]
    except KeyError:
        raise ValueError(
            f"Unknown transformation {transformation!r} "
            f"(expected one of {sorted(_STACKREG_TRANSFORMS)})"
        )
    return getattr(StackReg, attr)


def register_whole_frame(
    stack: np.ndarray,
    reference: str = "mean",
    mask: bool = False,
    transformation: str = "affine",
) -> RegistrationResult:
    """One rigid transform per frame vs the reference. SPEC §3 Stage II.

    Estimates one transform per frame against a fixed reference (mean image by
    default, or first frame). ``transformation`` selects the pystackreg model used
    for estimation — one of ``_STACKREG_TRANSFORMS`` (e.g. ``"affine"``,
    ``"rigid_body"``, ``"translation"``); the full estimated matrix is retained, so
    scale/shear carries through to warping and ROIs (see ``_matrix_to_rigid``).
    With ``mask=True`` the estimate runs on the per-frame tissue silhouette so it
    tracks the dim leaf boundary rather than locking onto the immobile bright
    background — see ``_silhouette_stack`` / ``segment_tissue``. The returned
    transforms still describe the full frame, so they apply unchanged to warping or
    ROIs.
    """
    from pystackreg import StackReg

    if reference not in ("mean", "first", "previous"):
        raise ValueError(f"Unknown reference {reference!r} (expected 'mean' | 'first' | 'previous')")

    est = _silhouette_stack(stack) if mask else np.asarray(stack, dtype=float)
    sr = StackReg(_resolve_transformation(transformation))
    with warnings.catch_warnings():
        # Our contract fixes time on axis 0; silence pystackreg's axis heuristic.
        warnings.filterwarnings("ignore", message=".*possible time axis.*")
        tmats = sr.register_stack(est, reference=reference)
    transforms = [_matrix_to_rigid(m) for m in tmats]
    return RegistrationResult(
        mode=RegistrationMode.WHOLE_FRAME, reference=reference, transforms=transforms
    )


def _drift_out_frames(transforms: list[RigidTransform], shape_yx, frac: float) -> list[int]:
    """Frames whose displacement approaches the box margin (drift-out-of-box).

    Heuristic: flag a frame when its translation magnitude relative to the
    reference exceeds ``frac`` of the box's smaller side (SPEC "Drift-out-of-box
    handling").
    """
    h, w = shape_yx
    thr = frac * min(h, w)
    return [i for i, t in enumerate(transforms) if float(np.hypot(t.dx, t.dy)) > thr]


def register_per_leaf(
    stack: np.ndarray,
    leaf_regions: list[LeafRegion],
    reference: str = "mean",
    drift_frac: float = 0.25,
    mask: bool = False,
    transformation: str = "affine",
) -> list[LeafRegion]:
    """Register each leaf box's sub-stack independently, in place. SPEC §3 Stage II.

    For each leaf: crop to bbox, build its own reference, estimate per-frame rigid
    transforms (reusing the whole-frame estimator on the sub-stack), and flag
    frames whose offset approaches the box margin as low-confidence. ``mask=True``
    segments the tissue inside each box before estimating, so a generously drawn
    box (mostly background) still tracks the leaf and not its static surroundings.
    """
    stack = np.asarray(stack)
    for leaf in leaf_regions:
        y0, y1, x0, x1 = leaf.bbox
        sub = stack[:, y0:y1, x0:x1]
        leaf.transforms = register_whole_frame(
            sub, reference, mask=mask, transformation=transformation
        ).transforms
        leaf.reference = make_reference(np.asarray(sub, dtype=float), reference)
        leaf.low_confidence_frames = _drift_out_frames(leaf.transforms, sub.shape[1:], drift_frac)
    return leaf_regions


def apply_per_leaf(stack: np.ndarray, leaf_regions: list[LeafRegion]) -> np.ndarray:
    """Composite stabilized stack: each leaf box replaced by its stabilized sub-stack.

    Pixels outside every leaf box keep their raw values; overlapping boxes are
    resolved last-box-wins. ROIs inside a box therefore sample stabilized tissue.
    """
    out = np.array(stack, dtype=float)
    for leaf in leaf_regions:
        if not leaf.transforms:
            continue
        y0, y1, x0, x1 = leaf.bbox
        sub = np.asarray(stack)[:, y0:y1, x0:x1]
        out[:, y0:y1, x0:x1] = apply_transforms(sub, leaf.transforms)
    return out


def apply_transforms(stack: np.ndarray, transforms: list[RigidTransform]) -> np.ndarray:
    """Warp each frame by its transform to produce a stabilized stack.

    Returns a float stack aligned to the registration reference. ``sr.transform``
    warps by the supplied ``tmat`` (the full estimated matrix, including any
    scale/shear) irrespective of the model the StackReg was constructed with, so
    non-rigid estimates are reproduced exactly and the pipeline round-trips.
    """
    from pystackreg import StackReg

    if len(transforms) != len(stack):
        raise ValueError(
            f"transforms ({len(transforms)}) must match frames ({len(stack)})"
        )
    sr = StackReg(StackReg.RIGID_BODY)
    out = np.empty(np.shape(stack), dtype=float)
    for i, (frame, tf) in enumerate(zip(stack, transforms)):
        out[i] = sr.transform(np.asarray(frame, dtype=float), tmat=_rigid_to_matrix(tf))
    return out
