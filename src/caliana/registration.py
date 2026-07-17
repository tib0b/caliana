"""Motion correction (leaf-motion registration).

Three modes — none / whole-frame / per-leaf. Estimation runs on the downsampled
stack via ``pystackreg`` (imported lazily). Per-leaf mode registers each leaf
box's sub-stack independently and flags drift-out-of-box frames as low-confidence.
"""
from __future__ import annotations

import warnings

import numpy as np

from .models import LeafRegion, RegistrationResult, RegistrationMode, RigidTransform


def make_reference(stack: np.ndarray, reference: str = "mean") -> np.ndarray:
    """Build the stored registration target image.

    reference: ``"mean"`` (mean over time), ``"first"``, or ``"previous"``.
    ``"previous"`` has no single target (each frame registers to its predecessor),
    so the first frame is stored as its representative reference.
    """
    if reference == "mean":
        return stack.mean(axis=0)
    if reference in ("first", "previous"):
        return stack[0]
    raise ValueError(f"Unknown reference {reference!r} (expected 'mean' | 'first' | 'previous')")


def map_point(point_yx, tf: RigidTransform, origin_yx=(0.0, 0.0)) -> tuple[float, float]:
    """Map a (y, x) point through transform ``tf`` about ``origin_yx``.

    ``tf`` maps raw→reference, so applied to a reference-frame point this returns
    the raw position — an ROI on the stabilized view thus follows the tissue as the
    leaf moves. Per-leaf transforms are box-local: pass the box's top-left
    ``(y0, x0)`` as ``origin_yx``; whole-frame uses ``(0, 0)``.
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

    Splits the histogram into two classes maximizing between-class variance.
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
    """Boolean mask of the leaf tissue in a 2-D image, by Otsu threshold.

    dark: True keeps pixels below the threshold (dark leaves on a bright
    background — the usual case here); False keeps pixels above it.
    close / open_: iterations of morphological closing / opening to fill small
    holes and remove speckle (0 disables either).
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
    """Per-frame binary tissue silhouette (0/1), used as the registration target.

    Registering silhouettes makes the moving leaf edge the dominant feature, so the
    estimate tracks the tissue instead of the static bright background. Opt-in
    (``mask=True``) because flickering segmentation would add jitter.
    """
    stack = np.asarray(stack, dtype=float)
    return np.stack([segment_tissue(f).astype(float) for f in stack])


def _matrix_to_rigid(m: np.ndarray) -> RigidTransform:
    """Wrap a pystackreg 3x3 homogeneous transform as a ``RigidTransform``.

    Keeps the matrix verbatim (authoritative, so scale/shear survives) and also
    reads off the rigid summary — translation from the last column, rotation from
    ``arctan2(m10, m00)`` — for the drift heuristic and display.
    """
    m = np.asarray(m, dtype=float)
    theta = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    return RigidTransform(dy=float(m[1, 2]), dx=float(m[0, 2]), theta=float(theta), matrix=m)


def _rigid_to_matrix(tf: RigidTransform) -> np.ndarray:
    """The 3x3 homogeneous transform for ``tf`` (acts on ``(x, y, 1)``).

    Uses ``tf.matrix`` when present (carrying scale/shear); otherwise rebuilds the
    pure rigid body from the scalar summary.
    """
    if tf.matrix is not None:
        return np.asarray(tf.matrix, dtype=float)
    t = np.radians(tf.theta)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, tf.dx], [s, c, tf.dy], [0.0, 0.0, 1.0]], dtype=float)


# Accepted ``transformation`` names → pystackreg model, so callers never import
# pystackreg to pick one. Ordered least→most free (translation ⊂ rigid ⊂ scaled
# rotation ⊂ affine). BILINEAR is excluded: its 4x4 form can't round-trip through
# ``RigidTransform.matrix``.
_STACKREG_TRANSFORMS = {
    "translation": "TRANSLATION",
    "rigid_body": "RIGID_BODY",
    "rigid": "RIGID_BODY",
    "scaled_rotation": "SCALED_ROTATION",
    "affine": "AFFINE",
}


def _resolve_transformation(transformation: str) -> int:
    """Map a ``transformation`` name (see ``_STACKREG_TRANSFORMS``) to its
    ``StackReg`` model constant. Raises ``ValueError`` on an unknown name.
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
    """Estimate one transform per frame against a fixed reference.

    reference: ``"mean"`` (default), ``"first"``, or ``"previous"``.
    transformation: pystackreg model — ``"translation"``, ``"rigid_body"``
        (alias ``"rigid"``), ``"scaled_rotation"``, or ``"affine"`` (default). The
        full estimated matrix is kept, so scale/shear survives to warping and ROIs.
    mask: estimate on the per-frame tissue silhouette instead of raw intensities,
        so registration tracks the dim leaf rather than the static bright
        background (recommended for these recordings).
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

    Flags a frame when its translation magnitude exceeds ``frac`` of the box's
    smaller side.
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
    """Register each leaf box's sub-stack independently, mutating ``leaf_regions``.

    For each leaf: crop to its bbox, estimate per-frame transforms, store its
    reference, and flag drift-out frames in ``low_confidence_frames``.
    reference / mask / transformation: as in ``register_whole_frame``.
    drift_frac: fraction of the box's smaller side a frame may shift before being
        flagged low-confidence.
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

    Pixels outside every box keep their raw values; overlapping boxes are resolved
    last-box-wins. Requires ``register_per_leaf`` to have run first.
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
    """Warp each frame by its transform to produce a stabilized float stack.

    Aligned to the registration reference. Raises ``ValueError`` if ``transforms``
    and ``stack`` differ in length. Non-rigid estimates (scale/shear) are applied
    exactly, since warping uses the stored matrix.
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
