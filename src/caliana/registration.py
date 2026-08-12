"""Motion correction (leaf-motion registration).

Three modes — none / whole-frame / per-leaf. Estimation runs on the downsampled
stack via ``crabstack`` (a Rust port of pystackreg's TurboReg core, imported
lazily through ``._stackreg``). Per-leaf mode registers each leaf box's sub-stack
independently and flags drift-out-of-box frames as low-confidence; a box may
carry a hand-drawn mask polygon naming the tissue to track inside it.
"""
from __future__ import annotations

import numpy as np

from .models import LeafRegion, RegistrationResult, RegistrationMode, RigidTransform
from .roi import polygon_mask


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


def leaf_mask(leaf: LeafRegion, shape_yx: tuple[int, int]) -> np.ndarray | None:
    """Rasterize a leaf's hand-drawn mask polygon over its *cropped* sub-stack.

    ``leaf.mask_polygon`` is stored in full-frame coordinates (that is what the
    widget draws in), while estimation runs on the box crop, so the outline is
    shifted by the box's top-left corner first; whatever falls outside the box is
    clipped away. Returns ``None`` when the leaf carries no usable outline —
    fewer than 3 vertices, or nothing of it inside the box — which is the "use
    every pixel" case.
    """
    verts = leaf.mask_polygon or []
    if len(verts) < 3:
        return None
    y0, _y1, x0, _x1 = leaf.bbox
    mask = polygon_mask([(y - y0, x - x0) for (y, x) in verts], shape_yx)
    return mask if mask.any() else None


def _matrix_to_rigid(m: np.ndarray) -> RigidTransform:
    """Wrap an estimated 3x3 homogeneous transform as a ``RigidTransform``.

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


# Accepted ``transformation`` names → TurboReg model, so callers never import the
# registration backend to pick one. Ordered least→most free (translation ⊂ rigid
# ⊂ scaled rotation ⊂ affine). BILINEAR is excluded: its 4x4 form can't
# round-trip through ``RigidTransform.matrix``.
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
    from ._stackreg import StackReg

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
    mask: np.ndarray | None = None,
    transformation: str = "affine",
) -> RegistrationResult:
    """Estimate one transform per frame against a fixed reference.

    reference: ``"mean"`` (default), ``"first"``, or ``"previous"``.
    transformation: TurboReg model — ``"translation"``, ``"rigid_body"``
        (alias ``"rigid"``), ``"scaled_rotation"``, or ``"affine"`` (default). The
        full estimated matrix is kept, so scale/shear survives to warping and ROIs.
    mask: optional frame-shaped ``(Y, X)`` boolean array, True where a pixel may
        take part in the fit. Static — the same pixels are compared in every frame
        — so the estimate tracks the tissue you outlined instead of a bright
        static background or a neighbouring leaf. Cut it a couple of pixels wider
        than what you are excluding; the spline interpolation and the pyramid
        spread a feature about that far past its own edge.
    """
    from ._stackreg import StackReg

    if reference not in ("mean", "first", "previous"):
        raise ValueError(f"Unknown reference {reference!r} (expected 'mean' | 'first' | 'previous')")

    est = np.asarray(stack, dtype=float)
    masks = None
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != est.shape[1:]:
            raise ValueError(f"mask shape {mask.shape} does not match frames {est.shape[1:]}")
        # One static mask, presented per-frame: a broadcast view, so a long
        # recording costs one frame's worth of booleans, not T of them.
        masks = np.broadcast_to(mask, est.shape)
    sr = StackReg(_resolve_transformation(transformation))
    tmats = sr.register_stack(est, reference=reference, masks=masks)
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
    transformation: str = "affine",
) -> list[LeafRegion]:
    """Register each leaf box's sub-stack independently, mutating ``leaf_regions``.

    For each leaf: crop to its bbox, estimate per-frame transforms, store its
    reference, and flag drift-out frames in ``low_confidence_frames``. A leaf
    carrying a ``mask_polygon`` is estimated on the pixels inside that outline
    only (see ``leaf_mask``); the whole box is still warped.
    reference / transformation: as in ``register_whole_frame``.
    drift_frac: fraction of the box's smaller side a frame may shift before being
        flagged low-confidence.
    """
    stack = np.asarray(stack)
    for leaf in leaf_regions:
        y0, y1, x0, x1 = leaf.bbox
        sub = stack[:, y0:y1, x0:x1]
        leaf.transforms = register_whole_frame(
            sub, reference, mask=leaf_mask(leaf, sub.shape[1:]),
            transformation=transformation,
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
    from ._stackreg import StackReg

    if len(transforms) != len(stack):
        raise ValueError(
            f"transforms ({len(transforms)}) must match frames ({len(stack)})"
        )
    # The model only matters for *estimating* a transform; `transform` just applies
    # the 3x3 tmat we hand it, so one instance warps translation/rigid/scaled-
    # rotation/affine matrices alike.
    sr = StackReg(StackReg.RIGID_BODY)
    out = np.empty(np.shape(stack), dtype=float)
    for i, (frame, tf) in enumerate(zip(stack, transforms)):
        out[i] = sr.transform(np.asarray(frame, dtype=float), tmat=_rigid_to_matrix(tf))
    return out
