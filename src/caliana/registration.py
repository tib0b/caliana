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
    """Build the registration target. SPEC §3: default mean image, fallback first frame."""
    if reference == "mean":
        return stack.mean(axis=0)
    if reference == "first":
        return stack[0]
    raise ValueError(f"Unknown reference {reference!r} (expected 'mean' | 'first')")


def _matrix_to_rigid(m: np.ndarray) -> RigidTransform:
    """Decompose a pystackreg rigid-body 3x3 homogeneous matrix.

    The matrix acts on (x, y, 1): rotation about the image origin then
    translation. For a pure rigid body this decomposition is exact, so the
    recomposed matrix (see ``_rigid_to_matrix``) round-trips identically.
    """
    theta = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    return RigidTransform(dy=float(m[1, 2]), dx=float(m[0, 2]), theta=float(theta))


def _rigid_to_matrix(tf: RigidTransform) -> np.ndarray:
    t = np.radians(tf.theta)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, tf.dx], [s, c, tf.dy], [0.0, 0.0, 1.0]], dtype=float)


def register_whole_frame(stack: np.ndarray, reference: str = "mean") -> RegistrationResult:
    """One rigid transform per frame vs the reference. SPEC §3 Stage II.

    Estimates per-frame (dy, dx, theta) with pystackreg's RIGID_BODY model
    against a fixed reference (mean image by default, or first frame).
    """
    from pystackreg import StackReg

    if reference not in ("mean", "first"):
        raise ValueError(f"Unknown reference {reference!r} (expected 'mean' | 'first')")

    sr = StackReg(StackReg.RIGID_BODY)
    with warnings.catch_warnings():
        # Our contract fixes time on axis 0; silence pystackreg's axis heuristic.
        warnings.filterwarnings("ignore", message=".*possible time axis.*")
        tmats = sr.register_stack(np.asarray(stack, dtype=float), reference=reference)
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
) -> list[LeafRegion]:
    """Register each leaf box's sub-stack independently, in place. SPEC §3 Stage II.

    For each leaf: crop to bbox, build its own reference, estimate per-frame rigid
    transforms (reusing the whole-frame estimator on the sub-stack), and flag
    frames whose offset approaches the box margin as low-confidence.
    """
    stack = np.asarray(stack)
    for leaf in leaf_regions:
        y0, y1, x0, x1 = leaf.bbox
        sub = stack[:, y0:y1, x0:x1]
        leaf.transforms = register_whole_frame(sub, reference).transforms
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
    """Warp each frame by its rigid transform to produce a stabilized stack.

    Returns a float stack aligned to the registration reference. Uses the same
    pystackreg backend (and matrix convention) as estimation, so it round-trips.
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
