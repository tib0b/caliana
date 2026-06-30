"""Whole-frame rigid registration. SPEC.md §3 Stage II.

Injects known motion into a structured base image, then checks that
registration recovers it and that applying the correction restores alignment.
Runnable with pytest or as a script (`python tests/test_registration.py`).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate, shift

import caliana
from caliana.models import LeafRegion
from caliana.registration import (
    apply_per_leaf,
    apply_transforms,
    register_per_leaf,
    register_whole_frame,
)


def _base_image(h=80, w=80):
    """A frame with off-centre structure so registration has features to lock on."""
    yy, xx = np.mgrid[:h, :w]
    blob = np.exp(-(((yy - 30) ** 2 + (xx - 50) ** 2) / (2 * 8.0 ** 2)))
    blob += 0.4 * np.exp(-(((yy - 55) ** 2 + (xx - 25) ** 2) / (2 * 5.0 ** 2)))
    return (100 + 150 * blob).astype(float)


def _interior_mse(a, b, m=12):
    """MSE on the interior only, ignoring warp border padding."""
    sl = (slice(m, -m), slice(m, -m))
    return float(np.mean((a[sl] - b[sl]) ** 2))


def test_translation_recovered():
    base = _base_image()
    shifts = [(0.0, 0.0), (3.0, -2.0), (-4.0, 5.0), (2.0, 6.0)]
    stack = np.stack([shift(base, s, order=1, mode="nearest") for s in shifts])

    reg = register_whole_frame(stack, reference="first")
    # Frame 0 is the reference -> ~zero transform.
    assert abs(reg.transforms[0].dx) < 0.5 and abs(reg.transforms[0].dy) < 0.5

    stabilized = apply_transforms(stack, reg.transforms)
    # Every stabilized frame should match the reference far better than the raw one.
    for i in range(1, len(stack)):
        before = _interior_mse(stack[i], base)
        after = _interior_mse(stabilized[i], base)
        assert after < 0.1 * before, f"frame {i}: after={after:.3g} before={before:.3g}"


def test_translation_plus_rotation_stabilizes():
    base = _base_image()
    frames = [base]
    for dy, dx, ang in [(2.0, -3.0, 4.0), (-3.0, 4.0, -5.0), (5.0, 2.0, 3.0)]:
        f = rotate(base, ang, reshape=False, order=1, mode="nearest")
        frames.append(shift(f, (dy, dx), order=1, mode="nearest"))
    stack = np.stack(frames)

    reg = register_whole_frame(stack, reference="first")
    stabilized = apply_transforms(stack, reg.transforms)
    for i in range(1, len(stack)):
        before = _interior_mse(stack[i], base)
        after = _interior_mse(stabilized[i], base)
        assert after < 0.5 * before, f"frame {i}: after={after:.3g} before={before:.3g}"


def test_session_uses_stabilized_stack():
    base = _base_image()
    stack = np.stack([shift(base, (i, -i), order=1, mode="nearest") for i in range(5)])

    s = caliana.Session()
    s.data = stack
    s.timeline = caliana.Timeline(n_frames=len(stack))
    s.register(caliana.RegistrationMode.WHOLE_FRAME, reference="first")

    assert s.registered_data is not None
    assert s.registered_data.shape == stack.shape
    # A static ROI on moving tissue should be steadier on the stabilized stack.
    s.add_roi(center=(30, 50), size=4)
    s.extract_traces()
    assert s.traces.raw.shape == (1, len(stack))


def _two_leaf_stack(shifts_a, shifts_b):
    """An 80x80 stack with two boxes whose contents move independently."""
    def patch(cy, cx, sig, h=30, w=30):
        yy, xx = np.mgrid[:h, :w]
        return 100 + 150 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)))

    base_a, base_b = patch(15, 15, 5), patch(12, 18, 4)
    T = len(shifts_a)
    stk = np.full((T, 80, 80), 100.0)
    for t in range(T):
        stk[t, 5:35, 5:35] = shift(base_a, shifts_a[t], order=1, mode="nearest")
        stk[t, 45:75, 45:75] = shift(base_b, shifts_b[t], order=1, mode="nearest")
    return stk


def _box_residual(arr, ref_stack, leaves, m=6):
    total = 0.0
    for leaf in leaves:
        y0, y1, x0, x1 = leaf.bbox
        ref = ref_stack[0, y0:y1, x0:x1]
        total += np.mean([_interior_mse(arr[t, y0:y1, x0:x1], ref, m)
                          for t in range(1, len(arr))])
    return total


def test_per_leaf_stabilizes_independent_motion():
    sa = [(0, 0), (2, -1), (-2, 2), (3, 1), (-1, -2), (1, 3)]
    sb = [(0, 0), (-2, 2), (2, -3), (-3, -1), (1, 2), (-1, -2)]
    stk = _two_leaf_stack(sa, sb)
    leaves = [LeafRegion(bbox=(5, 35, 5, 35)), LeafRegion(bbox=(45, 75, 45, 75))]

    register_per_leaf(stk, leaves, reference="first")
    comp = apply_per_leaf(stk, leaves)

    # Each leaf box is stabilized relative to its own frame 0.
    for leaf in leaves:
        y0, y1, x0, x1 = leaf.bbox
        ref = stk[0, y0:y1, x0:x1]
        before = np.mean([_interior_mse(stk[t, y0:y1, x0:x1], ref, 6) for t in range(1, len(stk))])
        after = np.mean([_interior_mse(comp[t, y0:y1, x0:x1], ref, 6) for t in range(1, len(stk))])
        assert after < 0.3 * before, f"{leaf.bbox}: after={after:.3g} before={before:.3g}"

    # Per-leaf beats a single whole-frame transform under independent motion.
    wf = apply_transforms(stk, register_whole_frame(stk, "first").transforms)
    assert _box_residual(comp, stk, leaves) < _box_residual(wf, stk, leaves)


def test_per_leaf_flags_drift_out():
    stk = _two_leaf_stack([(0, 0), (1, 0), (9, 0), (1, 0)], [(0, 0)] * 4)
    leaves = [LeafRegion(bbox=(5, 35, 5, 35))]  # 30 px box; 9 px > 0.25*30
    register_per_leaf(stk, leaves, reference="first", drift_frac=0.25)
    assert 2 in leaves[0].low_confidence_frames


def test_session_per_leaf_pipeline():
    stk = _two_leaf_stack([(0, 0), (2, -1), (-2, 2), (3, 1)],
                          [(0, 0), (-2, 2), (2, -3), (-3, -1)])
    s = caliana.Session()
    s.data = stk
    s.timeline = caliana.Timeline(n_frames=len(stk))
    s.add_leaf_region((5, 35, 5, 35))
    s.add_leaf_region((45, 75, 45, 75))
    s.register(caliana.RegistrationMode.PER_LEAF, reference="first")

    assert s.registered_data.shape == stk.shape
    roi = s.add_roi(center=(20, 20), size=4)        # inside leaf box 0
    assert roi.leaf_region == 0
    s.extract_traces()
    assert s.traces.raw.shape == (1, len(stk))
    assert len(s.provenance()["registration"]["leaf_regions"]) == 2


if __name__ == "__main__":
    test_translation_recovered()
    test_translation_plus_rotation_stabilizes()
    test_session_uses_stabilized_stack()
    test_per_leaf_stabilizes_independent_motion()
    test_per_leaf_flags_drift_out()
    test_session_per_leaf_pipeline()
    print("registration tests OK")
