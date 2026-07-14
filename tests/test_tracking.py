"""ROI motion tracking: move the ROI with the tissue, sample raw pixels.

Complements test_registration.py (which warps the stack). Here the stack is left
raw and each ROI is carried by the per-frame transforms, so a well-tracked ROI
sees a *constant* trace even though its tissue moves across the frame.
Runnable with pytest or as a script (`python tests/test_tracking.py`).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift

import caliana
from caliana.models import (
    ImportParams,
    LeafRegion,
    RegistrationMode,
    RegistrationResult,
    ROI,
    ROIShape,
    RigidTransform,
)
from caliana.registration import map_point, segment_tissue
from caliana.roi import extract_trace_tracked, move_roi


def _blob(h, w, cy, cx, sigma=4.0, amp=200.0):
    yy, xx = np.mgrid[:h, :w]
    return amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))


def test_map_point_matches_widget_convention():
    # Pure translation dx=+3: a point moves +3 in x, unchanged in y.
    y, x = map_point((16.0, 12.0), RigidTransform(dx=3.0), (0.0, 0.0))
    assert abs(y - 16.0) < 1e-9 and abs(x - 15.0) < 1e-9
    # Box-local origin offsets both in and out symmetrically.
    y, x = map_point((16.0, 12.0), RigidTransform(dx=3.0), (10.0, 4.0))
    assert abs(y - 16.0) < 1e-9 and abs(x - 15.0) < 1e-9


def test_move_polygon_translates_and_rotates_vertices():
    poly = ROI(center=(12.0, 8.0), size=0.0, shape=ROIShape.POLYGON,
               vertices=[(10, 6), (10, 10), (14, 10), (14, 6)])
    moved = move_roi(poly, RigidTransform(dx=3.0), (0.0, 0.0))
    got = {(round(y), round(x)) for y, x in moved.vertices}
    assert got == {(10, 9), (10, 13), (14, 13), (14, 9)}


def test_tracked_trace_is_flat_while_static_trace_decays():
    """A blob translating +2 px/frame; an ROI tracking it stays flat, a static
    ROI at the reference position loses the blob."""
    h = w = 48
    n = 8
    cx0, cy0 = 16.0, 24.0
    stack = np.stack([_blob(h, w, cy0, cx0 + 2.0 * k) for k in range(n)])
    # Transform mapping reference->raw for frame k is a +2k shift in x.
    transforms = [RigidTransform(dx=2.0 * k) for k in range(n)]
    roi = ROI(center=(cy0, cx0), size=3.0)

    tracked = extract_trace_tracked(stack, roi, transforms)
    static = np.stack([stack[k][_mask(roi, (h, w))] for k in range(n)]).mean(axis=1)

    # Tracked: nearly constant (follows the peak). Static: monotonically fading.
    assert tracked.std() / tracked.mean() < 0.02
    assert static[-1] < 0.3 * static[0]
    assert tracked.mean() > 2 * static.mean()


def _mask(roi, shape):
    from caliana.roi import roi_mask
    return roi_mask(roi, shape)


def test_session_tracking_pipeline_whole_frame():
    """register(apply=False) sets track_motion; extract_traces then follows tissue."""
    h = w = 40
    n = 6
    base = _blob(h, w, 20, 20, sigma=5) + _blob(h, w, 12, 28, sigma=3)
    shifts = [(0.0, 0.0), (1.0, 2.0), (2.0, 3.0), (2.0, 4.0), (3.0, 4.0), (3.0, 5.0)]
    stack = np.stack([shift(base, s, order=1, mode="nearest") for s in shifts])

    s = caliana.Session()
    s.data = stack
    from caliana.timeline import Timeline
    s.timeline = Timeline(n_frames=n)

    s.register(RegistrationMode.WHOLE_FRAME, reference="first", apply=False)
    assert s.track_motion and s.registered_data is None      # raw kept, no warp
    roi = s.add_roi(center=(20, 20), size=3, label="c")

    tracked = s.extract_traces().raw[0]
    # Compare to a naive static extraction on the raw stack.
    static = caliana.roi.extract_trace(stack, roi)
    assert tracked.std() < static.std()                       # steadier on tissue
    assert s.provenance()["registration"]["motion_tracking"] is True


def test_segment_tissue_finds_dark_low_texture_region():
    """Dark blob on a bright noisy background -> mask selects the dark region."""
    rng = np.random.default_rng(0)
    img = 200 + 20 * rng.standard_normal((60, 60))            # bright, textured bg
    img[20:40, 25:45] = 10.0                                  # dark tissue patch
    mask = segment_tissue(img, dark=True)
    assert mask[30, 35]                                       # inside the patch
    assert not mask[5, 5]                                     # background
    assert 0.05 < mask.mean() < 0.25                          # ~ the 0.11 patch fraction


def test_real_frames_tracking_runs_end_to_end():
    """Smoke test on the extracted recording frames, if present."""
    import glob
    from PIL import Image
    paths = sorted(glob.glob(
        __import__("os").path.join(__import__("os").path.dirname(__file__),
                                   "..", "..", "data", "compressed",
                                   "extracted_frames", "*.png")))
    if len(paths) < 3:
        print("real frames not present; skipping")
        return
    stack = np.stack([np.array(Image.open(p), dtype=float) for p in paths])
    s = caliana.Session()
    s.data = stack
    from caliana.timeline import Timeline
    s.timeline = Timeline(n_frames=len(stack))
    # One generous leaf box over the lower-left leaf; mask-driven per-leaf tracking.
    s.add_leaf_region((300, 470, 40, 210), label="lower-left")
    s.register(RegistrationMode.PER_LEAF, reference="mean", mask=True, apply=False)
    assert s.track_motion
    leaf = s.leaf_regions[0]
    assert len(leaf.transforms) == len(stack)
    s.add_roi(center=(360, 120), size=6, label="leaf-roi")
    tr = s.extract_traces().raw
    assert tr.shape == (1, len(stack)) and np.isfinite(tr).all()
    print("real-frames per-leaf tracking OK; "
          f"leaf drift dx span={max(t.dx for t in leaf.transforms) - min(t.dx for t in leaf.transforms):.2f}px")


if __name__ == "__main__":
    test_map_point_matches_widget_convention()
    test_move_polygon_translates_and_rotates_vertices()
    test_tracked_trace_is_flat_while_static_trace_decays()
    test_session_tracking_pipeline_whole_frame()
    test_segment_tissue_finds_dark_low_texture_region()
    test_real_frames_tracking_runs_end_to_end()
    print("all tracking tests passed")
