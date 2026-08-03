"""Loading & downsample-on-load, incl. scale metadata. SPEC.md §3 Stage I.

The TIFF path is always exercised; the nd2 path is skipped unless the `nd2`
package and the sample recording are both present. Run as a script or pytest.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import tifffile

import caliana
from caliana.io import _read_nd2, _read_tiff
from caliana.models import ImportParams
from caliana.io import _apply_import_params

ROOT = Path(__file__).resolve().parent.parent
ND2 = ROOT / "data" / "original" / "wtgcamp33_plant8_800ms_1fps_40C_fullmcs.nd2"

try:
    import nd2  # noqa: F401
    HAVE_ND2 = True
except Exception:  # pragma: no cover
    HAVE_ND2 = False


def test_import_params_downsample():
    # 10 frames, 8x8: temporal average by 2, spatial stride 2, crop window.
    raw = np.arange(10 * 8 * 8, dtype=np.uint16).reshape(10, 8, 8)
    out = _apply_import_params(raw, ImportParams(start=0, end=8, temporal_step=2, spatial_step=2))
    assert out.shape == (4, 4, 4)          # 8 frames -> 4 averaged; 8px -> 4 strided
    # 4D channel selection.
    raw4 = np.zeros((6, 2, 8, 8), dtype=np.uint16)
    raw4[:, 1] = 7
    out4 = _apply_import_params(raw4, ImportParams(channel=1))
    assert out4.shape == (6, 8, 8) and out4.mean() == 7


def test_tiff_scale_metadata():
    """Pixel size / frame interval are read from each TIFF dialect we support, in
    whatever unit it declares, and left None when the file says nothing."""
    data = np.zeros((4, 6, 6), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # ImageJ: pixels-per-unit in the resolution tag + the unit in its own block.
        ij = d / "imagej.tif"
        tifffile.imwrite(ij, data, imagej=True, resolution=(1 / 0.65, 1 / 0.65),
                         metadata={"unit": "micron", "finterval": 0.25, "axes": "TYX"})
        scale = _read_tiff(ij)[2]
        assert abs(scale.pixel_size - 0.65) < 1e-9
        assert scale.frame_interval == 0.25

        # OME-XML: units are explicit attributes.
        ome = d / "ome.ome.tif"
        tifffile.imwrite(ome, data, metadata={
            "axes": "TYX", "PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "µm",
            "TimeIncrement": 2.0, "TimeIncrementUnit": "s"})
        scale = _read_tiff(ome)[2]
        assert (scale.pixel_size, scale.frame_interval) == (0.5, 2.0)

        # Baseline tags: 100 px per inch -> 254 µm per pixel.
        inch = d / "inch.tif"
        tifffile.imwrite(inch, data, resolution=(100, 100), resolutionunit="INCH",
                         photometric="minisblack")
        assert abs(_read_tiff(inch)[2].pixel_size - 254.0) < 1e-9

        # "pixel" units mean uncalibrated space; fps still gives the time axis.
        fps = d / "fps.tif"
        tifffile.imwrite(fps, data, imagej=True,
                         metadata={"unit": "pixel", "fps": 4.0, "axes": "TYX"})
        scale = _read_tiff(fps)[2]
        assert scale.pixel_size is None and scale.frame_interval == 0.25

        # A bare TIFF stays frames/pixels — no guessing.
        plain = d / "plain.tif"
        tifffile.imwrite(plain, data, photometric="minisblack")
        scale = _read_tiff(plain)[2]
        assert scale.pixel_size is None and scale.frame_interval is None


def test_load_scales_follow_downsampling():
    """Downsampling on load rescales the calibration: each kept frame/pixel spans
    `*_step` native ones, and the file's own values stay on SourceInfo."""
    data = np.zeros((8, 8, 8), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "calibrated.tif"
        tifffile.imwrite(path, data, imagej=True, resolution=(1 / 0.5, 1 / 0.5),
                         metadata={"unit": "micron", "finterval": 0.2, "axes": "TYX"})

        s = caliana.Session.from_file(path)
        assert (s.timeline.frame_interval, s.space.pixel_size) == (0.2, 0.5)

        s = caliana.Session.from_file(path, temporal_step=2, spatial_step=4)
        assert abs(s.timeline.frame_interval - 0.4) < 1e-9
        assert abs(s.space.pixel_size - 2.0) < 1e-9
        # Native (pre-downsample) scale is preserved for provenance.
        assert s.source.scale.frame_interval == 0.2 and s.source.scale.pixel_size == 0.5
        prov = s.provenance()
        assert abs(prov["scale"]["pixel_size"] - 2.0) < 1e-9
        assert prov["source"]["file_scale"]["pixel_size"] == 0.5

        # An explicit calibration overrides whatever the file said.
        s.set_pixel_size(1.25).set_frame_interval(3.0)
        assert (s.space.pixel_size, s.timeline.frame_interval) == (1.25, 3.0)


def test_nd2_lazy_load():
    if not (HAVE_ND2 and ND2.exists()):
        print("nd2 sample/lib not available; skipping")
        return
    arr, meta, scale = _read_nd2(ND2)
    assert hasattr(arr, "compute")          # lazy (dask) — not materialized
    assert arr.ndim in (3, 4)
    assert "nd2_sizes" in meta
    # This recording is 1 fps with 4x4-binned 81.4 µm pixels.
    assert abs(scale.frame_interval - 1.0) < 1e-6
    assert scale.pixel_size > 0

    # Only a few frames are materialized, despite the multi-GB source.
    s = caliana.Session.from_file(ND2, start=0, end=6, temporal_step=2)
    assert s.data.ndim == 3 and s.data.shape[0] == 3
    assert isinstance(s.data, np.ndarray)
    assert abs(s.timeline.frame_interval - 2.0) < 1e-6   # 1 s/frame × temporal_step


if __name__ == "__main__":
    test_import_params_downsample()
    test_tiff_scale_metadata()
    test_load_scales_follow_downsampling()
    test_nd2_lazy_load()
    print("io tests OK")
