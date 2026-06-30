"""Loading & downsample-on-load. SPEC.md §3 Stage I.

The TIFF path is always exercised; the nd2 path is skipped unless the `nd2`
package and the sample recording are both present. Run as a script or pytest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import caliana
from caliana.io import _read_nd2
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


def test_nd2_lazy_load():
    if not (HAVE_ND2 and ND2.exists()):
        print("nd2 sample/lib not available; skipping")
        return
    arr, meta = _read_nd2(ND2)
    assert hasattr(arr, "compute")          # lazy (dask) — not materialized
    assert arr.ndim in (3, 4)
    assert "nd2_sizes" in meta

    # Only a few frames are materialized, despite the multi-GB source.
    s = caliana.Session.from_file(ND2, start=0, end=6, temporal_step=2)
    assert s.data.ndim == 3 and s.data.shape[0] == 3
    assert isinstance(s.data, np.ndarray)


if __name__ == "__main__":
    test_import_params_downsample()
    test_nd2_lazy_load()
    print("io tests OK")
