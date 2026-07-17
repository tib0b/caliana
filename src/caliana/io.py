"""Loading & downsample-on-load.

Reads ``.tif``/``.tiff`` and ``.nd2`` into a single-channel ``[T, Y, X]`` stack.
Optional readers (nd2) are imported lazily so ``import caliana`` works before they
are installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import ImportParams, SourceInfo


def load_stack(path, params: ImportParams | None = None) -> tuple[np.ndarray, SourceInfo]:
    """Load a ``.tif``/``.tiff``/``.nd2`` file → ``(data [T, Y, X], SourceInfo)``.

    ``params`` (default: no downsampling) selects the channel, temporal window and
    temporal/spatial downsampling. Raises ``ValueError`` on an unsupported suffix.
    """
    path = Path(path)
    params = params or ImportParams()

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        raw, meta = _read_tiff(path)
    elif suffix == ".nd2":
        raw, meta = _read_nd2(path)
    else:
        raise ValueError(f"Unsupported format {suffix!r} (expected .tif/.tiff/.nd2)")

    data = _apply_import_params(raw, params)
    return data, SourceInfo(path=path, import_params=params, metadata=meta)


def _read_tiff(path: Path) -> tuple[np.ndarray, dict]:
    import tifffile

    return np.asarray(tifffile.imread(str(path))), {}


def _read_nd2(path: Path) -> tuple[np.ndarray, dict]:
    """Read a Nikon .nd2 lazily as a dask-backed array in canonical axis order.

    Returns a (possibly dask) array with axes ``[T, (C,) Y, X]`` plus metadata.
    Lazy reading lets the import-time temporal crop load only the needed frames
    (multi-GB nd2 files never fully materialize unless the params ask for it).
    Extra dimensions (e.g. Z, multipoint) are reduced to their first index.
    """
    import nd2

    xarr = nd2.imread(str(path), dask=True, xarray=True)  # named dims, lazy
    keep = ("T", "C", "Y", "X")
    # Reduce any dim we don't model (Z, P, ...) to its first index.
    for dim in [d for d in xarr.dims if d not in keep]:
        xarr = xarr.isel({dim: 0})
    xarr = xarr.transpose(*[d for d in keep if d in xarr.dims])

    arr = xarr.data  # dask array
    if "T" not in xarr.dims:
        arr = arr[None, ...]  # ensure a leading time axis
    return arr, {"nd2_sizes": {k: int(v) for k, v in dict(xarr.sizes).items()}}


def _materialize(arr) -> np.ndarray:
    """Compute a dask array if needed, else pass a numpy array through."""
    arr = arr.compute() if hasattr(arr, "compute") else arr
    return np.ascontiguousarray(arr)


def _apply_import_params(raw, params: ImportParams) -> np.ndarray:
    """Apply channel select, temporal crop, temporal/spatial downsample, crop window.

    Operations are backend-agnostic (numpy or lazy dask); the result is
    materialized only at the end, after crops have shrunk it.
    """
    arr = raw
    orig_dtype = arr.dtype

    # Single-channel model: collapse an extra channel axis if present.
    if arr.ndim == 4:
        arr = arr[:, params.channel]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D [T,Y,X] (or 4D [T,C,Y,X]); got shape {arr.shape}")

    # Temporal crop (applied first so lazy backends load only these frames).
    arr = arr[params.start:params.end]

    # Temporal downsample: average every `temporal_step` frames.
    step = params.temporal_step
    if step > 1:
        n = len(arr) // step
        arr = arr[:n * step].reshape(n, step, *arr.shape[1:]).mean(axis=1).astype(orig_dtype)

    # Spatial crop window (y0, y1, x0, x1).
    if params.spatial_window is not None:
        y0, y1, x0, x1 = params.spatial_window
        arr = arr[:, y0:y1, x0:x1]

    # Spatial downsample (stride subsample).
    s = params.spatial_step
    if s > 1:
        arr = arr[:, ::s, ::s]

    return _materialize(arr)
