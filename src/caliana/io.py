"""Loading & downsample-on-load.

Reads ``.tif``/``.tiff`` and ``.nd2`` into a single-channel ``[T, Y, X]`` stack.
Optional readers (nd2) are imported lazily so ``import caliana`` works before they
are installed.

Each reader also reports the recording's physical scale (µm per pixel, seconds
per frame) when the file declares it, so ``Session.load`` can calibrate the time
and space axes automatically. Scale reading is best-effort: files carry these
tags in several dialects and often not at all, so a missing or unparsable value
leaves that axis in frames/pixels rather than failing the load.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import FileScale, ImportParams, SourceInfo

# Length units → µm, spelled the way ImageJ / OME / TIFF tags spell them.
_LENGTH_TO_MICRON = {
    "m": 1e6, "meter": 1e6, "metre": 1e6,
    "cm": 1e4, "centimeter": 1e4, "centimetre": 1e4,
    "mm": 1e3, "millimeter": 1e3, "millimetre": 1e3,
    "µm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0,
    "micrometer": 1.0, "micrometre": 1.0,
    "nm": 1e-3, "nanometer": 1e-3, "nanometre": 1e-3,
    "inch": 25400.0, "in": 25400.0, '"': 25400.0,
}

# Time units → seconds (OME's TimeIncrementUnit; ImageJ's finterval is seconds).
_TIME_TO_SECOND = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "µs": 1e-6, "us": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
}

# TIFF baseline ResolutionUnit tag values (1 = none/unspecified → unusable).
_RESOLUTION_UNIT = {2: "inch", 3: "cm"}


def load_stack(path, params: ImportParams | None = None) -> tuple[np.ndarray, SourceInfo]:
    """Load a ``.tif``/``.tiff``/``.nd2`` file → ``(data [T, Y, X], SourceInfo)``.

    ``params`` (default: no downsampling) selects the channel, temporal window and
    temporal/spatial downsampling. The returned ``SourceInfo.scale`` carries the
    file's *native* µm/px and s/frame when declared (see ``FileScale``). Raises
    ``ValueError`` on an unsupported suffix.
    """
    path = Path(path)
    params = params or ImportParams()

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        raw, meta, scale = _read_tiff(path)
    elif suffix == ".nd2":
        raw, meta, scale = _read_nd2(path)
    else:
        raise ValueError(f"Unsupported format {suffix!r} (expected .tif/.tiff/.nd2)")

    data = _apply_import_params(raw, params)
    return data, SourceInfo(path=path, import_params=params, metadata=meta, scale=scale)


def _read_tiff(path: Path) -> tuple[np.ndarray, dict, FileScale]:
    """Read a (possibly ImageJ/OME) TIFF → ``(array, metadata, scale)``."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        arr = np.asarray(tif.asarray())
        ij = tif.imagej_metadata or {}
        meta = {
            "tiff_flavor": "imagej" if ij else ("ome" if tif.is_ome else "plain"),
            # `Info` holds the acquisition dump verbatim (kilobytes); keep the keys.
            "imagej_keys": sorted(ij),
        }
        scale = _scale_from_tiff(tif, ij)
    return arr, meta, scale


def _scale_from_tiff(tif, ij: dict) -> FileScale:
    """µm/px + s/frame from a TIFF, trying each dialect in turn.

    ImageJ first (what Fiji-exported recordings carry, and the most explicit
    about frame interval), then OME-XML, then the baseline resolution tags. Each
    axis is filled independently: a file may declare a pixel size but no frame
    interval, or the reverse.
    """
    pixel_size = _imagej_pixel_size(tif, ij)
    frame_interval = _imagej_frame_interval(ij)

    if tif.is_ome and (pixel_size is None or frame_interval is None):
        ome_pixel, ome_interval = _ome_scale(tif.ome_metadata)
        pixel_size = pixel_size if pixel_size is not None else ome_pixel
        frame_interval = frame_interval if frame_interval is not None else ome_interval

    if pixel_size is None:
        pixel_size = _resolution_tag_pixel_size(tif.pages[0])
    return FileScale(pixel_size=pixel_size, frame_interval=frame_interval)


def _imagej_frame_interval(ij: dict) -> float | None:
    """ImageJ's seconds-per-frame: ``finterval``, else the reciprocal of ``fps``."""
    interval = _positive(ij.get("finterval"))
    if interval is not None:
        return interval
    fps = _positive(ij.get("fps"))
    return 1.0 / fps if fps else None


def _imagej_pixel_size(tif, ij: dict) -> float | None:
    """µm per pixel from the XResolution tag read in ImageJ's declared ``unit``.

    ImageJ writes the resolution as pixels-per-unit in the standard tag and names
    the unit in its own metadata block (leaving ResolutionUnit as "none"), so the
    two have to be read together. ``unit`` of "pixel"/"" means uncalibrated.
    """
    if not ij:
        return None
    factor = _LENGTH_TO_MICRON.get(_normalize_unit(ij.get("unit")))
    if factor is None:
        return None
    per_unit = _tag_resolution(tif.pages[0], "XResolution")
    return factor / per_unit if per_unit else None


def _resolution_tag_pixel_size(page) -> float | None:
    """µm per pixel from the baseline XResolution + ResolutionUnit tags.

    The fallback for plain TIFFs, which can only express inches or centimetres —
    microscopy files usually land here only when written by generic tooling.
    """
    unit = _RESOLUTION_UNIT.get(_tag_value(page, "ResolutionUnit"))
    per_unit = _tag_resolution(page, "XResolution")
    if unit is None or not per_unit:
        return None
    return _LENGTH_TO_MICRON[unit] / per_unit


def _ome_scale(xml: str | None) -> tuple[float | None, float | None]:
    """``(µm/px, s/frame)`` from the first ``Pixels`` element of an OME-XML header.

    Units are explicit in OME (``PhysicalSizeXUnit``, ``TimeIncrementUnit``), both
    defaulting to µm/s per the spec.
    """
    if not xml:
        return None, None
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None, None
    pixels = next((el for el in root.iter() if el.tag.rsplit("}", 1)[-1] == "Pixels"), None)
    if pixels is None:
        return None, None

    size = _positive(pixels.get("PhysicalSizeX"))
    unit = _LENGTH_TO_MICRON.get(_normalize_unit(pixels.get("PhysicalSizeXUnit", "µm")))
    pixel_size = size * unit if (size and unit) else None

    increment = _positive(pixels.get("TimeIncrement"))
    tunit = _TIME_TO_SECOND.get(_normalize_unit(pixels.get("TimeIncrementUnit", "s")))
    frame_interval = increment * tunit if (increment and tunit) else None
    return pixel_size, frame_interval


def _read_nd2(path: Path) -> tuple[np.ndarray, dict, FileScale]:
    """Read a Nikon .nd2 lazily as a dask-backed array in canonical axis order.

    Returns a (possibly dask) array with axes ``[T, (C,) Y, X]``, metadata, and the
    acquisition's scale. Lazy reading lets the import-time temporal crop load only
    the needed frames (multi-GB nd2 files never fully materialize unless the params
    ask for it). Extra dimensions (e.g. Z, multipoint) are reduced to their first
    index.
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
    meta = {"nd2_sizes": {k: int(v) for k, v in dict(xarr.sizes).items()}}
    return arr, meta, _nd2_scale(path)


def _nd2_scale(path: Path) -> FileScale:
    """µm/px + s/frame from an nd2's acquisition settings.

    Reopens the file (cheap — headers only; the pixel data stays lazy) because
    ``nd2.imread`` returns the array alone. ``voxel_size().x`` is µm per pixel
    *as acquired*, i.e. already accounting for camera binning and zoom.
    """
    import nd2

    try:
        with nd2.ND2File(str(path)) as f:
            pixel_size = _positive(f.voxel_size().x)
            frame_interval = _nd2_frame_interval(f.experiment)
    except (OSError, ValueError, TypeError, AttributeError, IndexError, KeyError):
        # Absent/malformed acquisition metadata must not fail an otherwise fine
        # load; the caller just gets an uncalibrated recording.
        return FileScale()
    return FileScale(pixel_size=pixel_size, frame_interval=frame_interval)


def _nd2_frame_interval(experiment) -> float | None:
    """Seconds per frame from the nd2 experiment's time loop, if it has one.

    Two loop flavors carry it: ``TimeLoop`` (a single period) and ``NETimeLoop``
    (non-equidistant: a list of periods). We take the first period's *nominal*
    interval — the recordings this targets use one period, and their measured
    per-frame jitter is sub-millisecond around it. Loops without a usable period
    (z-stacks, multipoint) are skipped.
    """
    for loop in experiment or []:
        params = getattr(loop, "parameters", None)
        periods = getattr(params, "periods", None)
        if periods:                                  # NETimeLoop
            params = periods[0]
        period_ms = _positive(getattr(params, "periodMs", None))
        if period_ms:
            return period_ms / 1000.0
    return None


def _normalize_unit(unit) -> str:
    """Fold a unit string to the spelling used in the conversion tables.

    Both micro signs (U+00B5 and Greek mu) appear in the wild; they must land on
    the same key.
    """
    if not isinstance(unit, str):
        return ""
    return unit.strip().lower().replace("μ", "µ")


def _positive(value) -> float | None:
    """``value`` as a float if it is finite and > 0, else ``None``.

    Scale metadata is routinely present but meaningless (0, -1, NaN, a string);
    those all mean "not calibrated" here.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _tag_value(page, name):
    """Raw value of a TIFF tag, or ``None`` when the page lacks it."""
    tag = page.tags.get(name)
    return None if tag is None else tag.value


def _tag_resolution(page, name) -> float | None:
    """A resolution tag as pixels-per-unit (TIFF stores it as a rational pair)."""
    value = _tag_value(page, name)
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        return _positive(num / den) if den else None
    return _positive(value)


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
