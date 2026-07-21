"""Export & provenance.

- traces → CSV (raw F + ΔF/F, rows = frames, cols = ROIs)
- processed/registered stack → TIFF
- provenance → JSON sidecar so any analysis is reproducible
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .models import Traces


def traces_to_csv(traces: Traces, path, timeline=None, frames=None) -> None:
    """Write per-ROI raw F, ΔF/F, and smoothed traces over time to a CSV at ``path``.

    Columns: ``frame`` (plus ``seconds`` if ``timeline`` is calibrated), then
    ``<label>_F``, ``<label>_dFF`` (if computed), and ``<label>_smoothed`` (if
    ``smooth_traces`` was run) per ROI; rows are timepoints. ``frames`` gives the
    original frame index of each column (defaults to ``0..T-1``) so a cropped
    window still reports true recording frames/seconds. Raises ``ValueError`` if
    there are no traces.
    """
    raw = traces.raw
    if raw.size == 0:
        raise ValueError("No traces to export")
    n_roi, T = raw.shape
    labels = traces.labels or [f"roi_{i}" for i in range(n_roi)]
    has_dff = traces.dff is not None
    has_smoothed = traces.smoothed is not None

    frames = np.arange(T) if frames is None else np.asarray(frames)
    seconds = timeline.seconds_for(frames) if timeline is not None else None

    header = ["frame"] + (["seconds"] if seconds is not None else [])
    for lab in labels:
        header.append(f"{lab}_F")
        if has_dff:
            header.append(f"{lab}_dFF")
        if has_smoothed:
            header.append(f"{lab}_smoothed")

    with open(Path(path), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for t in range(T):
            row = [int(frames[t])] + ([float(seconds[t])] if seconds is not None else [])
            for i in range(n_roi):
                row.append(raw[i, t])
                if has_dff:
                    row.append(traces.dff[i, t])
                if has_smoothed:
                    row.append(traces.smoothed[i, t])
            writer.writerow(row)


def stack_to_tiff(stack: np.ndarray, path) -> None:
    """Write an image stack to a TIFF at ``path``."""
    import tifffile

    tifffile.imwrite(str(Path(path)), np.asarray(stack))


def write_provenance(session, path) -> None:
    """Dump ``session.provenance()`` as a JSON sidecar at ``path``."""
    Path(path).write_text(json.dumps(session.provenance(), indent=2))
