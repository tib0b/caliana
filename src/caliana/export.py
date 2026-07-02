"""Export & provenance. SPEC.md §4.

- traces -> CSV (raw F + ΔF/F, rows = frames, cols = ROIs)
- processed/registered stack -> TIFF
- provenance -> JSON sidecar so any analysis is reproducible
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .models import Traces


def traces_to_csv(traces: Traces, path, timeline=None, frames=None) -> None:
    """Write per-ROI raw F and ΔF/F over time to CSV. SPEC §4.

    Columns: a frame (and seconds, if calibrated) axis, then ``<label>_F`` and
    ``<label>_dFF`` per ROI. Rows are timepoints.

    ``frames`` gives the original frame index of each trace column so a cropped
    window still reports the true recording frames/seconds (see
    ``Session.trace_frames``); it defaults to ``0..T-1`` when omitted.
    """
    raw = traces.raw
    if raw.size == 0:
        raise ValueError("No traces to export")
    n_roi, T = raw.shape
    labels = traces.labels or [f"roi_{i}" for i in range(n_roi)]
    has_dff = traces.dff is not None

    frames = np.arange(T) if frames is None else np.asarray(frames)
    seconds = timeline.seconds_for(frames) if timeline is not None else None

    header = ["frame"] + (["seconds"] if seconds is not None else [])
    for lab in labels:
        header.append(f"{lab}_F")
        if has_dff:
            header.append(f"{lab}_dFF")

    with open(Path(path), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for t in range(T):
            row = [int(frames[t])] + ([float(seconds[t])] if seconds is not None else [])
            for i in range(n_roi):
                row.append(raw[i, t])
                if has_dff:
                    row.append(traces.dff[i, t])
            writer.writerow(row)


def stack_to_tiff(stack: np.ndarray, path) -> None:
    """Write a (registered/downsampled) image stack to TIFF. SPEC §4."""
    import tifffile

    tifffile.imwrite(str(Path(path)), np.asarray(stack))


def write_provenance(session, path) -> None:
    """Dump the run's full parameter record as a JSON sidecar. SPEC §4."""
    Path(path).write_text(json.dumps(session.provenance(), indent=2))
