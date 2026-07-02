"""Publication-grade static figures (matplotlib). SPEC.md §4 (export).

pyqtgraph drives the *interactive* widgets; for the figures that go into a paper
we render the same ``Session`` arrays through matplotlib, which gives true vector
PDF/SVG with embedded, editable fonts and exact column-width sizing — the things
journals (and bioelectronics journals in particular) check. Keeping this separate
from the widgets follows the SPEC §2.2 rule that no business logic lives in the UI.

matplotlib is a core dependency but imported lazily here, so ``import caliana``
stays cheap for headless/GUI use that never renders a paper figure. The four
entry points mirror the analysis the tool already produces:

    plot_traces            stacked / overlaid ΔF/F traces with event markers
    plot_propagation       cross-ROI onset map + propagation arrow
    plot_roi_overlay       a frame / max-projection with ROI shapes drawn on it
    plot_imaging_electrode imaging trace aligned with an auxiliary electrode signal

Each returns a matplotlib ``Figure``; pass ``save=`` to also write a vector file.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence

import numpy as np

from .models import ROIShape

# Okabe-Ito colourblind-safe palette (the de-facto standard for physiology
# figures). Shared by every figure function so an ROI is the same colour in the
# overlay panel, its trace, and the propagation map — the figures stay
# cross-readable, which matters more in print than matching the interactive
# pyqtgraph HSV colours (pg.intColor) used live in the widgets.
ROI_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Single-column / double-column widths in inches (Nature-ish defaults; override
# per journal). Height is yours to choose via the `height` argument.
COL_SINGLE = 3.5
COL_DOUBLE = 7.2


def roi_color(i: int) -> str:
    """Stable per-ROI colour, cycled through the palette."""
    return ROI_COLORS[i % len(ROI_COLORS)]


@contextmanager
def paper_style(base_font_size: float = 8.0):
    """rcParams tuned for print: thin black open axes, outward ticks, sans-serif,
    embedded TrueType fonts (``fonttype 42``, required by most journals)."""
    import matplotlib as mpl

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_font_size,
        "axes.titlesize": base_font_size,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.0,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # editable TrueType, not paths
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    with mpl.rc_context(rc):
        yield


def _finish(fig, save: Optional[str], dpi: int):
    if save is not None:
        # Vector formats (.pdf/.svg) ignore dpi for the vector parts; it only
        # affects any rasterized layers. Inferred from the extension.
        fig.savefig(save, dpi=dpi)
    return fig


def _time_axis(session):
    """(x, label) — real seconds if the Timeline is calibrated, else frames.

    The axis follows the current (possibly cropped) traces: it uses the original
    frame index of each trace column (``Session.trace_frames``), so a cropped
    window still plots against its true recording frames/seconds and ``x`` always
    matches the trace length.

    Electrode co-analysis (SPEC §6) will populate ``frame_interval``; until then
    the model is frames-only, so figures default to a frame axis automatically.
    """
    frames = session.trace_frames()
    tl = session.timeline
    if tl is not None and tl.frame_interval:
        return frames * tl.frame_interval, "Time (s)"
    return frames, "Frame"


def _event_overlay(ax, session, x):
    """Draw stimulus/event markers as thin labelled lines at their frame index.

    Events are in original frame coordinates; only those falling inside the
    plotted (possibly cropped) window are drawn, positioned on the same
    frame/seconds axis as the traces.
    """
    tl = session.timeline
    if tl is None or not tl.events or len(x) == 0:
        return
    frames = session.trace_frames()
    lo, hi = int(frames[0]), int(frames[-1])
    scale = tl.frame_interval if tl.frame_interval else 1.0
    for ev in tl.events:
        if lo <= ev.frame <= hi:
            xv = ev.frame * scale
            ax.axvline(xv, color="0.4", lw=0.7, ls="--", zorder=0)
            if ev.label:
                ax.annotate(
                    ev.label, xy=(xv, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points",
                    fontsize=6, color="0.4", va="top", rotation=0,
                )


# --------------------------------------------------------------------------- #
# 1. ΔF/F traces (single or stacked)
# --------------------------------------------------------------------------- #
def plot_traces(
    session,
    *,
    use_dff: bool = True,
    stacked: bool = True,
    offset: Optional[float] = None,
    rois: Optional[Sequence[int]] = None,
    width: float = COL_SINGLE,
    height: Optional[float] = None,
    scalebar: bool = False,
    save: Optional[str] = None,
    dpi: int = 600,
):
    """Per-ROI fluorescence traces.

    stacked=True draws traces vertically offset (the usual physiology layout);
    stacked=False overlays them on a shared axis with a legend. ``offset`` sets
    the vertical spacing between stacked traces (auto = ~1.1× the largest range).
    ``scalebar=True`` removes the boxed y-axis and draws a ΔF/F + time scale bar
    instead, which reads better for many stacked traces.
    """
    import matplotlib.pyplot as plt

    if session.traces is None:
        raise RuntimeError("No traces; call extract_traces()/compute_dff() first.")
    tr = session.traces
    data = tr.dff if (use_dff and tr.dff is not None) else tr.raw
    if use_dff and tr.dff is None:
        raise RuntimeError("ΔF/F requested but not computed; call compute_dff().")
    ylabel = "ΔF/F" if (use_dff and tr.dff is not None) else "F (a.u.)"

    idx = list(rois) if rois is not None else list(range(data.shape[0]))
    labels = tr.labels or [f"ROI {i}" for i in range(data.shape[0])]
    x, xlabel = _time_axis(session)

    with paper_style():
        h = height if height is not None else (0.5 * len(idx) + 0.8 if stacked else 2.0)
        fig, ax = plt.subplots(figsize=(width, h))

        if stacked:
            step = offset if offset is not None else 1.1 * max(
                float(np.ptp(data[i])) for i in idx
            ) or 1.0
            yticks, yticklabels = [], []
            for row, i in enumerate(idx):
                base = row * step
                ax.plot(x, data[i] + base, color=roi_color(i), lw=1.0)
                yticks.append(base)
                yticklabels.append(labels[i])
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            if scalebar:
                _trace_scalebar(ax, x, step, ylabel)
            else:
                ax.set_ylabel(ylabel)
        else:
            for i in idx:
                ax.plot(x, data[i], color=roi_color(i), lw=1.0, label=labels[i])
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False, loc="upper right", ncol=1)

        ax.set_xlabel(xlabel)
        ax.margins(x=0)
        _event_overlay(ax, session, x)
        fig.tight_layout()
    return _finish(fig, save, dpi)


def _trace_scalebar(ax, x, dff_span, ylabel):
    """Replace the y-axis with a corner ΔF/F + time scale bar."""
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    # vertical bar = one stacked offset of ΔF/F; horizontal bar = ~10% of span.
    x0 = x[0]
    xspan = (x[-1] - x[0]) * 0.1
    y0 = ax.get_ylim()[0]
    ax.plot([x0, x0], [y0, y0 + dff_span], color="k", lw=1.2, clip_on=False)
    ax.plot([x0, x0 + xspan], [y0, y0], color="k", lw=1.2, clip_on=False)
    ax.annotate(f"{dff_span:.2g} {ylabel}", xy=(x0, y0 + dff_span / 2),
                xytext=(-4, 0), textcoords="offset points",
                rotation=90, va="center", ha="right", fontsize=6)
    ax.annotate(f"{xspan:.2g}", xy=(x0 + xspan / 2, y0),
                xytext=(0, -4), textcoords="offset points",
                va="top", ha="center", fontsize=6)


# --------------------------------------------------------------------------- #
# 2. Cross-ROI propagation
# --------------------------------------------------------------------------- #
def plot_propagation(
    session,
    *,
    background: bool = True,
    arrow: bool = True,
    width: float = COL_SINGLE,
    height: Optional[float] = None,
    save: Optional[str] = None,
    dpi: int = 600,
):
    """ROI positions coloured by response onset time, with the fitted propagation
    direction arrow and a speed annotation. Reads the ``propagation`` analysis
    result (run ``session.cross_roi_propagation()`` first).
    """
    import matplotlib.pyplot as plt

    res = session.analyses.get("propagation")
    if res is None:
        raise RuntimeError("No propagation result; call cross_roi_propagation().")
    onsets = np.asarray(res["onsets"], dtype=float)
    coords = np.array([r.center for r in session.rois], dtype=float)  # (y, x)

    with paper_style():
        h = height if height is not None else width
        fig, ax = plt.subplots(figsize=(width, h), layout="constrained")

        if background and session.data is not None:
            ax.imshow(session.max_projection(), cmap="gray", origin="upper")

        valid = ~np.isnan(onsets)
        sc = ax.scatter(
            coords[valid, 1], coords[valid, 0],
            c=onsets[valid], cmap="viridis", s=60,
            edgecolors="white", linewidths=0.8, zorder=3,
        )
        # ROIs whose onset couldn't be detected: open grey markers.
        if (~valid).any():
            ax.scatter(coords[~valid, 1], coords[~valid, 0], facecolors="none",
                       edgecolors="0.6", s=60, linewidths=0.8, zorder=3)

        src = res.get("source_roi")
        if src is not None:
            ax.scatter(coords[src, 1], coords[src, 0], marker="*", s=160,
                       facecolor="none", edgecolor="red", linewidths=1.2, zorder=4)

        if arrow and res.get("direction") is not None:
            dy, dx = res["direction"]
            cy, cx = coords[valid].mean(axis=0)
            L = 0.25 * max(session.data.shape[1:]) if session.data is not None else 30
            ax.annotate("", xy=(cx + dx * L, cy + dy * L), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))

        speed = res.get("speed_px_per_frame")
        if speed is not None and np.isfinite(speed):
            unit = "px/frame"
            tl = session.timeline
            if tl is not None and tl.frame_interval:
                speed = speed / tl.frame_interval  # px/frame -> px/s
                unit = "px/s"
            ax.set_title(f"propagation speed ≈ {speed:.2g} {unit}")

        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("onset (frame)" if (session.timeline is None or not
                     session.timeline.frame_interval) else "onset (s)")
        cb.outline.set_linewidth(0.6)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_aspect("equal")
    return _finish(fig, save, dpi)


# --------------------------------------------------------------------------- #
# 3. Image + ROI overlay
# --------------------------------------------------------------------------- #
def plot_roi_overlay(
    session,
    *,
    frame: Optional[int] = None,
    cmap: str = "gray",
    show_labels: bool = True,
    show_leaf_boxes: bool = True,
    width: float = COL_SINGLE,
    height: Optional[float] = None,
    save: Optional[str] = None,
    dpi: int = 600,
):
    """A background image with the ROI shapes drawn on top, ROI colours matching
    the trace/propagation figures. ``frame=None`` uses the max-projection
    (best for showing where signal occurred); pass an int for a single frame.
    Leaf-registration boxes are outlined if present.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    if session.data is None:
        raise RuntimeError("No data loaded.")
    img = session.max_projection() if frame is None else session._working_stack()[frame]

    with paper_style():
        h = height if height is not None else width
        fig, ax = plt.subplots(figsize=(width, h))
        ax.imshow(img, cmap=cmap, origin="upper")

        if show_leaf_boxes:
            for lr in session.leaf_regions:
                y0, y1, x0, x1 = lr.bbox
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="white", lw=0.8, ls="--"))

        labels = (session.traces.labels if session.traces else None) or [
            r.label or f"ROI {i}" for i, r in enumerate(session.rois)
        ]
        for i, r in enumerate(session.rois):
            cy, cx = r.center
            col = roi_color(i)
            if r.shape == ROIShape.CIRCLE:
                ax.add_patch(Circle((cx, cy), r.size, fill=False, edgecolor=col, lw=1.2))
            else:
                ax.add_patch(Rectangle((cx - r.size, cy - r.size), 2 * r.size,
                                       2 * r.size, fill=False, edgecolor=col, lw=1.2))
            if show_labels:
                ax.annotate(labels[i], xy=(cx, cy - r.size), xytext=(0, 2),
                            textcoords="offset points", color=col, fontsize=6,
                            ha="center", va="bottom")

        # A 50-px scale bar (no physical calibration in this model — SPEC §3).
        ax.plot([0.05, 0.05], [0, 0], color="w")  # keep autoscale sane
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_aspect("equal")
        fig.tight_layout()
    return _finish(fig, save, dpi)


# --------------------------------------------------------------------------- #
# 4. Imaging + electrode overlay (SPEC §6 forward-compat)
# --------------------------------------------------------------------------- #
def plot_imaging_electrode(
    session,
    aux_time: np.ndarray,
    aux_signal: np.ndarray,
    *,
    roi: int = 0,
    use_dff: bool = True,
    aux_label: str = "Electrode (mV)",
    width: float = COL_SINGLE,
    height: float = 2.6,
    save: Optional[str] = None,
    dpi: int = 600,
):
    """Stack a calcium trace over an auxiliary electrode time-series on a shared
    real-time x-axis. The electrode signal is passed explicitly (``aux_time`` in
    seconds, ``aux_signal`` same length) because ``Session`` has no ``aux_signals``
    slot yet — this is the SPEC §6 forward-compat case; when that slot lands,
    swap the args for ``session.aux_signals[...]``.

    The imaging trace needs a real seconds axis to align with the electrode, so
    set ``session.timeline.frame_interval`` before calling (otherwise it falls
    back to frame indices, which won't line up with the electrode time base).
    """
    import matplotlib.pyplot as plt

    if session.traces is None:
        raise RuntimeError("No traces; call extract_traces()/compute_dff() first.")
    tr = session.traces
    y = (tr.dff if (use_dff and tr.dff is not None) else tr.raw)[roi]
    ylabel = "ΔF/F" if (use_dff and tr.dff is not None) else "F (a.u.)"
    x, xlabel = _time_axis(session)
    label = (tr.labels[roi] if tr.labels else f"ROI {roi}")

    with paper_style():
        fig, (ax_im, ax_el) = plt.subplots(
            2, 1, figsize=(width, height), sharex=True,
            gridspec_kw={"hspace": 0.12},
        )
        ax_im.plot(x, y, color=roi_color(roi), lw=1.0)
        ax_im.set_ylabel(f"{label}\n{ylabel}")
        ax_im.margins(x=0)

        ax_el.plot(aux_time, aux_signal, color="0.15", lw=0.8)
        ax_el.set_ylabel(aux_label)
        ax_el.set_xlabel(xlabel if xlabel == "Time (s)" else "Time (s)  [set frame_interval]")
        ax_el.margins(x=0)

        _event_overlay(ax_im, session, x)
        _event_overlay(ax_el, session, x)
        fig.align_ylabels([ax_im, ax_el])
        fig.tight_layout()
    return _finish(fig, save, dpi)
