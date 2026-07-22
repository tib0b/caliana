# Publication-grade static figures (matplotlib), rendered from ``Session`` arrays.

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence

import numpy as np

from .models import ROIShape

# Okabe-Ito colourblind-safe palette
# Shared by every figure function so an ROI is the same colour in the
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


def intensity_levels(image, pmax: float = 99.0):
    """Display range ``(min, pmax-th percentile)`` of an image's intensities.

    Anchoring the low end at the data minimum and clipping the high end at the
    ``pmax``-th percentile keeps a handful of outlier-bright pixels from
    flattening the contrast across the rest of the image. Falls back to the full
    range for near-flat images where that top percentile equals the minimum. This
    is the same scale the interactive preview/ROI widgets default to, shared here
    so a saved figure matches what was on screen.
    """
    image = np.asarray(image)
    lo = float(np.min(image))
    hi = float(np.percentile(image, pmax))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


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

    Follows the current (possibly cropped) traces via each column's original frame
    index (``Session.trace_frames``), so ``x`` always matches the trace length and
    a cropped window plots against its true recording frames/seconds.
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


# Clean matplotlib renderings that mirror the live pyqtgraph views — same data,
# overlays, and which-series-shown — restyled with the paper rcParams and the
# Okabe-Ito palette.

def _draw_overlay(ax, ov):
    """Draw one ROI/leaf overlay spec onto an image axis (see ``export_image``).

    ``ov`` is a dict with ``kind`` in {circle, rect, bbox, polygon}, a ``center``
    (y, x), a ``size`` (radius, px), optional ``bbox``/``vertices`` for the box
    and polygon kinds, and styling (``color``/``lw``/``ls``/``label``). Labels go
    in a top-right legend (``_overlay_legend``), not next to each shape — in-place
    labels collide as soon as ROIs sit close together.
    """
    from matplotlib.patches import Circle, Polygon, Rectangle

    color = ov.get("color", "white")
    lw = ov.get("lw", 0.8)
    ls = ov.get("ls", "-")
    kind = ov["kind"]
    if kind == "circle":
        cy, cx = ov["center"]
        ax.add_patch(Circle((cx, cy), ov["size"], fill=False, edgecolor=color, lw=lw, ls=ls))
    elif kind == "rect":
        cy, cx = ov["center"]
        r = ov["size"]
        ax.add_patch(Rectangle((cx - r, cy - r), 2 * r, 2 * r, fill=False,
                               edgecolor=color, lw=lw, ls=ls))
    elif kind == "bbox":
        y0, y1, x0, x1 = ov["bbox"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=color, lw=lw, ls=ls))
    elif kind == "polygon":
        ax.add_patch(Polygon([(x, y) for (y, x) in ov["vertices"]], closed=True,
                             fill=False, edgecolor=color, lw=lw, ls=ls))


def _overlay_legend(ax, overlays):
    """One legend entry per labelled overlay, stacked in the top-right corner.

    Each entry is just the label text in its overlay's colour — the line swatch
    is suppressed (``handlelength=0``), since the text colour already carries the
    mapping and dropping it keeps the legend compact over the image.
    """
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=ov.get("color", "white"), lw=1.0,
                      label=ov["label"]) for ov in overlays]
    leg = ax.legend(handles=handles, loc="upper right", ncol=1, frameon=False,
                    handlelength=0, handletextpad=0)
    for text, ov in zip(leg.get_texts(), overlays):
        text.set_color(ov.get("color", "white"))




def export_image(image, *, levels=None, cmap="gray", cbar_label=None,
                 overlays=None, title=None, width=COL_SINGLE, height=None,
                 save=None, dpi=600):
    """Clean render of a 2D image the way a pyqtgraph ImageView shows it.

    levels: ``(vmin, vmax)`` contrast pair (e.g. the view's current histogram
        levels); ``None`` autoscales. cmap: matplotlib colormap matching the
        view's gradient. cbar_label: draw a colourbar with this label if given.
    overlays: ROI/leaf shape specs drawn on top (see ``_draw_overlay``); their
        labels go in a legend above the axes. NaN pixels render transparent, as
        in the live view.
    """
    import matplotlib.pyplot as plt

    image = np.asarray(image, dtype=float)
    lo, hi = levels if levels is not None else (None, None)
    with paper_style():
        h = height if height is not None else width
        fig, ax = plt.subplots(figsize=(width, h))
        im = ax.imshow(image, cmap=cmap, origin="upper", vmin=lo, vmax=hi)
        for ov in overlays or []:
            _draw_overlay(ax, ov)
        if cbar_label is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(cbar_label)
            cb.outline.set_linewidth(0.6)
        if title:
            ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_aspect("equal")
        fig.tight_layout()
        labelled = [ov for ov in overlays or [] if ov.get("label")]
        if labelled:
            _overlay_legend(ax, labelled)
    return _finish(fig, save, dpi)


def export_traces(traces, *, x=None, xlabel="frame", ylabel="", labels=None,
                  events=None, regions=None, onsets=None, legend=True,
                  title=None, width=COL_DOUBLE, height=2.6, save=None, dpi=600):
    """Clean render of overlaid line traces as a pyqtgraph PlotWidget shows them.

    traces: iterable of 1D arrays (or a 2D ``[n, T]`` array). x: shared x values
        (frames or seconds); defaults to ``range``. events: ``(x, label)``
        vertical stimulus markers. regions: ``(lo, hi[, color])`` shaded bands
        (baseline / crop windows). onsets: per-trace onset x positions (dashed,
        coloured to match each trace; ``None``/NaN entries are skipped). Series
        colours use the Okabe-Ito palette, so ROI *i* matches the overlay and
        propagation figures.
    """
    import matplotlib.pyplot as plt

    data = [np.asarray(t) for t in traces]
    n = len(data)
    if x is None:
        x = np.arange(len(data[0])) if n else np.arange(0)
    labels = labels or [f"ROI {i}" for i in range(n)]
    with paper_style():
        fig, ax = plt.subplots(figsize=(width, height))
        for lo, hi, *rest in (regions or []):
            ax.axvspan(lo, hi, color=(rest[0] if rest else "0.6"),
                       alpha=0.18, lw=0, zorder=0)
        for i, y in enumerate(data):
            ax.plot(x, y, color=roi_color(i), lw=1.0,
                    label=labels[i] if i < len(labels) else f"ROI {i}")
        for i, ox in enumerate(onsets or []):
            if ox is None or (isinstance(ox, float) and np.isnan(ox)):
                continue
            ax.axvline(ox, color=roi_color(i), lw=0.8, ls="--", zorder=1)
        for ex, elabel in (events or []):
            ax.axvline(ex, color="0.4", lw=0.7, ls="--", zorder=0)
            if elabel:
                ax.annotate(elabel, xy=(ex, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(2, -2), textcoords="offset points",
                            fontsize=6, color="0.4", va="top")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(x=0)
        if legend and n:
            ax.legend(frameon=False, loc="upper right", ncol=1)
        if title:
            ax.set_title(title)
        fig.tight_layout()
    return _finish(fig, save, dpi)


def export_scatter(x, y, *, xlabel, ylabel, point_labels=None, fit=None,
                   title=None, width=COL_SINGLE, height=None, save=None, dpi=600):
    """Clean render of the propagation distance-vs-onset-delay scatter.

    x, y: point coordinates. point_labels: per-point text (ROI labels). fit:
        optional ``(x_pair, y_pair, label)`` line drawn through the points.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with paper_style():
        h = height if height is not None else width
        fig, ax = plt.subplots(figsize=(width, h))
        ax.scatter(x, y, s=28, color=ROI_COLORS[0], zorder=3)
        for xi, yi, lab in zip(x, y, point_labels or []):
            ax.annotate(lab, xy=(xi, yi), xytext=(3, 3), textcoords="offset points",
                        fontsize=6, color="0.4")
        if fit is not None:
            fx, fy, flabel = fit
            ax.plot(fx, fy, color=ROI_COLORS[1], lw=1.5, label=flabel)
            ax.legend(frameon=False, loc="best")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        fig.tight_layout()
    return _finish(fig, save, dpi)
