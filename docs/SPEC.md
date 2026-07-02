# Caliana — Plant Calcium Imaging Analysis

Specification — drafted 2026-06-22

## 1. Overview

Caliana is a Python tool for analyzing calcium-imaging recordings of plant
tissue (e.g. leaves expressing a Ca²⁺ fluorescence reporter). It covers the
full workflow: loading a recording, optionally stabilizing leaf movement,
placing regions of interest (ROIs), extracting fluorescence traces, computing
ΔF/F, and running response/propagation analyses with export for reproducibility.

The tool is delivered as a **library of reusable PyQt widgets** built around a
central **`Session`** object. Every widget can be:

- driven from a **Jupyter notebook** via thin blocking convenience wrappers, or
- embedded in a **standalone PyQt application**.

### Phasing

- **Phase 1 (lab/notebook tool):** the `Session` API, the widgets, and the
  notebook wrappers. Highly customizable, fits well into existing workflows.
- **Phase 2 (standalone app):** a self-sufficient PyQt application wrapping the
  same widgets, with installer-grade packaging, robust error handling, and UX
  for non-coders. Items below tagged **[Phase 2]** are deferred.

## 2. Architecture

### 2.1 The `Session` object (single source of truth)

A `Session` holds all state for one analysis and is the object both the notebook
and the app operate on:

```
Session
├── source            # file path + import parameters
├── data              # downsampled image stack, numpy array [T, Y, X]
├── registration      # mode (none | whole-frame | per-leaf) + settings
├── leaf_regions      # per-leaf boxes, each with its own per-frame rigid
│                     #   transforms + reference (per-leaf mode only)
├── rois              # ROI defs (shape, center, size, label, leaf_region)
├── traces            # raw F and ΔF/F per ROI, numpy arrays
├── events            # optional stimulus/event markers (frame indices)
└── analyses          # results of built-in / custom analyses
```

Widgets read from and write to the `Session`. Recomputation is explicit (e.g.
re-running registration or changing the baseline invalidates downstream
`traces`/`analyses`).

### 2.2 Dual-mode pattern

- **Notebook:** thin blocking wrappers, e.g. `session.select_rois()`, open a Qt
  window using IPython's Qt event-loop integration (`%gui qt`). The wrapper
  blocks the current cell until the window is closed, then returns the relevant
  result while also updating the `Session`. Mental model: one cell = one
  interactive step.
- **App [Phase 2]:** the same `QWidget`s are docked/paneled into the main
  window; they mutate the shared `Session` directly without blocking.

Each interactive step is implemented once as an embeddable `QWidget`; the
blocking wrapper is a small adapter around it. No business logic lives in the
wrapper or the app shell.

### 2.3 Tech stack

- Python 3.x, numpy as the core array type.
- **PyQt/PySide** for widgets; **pyqtgraph** for fast image display, scrubbing,
  and interactive plots.
- **tifffile** for (OME-)TIFF; **nd2** (or nd2reader) for Nikon `.nd2`.
- **pystackreg** (or scikit-image) for rigid registration.
- **scipy** for peak detection and signal utilities.

## 3. Workflow

### Stage I — Dataset selection & import

**Inputs**
- File: `.tif` / `.ome.tif` / `.nd2`. **Single fluorescence channel only.** If a
  file contains multiple channels, the user picks one at import; no
  multi-channel math.
- Import parameters (the "downsample on load" strategy — full-resolution data is
  never required to fit in RAM):
  - **Start / end frame index** (temporal crop).
  - **Temporal step** (sub-sample frames).
  - **Spatial step** (sub-sample pixels / binning).
  - **Spatial window** (crop a Y/X region).

**Behavior**
- Read the selected/cropped/downsampled data into a numpy array `[T, Y, X]`,
  stored in `Session.data`.
- **Units:** size is always in pixel, indices are in seconds if a time scale is specified, and falls
  back to frames otherwise.
  (µm/pixel, seconds) is required or used. Propagation is reported in px/frame,
  timing in frames.

**Visualization (Stage I widget)**
- **Frame scrubbing + playback:** time slider plus a play button to watch the
  movie.
- **Contrast / colormap controls:** adjustable display min/max and selectable
  colormap (raw Ca²⁺ signals are low-contrast).
- **Max-intensity heatmap:** a per-pixel projection over time (max projection),
  **normalized**, shown alongside the movie to reveal active regions before ROI
  placement.

### Stage II — Registration (optional) & ROI selection

Registration is the leaf-motion-tracking mechanism: by stabilizing the image,
static ROIs stay on the same tissue (no per-ROI template tracking). All modes
use **rigid (translation + rotation)** transforms only — affine and
non-rigid/elastic are out of scope, because elastic warping risks distorting the
intensity traces. Transforms are always estimated on the **downsampled** stack
(e.g. via pystackreg) and reference defaults to the **mean image** (fallback:
first frame). Per-frame transforms are stored on the `Session` so the stabilized
stack can be exported and the run is reproducible.

The user chooses one of **three registration modes** per dataset, because the
recordings contain multiple leaves that move *independently* but often only
slightly:

1. **None** — no registration; ROIs are placed directly on the raw downsampled
   stack. For already-stable recordings.
2. **Whole-frame** — one rigid transform per frame for the entire field of view.
   Correct when there is a single leaf, or all tissue moves as one unit (e.g.
   only stage drift). Cheapest; one reference for the whole frame.
3. **Per-leaf** — the user drags a **bounding box around each leaf**. Each box's
   sub-stack is registered **independently** (its own per-frame rigid transforms
   and its own reference = mean of that sub-stack). This handles leaves that
   translate/rotate independently. Pipeline:
   `Load → draw leaf boxes → register each box → place ROIs → traces`.
   This has been implemented, but the capabilities are currently very limited, and it serves more as a proof of concept.

**Per-leaf details**
- Leaf boxes are stored in `Session.leaf_regions`; each holds its bbox,
  per-frame `(dx, dy, θ)`, and reference image.
- Boxes should be drawn **generously** so the leaf stays inside its box across
  the whole recording even as it moves; tissue that drifts out of its box can't
  be stabilized.
- Leaf boxes are expected **not to overlap**. If they do, containment is
  resolved first-match (see ROI assignment below).
- Whole-frame mode is equivalent to per-leaf mode with a single box covering the
  full frame.

**Drift-out-of-box handling.** If a leaf moves outside its box during the
recording, registration and traces silently corrupt (the estimate locks onto
incoming background, the reference mean blurs, edge ROIs sample fill/background,
and ΔF/F picks up geometric drops mistaken for signal). Mitigation, in order:
- **Generous boxes** are the user-facing guidance (draw the box larger than the
  leaf's full range of motion). Primary defense.
- **Drift detection (baseline behavior):** track the cumulative per-frame offset
  `(dx, dy, θ)` for each leaf. When it approaches the box margin, **warn** the
  user and mark the affected frames / that leaf region as **low-confidence**.
  This flag is carried through to the traces and recorded in the provenance
  sidecar so a corrupted run is visible rather than silent.
- **Auto-grow box** (later enhancement, not Phase 1): a first motion-estimation
  pass could expand the box to the union of leaf positions. Deferred — it needs
  a pre-pass and raises cross-leaf contamination risk.

**ROI selection (Stage II widget)**
- User places ROIs by clicking on an interactive plot (the registered or raw
  view).
- **Fixed shapes: circle or square**, with user-chosen radius/size, plus
  **free-hand polygon ROIs** (trace an outline, e.g. around a whole leaf). For a
  polygon the centre is the outline's centroid (used for leaf assignment and
  propagation); circle/square keep centre + size.
- ROIs are independent; **overlap is permitted** (a pixel may belong to multiple
  ROIs). No disjointness is enforced.
- Each ROI has a label/index.
- **Leaf assignment (per-leaf mode):** an ROI is automatically linked to the
  leaf region whose box **contains its center** (first match if boxes overlap);
  the user can override the assignment. The ROI's trace is then extracted from
  that leaf's stabilized sub-stack, so it follows that leaf's motion. In
  whole-frame and none modes there is a single implicit region and no assignment
  is needed. An ROI whose center falls in no leaf box is flagged (unassigned →
  no per-leaf stabilization).
- **Live trace preview:** as ROIs are placed/moved, their mean-intensity traces
  update live so the user can judge ROI quality immediately.
- **Non-goals:** a dedicated "background ROI" role.

**Trace extraction**
- For each ROI, compute the **mean pixel intensity inside the ROI per frame** →
  raw F trace `[T]`.
- Store all raw F traces in `Session.traces` (numpy arrays) for Stage III.

### Stage III — Analysis

Analyses operate on the ROI traces (and may reference `Session.data`).

**ΔF/F computation (core)**
- ΔF/F = (F − F0) / F0, computed per ROI.
- **F0 baseline**, user-selectable between:
  - **Mean of the first N frames** (N user-specified), or
  - **User-selected baseline region**: the user drags a time window on the trace
    to define the baseline interval interactively.

**Built-in analyses**
- **Peak detection** per trace: amplitude, time-to-peak (frames), peak count
  (scipy-based, with adjustable threshold/prominence).
- **Cross-ROI propagation:** compare response timing across ROIs to estimate
  signal propagation between them (e.g. calcium-wave timing/direction), reported
  in px/frame using ROI pixel coordinates.

**Stimulus / event markers (optional)**
- The user may mark one or more event times (frame indices) on the timeline.
- When present, baseline windows, latency/time-to-peak, and propagation are
  measured relative to event onset. Analyses also work with no events defined.

**Custom analysis functions**
- Users supply **plain Python callables** operating on the numpy arrays, e.g.
  `def f(traces, data) -> result`. Natural for notebook use; full trust, no
  sandboxing.
- Results are surfaced through the same display/export path as built-ins.

**Display**
- Results render as **graphs** (traces, peak markers, propagation timing) or
  **heatmaps**, depending on their nature.

## 4. Export & reproducibility

- **Traces → CSV:** per-ROI raw F and ΔF/F over time (columns = ROIs, rows =
  frames), including the frame/time axis.
- **Processed stack → TIFF:** the registered and/or downsampled image stack
  (and heatmaps) for use in other tools.
- **Provenance → sidecar (JSON/YAML):** records source file, import parameters
  (start/end, temporal/spatial step, spatial window, selected channel),
  registration settings (mode, reference, leaf boxes), ROI definitions
  (including leaf assignment), baseline method
  and parameters, event markers, and analysis settings — so any analysis is
  reproducible.
- Figure export (PNG/SVG) of plots/heatmaps: nice-to-have, lower priority.

## 5. Key decisions & rationale

| Area | Decision | Rationale |
|---|---|---|
| Delivery | Reusable PyQt widgets + `Session`; notebook wrappers + app | One implementation usable from notebook and app |
| Memory | Downsample on load | Keep multi-GB recordings in RAM; full-res only for export |
| Formats | (OME-)TIFF + nd2, single channel | Covers common acquisition outputs |
| Motion | Rigid registration, 3 modes: none / whole-frame / per-leaf (box per leaf) | Multiple leaves move independently; per-leaf stabilizes each without distorting intensities |
| ROIs | Circle/square or free-hand polygon, overlap allowed | Fixed shapes are fast/comparable; polygons capture whole leaves / irregular regions |
| Signal | ΔF/F, F0 = first-N-frames or user-selected window | Standard, dataset-appropriate baselines |
| Analyses | Peak detection + cross-ROI propagation + custom callables | Matches plant systemic-signaling questions |
| Events | Optional stimulus markers | Supports wound/touch experiments without requiring them |

## 6. Open questions / future work

- Irregular frame timing (nd2 per-frame timestamps) is ignored under the
  pixels/frames model; revisit if physical-time analyses are needed later.
  (nd2 reading is implemented and lazy/dask-backed; Z and multipoint dims are
  currently reduced to their first index — revisit if Z-projection or
  multi-position handling is needed.)
- Registration quality on low-contrast tissue — may need a quality indicator or
  manual reference-frame selection.
- **Registration vs. the signal itself.** Intensity-based rigid registration
  aligns whatever image structure dominates. Where the calcium signal is the
  *only* strong structure (little stable tissue texture, or large transients),
  registration can lock onto the signal and distort traces rather than correct
  motion. Single-channel GCaMP has no separate structural channel to register
  on. Mitigations to consider: register on a baseline/low-pass version, restrict
  estimation to pre-stimulus frames, or expose a quality check + easy disable.
  (Observed starkly on the synthetic clip, which has no motion and only the
  growing blob as structure — see examples/quickstart.ipynb §2.)
- Algorithm for per-leaf tracking needs to be improved.
- **[Phase 2]** Standalone-app packaging (installer), error handling, and
  in-app custom-function support for non-coders.
- Multi-channel / ratiometric support, affine registration: out of current
  scope.

### Forward compatibility: electrode data (future, not implemented)

Calcium imaging may later be analyzed alongside **electrode recordings** (an
auxiliary time-series captured during the same experiment). Not built now, but
kept in mind so the design doesn't preclude it:

- **Auxiliary time-series on the `Session`.** The data model should be able to
  grow an `aux_signals` slot (named 1-D traces sampled over the recording)
  living next to `traces`, plotted on the same timeline and exportable to the
  same CSV. ROI traces and electrode traces would share one time axis.
- **Real-time axis tension.** We chose **pixels/frames only**. Electrodes sample
  in real time (Hz) at a rate unrelated to the frame rate, so correlating the
  two will eventually require a real (seconds) time axis and a
  frame↔time mapping. This reinforces keeping the time-axis representation
  isolated (frame index now, but not hard-coded everywhere) so a seconds axis
  can be added without rewriting analyses. Ties into the irregular-frame-timing
  item above.
- **Events as the alignment anchor.** The optional stimulus/event markers are
  the natural synchronization point between imaging and electrode streams; keep
  events first-class and timeline-based.
- No commitment to electrode file formats or hardware yet.
