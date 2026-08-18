---
name: adding-an-analysis
description: Add a new analysis to caliana — a computation over ROI traces or the image stack, exposed headlessly, in the notebook and in the app. Use when asked to implement, extend or expose any analysis (peak detection, correlation, latency maps, new propagation or kymograph variants), or when an analysis exists in one layer and must be wired into the others.
---

# Adding an analysis to caliana

Caliana is layered, and an analysis touches every layer in a fixed order. Work
top to bottom; each step has one job and its own invariants. Do not merge steps —
a computation written inside a widget, or a unit conversion written inside the
compute layer, is the failure mode this procedure exists to prevent.

Read `docs/SPEC.md` §3 Stage III before starting. It is the specification of
record; if it already describes the analysis, implement what it says rather than
inventing a variant, and update it when a decision changes.

## Step 0 — Check whether code is needed

A one-off, notebook-only computation needs no changes at all:

```python
def my_metric(traces, data):
    return traces.dff.max(axis=1)

session.apply(my_metric)          # analysis.apply_custom; full trust, no sandbox
```

Everything below applies to an analysis that should become part of the tool:
reusable, tested, GUI-exposed, recorded in provenance.

## Step 1 — Classify the analysis

Three shapes exist. Each has a working template to copy; pick one before writing
anything.

| Shape | Input → output | Existing example | Widget template |
|---|---|---|---|
| Per-ROI trace analysis | `Traces` (+ `list[ROI]`) → dict/array per ROI | `cross_roi_propagation` | Trace page, an entry in the analysis selector |
| Dataset-wide map | `[T, Y, X]` stack → 2D `[Y, X]` image | `onset_time_map` | Heatmaps page |
| Geometry-driven | stack + a user-drawn shape → image/series | `kymograph` | Kymograph page |

The classification decides the `Session` wrapper (traces vs. stack) and the GUI
surface (a selector entry vs. a whole tab). Getting it wrong means redoing steps
3 and 5.

## Step 2 — Write the pure function in `src/caliana/analysis.py`

This module is headless: numpy arrays and dataclasses in, plain data out. **No
Qt, no `Session`, no file I/O, no user interaction.** Signature convention is the
data first, then keyword parameters with defaults.

```python
def peak_stats(
    traces: Traces,
    signal: str = "dff",
    prominence: float = 0.05,
) -> dict:
    """One-line summary. SPEC.md §3 Stage III.

    Explain what each parameter is *for* and when it breaks, not just its type.
    """
```

Five conventions, all observable in the existing functions:

- **Pixels and frames are the model.** Never accept `frame_interval` or
  `pixel_size` as an argument, and never return a value in seconds or µm.
  `cross_roi_propagation` returns `speed_px_per_frame`; the display layer
  converts. A calibration change must alter displayed units and nothing else.
- **Signal-selection ladder.** If the analysis reads traces, copy this exactly so
  it agrees with what the plot shows:
  ```python
  if signal == "smoothed" and traces.smoothed is not None:
      data = traces.smoothed
  elif signal == "dff" and traces.dff is not None:
      data = traces.dff
  else:
      data = traces.raw
  ```
- **Return a dict for multi-field results, and include the parameters it was
  measured with.** `kymograph` returns `width`/`step`/`baseline` alongside
  `values` so an export or a figure can be reconstructed from the result alone.
- **Raise `ValueError` on bad input**, with the offending value in the message;
  do not silently clamp. The exception is degenerate-but-meaningful cases, which
  return `NaN`/`None` (`onset_time` returns NaN when nothing rises).
- **Import heavy or optional dependencies inside the function** (`scipy`, as
  `smooth_traces` does). `import caliana` must stay cheap and work without the
  GUI extras. A genuinely new third-party dependency must also be added to
  `caliana.spec`'s `hiddenimports` if it is imported lazily or by plugin.

## Step 3 — Expose it as a `Session` method

In `src/caliana/session.py`, under the Stage III banner. The wrapper contains
**no analysis logic** — it selects inputs, calls the compute function, stores the
result:

```python
def peak_stats(self, **kwargs):
    """Per-ROI peak statistics; stores them under ``analyses``.

    Keyword args are forwarded to ``analysis.peak_stats``.
    """
    if self.traces is None:
        self.extract_traces()
    result = analysis.peak_stats(self.traces, **kwargs)
    self.analyses["peaks"] = result
    return result
```

- **Trace analyses auto-extract**: `if self.traces is None: self.extract_traces()`,
  so callers never hit an ordering error.
- **Stack analyses must go through the helpers**, never `self.data` directly:
  ```python
  self._require_data()
  start, end = self._crop_bounds()
  stack = self._working_stack()[start:end]
  ```
  `_working_stack()` returns the stabilized stack when one exists, else the raw
  one. `_crop_bounds()` keeps the analysis on the same interval as the traces.
- **Note the tracking caveat** in the docstring if the analysis is fixed in stack
  coordinates. With `register(apply=False)` the pixels stay raw and the *ROIs*
  move, so anything anchored to the frame (a path, a per-pixel map) needs
  `apply=True` to be motion-corrected. `Session.kymograph`'s docstring is the
  model.

## Step 4 — Store the result under `analyses`

Assigning `self.analyses["<key>"] = result` gets two behaviours for free:

- `_invalidate_traces()` calls `self.analyses.clear()`, so adding an ROI,
  re-registering or re-cropping drops the stale result. **Do not add a bespoke
  invalidation hook, a cache, or widget-to-widget signals** — the `_revision`
  counter and this dict are the whole mechanism.
- `Session.provenance()` emits `sorted(self.analyses.keys())`, so the JSON
  sidecar records that the analysis ran.

Provenance records key names only. Parameters worth reproducing belong *inside*
the stored dict (step 2).

## Step 5 — Wire the GUI in `src/caliana/widgets/analysis_widget.py`

Widgets read controls, call the `Session` method and draw the result. Anything
computational belongs in step 2.

**Per-ROI analysis** — one more entry in the existing selector:

```python
self.analysis_box.addItems(["(select analysis)", "Cross-ROI propagation", "Peak detection"])
...
self.param_stack.addWidget(self._build_peaks_panel())   # 2: peaks
```

The stacked parameter pages line up **1:1** with the combo items;
`_on_analysis_changed(index)` depends on that and must be updated to clear the
other analyses' overlays when a new index is added.

**Dataset-wide or geometry-driven analysis** — a new
`self.tabs.addTab(self._build_x_page(), "…")`, plus the tab gating in `app.py`.
The rule there: pages that need ROIs gate themselves inside `reload()`; only
stack-level prerequisites are enforced by the app.

The compute handler:

```python
def compute_peaks(self):
    """Detect peaks on the displayed signal and list them."""
    if self.session.traces is None:
        self.status.setText("No traces.")
        return None
    result = self.session.peak_stats(
        signal=self._displayed().signal,
        prominence=self.peak_prom_box.value(),
    )
    self._show_peaks(result)
    return result
```

Six rules:

1. **Run on `self._displayed().signal`** so the analysis matches the trace
   currently plotted. That helper is the single source of truth for
   raw / ΔF/F / smoothed.
2. **Convert frame coordinates.** Spin boxes and draggable regions are in
   *original* frame indices; analysis arguments (`baseline_region`, windows) are
   *post-crop trace-column* indices. Subtract `self._crop_start()` going in, add
   it back coming out. This is the most common bug in this area; `compute_onset_heatmap`
   and `_show_heatmap` are the reference pair.
3. **Units only at display time**: `_to_time`, `_time_unit`, `_distance_units`,
   `_speed_str`.
4. **Put the logic in a named method**, not a lambda inside `clicked.connect`, so
   tests can drive it without a mouse.
5. **Extend `reload()`**: clear the analysis' overlays at the top, and re-range
   any control against `_crop_start()` / `_n_frames()`. The contract every panel
   satisfies is: constructs on an empty `Session`, and `reload()` is safe to call
   any number of times.
6. **Never call `close()` on the widget.** The notebook path closes it via
   `run_widget_blocking(..., close_on=...)`; the app keeps it open and refreshes
   on the same signal.

**Long computations** (anything per-pixel over a whole stack) go through
`widgets/_task.py`'s `run_in_background` in the app path only. Never wire it into
`Session` — the notebook path must stay synchronous.

If the analysis is worth a blocking notebook entry point, add a thin `[notebook]`
`Session` method using `widgets._qt.run_widget_blocking`; do not duplicate the
widget.

## Step 6 — Figure export

Every on-screen result gets a "Save …" button rendering a matplotlib figure that
matches the display — same axes, same units, same contrast. Reuse
`src/caliana/figures.py` (`export_image`, `export_traces`, `export_kymograph`,
`export_scatter`); add a new `export_*` there only if none fits.

The handler shape: build a `render(path)` closure and hand it to
`save_figure_dialog` (see `_save_kymograph`). Pass `levels=` from the live colour
bar so the saved figure is WYSIWYG.

## Step 7 — Tests

New file `tests/test_<analysis>.py`, following `tests/test_kymograph.py`:

- Build a **synthetic input with an analytically known answer** (see
  `_travelling_wave`, a wave whose speed is known exactly) and assert against the
  true value, not a golden number recorded from a previous run.
- Cover the pure function, its error paths, and the `Session` wrapper —
  specifically that it **stores under `analyses`**, **honors `crop_window`**, and
  **uses the stabilized stack** where applicable. Those three are the wrapper's
  entire job.
- Widget tests go in `tests/test_widgets.py`, driving the plain methods
  (`widget.compute_peaks()`). They self-configure an offscreen Qt platform and
  skip if the Qt stack is missing.
- **Add every test to the `if __name__ == "__main__":` block** at the bottom —
  each test file must also run as a plain script.

Run `pytest` from the `caliana/` checkout.

## Step 8 — Documentation

1. **`docs/SPEC.md` §3 Stage III** — the specification of record. Update it when
   the analysis introduces or changes a decision; do not let code and spec drift.
2. **Module docstrings** — the "Built-ins:" list at the top of `analysis.py` and
   the page-by-page summary at the top of `analysis_widget.py`. Both are
   maintained indexes, not decoration.
3. **Function docstring** — cite the spec section (`SPEC.md §3 Stage III`) and
   document *why* each parameter exists and when it misbehaves.
   `cross_roi_propagation`'s `direction_mode` documentation is the standard: it
   states which layouts each mode is valid for.

## Checklist

- [ ] Shape classified; template chosen
- [ ] Pure function in `analysis.py`; no Qt, no I/O, no unit conversion
- [ ] Pixels/frames only in and out
- [ ] `Session` method: auto-extract or `_working_stack()` + `_crop_bounds()`
- [ ] Result stored under `session.analyses["<key>"]`, parameters inside it
- [ ] Widget reads controls only; frame↔column offset applied; `reload()` extended
- [ ] Long compute via `run_in_background` (app path only)
- [ ] Figure export reusing `figures.py`
- [ ] Tests with a known-answer synthetic, plus the `__main__` block
- [ ] `SPEC.md`, module docstrings and the function docstring updated
