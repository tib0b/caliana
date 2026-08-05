# Caliana

Analysis of plant calcium imaging data — load a recording, stabilize leaf
movement, place ROIs, extract fluorescence traces, compute ΔF/F, and run
response/propagation analyses with reproducible export. Usable as a standalone
app, from a Jupyter notebook, or as a headless library — all three over the same
reusable PyQt widgets and the same `Session` object.

See [`docs/SPEC.md`](docs/SPEC.md) for the full specification.

## Install

The following commands must be run in your active python/conda environment.

>Note: if installing into a conda environment, you might first need to 
>install the git and pip packages, which you can do with:
>
>```bash
>conda install git pip
>```

Directly from the repo, without cloning:

```bash
pip install git+https://github.com/tib0b/caliana.git
```


Or from a local checkout (editable):

```bash
pip install -e .            # everything except test tooling
pip install -e '.[dev]'     # + pytest
```

## Quickstart

Two ways in, over the same `Session`: the app if you'd rather not write Python,
the notebook if you want the steps under your own control.

### The app

```bash
caliana                 # or: caliana path/to/movie.nd2
```

One window with the workflow as tabs — import → registration → ROIs → crop →
analysis — and export from the `File` menu. Nothing else to set up; see
[Standalone app](#standalone-app) below for what each tab expects.

### A notebook

Each wrapper opens one window, blocks the cell until you close it, and leaves its
result on the `Session` (after `%gui qt`):

```python
%gui qt
import caliana

s = caliana.open_session()   # pick the file + import parameters in a window
s.preview()                  # scrub the movie, check the max projection
s.select_rois()              # click ROIs; their traces preview live
s.analyze()                  # ΔF/F, smoothing, propagation, heatmaps, kymographs
s.export_traces("traces.csv")
```

`s.select_leaves()` and `s.crop_traces()` cover the two optional steps (per-leaf
boxes, restricting every trace to one time window).

Because it is all one `Session`, widget steps and plain calls mix freely — do the
ROIs by hand and the rest in code, or skip the windows entirely:

```python
s = caliana.Session.from_file("movie.nd2", temporal_step=2, spatial_step=2)
s.register(caliana.RegistrationMode.WHOLE_FRAME)
s.add_roi(center=(32, 32), size=4, label="centre")
s.compute_dff(n=12)
s.cross_roi_propagation(signal="dff")                 # speed, direction, source ROI
s.kymograph([(10, 5), (30, 40)], baseline=(0, 12))    # distance × time image
s.export_provenance("provenance.json")
```

Two things the API does quietly:

- **Import parameters.** `from_file`/`load` take `start`/`end`, `temporal_step`,
  `spatial_step`, `spatial_window` and `channel`, applied in that order, so a
  multi-GB recording never has to fit in RAM — and so an nd2 only reads the
  frames you keep. `spatial_window` is in file pixels; everything afterwards (ROI
  centres, leaf boxes, crop windows) is in the loaded stack's coordinates.
  `help(caliana.Session.from_file)` takes them field by field, and `provenance()`
  records what a run actually used.
- **Time and space scales.** Analyses always compute in frames and pixels;
  `timeline.frame_interval` and `space.pixel_size` convert the *reported* numbers
  to seconds, µm and µm/s. Both are read from the file's metadata when it
  declares one (and rescaled for the downsampling), or set them yourself with
  `set_frame_interval` / `set_pixel_size`. Each axis degrades independently to
  frames/pixels when uncalibrated.

[`examples/interactive.ipynb`](examples/interactive.ipynb) walks the widget
workflow; [`examples/quickstart.ipynb`](examples/quickstart.ipynb) walks the
headless one end-to-end with rendered plots.

## Standalone app

The same widgets the notebook drives, wrapped in one window — no Python required
to run an analysis.

Each tab stays closed until its prerequisite exists: no ROIs before a stack, no
cropping before ROIs. Analysis opens on the stack alone, since its onset-heatmap
and kymograph pages are dataset-wide — only its trace page waits for ROIs. The
status bar carries the run — source, `[T,Y,X]`, calibration, registration mode,
ROI count, crop — and `File ▸ Export` writes the traces CSV, the stack TIFF and
the provenance JSON, together or one at a time. Long steps (loading,
registration, export) run off the UI thread behind a progress dialog, and
failures are reported in a dialog rather than a traceback.

The session is shared by every tab: registering on tab 2 invalidates the traces
tab 5 draws, and each panel re-reads the session when it is next opened.

To build a self-contained bundle needing no Python install (see
[`caliana.spec`](caliana.spec) for what it pins and why):

```bash
pip install -e . pyinstaller
pyinstaller caliana.spec        # -> dist/caliana/
```

## Package layout

| Module (`src/caliana/`) | Responsibility (SPEC ref) |
| --- | --- |
| `models.py` | Core dataclasses/enums: `Session` state pieces (§2.1) |
| `timeline.py` | Time axis (frames, optionally calibrated to seconds) + events (§3, §6) |
| `space.py` | Space axis (pixels, optionally calibrated to µm) + unit helpers (§3) |
| `io.py` | Load TIFF/nd2 + downsample-on-load + scale metadata (§3 Stage I) |
| `registration.py` | Rigid motion correction: none / whole-frame / per-leaf (§3 Stage II) |
| `roi.py` | ROI masks, trace extraction, leaf assignment (§3 Stage II) |
| `analysis.py` | ΔF/F, response-onset timing, propagation, kymographs, custom callables (§3 Stage III) |
| `export.py` | Traces CSV, stack TIFF, provenance JSON (§4) |
| `session.py` | `Session`: single source of truth tying it together (§2.1) |
| `widgets/` | Embeddable PyQt widgets + notebook blocking wrappers (§2.2) |
| `app.py` | Standalone app: window, tabs, menus, error handling (§1 Phase 2) |

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The nd2 test ([`tests/test_io.py`](tests/test_io.py)) auto-skips unless a real
`.nd2` file is present (the `nd2` reader itself ships with the core install).
GUI tests run headless with `QT_QPA_PLATFORM=offscreen`.

## License

MIT — see [`LICENSE`](LICENSE).
