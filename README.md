# Caliana

Analysis of plant calcium imaging data — load a recording, stabilize leaf
movement, place ROIs, extract fluorescence traces, compute ΔF/F, and run
response/propagation analyses with reproducible export. Usable as a headless
library, from a Jupyter notebook, or via embeddable PyQt widgets.

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

```python
import caliana

s = caliana.Session.from_file("movie.tif", temporal_step=2)   # load + downsample
s.register(caliana.RegistrationMode.WHOLE_FRAME, reference="mean")
s.add_roi(center=(32, 32), size=4, label="centre")
s.extract_traces()
s.compute_dff(n=12)
res = s.cross_roi_propagation(signal="dff")     # speed, direction, source ROI
# direction_mode="roi_line" (default) fits the speed along the line the ROIs sit
# on; "auto" fits a free 2D direction, which needs ROIs spread in two dimensions.
kymo = s.kymograph([(10, 5), (30, 40)], baseline=(0, 12))   # distance × time image
s.export_traces("traces.csv")
s.export_provenance("provenance.json")
```

### Import parameters

`from_file` (and `load`) take the `ImportParams` fields as keywords — the
"downsample on load" controls, so a multi-GB recording never has to fit in RAM:

| Keyword | Meaning |
| --- | --- |
| `start`, `end` | keep frames `[start, end)` (`end=None` ⇒ to the last frame) |
| `temporal_step` | average every N frames into one (`1` = off) |
| `spatial_step` | keep every Nth pixel along Y and X (`1` = full resolution) |
| `spatial_window` | `(y0, y1, x0, x1)` crop of the field of view |
| `channel` | which channel to keep from a multi-channel file |

They apply in that order — `channel` → temporal crop → `temporal_step` →
`spatial_window` → `spatial_step` — each acting on the result of the previous.
The temporal crop comes first, so an nd2 only reads the frames you keep.
`spatial_window` is in file pixels; everything afterwards (ROI centres, leaf
boxes, crop windows) is in the loaded stack's coordinates. See
`help(caliana.Session.from_file)` for the field-by-field guide, and
`provenance()` for the parameters a run actually used.

```python
caliana.Session.from_file("movie.nd2", start=100, end=1600)     # a 1500-frame window
caliana.Session.from_file("movie.nd2", temporal_step=2, spatial_step=2)
caliana.Session.from_file("two_channel.tif", channel=1)
```

### Time and space scales

Analyses always compute in frames and pixels; two optional scales convert the
*reported* numbers. Both are read from the file's metadata at import when it
declares them (nd2 acquisition settings, ImageJ/OME/resolution tags in TIFF) and
adjusted for the downsampling, so they describe the stack you loaded:

```python
s = caliana.Session.from_file("movie.nd2", spatial_step=2)
s.timeline.frame_interval    # seconds per frame, e.g. 1.0
s.space.pixel_size           # µm per pixel, e.g. 162.8 (81.4 in the file, ×2)

s.set_frame_interval(0.5)    # override / supply either one
s.set_pixel_size(81.4)
```

Once set, traces read in seconds (plot axes, the CSV's `seconds` column), ROI
distances in µm, propagation speed in µm/s — each axis degrading independently to
frames/pixels when uncalibrated — and saved images carry a scale bar.

Interactive (after `%gui qt` in a notebook): `caliana.open_session()` to pick a
file and its import parameters in a window, then `s.preview()` (Stage I),
`s.select_rois()` (Stage II), `s.analyze()` (Stage III). Each reads and writes the
same `Session`, so widgets and API calls mix freely.
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) walks the full headless
workflow end-to-end with rendered plots. [`examples/interactive.ipynb`](examples/interactive.ipynb) shows an example workflow using the interactive PyQt widgets.

## Standalone app

The same widgets, wrapped in one window with the workflow as tabs — no Python
required to run an analysis:

```bash
caliana                 # or: caliana path/to/movie.nd2
```

Each tab stays closed until its prerequisite exists (no ROIs before a stack, no
analysis before ROIs), the status bar carries the run — source, `[T,Y,X]`,
calibration, registration mode, ROI count, crop — and `File ▸ Export` writes the
traces CSV, the stack TIFF and the provenance JSON, together or one at a time.
Long steps (loading, registration, export) run off the UI thread behind a
progress dialog, and failures are reported in a dialog rather than a traceback.

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
