# Caliana

Analysis of plant calcium imaging data — load a recording, stabilize leaf
movement, place ROIs, extract fluorescence traces, compute ΔF/F, and run
response/propagation analyses with reproducible export. Usable as a headless
library, from a Jupyter notebook, or via embeddable PyQt widgets.

See [`docs/SPEC.md`](docs/SPEC.md) for the full specification.

## Install

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
s.export_traces("traces.csv")
s.export_provenance("provenance.json")
```

Interactive (after `%gui qt` in a notebook): `s.preview()` (Stage I),
`s.select_rois()` (Stage II), `s.analyze()` (Stage III). Each reads and writes the
same `Session`, so widgets and API calls mix freely.
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) walks the full headless
workflow end-to-end with rendered plots.

## Package layout

| Module (`src/caliana/`) | Responsibility (SPEC ref) |
| --- | --- |
| `models.py` | Core dataclasses/enums: `Session` state pieces (§2.1) |
| `timeline.py` | Time axis (frames now; seconds later for electrodes) + events (§3, §6) |
| `io.py` | Load TIFF/nd2 + downsample-on-load (§3 Stage I) |
| `registration.py` | Rigid motion correction: none / whole-frame / per-leaf (§3 Stage II) |
| `roi.py` | ROI masks, trace extraction, leaf assignment (§3 Stage II) |
| `analysis.py` | ΔF/F, peak detection, propagation, custom callables (§3 Stage III) |
| `export.py` | Traces CSV, stack TIFF, provenance JSON (§4) |
| `session.py` | `Session`: single source of truth tying it together (§2.1) |
| `widgets/` | Embeddable PyQt widgets + notebook blocking wrappers (§2.2) |
| `app.py` | Standalone app entry point (Phase 2) |

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
