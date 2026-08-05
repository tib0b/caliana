# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the standalone Caliana app. SPEC.md §1 (Phase 2).

Build a one-folder distribution from a checkout with the package installed:

    pip install -e . pyinstaller
    pyinstaller caliana.spec              # -> dist/caliana/

One folder rather than one file: a single executable unpacks the whole bundle
(Qt, numpy, scipy, matplotlib — hundreds of MB) into a temp directory on every
launch, which is slow and trips antivirus software. Build on the OS you are
shipping to; PyInstaller does not cross-compile.

The three things this spec exists to fix, all of them cases where PyInstaller's
static analysis cannot see an import:

- **nd2 / pystackreg** are imported lazily, inside the functions that need them
  (so ``import caliana`` works without them), which means nothing references
  them at module level for the analyser to follow.
- **qtpy** picks its binding at runtime from what is installed, so no Qt import
  is visible statically either. The binding is pinned to PySide6 here — bundling
  two would be wasteful, and qtpy would then have to choose inside the bundle.
- **matplotlib** picks a backend at import; pinned to Agg by a runtime hook,
  since figures are only ever rendered to files (see rthook_matplotlib.py).
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

hiddenimports = [
    # Lazily imported readers/estimators (caliana/io.py, caliana/registration.py).
    *collect_submodules("nd2"),
    *collect_submodules("pystackreg"),
    "pystackreg.turboreg",            # the C extension doing the estimation
    # nd2 hands back dask-backed arrays; dask's scheduler is plugin-loaded.
    *collect_submodules("dask"),
    "xarray",
    # qtpy resolves the binding at runtime.
    *collect_submodules("qtpy"),
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtSvg",
    # Figure export only. The Qt backend is excluded below.
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
]

# Compiled extensions PyInstaller can miss when the module is imported lazily.
binaries = collect_dynamic_libs("pystackreg")

excludes = [
    # Other Qt bindings: qtpy would have a choice to make inside the bundle, and
    # two Qt copies in one process is a reliable way to crash.
    "PyQt5", "PyQt6", "PySide2",
    # matplotlib GUI backends (Agg only) and the interactive stack around them.
    "tkinter", "matplotlib.backends.backend_qt5agg", "matplotlib.backends.backend_tkagg",
    "IPython", "jupyter", "notebook", "ipykernel",
    "pytest",
]

a = Analysis(
    ["packaging/caliana_launcher.py"],
    pathex=["src"],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["packaging/rthook_matplotlib.py"],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="caliana",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX-packed Qt plugins are a known crash source
    console=False,             # a GUI app: no console window on Windows
    disable_windowed_traceback=False,
    # A crash before Qt starts has nowhere to report to with console=False, so
    # keep the OS-level traceback dialog available.
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="caliana",
)
