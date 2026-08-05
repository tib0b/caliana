"""Frozen-app entry point (PyInstaller analyses a script, not a console script).

Deliberately trivial: everything real lives in ``caliana.app.main``, so a bundled
build and ``pip install caliana`` run the same code.
"""
from __future__ import annotations

import multiprocessing
import sys

from caliana.app import main

if __name__ == "__main__":
    # Windows/macOS spawn re-executes this file in each child process; without
    # this a frozen build would open a new window instead of a worker.
    multiprocessing.freeze_support()
    sys.exit(main())
