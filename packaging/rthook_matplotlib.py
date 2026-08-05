"""Runtime hook: pin matplotlib to Agg in a frozen build.

Caliana only ever uses matplotlib to *write files* (``figures.py`` renders the
static PNG/PDF/SVG exports; the on-screen plots are pyqtgraph). A GUI backend
would therefore buy nothing and cost a lot: matplotlib probes for one at import,
which in a bundle means either dragging Tk/Qt backends in or failing on a machine
that has neither. Agg renders to a buffer and needs no display at all.

Set before matplotlib is first imported, and only if the user has not chosen
otherwise.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
