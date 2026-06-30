"""Embeddable PyQt widgets for the Caliana workflow. SPEC.md §2.2.

Each interactive step is one QWidget that reads/writes a Session. The same widget
is used by the notebook blocking wrappers and (Phase 2) the standalone app.

Importing this subpackage pulls in a Qt binding (via qtpy); the core `caliana`
package does not, so headless use never requires Qt.
"""
