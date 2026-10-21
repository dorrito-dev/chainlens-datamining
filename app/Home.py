"""Home entrypoint.

Streamlit derives the sidebar label for the main script from its filename, so
the console entrypoint lives here (``Home.py``). The landing implementation is
still ``app/streamlit_app.py``; ``make app`` runs this wrapper.
"""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).with_name("streamlit_app.py")),
    run_name="__main__",
)
