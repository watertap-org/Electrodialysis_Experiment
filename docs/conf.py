from __future__ import annotations

import sys
from pathlib import Path

# Ensure local package imports work if autodoc is used later.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

project = "Electrodialysis Experiment"
author = "WaterTAP Organization and Electrodialysis Experiment Contributors"
copyright = (
    "2026, WaterTAP Organization and Electrodialysis Experiment Contributors"
)
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
