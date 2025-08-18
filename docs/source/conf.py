from __future__ import annotations
import sys
import os
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../fem'))

project = 'ZeroHeliumKit'
copyright = '2025, EeroQ'
author = 'EeroQ'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.todo',
              'nbsphinx',
              ]

autosummary_generate = True

# templates_path = ['_templates']
# exclude_patterns = []

# pygments_style = 'sphinx'

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

nbsphinx_execute = 'never'
nbsphinx_requirejs_path = ''

suppress_warnings = [
    'ref.python',
    'docutils',
    'ref.ref'
]

html_theme_options = {
    "path_to_docs": "docs", 
    "repository_url": "https://github.com/eeroqlab/zeroheliumkit",
    "use_repository_button": True
}


autodoc_default_options = {
    'members': True,
    'inherited-members': False,
    'show-inheritance': True,
    'member-order': "bysource",
    'exclude-member': ['__init__']
    # 'undoc-members': False,
}

add_module_names = True
# autodoc_member_order = "bysource"
# napoleon_use_ivar = True


autodoc_mock_imports = [
    "gmsh", "gdspy", "matplotlib", "numpy", "scipy", "Shapely", "ezdxf", "svgpathtools", "polars", "ipywidgets"
]


from pathlib import Path
from typing import Any, List
from sphinx.application import Sphinx

# --- helpers ---------------------------------------------------------------

def remove_line_containing_string(lines: List[str], excludetext: str) -> List[str]:
    return [ln for ln in lines if excludetext not in ln]

def insert_line_after_match(lines: List[str], match_text: str, new_line_text: str) -> List[str]:
    out: List[str] = []
    inserted = False
    to_insert = new_line_text if new_line_text.endswith("\n") else new_line_text + "\n"

    for ln in lines:
        out.append(ln)
        if not inserted and match_text in ln:
            out.append(to_insert)
            inserted = True
    return out

# --- core ------------------------------------------------------------------

def shorten_autosummary_titles(autosummary_dir: Path) -> None:
    """
    Rewrite autosummary stubs in `autosummary_dir`:
      - title: keep only the last dotted part (class/function name)
      - underline: match new title length
      - drop lines containing '__init__'
      - add ':member-order: bysource' after '.. autoclass::'
    """
    if not autosummary_dir.exists():
        return

    for p in autosummary_dir.rglob("*.rst"):  # only .rst files (recurses into subdirs)
        # Read safely as UTF-8 text; skip files that are not UTF-8
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.warning("[shorten titles] Skipping non-UTF8 file: %s", p)
            continue

        lines = text.splitlines(keepends=True)
        if len(lines) < 2:
            continue  # nothing to do

        # Title -> last dotted component
        title = lines[0].rstrip("\n")
        short = title.rsplit(".", 1)[-1]
        lines[0] = short + "\n"
        lines[1] = "=" * len(short) + "\n"  # underline length must match

        # Remove any line that mentions __init__ (e.g. special-members)
        lines = remove_line_containing_string(lines, "__init__")

        # After an autoclass directive, enforce :member-order: bysource
        lines = insert_line_after_match(lines, ".. autoclass::", "   :member-order: bysource")

        # Write back as UTF-8
        p.write_text("".join(lines), encoding="utf-8")

def shorten_autosummary_titles_all(app: Sphinx, *args: Any) -> None:
    """Hook: run after autosummary has generated stubs, before reading docs."""
    # logger = app.logger
    for rel in ("reference", "functions"):
        shorten_autosummary_titles(Path(app.srcdir) / rel)


def setup(app: Sphinx) -> None:
    app.connect("env-before-read-docs", shorten_autosummary_titles_all)
    app.add_css_file("custom.css")