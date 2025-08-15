import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

project = 'ZeroHeliumKit'
copyright = '2025, EeroQ'
author = 'EeroQ'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'nbsphinx']

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

pygments_style = 'sphinx'

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

# def setup(app):
#     app.add_css_file("custom.css")

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
}

add_module_names = False

from typing import Any
from sphinx.application import Sphinx

def shorten_autosummary_titles(app: Sphinx, *args: Any) -> None:
    """Remove module and class from the autosummary titles."""
    autosummary_dir = os.path.join(app.srcdir, "reference") # Adjust "api" and "_autosummary" if your autosummary output directory is different
    if not os.path.exists(autosummary_dir):
        return

    for filename in os.listdir(autosummary_dir):
        if not filename.endswith(".rst"):
            continue

        path = os.path.join(autosummary_dir, filename)
        with open(path, "r") as f:
            lines = f.readlines()

        # Skip if missing a title or if already shortened
        if not lines or lines[0].count(".") < 2:
            continue

        short = lines[0].strip().rsplit(".", 1)[-1]
        lines[0] = short + "\n"
        lines[1] = "=" * len(short) + "\n" # Adjust for underline length

        with open(path, "w") as f:
            f.writelines(lines)

def setup(app: Sphinx) -> None:
    app.connect("env-before-read-docs", shorten_autosummary_titles)
    app.add_css_file("custom.css")