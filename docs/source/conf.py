import sys
import os

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

from typing import Any
from sphinx.application import Sphinx


def remove_line_containing_string(multilines: str, excludetext: str):
    cleaned_lines = [line for line in multilines if excludetext not in line]
    return cleaned_lines


def insert_line_after_match(multilines: str, match_text: str, new_line_text: str):
    """
    Inserts a new line of text after the first line that contains the match_text.

    Parameters
    ----------
    multilines : str
        Target multilines.
    match_text : str
        Text to search for in lines.
    new_line_text : str
        The line to insert after the match (no newline needed).

    Returns
    -------
    bool
        True if insertion occurred, False if match_text was not found.
    """
    newlines = []
    inserted = False
    
    for line in multilines:
        newlines.append(line)
        if not inserted and match_text in line:
            newlines.append(new_line_text)
            inserted = True

    return newlines


def shorten_autosummary_titles(autosummary_dir) -> None:

    for filename in os.listdir(autosummary_dir):

        path = os.path.join(autosummary_dir, filename)
        with open(path, "r") as f:
            lines = f.readlines()
        
        short = lines[0].strip().rsplit(".", 1)[-1]
        lines[0] = short + "\n"
        lines[1] = "=" * len(short) + "\n" # Adjust for underline length

        lines = remove_line_containing_string(lines, "__init__")
        lines = insert_line_after_match(lines, ".. autoclass::", "   :member-order: bysource")

        with open(path, "w") as f:
            f.writelines(lines)


def shorten_autosummary_titles_all(app: Sphinx, *args: Any) -> None:
    """Remove module and class from the autosummary titles."""
    autosummary_dir = os.path.join(app.srcdir, "reference")
    shorten_autosummary_titles(autosummary_dir)

    autosummary_dir = os.path.join(app.srcdir, "functions")
    shorten_autosummary_titles(autosummary_dir)


def setup(app: Sphinx) -> None:
    app.connect("env-before-read-docs", shorten_autosummary_titles_all)
    app.add_css_file("custom.css")