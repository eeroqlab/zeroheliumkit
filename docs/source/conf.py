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

def setup(app):
    app.add_css_file("custom.css")

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
}