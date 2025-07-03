# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../zeroheliumkit/src'))

project = 'ZeroHeliumKit'
author = 'EeroQ'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',   
    'sphinx.ext.napoleon',
    "sphinx.ext.viewcode"
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'

html_logo = "_static/zhk.png"
html_static_path = ["_static"]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
