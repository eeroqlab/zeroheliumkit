# Configuration file for the Sphinx documentation builder.

project = 'ZeroHeliumKit'
author = 'EeroQ'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',   
    'sphinx.ext.napoleon'  
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
