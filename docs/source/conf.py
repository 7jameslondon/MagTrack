import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

project = 'MagTrack'
copyright = '2025, James London'
author = 'James London'
release = '0.7.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    'autoapi.extension',
]

autoapi_type = "python"
autoapi_dirs = ["../../magtrack"]
autoapi_add_toctree_entry = True
autoapi_root = "api"

autodoc_mock_imports = ["cupy"]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']


autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True