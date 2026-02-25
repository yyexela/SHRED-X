import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SHRED-X"
copyright = "2026, Kutz Lab"
author = "Kutz Lab"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx_design",
]

nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_options = {
    # Do not add members on automodule pages; rely on per-object stubs instead.
    "inherited-members": False,  # do not pull in torch.nn.Module methods
    "show-inheritance": False,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "SHRED-X Documentation"
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False

html_theme_options = {
    "logo": {
        "image_light": "_static/images/light_logo.svg",
        "image_dark": "_static/images/dark_logo.svg",
    },
    "github_url": "https://github.com/CTF-for-Science/ctf4science",
}

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Numpydoc configuration: do not show or link inherited methods
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
