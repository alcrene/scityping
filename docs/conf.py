# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
from importlib.metadata import version

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Scityping'
author = 'Alexandre René'

# The full version, including alpha/beta/rc tags

try:
    version = version("scityping")
except Exception:
    # Most likely project was not installed with setuptools
    version = "unknown"
release = version
this_year = datetime.date.today().year
copyright = f"2022-{this_year}, Alexandre René"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# autosummary config
autosummary_imported_members = True

# Intersphinx config
intersphinx_mapping = {
    'python': ("https://docs.python.org/3", None),
    'pydantic': ("https://docs.pydantic.dev/latest", None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}

intersphinx_disabled_reftypes = ["*"]

# Myst config
myst_heading_anchors = 3
myst_enable_extensions = [
    "dollarmath"
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

