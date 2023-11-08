# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sphinx_rtd_theme
import sys
sys.path.insert(0, os.path.abspath('/Users/egharibn/RESEARCH/ml/projects/TelescopeML_project/TelescopeML/'))
autodoc_mock_imports = ["sklearn", "tensorflow", "bokeh", "matplotlib", "ipython3"]


# -- Project information -----------------------------------------------------

project = 'TelescopeML'
copyright = '2023, Ehsan (Sam) Gharib-Nezhad'
author = 'Ehsan (Sam) Gharib-Nezhad'

# The full version, including alpha/beta/rc tags
release = '0.0.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [  'sphinx_copybutton',
		'sphinx.ext.autodoc', 
		'sphinx.ext.coverage', 
		'sphinx.ext.napoleon', 
 		'sphinx_gallery.load_style',
		'nbsphinx',
		'sphinx.ext.mathjax',
		]


nbsphinx_allow_errors = False

nbsphinx_execute = 'always'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['**.ipynb_checkpoints']

#exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# ------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'TelescopeMLdocs'

highlight_language = 'none'

html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'sticky_navigation': False,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False,
}


#nbsphinx_prolog = """
#{% set docname = env.doc2path(env.docname, base=None) %}
#.. note::  `Download full notebook here <https://github.com/EhsanGharibNezhad/TelescopeML/tree/master/docs/{{ docname }}>`_
#"""
###.. only:: html
