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
import sys
sys.path.insert(0, os.path.abspath('../..'))
autodoc_mock_imports = ["sklearn", "tensorflow", "bokeh", "matplotlib", "ipython3"]


# -- Project information -----------------------------------------------------

project = 'TelescopeML'
copyright = '2021, Ehsan (Sam) Gharib-Nezhad'
author = 'Ehsan (Sam) Gharib-Nezhad'
version = '0.0.0'

# The short X.Y version.
version = '0.0.0'
# The full version, including alpha/beta/rc tags.
release = '0.0.0'

# -- General configuration ---------------------------------------------------

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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

#extensions = [
#]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


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
# html_static_path = ['_static']


# Output file base name for HTML help builder.
htmlhelp_basename = 'TelescopeMLdocs'

highlight_language = 'none'

# html_theme_options = {
#     'display_version': True,
#     'prev_next_buttons_location': 'both',
#     'sticky_navigation': False,
#     'navigation_depth': 3,
#     'includehidden': True,
#     'titles_only': False,
# 	'sidebar_span': 6,  # 1(min) - 12(max)
#
# }

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    html_theme = 'default'
else:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
   'collapse_navigation': True,
	'display_version': True,
	'prev_next_buttons_location': 'both',
	'sticky_navigation': False,
	'navigation_depth': 3,
	'includehidden': True,
	'titles_only': False,
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.
# "<project> v<release> documentation" by default.
#
html_title = 'TelescopeMLdoc'

# A shorter title for the navigation bar.  Default is the same as html_title.
#
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
# html_logo = None

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#
# html_favicon = None

html_static_path = ['_static']


latex_elements = {
     # The paper size ('letterpaper' or 'a4paper').
     #
     'papersize': 'letterpaper',

     # The font size ('10pt', '11pt' or '12pt').
     #
     'pointsize': '10pt',

     # Additional stuff for the LaTeX preamble.
     #
     # 'preamble': '',

     # Latex figure (float) alignment
     #
     # 'figure_align': 'htbp',
}