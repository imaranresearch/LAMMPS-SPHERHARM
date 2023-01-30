# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
from datetime import date

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'The implementation of DEM using Spherical Harmonic  in LAMMPS'
copyright = '2023, Mohammad Imaran'.format(date.today().year)
author = 'Mohammad Imaran, Kevin J Hanley, Kevin Strford'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = ['sphinx.ext.mathjax']
# extensions = ['sphinx.ext.autodoc',
# 'sphinx.ext.doctest',
# 'sphinx.ext.intersphinx',
# 'sphinx.ext.todo',
# 'sphinx.ext.coverage',
# 'sphinx.ext.ifconfig',
# 'sphinx.ext.viewcode',
# 'sphinx.ext.githubpages',
# 'sphinx.ext.imgmath']
extensions =['sphinxcontrib.bibtex']
bibtex_bibfiles = ['./refs.bib']
bibtex_default_style = 'unsrt'

# The master toctree document.
master_doc = 'index'


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'sphinxdoc'
html_theme = 'lammps_theme'
html_theme_options = {
   'logo_only' : True,
   'navigation_depth': 3,
   'collapse_navigation': True
}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['_themes']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "SPHERHARM documentation"
html_logo = '_static/logo.png'

html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/lammps.css',
]


# # -- Options for LaTeX output ---------------------------------------------

# latex_toplevel_sectioning = 'section'
# latex_engine = 'pdflatex'

#     # 'fncychap': '\\usepackage[Conny]{fncychap}',
# }
# # The paper size ('letterpaper' or 'a4paper').
# #
# 'papersize': 'a4paper',
# 'releasename':" ",
# # Sonny, Lenny, Glenn, Conny, Rejne, Bjarne and Bjornstrup
# # 'fncychap': '\\usepackage[Lenny]{fncychap}',
# # 'fncychap': '\\usepackage{fncychap}',
# 'fontpkg': '\\usepackage{amsmath,amsfonts,amssymb,amsthm}',
# 'figure_align':'htbp',
# # The font size ('10pt', '11pt' or '12pt').
# #
# 'pointsize': '12pt',
# # Additional stuff for the LaTeX preamble.
# #

# # 'sphinxsetup': \
# # 'hmargin={0.7in,0.7in}, vmargin={1in,1in}, \
# # verbatimwithframe=true, \
# # TitleColor={rgb}{0,0,0}, \
# # HeaderFamily=\\rmfamily\\bfseries, \
# # InnerLinkColor={rgb}{0,0,1}, \
# # OuterLinkColor={rgb}{0,0,1}',
# # 'tableofcontents':' ',

# }
# # latex_logo = 'logo.png'
# # Grouping the document tree into LaTeX files. List of tuples
# # (source start file, target name, title,
# # author, documentclass [howto, manual, or own class]).

latex_documents = [
(master_doc, 'main.tex', 'Manual for SPHERHARM user package',
'Mohammad Imaran, Kevin J Hanley', 'manual')
]
latex_toplevel_sectioning = 'section'