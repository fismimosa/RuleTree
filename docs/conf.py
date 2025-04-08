# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'RuleTree'
copyright = '2025, Cristiano Landi, Alessio Cascione, Riccardo Guidotti'
author = 'Cristiano Landi, Alessio Cascione, Riccardo Guidotti'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',  # Support for todo items
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.autodoc',  #Include documentation from docstrings
    'sphinx.ext.githubpages',  # Publish HTML docs in GitHub Pages
    'sphinx_favicon',
    'sphinx_copybutton',
    'sphinx_prompt',
    'nbsphinx',
    'versionwarning.extension',
    'sphinx_last_updated_by_git',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "RuleTree Documentation"
html_logo = "_static/logo.png"

html_theme_options = {
    "repository_url": "https://github.com/fismimosa/RuleTree-dev",
    "use_repository_button": True,
}

favicons = [
    {"href": "icon.svg"},
]
