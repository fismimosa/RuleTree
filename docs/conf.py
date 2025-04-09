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
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Display only class name in headings, similar to scikit-learn
add_module_names = False  # Don't prefix members with module name
autodoc_typehints = 'signature'  # Show type hints in signature
autodoc_member_order = 'bysource'  # Preserve the order of members as in the source

# -- Options for Napoleon extension ------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

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
