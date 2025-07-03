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
    'sphinx.ext.napoleon',
    # Additional useful extensions
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.mathjax',      # Better math support
    'sphinx_design',           # Responsive web components
    'sphinx.ext.autosummary',  # Generate summary tables
    'sphinx_togglebutton',     # Add toggle buttons to content
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Display only class name in headings, similar to scikit-learn
add_module_names = False  # Don't prefix members with module name
autodoc_typehints = 'signature'  # Show type hints in signature
autodoc_member_order = 'bysource'  # Preserve the order of members as in the source

# NumPy-style table of contents configuration
toc_object_entries = True  # Show objects (classes, functions) in TOC
toc_object_entries_show_parents = "hide"  # Hide the parent module/class names

# Configure autodoc defaults to control what gets documented
autodoc_default_options = {
    'members': True,  # Document all members
    'undoc-members': True,  # Document members without docstrings
    'show-inheritance': True,  # Show base classes
    'inherited-members': False,  # Don't show inherited members by default
    'member-order': 'bysource',  # Keep original source order
    'special-members': '__init__',  # Document special methods like __init__
    'private-members': False,  # Don't document _private members
}

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

# Load custom CSS
html_css_files = [
    'custom.css',
]

html_theme_options = {
    "repository_url": "https://github.com/fismimosa/RuleTree",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "announcement": "This project is in active development. API may change.",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/fismimosa/RuleTree",
            "icon": "fa-brands fa-github",
        }
    ],
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo_dark.png",  # Add dark mode logo if available
    },
}

# Favicon configuration
favicons = [
    {"href": "icon.png"},
]

# Intersphinx configuration to link to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# Code block styling and copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
copybutton_line_continuation_character = "\\"


# Enable autosectionlabel to make it easier to link to sections
autosectionlabel_prefix_document = True

# Enable autosummary features
autosummary_generate = True

# Add last updated timestamp to each page
html_last_updated_fmt = "%b %d, %Y"


