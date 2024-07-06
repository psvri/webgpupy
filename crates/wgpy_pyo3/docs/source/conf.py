# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "webgpupy"
copyright = "2024, psvri"
author = "psvri"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    # Third-party extensions
    "numpydoc",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_title = project
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "collapse_navigation": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/psvri/webgpupy",
            "icon": "fa-brands fa-github",
        },
    ],
}

# Github integration
html_context = {
    "display_github": True,
    "github_user": "psvri",
    "github_repo": "webgpupy",
    "github_version": "main",
    "doc_path": "/crates/wgpy_pyo3/docs/source/",
}
