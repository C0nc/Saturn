"""Sphinx configuration. Notebooks are stored with their outputs and are not executed; heavy imports are mocked for autodoc."""
import os, sys
sys.path.insert(0, os.path.abspath("../src"))

project = "Saturn"
author = "Yan Cui"
copyright = "2026, Yan Cui"
release = "1.0.0"

extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx_design", "sphinx_copybutton"]
autodoc_mock_imports = ["torch", "numpy", "scipy", "pandas", "sklearn", "matplotlib", "gaston", "glmpca", "anndata", "mudata", "snapatac2"]
autodoc_member_order = "bysource"
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath", "html_image"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "jupyter_execute", "tutorials/_src", "examples", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.header", "misc.highlighting_failure"]

html_theme = "furo"
html_title = "Saturn"
html_logo = "_static/logo.png"
html_favicon = "_static/logo_mark.png"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/C0nc/Saturn/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {"color-brand-primary": "#2B3F6B", "color-brand-content": "#3E7987"},
    "dark_css_variables": {"color-brand-primary": "#9CC5C9", "color-brand-content": "#9CC5C9"},
}
