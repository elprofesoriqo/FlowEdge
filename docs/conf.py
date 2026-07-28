project = "FlowEdge"
author = "FlowEdge"
copyright = "2026, FlowEdge"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.mathjax",
]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
source_suffix = {".md": "markdown"}

html_theme = "furo"
html_title = "FlowEdge"
html_logo = "_static/surfingtux.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7b2733",
        "color-brand-content": "#7b2733",
        "color-background-primary": "#f7efdc",
        "color-background-secondary": "#f1e6cc",
        "color-background-hover": "#ece0c4",
        "color-background-border": "#e2d4b6",
        "color-foreground-primary": "#2b2521",
        "color-foreground-secondary": "#5c5248",
        "color-foreground-muted": "#8a7d6d",
        "color-code-background": "#f2e8d0",
        "color-code-foreground": "#5e1f2b",
        "color-admonition-background": "#f1e6cc",
        "color-highlight-on-target": "#fcefd4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#dd94a0",
        "color-brand-content": "#e3a6b0",
        "color-background-primary": "#211b19",
        "color-background-secondary": "#2a2320",
        "color-background-hover": "#342a25",
        "color-background-border": "#3b302b",
        "color-foreground-primary": "#ece0cb",
        "color-foreground-secondary": "#c9bba6",
        "color-foreground-muted": "#9a8b78",
        "color-code-background": "#2a2320",
        "color-code-foreground": "#e6c9b8",
        "color-admonition-background": "#2a2320",
    },
}

exclude_patterns = ["_build", "skill.md", "overview.md", "tasks.md", "README.md"]
