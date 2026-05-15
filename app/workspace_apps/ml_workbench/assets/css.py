"""Standalone CSS helpers for the ML Workbench app.

This module keeps the ML Workbench styling self-contained so the app can look
polished in standalone mode without depending on AgentShell. The structure is
kept intentionally similar to the broader shell theme system: theme tokens,
small helpers, one CSS builder, and one public injector.
"""

from __future__ import annotations

import streamlit as st


THEMES = {
    "light": {
        "bg": "#f7f8fc",
        "bg_top": "#ffffff",
        "surface": "rgba(255, 255, 255, 0.92)",
        "surface_strong": "rgba(255, 255, 255, 0.98)",
        "surface_soft": "#f8fbff",
        "surface_alt": "#eef4ff",
        "border": "rgba(28, 39, 60, 0.10)",
        "border_strong": "rgba(43, 92, 230, 0.24)",
        "text": "#1b2436",
        "muted": "#6b7280",
        "heading": "#14213d",
        "primary": "#2563eb",
        "primary_2": "#0ea5e9",
        "success": "#10b981",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "shadow": "0 22px 48px rgba(15, 23, 42, 0.08)",
        "shadow_soft": "0 10px 24px rgba(15, 23, 42, 0.06)",
        "hero_bg": "linear-gradient(135deg, #eff6ff 0%, #ffffff 52%, #f0fdf4 100%)",
        "hero_border": "rgba(37, 99, 235, 0.14)",
        "button_secondary_bg": "rgba(255, 255, 255, 0.98)",
        "button_secondary_text": "#23419c",
        "button_secondary_border": "rgba(28, 39, 60, 0.12)",
        "button_secondary_hover": "#eef4ff",
        "button_primary_text": "#ffffff",
        "uploader_bg": "rgba(239, 246, 255, 0.92)",
        "uploader_border": "rgba(37, 99, 235, 0.35)",
        "chip_bg": "rgba(37, 99, 235, 0.08)",
        "chip_text": "#23419c",
        "info_bg": "rgba(37, 99, 235, 0.07)",
        "info_border": "rgba(37, 99, 235, 0.18)",
        "success_bg": "rgba(16, 185, 129, 0.08)",
        "success_border": "rgba(16, 185, 129, 0.18)",
        "error_bg": "rgba(239, 68, 68, 0.07)",
        "error_border": "rgba(239, 68, 68, 0.18)",
    },
    "dark": {
        "bg": "#0b1220",
        "bg_top": "#0f172a",
        "surface": "rgba(17, 24, 39, 0.94)",
        "surface_strong": "rgba(15, 23, 42, 0.98)",
        "surface_soft": "rgba(30, 41, 59, 0.92)",
        "surface_alt": "rgba(30, 41, 59, 0.98)",
        "border": "rgba(148, 163, 184, 0.18)",
        "border_strong": "rgba(96, 165, 250, 0.34)",
        "text": "#e5eefc",
        "muted": "#94a3b8",
        "heading": "#f8fbff",
        "primary": "#60a5fa",
        "primary_2": "#34d399",
        "success": "#34d399",
        "warning": "#f59e0b",
        "danger": "#f87171",
        "shadow": "0 22px 48px rgba(2, 6, 23, 0.28)",
        "shadow_soft": "0 10px 24px rgba(2, 6, 23, 0.22)",
        "hero_bg": "linear-gradient(135deg, rgba(30, 41, 59, 0.98) 0%, rgba(15, 23, 42, 0.98) 55%, rgba(17, 24, 39, 0.98) 100%)",
        "hero_border": "rgba(96, 165, 250, 0.18)",
        "button_secondary_bg": "rgba(15, 23, 42, 0.96)",
        "button_secondary_text": "#dbeafe",
        "button_secondary_border": "rgba(148, 163, 184, 0.18)",
        "button_secondary_hover": "rgba(30, 41, 59, 0.98)",
        "button_primary_text": "#ffffff",
        "uploader_bg": "rgba(15, 23, 42, 0.82)",
        "uploader_border": "rgba(96, 165, 250, 0.34)",
        "chip_bg": "rgba(96, 165, 250, 0.12)",
        "chip_text": "#dbeafe",
        "info_bg": "rgba(59, 130, 246, 0.09)",
        "info_border": "rgba(96, 165, 250, 0.20)",
        "success_bg": "rgba(16, 185, 129, 0.09)",
        "success_border": "rgba(52, 211, 153, 0.20)",
        "error_bg": "rgba(239, 68, 68, 0.09)",
        "error_border": "rgba(248, 113, 113, 0.20)",
    },
}

BASE_TOKENS = {
    "radius": "20px",
    "radius_sm": "14px",
    "radius_xs": "10px",
    "radius_pill": "999px",
    "max_width": "100%",
    "hero_height": "minmax(118px, auto)",
    "font_sans": '"DM Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    "font_serif": '"Playfair Display", Georgia, serif',
    "font_mono": '"JetBrains Mono", "SFMono-Regular", Consolas, monospace',
}

FONT_IMPORTS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Playfair+Display:wght@600;700&display=swap');
"""


def _rgb(hex_color: str) -> str:
    """Convert a hex color like '#2563eb' into an RGB triplet string."""
    color = hex_color.lstrip("#")
    return ", ".join(str(int(color[i : i + 2], 16)) for i in (0, 2, 4))



def _theme_vars(theme: dict[str, str]) -> str:
    """Return CSS variable declarations for the supplied theme."""
    merged = {**BASE_TOKENS, **theme}
    vars_lines = [f"  --mlw-{key.replace('_', '-')}: {value};" for key, value in merged.items()]
    vars_lines.append(f"  --mlw-primary-rgb: {_rgb(merged['primary'])};")
    vars_lines.append(f"  --mlw-heading-rgb: {_rgb(merged['heading'])};")
    return "\n".join(vars_lines)



def build_css(theme_name: str = "light", *, hosted: bool = False) -> str:
    """Build the full CSS payload for the requested theme name."""
    theme = THEMES[theme_name]
    hosted_scope = ".st-key-workspace_host_mount"
    scope_prefix = f"{hosted_scope} " if hosted else ""
    vars_scope = hosted_scope if hosted else ":root"
    app_background_css = ""
    standalone_button_css = ""
    standalone_tab_css = ""
    global_typography_css = ""
    hosted_typography_css = ""
    block_container_css = ""
    hosted_wrapper_surface_css = ""

    if not hosted:
        app_background_css = """
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, var(--mlw-bg-top) 0%, var(--mlw-bg) 100%);
}
"""
        block_container_css = """
.main .block-container {
  max-width: var(--mlw-max-width);
  padding-top: 1.05rem;
  padding-bottom: 2.25rem;
  padding-left: 1.15rem;
  padding-right: 1.15rem;
}
"""
        global_typography_css = """
html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], [data-testid="stApp"] *, button, input, textarea, select {
  font-family: var(--mlw-font-sans) !important;
}
"""
        standalone_button_css = """
div.stButton > button {
  border-radius: var(--mlw-radius-sm) !important;
  border: 1px solid var(--mlw-button-secondary-border) !important;
  background: var(--mlw-button-secondary-bg) !important;
  color: var(--mlw-button-secondary-text) !important;
  font-weight: 700 !important;
  min-height: 2.56rem !important;
  padding: 0.35rem 0.95rem !important;
  transition: all 0.18s ease !important;
  box-shadow: none !important;
}

div.stButton > button span,
div.stButton > button p,
div.stButton > button div {
  color: var(--mlw-button-secondary-text) !important;
  font-weight: 700 !important;
}

div.stButton > button:hover {
  background: var(--mlw-button-secondary-hover) !important;
  border-color: var(--mlw-border-strong) !important;
  transform: translateY(-1px);
}

div.stButton > button[kind="secondary"] {
  background: var(--mlw-button-secondary-bg) !important;
  color: var(--mlw-button-secondary-text) !important;
  border-color: var(--mlw-button-secondary-border) !important;
}

div.stButton > button[kind="secondary"] span,
div.stButton > button[kind="secondary"] p,
div.stButton > button[kind="secondary"] div {
  color: var(--mlw-button-secondary-text) !important;
}

div.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, var(--mlw-primary) 0%, var(--mlw-primary-2) 100%) !important;
  border-color: transparent !important;
  color: var(--mlw-button-primary-text) !important;
  box-shadow: 0 12px 28px rgba(var(--mlw-primary-rgb), 0.18) !important;
}

div.stButton > button[kind="primary"] span,
div.stButton > button[kind="primary"] p,
div.stButton > button[kind="primary"] div {
  color: var(--mlw-button-primary-text) !important;
}
"""
        standalone_tab_css = """
button[data-baseweb="tab"] {
  border-radius: var(--mlw-radius-sm) var(--mlw-radius-sm) 0 0 !important;
  color: var(--mlw-muted) !important;
  background: transparent !important;
  border: 0 !important;
  font-weight: 700 !important;
  padding-left: 0.95rem !important;
  padding-right: 0.95rem !important;
  transition: all 0.18s ease !important;
}

button[data-baseweb="tab"]:hover {
  color: var(--mlw-primary) !important;
  background: rgba(var(--mlw-primary-rgb), 0.06) !important;
}

button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span,
button[data-baseweb="tab"] div {
  color: inherit !important;
  font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--mlw-primary) !important;
  background: rgba(var(--mlw-primary-rgb), 0.08) !important;
  box-shadow: 0 0 0 1px rgba(var(--mlw-primary-rgb), 0.10) inset !important;
}

button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span,
button[data-baseweb="tab"][aria-selected="true"] div {
  color: var(--mlw-primary) !important;
}

[data-baseweb="tab-highlight"] {
  background: linear-gradient(90deg, var(--mlw-primary) 0%, var(--mlw-primary-2) 100%) !important;
  height: 3px !important;
  border-radius: 999px !important;
}
"""
        hosted_wrapper_surface_css = """
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stFileUploader"]),
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stDataFrame"]),
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stInfo"]),
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stSuccess"]),
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stError"]) {
          background: linear-gradient(180deg, var(--mlw-surface-strong) 0%, var(--mlw-surface) 100%);
          border: 1px solid var(--mlw-border);
          box-shadow: var(--mlw-shadow);
        }
"""
    else:
        hosted_typography_css = f"""
{hosted_scope},
{hosted_scope} *,
{hosted_scope} [data-testid="stMarkdownContainer"],
{hosted_scope} [data-testid="stMarkdownContainer"] *,
{hosted_scope} button,
{hosted_scope} input,
{hosted_scope} textarea,
{hosted_scope} select {{
  font-family: var(--mlw-font-sans) !important;
}}
"""

    return f"""
<style>
{FONT_IMPORTS}

{vars_scope} {{
{_theme_vars(theme)}
}}

{global_typography_css}
{app_background_css}
{hosted_typography_css}

{block_container_css}

{scope_prefix}h1, {scope_prefix}h2, {scope_prefix}h3, {scope_prefix}h4, {scope_prefix}h5, {scope_prefix}h6,
{scope_prefix}[data-testid="stMarkdownContainer"] h1,
{scope_prefix}[data-testid="stMarkdownContainer"] h2,
{scope_prefix}[data-testid="stMarkdownContainer"] h3,
{scope_prefix}[data-testid="stMarkdownContainer"] h4 {{
  color: var(--mlw-heading) !important;
  letter-spacing: -0.02em;
}}

{scope_prefix}h1, {scope_prefix}h2 {{
  font-family: var(--mlw-font-serif) !important;
}}

{scope_prefix}p, {scope_prefix}li, {scope_prefix}label, {scope_prefix}span, {scope_prefix}div, {scope_prefix}small {{
  color: var(--mlw-text);
}}

{scope_prefix}button, {scope_prefix}button span, {scope_prefix}button p, {scope_prefix}[role="button"], {scope_prefix}[role="button"] span {{
  font-family: var(--mlw-font-sans) !important;
}}

/* Caption text */
{scope_prefix}.stCaption, {scope_prefix}[data-testid="stCaptionContainer"], {scope_prefix}.stMarkdown small {{
  color: var(--mlw-muted) !important;
}}

{scope_prefix}code, {scope_prefix}pre {{
  font-family: var(--mlw-font-mono) !important;
}}

{scope_prefix}hr {{
  border-color: var(--mlw-border);
}}

/* Generic card surfaces */
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: var(--mlw-radius);
}}

{hosted_wrapper_surface_css}

        {scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) {{
          background: linear-gradient(180deg, var(--mlw-surface-strong) 0%, var(--mlw-surface) 100%);
          border: 1px solid var(--mlw-border);
          box-shadow: var(--mlw-shadow);
        }}

        {scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.stSelectbox):not(:has(.mlw-surface-panel-marker)),
        {scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.stMultiSelect):not(:has(.mlw-surface-panel-marker)) {{
          background: transparent !important;
          border: 0 !important;
          box-shadow: none !important;
        }}

/* Hero helpers for future markdown/HTML blocks */
.mlw-hero {{
  display: grid;
  align-items: center;
  min-height: var(--mlw-hero-height);
  padding: 1.1rem 1.2rem;
  border-radius: calc(var(--mlw-radius) + 4px);
  border: 1px solid var(--mlw-hero-border);
  background: var(--mlw-hero-bg);
  box-shadow: var(--mlw-shadow);
  margin-bottom: 0.8rem;
}}

.mlw-hero-title {{
  margin: 0 0 0.18rem 0;
  font-family: var(--mlw-font-serif);
  font-size: 1.8rem;
  line-height: 1.08;
  color: var(--mlw-heading);
}}

.mlw-hero-subtitle {{
  margin: 0;
  font-size: 0.98rem;
  color: var(--mlw-muted);
}}

/* Metric cards styled like shell/admin cards */
{scope_prefix}[data-testid="stMetric"] {{
  background: linear-gradient(180deg, var(--mlw-surface-strong) 0%, var(--mlw-surface) 100%) !important;
  border: 1px solid var(--mlw-border) !important;
  border-radius: var(--mlw-radius-sm) !important;
  padding: 0.92rem 0.98rem !important;
  box-shadow: var(--mlw-shadow-soft) !important;
}}

{scope_prefix}[data-testid="stMetricLabel"] {{
  color: var(--mlw-muted) !important;
  font-weight: 700 !important;
}}

{scope_prefix}[data-testid="stMetricValue"] {{
  color: var(--mlw-heading) !important;
}}

/* Hosted workflow stage selector buttons */
{scope_prefix}.st-key-ml_workbench_stage_button_profile button,
{scope_prefix}.st-key-ml_workbench_stage_button_preprocess button,
{scope_prefix}.st-key-ml_workbench_stage_button_features button,
{scope_prefix}.st-key-ml_workbench_stage_button_modeling button,
{scope_prefix}.st-key-ml_workbench_stage_button_results button {{
  min-height: 2.5rem !important;
  border-radius: var(--mlw-radius-sm) !important;
  border: 1px solid var(--mlw-button-secondary-border) !important;
  background: var(--mlw-button-secondary-bg) !important;
  color: var(--mlw-button-secondary-text) !important;
  border-color: var(--mlw-button-secondary-border) !important;
  box-shadow: none !important;
}}

{scope_prefix}.st-key-ml_workbench_stage_button_profile button span,
{scope_prefix}.st-key-ml_workbench_stage_button_profile button p,
{scope_prefix}.st-key-ml_workbench_stage_button_profile button div,
{scope_prefix}.st-key-ml_workbench_stage_button_preprocess button span,
{scope_prefix}.st-key-ml_workbench_stage_button_preprocess button p,
{scope_prefix}.st-key-ml_workbench_stage_button_preprocess button div,
{scope_prefix}.st-key-ml_workbench_stage_button_features button span,
{scope_prefix}.st-key-ml_workbench_stage_button_features button p,
{scope_prefix}.st-key-ml_workbench_stage_button_features button div,
{scope_prefix}.st-key-ml_workbench_stage_button_modeling button span,
{scope_prefix}.st-key-ml_workbench_stage_button_modeling button p,
{scope_prefix}.st-key-ml_workbench_stage_button_modeling button div,
{scope_prefix}.st-key-ml_workbench_stage_button_results button span,
{scope_prefix}.st-key-ml_workbench_stage_button_results button p,
{scope_prefix}.st-key-ml_workbench_stage_button_results button div {{
  color: var(--mlw-button-secondary-text) !important;
}}

{scope_prefix}.st-key-ml_workbench_stage_button_profile button[kind="primary"],
{scope_prefix}.st-key-ml_workbench_stage_button_preprocess button[kind="primary"],
{scope_prefix}.st-key-ml_workbench_stage_button_features button[kind="primary"],
{scope_prefix}.st-key-ml_workbench_stage_button_modeling button[kind="primary"],
{scope_prefix}.st-key-ml_workbench_stage_button_results button[kind="primary"] {{
  background: linear-gradient(90deg, var(--mlw-primary) 0%, var(--mlw-primary-2) 100%) !important;
  color: var(--mlw-button-primary-text) !important;
  border-color: transparent !important;
  box-shadow: 0 12px 24px rgba(var(--mlw-primary-rgb), 0.18) !important;
}}

{scope_prefix}.st-key-ml_workbench_stage_button_profile button[kind="primary"] span,
{scope_prefix}.st-key-ml_workbench_stage_button_profile button[kind="primary"] p,
{scope_prefix}.st-key-ml_workbench_stage_button_profile button[kind="primary"] div,
{scope_prefix}.st-key-ml_workbench_stage_button_preprocess button[kind="primary"] span,
{scope_prefix}.st-key-ml_workbench_stage_button_preprocess button[kind="primary"] p,
{scope_prefix}.st-key-ml_workbench_stage_button_preprocess button[kind="primary"] div,
{scope_prefix}.st-key-ml_workbench_stage_button_features button[kind="primary"] span,
{scope_prefix}.st-key-ml_workbench_stage_button_features button[kind="primary"] p,
{scope_prefix}.st-key-ml_workbench_stage_button_features button[kind="primary"] div,
{scope_prefix}.st-key-ml_workbench_stage_button_modeling button[kind="primary"] span,
{scope_prefix}.st-key-ml_workbench_stage_button_modeling button[kind="primary"] p,
{scope_prefix}.st-key-ml_workbench_stage_button_modeling button[kind="primary"] div,
{scope_prefix}.st-key-ml_workbench_stage_button_results button[kind="primary"] span,
{scope_prefix}.st-key-ml_workbench_stage_button_results button[kind="primary"] p,
{scope_prefix}.st-key-ml_workbench_stage_button_results button[kind="primary"] div {{
  color: var(--mlw-button-primary-text) !important;
}}

{standalone_button_css}

/* Buttons / workflow nav */
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button {{
  border-radius: var(--mlw-radius-sm) !important;
  border: 1px solid var(--mlw-button-secondary-border) !important;
  background: var(--mlw-button-secondary-bg) !important;
  color: var(--mlw-button-secondary-text) !important;
  font-weight: 700 !important;
  min-height: 2.56rem !important;
  padding: 0.35rem 0.95rem !important;
  transition: all 0.18s ease !important;
  box-shadow: none !important;
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button div,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button div,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button div {{
  color: var(--mlw-button-secondary-text) !important;
  font-weight: 700 !important;
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button:hover,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button:hover,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button:hover {{
  background: var(--mlw-button-secondary-hover) !important;
  border-color: var(--mlw-border-strong) !important;
  transform: translateY(-1px);
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button[kind="secondary"],
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button[kind="secondary"],
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button[kind="secondary"] {{
  background: var(--mlw-button-secondary-bg) !important;
  color: var(--mlw-button-secondary-text) !important;
  border-color: var(--mlw-button-secondary-border) !important;
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button[kind="secondary"] span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button[kind="secondary"] p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button[kind="secondary"] div,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button[kind="secondary"] span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button[kind="secondary"] p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button[kind="secondary"] div,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button[kind="secondary"] span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button[kind="secondary"] p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button[kind="secondary"] div {{
  color: var(--mlw-button-secondary-text) !important;
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button[kind="primary"],
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button[kind="primary"],
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button[kind="primary"] {{
  background: linear-gradient(90deg, var(--mlw-primary) 0%, var(--mlw-primary-2) 100%) !important;
  border-color: transparent !important;
  color: var(--mlw-button-primary-text) !important;
  box-shadow: 0 12px 28px rgba(var(--mlw-primary-rgb), 0.18) !important;
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button[kind="primary"] span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button[kind="primary"] p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-surface-panel-marker) div.stButton > button[kind="primary"] div,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button[kind="primary"] span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button[kind="primary"] p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-hero) div.stButton > button[kind="primary"] div,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button[kind="primary"] span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button[kind="primary"] p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-status-message) div.stButton > button[kind="primary"] div {{
  color: var(--mlw-button-primary-text) !important;
}}

{scope_prefix}div.stDownloadButton > button {{
  border-radius: var(--mlw-radius-sm) !important;
  border: 1px solid var(--mlw-button-secondary-border) !important;
  background: var(--mlw-button-secondary-bg) !important;
  color: var(--mlw-button-secondary-text) !important;
  font-weight: 700 !important;
  min-height: 2.56rem !important;
  padding: 0.35rem 0.95rem !important;
  transition: all 0.18s ease !important;
  box-shadow: none !important;
}}

{scope_prefix}div.stDownloadButton > button span,
{scope_prefix}div.stDownloadButton > button p,
{scope_prefix}div.stDownloadButton > button div {{
  color: var(--mlw-button-secondary-text) !important;
  font-weight: 700 !important;
}}

{scope_prefix}div.stDownloadButton > button:hover {{
  background: var(--mlw-button-secondary-hover) !important;
  border-color: var(--mlw-border-strong) !important;
  transform: translateY(-1px);
}}

/* Compact toggle buttons used in the modeling training settings area */
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) div.stButton > button {{
  min-height: 2.18rem !important;
  padding: 0.2rem 0.72rem !important;
  border-radius: var(--mlw-radius-xs) !important;
  font-size: 0.86rem !important;
  line-height: 1.1 !important;
  box-shadow: none !important;
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) div.stButton > button[kind="secondary"] {{
  background: var(--mlw-surface-soft) !important;
  color: var(--mlw-chip-text) !important;
  border-color: rgba(var(--mlw-primary-rgb), 0.18) !important;
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) div.stButton > button[kind="secondary"] span,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) div.stButton > button[kind="secondary"] p,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) div.stButton > button[kind="secondary"] div {{
  color: var(--mlw-chip-text) !important;
}}

{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) div.stButton > button[kind="primary"] {{
  box-shadow: 0 8px 18px rgba(var(--mlw-primary-rgb), 0.16) !important;
}}

/* Keep short input controls in the training settings area compact and left-packed */
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) .stNumberInput,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) .stSlider,
{scope_prefix}[data-testid="stVerticalBlockBorderWrapper"]:has(.mlw-compact-toggle-group) .stTextInput {{
  max-width: 13rem !important;
}}

/* Inputs */
{scope_prefix}[data-baseweb="input"] > div,
{scope_prefix}[data-baseweb="select"] > div,
{scope_prefix}textarea,
{scope_prefix}.stTextInput input,
{scope_prefix}.stNumberInput input {{
  background: var(--mlw-surface-soft) !important;
  border: 1px solid var(--mlw-border) !important;
  color: var(--mlw-text) !important;
  border-radius: var(--mlw-radius-sm) !important;
}}

{scope_prefix}[data-baseweb="input"] input,
{scope_prefix}[data-baseweb="select"] input,
{scope_prefix}.stTextInput input,
{scope_prefix}.stNumberInput input,
{scope_prefix}textarea,
{scope_prefix}[data-baseweb="select"] span {{
  color: var(--mlw-text) !important;
  font-family: var(--mlw-font-sans) !important;
}}

{scope_prefix}label, {scope_prefix}.stSelectbox label, {scope_prefix}.stMultiSelect label, {scope_prefix}.stTextInput label, {scope_prefix}.stSlider label {{
  color: var(--mlw-heading) !important;
  font-weight: 700 !important;
}}

{scope_prefix}[data-baseweb="tag"] {{
  background: var(--mlw-chip-bg) !important;
  color: var(--mlw-chip-text) !important;
  border: 1px solid rgba(var(--mlw-primary-rgb), 0.18) !important;
  border-radius: var(--mlw-radius-pill) !important;
}}

/* Simple radio buttons */
{scope_prefix}[data-testid="stRadio"] label,
{scope_prefix}[data-testid="stRadio"] [role="radiogroup"] label {{
  color: var(--mlw-heading) !important;
  font-weight: 700 !important;
}}

{scope_prefix}[data-testid="stRadio"] [role="radiogroup"] {{
  gap: 0.45rem !important;
}}

{scope_prefix}[data-testid="stRadio"] [role="radiogroup"] > label {{
  border-radius: var(--mlw-radius-sm) !important;
  padding: 0.22rem 0.42rem !important;
  transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
}}

{scope_prefix}[data-testid="stRadio"] [role="radiogroup"] > label:hover {{
  background: rgba(var(--mlw-primary-rgb), 0.05) !important;
}}

 {scope_prefix}[data-testid="stRadio"] input[type="radio"] {{
   accent-color: var(--mlw-primary) !important;
 }}

 {scope_prefix}[data-testid="stRadio"] input[type="radio"]:checked {{
   accent-color: var(--mlw-primary) !important;
 }}

 {scope_prefix}[data-testid="stRadio"] [role="radiogroup"] input[type="radio"] {{
   accent-color: var(--mlw-primary) !important;
   border-color: var(--mlw-primary) !important;
 }}

 {scope_prefix}[data-testid="stRadio"] [role="radiogroup"] input[type="radio"]:checked {{
   accent-color: var(--mlw-primary) !important;
   border-color: var(--mlw-primary) !important;
   box-shadow: 0 0 0 1px rgba(var(--mlw-primary-rgb), 0.18) !important;
 }}

 {scope_prefix}[data-testid="stRadio"] [role="radiogroup"] > label svg,
 {scope_prefix}[data-testid="stRadio"] [role="radiogroup"] > label circle {{
   fill: var(--mlw-primary) !important;
   stroke: var(--mlw-primary) !important;
 }}

{scope_prefix}[data-testid="stRadio"] input[type="radio"] + div,
{scope_prefix}[data-testid="stRadio"] input[type="radio"] + div p,
{scope_prefix}[data-testid="stRadio"] input[type="radio"] + div span {{
  color: var(--mlw-text) !important;
}}

{scope_prefix}[data-testid="stRadio"] input[type="radio"]:checked + div,
{scope_prefix}[data-testid="stRadio"] input[type="radio"]:checked + div p,
{scope_prefix}[data-testid="stRadio"] input[type="radio"]:checked + div span {{
  color: var(--mlw-primary) !important;
  font-weight: 700 !important;
}}

/* File uploader */
{scope_prefix}[data-testid="stFileUploaderDropzone"] {{
  border-radius: var(--mlw-radius-sm) !important;
  border: 1px dashed var(--mlw-uploader-border) !important;
  background: var(--mlw-uploader-bg) !important;
}}

/* Alerts */
{scope_prefix}[data-testid="stInfo"] {{
  border-radius: var(--mlw-radius-sm) !important;
  border: 1px solid var(--mlw-info-border) !important;
  background: var(--mlw-info-bg) !important;
}}

{scope_prefix}[data-testid="stSuccess"] {{
  border-radius: var(--mlw-radius-sm) !important;
  border: 1px solid var(--mlw-success-border) !important;
  background: var(--mlw-success-bg) !important;
}}

{scope_prefix}[data-testid="stError"] {{
  border-radius: var(--mlw-radius-sm) !important;
  border: 1px solid var(--mlw-error-border) !important;
  background: var(--mlw-error-bg) !important;
}}

/* Progress bars */
{scope_prefix}[data-testid="stProgress"] {{
  margin: 0.2rem 0 0.75rem !important;
}}

{scope_prefix}[data-testid="stProgress"] > div {{
  background: transparent !important;
  box-shadow: none !important;
}}

{scope_prefix}[data-testid="stProgress"] > div > div {{
  background: #e5e7eb !important;
  border-radius: var(--mlw-radius-pill) !important;
  overflow: hidden !important;
  box-shadow: 0 4px 12px rgba(var(--mlw-heading-rgb), 0.06) !important;
  min-height: 0.72rem !important;
}}

{scope_prefix}[data-testid="stProgress"] div[role="progressbar"] {{
  background: linear-gradient(90deg, var(--mlw-primary) 0%, var(--mlw-primary-2) 100%) !important;
  border-radius: var(--mlw-radius-pill) !important;
  box-shadow: 0 6px 16px rgba(var(--mlw-primary-rgb), 0.18) !important;
  min-height: 0.72rem !important;
}}

.mlw-training-progress {{
  margin: 0.2rem 0 0.75rem;
}}

.mlw-training-progress__track {{
  width: 100%;
  min-height: 0.72rem;
  border-radius: var(--mlw-radius-pill);
  background: #e5e7eb;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(var(--mlw-heading-rgb), 0.06);
}}

.mlw-training-progress__fill {{
  min-height: 0.72rem;
  border-radius: var(--mlw-radius-pill);
  background: linear-gradient(90deg, var(--mlw-primary) 0%, var(--mlw-primary-2) 100%);
  box-shadow: 0 6px 16px rgba(var(--mlw-primary-rgb), 0.18);
  transition: width 0.18s ease;
}}

/* Dataframe wrapper only. The inner grid will still look Streamlit until replaced. */
{scope_prefix}[data-testid="stDataFrame"] {{
  border: 1px solid var(--mlw-border) !important;
  border-radius: var(--mlw-radius-sm) !important;
  overflow: hidden !important;
  box-shadow: var(--mlw-shadow-soft) !important;
}}

{standalone_tab_css}

/* Tabs */
{scope_prefix}.stTabs button[data-baseweb="tab"] {{
  border-radius: var(--mlw-radius-sm) var(--mlw-radius-sm) 0 0 !important;
  color: var(--mlw-muted) !important;
  background: transparent !important;
  border: 0 !important;
  font-weight: 700 !important;
  padding-left: 0.95rem !important;
  padding-right: 0.95rem !important;
  transition: all 0.18s ease !important;
}}

{scope_prefix}.stTabs button[data-baseweb="tab"]:hover {{
  color: var(--mlw-primary) !important;
  background: rgba(var(--mlw-primary-rgb), 0.06) !important;
}}

{scope_prefix}.stTabs button[data-baseweb="tab"] p,
{scope_prefix}.stTabs button[data-baseweb="tab"] span,
{scope_prefix}.stTabs button[data-baseweb="tab"] div {{
  color: inherit !important;
  font-weight: 700 !important;
}}

{scope_prefix}.stTabs button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--mlw-primary) !important;
  background: rgba(var(--mlw-primary-rgb), 0.08) !important;
  box-shadow: 0 0 0 1px rgba(var(--mlw-primary-rgb), 0.10) inset !important;
}}

{scope_prefix}.stTabs button[data-baseweb="tab"][aria-selected="true"] p,
{scope_prefix}.stTabs button[data-baseweb="tab"][aria-selected="true"] span,
{scope_prefix}.stTabs button[data-baseweb="tab"][aria-selected="true"] div {{
  color: var(--mlw-primary) !important;
}}

{scope_prefix}.stTabs [data-baseweb="tab-highlight"] {{
  background: linear-gradient(90deg, var(--mlw-primary) 0%, var(--mlw-primary-2) 100%) !important;
  height: 3px !important;
  border-radius: 999px !important;
}}

/* Custom HTML-backed layout elements */
.mlw-metric-grid {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.9rem;
  margin: 0.15rem 0 0.95rem;
}}

.mlw-m-card {{
  border: 1px solid var(--mlw-border);
  border-radius: var(--mlw-radius-sm);
  background: linear-gradient(180deg, var(--mlw-surface-strong) 0%, var(--mlw-surface) 100%);
  box-shadow: var(--mlw-shadow-soft);
  padding: 0.95rem 1rem;
}}

.mlw-m-val {{
  color: var(--mlw-heading);
  font-size: 1.5rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.05;
}}

.mlw-m-label {{
  margin-top: 0.3rem;
  color: var(--mlw-muted);
  font-size: 0.84rem;
  font-weight: 700;
}}

.mlw-metric-delta {{
  margin-top: 0.32rem;
  color: var(--mlw-primary);
  font-size: 0.8rem;
  font-weight: 700;
}}

.mlw-status-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin: 0.25rem 0 0.95rem;
}}

.mlw-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.34rem 0.78rem;
  border-radius: var(--mlw-radius-pill);
  border: 1px solid var(--mlw-border);
  background: var(--mlw-chip-bg);
  color: var(--mlw-chip-text);
  font-size: 0.82rem;
  font-weight: 700;
}}

.mlw-badge.info {{
  background: var(--mlw-chip-bg);
  color: var(--mlw-chip-text);
}}

.mlw-badge.success {{
  background: var(--mlw-success-bg);
  color: var(--mlw-success);
  border-color: var(--mlw-success-border);
}}


.mlw-badge.error {{
  background: var(--mlw-error-bg);
  color: var(--mlw-danger);
  border-color: var(--mlw-error-border);
}}

.mlw-status-message {{
  display: inline-flex;
  align-items: center;
  gap: 0.8rem;
  width: auto;
  max-width: 100%;
  box-sizing: border-box;
  border: 1px solid var(--mlw-border);
  border-radius: var(--mlw-radius-sm);
  background: #f3f4f6;
  box-shadow: var(--mlw-shadow-soft);
  padding: 0.9rem 0.95rem;
  margin: 0.2rem 0 0.95rem;
  position: relative;
}}

.mlw-status-message__icon {{
  flex: 0 0 auto;
  width: 0.72rem;
  height: 0.72rem;
  border-radius: 999px;
  background: var(--mlw-muted);
  box-shadow: 0 0 0 4px rgba(var(--mlw-heading-rgb), 0.04);
}}

.mlw-status-message__body {{
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.12rem;
}}

.mlw-status-message__title {{
  color: var(--mlw-heading);
  font-size: 0.92rem;
  font-weight: 800;
  line-height: 1.2;
}}

.mlw-status-message__text {{
  color: var(--mlw-text);
  font-size: 0.92rem;
  font-weight: 600;
  line-height: 1.35;
}}

.mlw-status-message--info {{
  border-color: rgba(var(--mlw-primary-rgb), 0.24);
  box-shadow: 0 0 0 1px rgba(var(--mlw-primary-rgb), 0.06), var(--mlw-shadow-soft);
}}

.mlw-status-message--info .mlw-status-message__icon {{
  background: var(--mlw-primary);
  box-shadow: 0 0 0 4px rgba(var(--mlw-primary-rgb), 0.10);
}}

.mlw-status-message--success {{
  border-color: var(--mlw-success-border);
  box-shadow: 0 0 0 1px rgba(16, 185, 129, 0.08), var(--mlw-shadow-soft);
}}

.mlw-status-message--success .mlw-status-message__icon {{
  background: var(--mlw-success);
  box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.10);
}}

.mlw-status-message--warning {{
  border-color: rgba(245, 158, 11, 0.22);
  box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.08), var(--mlw-shadow-soft);
}}

.mlw-status-message--warning .mlw-status-message__icon {{
  background: var(--mlw-warning);
  box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.10);
}}

.mlw-status-message--error {{
  border-color: var(--mlw-error-border);
  box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.08), var(--mlw-shadow-soft);
}}

.mlw-status-message--error .mlw-status-message__icon {{
  background: var(--mlw-danger);
  box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.10);
}}

.mlw-app-panel {{
  border: 1px solid var(--mlw-border);
  border-radius: var(--mlw-radius);
  background: linear-gradient(180deg, var(--mlw-surface-strong) 0%, var(--mlw-surface) 100%);
  box-shadow: var(--mlw-shadow);
  padding: 1rem 1.05rem;
  margin: 0.15rem 0 0.95rem;
}}

.mlw-app-panel.bordered {{
  border-color: var(--mlw-border);
}}

.mlw-panel-title {{
  margin: 0 0 0.2rem;
  color: var(--mlw-heading);
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}}

.mlw-panel-subtitle {{
  margin: 0 0 0.8rem;
  color: var(--mlw-muted);
  font-size: 0.92rem;
}}

.mlw-panel-body {{
  color: var(--mlw-text);
  font-size: 0.95rem;
}}

.mlw-empty-state {{
  display: inline-block;
  border: 1px dashed var(--mlw-border);
  border-radius: var(--mlw-radius-sm);
  background: var(--mlw-surface-soft);
  padding: 0.9rem 0.95rem;
  color: var(--mlw-muted);
  font-size: 0.93rem;
  margin-bottom: 0.35rem;
  max-width: 100%;
}}

.mlw-kv-grid {{
  display: grid;
  gap: 0.8rem;
  margin: 0.15rem 0 0.9rem;
}}

.mlw-kv-grid.cols-1 {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
.mlw-kv-grid.cols-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.mlw-kv-grid.cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.mlw-kv-grid.cols-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}

.mlw-kv-card {{
  border: 1px solid var(--mlw-border);
  border-radius: var(--mlw-radius-sm);
  background: linear-gradient(180deg, var(--mlw-surface-strong) 0%, var(--mlw-surface) 100%);
  box-shadow: var(--mlw-shadow-soft);
  padding: 0.9rem 0.95rem;
}}

.mlw-kv-label {{
  color: var(--mlw-muted);
  font-size: 0.82rem;
  font-weight: 700;
}}

.mlw-kv-value {{
  margin-top: 0.28rem;
  color: var(--mlw-heading);
  font-size: 1rem;
  font-weight: 700;
}}

.mlw-results-band {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
  margin: 0.2rem 0 0.75rem;
}}

.mlw-result-card {{
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  min-height: 100%;
  border: 1px solid var(--mlw-border);
  border-radius: var(--mlw-radius);
  background: linear-gradient(180deg, var(--mlw-surface-strong) 0%, var(--mlw-surface-soft) 100%);
  box-shadow: var(--mlw-shadow-soft);
  padding: 1rem 1rem 1.05rem;
}}

.mlw-result-card--best {{
  border-color: rgba(16, 185, 129, 0.30);
  background:
    linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.03) 24%, var(--mlw-surface-strong) 100%);
  box-shadow:
    0 0 0 1px rgba(16, 185, 129, 0.10),
    0 18px 38px rgba(16, 185, 129, 0.14);
}}

.mlw-result-card__header {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
}}

.mlw-result-card__title-group {{
  min-width: 0;
}}

.mlw-result-card__title {{
  color: var(--mlw-heading);
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.2;
}}

.mlw-result-card__subtitle {{
  margin-top: 0.18rem;
  color: var(--mlw-muted);
  font-size: 0.86rem;
  font-weight: 600;
}}

.mlw-result-card__meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}}

.mlw-result-card__metrics {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.7rem;
}}

.mlw-result-card__metric {{
  border: 1px solid var(--mlw-border);
  border-radius: var(--mlw-radius-sm);
  background: rgba(var(--mlw-heading-rgb), 0.02);
  padding: 0.82rem 0.85rem;
}}

.mlw-result-card__metric-label {{
  color: var(--mlw-muted);
  font-size: 0.79rem;
  font-weight: 700;
}}

.mlw-result-card__metric-value {{
  margin-top: 0.28rem;
  color: var(--mlw-heading);
  font-size: 1.08rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}}

@media (max-width: 900px) {{
  .main .block-container {{
    padding-top: 0.85rem;
    padding-left: 0.85rem;
    padding-right: 0.85rem;
  }}

  .mlw-metric-grid {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}

  .mlw-kv-grid.cols-2,
  .mlw-kv-grid.cols-3,
  .mlw-kv-grid.cols-4 {{
    grid-template-columns: repeat(1, minmax(0, 1fr));
  }}

  .mlw-result-card__metrics {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}

  .mlw-hero-title {{
    font-size: 1.5rem;
  }}
}}
</style>
"""



def build_ml_workbench_css(*, use_dark_mode: bool = False, hosted: bool = False) -> str:
    """Return the full CSS payload for the requested theme mode."""
    return build_css("dark" if use_dark_mode else "light", hosted=hosted)



def inject_ml_workbench_css(*, use_dark_mode: bool = False, hosted: bool = False) -> None:
    """Inject the ML Workbench CSS into the current Streamlit app.

    Parameters
    ----------
    use_dark_mode:
        Defaults to False so the standalone app remains light by default.
        AgentShell can opt into dark mode later by calling this function with
        ``use_dark_mode=True``.
    hosted:
        When True, avoid page-wide standalone overrides so AgentShell remains
        the owner of the outer shell styling.
    """
    st.markdown(
        build_ml_workbench_css(use_dark_mode=use_dark_mode, hosted=hosted),
        unsafe_allow_html=True,
    )
