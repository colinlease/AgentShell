"""Layout helpers for the ML Workbench UI.

These helpers keep the shell-facing app adapter thin by centralizing reusable,
HTML-backed presentation blocks that match the standalone ML Workbench design
language more closely than default Streamlit widgets.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from app.workspace_apps.ml_workbench.constants import APP_LABEL



def _escape(value: Any) -> str:
    """Return an HTML-escaped string representation of a value."""
    return html.escape("" if value is None else str(value))




def _build_badge_html(*, label: str, variant: str = "info") -> str:
    """Build one reusable pill badge."""
    safe_label = _escape(label)
    safe_variant = _escape(variant)
    return f'<div class="mlw-badge {safe_variant}">{safe_label}</div>'


def _build_status_message_html(*, message: str, variant: str = "info", title: str | None = None) -> str:
    """Build one reusable themed status message block."""
    safe_message = _escape(message)
    safe_variant = _escape(variant)
    title_html = ""
    if title:
        title_html = f'<div class="mlw-status-message__title">{_escape(title)}</div>'

    return (
        f'<div class="mlw-status-message mlw-status-message--{safe_variant}">'
        '<div class="mlw-status-message__icon" aria-hidden="true"></div>'
        '<div class="mlw-status-message__body">'
        f'{title_html}'
        f'<div class="mlw-status-message__text">{safe_message}</div>'
        '</div>'
        '</div>'
    )



def render_compact_title(description: str) -> None:
    """Render the standalone ML Workbench hero block."""
    safe_title = _escape(APP_LABEL)
    safe_description = _escape(description)
    st.markdown(
        f"""
        <div class="mlw-hero">
            <div>
                <div class="mlw-hero-title">{safe_title}</div>
                <div class="mlw-hero-subtitle">{safe_description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_metric_row(metrics: list[dict[str, Any]]) -> None:
    """Render a row of custom metric cards.

    Each metric dict should contain:
    - label: required display label
    - value: required metric value
    - delta: optional metric delta
    """
    if not metrics:
        return

    cards_html: list[str] = []
    for metric in metrics:
        label = _escape(metric.get("label", ""))
        value = _escape(metric.get("value", ""))
        delta_value = metric.get("delta")
        delta_html = ""
        if delta_value not in (None, ""):
            delta_html = f'<div class="mlw-metric-delta">{_escape(delta_value)}</div>'

        card_html = (
            '<div class="mlw-m-card">'
            f'<div class="mlw-m-val">{value}</div>'
            f'<div class="mlw-m-label">{label}</div>'
            f'{delta_html}'
            '</div>'
        )
        cards_html.append(card_html)

    metrics_html = ''.join(cards_html)
    st.markdown(
        '<div class="mlw-metric-grid">' + metrics_html + '</div>',
        unsafe_allow_html=True,
    )




def render_section_header(*, title: str, subtitle: str | None = None) -> None:
    """Render a standardized section header."""
    safe_title = _escape(title)
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="mlw-panel-subtitle">{_escape(subtitle)}</div>'

    st.markdown(
        f"""
        <div class="mlw-panel-title">{safe_title}</div>
        {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


# --- Helper containers for explicit surface/flat grouping ---

def create_surface_panel(*, title: str | None = None, subtitle: str | None = None):
    """Create an explicitly surfaced Streamlit container.

    This should be used for sections that intentionally need a bordered card-like
    wrapper. It avoids relying on application-wide CSS rules that try to infer
    whether a block should have a surface based on the widgets inside it.
    """
    panel = st.container(border=True)
    with panel:
        st.markdown(
            '<div class="mlw-surface-panel-marker" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        if title or subtitle:
            render_section_header(title=title or "", subtitle=subtitle)
    return panel


def create_plain_group(*, title: str | None = None, subtitle: str | None = None):
    """Create an explicitly plain Streamlit container.

    This should be used for grouped controls or content that should remain
    visually flat inside another surfaced section.
    """
    group = st.container(border=False)
    with group:
        if title or subtitle:
            render_section_header(title=title or "", subtitle=subtitle)
    return group


def render_empty_state_block(message: str) -> None:
    """Render a simple inline empty-state block.

    This is safer for nested use inside other surfaced panels than rendering a
    second full card inside the current panel.
    """
    safe_message = _escape(message)
    st.markdown(
        f'<div class="mlw-empty-state">{safe_message}</div>',
        unsafe_allow_html=True,
    )



def render_section_card(
    title: str,
    body: str | None = None,
    *,
    caption: str | None = None,
    border: bool = True,
) -> None:
    """Render a compact titled section card using custom HTML styling."""
    safe_title = _escape(title)
    safe_caption = _escape(caption) if caption else ""
    safe_body = _escape(body) if body else ""
    bordered_class = " bordered" if border else ""

    caption_html = (
        f'<div class="mlw-panel-subtitle">{safe_caption}</div>' if caption else ""
    )
    body_html = f'<div class="mlw-panel-body">{safe_body}</div>' if body else ""

    st.markdown(
        f"""
        <div class="mlw-app-panel{bordered_class}">
            <div class="mlw-panel-title">{safe_title}</div>
            {caption_html}
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_info_card(
    title: str,
    message: str,
    *,
    caption: str | None = None,
    border: bool = True,
) -> None:
    """Render an informational section.

    When `border` is True, render a full surfaced card. When `border` is False,
    render only an inline empty-state block so the helper can be used safely
    inside an existing surfaced panel.
    """
    safe_title = _escape(title)
    safe_message = _escape(message)
    caption_html = (
        f'<div class="mlw-panel-subtitle">{_escape(caption)}</div>' if caption else ""
    )

    if not border:
        if title:
            st.markdown(
                f'<div class="mlw-panel-title">{safe_title}</div>',
                unsafe_allow_html=True,
            )
        if caption_html:
            st.markdown(caption_html, unsafe_allow_html=True)
        render_empty_state_block(message)
        return

    st.markdown(
        f"""
        <div class="mlw-app-panel bordered">
            <div class="mlw-panel-title">{safe_title}</div>
            {caption_html}
            <div class="mlw-empty-state">{safe_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_key_value_grid(items: list[dict[str, Any]], columns: int = 3) -> None:
    """Render a compact key/value summary grid using custom cards.

    Each item dict should contain:
    - label: required
    - value: required
    """
    if not items:
        return

    safe_columns = max(1, min(columns, len(items)))
    cards_html: list[str] = []

    for item in items:
        label = _escape(item.get("label", ""))
        value = _escape(item.get("value", ""))
        card_html = (
            '<div class="mlw-kv-card">'
            f'<div class="mlw-kv-label">{label}</div>'
            f'<div class="mlw-kv-value">{value}</div>'
            '</div>'
        )
        cards_html.append(card_html)

    grid_html = ''.join(cards_html)
    st.markdown(
        f'<div class="mlw-kv-grid cols-{safe_columns}">' + grid_html + '</div>',
        unsafe_allow_html=True,
    )




def render_badge_row(labels: list[str], *, variant: str = "info") -> None:
    """Render a reusable row of pill badges."""
    if not labels:
        return

    badges_html = ''.join(_build_badge_html(label=label, variant=variant) for label in labels)
    st.markdown(
        '<div class="mlw-status-row">' + badges_html + '</div>',
        unsafe_allow_html=True,
    )


def render_status_message(
    message: str,
    *,
    variant: str = "info",
    title: str | None = None,
) -> None:
    """Render one reusable app-native status message block.

    Supported variants are typically: info, success, warning, error.
    """
    st.markdown(
        _build_status_message_html(message=message, variant=variant, title=title),
        unsafe_allow_html=True,
    )



def render_placeholder_stage_panel(title: str, message: str) -> None:
    """Render a standard placeholder panel for unfinished workflow stages."""
    render_info_card(title=title, message=message)
