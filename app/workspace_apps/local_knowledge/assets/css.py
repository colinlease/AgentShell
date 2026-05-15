from __future__ import annotations

import streamlit as st


def inject_local_knowledge_css() -> None:
    st.markdown(
        """
        <style>
        .lkw-hero {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.4rem 1.45rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .lkw-eyebrow {
            color: var(--primary);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .lkw-title {
            color: var(--text);
            font-family: var(--font-display);
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.45rem;
        }
        .lkw-subtitle {
            color: var(--slate);
            font-size: 0.98rem;
            line-height: 1.6;
            max-width: 760px;
        }
        .lkw-panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.15rem;
            margin-bottom: 1rem;
        }
        .lkw-panel-title {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .lkw-panel-subtitle {
            color: var(--slate);
            font-size: 0.9rem;
            line-height: 1.55;
            margin-bottom: 0.85rem;
        }
        .lkw-folder-copy {
            margin: 0 0 0.75rem 0;
        }
        .lkw-folder-title {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .lkw-folder-subtitle {
            color: var(--slate);
            font-size: 0.9rem;
            line-height: 1.55;
            max-width: 760px;
        }
        .lkw-recent-label {
            color: var(--slate);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0.7rem 0 0.25rem 0.05rem;
        }
        .lkw-metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.85rem;
            margin-bottom: 1rem;
        }
        .lkw-metric-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
        }
        .lkw-metric-value {
            color: var(--text);
            font-family: var(--font-mono);
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .lkw-metric-label {
            color: var(--slate);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .lkw-status {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--surface);
            color: var(--text);
            padding: 0.75rem 0.85rem;
            font-size: 0.88rem;
            line-height: 1.45;
            margin: 0.85rem 0 1.1rem 0;
        }
        .lkw-status.error {
            background: var(--error-bg);
        }
        .lkw-status.success {
            background: var(--success-bg);
        }
        .lkw-status.info {
            background: var(--info-bg);
        }
        .lkw-mini-status {
            color: var(--slate);
            font-size: 0.82rem;
            line-height: 1.45;
            margin: -0.35rem 0 1rem 0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
