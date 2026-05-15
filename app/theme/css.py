# ─────────────────────────────────────────────
# THEME DEFINITIONS
# ─────────────────────────────────────────────
# Centralized theme tokens for the full app shell. The values are intentionally
# similar in spirit to your professor's shell, but expanded slightly so the same
# structure can support a broader set of app components later, including chat,
# tool traces, status badges, and agent-related UI.
THEMES = {
    "dark": dict(
        bg="#0b0f19",
        card="#111827",
        card_hover="#1a2235",
        surface="#161e2e",
        border="rgba(56,189,248,0.08)",
        border_glow="rgba(56,189,248,0.25)",
        primary="#38bdf8",
        primary_dim="rgba(56,189,248,0.6)",
        amber="#fbbf24",
        emerald="#34d399",
        rose="#fb7185",
        violet="#a78bfa",
        slate="#94a3b8",
        text="#f1f5f9",
        muted="#64748b",
        grid="rgba(148,163,184,0.06)",
        line="rgba(148,163,184,0.1)",
        sidebar_bg="linear-gradient(180deg, #0d1321 0%, #111827 100%)",
        card_shadow="rgba(56,189,248,0.08)",
        tab_active_shadow="rgba(56,189,248,0.08)",
        insight_bg_alpha="0.04",
        insight_bg_alpha2="0.01",
        insight_border_alpha="0.1",
        pie_line="#0b0f19",
        bar_text="#f1f5f9",
        bar_text_alt="#94a3b8",
        heatmap_scale=["#0b0f19", "#1e3a5f", "#38bdf8", "#e0f2fe"],
        matrix_scale=["#0b0f19", "#1e1b4b", "#a78bfa", "#e9d5ff"],
        footer_color="rgba(148,163,184,0.35)",
        stat_border="rgba(148,163,184,0.06)",
        input_bg="#0f172a",
        chat_user_bg="rgba(56,189,248,0.12)",
        chat_assistant_bg="rgba(148,163,184,0.08)",
        chat_border="rgba(148,163,184,0.10)",
        success_bg="rgba(52,211,153,0.12)",
        warning_bg="rgba(251,191,36,0.12)",
        error_bg="rgba(251,113,133,0.12)",
        info_bg="rgba(56,189,248,0.10)",
    ),
    "light": dict(
        bg="#f8fafc",
        card="#ffffff",
        card_hover="#f1f5f9",
        surface="#eef2f7",
        border="rgba(15,23,42,0.08)",
        border_glow="rgba(14,116,144,0.2)",
        primary="#0891b2",
        primary_dim="rgba(8,145,178,0.6)",
        amber="#d97706",
        emerald="#059669",
        rose="#e11d48",
        violet="#7c3aed",
        slate="#475569",
        text="#0f172a",
        muted="#64748b",
        grid="rgba(15,23,42,0.05)",
        line="rgba(15,23,42,0.08)",
        sidebar_bg="linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%)",
        card_shadow="rgba(0,0,0,0.06)",
        tab_active_shadow="rgba(14,116,144,0.1)",
        insight_bg_alpha="0.06",
        insight_bg_alpha2="0.02",
        insight_border_alpha="0.15",
        pie_line="#ffffff",
        bar_text="#0f172a",
        bar_text_alt="#475569",
        heatmap_scale=["#f0fdfa", "#99f6e4", "#0891b2", "#164e63"],
        matrix_scale=["#faf5ff", "#ddd6fe", "#7c3aed", "#4c1d95"],
        footer_color="rgba(71,85,105,0.4)",
        stat_border="rgba(15,23,42,0.06)",
        input_bg="#ffffff",
        chat_user_bg="rgba(8,145,178,0.10)",
        chat_assistant_bg="rgba(71,85,105,0.06)",
        chat_border="rgba(15,23,42,0.08)",
        success_bg="rgba(5,150,105,0.10)",
        warning_bg="rgba(217,119,6,0.10)",
        error_bg="rgba(225,29,72,0.10)",
        info_bg="rgba(8,145,178,0.08)",
    ),
}


# ─────────────────────────────────────────────
# COLOR UTILITIES
# ─────────────────────────────────────────────
# Helper functions for converting and reusing colors across charts, badges,
# and custom HTML components.

def _rgb(hex_color: str) -> str:
    """Convert #hex to an r,g,b string for rgba() usage."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# ─────────────────────────────────────────────
# DYNAMIC CSS
# ─────────────────────────────────────────────
# Generates the global CSS injected into Streamlit. This styles both native
# Streamlit components and custom HTML blocks so the app feels cohesive.

def build_css(t: dict, dark: bool) -> str:
    card_hover_shadow = (
        f"0 8px 30px {t['card_shadow']}" if dark else f"0 4px 20px {t['card_shadow']}"
    )

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&family=Playfair+Display:wght@600;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
    --bg: {t['bg']};
    --card: {t['card']};
    --card-hover: {t['card_hover']};
    --surface: {t['surface']};
    --primary: {t['primary']};
    --primary-dim: {t['primary_dim']};
    --border: {t['border']};
    --border-glow: {t['border_glow']};
    --amber: {t['amber']};
    --emerald: {t['emerald']};
    --rose: {t['rose']};
    --violet: {t['violet']};
    --slate: {t['slate']};
    --text: {t['text']};
    --muted: {t['muted']};
    --input-bg: {t['input_bg']};
    --chat-user-bg: {t['chat_user_bg']};
    --chat-assistant-bg: {t['chat_assistant_bg']};
    --chat-border: {t['chat_border']};
    --success-bg: {t['success_bg']};
    --warning-bg: {t['warning_bg']};
    --error-bg: {t['error_bg']};
    --info-bg: {t['info_bg']};
    --font-body: 'DM Sans', sans-serif;
    --font-display: 'Playfair Display', serif;
    --font-mono: 'JetBrains Mono', monospace;
}}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {{
    background-color: var(--bg) !important;
    color: var(--slate) !important;
    font-family: var(--font-body) !important;
}}

#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 2000px !important;
}}
div[data-testid="stMetric"] {{ display: none; }}


/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background: {t['sidebar_bg']} !important;
    border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"] * {{
    color: var(--slate) !important;
}}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {{
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em;
}}
section[data-testid="stSidebar"] hr {{
    border-color: var(--border) !important;
}}


/* METRIC CARDS */
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-bottom: 2rem;
}}
@media (max-width: 900px) {{
    .metric-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}
.m-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    {"" if dark else "box-shadow: 0 1px 4px rgba(0,0,0,0.04);"}
}}
.m-card:hover {{
    border-color: var(--border-glow);
    background: var(--card-hover);
    transform: translateY(-3px);
    box-shadow: {card_hover_shadow};
}}
.m-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}}
.m-card:nth-child(1)::before {{ background: var(--primary); }}
.m-card:nth-child(2)::before {{ background: var(--emerald); }}
.m-card:nth-child(3)::before {{ background: var(--violet); }}
.m-card:nth-child(4)::before {{ background: var(--amber); }}
.m-card:nth-child(5)::before {{ background: var(--rose); }}
.m-card:nth-child(6)::before {{ background: var(--slate); }}
.m-icon {{
    font-size: 1.4rem;
    margin-bottom: 0.3rem;
}}
.m-val {{
    font-family: var(--font-mono);
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
}}
.m-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--slate);
    margin-top: 0.35rem;
    font-weight: 500;
}}


/* GENERIC PANELS */
.app-panel {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}}
.panel-title {{
    font-family: var(--font-display);
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.35rem;
}}
.panel-subtitle {{
    color: var(--slate);
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}}


/* TABS */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: var(--surface);
    border-radius: 14px;
    padding: 5px;
    border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: var(--slate) !important;
    background: transparent !important;
    border: none !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: var(--card) !important;
    color: var(--primary) !important;
    box-shadow: 0 2px 12px {t['tab_active_shadow']};
    font-weight: 700 !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

/* INPUTS */
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div,
.stTextInput input,
.stTextArea textarea {{
    background: var(--input-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {{
    color: var(--muted) !important;
}}

.stTextInput label,
.stTextArea label {{
    color: var(--text) !important;
}}

[data-testid="stForm"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}

/* BUTTONS */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {{
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    min-height: 2.7rem !important;
}}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
    border-color: var(--border-glow) !important;
    transform: translateY(-1px);
    box-shadow: {card_hover_shadow};
}}

.stButton > button:disabled,
[data-testid="stFormSubmitButton"] > button:disabled {{
    background: var(--surface) !important;
    border-color: var(--border-glow) !important;
    color: var(--text) !important;
    opacity: 1 !important;
    box-shadow:
        inset 0 1px 0 rgba({_rgb(t['text'])}, 0.04),
        0 2px 10px rgba({_rgb(t['text'])}, 0.06) !important;
    cursor: default !important;
}}

.stButton > button:disabled:hover,
[data-testid="stFormSubmitButton"] > button:disabled:hover {{
    transform: none !important;
    box-shadow:
        inset 0 1px 0 rgba({_rgb(t['text'])}, 0.04),
        0 2px 10px rgba({_rgb(t['text'])}, 0.06) !important;
}}


/* CHAT PANEL */
.chat-shell {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-top: 1rem;
    margin-bottom: 1rem;
    height: 70vh;
    min-height: 520px;
    max-height: 760px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}}

.chat-messages {{
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}}

.chat-input-shell {{
    border-top: 1px solid var(--border);
    background: var(--card);
    padding: 0.9rem 1rem 1rem 1rem;
    position: sticky;
    bottom: 0;
    z-index: 5;
}}

.chat-row {{
    display: flex;
    width: 100%;
    margin-bottom: 0.85rem;
}}

.chat-row.left {{
    justify-content: flex-start;
}}

.chat-row.right {{
    justify-content: flex-end;
}}

.chat-message {{
    border: 1px solid var(--chat-border);
    border-radius: 16px;
    padding: 0.95rem 1rem;
    max-width: 72%;
    min-width: 220px;
}}

.chat-message.user {{
    background: var(--chat-user-bg);
}}

.chat-message.assistant {{
    background: var(--chat-assistant-bg);
}}

.chat-role {{
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 0.4rem;
}}

.chat-body {{
    color: var(--text);
    line-height: 1.65;
    font-size: 0.96rem;
}}

.assistant-pane-wrap {{
    margin-top: 1.1rem;
    margin-right: 0.6rem;
}}




/* STATUS / TOOL BADGES */
.status-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0.75rem 0 1rem 0;
}}
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0.45rem 0.7rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    font-size: 0.78rem;
    font-weight: 600;
}}

.badge.success {{ background: var(--success-bg); }}
.badge.warning {{ background: var(--warning-bg); }}
.badge.error {{ background: var(--error-bg); }}
.badge.info {{ background: var(--info-bg); }}

.badge-tooltip-wrap {{
    position: relative;
    display: inline-flex;
    align-items: center;
}}

.badge-tooltip {{
    position: absolute;
    left: 50%;
    bottom: calc(100% + 10px);
    transform: translateX(-50%) translateY(4px);
    min-width: 220px;
    max-width: 320px;
    padding: 0.7rem 0.85rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text);
    font-size: 0.8rem;
    font-weight: 500;
    line-height: 1.5;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transition: opacity 0.18s ease, transform 0.18s ease, visibility 0.18s ease;
    z-index: 1000;
    text-align: left;
    white-space: pre-line;
}}

.badge-tooltip::after {{
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-width: 6px;
    border-style: solid;
    border-color: var(--card) transparent transparent transparent;
}}

.badge-tooltip-wrap:hover .badge-tooltip {{
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(0);
}}

.badge-tooltip-wrap:hover .badge {{
    background: var(--card-hover);
    border-color: var(--border-glow);
    box-shadow: 0 6px 16px rgba(0,0,0,0.14);
}}

.trace-stack .badge {{
    margin-bottom: 0.45rem;
}}

.trace-stack {{
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 0.5rem;
}}

.trace-section {{
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--card);
    overflow: hidden;
}}

.trace-section[open] {{
    border-color: var(--border-glow);
    background: var(--card-hover);
}}

.trace-section-summary {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 0.8rem 0.9rem;
    cursor: pointer;
    list-style: none;
    color: var(--text);
}}

.trace-section-summary::-webkit-details-marker {{
    display: none;
}}

.trace-section-summary::after {{
    content: "▸";
    color: var(--primary);
    font-family: var(--font-mono);
    font-size: 0.82rem;
    transition: transform 0.16s ease;
}}

.trace-section[open] > .trace-section-summary::after {{
    transform: rotate(90deg);
}}

.trace-section-title {{
    font-weight: 700;
    color: var(--text);
    min-width: 92px;
}}

.trace-section-meta {{
    display: flex;
    flex: 1;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}}

.trace-section-meta .badge {{
    margin-bottom: 0;
}}

.trace-section-body {{
    padding: 0.15rem 0.9rem 0.95rem 0.9rem;
}}

.trace-item-header {{
    margin: 0.35rem 0 0.55rem 0;
    color: var(--slate);
    font-family: var(--font-mono);
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

.trace-item-header.success {{
    color: var(--emerald);
}}

.trace-item-header.error {{
    color: var(--rose);
}}

.trace-kv-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 0.8rem;
    margin: 0 0 0.65rem 0;
    color: var(--text);
    font-size: 0.84rem;
    line-height: 1.45;
}}

.trace-kv {{
    display: inline-flex;
    align-items: baseline;
    gap: 0.32rem;
    min-width: 0;
}}

.trace-kv-label {{
    color: var(--slate);
    font-weight: 600;
}}

.trace-kv-label::after {{
    content: ":";
}}

.trace-kv-value {{
    color: var(--text);
    overflow-wrap: anywhere;
}}

.trace-plain-text-panel {{
    max-width: 100%;
    margin: 0 0 0.75rem 0;
    padding: 0.7rem 0.8rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--surface);
}}

.trace-plain-text {{
    color: var(--text);
    font-size: 0.84rem;
    line-height: 1.45;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}}

.trace-markdown-panel {{
    max-width: 100%;
    margin: 0 0 0.75rem 0;
    padding: 0.75rem 0.85rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface);
}}

.trace-markdown {{
    color: var(--text);
    font-size: 0.86rem;
    line-height: 1.5;
    word-break: break-word;
}}

.trace-markdown p {{
    margin: 0 0 0.55rem 0;
}}

.trace-markdown p:last-child {{
    margin-bottom: 0;
}}

.trace-markdown ul,
.trace-markdown ol {{
    margin: 0.15rem 0 0.65rem 1.15rem;
    padding: 0;
}}

.trace-markdown li {{
    margin: 0 0 0.25rem 0;
}}

.trace-markdown strong {{
    font-weight: 700;
}}

.trace-markdown em {{
    font-style: italic;
}}

.trace-markdown code {{
    font-family: var(--font-mono);
    font-size: 0.88em;
    padding: 0.08rem 0.3rem;
    border-radius: 7px;
    background: var(--card);
    border: 1px solid var(--border);
}}

.trace-markdown pre {{
    margin: 0.15rem 0 0.7rem 0;
    padding: 0.7rem 0.8rem;
    border-radius: 10px;
    background: var(--card);
    border: 1px solid var(--border);
    overflow-x: auto;
}}

.trace-markdown pre code {{
    display: block;
    padding: 0;
    border: 0;
    background: transparent;
    font-size: 0.8rem;
    line-height: 1.45;
    white-space: pre;
}}

.trace-markdown-heading {{
    margin: 0.15rem 0 0.5rem 0;
    color: var(--text);
    font-weight: 700;
    line-height: 1.25;
}}

.trace-markdown-heading.heading-1 {{
    font-size: 1rem;
}}

.trace-markdown-heading.heading-2 {{
    font-size: 0.96rem;
}}

.trace-markdown-heading.heading-3,
.trace-markdown-heading.heading-4 {{
    font-size: 0.9rem;
}}

.trace-markdown-table-wrapper {{
    width: 100%;
    overflow-x: auto;
    margin: 0.15rem 0 0.7rem 0;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--card);
}}

.trace-markdown-table-wrapper table {{
    width: 100%;
    min-width: 420px;
    border-collapse: collapse;
    font-size: 0.82rem;
}}

.trace-markdown-table-wrapper th,
.trace-markdown-table-wrapper td {{
    padding: 0.5rem 0.6rem;
    text-align: left;
    vertical-align: top;
    border-bottom: 1px solid var(--border);
}}

.trace-markdown-table-wrapper th {{
    color: var(--text);
    font-weight: 700;
    white-space: nowrap;
}}

.trace-markdown-table-wrapper tr:last-child td {{
    border-bottom: 0;
}}

.code-panel {{
    display: block;
    max-width: 100%;
    margin: 0 0 0.75rem 0;
}}

.code-panel-text {{
    display: inline-block;
    max-width: 100%;
    padding: 0.7rem 0.85rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 0.8rem;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: auto;
}}

.section-divider {{
    width: 100%;
    height: 1px;
    margin: 1rem 0 1.15rem 0;
    background: linear-gradient(
        90deg,
        transparent 0%,
        var(--border) 8%,
        var(--border) 92%,
        transparent 100%
    );
}}



/* EXPANDERS */
details {{
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--card);
}}


/* THEME TOGGLE LABEL */
.theme-toggle {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 0;
    margin-bottom: 0.8rem;
}}
.theme-toggle-label {{
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--primary);
}}


/* SCROLLBAR */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--muted); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--slate); }}
</style>
"""
