import streamlit as st

from app.theme.css import THEMES, build_css


def get_theme_tokens(theme_name: str = "light") -> tuple[dict, bool]:
    """
    Return the selected theme token dictionary and whether it is a dark theme.
    """
    selected = theme_name if theme_name in THEMES else "light"
    tokens = THEMES[selected]
    is_dark = selected == "dark"
    return tokens, is_dark


def load_theme(theme_name: str = "light") -> None:
    """
    Build and inject the CSS for the selected theme into the Streamlit app.
    """
    tokens, is_dark = get_theme_tokens(theme_name)
    st.markdown(build_css(tokens, is_dark), unsafe_allow_html=True)
