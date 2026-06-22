from pathlib import Path
import streamlit as st


def load_css():
    css_path = Path(__file__).resolve().parents[2] / "assets" / "CSS" / "styles.css"

    if not css_path.exists():
        st.error(f"CSS file not found: {css_path}")
        return

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )