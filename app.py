"""
RadioAI — Main Streamlit Application
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from components.theme import GLOBAL_CSS, DISCLAIMER, CARD, BORDER, TEXT, SUBTEXT, ACCENT
from components.hero  import HERO_HTML, UPLOAD_ANIMATION_CSS
from pages.analyser   import render_analyser
from pages.about      import render_about

st.set_page_config(
    page_title="RadioAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS,           unsafe_allow_html=True)
st.markdown(UPLOAD_ANIMATION_CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:20px 4px 20px 4px;">'
        f'<div style="font-family:Inter,sans-serif;font-size:1.4rem;'
        f'font-weight:600;color:{TEXT};letter-spacing:-0.02em;">'
        f'Radio<span style="color:{ACCENT};">AI</span>'
        f'</div>'
        f'<div style="color:{SUBTEXT};font-size:0.75rem;margin-top:3px;">'
        f'Medical Imaging · Research Demo'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<hr style="border-color:{BORDER};margin:0 0 12px 0;">',
                unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🩺 Analyse Scan", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown(f'<hr style="border-color:{BORDER};margin:12px 0;">',
                unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.72rem;color:{SUBTEXT};line-height:1.7;">'
        f'Not for clinical use.<br>'
        f'Research demonstration only.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Pages ─────────────────────────────────────────────────────────────────────
if page == "🩺 Analyse Scan":
    st.markdown(HERO_HTML, unsafe_allow_html=True)
    render_analyser()
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
else:
    render_about()
