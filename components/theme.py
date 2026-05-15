"""
RadioAI theme — clean dark blue, single accent, no gradients.
"""

# ── Colour tokens ─────────────────────────────────────────────────────────────
BG      = "#0f172a"
CARD    = "#1e293b"
CARD2   = "#263248"
BORDER  = "#334155"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"
ACCENT  = "#3b82f6"
ACCENT2 = "#60a5fa"
SUCCESS = "#22c55e"
WARN    = "#f59e0b"
DANGER  = "#ef4444"

MODALITY_COLORS = {"CT Scan": ACCENT, "Ultrasound": ACCENT, "MRI": ACCENT}
MODALITY_ICONS  = {"CT Scan": "🫁", "Ultrasound": "🔬", "MRI": "🧠"}

# ── Global CSS ────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background: #0f172a !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}
[data-testid="stHeader"] {
    background: #0f172a !important;
    border-bottom: 1px solid #1e293b;
}
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: #0b1120 !important;
    border-right: 1px solid #1e293b;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}
p, li, span { color: #e2e8f0; }

/* Buttons */
.stButton > button {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.6rem !important;
    font-size: 0.9rem !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #2563eb !important;
    transform: scale(1.02) !important;
    box-shadow: none !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #334155 !important;
    border-radius: 10px !important;
    background: #1e293b !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #1e293b;
    border-radius: 10px;
    border: 1px solid #334155;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #3b82f6 !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stSelectbox > div > div:hover { border-color: #3b82f6 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1e293b !important;
    border-radius: 8px;
    gap: 2px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    transition: background 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #263248 !important;
    color: #e2e8f0 !important;
}
.stTabs [aria-selected="true"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    font-family: 'Inter', sans-serif !important;
}
.streamlit-expanderHeader:hover { background: #263248 !important; }

/* Divider */
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }

/* Alert */
.stAlert {
    background: #1e293b !important;
    border-left-color: #3b82f6 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Sidebar radio */
[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #94a3b8 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* Fade-in animation — only for result card */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.25s ease-out both; }
</style>
"""

# ── HTML helpers ──────────────────────────────────────────────────────────────

def card(content: str, border_left: str = ACCENT) -> str:
    return (
        f'<div style="background:{CARD};border:1px solid {BORDER};'
        f'border-left:3px solid {border_left};border-radius:10px;'
        f'padding:16px 20px;margin:12px 0;">'
        f'{content}</div>'
    )

def section_header(title: str, subtitle: str = "") -> str:
    sub = (f'<div style="color:{SUBTEXT};font-size:0.82rem;margin-top:3px;">'
           f'{subtitle}</div>') if subtitle else ""
    return (
        f'<div style="margin:24px 0 12px 0;">'
        f'<div style="font-size:1rem;font-weight:600;color:{TEXT};">{title}</div>'
        f'{sub}</div>'
    )

def warning_banner(text: str) -> str:
    return (
        f'<div style="background:#1c1a0f;border:1px solid {WARN};'
        f'border-left:3px solid {WARN};border-radius:8px;'
        f'padding:12px 16px;margin:12px 0;color:#fbbf24;font-size:0.875rem;">'
        f'⚠ {text}</div>'
    )

def danger_banner(text: str) -> str:
    return (
        f'<div style="background:#1c0f0f;border:1px solid {DANGER};'
        f'border-left:3px solid {DANGER};border-radius:8px;'
        f'padding:12px 16px;margin:12px 0;color:#fca5a5;font-size:0.875rem;">'
        f'⛔ {text}</div>'
    )

DISCLAIMER = (
    f'<div style="background:{CARD};border:1px solid {BORDER};'
    f'border-radius:8px;padding:14px 18px;margin-top:24px;'
    f'font-size:0.78rem;color:{SUBTEXT};line-height:1.6;">'
    f'<strong style="color:{TEXT};">Medical disclaimer</strong> — '
    f'RadioAI is a research tool. Results are for illustrative purposes only '
    f'and must never be used for clinical decision-making. '
    f'Always consult a qualified radiologist.'
    f'</div>'
)
