"""
About page — plain summary of the three models. No jargon.
"""

import streamlit as st
from components.theme import (card, section_header, ACCENT, CARD, BORDER,
                               TEXT, SUBTEXT, BG)

MODALITY_ICONS = {"CT Scan": "🫁", "Ultrasound": "🔬", "MRI": "🧠"}

MODELS = [
    {
        "modality": "CT Scan",
        "model":    "DPMS-LSW",
        "classes":  ["Adenocarcinoma", "Large Cell Carcinoma",
                     "Squamous Cell Carcinoma", "Normal"],
        "what_it_does": (
            "Analyses chest CT scans for four types of lung condition. "
            "It looks at a scan from two angles simultaneously — fine local texture "
            "and the broader picture across multiple slices — then combines both views "
            "to make a decision."
        ),
        "why_trustworthy": (
            "Each component was tested individually. Removing the multi-scale texture "
            "analysis dropped accuracy by 5.6%. Removing the cross-slice view dropped "
            "it by a further 2.8%. Every part of the model contributes measurably."
        ),
        "dataset": "613 training scans · 315 test scans",
    },
    {
        "modality": "Ultrasound",
        "model":    "TARNet",
        "classes":  ["Benign", "Malignant", "Normal"],
        "what_it_does": (
            "Analyses breast ultrasound images for three categories. "
            "The model was designed around the physical properties of sound waves — "
            "how they are absorbed at depth, how they scatter off different tissues, "
            "and how they cast acoustic shadows behind certain masses. "
            "These are the same cues a radiologist uses."
        ),
        "why_trustworthy": (
            "The acoustic shadow branch — which detects the dark cone behind a malignant "
            "mass — contributed most to accuracy. Removing it dropped performance by 7.7%, "
            "the largest single drop across all three models. This matches what radiologists "
            "know: the shadow is the primary diagnostic cue."
        ),
        "dataset": "546 training images · 117 test images",
    },
    {
        "modality": "MRI",
        "model":    "MSCAF",
        "classes":  ["Glioma", "Meningioma", "Pituitary Tumour", "No Tumour"],
        "what_it_does": (
            "Analyses brain MRI scans for four conditions. "
            "Different brain tumours appear at different scales — a glioma is large "
            "and irregular, a pituitary tumour is small and central. The model examines "
            "the scan at three resolutions simultaneously, then combines the results "
            "with a global view of the whole image."
        ),
        "why_trustworthy": (
            "When the multi-scale analysis was disabled and replaced with a single view, "
            "accuracy dropped by 11.6% — the largest single-component drop in this entire "
            "research project. The multi-resolution design is not decorative; it is essential."
        ),
        "dataset": "5,600 training scans · 800 test scans · perfectly balanced",
    },
]


def render_about():
    st.markdown(
        f'<div style="padding:8px 0 24px 0;">'
        f'<div style="font-size:1.4rem;font-weight:600;color:{TEXT};">'
        f'About these models</div>'
        f'<div style="font-size:0.875rem;color:{SUBTEXT};margin-top:4px;">'
        f'Three AI models built specifically for medical imaging — '
        f'each designed to be interpretable, not just accurate.'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for m in MODELS:
        icon = MODALITY_ICONS[m["modality"]]
        with st.expander(f'{icon}  {m["modality"]}  —  {m["model"]}', expanded=False):

            # Classes
            class_pills = "".join(
                f'<span style="background:#0f172a;border:1px solid {BORDER};'
                f'color:{SUBTEXT};border-radius:6px;padding:2px 10px;'
                f'font-size:0.78rem;margin:2px;display:inline-block;">{c}</span>'
                for c in m["classes"]
            )
            st.markdown(
                f'<div style="margin-bottom:16px;">'
                f'<div style="font-size:0.75rem;color:{SUBTEXT};'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">'
                f'Detects</div>'
                f'{class_pills}</div>',
                unsafe_allow_html=True,
            )

            # What it does
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown(
                    f'<div style="background:{CARD};border:1px solid {BORDER};'
                    f'border-radius:10px;padding:16px 18px;height:100%;">'
                    f'<div style="font-size:0.75rem;color:{SUBTEXT};'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">'
                    f'How it works</div>'
                    f'<div style="font-size:0.875rem;color:{TEXT};line-height:1.7;">'
                    f'{m["what_it_does"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div style="background:{CARD};border:1px solid {BORDER};'
                    f'border-radius:10px;padding:16px 18px;height:100%;">'
                    f'<div style="font-size:0.75rem;color:{SUBTEXT};'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">'
                    f'Why it can be trusted</div>'
                    f'<div style="font-size:0.875rem;color:{TEXT};line-height:1.7;">'
                    f'{m["why_trustworthy"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div style="font-size:0.78rem;color:{SUBTEXT};margin-top:10px;">'
                f'📊 {m["dataset"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{CARD};border:1px solid {BORDER};'
        f'border-radius:10px;padding:16px 20px;">'
        f'<div style="font-size:0.85rem;font-weight:600;color:{TEXT};margin-bottom:6px;">'
        f'What these models are — and are not</div>'
        f'<div style="font-size:0.85rem;color:{SUBTEXT};line-height:1.8;">'
        f'These models were built to support research into interpretable medical AI. '
        f'They are not clinical tools. All three were trained on publicly available '
        f'datasets and use trained weights that reproduce the published results. '
        f'None of them can replace a radiologist — but they were designed to show '
        f'<em>why</em> they reached a conclusion, which is the first step toward '
        f'models that a radiologist can actually verify and trust.'
        f'</div></div>',
        unsafe_allow_html=True,
    )
