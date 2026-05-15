"""
Analyser page — clean, minimal, user-friendly.
No architecture jargon. Real checkpoints only.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

from components.theme    import (card, section_header, warning_banner, danger_banner,
                                 ACCENT, CARD, BORDER, TEXT, SUBTEXT, BG)
from models.loader       import (get_gatekeeper, get_model, run_inference,
                                 MODALITY_META, checkpoint_status, missing_checkpoints)
from utils.preprocessing import preprocess, get_display_image
from utils.explainability import compute_gradcam, make_gradcam_figure, make_branch_radar
from utils.visualisations import make_confidence_bar, make_confidence_gauge
from utils.explanations   import get_explanation


# ── helpers ───────────────────────────────────────────────────────────────────

MODALITY_ICONS = {"CT Scan": "🫁", "Ultrasound": "🔬", "MRI": "🧠"}

URGENT_CLASSES = {"malignant", "glioma", "adenocarcinoma",
                  "large cell carcinoma", "squamous cell carcinoma"}


def _is_urgent(pred_class: str) -> bool:
    return pred_class.lower() in URGENT_CLASSES


def _routing_bar(gk_result: dict) -> str:
    probs     = gk_result["all_probs"]
    detected  = gk_result["modality"]
    label_map = {"ct": "CT Scan", "mri": "MRI", "ultrasound": "Ultrasound"}

    rows = ""
    for raw, p in sorted(probs.items(), key=lambda x: -x[1]):
        display = label_map[raw]
        w       = int(p * 100)
        is_top  = display == detected
        bar_col = ACCENT if is_top else "#334155"
        txt_col = TEXT   if is_top else SUBTEXT
        rows += (
            f'<div style="display:flex;align-items:center;gap:10px;margin:5px 0;">'
            f'<span style="color:{txt_col};font-size:0.8rem;width:88px;'
            f'font-weight:{"600" if is_top else "400"};">{display}</span>'
            f'<div style="flex:1;background:#1e293b;border-radius:3px;height:6px;">'
            f'<div style="width:{w}%;background:{bar_col};height:6px;border-radius:3px;"></div>'
            f'</div>'
            f'<span style="color:{txt_col};font-size:0.8rem;width:34px;text-align:right;">'
            f'{p:.0%}</span>'
            f'</div>'
        )

    icon = MODALITY_ICONS[detected]
    return (
        f'<div style="background:{CARD};border:1px solid {BORDER};'
        f'border-left:3px solid {ACCENT};border-radius:10px;'
        f'padding:14px 18px;margin:12px 0;">'
        f'<div style="font-size:0.75rem;color:{SUBTEXT};'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">'
        f'Scan type detected</div>'
        f'<div style="font-size:1rem;font-weight:600;color:{TEXT};margin-bottom:10px;">'
        f'{icon} {detected}</div>'
        f'{rows}'
        f'</div>'
    )


def _no_gatekeeper_note() -> str:
    return (
        f'<div style="background:{CARD};border:1px solid #f59e0b;'
        f'border-radius:8px;padding:12px 16px;margin:12px 0;'
        f'color:#fbbf24;font-size:0.85rem;">'
        f'⚠ <strong>gatekeeper.pth</strong> not found — '
        f'select scan type manually below.'
        f'</div>'
    )


# ── main ──────────────────────────────────────────────────────────────────────

def render_analyser():
    # ── Checkpoint guard ────────────────────────────────────────────────────
    missing = missing_checkpoints()
    if missing:
        st.markdown(
            f'<div style="background:{CARD};border:1px solid #ef4444;'
            f'border-left:3px solid #ef4444;border-radius:10px;'
            f'padding:16px 20px;margin:0 0 20px 0;">'
            f'<div style="font-size:0.95rem;font-weight:600;color:#fca5a5;'
            f'margin-bottom:8px;">Checkpoints required</div>'
            f'<div style="font-size:0.85rem;color:#94a3b8;line-height:1.7;">'
            f'Place these files in <code style="color:#e2e8f0;background:#0f172a;'
            f'padding:1px 6px;border-radius:4px;">radioai/checkpoints/</code>'
            f' to use the app:<br>'
            + "".join(
                f'<span style="color:#fca5a5;">✗ {k}</span>'
                f'<span style="color:#64748b;"> → '
                f'<code style="color:#94a3b8;background:#0f172a;padding:1px 6px;'
                f'border-radius:3px;">{_ckpt_filename(k)}</code></span><br>'
                for k in missing
            )
            + f'</div></div>',
            unsafe_allow_html=True,
        )
        if any(k != "gatekeeper" for k in missing):
            return   # can't run inference without model checkpoints

    # ── Upload ───────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a scan — CT, MRI, or Ultrasound (PNG · JPG · TIFF)",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        key="main_upload",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div style="font-size:0.82rem;color:{SUBTEXT};margin-bottom:20px;">'
        f'Upload a scan — CT, MRI, or Ultrasound</div>',
        unsafe_allow_html=True,
    )

    if uploaded is None:
        st.markdown(
            f'<div style="text-align:center;padding:40px 0;color:{SUBTEXT};">'
            f'<div style="font-size:2.5rem;margin-bottom:10px;opacity:0.4;">📂</div>'
            f'<div style="font-size:0.9rem;">Drop a scan file above to begin</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    image_bytes = uploaded.read()
    display_img = get_display_image(image_bytes)

    # ── Routing ──────────────────────────────────────────────────────────────
    router   = get_gatekeeper()
    modality = None

    if router is not None:
        with st.spinner("Identifying scan type…"):
            gk = router.route_bytes(image_bytes)
        modality = gk["modality"]
        st.markdown(_routing_bar(gk), unsafe_allow_html=True)

        with st.expander("Override scan type"):
            override = st.selectbox(
                "Scan type",
                ["— keep auto-detected —"] + list(MODALITY_META.keys()),
                key="override_sel",
                label_visibility="collapsed",
            )
            if override != "— keep auto-detected —":
                modality = override
    else:
        st.markdown(_no_gatekeeper_note(), unsafe_allow_html=True)
        modality = st.selectbox(
            "Select scan type",
            list(MODALITY_META.keys()),
            key="manual_mod",
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div style="font-size:0.82rem;color:{SUBTEXT};margin-bottom:4px;">'
            f'Select scan type</div>',
            unsafe_allow_html=True,
        )

    if modality is None:
        return

    # ── Scan preview + run ───────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_img, col_run = st.columns([1, 1], gap="medium")

    with col_img:
        st.image(
            Image.fromarray(display_img).convert("RGB"),
            caption=uploaded.name,
            use_container_width=True,
        )

    with col_run:
        st.markdown(
            f'<div style="font-size:0.82rem;color:{SUBTEXT};margin-bottom:16px;">'
            f'{MODALITY_ICONS.get(modality,"")} '
            f'<strong style="color:{TEXT};">{modality}</strong> — '
            f'ready to analyse</div>',
            unsafe_allow_html=True,
        )
        run = st.button("Analyse scan", use_container_width=True,
                        type="primary", key="run_btn")

    if run:
        with st.spinner("Analysing…"):
            try:
                tensor = preprocess(image_bytes, modality)
                model  = get_model(modality)          # raises if .pth missing
                result = run_inference(model, tensor)
                cam    = compute_gradcam(model, tensor, result["pred_idx"])
                st.session_state.update({
                    "result":   result,
                    "cam":      cam,
                    "disp_img": display_img,
                    "mod_used": modality,
                })
            except FileNotFoundError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Analysis error: {e}")
                return

    # ── Results ──────────────────────────────────────────────────────────────
    if "result" not in st.session_state:
        return

    result   = st.session_state["result"]
    cam      = st.session_state["cam"]
    disp_img = st.session_state["disp_img"]
    mod_used = st.session_state["mod_used"]

    if mod_used != modality:
        st.info("Showing results from previous run. Click Analyse again to refresh.")

    pred    = result["pred_class"]
    conf    = result["confidence"]
    probs   = result["probs"]
    classes = MODALITY_META[mod_used]["classes"]

    _render_results(pred, conf, probs, classes, cam, disp_img, mod_used,
                    result.get("branch_weights"))


# ── results layout ────────────────────────────────────────────────────────────

def _render_results(pred, conf, probs, classes, cam, disp_img, modality, branch_w):
    try:
        exp = get_explanation(modality, pred)
    except KeyError:
        exp = None

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(f'<hr style="border-color:{BORDER};margin:0 0 24px 0;">',
                unsafe_allow_html=True)

    # ── Result card ───────────────────────────────────────────────────────────
    urgent     = _is_urgent(pred)
    card_border = "#ef4444" if urgent else ACCENT
    conf_color  = ("#22c55e" if conf >= 0.70
                   else "#f59e0b" if conf >= 0.50 else "#ef4444")

    st.markdown(
        f'<div class="fade-up" style="background:{CARD};border:1px solid {BORDER};'
        f'border-left:4px solid {card_border};border-radius:10px;'
        f'padding:20px 24px;margin-bottom:20px;">'

        # title
        f'<div style="font-size:0.75rem;color:{SUBTEXT};text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:6px;">{modality} result</div>'
        f'<div style="font-size:1.6rem;font-weight:600;color:{TEXT};'
        f'margin-bottom:4px;">'
        f'{exp.title if exp else pred}'
        f'</div>'

        # confidence pill
        f'<div style="margin-bottom:14px;">'
        f'<span style="background:{conf_color}20;border:1px solid {conf_color};'
        f'color:{conf_color};border-radius:20px;padding:2px 12px;'
        f'font-size:0.82rem;font-weight:600;">{conf:.0%} confidence</span>'
        f'</div>'

        # summary
        + (f'<div style="font-size:0.9rem;color:{SUBTEXT};'
           f'line-height:1.6;margin-bottom:14px;">'
           f'{exp.summary}</div>' if exp else "")

        # key findings
        + (f'<div style="font-size:0.85rem;color:{TEXT};line-height:1.8;">'
           + "".join(
               f'<div style="display:flex;gap:8px;margin:2px 0;">'
               f'<span style="color:{ACCENT};margin-top:2px;">›</span>'
               f'<span>{f}</span></div>'
               for f in exp.key_findings
           )
           + f'</div>' if exp else "")

        + f'</div>',
        unsafe_allow_html=True,
    )

    # urgent warning
    if urgent:
        st.markdown(danger_banner(
            "These findings need prompt clinical review. "
            "This is a screening tool — not a diagnosis."
        ), unsafe_allow_html=True)

    # ── Two columns: meaning + next step / confidence visual ─────────────────
    if exp:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            st.markdown(
                f'<div style="background:{CARD};border:1px solid {BORDER};'
                f'border-radius:10px;padding:16px 20px;margin-bottom:12px;">'
                f'<div style="font-size:0.75rem;color:{SUBTEXT};'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">'
                f'What this means</div>'
                f'<div style="font-size:0.875rem;color:{TEXT};line-height:1.7;">'
                f'{exp.meaning}</div>'
                f'</div>'
                f'<div style="background:{CARD};border:1px solid {BORDER};'
                f'border-radius:10px;padding:16px 20px;">'
                f'<div style="font-size:0.75rem;color:{SUBTEXT};'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">'
                f'Suggested next step</div>'
                f'<div style="font-size:0.875rem;color:{TEXT};line-height:1.7;">'
                f'→ {exp.next_step}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            gauge = make_confidence_gauge(conf, pred)
            st.image(gauge, use_container_width=True)
            st.markdown(
                f'<div style="font-size:0.78rem;color:{SUBTEXT};'
                f'text-align:center;margin-top:4px;">'
                f'{exp.confidence_note}</div>',
                unsafe_allow_html=True,
            )

    # ── Expandable: probabilities + activation map ────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.expander("View detailed scores"):
        d1, d2 = st.columns([1.4, 1], gap="medium")
        with d1:
            st.markdown(
                f'<div style="font-size:0.8rem;color:{SUBTEXT};margin-bottom:8px;">'
                f'All class probabilities</div>',
                unsafe_allow_html=True,
            )
            st.image(make_confidence_bar(classes, probs,
                     classes.index(pred) if pred in classes else 0),
                     use_container_width=True)
        with d2:
            if modality == "Ultrasound" and branch_w:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:{SUBTEXT};margin-bottom:8px;">'
                    f'Signal branch weights</div>',
                    unsafe_allow_html=True,
                )
                st.image(make_branch_radar(branch_w), use_container_width=True)

    with st.expander("View activation map"):
        if cam is not None:
            fig = make_gradcam_figure(disp_img, cam)
            st.image(fig, use_container_width=True)
            st.markdown(
                f'<div style="font-size:0.78rem;color:{SUBTEXT};margin-top:6px;">'
                f'Highlighted areas show which parts of the scan influenced the result most.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="color:{SUBTEXT};font-size:0.85rem;">'
                f'Activation map unavailable for this image.</div>',
                unsafe_allow_html=True,
            )


# ── helpers ───────────────────────────────────────────────────────────────────

def _ckpt_filename(key: str) -> str:
    return {
        "gatekeeper": "gatekeeper.pth",
        "CT Scan":    "dpms_lsw.pth",
        "Ultrasound": "tarnet.pth",
        "MRI":        "mscaf.pth",
    }.get(key, key)
