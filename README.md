# RadioAI — Medical Imaging Intelligence

> *Novel architectures designed to build radiologist trust through physics-grounded, interpretable AI*

---

## Overview

RadioAI is a Streamlit-based diagnostic demonstration app built around three novel deep learning architectures for medical image classification across three imaging modalities. The design philosophy prioritises **explainability and radiologist trust** over raw accuracy.

| Modality | Architecture | Full Name | Classes |
|----------|-------------|-----------|---------|
| 🫁 CT Scan | **DPMS-LSW** | Dual-Path Multi-Scale Attention + Learnable Slice Weighting | Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal |
| 🔬 Ultrasound | **TARNet** | Tissue Acoustic Response Network | Benign, Malignant, Normal |
| 🧠 MRI | **MSCAF** | Multi-Scale CNN-Attention Fusion | Glioma, Meningioma, Pituitary, No Tumor |

---

## Features

- **Modality-specific preprocessing** — MRI uses NLM denoising + CLAHE; CT generates 8-slice pseudo-volumes; Ultrasound uses grayscale z-score normalisation
- **Activation map visualisation** — Channel-mean activation maps showing which spatial regions drove the prediction (3-panel: original | heatmap | overlay)
- **TARNet branch radar** — Polar chart showing relative contribution of each acoustic physics branch
- **Confidence gauge** — Semi-circular visual confidence meter
- **Structured clinical explanations** — "What the model saw", clinical significance, architecture insight, confidence notes
- **Radiologist trust indicators** — Checklist of explainability, auditability, and physics-alignment properties
- **Architecture deep-dives** — Full component breakdowns with ablation study findings

---

## Installation

```bash
# Clone / download the radioai/ folder, then:
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Requirements
```
streamlit>=1.32.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## Project Structure

```
radioai/
├── app.py                          ← Main entry point
├── requirements.txt
│
├── models/
│   ├── architectures.py            ← All 3 novel model implementations
│   └── loader.py                   ← Model registry, inference runner
    └── gatekeeper.py 
│
├── utils/
│   ├── preprocessing.py            ← Per-modality image preprocessing
│   ├── explainability.py           ← Activation maps, branch radar chart
│   ├── visualisations.py           ← Confidence gauge, probability bars
│   └── explanations.py             ← Clinical explanation templates (11 classes)
│
├── components/
│   ├── theme.py                    ← Global CSS, color tokens, HTML helpers
│   └── hero.py                     ← Animated landing banner (ECG, pulse rings)
│
└── pages/
    ├── analyser.py                 ← Main upload → infer → explain page
    └── about.py                    ← Architecture documentation page
```


## Architecture Highlights

### DPMS-LSW (CT)
- **Local path**: EfficientNet-B2 + MultiScaleSpatialAttention (1×1, 3×3, 5×5 parallel branches)
- **Global path**: ResNet-34 per-slice + LearnableSliceWeighting + 3-layer Transformer
- **Fusion**: CrossPathFusion with cross-attention and learned gate
- **Key ablation**: MultiScaleAttention removal → −5.56% accuracy

### TARNet (Ultrasound)
- **AttenuationBranch**: Learnable depth-decay curve modelling I(z) = I₀ × e^(−2αz)
- **BackscatterBranch**: Multi-scale echogenicity (3×3, 5×5, 7×7) with acoustic impedance weights
- **AcousticShadowBranch**: Posterior artifact detection via learnable posterior mask
- **AcousticCompositionGate**: Dynamic per-image branch routing (3-layer MLP)
- **Key ablation**: AcousticShadowBranch removal → −7.69% accuracy (largest drop in entire study)

### MSCAF (MRI)
- **Scale 1** (64ch, 224×224): Fine texture — necrosis, cell density
- **Scale 2** (128ch, 112×112): Boundaries — tumor margins, tissue patterns  
- **Scale 3** (256ch, 56×56): Morphology — overall shape, skull-relative location
- **Transformer** (3 layers, 6 heads, 384d): Global positional context from 196 patches
- **CrossAttentionModule (×3)**: CNN queries Transformer per scale
- **Key ablation**: Single-scale → −11.62% accuracy (largest single-component drop across all models)

---

## Design Philosophy

The feminine colour palette (rose pink `#e91e8c`, lavender `#9c27b0`, wine `#8b1a4a`, butter `#f5c518`) with deep plum backgrounds was chosen intentionally: medical AI tools are too often austere and intimidating. A warmer, refined aesthetic makes the tool more approachable for clinical review contexts.

---

## ⚕️ Medical Disclaimer

RadioAI is a **research demonstration tool only**. **Never use this tool for clinical decision-making.** Always consult qualified radiologists and physicians for medical diagnosis.
