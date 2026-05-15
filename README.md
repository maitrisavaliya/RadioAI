# 🩺 RadioAI — Medical Imaging Intelligence

> *Novel architectures designed to build radiologist trust through physics-grounded, interpretable AI*

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/streamlit-1.32.0-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.0-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## 🚀 Quick Start

Get RadioAI running in 3 steps:

### Step 1️⃣ Clone the Repository
```bash
git clone https://github.com/maitrisavaliya/RadioAI.git
cd RadioAI
```

### Step 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3️⃣ Launch the App
```bash
streamlit run app.py
```

✅ Done! Your app opens at `http://localhost:8501`

---

## 📊 Overview

RadioAI is a **Streamlit-based diagnostic demonstration** app built around three novel deep learning architectures for medical image classification. The design philosophy prioritises **explainability and radiologist trust** over raw accuracy.

### 🎯 Supported Imaging Modalities

| 🫁 CT Scan | 🔬 Ultrasound | 🧠 MRI |
|:---:|:---:|:---:|
| **DPMS-LSW** | **TARNet** | **MSCAF** |
| Dual-Path Multi-Scale Attention | Tissue Acoustic Response | Multi-Scale CNN-Attention |
| 4 Classes | 3 Classes | 4 Classes |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Multi-Modality Support** | CT, Ultrasound, and MRI with modality-specific preprocessing |
| 🎨 **Activation Visualisation** | See what regions drove each prediction with 3-panel heatmaps |
| 📡 **Branch Radar Charts** | Understand model reasoning via acoustic physics branches (TARNet) |
| 🎯 **Confidence Metrics** | Semi-circular gauges showing model confidence levels |
| 📝 **Clinical Explanations** | Structured breakdowns: "What the model saw", significance, architecture insights |
| ✅ **Trust Indicators** | Explainability checklist for radiologist validation |
| 🔬 **Architecture Deep-Dives** | Full component documentation with ablation study results |

---

## 🛠️ Installation & Setup

### Requirements
```
✓ Python 3.9+
✓ PyTorch 2.0+
✓ Streamlit 1.32+
✓ CUDA (optional, for GPU acceleration)
```

### Full Installation Guide

#### Option A: Local Development
```bash
# Clone repo
git clone https://github.com/maitrisavaliya/RadioAI.git
cd RadioAI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

#### Option B: Deploy to Streamlit Cloud ☁️
```bash
# Push to GitHub (already set up!)
git add .
git commit -m "Your changes"
git push origin main

# Go to https://share.streamlit.io
# Connect your GitHub repo → automatic deployment! 🚀
```

---

## 📚 Tutorial: Your First Prediction

### Step 1: Upload an Image

1. Open the app (`http://localhost:8501`)
2. Click **"📤 Upload Medical Image"**
3. Select a medical image (JPEG, PNG, DICOM)
4. Choose the imaging modality:
   - 🫁 **CT Scan** — For chest/lung images
   - 🔬 **Ultrasound** — For tissue/organ images  
   - 🧠 **MRI** — For brain/tissue images

### Step 2: Run Inference

Click **"🔍 Analyze Image"** and watch the model process it:

```
✓ Image loaded
✓ Preprocessing complete (modality-specific normalization)
✓ Model inference (~2-3 seconds)
✓ Generating explanations
```

### Step 3: Interpret Results

You'll see:
- **Prediction** — Top 3 likely diagnoses with probabilities
- **Confidence Gauge** — Visual confidence meter (0-100%)
- **Activation Map** — Where the model focused:
  - Left: Original image
  - Center: Heatmap (red = high attention)
  - Right: Overlay for comparison
- **Clinical Explanation** — Structured text breakdown
- **Model Confidence Notes** — What made the model confident (or uncertain)

---

## 🏗️ Project Structure

```
RadioAI/
│
├── 📄 app.py                    ← Main entry point (Streamlit app)
├── 📋 requirements.txt          ← Python dependencies
├── 📦 checkpoints/              ← Pre-trained model weights
│   ├── dpms_lsw.pth            (CT model - 157 MB)
│   ├── tarnet.pth              (Ultrasound model - 37 MB)
│   ├── mscaf.pth               (MRI model - 55 MB)
│   └── gatekeeper.pth          (Confidence gating model)
│
├── 🤖 models/
│   ├── architectures.py         ← All 3 novel architectures
│   ├── loader.py                ← Model registry & inference
│   └── gatekeeper.py            ← Confidence validation model
│
├── 🛠️ utils/
│   ├── preprocessing.py         ← Image preprocessing (per-modality)
│   ├── explainability.py        ← Activation map generation
│   ├── visualisations.py        ← Confidence gauges, charts
│   └── explanations.py          ← Clinical explanation templates
│
├── 🎨 components/
│   ├── theme.py                 ← Global CSS & colors
│   └── hero.py                  ← Landing animation
│
└── 📄 pages/
    ├── analyser.py              ← Main upload → infer page
    └── about.py                 ← Architecture documentation
```

---

## 🧠 Architecture Highlights

### 🫁 DPMS-LSW (CT Scans)

**Problem**: CT slices need both local details AND global context

**Solution**: Dual-path fusion
- 🔵 **Local Path** — EfficientNet-B2 + MultiScale Spatial Attention (1×1, 3×3, 5×5 branches)
- 🟣 **Global Path** — ResNet-34 per-slice + Learnable Slice Weighting + Transformer
- 🟡 **Fusion** — Cross-attention gates decide which path to trust

**Key Finding**: MultiScale Attention is critical → removing it drops accuracy by **5.56%**

### 🔬 TARNet (Ultrasound)

**Problem**: Ultrasound is physics-based (echoes, attenuation, artifacts) — generic CNNs miss this!

**Solution**: Three acoustic-physics branches
- 🟢 **Attenuation Branch** — Models signal decay with depth: I(z) = I₀ × e^(−2αz)
- 🔵 **Backscatter Branch** — Echogenicity at multiple scales (3×3, 5×5, 7×7)
- 🟠 **Acoustic Shadow Branch** — Detects posterior artifacts (largest impact!)

**Dynamic Routing**: A gating network decides which branch contributes most per image

**Key Finding**: Acoustic Shadow Branch is essential → removing it drops accuracy by **7.69%** (largest single drop!)

### 🧠 MSCAF (MRI Scans)

**Problem**: MRI requires understanding features at multiple scales simultaneously

**Solution**: Multi-scale pyramid with cross-attention
- **Scale 1** (64 channels, 224×224) — Fine texture (necrosis, cell density)
- **Scale 2** (128 channels, 112×112) — Boundaries (tumor margins)
- **Scale 3** (256 channels, 56×56) — Morphology (overall shape, location)
- **Transformer** — Captures global spatial context across all scales

**Cross-Scale Fusion**: Each scale queries the Transformer independently

**Key Finding**: Multi-scale is critical → single-scale drops accuracy by **11.62%** (biggest loss!)

---

## 🎨 Design Philosophy

The **feminine colour palette** was chosen intentionally:
- Rose Pink `#e91e8c` — Attention & importance
- Lavender `#9c27b0` — Trust & calm
- Wine `#8b1a4a` — Confidence
- Butter `#f5c518` — Highlights

Medical AI tools are often austere and intimidating. This warmer, refined aesthetic makes clinical review more approachable. ✨

---

## 📖 Educational Resources

### Understanding the Models

**New to medical AI?** Start here:
1. Read the [About page](pages/about.py) — Visual explanations of each architecture
2. Upload test images — See predictions in real-time
3. Study activation maps — Understand where the model looks
4. Review ablation results — See what components matter most

### For Researchers

- `models/architectures.py` — Full model implementations with comments
- `utils/explainability.py` — GradCAM and attention visualization code
- Ablation results — In the About page documentation

---

## ⚠️ Medical Disclaimer

**RadioAI is a RESEARCH DEMONSTRATION TOOL ONLY.**

🚫 **NEVER use for clinical decision-making**
✅ **ALWAYS consult qualified radiologists and physicians**

This tool is designed for educational purposes and research demonstration. Medical AI requires rigorous validation and regulatory approval before clinical use.

---

## 📞 Support & Issues

Found a bug? Have a question?
- 📧 Open an issue on GitHub
- 💬 Check the About page for troubleshooting
- 🔧 See `requirements.txt` for dependency versions

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🌟 Highlights & Performance

| Model | Modality | Classes | Key Advantage | Ablation Impact |
|-------|----------|---------|---------------|-----------------|
| 🫁 DPMS-LSW | CT | 4 | Dual-path attention | -5.56% (MultiScale) |
| 🔬 TARNet | Ultrasound | 3 | Physics-grounded branches | -7.69% (AcousticShadow) |
| 🧠 MSCAF | MRI | 4 | Multi-scale CNN-Attention fusion | -11.62% (Single-scale) |

---

**Built with ❤️ for radiologists who value interpretability** 🩺✨
