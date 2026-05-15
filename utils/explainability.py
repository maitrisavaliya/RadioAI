"""
Activation map visualisation for RadioAI.
Clean dark-blue palette, no complex animations.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
from PIL import Image

BG   = "#0f172a"
CARD = "#1e293b"
SUB  = "#94a3b8"
TEXT = "#e2e8f0"


# ── Activation map ────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, input_tensor, pred_idx):
        input_tensor.requires_grad_()  # Enable gradients for input
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]  # In case of TARNet with branch weights
        score = output[0, pred_idx]

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Compute GradCAM
        if self.activations is None or self.gradients is None:
            return None

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).relu()

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def _get_target_layer(model):
    name = model.__class__.__name__
    if name == "DPMS_LSW":
        return model.local_attention.scale3[0]  # First conv in scale3
    elif name == "TARNet":
        return model.bsc_branch.fine[0]  # First conv in backscatter fine branch
    elif name == "MSCAF":
        return model.cnn_scale3[3]  # Second conv in scale3
    raise ValueError(f"No target layer for {name}")


def compute_gradcam(model, input_tensor, pred_idx):
    try:
        target_layer = _get_target_layer(model)
        gradcam = GradCAM(model, target_layer)
        cam = gradcam(input_tensor, pred_idx)
        gradcam.remove()
        if cam is None:
            return None
        # Upsample to 224x224
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_up = F.interpolate(cam_tensor, (224, 224), mode="bilinear", align_corners=False)
        return cam_up.squeeze().numpy()
    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None


# ── Overlay + figure ──────────────────────────────────────────────────────────

def _overlay(gray: np.ndarray, cam: np.ndarray, alpha=0.5) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    heat = cm.get_cmap("Blues")(cam)[..., :3]
    blended = (1 - alpha) * base + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def make_gradcam_figure(original_gray: np.ndarray, cam: np.ndarray,
                        title: str = "") -> BytesIO:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), facecolor=CARD)
    panels = [
        (original_gray, "gray",   "Original"),
        (cam,           "Blues",  "Activation"),
        (_overlay(original_gray, cam), None, "Overlay"),
    ]
    for ax, (data, cmap, label) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(label, color=SUB, fontsize=8.5, pad=5)
        ax.axis("off")
        ax.set_facecolor(CARD)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=10, fontweight="600", y=1.01)
    fig.patch.set_facecolor(CARD)
    plt.tight_layout(pad=0.5)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=CARD, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── TARNet branch radar ───────────────────────────────────────────────────────

BRANCH_LABELS = ["Attenuation", "Backscatter", "Shadow"]


def make_branch_radar(weights: list) -> BytesIO:
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True),
                           facecolor=CARD)
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]
    vals   = list(weights) + [weights[0]]

    ax.set_facecolor(BG)
    ax.plot(angles, vals, "o-", color="#3b82f6", linewidth=2)
    ax.fill(angles, vals, alpha=0.2, color="#3b82f6")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(BRANCH_LABELS, color=SUB, size=8)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#334155")
    ax.spines["polar"].set_color("#334155")
    ax.set_title("Branch weights", color=SUB, fontsize=9, pad=12)
    for r in ax.yaxis.get_gridlines():
        r.set_color("#1e293b")
    fig.patch.set_facecolor(CARD)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=CARD, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf
