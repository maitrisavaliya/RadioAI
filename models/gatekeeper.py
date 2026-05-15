"""
Gatekeeper — robust MobileNetV3-Small router (CT / MRI / Ultrasound)

✔ Matches training pipeline exactly
✔ Fixes class_names mismatch bug
✔ Handles torchvision version differences
✔ Safe + debuggable
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io


# ImageNet stats (same as training)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# Output mapping
GATEKEEPER_TO_MODALITY = {
    "ct":         "CT Scan",
    "mri":        "MRI",
    "ultrasound": "Ultrasound",
}


# ─────────────────────────────────────────────────────────────
# Model builder (ROBUST)
# ─────────────────────────────────────────────────────────────
def _build_model(num_classes: int):
    model = models.mobilenet_v3_small(weights=None)

    # 🔥 always replace LAST layer (safe across versions)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────
class GatekeeperRouter:
    def __init__(self, checkpoint_path: str, device: str = "cpu", debug: bool = False):
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # 🔥 FIX: enforce sorted class order (matches training labels)
        raw_class_names = ckpt["class_names"]
        self.class_names = sorted(raw_class_names)

        self.img_size   = ckpt.get("img_size", 224)
        num_classes     = ckpt.get("num_classes", len(self.class_names))

        # Build model
        self.model = _build_model(num_classes).to(self.device)

        # Load weights safely
        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )

        self.model.eval()

        # Transform (EXACT match to val/test pipeline)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

        # Debug info
        if debug:
            print("\n--- GATEKEEPER DEBUG ---")
            print("Checkpoint classes :", raw_class_names)
            print("Sorted classes     :", self.class_names)
            print("Missing keys       :", missing)
            print("Unexpected keys    :", unexpected)
            print("-------------------------\n")

    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def route_bytes(self, image_bytes: bytes, debug: bool = False) -> dict:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image: {e}")

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx   = int(np.argmax(probs))
        label = self.class_names[idx]

        if debug:
            print("Probabilities:", {
                c: float(p) for c, p in zip(self.class_names, probs)
            })

        return {
            "gatekeeper_label": label,
            "modality":         GATEKEEPER_TO_MODALITY[label],
            "confidence":       float(probs[idx]),
            "all_probs":        {
                c: float(p) for c, p in zip(self.class_names, probs)
            },
        }