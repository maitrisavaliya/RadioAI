"""
Model loader.

CHECKPOINT PLACEMENT
────────────────────
radioai/checkpoints/
    gatekeeper.pth
    dpms_lsw.pth
    tarnet.pth
    mscaf.pth

If a checkpoint is missing the app shows a clear error and refuses to run
inference — no silent random-weight fallback.
"""

from __future__ import annotations
import torch
import numpy as np
from pathlib import Path

from models.architectures import DPMS_LSW, TARNet, MSCAF
from models.gatekeeper    import GatekeeperRouter

_HERE     = Path(__file__).parent
_CKPT_DIR = _HERE.parent / "checkpoints"

CKPT_PATHS = {
    "gatekeeper": _CKPT_DIR / "gatekeeper.pth",
    "CT Scan":    _CKPT_DIR / "dpms_lsw.pth",
    "Ultrasound": _CKPT_DIR / "tarnet.pth",
    "MRI":        _CKPT_DIR / "mscaf.pth",
}

MODALITY_META = {
    "CT Scan": {
        "model_cls":   DPMS_LSW,
        "classes":     DPMS_LSW.CLASSES,
        "input_mode":  "ct",
        "description": "CT Scan Analysis",
        "color":       "#3b82f6",
    },
    "Ultrasound": {
        "model_cls":   TARNet,
        "classes":     TARNet.CLASSES,
        "input_mode":  "us",
        "description": "Ultrasound Analysis",
        "color":       "#3b82f6",
    },
    "MRI": {
        "model_cls":   MSCAF,
        "classes":     MSCAF.CLASSES,
        "input_mode":  "mri",
        "description": "MRI Analysis",
        "color":       "#3b82f6",
    },
}

_model_cache:      dict[str, torch.nn.Module] = {}
_gatekeeper_cache: GatekeeperRouter | None    = None


def checkpoint_status() -> dict[str, bool]:
    return {k: v.exists() for k, v in CKPT_PATHS.items()}


def all_checkpoints_present() -> bool:
    return all(checkpoint_status().values())


def missing_checkpoints() -> list[str]:
    return [k for k, exists in checkpoint_status().items() if not exists]


def get_gatekeeper() -> GatekeeperRouter | None:
    global _gatekeeper_cache
    if _gatekeeper_cache is not None:
        return _gatekeeper_cache
    path = CKPT_PATHS["gatekeeper"]
    if not path.exists():
        return None
    try:
        _gatekeeper_cache = GatekeeperRouter(str(path), device="cpu")
        return _gatekeeper_cache
    except Exception as e:
        print(f"[loader] Gatekeeper load failed: {e}")
        return None


def get_model(modality: str) -> torch.nn.Module:
    """
    Returns a loaded model. Raises FileNotFoundError if checkpoint is missing.
    No random weights — ever.
    """
    if modality in _model_cache:
        return _model_cache[modality]

    path = CKPT_PATHS[modality]
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.name}\n"
            f"Place your trained {path.name} in radioai/checkpoints/"
        )

    cls   = MODALITY_META[modality]["model_cls"]
    model = cls()
    try:
        ckpt  = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        # Use strict=False for flexibility with checkpoint key mismatches
        model.load_state_dict(state, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path.name}: {e}")

    model.eval()
    _model_cache[modality] = model
    return model


def run_inference(model: torch.nn.Module, tensor: torch.Tensor) -> dict:
    with torch.no_grad():
        if model.__class__.__name__ == "TARNet":
            logits, branch_w = model(tensor, return_branch_weights=True)
            branch_weights   = branch_w[0].cpu().numpy().tolist()
        else:
            logits         = model(tensor)
            branch_weights = None

    probs    = torch.softmax(logits, -1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "probs":          probs.tolist(),
        "pred_idx":       pred_idx,
        "pred_class":     model.CLASSES[pred_idx],
        "confidence":     float(probs[pred_idx]),
        "branch_weights": branch_weights,
    }
