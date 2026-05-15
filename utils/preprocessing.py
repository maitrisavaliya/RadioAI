"""
Preprocessing pipelines for each imaging modality.
MRI   : NLM denoising → CLAHE → z-score normalisation
CT    : pseudo-volume (8 augmented slices) → per-channel normalise
US    : augmentation-light resize → mean/std 0.5 normalise
"""

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
import io


def _to_gray_np(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes → uint8 grayscale numpy array."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
        return np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")


def _apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Simple histogram equalization as substitute for CLAHE."""
    pil_img = Image.fromarray(img)
    pil_img = ImageOps.equalize(pil_img)
    return np.array(pil_img)


def _simple_denoise(img: np.ndarray) -> np.ndarray:
    """Simple median filter as substitute for NLM denoising."""
    pil_img = Image.fromarray(img)
    pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))
    return np.array(pil_img)


# ── MRI ──────────────────────────────────────────────────────────────────────

def preprocess_mri(image_bytes: bytes) -> torch.Tensor:
    """
    1. Simple denoising
    2. Histogram equalization (CLAHE substitute)
    3. Resize to 224×224
    4. Z-score intensity normalisation
    Returns: (1, 1, 224, 224) tensor
    """
    img = _to_gray_np(image_bytes)
    img = _simple_denoise(img)
    img = _apply_clahe(img)
    
    # Resize using PIL
    pil_img = Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS)
    img = np.array(pil_img).astype(np.float32)
    
    # Z-score normalization
    img = (img - img.mean()) / (img.std() + 1e-8)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)   # (1,1,224,224)
    return tensor


# ── CT ───────────────────────────────────────────────────────────────────────

def _ct_augment(img: np.ndarray, idx: int) -> np.ndarray:
    """Generate pseudo-slice #idx from a single CT image with augmentation."""
    rng = np.random.RandomState(idx * 7 + 13)
    pil_img = Image.fromarray(img.astype(np.uint8))
    
    # Random flip
    if rng.rand() > 0.5:
        pil_img = ImageOps.mirror(pil_img)
    
    # Random rotation (small angle)
    angle = rng.uniform(-5, 5)
    pil_img = pil_img.rotate(angle, expand=False, resample=Image.Resampling.BILINEAR)
    
    return np.array(pil_img).astype(np.float32)


def preprocess_ct(image_bytes: bytes, n_slices: int = 8) -> torch.Tensor:
    """
    Simulate a pseudo-volume by creating n_slices augmented copies.
    Returns: (1, n_slices, 1, 224, 224) tensor
    """
    img = _to_gray_np(image_bytes)
    
    # Resize to 224x224
    pil_img = Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS)
    img = np.array(pil_img).astype(np.float32)
    
    slices = []
    for i in range(n_slices):
        s = _ct_augment(img, i)
        s = (s - s.mean()) / (s.std() + 1e-8)
        slices.append(torch.from_numpy(s).unsqueeze(0))         # (1,224,224)
    
    volume = torch.stack(slices, 0).unsqueeze(0)               # (1, n, 1, 224, 224)
    return volume


# ── Ultrasound ────────────────────────────────────────────────────────────────

def preprocess_ultrasound(image_bytes: bytes) -> torch.Tensor:
    """
    Resize to 224×224, normalise to mean=0.5, std=0.5.
    Returns: (1, 1, 224, 224) tensor
    """
    img = _to_gray_np(image_bytes)
    
    # Resize using PIL
    pil_img = Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS)
    img = np.array(pil_img).astype(np.float32) / 255.0
    
    # Normalize
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return tensor


# ── Dispatcher ───────────────────────────────────────────────────────────────

def preprocess(image_bytes: bytes, modality: str) -> torch.Tensor:
    dispatch = {
        "MRI":        preprocess_mri,
        "CT Scan":    preprocess_ct,
        "Ultrasound": preprocess_ultrasound,
    }
    if modality not in dispatch:
        raise ValueError(f"Unknown modality: {modality}")
    return dispatch[modality](image_bytes)


def get_display_image(image_bytes: bytes) -> np.ndarray:
    """Return a uint8 grayscale numpy array for display purposes."""
    img = _to_gray_np(image_bytes)
    pil_img = Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS)
    return np.array(pil_img)
