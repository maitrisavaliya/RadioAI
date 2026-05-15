"""
Preprocessing pipelines for each imaging modality.
MRI   : NLM denoising → CLAHE → z-score normalisation
CT    : pseudo-volume (8 augmented slices) → per-channel normalise
US    : augmentation-light resize → mean/std 0.5 normalise
"""

import cv2
import numpy as np
import torch
from PIL import Image
import io


def _to_gray_np(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes → uint8 grayscale numpy array."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pil = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = np.array(pil)
    return img


# ── MRI ──────────────────────────────────────────────────────────────────────

def preprocess_mri(image_bytes: bytes) -> torch.Tensor:
    """
    1. NLM denoising (h=10)
    2. CLAHE (clipLimit=2.0, tileGridSize=8×8)
    3. Lanczos4 resize to 224×224
    4. Z-score intensity normalisation
    Returns: (1, 1, 224, 224) tensor
    """
    img = _to_gray_np(image_bytes)
    img = cv2.fastNlMeansDenoising(img, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-8)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)   # (1,1,224,224)
    return tensor


# ── CT ───────────────────────────────────────────────────────────────────────

def _ct_augment(img: np.ndarray, idx: int) -> np.ndarray:
    """Generate pseudo-slice #idx from a single CT image."""
    rng = np.random.RandomState(idx * 7 + 13)
    aug = img.copy().astype(np.float32)
    if rng.rand() > 0.5:
        aug = np.fliplr(aug)
    angle = rng.uniform(-5, 5)
    h, w = aug.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h))
    tx, ty = rng.uniform(-0.05, 0.05, 2) * np.array([w, h])
    M2 = np.float32([[1, 0, tx], [0, 1, ty]])
    aug = cv2.warpAffine(aug, M2, (w, h))
    return aug


def preprocess_ct(image_bytes: bytes, n_slices: int = 8) -> torch.Tensor:
    """
    Simulate a pseudo-volume by creating n_slices augmented copies.
    Returns: (1, n_slices, 1, 224, 224) tensor
    """
    img = _to_gray_np(image_bytes)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
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
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
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
    return cv2.resize(img, (224, 224))
