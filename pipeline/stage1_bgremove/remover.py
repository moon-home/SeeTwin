"""
Stage 1 — Background removal using BRIA-RMBG-2.0.

Responsibilities:
- Load the model once and cache it (slow first load, fast subsequent calls)
- Resize input to max 1024px before inference
- Return a float32 alpha mask in [0, 1] range (the raw confidence map)
- NOT responsible for brush corrections or thresholding — see mask_editor.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Model is loaded lazily on first call and cached here
_model = None
_processor = None


def _load_model(device: str = "cpu"):
    """Load BRIA-RMBG-2.0 from HuggingFace (cached after first download)."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    try:
        from transformers import AutoModelForImageSegmentation
        from torchvision.transforms.functional import normalize
        import torch
    except ImportError as e:
        raise ImportError(
            "Missing dependencies. Run: pip install transformers torch torchvision"
        ) from e

    logger.info("Loading BRIA-RMBG-2.0 model (first load may take a minute)...")

    _model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
    )
    _model.to(device)
    _model.eval()

    # Store normalize function as a callable so we don't re-import it later
    _processor = normalize
    logger.info("Model loaded successfully.")
    return _model, _processor


def _preprocess(image: Image.Image, model_size: int = 1024):
    """
    Resize and normalise image for BRIA-RMBG-2.0.
    Returns a (1, 3, H, W) float32 tensor and the resized PIL image.
    """
    import torch
    import torchvision.transforms as T

    resized = image.resize((model_size, model_size), Image.BILINEAR)
    tensor = T.ToTensor()(resized).unsqueeze(0)          # (1, 3, H, W), [0, 1]
    tensor = _processor(tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return tensor, resized


def _postprocess(raw_output, original_size: tuple[int, int]) -> np.ndarray:
    """
    Convert raw model output to a float32 alpha mask resized to original_size.
    Returns ndarray of shape (H, W) with values in [0, 1].
    original_size: (width, height)
    """
    import torch

    # raw_output is a list of tensors; take the last one (finest prediction)
    pred = raw_output[-1].sigmoid().squeeze().cpu().numpy()  # (1024, 1024) float32

    # Resize back to original image dimensions
    mask_pil = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    mask_pil = mask_pil.resize(original_size, Image.BILINEAR)
    return np.array(mask_pil).astype(np.float32) / 255.0


def resize_to_max(image: Image.Image, max_px: int = 1024) -> Image.Image:
    """
    Proportionally resize an image so neither dimension exceeds max_px.
    Returns the image unchanged if it is already within bounds.
    """
    w, h = image.size
    if max(w, h) <= max_px:
        return image
    scale = max_px / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def run_model(
    image: Image.Image,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run BRIA-RMBG-2.0 on a single PIL image.

    Args:
        image:  PIL Image (RGB). Will be resized to 1024px for inference,
                but the returned mask matches the INPUT image dimensions.
        device: "cpu" or "cuda"

    Returns:
        alpha_mask: float32 ndarray of shape (H, W), values in [0, 1].
                    1.0 = foreground (keep), 0.0 = background (remove).
                    This is the RAW confidence map — apply threshold in mask_editor.
    """
    import torch

    model, processor = _load_model(device)

    original_size = image.size  # (width, height)
    image_rgb = image.convert("RGB")

    with torch.no_grad():
        tensor, _ = _preprocess(image_rgb)
        tensor = tensor.to(device)
        output = model(tensor)

    alpha = _postprocess(output, original_size)
    return alpha


def apply_alpha_to_image(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """
    Composite an RGBA image using the given alpha mask.

    Args:
        image: PIL Image (any mode)
        alpha: float32 ndarray (H, W), values in [0, 1]

    Returns:
        PIL Image in RGBA mode with transparent background.
    """
    rgba = image.convert("RGBA")
    alpha_uint8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    alpha_channel = Image.fromarray(alpha_uint8, mode="L")
    rgba.putalpha(alpha_channel)
    return rgba
