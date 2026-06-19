"""
CLIP-based zero-shot classifier for Stage 3.
Runs on CPU by default; uses GPU if available.
Model downloads ~350 MB on first call (ViT-B-32, OpenAI weights).
"""

from __future__ import annotations
import numpy as np
from functools import lru_cache
from PIL import Image

from .asset_library import (
    HAIR_ASSETS, FACE_ASSETS, GARMENT_ASSETS, ACCESSORY_ASSETS,
    ACCESSORY_DETECTION_THRESHOLD,
)

CONFIDENCE_THRESHOLD = 0.85  # auto-confirm above this; surface alternatives below


@lru_cache(maxsize=1)
def _load_model():
    import open_clip, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, device


def _scores(image: Image.Image, labels: list[str]) -> list[float]:
    import torch
    model, preprocess, tokenizer, device = _load_model()
    img_rgb = image.convert("RGB")
    with torch.no_grad():
        img_t = preprocess(img_rgb).unsqueeze(0).to(device)
        txt_t = tokenizer(labels).to(device)
        img_f = model.encode_image(img_t)
        txt_f = model.encode_text(txt_t)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        probs = (img_f @ txt_f.T).softmax(dim=-1)[0].cpu().tolist()
    return probs


def _crop(img: Image.Image,
          top: float, bottom: float,
          left: float = 0.1, right: float = 0.9) -> Image.Image:
    w, h = img.size
    return img.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))


def sample_dominant_color(region: Image.Image) -> str:
    """Return hex string of the median RGB in the region, ignoring near-black/white."""
    small = region.resize((40, 40)).convert("RGB")
    arr = np.array(small, dtype=float)
    brightness = arr.mean(axis=2)
    mask = (brightness > 25) & (brightness < 230)
    if mask.sum() < 5:
        return "#888888"
    rgb = arr[mask].mean(axis=0)
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


# ── Per-category classifiers ──────────────────────────────────────────────────

def classify_face(img: Image.Image, top_n: int = 3) -> tuple[Image.Image, list[dict]]:
    crop = _crop(img, 0.04, 0.38, 0.18, 0.82)
    labels = [f"a face with {a['label']}" for a in FACE_ASSETS]
    scores = _scores(crop, labels)
    ranked = sorted(
        [{"id": a["id"], "label": a["label"], "confidence": s}
         for a, s in zip(FACE_ASSETS, scores)],
        key=lambda x: x["confidence"], reverse=True,
    )
    return crop, ranked[:top_n]


def classify_hair(img: Image.Image, top_n: int = 5) -> tuple[Image.Image, list[dict], str]:
    crop = _crop(img, 0.0, 0.28, 0.12, 0.88)
    labels = [f"a photo of a person with {a['label']}" for a in HAIR_ASSETS]
    scores = _scores(crop, labels)
    ranked = sorted(
        [{"id": a["id"], "label": a["label"], "confidence": s}
         for a, s in zip(HAIR_ASSETS, scores)],
        key=lambda x: x["confidence"], reverse=True,
    )
    color_hex = sample_dominant_color(crop)
    return crop, ranked[:top_n], color_hex


def classify_upper_garment(img: Image.Image, top_n: int = 4) -> tuple[Image.Image, list[dict], str]:
    crop = _crop(img, 0.18, 0.62)
    assets = [a for a in GARMENT_ASSETS if a["category"] in ("top", "outerwear", "dress")]
    labels = [f"a person wearing a {a['label']}" for a in assets]
    scores = _scores(crop, labels)
    ranked = sorted(
        [{"id": a["id"], "label": a["label"], "category": a["category"], "confidence": s}
         for a, s in zip(assets, scores)],
        key=lambda x: x["confidence"], reverse=True,
    )
    color_hex = sample_dominant_color(crop)
    return crop, ranked[:top_n], color_hex


def classify_lower_garment(img: Image.Image, top_n: int = 4) -> tuple[Image.Image, list[dict], str]:
    crop = _crop(img, 0.50, 0.95)
    assets = [a for a in GARMENT_ASSETS if a["category"] in ("bottom", "footwear")]
    labels = [f"a person wearing {a['label']}" for a in assets]
    scores = _scores(crop, labels)
    ranked = sorted(
        [{"id": a["id"], "label": a["label"], "category": a["category"], "confidence": s}
         for a, s in zip(assets, scores)],
        key=lambda x: x["confidence"], reverse=True,
    )
    color_hex = sample_dominant_color(crop)
    return crop, ranked[:top_n], color_hex


def detect_accessories(img: Image.Image) -> list[dict]:
    """
    For each accessory category, run a binary detection pass:
    "wearing X" vs "not wearing any X". Categories above threshold
    are included with their top variant.
    """
    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    for a in ACCESSORY_ASSETS:
        by_cat[a["category"]].append(a)

    detected = []
    for cat, assets in by_cat.items():
        present_labels = [f"a person wearing {a['label']}" for a in assets]
        absent_label = f"a person not wearing any {cat}"
        all_labels = present_labels + [absent_label]
        scores = _scores(img, all_labels)
        present_scores = scores[:-1]
        absent_score = scores[-1]

        best_score = max(present_scores)
        if best_score >= ACCESSORY_DETECTION_THRESHOLD and best_score > absent_score:
            best_idx = int(np.argmax(present_scores))
            alternatives = sorted(
                [{"id": a["id"], "label": a["label"], "confidence": s}
                 for a, s in zip(assets, present_scores)],
                key=lambda x: x["confidence"], reverse=True,
            )[:3]
            detected.append({
                "category": cat,
                "id": assets[best_idx]["id"],
                "label": assets[best_idx]["label"],
                "confidence": best_score,
                "alternatives": alternatives,
            })

    return sorted(detected, key=lambda x: x["confidence"], reverse=True)


# ── Master runner ─────────────────────────────────────────────────────────────

def classify_all(img_front: Image.Image) -> dict:
    """
    Run all classifiers on the front photo. Returns a dict with keys:
    face_crop, face_preds,
    hair_crop, hair_preds, hair_color,
    upper_crop, upper_preds, upper_color,
    lower_crop, lower_preds, lower_color,
    accessories
    """
    face_crop, face_preds                     = classify_face(img_front)
    hair_crop, hair_preds, hair_color         = classify_hair(img_front)
    upper_crop, upper_preds, upper_color      = classify_upper_garment(img_front)
    lower_crop, lower_preds, lower_color      = classify_lower_garment(img_front)
    accessories                               = detect_accessories(img_front)

    return {
        "face_crop":   face_crop,
        "face_preds":  face_preds,
        "hair_crop":   hair_crop,
        "hair_preds":  hair_preds,
        "hair_color":  hair_color,
        "upper_crop":  upper_crop,
        "upper_preds": upper_preds,
        "upper_color": upper_color,
        "lower_crop":  lower_crop,
        "lower_preds": lower_preds,
        "lower_color": lower_color,
        "accessories": accessories,
    }
