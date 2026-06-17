"""
Stage 2 — Keypoint extraction via MediaPipe Pose (Tasks API, v0.10+).

Produces 33 normalised landmarks per image and an annotated PIL image
showing the skeleton overlay with landmark labels.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp

logger = logging.getLogger(__name__)

_MODEL_PATH = str(Path(__file__).parents[2] / "models" / "mediapipe" / "pose_landmarker_heavy.task")
_POSE = None  # module-level singleton


def _get_pose():
    global _POSE
    if _POSE is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"MediaPipe pose model not found at {_MODEL_PATH}. "
                "Download it from the mediapipe model card and place it there."
            )
        base_opts = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        _POSE = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
    return _POSE


# MediaPipe landmark index → short label
LANDMARK_NAMES: dict[int, str] = {
    0:  "nose",
    1:  "L-eye-in", 2: "L-eye", 3: "L-eye-out",
    4:  "R-eye-in", 5: "R-eye", 6: "R-eye-out",
    7:  "L-ear",    8: "R-ear",
    9:  "mouth-L", 10: "mouth-R",
    11: "L-shldr", 12: "R-shldr",
    13: "L-elbow", 14: "R-elbow",
    15: "L-wrist", 16: "R-wrist",
    17: "L-pinky", 18: "R-pinky",
    19: "L-index", 20: "R-index",
    21: "L-thumb", 22: "R-thumb",
    23: "L-hip",   24: "R-hip",
    25: "L-knee",  26: "R-knee",
    27: "L-ankle", 28: "R-ankle",
    29: "L-heel",  30: "R-heel",
    31: "L-foot",  32: "R-foot",
}

# Skeleton edges to draw (pairs of landmark indices)
_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

_EDGE_SIDE: dict[tuple[int, int], bool | None] = {
    (11, 12): None, (23, 24): None,
    (11, 13): True,  (13, 15): True,
    (12, 14): False, (14, 16): False,
    (11, 23): True,  (23, 25): True,  (25, 27): True,  (27, 29): True,  (27, 31): True,
    (12, 24): False, (24, 26): False, (26, 28): False, (28, 30): False, (28, 32): False,
}

_LEFT_COLOR  = (0, 200, 120)
_RIGHT_COLOR = (220, 80, 80)
_DOT_COLOR   = (255, 220, 50)


def extract_keypoints(image: Image.Image) -> dict:
    """
    Run MediaPipe Pose on a PIL image (RGBA or RGB).

    Returns:
        {detected: bool, landmarks: list of 33 dicts with x,y,z,visibility}
    """
    rgb = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_pose().detect(mp_image)

    if not result.pose_landmarks:
        return {"detected": False, "landmarks": None}

    lms = []
    for lm in result.pose_landmarks[0]:
        lms.append({
            "x":          float(lm.x),
            "y":          float(lm.y),
            "z":          float(lm.z),
            "visibility": float(getattr(lm, "visibility", 1.0)),
        })
    return {"detected": True, "landmarks": lms}


def _checker_background(w: int, h: int, tile: int = 20) -> Image.Image:
    """Light grey checker — standard transparency indicator."""
    light, dark = (220, 220, 220), (190, 190, 190)
    arr = np.full((h, w, 3), light, dtype=np.uint8)
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            if (r // tile + c // tile) % 2 == 1:
                arr[r:r+tile, c:c+tile] = dark
    return Image.fromarray(arr).convert("RGBA")


def draw_landmarks(image: Image.Image, kp_result: dict,
                   selected_idx: int = -1) -> Image.Image:
    """
    Return an RGB annotated image with the 33 pose landmarks and skeleton.

    If the input is RGBA (transparent-background photo from Stage 1), the
    subject is composited onto a light checker background so the landmark
    overlay is easy to read without background clutter.

    Args:
        image:        PIL Image, RGB or RGBA
        kp_result:    dict from extract_keypoints()
        selected_idx: landmark index to highlight in white (for correction UI)
    """
    if image.mode == "RGBA":
        bg = _checker_background(image.width, image.height)
        base = Image.alpha_composite(bg, image)
    else:
        base = image.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if not kp_result.get("detected"):
        out = base.convert("RGB")
        ImageDraw.Draw(out).text((10, 10), "No person detected", fill=(255, 80, 80))
        return out

    lms = kp_result["landmarks"]
    W, H = image.size

    def px(idx):
        return int(lms[idx]["x"] * W), int(lms[idx]["y"] * H)

    def vis(idx):
        return max(80, int(lms[idx]["visibility"] * 255))

    # Skeleton edges — 5 px wide
    for a, b in _EDGES:
        side = _EDGE_SIDE.get((a, b))
        col = _LEFT_COLOR if side is True else (_RIGHT_COLOR if side is False else (180, 180, 180))
        al = min(vis(a), vis(b))
        draw.line([px(a), px(b)], fill=(*col, al), width=5)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    # Dots + labels
    for idx, name in LANDMARK_NAMES.items():
        x, y = px(idx)
        al = vis(idx)
        r = 10 if idx == selected_idx else 7
        dot_col = (255, 255, 255) if idx == selected_idx else _DOT_COLOR

        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(*dot_col, al), outline=(0, 0, 0, min(al + 60, 255)), width=2)

        # Text shadow → readability on any background
        for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            draw.text((x + 9 + dx, y - 7 + dy), name,
                      fill=(0, 0, 0, min(al, 200)), font=font)
        draw.text((x + 9, y - 7), name, fill=(*dot_col, al), font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")
