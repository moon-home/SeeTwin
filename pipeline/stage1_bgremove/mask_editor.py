"""
Stage 1 — Mask editor.

Handles all mask manipulation that happens AFTER the model runs:
- Thresholding the raw float confidence map into a binary mask
- Painting foreground / background brush strokes on top of the model mask
- Merging stroke layer with model mask (strokes always win)
- Feathering the final mask edge

The stroke layer and model mask are kept separate so the user can
re-run the model at any time without losing their painted corrections.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

def apply_threshold(alpha: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert a float [0,1] confidence map to a binary float mask using threshold.

    Args:
        alpha:      float32 (H, W), raw model output
        threshold:  float in [0, 1]; pixels above → 1.0, below → 0.0

    Returns:
        float32 (H, W) binary mask
    """
    return (alpha >= threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# Brush strokes
# ---------------------------------------------------------------------------

class StrokeLayer:
    """
    Tracks user brush corrections as a pair of boolean masks:
    - fg_strokes: pixels the user painted as foreground (keep)
    - bg_strokes: pixels the user painted as background (remove)

    Strokes are stored at the same resolution as the working image.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.fg_strokes: np.ndarray = np.zeros((height, width), dtype=bool)
        self.bg_strokes: np.ndarray = np.zeros((height, width), dtype=bool)

    def paint(
        self,
        cx: int,
        cy: int,
        radius: int,
        mode: str,  # "fg" or "bg"
    ) -> None:
        """
        Paint a circular brush stroke at pixel (cx, cy).

        Args:
            cx, cy:  centre pixel coordinates (x=col, y=row)
            radius:  brush radius in pixels
            mode:    "fg" to mark as foreground, "bg" to mark as background
        """
        h, w = self.height, self.width
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

        if mode == "fg":
            self.fg_strokes[y_min:y_max, x_min:x_max][circle] = True
            # A fg stroke cancels any bg stroke in the same spot
            self.bg_strokes[y_min:y_max, x_min:x_max][circle] = False
        else:
            self.bg_strokes[y_min:y_max, x_min:x_max][circle] = True
            self.fg_strokes[y_min:y_max, x_min:x_max][circle] = False

    def clear(self) -> None:
        """Remove all painted strokes."""
        self.fg_strokes[:] = False
        self.bg_strokes[:] = False

    def is_empty(self) -> bool:
        return not (self.fg_strokes.any() or self.bg_strokes.any())

    def stroke_count(self) -> int:
        """Approximate number of painted regions (connected components would be
        more accurate, but pixel count is fast and good enough for the status bar)."""
        return int(self.fg_strokes.sum() + self.bg_strokes.sum())


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(
    model_mask: np.ndarray,
    strokes: StrokeLayer,
    threshold: float = 0.5,
    feather_px: int = 0,
) -> np.ndarray:
    """
    Produce the final compositing mask by merging model output and brush strokes.

    Priority:
      1. bg_strokes → force 0 (remove)
      2. fg_strokes → force 1 (keep)
      3. model_mask thresholded at `threshold`

    Args:
        model_mask:  float32 (H, W) raw model confidence map
        strokes:     StrokeLayer instance
        threshold:   float in [0, 1]
        feather_px:  Gaussian feather radius applied to final edge (0 = none)

    Returns:
        float32 (H, W) mask in [0, 1] ready for compositing
    """
    binary = apply_threshold(model_mask, threshold)

    # Apply stroke overrides
    binary[strokes.bg_strokes] = 0.0
    binary[strokes.fg_strokes] = 1.0

    if feather_px > 0:
        binary = _feather(binary, feather_px)

    return binary


# ---------------------------------------------------------------------------
# Feathering
# ---------------------------------------------------------------------------

def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Soften the mask edge with a Gaussian blur, preserving the core regions.
    Only the edge band (within `radius` pixels of the boundary) is blurred.
    """
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Visualisation helpers (used by the Gradio UI)
# ---------------------------------------------------------------------------

def overlay_strokes(
    image_rgba: Image.Image,
    strokes: StrokeLayer,
    fg_color: tuple = (30, 200, 120, 160),   # green, 63% opacity
    bg_color: tuple = (220, 60, 60, 160),     # red, 63% opacity
) -> Image.Image:
    """
    Overlay stroke regions on an RGBA image for display in the canvas.
    Returns a new RGBA image with coloured stroke overlays.
    """
    result = image_rgba.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)

    if strokes.fg_strokes.any():
        overlay_arr[strokes.fg_strokes] = fg_color
    if strokes.bg_strokes.any():
        overlay_arr[strokes.bg_strokes] = bg_color

    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    return Image.alpha_composite(result, overlay)


def checkerboard_background(width: int, height: int, tile: int = 16) -> Image.Image:
    """
    Generate a grey checkerboard pattern (standard transparency indicator).
    Used as the canvas background behind the masked image preview.
    """
    light, dark = 200, 160
    arr = np.full((height, width, 3), light, dtype=np.uint8)
    for r in range(0, height, tile):
        for c in range(0, width, tile):
            if (r // tile + c // tile) % 2 == 1:
                arr[r:r + tile, c:c + tile] = dark
    return Image.fromarray(arr, mode="RGB")


def composite_on_checker(image_rgba: Image.Image) -> Image.Image:
    """Composite an RGBA image onto a checkerboard for display."""
    bg = checkerboard_background(image_rgba.width, image_rgba.height)
    bg = bg.convert("RGBA")
    return Image.alpha_composite(bg, image_rgba.convert("RGBA"))
