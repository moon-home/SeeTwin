"""
Stage 1 — Background removal UI (Gradio).

Layout:
  Left col  — upload slots for 3 photos (front, side, back)
  Centre    — canvas showing active photo with mask preview and stroke overlay
  Right col — brush mode, brush size, threshold slider, feather slider, actions

State that lives in Gradio:
  - Uploaded PIL images (3 slots)
  - Raw model alpha masks (3 floats arrays, one per photo)
  - StrokeLayer per photo (brush corrections)
  - Active photo index (0 = front, 1 = side, 2 = back)
  - Current threshold and feather values
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from pipeline.stage1_bgremove import (
    StrokeLayer,
    apply_alpha_to_image,
    composite_on_checker,
    merge,
    overlay_strokes,
    resize_to_max,
    run_model,
)

MAX_PX = 1024
PHOTO_LABELS = ["Front", "Side", "Back"]
BRUSH_SIZES = [8, 16, 24, 36]  # pixel radii
DEFAULT_THRESHOLD = 0.5
DEFAULT_FEATHER = 2


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _empty_session() -> dict:
    return {
        "images": [None, None, None],       # PIL Image per slot
        "alpha_masks": [None, None, None],  # float32 ndarray per slot
        "strokes": [None, None, None],      # StrokeLayer per slot
        "active": 0,                        # currently viewed photo index
    }


def _get_strokes(session: dict, idx: int) -> Optional[StrokeLayer]:
    return session["strokes"][idx]


def _ensure_strokes(session: dict, idx: int) -> StrokeLayer:
    if session["strokes"][idx] is None and session["images"][idx] is not None:
        img = session["images"][idx]
        session["strokes"][idx] = StrokeLayer(img.height, img.width)
    return session["strokes"][idx]


# ---------------------------------------------------------------------------
# Canvas rendering
# ---------------------------------------------------------------------------

def _render_canvas(session: dict, threshold: float, feather: int) -> Optional[Image.Image]:
    """
    Produce the canvas image for the active photo slot.
    Shows the original image composited on a checkerboard with stroke overlay.
    """
    idx = session["active"]
    image = session["images"][idx]
    alpha = session["alpha_masks"][idx]
    strokes = session["strokes"][idx]

    if image is None:
        return None

    if alpha is None:
        # Model hasn't run yet — show original on checkerboard
        return composite_on_checker(image.convert("RGBA"))

    # Build final mask
    sl = strokes if strokes is not None else StrokeLayer(image.height, image.width)
    final_mask = merge(alpha, sl, threshold=threshold, feather_px=feather)

    # Composite with transparency
    rgba = apply_alpha_to_image(image, final_mask)

    # Overlay stroke visualisation
    if strokes is not None and not strokes.is_empty():
        rgba = overlay_strokes(rgba, strokes)

    return composite_on_checker(rgba)


def _status_text(session: dict) -> str:
    idx = session["active"]
    alpha = session["alpha_masks"][idx]
    strokes = session["strokes"][idx]

    if alpha is None:
        return "Upload a photo and click **Run model**"

    stroke_px = strokes.stroke_count() if strokes else 0
    parts = [f"Photo: {PHOTO_LABELS[idx]}"]
    if stroke_px > 0:
        parts.append(f"{stroke_px:,} painted pixels")
    else:
        parts.append("No corrections yet")

    ready = all(session["alpha_masks"])
    parts.append("✓ All 3 photos ready" if ready else "Waiting for all 3 photos")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def handle_upload(file_path: str, slot_idx: int, session: dict) -> tuple:
    """Handle photo upload for a given slot."""
    if file_path is None:
        return session, None, _status_text(session)

    image = Image.open(file_path).convert("RGB")
    image = resize_to_max(image, MAX_PX)

    session["images"][slot_idx] = image
    session["alpha_masks"][slot_idx] = None  # reset mask on new upload
    session["strokes"][slot_idx] = None
    session["active"] = slot_idx

    canvas = _render_canvas(session, DEFAULT_THRESHOLD, DEFAULT_FEATHER)
    return session, canvas, _status_text(session)


def handle_run_model(session: dict, threshold: float, feather: int, device: str) -> tuple:
    """Run the model on the currently active photo."""
    idx = session["active"]
    image = session["images"][idx]

    if image is None:
        return session, None, "No photo loaded for this slot."

    logger.info(f"Running model on {PHOTO_LABELS[idx]} photo...")
    alpha = run_model(image, device=device)
    session["alpha_masks"][idx] = alpha

    # Preserve existing strokes — don't reset them on re-run
    _ensure_strokes(session, idx)

    canvas = _render_canvas(session, threshold, feather)
    return session, canvas, _status_text(session)


def handle_run_all(session: dict, threshold: float, feather: int, device: str) -> tuple:
    """Run the model on all uploaded photos sequentially."""
    for idx in range(3):
        if session["images"][idx] is not None:
            session["active"] = idx
            alpha = run_model(session["images"][idx], device=device)
            session["alpha_masks"][idx] = alpha
            _ensure_strokes(session, idx)

    canvas = _render_canvas(session, threshold, feather)
    return session, canvas, _status_text(session)


def handle_paint(
    session: dict,
    evt: gr.SelectData,
    brush_mode: str,
    brush_size: int,
    threshold: float,
    feather: int,
) -> tuple:
    """
    Handle a click/drag on the canvas image.
    Gradio SelectData gives us the pixel coordinates of the click.
    """
    idx = session["active"]
    if session["images"][idx] is None or session["alpha_masks"][idx] is None:
        return session, _render_canvas(session, threshold, feather)

    strokes = _ensure_strokes(session, idx)
    # evt.index is (x, y) for Image components
    cx, cy = int(evt.index[0]), int(evt.index[1])
    strokes.paint(cx, cy, radius=brush_size, mode=brush_mode)

    canvas = _render_canvas(session, threshold, feather)
    return session, canvas


def handle_clear_strokes(session: dict, threshold: float, feather: int) -> tuple:
    idx = session["active"]
    strokes = session["strokes"][idx]
    if strokes is not None:
        strokes.clear()
    canvas = _render_canvas(session, threshold, feather)
    return session, canvas, _status_text(session)


def handle_switch_photo(session: dict, slot_idx: int, threshold: float, feather: int) -> tuple:
    session["active"] = slot_idx
    canvas = _render_canvas(session, threshold, feather)
    return session, canvas, _status_text(session)


def handle_threshold_change(session: dict, threshold: float, feather: int) -> Image.Image:
    return _render_canvas(session, threshold, feather)


def handle_export(session: dict, threshold: float, feather: int) -> list[str]:
    """Export all completed photos as PNG files with transparency."""
    paths = []
    import tempfile, os

    for idx in range(3):
        image = session["images"][idx]
        alpha = session["alpha_masks"][idx]
        if image is None or alpha is None:
            continue

        strokes = session["strokes"][idx] or StrokeLayer(image.height, image.width)
        final_mask = merge(alpha, strokes, threshold=threshold, feather_px=feather)
        rgba = apply_alpha_to_image(image, final_mask)

        with tempfile.NamedTemporaryFile(
            suffix=f"_seetwin_{PHOTO_LABELS[idx].lower()}.png",
            delete=False
        ) as f:
            rgba.save(f.name, format="PNG")
            paths.append(f.name)

    return paths


# ---------------------------------------------------------------------------
# Build the Gradio tab
# ---------------------------------------------------------------------------

def build_stage1_tab(device: str = "cpu") -> gr.Tab:
    """
    Returns a gr.Tab component containing the full Stage 1 UI.
    Call this from app.py and mount it in a gr.Blocks + gr.Tabs context.
    """

    with gr.Tab("1 — Background removal") as tab:
        session_state = gr.State(_empty_session())

        with gr.Row():
            # ── Left column: uploads ────────────────────────────────────────
            with gr.Column(scale=2, min_width=200):
                gr.Markdown("### Input photos")
                upload_front = gr.File(
                    label="Front (T-pose)", file_types=["image"], type="filepath"
                )
                upload_side = gr.File(
                    label="Side (T-pose)", file_types=["image"], type="filepath"
                )
                upload_back = gr.File(
                    label="Back (T-pose)", file_types=["image"], type="filepath"
                )

                gr.Markdown("**Active photo**")
                with gr.Row():
                    btn_front = gr.Button("Front", size="sm")
                    btn_side  = gr.Button("Side",  size="sm")
                    btn_back  = gr.Button("Back",  size="sm")

            # ── Centre: canvas ───────────────────────────────────────────────
            with gr.Column(scale=5):
                canvas = gr.Image(
                    label="Canvas (click to paint)",
                    type="pil",
                    interactive=False,
                    height=520,
                )

            # ── Right column: tools ─────────────────────────────────────────
            with gr.Column(scale=2, min_width=180):
                gr.Markdown("### Brush mode")
                brush_mode = gr.Radio(
                    choices=["fg", "bg"],
                    value="fg",
                    label="",
                    info="fg = keep (green)  ·  bg = remove (red)",
                )

                gr.Markdown("### Brush size")
                brush_size = gr.Slider(
                    minimum=4, maximum=60, value=16, step=4, label="Radius (px)"
                )

                gr.Markdown("### Threshold")
                threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=DEFAULT_THRESHOLD,
                    step=0.01, label="Edge confidence cutoff"
                )

                gr.Markdown("### Feather")
                feather = gr.Slider(
                    minimum=0, maximum=20, value=DEFAULT_FEATHER,
                    step=1, label="Edge softness (px)"
                )

                gr.Markdown("---")
                btn_run     = gr.Button("▶ Run model (this photo)", variant="primary")
                btn_run_all = gr.Button("▶ Run all 3 photos")
                btn_clear   = gr.Button("Clear strokes")
                btn_export  = gr.Button("Export PNGs", variant="secondary")
                export_files = gr.Files(label="Downloads", visible=False)

        # Status bar
        status = gr.Markdown("Upload photos and click **Run model** to begin.")

        # ── Wire up events ────────────────────────────────────────────────────

        # Uploads
        for slot_idx, upload_widget in enumerate(
            [upload_front, upload_side, upload_back]
        ):
            upload_widget.change(
                fn=lambda f, s, si=slot_idx: handle_upload(f, si, s),
                inputs=[upload_widget, session_state],
                outputs=[session_state, canvas, status],
            )

        # Photo switcher buttons
        btn_front.click(
            fn=lambda s, t, f: handle_switch_photo(s, 0, t, f),
            inputs=[session_state, threshold, feather],
            outputs=[session_state, canvas, status],
        )
        btn_side.click(
            fn=lambda s, t, f: handle_switch_photo(s, 1, t, f),
            inputs=[session_state, threshold, feather],
            outputs=[session_state, canvas, status],
        )
        btn_back.click(
            fn=lambda s, t, f: handle_switch_photo(s, 2, t, f),
            inputs=[session_state, threshold, feather],
            outputs=[session_state, canvas, status],
        )

        # Model run
        btn_run.click(
            fn=lambda s, t, fe: handle_run_model(s, t, fe, device),
            inputs=[session_state, threshold, feather],
            outputs=[session_state, canvas, status],
        )
        btn_run_all.click(
            fn=lambda s, t, fe: handle_run_all(s, t, fe, device),
            inputs=[session_state, threshold, feather],
            outputs=[session_state, canvas, status],
        )

        # Brush painting — fires on canvas click
        canvas.select(
            fn=handle_paint,
            inputs=[session_state, brush_mode, brush_size, threshold, feather],
            outputs=[session_state, canvas],
        )

        # Slider live update
        threshold.change(
            fn=handle_threshold_change,
            inputs=[session_state, threshold, feather],
            outputs=[canvas],
        )
        feather.change(
            fn=handle_threshold_change,
            inputs=[session_state, threshold, feather],
            outputs=[canvas],
        )

        # Clear strokes
        btn_clear.click(
            fn=handle_clear_strokes,
            inputs=[session_state, threshold, feather],
            outputs=[session_state, canvas, status],
        )

        # Export
        btn_export.click(
            fn=lambda s, t, f: (handle_export(s, t, f), gr.update(visible=True)),
            inputs=[session_state, threshold, feather],
            outputs=[export_files, export_files],
        )

    return tab