"""Stage 2 — Body shape estimation UI (Gradio)."""

from __future__ import annotations

import logging
import numpy as np
import gradio as gr
from PIL import Image

from pipeline.stage2_body_shape import (
    extract_keypoints, draw_landmarks,
    fit_body_shape, beta_to_measurements,
)
from pipeline.stage2_body_shape.keypoint_extractor import LANDMARK_NAMES

logger = logging.getLogger(__name__)

NUM_BETAS = 10

BETA_LABELS = [
    "β0 — overall size",
    "β1 — height vs weight",
    "β2 — torso width",
    "β3 — limb length",
    "β4 — shoulder breadth",
    "β5 — hip breadth",
    "β6 — belly",
    "β7 — leg shape",
    "β8 — arm thickness",
    "β9 — upper/lower ratio",
]

# Dropdown choices: (display label, integer value)
_LM_CHOICES = [("(select landmark to correct)", -1)] + [
    (f"[{idx:02d}] {name}", idx) for idx, name in LANDMARK_NAMES.items()
]

_STYLE = """<style>
.st2-label { font-size: 0.78rem; color: #aaa; text-transform: uppercase;
             letter-spacing: .05em; margin: 8px 0 2px; }
.st2-meas-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.st2-meas-table td, .st2-meas-table th {
    padding: 4px 10px; border-bottom: 1px solid #333; }
.st2-meas-table th { color: #aaa; font-weight: normal; text-align: left; }
.st2-meas-table td:last-child { text-align: right; font-weight: 600;
    color: #1EC878; }
.st2-beta-note { font-size: 0.78rem; color: #888; margin: 4px 0 10px;
    border-left: 3px solid #444; padding-left: 8px; line-height: 1.5; }
</style>"""


def _measurements_html(m: dict) -> str:
    rows = "".join(
        f"<tr><td>{k.replace('_', ' ').replace(' cm', '')}</td>"
        f"<td>{v} cm</td></tr>"
        for k, v in m.items()
    )
    return (
        '<table class="st2-meas-table">'
        "<tr><th>Measurement</th><th>Estimated</th></tr>"
        f"{rows}</table>"
    )


def _empty_state() -> dict:
    return {
        "kp_front": None, "kp_side": None, "kp_back": None,
        "img_front": None, "img_side": None, "img_back": None,
        "beta": None,
        "img_w": 512, "img_h": 768,
    }


def _render_ann(state: dict, view: str, selected_idx: int = -1) -> Image.Image | None:
    img = state.get(f"img_{view}")
    kp  = state.get(f"kp_{view}")
    if img is None or kp is None:
        return None
    return draw_landmarks(img, kp, selected_idx=selected_idx)


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_extract(img_front, img_side, img_back, state: dict):
    """Extract MediaPipe keypoints from uploaded photos."""
    photos = {"front": img_front, "side": img_side, "back": img_back}

    for view, img in photos.items():
        state[f"img_{view}"] = img  # store original (may be RGBA from Stage 1)
        if img is None:
            state[f"kp_{view}"] = None
            continue
        kp = extract_keypoints(img)
        state[f"kp_{view}"] = kp
        logger.info("Keypoints %s: detected=%s", view, kp.get("detected"))

    if img_front is not None:
        state["img_w"], state["img_h"] = img_front.size

    detected = sum(
        1 for v in ["front", "side", "back"]
        if state.get(f"kp_{v}") and state[f"kp_{v}"].get("detected")
    )
    status = (f"Keypoints extracted — {detected}/3 views detected."
              if detected > 0 else
              "No person detected in any photo. Check photo quality.")

    ann_f = _render_ann(state, "front")
    ann_s = _render_ann(state, "side")
    ann_b = _render_ann(state, "back")
    return state, ann_f, ann_s, ann_b, status


def handle_fit(state: dict):
    """Run SMPL-X fitting and populate β sliders + measurements."""
    beta = fit_body_shape(
        kp_front=state.get("kp_front"),
        kp_side=state.get("kp_side"),
        kp_back=state.get("kp_back"),
        img_w=state.get("img_w", 512),
        img_h=state.get("img_h", 768),
    )
    state["beta"] = beta.tolist()
    measurements = beta_to_measurements(beta)
    logger.info("Fitting complete. Measurements: %s", measurements)
    return [state, _measurements_html(measurements), gr.update(visible=True)] + list(beta)


def handle_beta_update(state: dict, *slider_vals):
    """Recompute measurements when β sliders change. Landmarks are unaffected
    because they reflect MediaPipe detections, not the 3D model pose."""
    beta = np.array(slider_vals, dtype=np.float32)
    state["beta"] = beta.tolist()
    return state, _measurements_html(beta_to_measurements(beta))


def handle_lm_select(state: dict, lm_idx):
    """Highlight the selected landmark (white dot) in all three views."""
    idx = int(lm_idx) if lm_idx is not None else -1
    ann_f = _render_ann(state, "front", selected_idx=idx)
    ann_s = _render_ann(state, "side",  selected_idx=idx)
    ann_b = _render_ann(state, "back",  selected_idx=idx)

    if idx < 0:
        msg = "Select a landmark from the dropdown, then click on an annotated image to reposition it."
    else:
        name = LANDMARK_NAMES.get(idx, str(idx))
        msg = f"**{name}** selected (white dot). Click anywhere on an annotated image to move it there."

    return (ann_f if ann_f is not None else gr.update(),
            ann_s if ann_s is not None else gr.update(),
            ann_b if ann_b is not None else gr.update(),
            msg)


def _do_landmark_click(evt: gr.SelectData, state: dict, view: str, lm_idx):
    """Move the chosen landmark to the pixel the user clicked."""
    idx = int(lm_idx) if lm_idx is not None else -1
    if idx < 0:
        return (state,
                gr.update(), gr.update(), gr.update(),
                "Select a landmark from the dropdown first, then click to place it.")

    kp  = state.get(f"kp_{view}")
    img = state.get(f"img_{view}")
    if not kp or not kp.get("detected") or img is None:
        return (state,
                gr.update(), gr.update(), gr.update(),
                f"No keypoints detected for the {view} view.")

    # evt.index is (x, y) in original image pixel coordinates
    x_norm = evt.index[0] / img.width
    y_norm = evt.index[1] / img.height
    kp["landmarks"][idx]["x"] = x_norm
    kp["landmarks"][idx]["y"] = y_norm
    kp["landmarks"][idx]["visibility"] = 1.0  # trust manual correction
    state[f"kp_{view}"] = kp

    ann_f = _render_ann(state, "front", selected_idx=idx)
    ann_s = _render_ann(state, "side",  selected_idx=idx)
    ann_b = _render_ann(state, "back",  selected_idx=idx)

    name = LANDMARK_NAMES.get(idx, str(idx))
    return (state,
            ann_f if ann_f is not None else gr.update(),
            ann_s if ann_s is not None else gr.update(),
            ann_b if ann_b is not None else gr.update(),
            f"Moved **{name}** in {view} view → ({x_norm:.3f}, {y_norm:.3f}).")


def handle_confirm(state: dict):
    beta = state.get("beta")
    if beta is None:
        return state, "Run body shape fitting first."
    return state, f"✓ Body shape confirmed. β = {[round(b, 3) for b in beta]}"


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_stage2_tab(stage1_state: gr.State | None = None) -> gr.Tab:
    with gr.Tab("2 — Body shape") as tab:
        state = gr.State(_empty_state())
        gr.HTML(_STYLE)

        with gr.Row():
            # ── Left: photo inputs ─────────────────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown("### Input photos")
                gr.HTML('<p class="st2-label">Transparent PNGs from Stage 1 (or upload directly)</p>')
                img_front = gr.Image(label="Front", type="pil", height=260)
                img_side  = gr.Image(label="Side",  type="pil", height=260)
                img_back  = gr.Image(label="Back",  type="pil", height=260)
                btn_extract = gr.Button("1 — Extract keypoints", variant="secondary")

            # ── Centre: annotated landmark views ───────────────────────────
            with gr.Column(scale=4):
                gr.Markdown("### Landmark overlay")
                gr.HTML(
                    '<p style="font-size:0.82rem;color:#aaa;margin:2px 0 10px">'
                    "Green = left side · Red = right side · Yellow = landmark<br>"
                    "Select a landmark below, then click an annotated view to reposition it."
                    "</p>"
                )
                ann_front = gr.Image(label="Front", type="pil", height=300,
                                     show_label=True, interactive=False)
                ann_side  = gr.Image(label="Side",  type="pil", height=300,
                                     show_label=True, interactive=False)
                ann_back  = gr.Image(label="Back",  type="pil", height=300,
                                     show_label=True, interactive=False)

                gr.HTML('<hr style="border-color:#333;margin:12px 0 8px">')
                gr.HTML('<p class="st2-label">Correct a landmark</p>')
                selected_lm = gr.Dropdown(
                    choices=_LM_CHOICES,
                    value=-1,
                    label="Landmark to reposition",
                )
                gr.HTML(
                    '<p style="font-size:0.8rem;color:#888;margin:4px 0 10px">'
                    "After selecting a landmark it will highlight in white. "
                    "Click anywhere on the annotated image above to move it."
                    "</p>"
                )
                btn_refit = gr.Button("Re-fit with corrected landmarks", variant="secondary")

            # ── Right: fit controls + measurements ─────────────────────────
            with gr.Column(scale=3):
                gr.Markdown("### Body shape")
                btn_fit = gr.Button("2 — Fit body shape (SMPL-X)", variant="primary")
                gr.HTML('<p class="st2-label">Estimated measurements</p>')
                meas_html = gr.HTML(
                    "<p style='color:#666'>Run fitting to see measurements.</p>"
                )

                with gr.Group(visible=False) as beta_group:
                    gr.HTML('<p class="st2-label">Manual β correction</p>')
                    gr.HTML(
                        '<p class="st2-beta-note">'
                        "β controls the 3D SMPL-X body shape — measurements update live "
                        "as you drag these sliders.<br><br>"
                        "The landmark overlay positions come from MediaPipe photo detection "
                        "and do <em>not</em> change with β. "
                        "To correct wrong landmarks, use the <strong>Correct a landmark</strong> "
                        "tool in the centre panel, then click "
                        "<strong>Re-fit with corrected landmarks</strong>."
                        "</p>"
                    )
                    beta_sliders = [
                        gr.Slider(-3.0, 3.0, value=0.0, step=0.05,
                                  label=BETA_LABELS[i], show_label=True)
                        for i in range(NUM_BETAS)
                    ]

                gr.HTML('<hr style="border-color:#444;margin:12px 0">')
                btn_confirm = gr.Button("3 — Confirm & continue", variant="primary")

        status = gr.Markdown(
            "Upload photos (or load from Stage 1 export), then click **Extract keypoints**."
        )

        # ── Wiring ────────────────────────────────────────────────────────

        btn_extract.click(
            fn=handle_extract,
            inputs=[img_front, img_side, img_back, state],
            outputs=[state, ann_front, ann_side, ann_back, status],
        )

        btn_fit.click(
            fn=handle_fit,
            inputs=[state],
            outputs=[state, meas_html, beta_group] + beta_sliders,
        )

        # Landmark dropdown → highlight in all views
        selected_lm.change(
            fn=handle_lm_select,
            inputs=[state, selected_lm],
            outputs=[ann_front, ann_side, ann_back, status],
        )

        # Click on annotated image → move selected landmark
        def click_front(evt: gr.SelectData, s, lm):
            return _do_landmark_click(evt, s, "front", lm)

        def click_side(evt: gr.SelectData, s, lm):
            return _do_landmark_click(evt, s, "side", lm)

        def click_back(evt: gr.SelectData, s, lm):
            return _do_landmark_click(evt, s, "back", lm)

        ann_front.select(
            fn=click_front,
            inputs=[state, selected_lm],
            outputs=[state, ann_front, ann_side, ann_back, status],
        )
        ann_side.select(
            fn=click_side,
            inputs=[state, selected_lm],
            outputs=[state, ann_front, ann_side, ann_back, status],
        )
        ann_back.select(
            fn=click_back,
            inputs=[state, selected_lm],
            outputs=[state, ann_front, ann_side, ann_back, status],
        )

        # Re-fit with (possibly corrected) keypoints
        btn_refit.click(
            fn=handle_fit,
            inputs=[state],
            outputs=[state, meas_html, beta_group] + beta_sliders,
        )

        # Live β adjustment → measurements only
        for sl in beta_sliders:
            sl.change(
                fn=handle_beta_update,
                inputs=[state] + beta_sliders,
                outputs=[state, meas_html],
            )

        btn_confirm.click(
            fn=handle_confirm,
            inputs=[state],
            outputs=[state, status],
        )

    return tab
