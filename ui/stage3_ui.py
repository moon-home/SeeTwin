"""
Stage 3 — Classification & Confirmation UI.

Flow:
  1. Receive front photo + beta from Stage 2 session OR upload seetwin_body_*.json + photo
  2. "Run classification" → CLIP zero-shot over hair/face/garments/accessories
  3. Each section shows a photo crop alongside top predictions; user adjusts via radio
  4. Color pickers let user correct the sampled garment/hair colors
  5. "Generate manifest" → assembles and downloads the JSON asset manifest

Note: only the front photo is used for classification. Side/back are not needed here
(they're used in Stage 2 for body-shape fitting and in Stage 4 for texture extraction).
"""

from __future__ import annotations
import json
import tempfile

import gradio as gr
import numpy as np
from PIL import Image

from pipeline.stage3_classification.classifier import (
    classify_all, CONFIDENCE_THRESHOLD,
)
from pipeline.stage3_classification.manifest import build_manifest
from pipeline.stage3_classification.asset_library import GARMENT_FIT_OPTIONS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_badge(conf: float) -> str:
    color = "#4caf50" if conf >= CONFIDENCE_THRESHOLD else "#ff9800" if conf >= 0.5 else "#f44336"
    return f'<span style="color:{color};font-size:0.8rem">({conf:.0%})</span>'


def _pred_html(preds: list[dict]) -> str:
    if not preds:
        return ""
    top = preds[0]
    badge = _confidence_badge(top["confidence"])
    return (
        f"<p style='font-size:0.85rem;color:#ccc;margin:4px 0'>"
        f"Top prediction: <b>{top['label']}</b> {badge}</p>"
    )


def _radio_choices(preds: list[dict]) -> list[str]:
    return [f"{p['label']}  ({p['confidence']:.0%})" for p in preds]


def _selected_id(preds: list[dict], choice: str | None) -> tuple[str, str]:
    if not choice or not preds:
        return (preds[0]["id"] if preds else "none",
                preds[0]["label"] if preds else "none")
    for p in preds:
        if p["label"] in choice:
            return p["id"], p["label"]
    return preds[0]["id"], preds[0]["label"]


def _parse_s2_json(path: str | None) -> tuple[list[float] | None, str]:
    if not path:
        return None, ""
    try:
        with open(path) as f:
            data = json.load(f)
        beta_dict = data.get("beta_parameters", {})
        beta = [beta_dict.get(f"beta_{i}", 0.0) for i in range(10)]
        measurements = data.get("measurements_cm", {})
        rows = "".join(
            f"<tr><td style='padding:2px 8px'>{k}</td>"
            f"<td style='padding:2px 8px;text-align:right'><b>{v:.1f} cm</b></td></tr>"
            for k, v in measurements.items()
        )
        return beta, (
            "<table style='font-size:0.85rem;border-collapse:collapse'>"
            + rows + "</table>"
        )
    except Exception as e:
        return None, f"<p style='color:red'>Failed to parse JSON: {e}</p>"


def _measurements_html(beta: list[float]) -> str:
    try:
        from pipeline.stage2_body_shape.smplx_fitter import beta_to_measurements
        m = beta_to_measurements(np.array(beta, dtype=np.float32))
        rows = "".join(
            f"<tr><td style='padding:2px 8px'>{k}</td>"
            f"<td style='padding:2px 8px;text-align:right'><b>{v:.1f} cm</b></td></tr>"
            for k, v in m.items()
        )
        return (
            "<table style='font-size:0.85rem;border-collapse:collapse'>"
            + rows + "</table>"
        )
    except Exception:
        return ""


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_stage3_tab(
    stage2_state: gr.State | None = None,
    main_tabs: gr.Tabs | None = None,
) -> gr.Tab:

    with gr.Tab("3 — Parts manifest", id="stage3") as tab:

        state = gr.State({})

        gr.Markdown(
            "CLIP zero-shot classification identifies face type, hair style, "
            "garments, and accessories from your **front photo**. "
            "Review each prediction, adjust selections and colors as needed, "
            "then generate the asset manifest."
        )

        # ── Input ─────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Front photo")
                gr.Markdown(
                    "_Auto-populated from Stage 2. "
                    "Re-upload here to use a different photo._"
                )
                photo_front = gr.Image(
                    label="Front (required for classification)",
                    type="pil", height=260, image_mode="RGBA",
                )

            with gr.Column(scale=3):
                gr.Markdown("### Body shape (from Stage 2)")
                body_summary = gr.HTML(
                    "<p style='color:#aaa'>Complete Stage 2 first, "
                    "or upload a <code>seetwin_body_*.json</code> below.</p>"
                )
                gr.Markdown("#### Upload Stage 2 JSON (cross-session)")
                s2_json = gr.File(label="seetwin_body_*.json", file_types=[".json"])
                s2_status = gr.Markdown("")

            with gr.Column(scale=2):
                gr.Markdown("### Run")
                btn_classify = gr.Button(
                    "Run all classifiers", variant="primary", size="lg"
                )
                classify_status = gr.Markdown("")
                gr.HTML(
                    "<p style='font-size:0.8rem;color:#aaa'>"
                    "First run downloads ~350 MB (ViT-B-32 CLIP weights).<br>"
                    "Subsequent runs are fast (~1–2 s on CPU).</p>"
                )

        gr.Markdown("---")

        # ── Results (hidden until classification runs) ─────────────────────────
        results_col = gr.Column(visible=False)
        with results_col:

            # Face ─────────────────────────────────────────────────────────────
            with gr.Accordion("Face type", open=True):
                with gr.Row():
                    face_crop_img = gr.Image(
                        label="Face region crop", height=240, interactive=False,
                    )
                    with gr.Column():
                        face_conf_html = gr.HTML("")
                        face_radio = gr.Radio(choices=[], label="Select face type")

            # Hair ─────────────────────────────────────────────────────────────
            with gr.Accordion("Hair style", open=True):
                with gr.Row():
                    hair_crop_img = gr.Image(
                        label="Hair region crop", height=240, interactive=False,
                    )
                    with gr.Column():
                        hair_conf_html = gr.HTML("")
                        hair_radio = gr.Radio(choices=[], label="Select hair style")
                        hair_color_picker = gr.ColorPicker(
                            label="Hair color (auto-sampled — adjust if wrong)",
                            value="#888888",
                        )

            # Upper garment ────────────────────────────────────────────────────
            with gr.Accordion("Upper body garment", open=True):
                with gr.Row():
                    upper_crop_img = gr.Image(
                        label="Upper body crop", height=240, interactive=False,
                    )
                    with gr.Column():
                        upper_conf_html = gr.HTML("")
                        upper_radio = gr.Radio(choices=[], label="Select upper garment")
                        upper_fit_radio = gr.Radio(
                            choices=GARMENT_FIT_OPTIONS, value="regular", label="Fit",
                        )
                        upper_color_picker = gr.ColorPicker(
                            label="Garment color (auto-sampled — adjust if wrong)",
                            value="#888888",
                        )

            # Lower garment ────────────────────────────────────────────────────
            with gr.Accordion("Lower body garment", open=True):
                with gr.Row():
                    lower_crop_img = gr.Image(
                        label="Lower body crop", height=240, interactive=False,
                    )
                    with gr.Column():
                        lower_conf_html = gr.HTML("")
                        lower_radio = gr.Radio(choices=[], label="Select lower garment")
                        lower_fit_radio = gr.Radio(
                            choices=GARMENT_FIT_OPTIONS, value="regular", label="Fit",
                        )
                        lower_color_picker = gr.ColorPicker(
                            label="Garment color (auto-sampled — adjust if wrong)",
                            value="#888888",
                        )

            # Accessories ──────────────────────────────────────────────────────
            with gr.Accordion("Accessories", open=True):
                with gr.Row():
                    acc_full_img = gr.Image(
                        label="Full photo", height=320, interactive=False,
                    )
                    with gr.Column():
                        acc_info_html = gr.HTML(
                            "<p style='color:#aaa'>No accessories detected.</p>"
                        )
                        acc_checkboxes = gr.CheckboxGroup(
                            choices=[], value=[],
                            label="Detected accessories "
                                  "(uncheck to remove — re-check to add back)",
                        )

            gr.Markdown("---")

            # Manifest ─────────────────────────────────────────────────────────
            btn_manifest = gr.Button(
                "Generate asset manifest", variant="primary", size="lg"
            )
            manifest_status = gr.Markdown("")

            with gr.Row(visible=False) as manifest_row:
                with gr.Column(scale=3):
                    manifest_code = gr.Code(
                        label="Asset manifest (JSON)",
                        language="json",
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    manifest_file = gr.File(
                        label="Download manifest.json",
                        visible=False,
                    )

        # ── Handlers ──────────────────────────────────────────────────────────

        def on_s2_json_upload(path, cur_state):
            beta, html = _parse_s2_json(path)
            if beta is None:
                return cur_state, html, "Upload failed."
            cur_state = dict(cur_state)
            cur_state["beta"] = beta
            return cur_state, html, "✓ Body shape loaded from JSON."

        s2_json.change(
            fn=on_s2_json_upload,
            inputs=[s2_json, state],
            outputs=[state, body_summary, s2_status],
        )

        if stage2_state is not None:
            def _populate_from_s2(s2, s3):
                beta = s2.get("beta")
                s3 = dict(s3)
                if beta:
                    s3["beta"] = beta
                return (
                    s3,
                    _measurements_html(beta) if beta else gr.update(),
                    s2.get("img_front") or gr.update(),
                )

            tab.select(
                fn=_populate_from_s2,
                inputs=[stage2_state, state],
                outputs=[state, body_summary, photo_front],
            )

        def on_classify(img_front, s3_state):
            _no_results = (
                s3_state, gr.update(visible=False),
                None, "", gr.update(choices=[], value=None),
                None, "", gr.update(choices=[], value=None), "#888888",
                None, "", gr.update(choices=[], value=None), "regular", "#888888",
                None, "", gr.update(choices=[], value=None), "regular", "#888888",
                None, "<p style='color:#aaa'>No accessories detected.</p>",
                gr.update(choices=[], value=[]),
            )

            if img_front is None:
                return ("Upload a front photo first.",) + _no_results

            try:
                results = classify_all(img_front)
            except Exception as e:
                return (f"Classification failed: {e}",) + _no_results

            s3 = dict(s3_state)
            s3["results"] = results
            s3["accessories_raw"] = results["accessories"]

            face_preds  = results["face_preds"]
            hair_preds  = results["hair_preds"]
            upper_preds = results["upper_preds"]
            lower_preds = results["lower_preds"]
            accs        = results["accessories"]

            acc_choices = [f"{a['label']}  ({a['confidence']:.0%})" for a in accs]
            if accs:
                rows = "".join(
                    f"<tr>"
                    f"<td style='padding:2px 8px'><b>{a['label']}</b></td>"
                    f"<td style='padding:2px 8px;color:#aaa'>{a['category']}</td>"
                    f"<td style='padding:2px 8px'>{_confidence_badge(a['confidence'])}</td>"
                    f"</tr>"
                    for a in accs
                )
                acc_info = (
                    "<table style='font-size:0.85rem;border-collapse:collapse'>"
                    + rows + "</table>"
                )
            else:
                acc_info = "<p style='color:#aaa;font-size:0.85rem'>No accessories detected above threshold.</p>"

            return (
                "✓ Classification complete. Review each section and adjust if needed.",
                s3,
                gr.update(visible=True),
                # face
                results["face_crop"],
                _pred_html(face_preds),
                gr.update(choices=_radio_choices(face_preds),
                          value=_radio_choices(face_preds)[0] if face_preds else None),
                # hair
                results["hair_crop"],
                _pred_html(hair_preds),
                gr.update(choices=_radio_choices(hair_preds),
                          value=_radio_choices(hair_preds)[0] if hair_preds else None),
                results["hair_color"],   # → hair_color_picker
                # upper garment
                results["upper_crop"],
                _pred_html(upper_preds),
                gr.update(choices=_radio_choices(upper_preds),
                          value=_radio_choices(upper_preds)[0] if upper_preds else None),
                "regular",               # reset fit
                results["upper_color"],  # → upper_color_picker
                # lower garment
                results["lower_crop"],
                _pred_html(lower_preds),
                gr.update(choices=_radio_choices(lower_preds),
                          value=_radio_choices(lower_preds)[0] if lower_preds else None),
                "regular",
                results["lower_color"],  # → lower_color_picker
                # accessories
                img_front,
                acc_info,
                gr.update(choices=acc_choices, value=acc_choices),
            )

        btn_classify.click(
            fn=on_classify,
            inputs=[photo_front, state],
            outputs=[
                classify_status, state, results_col,
                face_crop_img, face_conf_html, face_radio,
                hair_crop_img, hair_conf_html, hair_radio, hair_color_picker,
                upper_crop_img, upper_conf_html, upper_radio, upper_fit_radio, upper_color_picker,
                lower_crop_img, lower_conf_html, lower_radio, lower_fit_radio, lower_color_picker,
                acc_full_img, acc_info_html, acc_checkboxes,
            ],
        )

        def on_generate_manifest(
            s3_state,
            face_choice, hair_choice, hair_color,
            upper_choice, upper_fit, upper_color,
            lower_choice, lower_fit, lower_color,
            acc_checked,
        ):
            results = s3_state.get("results")
            if not results:
                return (
                    s3_state, "Run classification first.",
                    gr.update(visible=False), "", gr.update(visible=False),
                )

            face_id,  face_label  = _selected_id(results["face_preds"],  face_choice)
            hair_id,  hair_label  = _selected_id(results["hair_preds"],  hair_choice)
            upper_id, upper_label = _selected_id(results["upper_preds"], upper_choice)
            lower_id, lower_label = _selected_id(results["lower_preds"], lower_choice)

            accs_raw = s3_state.get("accessories_raw", [])
            confirmed_accs = [
                acc for acc in accs_raw
                if f"{acc['label']}  ({acc['confidence']:.0%})" in (acc_checked or [])
            ]

            manifest = build_manifest(
                beta=s3_state.get("beta"),
                face_id=face_id,   face_label=face_label,
                hair_id=hair_id,   hair_label=hair_label,   hair_color=hair_color or "#888888",
                upper_id=upper_id, upper_label=upper_label, upper_fit=upper_fit or "regular",
                upper_color=upper_color or "#888888",
                lower_id=lower_id, lower_label=lower_label, lower_fit=lower_fit or "regular",
                lower_color=lower_color or "#888888",
                accessories=confirmed_accs,
            )

            manifest_str = json.dumps(manifest, indent=2)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="seetwin_manifest_"
            )
            tmp.write(manifest_str)
            tmp.close()

            s3_state = dict(s3_state)
            s3_state["manifest"] = manifest

            return (
                s3_state,
                "✓ Asset manifest generated.",
                gr.update(visible=True),
                manifest_str,
                gr.update(value=tmp.name, visible=True),
            )

        btn_manifest.click(
            fn=on_generate_manifest,
            inputs=[
                state,
                face_radio,
                hair_radio, hair_color_picker,
                upper_radio, upper_fit_radio, upper_color_picker,
                lower_radio, lower_fit_radio, lower_color_picker,
                acc_checkboxes,
            ],
            outputs=[
                state, manifest_status,
                manifest_row, manifest_code, manifest_file,
            ],
        )

    return tab
