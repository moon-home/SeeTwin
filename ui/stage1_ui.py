"""Stage 1 — Background removal UI (Gradio)."""

from __future__ import annotations

import json
import logging
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

from pipeline.stage1_bgremove import (
    StrokeLayer,
    apply_alpha_to_image,
    composite_on_checker,
    merge,
    resize_to_max,
    run_model,
)

MAX_PX = 1024
PHOTO_LABELS = ["Front", "Side", "Back"]
DEFAULT_THRESHOLD = 0.5
DEFAULT_FEATHER = 2

# ---------------------------------------------------------------------------
# Injected JS — transparent canvas overlay for real-time brush painting.
# Strokes accumulate in window.seetwin_strokes and are synced to a hidden
# Gradio Textbox so Python can read them on the next button click.
# The MutationObserver auto-clears the overlay whenever the canvas image
# src changes (i.e. after any Python handler updates the canvas).
# ---------------------------------------------------------------------------
STAGE1_JS = """() => {
  var overlay = null, painting = false;
  window.seetwin_strokes = [];

  function getContainer() {
    return document.querySelector('#seetwin-canvas');
  }

  /* Find natural image dimensions — only used for coordinate scaling.
     We walk up from the container to find the first img with real pixels. */
  function getNatDims() {
    var c = getContainer();
    if (!c) return null;
    var el = c;
    for (var depth = 0; depth < 6; depth++) {
      var imgs = el.querySelectorAll('img');
      for (var i = 0; i < imgs.length; i++) {
        if (imgs[i].naturalWidth > 10) {
          return { w: imgs[i].naturalWidth, h: imgs[i].naturalHeight,
                   img: imgs[i] };
        }
      }
      if (!el.parentElement || el.tagName === 'BODY') break;
      el = el.parentElement;
    }
    return null;
  }

  /* Visual rect of the photo within the container, using the container's
     own bounding rect (always reliable) plus the natural image aspect ratio
     to compute letterbox offsets. */
  function getVisualRect() {
    var c = getContainer();
    if (!c) return null;
    var cRect = c.getBoundingClientRect();
    if (!cRect.width || !cRect.height) return null;
    var nd = getNatDims();
    if (!nd) return null;
    var natR = nd.w / nd.h, cR = cRect.width / cRect.height;
    var dispW, dispH, offX, offY;
    if (natR > cR) {
      dispW = cRect.width;  dispH = cRect.width  / natR;
      offX  = 0;            offY  = (cRect.height - dispH) / 2;
    } else {
      dispH = cRect.height; dispW = cRect.height * natR;
      offY  = 0;            offX  = (cRect.width  - dispW) / 2;
    }
    var top  = cRect.top  + offY;
    var left = cRect.left + offX;
    return { top: top, left: left, right: left + dispW, bottom: top + dispH,
             width: dispW, height: dispH, natW: nd.w, natH: nd.h,
             img: nd.img };
  }

  function getBrushRadius() {
    var el = document.querySelector('#seetwin-radius input[type=range]');
    return el ? parseFloat(el.value) : 24;
  }
  function getBrushMode() {
    var rs = document.querySelectorAll('#seetwin-mode input[type=radio]');
    for (var i = 0; i < rs.length; i++) {
      if (rs[i].checked) {
        var lbl = rs[i].closest('label');
        var txt = lbl ? (lbl.textContent || lbl.innerText) : '';
        return txt.toLowerCase().indexOf('keep') >= 0 ? 'fg' : 'bg';
      }
    }
    return 'bg';
  }

  function setupOverlay() {
    var c = getContainer();
    if (!c) { setTimeout(setupOverlay, 600); return; }
    var cRect = c.getBoundingClientRect();
    if (!cRect.width) { setTimeout(setupOverlay, 600); return; }
    var nd = getNatDims();
    if (!nd) { setTimeout(setupOverlay, 600); return; }

    if (overlay) { overlay.remove(); overlay = null; }

    overlay = document.createElement('canvas');
    overlay.id = 'seetwin-overlay';
    overlay.style.cssText = (
      'position:fixed;z-index:9999;pointer-events:none;' +
      'top:'    + cRect.top    + 'px;' +
      'left:'   + cRect.left   + 'px;' +
      'width:'  + cRect.width  + 'px;' +
      'height:' + cRect.height + 'px;'
    );
    overlay.width  = nd.w;
    overlay.height = nd.h;
    document.body.appendChild(overlay);

    nd.img.draggable = false;
  }

  function repositionOverlay() {
    if (!overlay) return;
    var c = getContainer();
    if (!c) return;
    var cRect = c.getBoundingClientRect();
    overlay.style.top    = cRect.top    + 'px';
    overlay.style.left   = cRect.left   + 'px';
    overlay.style.width  = cRect.width  + 'px';
    overlay.style.height = cRect.height + 'px';
  }
  window.addEventListener('scroll', repositionOverlay, true);
  window.addEventListener('resize', repositionOverlay);

  /* Capture-phase listeners — fire before any Gradio handler. */
  document.addEventListener('mousedown', function (e) {
    var r = getVisualRect();
    if (!r || e.clientX < r.left || e.clientX > r.right ||
               e.clientY < r.top  || e.clientY > r.bottom) return;
    e.preventDefault();
    e.stopPropagation();
    painting = true;
    paint(e);
  }, true);

  document.addEventListener('mousemove', function (e) {
    if (painting) paint(e);
  });

  document.addEventListener('mouseup', function () {
    if (painting) { painting = false; syncBox(); }
  });

  function paint(e) {
    var r = getVisualRect();
    if (!r) return;
    var sx = r.natW / r.width;
    var sy = r.natH / r.height;
    var x  = Math.round((e.clientX - r.left) * sx);
    var y  = Math.round((e.clientY - r.top)  * sy);
    if (x < 0 || y < 0 || x >= r.natW || y >= r.natH) return;
    var br   = Math.round(getBrushRadius());
    var mode = getBrushMode();

    /* Always record — strokes reach Python even if overlay is missing */
    window.seetwin_strokes.push({ x: x, y: y, r: br, mode: mode });

    /* Draw visually only if overlay canvas exists */
    if (overlay) {
      var color = (mode === 'fg') ? '#1EC878' : '#DC3C3C';
      var ctx = overlay.getContext('2d');
      ctx.globalAlpha = 0.65;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, br, 0, 2 * Math.PI); ctx.fill();
    }
  }

  function syncBox() {
    var tb = document.querySelector('#seetwin-strokes-box textarea');
    if (!tb) return;
    var setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(tb, JSON.stringify(window.seetwin_strokes));
    tb.dispatchEvent(new Event('input', { bubbles: true }));
  }

  window.seetwin_clear = function () {
    window.seetwin_strokes = [];
    if (overlay) {
      overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
    }
    syncBox();
  };

  /* Watch only the img's src attribute — avoids firing on every Gradio
     internal DOM update that would destroy/recreate the overlay in a loop. */
  function startWatcher() {
    var imgs = document.querySelectorAll('#seetwin-canvas img');
    if (!imgs.length) { setTimeout(startWatcher, 500); return; }
    new MutationObserver(function () {
      if (overlay) { overlay.remove(); overlay = null; }
      setTimeout(setupOverlay, 400);
    }).observe(imgs[0], { attributes: true, attributeFilter: ['src'] });
  }

  setTimeout(setupOverlay, 1500);
  setTimeout(startWatcher,  1500);
}
"""

_STYLE = """<style>
.st1-label { font-size: 0.78rem; color: #aaa; text-transform: uppercase;
             letter-spacing: .05em; margin: 8px 0 2px; }
#seetwin-mode label:first-of-type span { color: #1EC878 !important; font-weight: 600; }
#seetwin-mode label:last-of-type  span { color: #DC3C3C !important; font-weight: 600; }
/* Crosshair cursor + no native drag on the canvas image */
#seetwin-canvas img, #seetwin-canvas ~ * img { cursor: crosshair !important; }
#seetwin-canvas img { -webkit-user-drag: none; user-drag: none; }
</style>"""


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _empty_session() -> dict:
    return {"images": [None]*3, "alpha_masks": [None]*3,
            "strokes": [None]*3, "active": 0}

def _ensure_strokes(session, idx):
    if session["strokes"][idx] is None and session["images"][idx] is not None:
        img = session["images"][idx]
        session["strokes"][idx] = StrokeLayer(img.height, img.width)
    return session["strokes"][idx]

def _apply_json_strokes(strokes_json: str, session: dict) -> None:
    try:
        items = json.loads(strokes_json or "[]")
    except Exception:
        return
    if not items:
        return
    idx = session["active"]
    if session["images"][idx] is None:
        return
    sl = _ensure_strokes(session, idx)
    for s in items:
        sl.paint(int(s["x"]), int(s["y"]), radius=int(s["r"]), mode=s["mode"])
    logger.info("Applied %d JS strokes to slot %d", len(items), idx)


# ---------------------------------------------------------------------------
# Canvas rendering  (no Python-side stroke overlay — JS handles that)
# ---------------------------------------------------------------------------

def _render_canvas(session, threshold, feather) -> Optional[Image.Image]:
    idx    = session["active"]
    image  = session["images"][idx]
    alpha  = session["alpha_masks"][idx]
    strokes = session["strokes"][idx]

    if image is None:
        return None

    if alpha is None:
        # Before model runs: treat full image as foreground, apply any strokes
        base = np.ones((image.height, image.width), dtype=np.float32)
        sl   = strokes or StrokeLayer(image.height, image.width)
        mask = merge(base, sl, threshold=0.5, feather_px=0)
    else:
        sl   = strokes or StrokeLayer(image.height, image.width)
        mask = merge(alpha, sl, threshold=threshold, feather_px=feather)

    return composite_on_checker(apply_alpha_to_image(image, mask))

def _status(session) -> str:
    idx     = session["active"]
    alpha   = session["alpha_masks"][idx]
    strokes = session["strokes"][idx]
    if alpha is None:
        return "Paint then click **Apply strokes** — or run the model first"
    px = strokes.stroke_count() if strokes else 0
    parts = [f"Photo: {PHOTO_LABELS[idx]}",
             f"{px:,} stroke px" if px else "no corrections"]
    ready = all(m is not None for m in session["alpha_masks"])
    parts.append("✓ all 3 ready" if ready else "waiting for all 3")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_upload(path, slot, session, threshold, feather):
    if path is None:
        return session, _render_canvas(session, threshold, feather), _status(session)
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = resize_to_max(img, MAX_PX)
    session["images"][slot]      = img
    session["alpha_masks"][slot] = None
    session["strokes"][slot]     = None
    session["active"]            = slot
    return session, _render_canvas(session, threshold, feather), _status(session)

def handle_run_model(strokes_json, session, threshold, feather, device):
    _apply_json_strokes(strokes_json, session)
    idx = session["active"]
    if session["images"][idx] is None:
        return session, _render_canvas(session, threshold, feather), "No photo loaded."
    logger.info("Running model on %s…", PHOTO_LABELS[idx])
    session["alpha_masks"][idx] = run_model(session["images"][idx], device=device)
    _ensure_strokes(session, idx)
    return session, _render_canvas(session, threshold, feather), _status(session)

def handle_run_all(strokes_json, session, threshold, feather, device):
    _apply_json_strokes(strokes_json, session)
    for idx in range(3):
        if session["images"][idx] is not None:
            session["active"] = idx
            session["alpha_masks"][idx] = run_model(session["images"][idx], device=device)
            _ensure_strokes(session, idx)
    return session, _render_canvas(session, threshold, feather), _status(session)

def handle_apply(strokes_json, session, threshold, feather):
    _apply_json_strokes(strokes_json, session)
    return session, _render_canvas(session, threshold, feather), _status(session)

def handle_clear(session, threshold, feather):
    idx = session["active"]
    if session["strokes"][idx]:
        session["strokes"][idx].clear()
    return session, _render_canvas(session, threshold, feather), _status(session)

def handle_switch(strokes_json, session, slot, threshold, feather):
    _apply_json_strokes(strokes_json, session)
    session["active"] = slot
    return session, _render_canvas(session, threshold, feather), _status(session)

def handle_threshold(session, threshold, feather):
    return _render_canvas(session, threshold, feather)

def handle_export(session, threshold, feather):
    import tempfile
    paths = []
    for idx in range(3):
        img   = session["images"][idx]
        alpha = session["alpha_masks"][idx]
        if img is None or alpha is None:
            continue
        sl   = session["strokes"][idx] or StrokeLayer(img.height, img.width)
        mask = merge(alpha, sl, threshold=threshold, feather_px=feather)
        rgba = apply_alpha_to_image(img, mask)
        with tempfile.NamedTemporaryFile(
                suffix=f"_seetwin_{PHOTO_LABELS[idx].lower()}.png", delete=False) as f:
            rgba.save(f.name, "PNG")
            paths.append(f.name)
    return paths


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_stage1_tab(device: str = "cpu") -> gr.Tab:
    with gr.Tab("1 — Background removal") as tab:
        session_state = gr.State(_empty_session())

        # Inject CSS only — JS runs via gr.Blocks(js=STAGE1_JS) in app.py
        gr.HTML(_STYLE)

        with gr.Row():
            # ── Left: uploads ───────────────────────────────────────────────
            with gr.Column(scale=2, min_width=200):
                gr.Markdown("### Input photos")
                upload_front = gr.File(label="Front (T-pose)", file_types=["image"], type="filepath")
                upload_side  = gr.File(label="Side (T-pose)",  file_types=["image"], type="filepath")
                upload_back  = gr.File(label="Back (T-pose)",  file_types=["image"], type="filepath")
                gr.HTML('<p class="st1-label">Active photo</p>')
                with gr.Row():
                    btn_front = gr.Button("Front", size="sm")
                    btn_side  = gr.Button("Side",  size="sm")
                    btn_back  = gr.Button("Back",  size="sm")

            # ── Centre: canvas only ──────────────────────────────────────────
            with gr.Column(scale=5):
                canvas = gr.Image(
                    label="Canvas",
                    interactive=False,
                    type="pil",
                    height=520,
                    show_download_button=False,
                    elem_id="seetwin-canvas",
                )
                # Hidden textbox — JS writes stroke JSON here on each stroke end
                strokes_box = gr.Textbox(
                    value="[]", visible=False, elem_id="seetwin-strokes-box"
                )

            # ── Right: brush + sliders + actions ────────────────────────────
            with gr.Column(scale=2, min_width=180):
                gr.HTML('<p class="st1-label">Brush mode</p>')
                mode_radio = gr.Radio(
                    choices=["🟢 Keep", "🔴 Remove"],
                    value="🔴 Remove",
                    label="",
                    elem_id="seetwin-mode",
                    interactive=True,
                )
                gr.HTML('<p class="st1-label">Brush radius (px)</p>')
                brush_slider = gr.Slider(
                    8, 80, value=24, step=4,
                    label="",
                    elem_id="seetwin-radius",
                    interactive=True,
                )
                gr.HTML('<hr style="border-color:#444;margin:8px 0">')
                gr.HTML('<p class="st1-label">Threshold</p>')
                threshold = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD,
                                      step=0.01, label="",
                                      show_label=False)
                gr.HTML('<p class="st1-label">Feather</p>')
                feather = gr.Slider(0, 20, value=DEFAULT_FEATHER,
                                    step=1, label="",
                                    show_label=False)
                gr.HTML('<hr style="border-color:#444;margin:8px 0">')
                btn_apply   = gr.Button("✓ Apply strokes",          variant="primary")
                btn_run     = gr.Button("▶ Run model (this photo)", variant="secondary")
                btn_run_all = gr.Button("▶ Run all 3 photos",       variant="secondary")
                btn_clear   = gr.Button("↺ Reset corrections",      variant="secondary")
                btn_export  = gr.Button("Export PNGs",              variant="secondary")
                export_files = gr.Files(label="Downloads", visible=False)

        status = gr.Markdown(
            "Upload a photo. Paint green to keep, red to remove. Click **Apply strokes**."
        )

        # ── Wiring ─────────────────────────────────────────────────────────

        def _upload_and_run(path, session, slot):
            session, img, st = handle_upload(path, slot, session,
                                             DEFAULT_THRESHOLD, DEFAULT_FEATHER)
            if path is not None:
                session, img, st = handle_run_model(
                    "[]", session, DEFAULT_THRESHOLD, DEFAULT_FEATHER, device)
            return session, img, st

        for si, w in enumerate([upload_front, upload_side, upload_back]):
            w.change(fn=lambda f, s, i=si: _upload_and_run(f, s, i),
                     inputs=[w, session_state],
                     outputs=[session_state, canvas, status])

        _clear_js = "() => { if (window.seetwin_clear) window.seetwin_clear(); }"

        btn_front.click(fn=lambda sb,s,t,f: handle_switch(sb,s,0,t,f),
                        inputs=[strokes_box, session_state, threshold, feather],
                        outputs=[session_state, canvas, status]).then(fn=None, js=_clear_js)
        btn_side.click( fn=lambda sb,s,t,f: handle_switch(sb,s,1,t,f),
                        inputs=[strokes_box, session_state, threshold, feather],
                        outputs=[session_state, canvas, status]).then(fn=None, js=_clear_js)
        btn_back.click( fn=lambda sb,s,t,f: handle_switch(sb,s,2,t,f),
                        inputs=[strokes_box, session_state, threshold, feather],
                        outputs=[session_state, canvas, status]).then(fn=None, js=_clear_js)

        btn_apply.click(fn=handle_apply,
                        inputs=[strokes_box, session_state, threshold, feather],
                        outputs=[session_state, canvas, status]).then(fn=None, js=_clear_js)

        btn_run.click(fn=lambda sb,s,t,fe: handle_run_model(sb,s,t,fe,device),
                      inputs=[strokes_box, session_state, threshold, feather],
                      outputs=[session_state, canvas, status]).then(fn=None, js=_clear_js)

        btn_run_all.click(fn=lambda sb,s,t,fe: handle_run_all(sb,s,t,fe,device),
                          inputs=[strokes_box, session_state, threshold, feather],
                          outputs=[session_state, canvas, status]).then(fn=None, js=_clear_js)

        btn_clear.click(fn=handle_clear,
                        inputs=[session_state, threshold, feather],
                        outputs=[session_state, canvas, status]).then(fn=None, js=_clear_js)

        threshold.change(fn=handle_threshold,
                         inputs=[session_state, threshold, feather],
                         outputs=[canvas])
        feather.change(fn=handle_threshold,
                       inputs=[session_state, threshold, feather],
                       outputs=[canvas])

        btn_export.click(
            fn=lambda s,t,f: (handle_export(s,t,f), gr.update(visible=True)),
            inputs=[session_state, threshold, feather],
            outputs=[export_files, export_files])

    return tab
