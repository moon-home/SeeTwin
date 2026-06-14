"""Stage 1 — Background removal UI (Gradio)."""

from __future__ import annotations

import json
import logging
import zipfile
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
  var overlay = null, cursorEl = null, cachedImg = null;
  var painting = false;
  var zoom = 1, panX = 0, panY = 0;
  window.seetwin_strokes = [];

  // ── DOM helpers ─────────────────────────────────────────────────────────

  function getContainer() { return document.querySelector('#seetwin-canvas'); }

  function getNatImg() {
    var c = getContainer();
    if (!c) return null;
    var imgs = c.querySelectorAll('img');
    for (var i = 0; i < imgs.length; i++) {
      if (imgs[i].naturalWidth > 10) return imgs[i];
    }
    return null;
  }

  // Letterbox bounds at zoom=1 / pan=0
  function getBaseRect() {
    var c = getContainer();
    if (!c) return null;
    var cr = c.getBoundingClientRect();
    if (!cr.width || !cr.height) return null;
    var img = getNatImg();
    if (!img) return null;
    var nw = img.naturalWidth, nh = img.naturalHeight;
    var natR = nw / nh, cR = cr.width / cr.height;
    var bw, bh;
    if (natR > cR) { bw = cr.width;  bh = cr.width  / natR; }
    else           { bh = cr.height; bw = cr.height * natR; }
    return {
      cr: cr,
      cx: cr.left + cr.width  / 2,
      cy: cr.top  + cr.height / 2,
      bw: bw, bh: bh, nw: nw, nh: nh, img: img
    };
  }

  // Actual image rect in screen coords, with zoom/pan applied
  function getVisualRect() {
    var b = getBaseRect();
    if (!b) return null;
    var w = b.bw * zoom, h = b.bh * zoom;
    var cx = b.cx + panX, cy = b.cy + panY;
    return {
      left: cx - w/2, top: cy - h/2, right: cx + w/2, bottom: cy + h/2,
      width: w, height: h, nw: b.nw, nh: b.nh, cr: b.cr
    };
  }

  // ── Brush helpers ────────────────────────────────────────────────────────

  function getBrushRadius() {
    var el = document.querySelector('#seetwin-radius input[type=range]');
    return el ? parseFloat(el.value) : 10;
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

  // Brush radius in screen pixels at current zoom
  function getScreenRadius() {
    var b = getBaseRect();
    return b ? getBrushRadius() * (b.bw * zoom) / b.nw : 0;
  }

  // ── Cursor circle ────────────────────────────────────────────────────────

  function ensureCursor() {
    cursorEl = document.getElementById('seetwin-cursor');
    if (cursorEl) return;
    cursorEl = document.createElement('div');
    cursorEl.id = 'seetwin-cursor';
    cursorEl.style.cssText = (
      'position:fixed;z-index:10000;pointer-events:none;border-radius:50%;' +
      'border:2px solid #DC3C3C;display:none;' +
      'transform:translate(-50%,-50%);box-sizing:border-box;'
    );
    document.body.appendChild(cursorEl);
  }

  function updateCursor(e) {
    if (!cursorEl) return;
    var vr = getVisualRect();
    var b  = getBaseRect();
    if (!vr || !b) { cursorEl.style.display = 'none'; return; }
    var cr = b.cr;
    var inView = (e.clientX >= Math.max(vr.left,  cr.left)  &&
                  e.clientX <= Math.min(vr.right,  cr.right) &&
                  e.clientY >= Math.max(vr.top,    cr.top)   &&
                  e.clientY <= Math.min(vr.bottom, cr.bottom));
    if (!inView) { cursorEl.style.display = 'none'; return; }
    var sr    = getScreenRadius();
    var mode  = getBrushMode();
    var color = mode === 'fg' ? '#1EC878' : '#DC3C3C';
    var diam  = Math.max(4, sr * 2);
    cursorEl.style.display     = 'block';
    cursorEl.style.left        = e.clientX + 'px';
    cursorEl.style.top         = e.clientY + 'px';
    cursorEl.style.width       = diam + 'px';
    cursorEl.style.height      = diam + 'px';
    cursorEl.style.borderColor = color;
  }

  // ── Overlay canvas ───────────────────────────────────────────────────────

  function setupOverlay() {
    var b = getBaseRect();
    if (!b) { setTimeout(setupOverlay, 600); return; }
    if (overlay) { overlay.remove(); overlay = null; }
    overlay = document.createElement('canvas');
    overlay.id     = 'seetwin-overlay';
    overlay.width  = b.nw;
    overlay.height = b.nh;
    overlay.style.cssText = 'position:fixed;z-index:9998;pointer-events:none;';
    document.body.appendChild(overlay);
    cachedImg = b.img;
    b.img.draggable = false;
    repositionOverlay();
    hideFullscreenButton();
  }

  function repositionOverlay() {
    if (!overlay) return;
    var vr = getVisualRect();
    var b  = getBaseRect();
    if (!vr || !b) return;
    var cr = b.cr;
    overlay.style.top    = vr.top    + 'px';
    overlay.style.left   = vr.left   + 'px';
    overlay.style.width  = vr.width  + 'px';
    overlay.style.height = vr.height + 'px';
    // Clip brush marks to the container's visible area
    var ct = Math.max(0, cr.top    - vr.top);
    var cl = Math.max(0, cr.left   - vr.left);
    var cb = Math.max(0, vr.top    + vr.height - cr.bottom);
    var crt = Math.max(0, vr.left  + vr.width  - cr.right);
    overlay.style.clipPath = (
      'inset(' + ct + 'px ' + crt + 'px ' + cb + 'px ' + cl + 'px)'
    );
  }

  // ── Zoom / pan ───────────────────────────────────────────────────────────

  function applyTransform() {
    var b = getBaseRect();
    if (!b || !b.img) return;
    cachedImg = b.img;
    b.img.style.transformOrigin = 'center center';
    b.img.style.transform = (
      'translate(' + panX + 'px,' + panY + 'px) scale(' + zoom + ')'
    );
    repositionOverlay();
  }

  function resetZoom() {
    zoom = 1; panX = 0; panY = 0;
    if (cachedImg) cachedImg.style.transform = '';
    repositionOverlay();
  }

  window.addEventListener('wheel', function (e) {
    var b = getBaseRect();
    if (!b) return;
    var cr = b.cr;
    if (e.clientX < cr.left || e.clientX > cr.right ||
        e.clientY < cr.top  || e.clientY > cr.bottom) return;
    e.preventDefault();

    if (e.ctrlKey) {
      // Pinch-to-zoom: hold cursor point fixed
      var factor  = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      var newZoom = Math.max(1, Math.min(20, zoom * factor));
      var oldCX   = b.cx + panX, oldCY = b.cy + panY;
      panX += (e.clientX - oldCX) * (1 - newZoom / zoom);
      panY += (e.clientY - oldCY) * (1 - newZoom / zoom);
      zoom = newZoom;
    } else {
      panX -= e.deltaX;
      panY -= e.deltaY;
    }
    // Clamp: image content always fills the container (no gray bars)
    var maxX = Math.max(0, (zoom * b.bw - cr.width)  / 2);
    var maxY = Math.max(0, (zoom * b.bh - cr.height) / 2);
    panX = Math.max(-maxX, Math.min(maxX, panX));
    panY = Math.max(-maxY, Math.min(maxY, panY));
    applyTransform();
  }, { passive: false });

  window.addEventListener('scroll', repositionOverlay, true);
  window.addEventListener('resize', function () { resetZoom(); setupOverlay(); });

  // ── Brush painting ───────────────────────────────────────────────────────

  document.addEventListener('mousedown', function (e) {
    var vr = getVisualRect();
    var b  = getBaseRect();
    if (!vr || !b) return;
    var cr = b.cr;
    if (e.clientX < Math.max(vr.left,  cr.left)  ||
        e.clientX > Math.min(vr.right,  cr.right) ||
        e.clientY < Math.max(vr.top,    cr.top)   ||
        e.clientY > Math.min(vr.bottom, cr.bottom)) return;
    e.preventDefault();
    e.stopPropagation();
    painting = true;
    paint(e);
  }, true);

  document.addEventListener('mousemove', function (e) {
    updateCursor(e);
    if (painting) paint(e);
  });

  document.addEventListener('mouseup', function () {
    if (painting) { painting = false; syncBox(); }
  });

  function paint(e) {
    var vr = getVisualRect();
    var b  = getBaseRect();
    if (!vr || !b) return;
    var x = Math.round((e.clientX - vr.left) * b.nw / vr.width);
    var y = Math.round((e.clientY - vr.top)  * b.nh / vr.height);
    if (x < 0 || y < 0 || x >= b.nw || y >= b.nh) return;
    var br   = Math.round(getBrushRadius());
    var mode = getBrushMode();
    window.seetwin_strokes.push({ x: x, y: y, r: br, mode: mode });
    if (overlay) {
      var color = mode === 'fg' ? '#1EC878' : '#DC3C3C';
      var ctx = overlay.getContext('2d');
      ctx.globalAlpha = 0.65;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, br, 0, 2 * Math.PI); ctx.fill();
    }
  }

  // ── Sync / clear ─────────────────────────────────────────────────────────

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
    if (overlay) overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
    syncBox();
  };

  // ── Setup ────────────────────────────────────────────────────────────────

  function hideFullscreenButton() {
    var c = getContainer();
    if (!c) return;
    c.querySelectorAll('button').forEach(function (btn) {
      var t = (btn.getAttribute('title') || btn.getAttribute('aria-label') || '').toLowerCase();
      if (t.indexOf('full') >= 0 || t.indexOf('screen') >= 0) btn.style.display = 'none';
    });
  }

  function startWatcher() {
    var img = getNatImg();
    if (!img) { setTimeout(startWatcher, 500); return; }
    new MutationObserver(function () {
      resetZoom();
      if (overlay) { overlay.remove(); overlay = null; }
      setTimeout(setupOverlay, 400);
    }).observe(img, { attributes: true, attributeFilter: ['src'] });
  }

  setTimeout(function () { setupOverlay(); ensureCursor(); }, 1500);
  setTimeout(startWatcher, 1500);
}
"""

_STYLE = """<style>
.st1-label { font-size: 0.78rem; color: #aaa; text-transform: uppercase;
             letter-spacing: .05em; margin: 8px 0 2px; }
#seetwin-mode label:first-of-type span { color: #1EC878 !important; font-weight: 600; }
#seetwin-mode label:last-of-type  span { color: #DC3C3C !important; font-weight: 600; }
/* Hide native cursor on canvas — JS circle cursor takes over */
#seetwin-canvas img { cursor: none !important; -webkit-user-drag: none; user-drag: none; }
/* Ensure zoomed image clips to the container */
#seetwin-canvas { overflow: hidden !important; }
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
    logger.info("Running model on slot %d (%s)…", idx, PHOTO_LABELS[idx])
    session["alpha_masks"][idx] = run_model(session["images"][idx], device=device)
    _ensure_strokes(session, idx)
    logger.info("Model done. Alpha masks set for slots: %s",
                [i for i, m in enumerate(session["alpha_masks"]) if m is not None])
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

def handle_export(strokes_json, session, threshold, feather):
    logger.info("=== handle_export CALLED ===")
    _apply_json_strokes(strokes_json, session)
    import tempfile
    png_items = []
    for idx in range(3):
        img   = session["images"][idx]
        alpha = session["alpha_masks"][idx]
        sl    = session["strokes"][idx]
        logger.info("Export slot %d: img=%s  alpha=%s  strokes=%s",
                    idx, img is not None, alpha is not None,
                    sl.stroke_count() if sl else None)
        if img is None or alpha is None:
            continue
        sl   = sl or StrokeLayer(img.height, img.width)
        mask = merge(alpha, sl, threshold=threshold, feather_px=feather)
        rgba = apply_alpha_to_image(img, mask)
        png_items.append((f"seetwin_{PHOTO_LABELS[idx].lower()}.png", rgba))

    if not png_items:
        logger.info("Export: nothing ready to export")
        return gr.update(visible=False)

    with tempfile.NamedTemporaryFile(suffix="_seetwin_export.zip", delete=False) as f:
        zip_path = f.name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, rgba in png_items:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pf:
                rgba.save(pf.name, "PNG")
                zf.write(pf.name, arcname=fname)
    logger.info("Export ZIP ready: %s  (%d photo(s))", zip_path, len(png_items))
    return gr.update(value=zip_path, visible=True)


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
                    btn_front = gr.Button("Front", variant="secondary")
                    btn_side  = gr.Button("Side",  variant="secondary")
                    btn_back  = gr.Button("Back",  variant="secondary")

            # ── Centre: canvas only ──────────────────────────────────────────
            with gr.Column(scale=5):
                canvas = gr.Image(
                    label="Canvas",
                    show_label=False,
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
                    show_label=False,
                    elem_id="seetwin-mode",
                    interactive=True,
                )
                gr.HTML('<p class="st1-label">Brush radius (px)</p>')
                brush_slider = gr.Slider(
                    0, 80, value=20, step=1,
                    label="",
                    show_label=False,
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
                btn_export  = gr.Button("Export ZIP",               variant="secondary")
                export_file  = gr.File(label="Download", visible=False)

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
            fn=handle_export,
            inputs=[strokes_box, session_state, threshold, feather],
            outputs=[export_file])

    return tab
