"""Stage 2 — Body shape estimation UI (Gradio)."""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import numpy as np
import gradio as gr
from PIL import Image

from pipeline.stage2_body_shape import (
    extract_keypoints, draw_landmarks,
    fit_body_shape, beta_to_measurements,
)
from pipeline.stage2_body_shape.keypoint_extractor import (
    LANDMARK_NAMES, _checker_background,
)

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

# ---------------------------------------------------------------------------
# Stage 2 JS — interactive landmark canvas overlay.
#
# For each of the three annotated views (front/side/back):
#   · A <canvas> is injected over the gr.Image display.
#   · The canvas draws: base photo | skeleton edges | landmark dots (all
#     scaled by zoom/pan transform) | text labels (fixed 13px, NOT scaled).
#   · Wheel to zoom (centred on cursor), drag on empty area to pan,
#     drag a landmark dot to reposition it.
#   · On drag-end the new position is synced to a hidden Gradio Textbox;
#     a Python .change() handler updates the session state.
#   · Landmark data (from Python) arrives as base64 JSON in a hidden
#     gr.HTML element; a MutationObserver triggers re-draw on update.
# ---------------------------------------------------------------------------
STAGE2_JS = r"""() => {
  var VIEWS = ['front','side','back'];

  var EDGES = [
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],
    [23,25],[25,27],[27,29],[27,31],
    [24,26],[26,28],[28,30],[28,32]
  ];

  var EDGE_SIDE = {
    '11,12':'C','23,24':'C',
    '11,13':'L','13,15':'L','11,23':'L','23,25':'L','25,27':'L','27,29':'L','27,31':'L',
    '12,14':'R','14,16':'R','12,24':'R','24,26':'R','26,28':'R','28,30':'R','28,32':'R'
  };
  var ECOL = {L:'#00C878', R:'#DC5050', C:'#A0A0A0'};

  var NAMES = {
    0:'nose',1:'L-eye-in',2:'L-eye',3:'L-eye-out',
    4:'R-eye-in',5:'R-eye',6:'R-eye-out',
    7:'L-ear',8:'R-ear',9:'mouth-L',10:'mouth-R',
    11:'L-shldr',12:'R-shldr',13:'L-elbow',14:'R-elbow',
    15:'L-wrist',16:'R-wrist',17:'L-pinky',18:'R-pinky',
    19:'L-index',20:'R-index',21:'L-thumb',22:'R-thumb',
    23:'L-hip',24:'R-hip',25:'L-knee',26:'R-knee',
    27:'L-ankle',28:'R-ankle',29:'L-heel',30:'R-heel',
    31:'L-foot',32:'R-foot'
  };

  var VS = {};
  VIEWS.forEach(function(v) {
    VS[v] = {
      canvas:null, ctx:null,
      img:null, imgW:0, imgH:0,
      landmarks:null,
      fitZoom:1, zoom:1, panX:0, panY:0,
      dragging:-1,
      panning:false, panStart:null,
      _lastSrc:'', _lastB64:'',
      _toast:null, _toastTimer:null
    };
  });

  // ── Drawing ───────────────────────────────────────────────────────────────

  function drawView(v) {
    var s = VS[v];
    if (!s.canvas || !s.ctx) return;
    var ctx = s.ctx;
    var W = s.canvas.width, H = s.canvas.height;
    ctx.clearRect(0, 0, W, H);

    ctx.save();
    ctx.translate(s.panX, s.panY);
    ctx.scale(s.zoom, s.zoom);

    if (s.img && s.img.complete && s.imgW > 0) {
      ctx.drawImage(s.img, 0, 0, s.imgW, s.imgH);
    }

    if (s.landmarks) {
      // Lines and dots are drawn in image-space but sized to be constant screen pixels
      ctx.lineWidth = 3.5 / s.zoom;
      EDGES.forEach(function(pair) {
        var a = pair[0], b = pair[1];
        var la = s.landmarks[a], lb = s.landmarks[b];
        if (!la || !lb) return;
        var key = a + ',' + b;
        var side = EDGE_SIDE[key] || 'C';
        var vis = Math.min(la.visibility, lb.visibility);
        ctx.globalAlpha = Math.max(0.35, vis);
        ctx.strokeStyle = ECOL[side];
        ctx.beginPath();
        ctx.moveTo(la.x * s.imgW, la.y * s.imgH);
        ctx.lineTo(lb.x * s.imgW, lb.y * s.imgH);
        ctx.stroke();
      });

      var dotR = 7 / s.zoom;
      ctx.lineWidth = 2 / s.zoom;
      s.landmarks.forEach(function(lm, idx) {
        if (!lm) return;
        if (v === 'back' && idx < 11) return; // face not visible from back
        var x = lm.x * s.imgW, y = lm.y * s.imgH;
        var isDrag = (idx === s.dragging);
        ctx.globalAlpha = Math.max(0.4, lm.visibility);
        ctx.fillStyle   = isDrag ? '#FFFFFF' : '#FFDC32';
        ctx.strokeStyle = '#000000';
        ctx.beginPath();
        ctx.arc(x, y, isDrag ? dotR * 1.4 : dotR, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      });
    }

    ctx.restore(); // text is drawn below in canvas-pixel space (not scaled)

    if (s.landmarks) {
      ctx.font = '13px "Helvetica Neue",Helvetica,Arial,sans-serif';
      s.landmarks.forEach(function(lm, idx) {
        if (!lm) return;
        if (v === 'back' && idx < 11) return; // face not visible from back
        var name = NAMES[idx];
        if (!name) return;
        var cx = lm.x * s.imgW * s.zoom + s.panX;
        var cy = lm.y * s.imgH * s.zoom + s.panY;
        if (cx < -80 || cx > W + 80 || cy < -20 || cy > H + 20) return;
        var tx = cx + 9, ty = cy - 7;
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = '#000000';
        ctx.fillText(name, tx+1, ty+1);
        ctx.fillText(name, tx-1, ty-1);
        ctx.fillText(name, tx+1, ty-1);
        ctx.fillText(name, tx-1, ty+1);
        ctx.globalAlpha = 1;
        ctx.fillStyle = (idx === s.dragging) ? '#FFFFFF' : '#FFDC32';
        ctx.fillText(name, tx, ty);
      });
    }

    // Toast notification (zoom limit alert)
    if (s._toast) {
      var midX = W / 2;
      ctx.save();
      ctx.font = 'bold 12px "Helvetica Neue",Helvetica,Arial,sans-serif';
      ctx.textAlign = 'center';
      var tw = ctx.measureText(s._toast).width;
      ctx.globalAlpha = 0.82;
      ctx.fillStyle = '#1a1a1a';
      ctx.fillRect(midX - tw/2 - 12, 10, tw + 24, 28);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#FFDC32';
      ctx.fillText(s._toast, midX, 29);
      ctx.restore();
    }
  }

  // ── Canvas setup ──────────────────────────────────────────────────────────

  function setupCanvas(v) {
    var s    = VS[v];
    var host = document.querySelector('#s2-ann-' + v);
    if (!host) return;

    var img = host.querySelector('img');
    if (!img || !img.src || img.naturalWidth < 4) return;

    if (s.canvas && img.src === s._lastSrc && s.canvas.width > 0) return;
    s._lastSrc = img.src;

    if (s.canvas) { s.canvas.remove(); s.canvas = null; }

    // Size canvas to the FULL component container, not just the narrow <img> rect.
    // This gives a proper pan viewport for side-view photos with slim aspect ratios.
    if (host.style.position !== 'relative') host.style.position = 'relative';
    var hRect = host.getBoundingClientRect();
    if (!hRect.width || !hRect.height) return;

    var canvas = document.createElement('canvas');
    canvas.className = 's2-lm-canvas';
    canvas.width  = Math.round(hRect.width);
    canvas.height = Math.round(hRect.height);
    canvas.style.cssText = (
      'position:absolute;z-index:20;cursor:crosshair;pointer-events:auto;' +
      'top:0;left:0;width:100%;height:100%;'
    );
    host.appendChild(canvas);

    s.canvas = canvas;
    s.ctx    = canvas.getContext('2d');
    s.imgW   = img.naturalWidth;
    s.imgH   = img.naturalHeight;

    // Fit zoom: scale so the full image fills the canvas at zoom=1
    var fitZoom = Math.min(canvas.width / s.imgW, canvas.height / s.imgH);
    s.fitZoom = fitZoom;
    s.zoom    = fitZoom;
    s.panX    = (canvas.width  - s.imgW * fitZoom) / 2;
    s.panY    = (canvas.height - s.imgH * fitZoom) / 2;

    var ci = new Image();
    ci.crossOrigin = 'anonymous';
    ci.onload = function() {
      s.img = ci;
      img.style.visibility = 'hidden'; // canvas draws the image; hide the <img>
      drawView(v);
    };
    ci.onerror = function() {
      // CORS fallback: keep <img> visible, draw only skeleton on transparent canvas
      s.img = null;
      img.style.visibility = '';
      drawView(v);
    };
    ci.src = img.src;
    if (ci.complete && ci.naturalWidth > 0) {
      s.img = ci;
      img.style.visibility = 'hidden';
      drawView(v);
    }

    canvas.addEventListener('mousedown',  function(e){ onDown(e,v); });
    canvas.addEventListener('mousemove',  function(e){ onMove(e,v); });
    canvas.addEventListener('mouseup',    function(e){ onUp(e,v); });
    canvas.addEventListener('mouseleave', function(e){ onLeave(e,v); });
    canvas.addEventListener('wheel',      function(e){ onWheel(e,v); },{passive:false});
    canvas.addEventListener('contextmenu',function(e){ e.preventDefault(); });
  }

  // ── Mouse helpers ─────────────────────────────────────────────────────────

  function canvasXY(e, canvas) {
    var r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  function nearestLM(cx, cy, s) {
    if (!s.landmarks) return -1;
    var SNAP = 15;
    var best = -1, bestD = Infinity;
    s.landmarks.forEach(function(lm, idx) {
      if (!lm) return;
      var sx = lm.x * s.imgW * s.zoom + s.panX;
      var sy = lm.y * s.imgH * s.zoom + s.panY;
      var d  = Math.sqrt((cx-sx)*(cx-sx) + (cy-sy)*(cy-sy));
      if (d < SNAP && d < bestD) { bestD = d; best = idx; }
    });
    return best;
  }

  function onDown(e, v) {
    var s = VS[v];
    var pos = canvasXY(e, s.canvas);
    var idx = nearestLM(pos[0], pos[1], s);
    if (idx >= 0) {
      s.dragging = idx;
      s.canvas.style.cursor = 'grabbing';
      e.preventDefault();
    } else {
      s.panning  = true;
      s.panStart = [e.clientX - s.panX, e.clientY - s.panY];
      s.canvas.style.cursor = 'grab';
    }
  }

  function onMove(e, v) {
    var s = VS[v];
    if (!s.canvas) return;

    if (s.dragging >= 0 && s.landmarks) {
      var pos = canvasXY(e, s.canvas);
      var nx = (pos[0] - s.panX) / (s.imgW * s.zoom);
      var ny = (pos[1] - s.panY) / (s.imgH * s.zoom);
      nx = Math.max(0, Math.min(1, nx));
      ny = Math.max(0, Math.min(1, ny));
      s.landmarks[s.dragging].x = nx;
      s.landmarks[s.dragging].y = ny;
      drawView(v);
      e.preventDefault();
    } else if (s.panning) {
      s.panX = e.clientX - s.panStart[0];
      s.panY = e.clientY - s.panStart[1];
      drawView(v);
    } else {
      var pos2 = canvasXY(e, s.canvas);
      var near = nearestLM(pos2[0], pos2[1], s);
      s.canvas.style.cursor = near >= 0 ? 'grab' : 'crosshair';
    }
  }

  function onUp(e, v) {
    var s = VS[v];
    if (s.dragging >= 0) {
      syncLandmark(v, s.dragging);
      s.dragging = -1;
      s.canvas.style.cursor = 'crosshair';
    }
    s.panning = false;
  }

  function onLeave(e, v) {
    VS[v].panning = false;
  }

  function showToast(v, msg) {
    var s = VS[v];
    s._toast = msg;
    if (s._toastTimer) clearTimeout(s._toastTimer);
    s._toastTimer = setTimeout(function() { s._toast = null; drawView(v); }, 1500);
  }

  function onWheel(e, v) {
    e.preventDefault();
    var s = VS[v];
    if (!s.canvas) return;
    var pos = canvasXY(e, s.canvas);
    var cx = pos[0], cy = pos[1];
    // Reduced sensitivity: 9% per tick (was 18%/15%)
    var f = e.deltaY > 0 ? 0.91 : 1.09;
    var minZ = s.fitZoom, maxZ = s.fitZoom * 10;
    var newZoom = s.zoom * f;

    if (newZoom <= minZ) {
      if (s.zoom === minZ) { showToast(v, 'Cannot zoom out further'); drawView(v); return; }
      newZoom = minZ;
    } else if (newZoom >= maxZ) {
      if (s.zoom === maxZ) { showToast(v, 'Maximum zoom reached'); drawView(v); return; }
      newZoom = maxZ;
    }

    // Zoom centred on cursor: keep the image point under the cursor fixed
    var scale = newZoom / s.zoom;
    s.panX = cx - (cx - s.panX) * scale;
    s.panY = cy - (cy - s.panY) * scale;
    s.zoom = newZoom;
    drawView(v);
  }

  // ── JS → Python sync ──────────────────────────────────────────────────────

  function syncLandmark(v, idx) {
    var lm = VS[v].landmarks && VS[v].landmarks[idx];
    if (!lm) return;
    var tb = document.querySelector('#s2-lm-up-' + v + ' textarea');
    if (!tb) return;
    var val = JSON.stringify({idx:idx, x:lm.x, y:lm.y});
    var setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(tb, val);
    tb.dispatchEvent(new Event('input', {bubbles:true}));
  }

  // ── Python → JS landmark data ─────────────────────────────────────────────

  function loadLmData(v) {
    var el = document.querySelector('#s2-lm-data-' + v);
    if (!el) return;
    var tag = el.querySelector('[data-b64]');
    if (!tag) return;
    var b64 = tag.getAttribute('data-b64');
    if (!b64 || b64 === VS[v]._lastB64) return;
    VS[v]._lastB64 = b64;
    try { VS[v].landmarks = JSON.parse(atob(b64)); }
    catch(err) { VS[v].landmarks = null; }
    drawView(v);
  }

  // ── Watchers ──────────────────────────────────────────────────────────────

  function watchView(v) {
    var annEl = document.querySelector('#s2-ann-' + v);
    if (annEl) {
      new MutationObserver(function() {
        setTimeout(function() { setupCanvas(v); loadLmData(v); }, 250);
      }).observe(annEl, {childList:true, subtree:true,
                          attributes:true, attributeFilter:['src']});

      // ResizeObserver re-triggers canvas setup when a hidden tab becomes visible
      if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(function() {
          var s = VS[v];
          if (!s.canvas || s.canvas.width === 0) {
            s._lastSrc = '';
            setTimeout(function() { setupCanvas(v); }, 150);
          }
        }).observe(annEl);
      }
    }

    var lmEl = document.querySelector('#s2-lm-data-' + v);
    if (lmEl) {
      new MutationObserver(function() {
        setTimeout(function() { loadLmData(v); }, 80);
      }).observe(lmEl, {childList:true, subtree:true});
    }
  }

  // ── Tab-advance signal ────────────────────────────────────────────────────
  // Python writes "advance" into #s2-advance (gr.HTML); observer clicks Stage 3 tab.
  // Same pattern as landmark data observers — set up once in init, lives for session.

  function setupAdvanceObserver() {
    var el = document.querySelector('#s2-advance');
    if (!el) { setTimeout(setupAdvanceObserver, 800); return; }
    new MutationObserver(function() {
      // Read textContent recursively — Svelte may nest content in child divs
      var txt = (el.textContent || '').trim();
      if (txt !== 'advance') return;
      // Find the "3 — Classification" tab button by partial text match
      document.querySelectorAll('button').forEach(function(b) {
        if (b.textContent.includes('Classification')) b.click();
      });
    }).observe(el, {childList: true, subtree: true, characterData: true});
  }

  function init() {
    setupAdvanceObserver();
    VIEWS.forEach(function(v) {
      watchView(v);
      setupCanvas(v);
      loadLmData(v);
    });
    window.addEventListener('resize', function() {
      VIEWS.forEach(function(v) {
        if (VS[v].canvas) { VS[v].canvas.remove(); VS[v].canvas = null; }
        VS[v]._lastSrc = '';
        var host = document.querySelector('#s2-ann-' + v);
        if (host) {
          var im = host.querySelector('img');
          if (im) im.style.visibility = '';
        }
        setTimeout(function() { setupCanvas(v); }, 300);
      });
    });
  }

  setTimeout(init, 1500);
}"""


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


def _base_photo(img: Image.Image) -> Image.Image:
    """Composite RGBA on checker background; return RGB for gr.Image display."""
    if img.mode == "RGBA":
        bg = _checker_background(img.width, img.height)
        return Image.alpha_composite(bg, img).convert("RGB")
    return img.convert("RGB")


def _lm_data_html(kp: dict | None) -> str:
    """Encode landmark list as base64 JSON inside an HTML data attribute."""
    if not kp or not kp.get("detected"):
        return ""
    b64 = base64.b64encode(
        json.dumps(kp["landmarks"]).encode()
    ).decode()
    return f'<span data-b64="{b64}"></span>'


_LR_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6),          # eyes
    (7, 8), (9, 10),                  # ears, mouth
    (11, 12), (13, 14), (15, 16),     # shoulder / elbow / wrist
    (17, 18), (19, 20), (21, 22),     # pinky / index / thumb
    (23, 24), (25, 26), (27, 28),     # hip / knee / ankle
    (29, 30), (31, 32),               # heel / foot
]


def _fix_back_view_landmarks(kp: dict) -> dict:
    """Swap left/right landmark identities and zero face landmarks for back-view photos.

    MediaPipe is trained on front-facing images, so in a back photo it labels
    the arm on the LEFT of the image as 'R-shldr'. Swapping each L/R pair
    restores anatomically correct labels without moving any points.

    Face landmarks (0-10: nose, eyes, ears, mouth) are not visible from the
    back; their visibility is set to 0 so they are hidden in the overlay and
    ignored in fitting.
    """
    if not kp or not kp.get("detected"):
        return kp
    lms = [dict(lm) for lm in kp["landmarks"]]
    for l_idx, r_idx in _LR_SWAP_PAIRS:
        lms[l_idx], lms[r_idx] = lms[r_idx], lms[l_idx]
    for i in range(11):          # indices 0–10 are face landmarks
        lms[i]["visibility"] = 0.0
    return {**kp, "landmarks": lms}


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_extract(img_front, img_side, img_back, state: dict):
    """Extract MediaPipe keypoints; return base photos + landmark data for JS."""
    photos = {"front": img_front, "side": img_side, "back": img_back}
    base, lm_html = {}, {}

    for view, img in photos.items():
        state[f"img_{view}"] = img
        if img is None:
            state[f"kp_{view}"] = None
            base[view]    = None
            lm_html[view] = ""
            continue
        kp = extract_keypoints(img)
        if view == "back":
            kp = _fix_back_view_landmarks(kp)
        state[f"kp_{view}"] = kp
        base[view]    = _base_photo(img)
        lm_html[view] = _lm_data_html(kp)
        logger.info("Keypoints %s: detected=%s", view, kp.get("detected"))

    if img_front is not None:
        state["img_w"], state["img_h"] = img_front.size

    required_ok = sum(
        1 for v in ["front", "side"]
        if state.get(f"kp_{v}") and state[f"kp_{v}"].get("detected")
    )
    back_ok = bool(state.get("kp_back") and state["kp_back"].get("detected"))
    if required_ok == 0:
        status = "No person detected in any photo. Check photo quality."
    else:
        back_note = " + back" if back_ok else (" (back: not detected)" if state.get("img_back") else "")
        status = (f"Keypoints extracted — {required_ok}/2 required views detected{back_note}. "
                  "Drag any landmark dot to correct its position.")

    return (state,
            base["front"], base["side"], base["back"],
            lm_html["front"], lm_html["side"], lm_html["back"],
            status,
            base["front"], base["side"], base["back"])


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
    """Recompute measurements when β sliders change.
    Landmark positions are MediaPipe detections and do not move with β."""
    beta = np.array(slider_vals, dtype=np.float32)
    state["beta"] = beta.tolist()
    return state, _measurements_html(beta_to_measurements(beta))


def _make_lm_update_handler(view: str):
    def handler(state: dict, correction_json: str):
        if not correction_json:
            return state
        try:
            data = json.loads(correction_json)
            idx  = int(data["idx"])
            kp   = state.get(f"kp_{view}")
            if kp and kp.get("detected") and 0 <= idx < 33:
                kp["landmarks"][idx]["x"] = float(data["x"])
                kp["landmarks"][idx]["y"] = float(data["y"])
                kp["landmarks"][idx]["visibility"] = 1.0
                state[f"kp_{view}"] = kp
        except Exception as exc:
            logger.warning("Bad landmark update (%s): %s", view, exc)
        return state
    return handler


def handle_confirm(state: dict):
    beta = state.get("beta")
    if beta is None:
        return state, "Run body shape fitting first."
    return state, f"✓ Body shape confirmed. β = {[round(b, 3) for b in beta]}"


def handle_download(state: dict):
    beta = state.get("beta")
    if beta is None:
        return gr.update(visible=False)
    measurements = beta_to_measurements(np.array(beta, dtype=np.float32))
    payload = {
        "beta_parameters": {f"beta_{i}": round(v, 6) for i, v in enumerate(beta)},
        "measurements_cm": measurements,
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="seetwin_body_"
    )
    json.dump(payload, tmp, indent=2)
    tmp.close()
    return gr.update(value=tmp.name, visible=True)


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_stage2_tab(stage1_state: gr.State | None = None,
                     main_tabs: gr.Tabs | None = None) -> gr.Tab:
    with gr.Tab("2 — Body shape") as tab:
        state = gr.State(_empty_state())
        gr.HTML(_STYLE)

        with gr.Row():
            # ── Left: photo inputs ─────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Input photos")
                gr.HTML(
                    '<p class="st2-label">Transparent PNGs from Stage 1 (or upload directly) '
                    '· Front + Side required · Back optional</p>'
                )
                img_front = gr.Image(label="Front", type="pil", height=160, image_mode="RGBA")
                img_side  = gr.Image(label="Side",  type="pil", height=160, image_mode="RGBA")
                img_back  = gr.Image(label="Back (optional)", type="pil", height=160, image_mode="RGBA")
                btn_extract = gr.Button("1 — Extract keypoints", variant="secondary")

            # ── Centre: annotated landmark views ───────────────────────────
            with gr.Column(scale=5):
                gr.Markdown("### Landmark overlay")
                gr.HTML(
                    '<p style="font-size:0.82rem;color:#aaa;margin:2px 0 10px">'
                    '<span style="color:#00C878">Green = left side</span> · '
                    '<span style="color:#DC5050">Red = right side</span> · '
                    '<span style="color:#FFDC32">Yellow = landmark</span><br>'
                    "<strong>Drag</strong> any dot to correct its position · "
                    "<strong>Scroll</strong> to zoom · <strong>Drag empty area</strong> to pan"
                    "</p>"
                )
                with gr.Tabs():
                    with gr.Tab("Front"):
                        ann_front = gr.Image(
                            type="pil", height=520,
                            show_label=False, interactive=False,
                            elem_id="s2-ann-front",
                        )
                        lm_html_f = gr.HTML("", visible=False, elem_id="s2-lm-data-front")
                        lm_up_f   = gr.Textbox("", visible=False, elem_id="s2-lm-up-front")
                    with gr.Tab("Side"):
                        ann_side = gr.Image(
                            type="pil", height=520,
                            show_label=False, interactive=False,
                            elem_id="s2-ann-side",
                        )
                        lm_html_s = gr.HTML("", visible=False, elem_id="s2-lm-data-side")
                        lm_up_s   = gr.Textbox("", visible=False, elem_id="s2-lm-up-side")
                    with gr.Tab("Back"):
                        ann_back = gr.Image(
                            type="pil", height=520,
                            show_label=False, interactive=False,
                            elem_id="s2-ann-back",
                        )
                        lm_html_b = gr.HTML("", visible=False, elem_id="s2-lm-data-back")
                        lm_up_b   = gr.Textbox("", visible=False, elem_id="s2-lm-up-back")


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
                        "β controls the 3D SMPL-X body shape — measurements update live.<br><br>"
                        "Drag landmark dots on the overlay to correct wrong detections, "
                        "then click <strong>2 — Fit body shape</strong> again to re-fit."
                        "</p>"
                    )
                    beta_sliders = [
                        gr.Slider(-3.0, 3.0, value=0.0, step=0.05,
                                  label=BETA_LABELS[i], show_label=True)
                        for i in range(NUM_BETAS)
                    ]

                gr.HTML('<hr style="border-color:#444;margin:12px 0">')
                btn_download = gr.Button("⬇ Download betas & measurements", variant="secondary")
                download_file = gr.File(visible=False, label="Download")
                gr.HTML('<div style="margin:6px 0"></div>')
                btn_confirm = gr.Button("3 — Confirm & continue", variant="primary")

        # Hidden signal: Python sets this to trigger JS tab advance
        adv_html = gr.HTML("", visible=False, elem_id="s2-advance")

        status = gr.Markdown(
            "Upload front + side photos (back is optional), then click **Extract keypoints**."
        )

        # ── Wiring ────────────────────────────────────────────────────────

        btn_extract.click(
            fn=handle_extract,
            inputs=[img_front, img_side, img_back, state],
            outputs=[state,
                     ann_front, ann_side, ann_back,
                     lm_html_f, lm_html_s, lm_html_b,
                     status,
                     img_front, img_side, img_back],
        )

        btn_fit.click(
            fn=handle_fit,
            inputs=[state],
            outputs=[state, meas_html, beta_group] + beta_sliders,
        )

        # JS drag corrections → update Python state (silent, no re-render)
        lm_up_f.change(
            fn=_make_lm_update_handler("front"),
            inputs=[state, lm_up_f],
            outputs=[state],
        )
        lm_up_s.change(
            fn=_make_lm_update_handler("side"),
            inputs=[state, lm_up_s],
            outputs=[state],
        )
        lm_up_b.change(
            fn=_make_lm_update_handler("back"),
            inputs=[state, lm_up_b],
            outputs=[state],
        )

        # Live β adjustment → measurements only
        for sl in beta_sliders:
            sl.change(
                fn=handle_beta_update,
                inputs=[state] + beta_sliders,
                outputs=[state, meas_html],
            )

        btn_download.click(
            fn=handle_download,
            inputs=[state],
            outputs=[download_file],
        )

        def _confirm_and_advance(state):
            s, msg = handle_confirm(state)
            if main_tabs is not None and s.get("beta") is not None:
                return s, msg, gr.update(selected="stage3")
            return s, msg, gr.update()

        _confirm_outputs = [state, status]
        if main_tabs is not None:
            _confirm_outputs.append(main_tabs)

        btn_confirm.click(
            fn=_confirm_and_advance,
            inputs=[state],
            outputs=_confirm_outputs,
        )

    return tab
