# SeeTwin — Photo-to-Rigged 3D Avatar Pipeline

> Turn everyday photos of a person into a game-ready, rigged, stylized 3D avatar using parametric mesh deformation and AI-driven texture extraction.

---

## Core Architecture Philosophy

SeeTwin does **not** generate 3D geometry from scratch. Instead, it follows the same pattern used by Fortnite, Ready Player Me, and Roblox:

- **Geometry is always pre-authored** — a fixed library of clean, rigged base meshes
- **AI classifies and selects** the right base mesh per body part, garment, and accessory
- **Parameters adjust** mesh shape within known bounds (SMPL-X body shape, FLAME face shape, garment preset blending)
- **Texture carries the visual identity** — AI extracts a clean, shadow-free albedo atlas from the input photo and applies it to the selected mesh
- **The rig is embedded** in the base meshes — rigging is always clean because the skeleton never changes

This means the output geometry is always predictable, always clean, and always game-ready. The AI never touches topology.

---

## Pipeline Overview

```
Photos (4 angles, T-pose)
        │
        ▼
Stage 1 — Background Removal
        │  BRIA-RMBG-2.0 · brush + threshold corrections
        ▼
Stage 2 — Body Shape Estimation
        │  SMPL-X parameter fitting (height, weight, proportions)
        │  No mesh export yet — parameters only
        ▼
Stage 3 — Classification & Confirmation   ← human confirms every result
        │  Body silhouette → SMPL-X shape params
        │  Face type       → FLAME shape params
        │  Hair style      → select from ~30 hair cap meshes
        │  Garments        → select base mesh + blend preset (5 per type)
        │  Accessories     → select from fixed library (~80 meshes)
        ▼
Stage 4 — Texture Extraction              ← quality-critical stage
        │  Intrinsic decomposition (albedo/shading separation)
        │  Shadow removal + lighting normalisation
        │  Per-part UV atlas generation
        │  Back/occluded region inpainting
        │  Stylization pass (cartoon, low-poly toon, ink, realistic)
        ▼
Stage 5 — Parametric Assembly
        │  All parts combined on shared SMPL-X skeleton
        │  Garment meshes fitted to body shape (5-preset blend)
        │  Textures applied to UV atlases
        │  Poly budget verification
        ▼
Stage 6 — Blender Fine-tune  (manual)
        │  Import script auto-places all parts
        │  Guided checklist: seams, UV overlaps, modifier apply
        ▼
Stage 7 — Auto-Rig
        │  Mixamo (recommended) or Rigify
        │  Skeleton already embedded — trivial clean result
        ▼
Stage 8 — Real-Time Motion Capture
        │  MediaPipe Holistic → skeleton keypoints
        │  Three.js web preview, WebSocket streaming
        ▼
Stage 9 — Pose Analysis (Kimodo)
           MediaPipe → SOMA skeleton → Kimodo constraints
           Sign language / yoga / dance demos
```

---

## Stage Details

### Stage 1 — Background Removal
**Input:** 3 full-body T-pose photos (front, back, side)

**Output:** 3 PNGs with transparent background, resized to max 1024px

**Model:** BRIA-RMBG-2.0, local inference via `transformers`

**UI:** Upload → auto-run → brush corrections (foreground green / background red) and threshold slider → re-run model preserving brush strokes → export

---

### Stage 2 — Body Shape Estimation
**Input:** Front + side + back transparent PNGs

**Output:** SMPL-X shape parameter vector β (10 values: height, weight, torso width, limb lengths, etc.)

**Model:** SMPL-X fitting via silhouette + keypoint optimization

**No mesh output at this stage** — parameters are passed to Stages 3 and 5


---

### Stage 3 — Classification & Confirmation
**Input:** 3 cleaned photos + SMPL-X β params

**Output:** A confirmed asset manifest (JSON) mapping each detected item to a base mesh ID + parameters

**Model:** The classifier runs sequentially. After each category, the user sees the result and confirms or corrects before the next runs.

**Classification order and tools:**

| Category | Classifier | Output |
|----------|-----------|--------|
| Body silhouette | SMPL-X fitting | Shape params β |
| Face type | FLAME fitting | Face shape params |
| Hair style | CLIP zero-shot over ~30 cap labels | Hair mesh ID |
| Each garment | Fine-tuned ViT on DeepFashion taxonomy | Garment type + mesh ID |
| Each accessory | CLIP zero-shot over accessory library labels | Accessory mesh ID |

**Confidence threshold:** Results below 85% confidence automatically surface the confirmation screen with alternatives ranked. Results above 85% still require confirmation but default to the top prediction.

**Asset manifest output (example):**
```json
{
  "body": { "mesh": "smplx_neutral", "beta": [0.3, -0.1, 0.8] },
  "face": { "mesh": "flame_base", "shape": [0.1, 0.4] },
  "hair": { "mesh": "hair_medium_wavy_001", "color_sample": "#3a2010" },
  "garments": [
    { "mesh": "shirt_fitted_M", "type": "tshirt", "layer": "base" },
    { "mesh": "pants_straight_M", "type": "jeans", "layer": "base" }
  ],
  "accessories": [
    { "mesh": "glasses_rectangle_001", "type": "glasses" }
  ]
}
```

---

### Stage 4 — Texture Extraction ⭐
**The quality-critical stage.** See `docs/texture_extraction_research.md` for full technical scaffold.

**Input:** Cleaned photos + confirmed asset manifest

**Output:** Per-part UV texture atlases (albedo, normal, roughness) at 1024×1024

**Sub-pipeline per part:**
1. **Segment** the part region from photo using SAM2 + manifest labels
2. **Intrinsic decomposition** — separate albedo from shading
3. **Shadow removal** — normalize lighting across region
4. **UV projection** — project visible pixels onto base mesh UV map
5. **Inpainting** — fill occluded/back-facing regions
6. **Stylization** — apply chosen style pass (toon, flat, ink)
7. **Pack atlas** — combine into final UV atlas per mesh

**Key tools:**
- Segmentation: SAM2
- Intrinsic decomposition: `compphoto/Intrinsic` (TOG 2024, open source)
- Human UV projection: SMPLitex or TexDreamer (image → SMPL UV map)
- Inpainting: Stable Diffusion inpaint (for back/occluded regions)
- Stylization: ControlNet depth/edge conditioned, or Blender toon shader

---

### Stage 5 — Parametric Assembly
**Input:** Asset manifest + UV atlases

**Output:** Single `.glb` with all parts, shared skeleton, textures applied

**Polygon budget targets:**

| Part | Target tris |
|------|------------|
| Body | 10,000–15,000 |
| Head + face | 3,000–5,000 |
| Hair cap | 1,500–3,000 |
| Each garment | 1,000–3,000 |
| Each accessory | 200–800 |
| **Total** | **~20,000–35,000** |

---

### Stage 6 — Blender Fine-tune (Manual)
**Input:** `.glb` from Stage 5 + `blender_scripts/import_and_assemble.py`

**Output:** Cleaned `.fbx` ready for rigging

Guided interactive checklist panel in Blender — seam inspection, UV overlap check, modifier apply, poly count verification, export.

---

### Stage 7 — Auto-Rig
**Recommended:** Mixamo web upload — auto-rigs humanoid mesh in ~2 minutes
**Alternative:** Rigify Blender add-on

Because all base meshes are SMPL-X-derived with consistent proportions, Mixamo auto-detection succeeds reliably.

---

### Stage 8 — Real-Time Motion Capture
MediaPipe Holistic (33 body + 21 hand keypoints) → WebSocket → Three.js browser viewer.
Phone support via QR code on same network.

---

### Stage 9 — Pose Analysis with Kimodo
MediaPipe keypoints → SOMA skeleton retarget → Kimodo constraint evaluation → per-joint deviation scoring → color-coded overlay.

**Demos:** ASL sign language (10 signs), yoga poses (8), simple dance (2 × 16-count).

---

## Fixed Asset Library

Must be authored by a 3D artist. This is the one hard dependency AI cannot replace.

| Asset type | Count | Notes |
|-----------|-------|-------|
| Hair cap meshes | ~30 | Major hair archetypes |
| Garment silhouettes | ~50 | ~5 presets × ~10 garment types |
| Accessory base meshes | ~80 | Glasses, bags, watches, headphones, hats, etc. |
| SMPL-X body model | 1 | Licensed via MPI |
| FLAME face model | 1 | Licensed via MPI |

All meshes share compatible UV layouts for texture swapping.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Background removal | BRIA-RMBG-2.0 (local) |
| Body shape | SMPL-X (MPI) |
| Face shape | FLAME (MPI) |
| Intrinsic decomposition | `compphoto/Intrinsic` (TOG 2024, MIT) |
| Human UV projection | SMPLitex / TexDreamer |
| Segmentation | SAM2 (Meta) |
| Inpainting | Stable Diffusion inpaint |
| Garment classification | ViT fine-tuned on DeepFashion |
| Accessory/hair classification | CLIP zero-shot |
| 3D viewer + mocap | Three.js + WebSocket |
| Motion capture | MediaPipe Holistic |
| Pose analysis | NVIDIA Kimodo |
| Web UI | Gradio |
| Backend | Python 3.11+, FastAPI |

---

## Hardware Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| RAM | 8 GB | 16 GB |
| VRAM | 4 GB | 8 GB (SD inpainting) |
| GPU | Optional | NVIDIA RTX 3060+ |
| Disk | 15 GB | 25 GB |
| Blender | 3.6 LTS | 4.x |


## Installation

```bash
git clone https://github.com/yourname/seetwin.git
cd SeeTwin
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
python app.py
```

Open `http://localhost:7860`


## Known Limitations

- **Loose/baggy clothing:** 5-preset system assumes semi-fitted silhouettes
- **Occlusion:** Hidden regions are inpainted — quality varies
- **Complex hair:** Very unusual styles may need Blender fine-tuning
- **SMPL-X / FLAME licenses:** Non-commercial research use only — users must register at mpi-inf.mpg.de



## License

- Project code: MIT
- BRIA-RMBG-2.0: BRIA AI license (non-commercial research)
- SMPL-X / FLAME: MPI license (non-commercial research)
- Kimodo: NVIDIA Research license (research use)
- `compphoto/Intrinsic`: MIT
- Mixamo: Adobe license (free personal + commercial)
