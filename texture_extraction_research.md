# Texture Extraction — Research Scaffold

Stage 4 is the quality-critical stage of the SeeTwin pipeline. Every other stage
produces deterministic or near-deterministic output. This one does not. The goal of
this document is to scaffold the research needed to make it robust before writing
production code.



## What "texture extraction" means in this pipeline

Given a real-world photo of a person (arbitrary lighting, background, pose, phone
camera quality), extract a clean, shadow-free, lighting-normalized color map that
can be applied to a known UV-unwrapped 3D mesh and look plausible under arbitrary
game-engine lighting.

The output is not a photo crop. It is an **albedo map** — the intrinsic surface color
of the material, independent of the scene's lighting conditions.


## The core problem: photos bake lighting into color

A red shirt photographed under warm indoor light will have orange highlights and
dark-red shadows baked into the pixels. If you apply those pixels directly to a 3D
mesh, the character will look like it was permanently lit from one direction. Under
any other game-engine light, it will look wrong.

The solution is **intrinsic image decomposition**: separate the image into:
- **Albedo** (true surface color, lighting-independent)
- **Shading** (how light fell on the surface in this specific photo)

We keep the albedo and discard the shading.



## Sub-problem breakdown

The full Stage 4 pipeline has 7 sub-steps. Each has its own research questions.


### Sub-step 1: Segmentation
**Task:** Given a photo and a manifest label (e.g. "tshirt"), isolate the pixel
region of that item.

**Current best tool:** SAM2 (Meta, 2024) — segment anything with a prompt point or
box. Given the manifest's bounding box estimate from the classification stage, SAM2
can produce a clean mask in one forward pass.

**Research questions:**
- How well does SAM2 handle thin straps, hair edges, and glasses frames?
- At what confidence threshold should we flag a mask for user correction?
- Can we use the SMPL-X body silhouette as a hard constraint to prevent over-segmentation?

**Known failure modes:**
- Transparent or reflective materials (glasses lenses, shiny watches)
- Hair overlapping garment neckline
- Dark clothing in dim photos (low contrast with background)

**Experiment to run:**
Test SAM2 on 50 real-world photos across each garment category. Measure IoU against
hand-labeled masks. Log failure modes by category.

---

### Sub-step 2: Intrinsic decomposition
**Task:** Separate the segmented region into albedo and shading layers.

**Research landscape:**
Two strong open-source options exist as of 2024–2025:

**Option A: `compphoto/Intrinsic` (Careaga & Aksoy, TOG 2024)**
- License: MIT
- Handles in-the-wild images, not just controlled studio shots
- Outputs: `hr_alb` (high-res albedo), `dif_shd` (diffuse shading), `residual`
- Works on full images — apply after masking, or mask the output
- GitHub: https://github.com/compphoto/Intrinsic

**Option B: SMPLitex (Casas et al., BMVC 2023)**
- Specific to clothed humans on the SMPL UV map
- Uses diffusion model to hallucinate a complete UV texture from a partial projection
- Does NOT separate albedo from shading — produces a texture that includes some baked lighting
- Better for completing occluded regions, worse for lighting normalization

**Recommendation:** Use `compphoto/Intrinsic` for albedo extraction, then use
SMPLitex/TexDreamer for UV completion of occluded regions. They solve different problems.

**Research questions:**
- Does `compphoto/Intrinsic` handle clothing textures (patterns, stripes, logos) without
  mistaking texture variation for shading variation?
- What happens to metallic materials (belt buckles, zippers)? Metallic specularity is
  NOT diffuse — the Lambertian assumption breaks.
- Is the albedo output stable across the 4 input angles of the same garment?
  (We need consistency — the front and back atlases should have the same "base color".)

**Experiment to run:**
Run `compphoto/Intrinsic` on 20 garment photos with known dominant colors (solid red,
blue, white, black, patterned). Measure whether the albedo output matches ground truth
color better than the raw photo pixel. Specifically test: stripes, logos, denim texture,
leather, and knit patterns.

**Known hard cases:**
- White clothing in bright light: albedo is overexposed before decomposition
- Black clothing in dim light: shading and albedo are both near-zero, hard to separate
- Patterned fabric: high-frequency texture variation may be misclassified as shading

---

### Sub-step 3: Shadow removal
**Task:** After intrinsic decomposition, the albedo map may still have residual cast
shadows (from the arm casting shadow on the torso, etc.) that the decomposition didn't
fully remove.

**Tools to evaluate:**
- `compphoto/Intrinsic`'s `residual` output contains view-dependent effects including
  specular highlights and cast shadows. Subtract from albedo.
- Shadow detection + inpainting as a second pass: detect shadow regions (dark boundaries
  with a color shift toward the light's complement), then inpaint with the local average
  albedo color.

**Research questions:**
- Is the `residual` subtraction from `compphoto/Intrinsic` sufficient, or do we need
  a second pass?
- Can we detect shadow vs. genuine dark regions (dark shirt vs. shadowed light shirt)
  without ground truth?
- Is per-garment color normalization (normalize mean albedo within the masked region)
  a sufficient proxy for shadow removal at the cost of losing genuine color gradients
  in gradient-dyed garments?

**Experiment to run:**
Create a test set of 10 garments photographed under (a) even studio light and
(b) harsh directional natural light. The studio version is the ground truth.
Measure color distance between decomposed albedo from the harsh-light photo vs.
the studio photo.

---

### Sub-step 4: UV projection
**Task:** Map the cleaned albedo pixels from the 2D photo onto the 3D mesh's UV space.

**The problem:** The photo is a 2D projection of a 3D surface. We need the inverse —
to "unwrap" the visible surface back onto the UV map. This requires knowing the
correspondence between each photo pixel and its UV coordinate on the mesh.

**Tools:**
- **SMPLitex approach:** Estimate DensePose-style pixel-to-surface correspondences
  (which UV coordinate does each pixel project onto), then rasterize into UV space.
  SMPLitex uses this as its conditioning signal.
- **TexDreamer (ECCV 2024):** Instead of direct projection, uses a trained feature
  translator to map image features to a semantically correct UV texture. Zero-shot,
  handles unusual poses better than direct projection.
- **Pix2Surf (CVPR 2020):** Learns UV mapping for garments from silhouette alone,
  without needing body pose or DensePose.

**Recommendation for initial implementation:**
Use TexDreamer's I2UV (image-to-UV) mode. It is:
- Zero-shot (no per-garment training)
- Open source (GitHub: https://github.com/ggxxii/texdreamer)
- Faster than optimization-based methods
- Handles partial views (only front is visible — TexDreamer hallucinates the back)

**Research questions:**
- TexDreamer was trained on DeepFashion-MultiModal. How well does it generalize to
  non-studio phone photos (harsh lighting, cluttered backgrounds, non-neutral poses)?
- Does TexDreamer's hallucinated back of the garment match our separately extracted
  back photo? Or should we use our back photo directly and only use TexDreamer for
  sides/occluded regions?
- What is the UV layout assumed by TexDreamer? Does it match our asset library's UV
  unwrap? (This is a hard constraint — if they don't match, we need a remapping step.)

**Experiment to run:**
Run TexDreamer on 10 garments where we have both front and back photos. Compare
TexDreamer's hallucinated back-UV to our directly extracted back-UV. Measure
perceptual similarity (LPIPS) and color accuracy.

---

### Sub-step 5: Inpainting for occluded regions
**Task:** Fill in UV regions that are not visible in any of the 4 input photos.
Examples: the inner back collar, the underside of a sleeve, between legs in the crotch region.

**Strategy:**
- After UV projection from all 4 angles, identify unfilled UV regions (zero coverage).
- Inpaint using Stable Diffusion inpaint, conditioned on:
  - The filled regions of the same garment's atlas (for color/pattern continuity)
  - A text prompt describing the garment type and apparent material
- For skin regions (back of neck, back of hand): use the body atlas's filled skin
  tone regions as color reference.

**Research questions:**
- At what coverage threshold should we trigger inpainting vs. mirror/tile from visible
  regions? (Mirroring is faster and more consistent for symmetric garments.)
- Does SD inpainting maintain garment pattern continuity (e.g. stripes continuing
  around the torso) or does it generate plausible but inconsistent fills?
- For patterned garments (plaid, floral, logo-print), is diffusion inpainting better
  or worse than simply tiling from the visible region?

**Experiment to run:**
Collect 10 garments with clear patterns. Compare 3 approaches for filling the back UV:
(a) mirror from front, (b) tile from front, (c) SD inpainting.
Rate outputs on: pattern continuity, color accuracy, artifact frequency.

---

### Sub-step 6: Stylization
**Task:** Convert a photorealistic albedo atlas to a chosen visual style
(cartoon, low-poly flat, ink sketch, realistic clean).

**This sub-step is intentionally last.** Stylization is applied to an already
lighting-normalized, UV-projected albedo — not to the raw photo. This ensures
the stylization works on clean input.

**Tools by style:**

| Style | Approach | Tool |
|-------|---------|------|
| Realistic clean | No stylization — albedo as-is | — |
| Low-poly flat | Quantize atlas colors to N palette | k-means on atlas pixels |
| Cartoon toon | Edge-preserve smooth + outline | bilateral filter + Canny |
| Ink sketch | Grayscale + ink line simulation | Controlnet scribble / custom |
| Custom | ControlNet depth+edge conditioned img2img | SD + ControlNet |

**Research questions:**
- For toon shading: how many quantized colors produces a readable but non-ugly
  palette? 8? 16? Does it vary by garment type vs. skin?
- ControlNet stylization changes colors arbitrarily. For our use case we want color
  PRESERVATION with style change. Can we constrain ControlNet with a color palette
  reference?
- Should stylization happen per-atlas (per garment/part), or on the final assembled
  render? Per-atlas is more controllable; assembled render gives better cross-part
  consistency.

**Experiment to run:**
Apply toon quantization at 4, 8, 16, and 32 colors to 5 garment types.
Rate perceived quality and identity preservation (does it still look like the same shirt?).

---

### Sub-step 7: Atlas packing and quality checks
**Task:** Combine sub-region outputs into a final 1024×1024 atlas per mesh.
Run automated quality gates before allowing the user to proceed.

**Quality checks to implement:**

| Check | Method | Threshold |
|-------|--------|-----------|
| UV coverage | Ratio of non-zero pixels in atlas | > 80% — flag otherwise |
| Seam continuity | Color delta across UV seam edges | < 15 delta-E |
| Artifact detection | Variance spikes in otherwise smooth regions | Flag top 1% variance regions |
| Color range sanity | Check for pure black (0,0,0) or pure white (255,255,255) large regions | > 5% coverage = flag |
| Back/front consistency | Mean color difference between front-projected and back-projected regions of same garment | < 20 delta-E |

**Experiment to run:**
Collect 20 completed atlases (mix of clean and known-bad). Tune thresholds so
the quality checker catches all known-bad cases with < 2 false positives.

---

## Recommended Research Sequencing

Run experiments in this order. Each informs the next.

1. **Segmentation accuracy** (Sub-step 1) — everything downstream depends on clean masks.
   Block: 1 week, 50 photos, measure IoU per category.

2. **Intrinsic decomposition on clothing** (Sub-step 2) — this is the core novel question.
   Block: 2 weeks, evaluate `compphoto/Intrinsic` on clothing specifically.

3. **UV projection method selection** (Sub-step 4) — TexDreamer vs. direct projection.
   Block: 1 week, 10 garments with ground-truth front+back.

4. **Inpainting strategy** (Sub-step 5) — mirror vs. tile vs. diffusion.
   Block: 1 week, 10 patterned garments.

5. **Stylization quality** (Sub-step 6) — last, after albedo pipeline is stable.
   Block: 1 week, palette quantization experiments.

6. **End-to-end integration test** — run the full sub-pipeline on 10 people.
   Block: 1 week, identify emergent failure modes.

**Total estimated research time before production code:** 7 weeks.
This is the single largest time investment in the project.

---

## Open questions that require decisions before implementation

**Q1: Single UV layout or per-part UV?**
SMPL-X has a single UV map for the entire body. Our garment meshes have separate
UV maps. TexDreamer outputs SMPL UV. Do we:
(a) Use SMPL UV for everything and bake garments into body UV space, or
(b) Maintain separate UV per mesh and use TexDreamer only for the body/face, with
    a separate UV projection for garments?

Recommendation: (b) — separate UV per mesh. Cleaner boundaries, easier to update
individual garments without re-baking the full body.

**Q2: How do we handle garment occlusion between layers?**
A jacket over a shirt means the shirt's back is occluded in the front photo AND
the jacket front is visible. The shirt's front UV is also partially occluded by
the jacket overlap. We need to decide: project shirt first (and let jacket projection
overwrite), or mask by layer order?

Recommendation: Project outermost layer first, then inward. The shirt's occluded
region falls back to inpainting. This matches how physical layering works.

**Q3: Are 4 input photos enough?**
For most garments: yes. Exceptions:
- Complex hairstyles with depth (afros, elaborate curls) — side photos help
- Accessories at unusual angles (backpack, belt from behind)

Consider: allow user to optionally upload additional photos at specific angles
for complex cases. The pipeline treats them as additional projection sources.

---

## Tools reference

| Tool | Purpose | License | Link |
|------|---------|---------|------|
| SAM2 | Segmentation | Apache 2.0 | https://github.com/facebookresearch/sam2 |
| compphoto/Intrinsic | Intrinsic decomp | MIT | https://github.com/compphoto/Intrinsic |
| TexDreamer | Image → SMPL UV | Research | https://github.com/ggxxii/texdreamer |
| SMPLitex | Human UV estimation | Research | https://github.com/dancasas/SMPLitex |
| Pix2Surf | Garment UV (silhouette) | Research | https://arxiv.org/abs/2003.02050 |
| Stable Diffusion inpaint | Occluded region fill | CreativeML | via diffusers |
| DensePose | Pixel → SMPL surface | CC-BY-NC | via detectron2 |
