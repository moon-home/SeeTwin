# SeeTwin — Repository Structure

```
seetwin/
│
├── README.md
├── requirements.txt
├── app.py                              # Main Gradio app — launches all stages
├── config.yaml                         # Model paths, device, poly budgets, confidence thresholds
│
├── pipeline/                           # Pure logic — no Gradio imports anywhere in here
│   ├── __init__.py
│   │
│   ├── stage1_bgremove/
│   │   ├── remover.py                  # BRIA-RMBG-2.0 inference wrapper
│   │   ├── mask_editor.py              # Brush stroke + threshold mask ops (numpy)
│   │   └── utils.py                    # Resize to 1024, padding, format helpers
│   │
│   ├── stage2_body_shape/
│   │   ├── smplx_fitter.py             # SMPL-X silhouette + keypoint fitting → β vector
│   │   ├── keypoint_extractor.py       # MediaPipe/OpenPose wrapper for fitting input
│   │   └── utils.py
│   │
│   ├── stage3_classification/
│   │   ├── classifier.py               # Orchestrator — runs all sub-classifiers in order
│   │   ├── face_classifier.py          # FLAME fitting from face crop
│   │   ├── hair_classifier.py          # CLIP zero-shot over hair cap library labels
│   │   ├── garment_classifier.py       # ViT fine-tuned on DeepFashion taxonomy
│   │   ├── accessory_classifier.py     # CLIP zero-shot over accessory library labels
│   │   ├── manifest.py                 # AssetManifest dataclass + JSON serialization
│   │   ├── taxonomy.py                 # Full taxonomy constants — garment types, accessory types
│   │   └── confidence.py              # Threshold logic, alternative ranking
│   │
│   ├── stage4_texture/                 # Quality-critical stage — see docs/texture_extraction_research.md
│   │   ├── pipeline.py                 # Orchestrator: runs sub-steps per part
│   │   ├── segmentor.py                # SAM2 wrapper — crop part region per manifest label
│   │   ├── intrinsic.py                # Albedo/shading separation (compphoto/Intrinsic)
│   │   ├── shadow_remover.py           # Lighting normalization on segmented region
│   │   ├── uv_projector.py             # Project photo pixels → SMPL UV space (SMPLitex/TexDreamer)
│   │   ├── inpainter.py                # SD inpaint for occluded/back-facing regions
│   │   ├── stylizer.py                 # Style pass: toon, flat, ink, realistic
│   │   ├── atlas_packer.py             # Combine per-region outputs into final 1024×1024 UV atlas
│   │   └── quality_checker.py          # Automated checks: seam continuity, coverage %, artifact flags
│   │
│   ├── stage5_assembly/
│   │   ├── assembler.py                # Main orchestrator: instantiate → fit → attach → export
│   │   ├── body_instantiator.py        # Apply β params to SMPL-X mesh
│   │   ├── face_instantiator.py        # Apply shape params to FLAME mesh
│   │   ├── garment_fitter.py           # GarmentFitter interface: blend 5 presets to body shape
│   │   │                               # Swap to TailorNet here later without touching other code
│   │   ├── accessory_placer.py         # Map accessory mesh to SMPL-X attachment joint
│   │   ├── texture_applicator.py       # Assign UV atlases to mesh materials
│   │   ├── poly_checker.py             # Verify budget targets per part
│   │   └── exporter.py                 # Export assembled scene as .glb
│   │
│   ├── stage7_rigging/
│   │   ├── mixamo_guide.py             # Generate step-by-step Mixamo upload instructions
│   │   └── rigify_script.py            # Blender Python script for Rigify auto-rig
│   │
│   ├── stage8_mocap/
│   │   ├── mediapipe_capture.py        # MediaPipe Holistic keypoint extraction
│   │   ├── skeleton_retarget.py        # MediaPipe 33-point → SMPL-X joint mapping
│   │   └── websocket_server.py         # FastAPI WebSocket — streams keypoints to browser
│   │
│   └── stage9_kimodo/
│       ├── kimodo_runner.py             # Kimodo Python API wrapper
│       ├── retarget_to_soma.py          # MediaPipe keypoints → SOMA skeleton format
│       ├── pose_scorer.py               # Per-joint deviation scoring + temporal smoothing
│       ├── feedback_overlay.py          # Color-coded skeleton overlay (green/amber/red)
│       └── reference_motions/
│           ├── sign_language/           # 10 ASL sign keyframe sequences (.json, SOMA format)
│           ├── yoga/                    # 8 yoga pose sequences (.json)
│           └── dance/                   # 2 dance choreographies (.json)
│
├── ui/                                 # Gradio components — imports pipeline, never the reverse
│   ├── stage1_ui.py
│   ├── stage2_ui.py
│   ├── stage3_ui.py                    # Confirmation flow: show result → user confirms/corrects
│   ├── stage4_ui.py                    # Per-part texture preview with quality warnings
│   ├── stage5_ui.py                    # 3D assembly preview (Three.js embed)
│   ├── stage6_ui.py                    # Blender instructions + download script button
│   ├── stage7_ui.py                    # Mixamo walkthrough or Rigify instructions
│   ├── stage8_ui.py                    # WebRTC webcam + Three.js avatar viewer
│   └── stage9_ui.py                    # Pose analysis dashboard + score overlay
│
├── web/                                # Static browser assets (Three.js, WebRTC)
│   ├── viewer3d/
│   │   ├── index.html                  # Three.js GLB viewer (Stage 5 preview)
│   │   └── viewer.js
│   ├── mocap/
│   │   ├── index.html                  # Live webcam → avatar (Stage 8)
│   │   └── mocap.js
│   └── pose_analysis/
│       ├── index.html                  # Pose scoring overlay (Stage 9)
│       └── overlay.js
│
├── asset_library/                      # Fixed authored meshes — NOT gitignored (ship with app)
│   ├── README.md                       # Artist brief: poly targets, UV layout spec, naming convention
│   ├── body/
│   │   └── smplx_neutral.glb           # Placeholder until SMPL-X weights configured
│   ├── face/
│   │   └── flame_base.glb
│   ├── hair/
│   │   ├── hair_short_straight_001.glb
│   │   ├── hair_medium_wavy_001.glb
│   │   └── ...                         # ~30 total
│   ├── garments/
│   │   ├── tshirt/
│   │   │   ├── shirt_fitted_S.glb
│   │   │   ├── shirt_fitted_M.glb
│   │   │   ├── shirt_fitted_L.glb
│   │   │   ├── shirt_loose_M.glb
│   │   │   └── shirt_loose_L.glb
│   │   ├── pants/
│   │   ├── dress/
│   │   ├── jacket/
│   │   └── ...                         # ~10 garment types
│   └── accessories/
│       ├── glasses/
│       ├── bags/
│       ├── watches/
│       ├── headphones/
│       ├── hats/
│       └── ...                         # ~80 total
│
├── blender_scripts/                    # Standalone .py files — run inside Blender text editor
│   ├── import_and_assemble.py          # Auto-import all parts from session, initial placement
│   ├── cleanup_checklist.py            # Interactive checklist side panel (Blender add-on style)
│   └── export_for_mixamo.py            # Clean FBX export: Y-up, apply transforms, correct scale
│
├── models/                             # Downloaded model weights — gitignored
│   ├── bria_rmbg/
│   ├── smplx/                          # Register at mpi-inf.mpg.de first
│   ├── flame/                          # Register at flame.is.tue.mpg.de first
│   ├── intrinsic/                      # compphoto/Intrinsic weights
│   ├── sam2/
│   ├── texdreamer/
│   └── kimodo/
│
├── data/
│   └── sessions/                       # Per-session working files — gitignored
│       └── {session_id}/
│           ├── input_photos/
│           ├── stage1_masks/
│           ├── stage2_body_params.json
│           ├── stage3_manifest.json
│           ├── stage4_atlases/
│           ├── stage5_assembly.glb
│           └── final/
│
├── scripts/
│   ├── download_models.py              # Downloads all model weights on first run
│   └── validate_asset_library.py       # Checks all authored meshes: poly count, UV coverage, naming
│
├── tests/
│   ├── test_stage1_bgremove.py
│   ├── test_stage3_classifier.py
│   ├── test_stage4_texture.py          # Most important test suite
│   ├── test_stage5_assembly.py
│   ├── test_skeleton_retarget.py
│   └── fixtures/
│       ├── sample_photos/
│       ├── expected_masks/
│       └── sample_manifest.json
│
└── docs/
    ├── texture_extraction_research.md  # Technical scaffold for Stage 4 research
    ├── taxonomy.md                     # Full garment + accessory taxonomy with mesh IDs
    ├── asset_library_brief.md          # Brief for the 3D artist authoring the base meshes
    ├── smplx_uv_layout.md              # SMPL-X UV map regions and seam locations
    ├── kimodo_integration.md           # SOMA skeleton mapping details
    ├── style_guide.md                  # UI design tokens and component specs
    └── blender_guide.md                # Detailed Stage 6 walkthrough with screenshots
```

---

## Key Design Conventions

**Pipeline/UI hard separation.** `pipeline/` has zero Gradio imports. `ui/` only calls `pipeline/`. This makes every pipeline stage independently testable as a Python function and callable as a REST endpoint.

**GarmentFitter is an interface.** `stage5_assembly/garment_fitter.py` exposes a single `fit(garment_mesh_id, body_beta) -> mesh` interface. The current implementation blends 5 presets. TailorNet is a drop-in replacement later — no other code changes.

**Asset library is version-controlled.** Unlike model weights, the authored mesh library is committed to the repo. It's small (<50MB total for low-poly meshes), deterministic, and the entire pipeline depends on it being stable.

**Sessions are isolated.** Every run creates `data/sessions/{uuid}/`. Nothing is written globally. Sessions resume by loading a session ID. Intermediate files at each stage are persisted so the user can go back without re-running prior stages.

**Confirmations are blocking.** Stage 3 classifications do not queue — each waits for explicit user confirmation before the next runs. This is enforced at the pipeline level, not just the UI.

**Texture quality is instrumented.** `stage4_texture/quality_checker.py` runs automated checks after every atlas is generated and surfaces warnings in the UI before the user can proceed to Stage 5. Catching texture problems early prevents wasted Blender time.
