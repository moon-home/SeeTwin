"""
Stage 2 — SMPL-X body shape fitting.

Given MediaPipe keypoints from front, side, and back views, optimises the
SMPL-X shape parameter vector β (10 values) so that the model's projected
joints match the observed 2D keypoints as closely as possible.

No mesh is exported here — the β vector is the only output, forwarded to
Stages 3 and 5.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import smplx

logger = logging.getLogger(__name__)

MODELS_DIR = str(Path(__file__).parents[2] / "models")
NUM_BETAS = 10

# ── MediaPipe → SMPL-X joint index mapping ───────────────────────────────────
# SMPL-X joint order (first 22 body joints only):
#   0=pelvis 1=L-hip 2=R-hip 3=spine1 4=L-knee 5=R-knee 6=spine2
#   7=L-ankle 8=R-ankle 9=spine3 10=L-foot 11=R-foot 12=neck
#   13=L-collar 14=R-collar 15=head 16=L-shldr 17=R-shldr
#   18=L-elbow 19=R-elbow 20=L-wrist 21=R-wrist
#
# MediaPipe landmark indices (subset used for fitting):
#   11=L-shldr 12=R-shldr 13=L-elbow 14=R-elbow 15=L-wrist 16=R-wrist
#   23=L-hip   24=R-hip   25=L-knee  26=R-knee  27=L-ankle 28=R-ankle

_MP_TO_SMPLX: list[tuple[int, int]] = [
    (11, 16), (12, 17),  # shoulders
    (13, 18), (14, 19),  # elbows
    (15, 20), (16, 21),  # wrists
    (23,  1), (24,  2),  # hips
    (25,  4), (26,  5),  # knees
    (27,  7), (28,  8),  # ankles
]

# Approximate real-world distance between pairs of landmarks (metres).
# Used to recover a scale factor from pixel-space keypoints.
_SCALE_PAIRS: list[tuple[int, int, float]] = [
    (11, 12, 0.42),   # shoulder width
    (23, 24, 0.28),   # hip width
]


def _pixel_to_normalised(lms: list[dict], img_w: int, img_h: int
                          ) -> np.ndarray:
    """Return (33, 3) array of [x_norm, y_norm, visibility] for each landmark."""
    arr = np.zeros((33, 3), dtype=np.float32)
    for i, lm in enumerate(lms):
        arr[i, 0] = lm["x"]   # already normalised [0,1]
        arr[i, 1] = lm["y"]
        arr[i, 2] = lm["visibility"]
    return arr


def _estimate_pixel_scale(lms_norm: np.ndarray, img_w: int) -> float:
    """
    Estimate metres-per-pixel using known reference distances.
    Returns 0 if landmarks are not visible enough.
    """
    scales = []
    for a, b, real_m in _SCALE_PAIRS:
        if lms_norm[a, 2] < 0.5 or lms_norm[b, 2] < 0.5:
            continue
        pixel_dist = abs(lms_norm[a, 0] - lms_norm[b, 0]) * img_w
        if pixel_dist < 5:
            continue
        scales.append(real_m / pixel_dist)
    return float(np.mean(scales)) if scales else 0.0


def _build_target_joints(lms_front: np.ndarray | None,
                          lms_side: np.ndarray | None,
                          lms_back: np.ndarray | None,
                          img_w: int, img_h: int,
                          scale_mpp: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (N, 3) target joint positions in metres and (N,) confidence weights.
    Uses front view for X (left-right) and Y (up-down);
    side view for Z (depth); back view adds redundancy to X/Y.
    """
    joints_3d = np.zeros((22, 3), dtype=np.float32)
    weights   = np.zeros(22, dtype=np.float32)

    def add(mp_idx, smplx_idx, lms, axis_map):
        if lms is None:
            return
        lm = lms[mp_idx]
        if lm[2] < 0.4:
            return
        for src_ax, dst_ax, sign in axis_map:
            val = (lm[src_ax] - 0.5) * sign * img_w * scale_mpp
            joints_3d[smplx_idx, dst_ax] += val * lm[2]
            weights[smplx_idx] = max(weights[smplx_idx], float(lm[2]))

    for mp_idx, sx_idx in _MP_TO_SMPLX:
        # front: image-x → body-x (negated), image-y → body-y (negated)
        if lms_front is not None:
            add(mp_idx, sx_idx, lms_front,
                [(0, 0, -1.0), (1, 1, -1.0)])
        # back: image-x mirrors → body-x (positive)
        if lms_back is not None:
            add(mp_idx, sx_idx, lms_back,
                [(0, 0, +1.0), (1, 1, -1.0)])
        # side: image-x → body-z
        if lms_side is not None:
            add(mp_idx, sx_idx, lms_side,
                [(0, 2, -1.0)])

    return joints_3d, weights


def fit_body_shape(
    kp_front: dict | None,
    kp_side:  dict | None,
    kp_back:  dict | None,
    img_w: int = 512,
    img_h: int = 768,
    n_iters: int = 300,
    lr: float = 0.05,
) -> np.ndarray:
    """
    Optimise SMPL-X β (10-dim) to match observed 2D/3D keypoints.

    Args:
        kp_front / kp_side / kp_back: output dicts from extract_keypoints()
                                       (any can be None if not available)
        img_w, img_h: pixel dimensions of the input photos
        n_iters:      gradient-descent iterations
        lr:           Adam learning rate

    Returns:
        beta: float32 numpy array, shape (10,)
    """
    logger.info("Fitting SMPL-X body shape (iters=%d)…", n_iters)

    def lms_or_none(kp):
        if kp and kp.get("detected"):
            return np.array([[l["x"], l["y"], l["visibility"]] for l in kp["landmarks"]],
                            dtype=np.float32)
        return None

    lms_f = lms_or_none(kp_front)
    lms_s = lms_or_none(kp_side)
    lms_b = lms_or_none(kp_back)

    # Estimate scale from front or back view
    scale_mpp = 0.0
    for lms in [lms_f, lms_b, lms_s]:
        if lms is not None:
            s = _estimate_pixel_scale(lms, img_w)
            if s > 0:
                scale_mpp = s
                break

    if scale_mpp == 0.0:
        logger.warning("Could not estimate scale from landmarks; using default 0.0005 m/px")
        scale_mpp = 0.0005

    target_joints, weights = _build_target_joints(
        lms_f, lms_s, lms_b, img_w, img_h, scale_mpp)

    target_t = torch.tensor(target_joints, dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)

    # Load SMPL-X model
    model = smplx.create(
        MODELS_DIR, model_type="smplx",
        gender="neutral", use_pca=False, num_betas=NUM_BETAS,
        batch_size=1,
    )
    model.eval()

    beta = nn.Parameter(torch.zeros(1, NUM_BETAS))
    optim = torch.optim.Adam([beta], lr=lr)

    for i in range(n_iters):
        optim.zero_grad()
        out = model(betas=beta, return_verts=False)
        # out.joints: (1, 127, 3) — first 22 are body joints
        pred = out.joints[0, :22, :]
        diff = (pred - target_t) * weights_t.unsqueeze(1)
        loss = (diff ** 2).mean()
        # Regularise β toward zero (stay close to mean body)
        loss = loss + 0.01 * (beta ** 2).mean()
        loss.backward()
        optim.step()

        if (i + 1) % 100 == 0:
            logger.info("  iter %d  loss=%.6f", i + 1, loss.item())

    result = beta.detach().numpy()[0]
    logger.info("Fitting done. β = %s", np.round(result, 3))
    return result.astype(np.float32)


# ── Measurement helpers ───────────────────────────────────────────────────────

def beta_to_measurements(beta: np.ndarray) -> dict[str, float]:
    """
    Forward-pass SMPL-X with the given β and derive human-readable
    body measurements in centimetres.

    Returns a dict with keys:
        height_cm, shoulder_width_cm, hip_width_cm,
        torso_length_cm, inseam_cm, arm_length_cm
    """
    model = smplx.create(
        MODELS_DIR, model_type="smplx",
        gender="neutral", use_pca=False, num_betas=NUM_BETAS,
        batch_size=1,
    )
    model.eval()
    with torch.no_grad():
        beta_t = torch.tensor(beta[np.newaxis], dtype=torch.float32)
        out = model(betas=beta_t, return_verts=True)
        joints = out.joints[0].numpy()   # (127, 3)
        verts  = out.vertices[0].numpy() # (10475, 3)

    def j(idx):
        return joints[idx]

    def dist(a, b):
        return float(np.linalg.norm(j(a) - j(b)))

    # Height: top of head vertex (highest Y) to bottom of foot (lowest Y)
    height_m = float(verts[:, 1].max() - verts[:, 1].min())

    # Shoulder width: L-shoulder (16) to R-shoulder (17)
    shoulder_m = dist(16, 17)

    # Hip width: L-hip (1) to R-hip (2)
    hip_m = dist(1, 2)

    # Torso length: neck (12) to pelvis (0)
    torso_m = dist(12, 0)

    # Inseam: L-hip (1) to L-ankle (7)
    inseam_m = dist(1, 7)

    # Arm length: L-shoulder (16) to L-wrist (20)
    arm_m = dist(16, 20)

    return {
        "height_cm":         round(height_m    * 100, 1),
        "shoulder_width_cm": round(shoulder_m  * 100, 1),
        "hip_width_cm":      round(hip_m       * 100, 1),
        "torso_length_cm":   round(torso_m     * 100, 1),
        "inseam_cm":         round(inseam_m    * 100, 1),
        "arm_length_cm":     round(arm_m       * 100, 1),
    }
