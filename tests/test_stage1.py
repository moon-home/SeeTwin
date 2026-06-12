"""
Tests for Stage 1 pipeline — mask_editor and remover utilities.

These tests do NOT require the BRIA-RMBG-2.0 model to be downloaded.
They test pure numpy/PIL logic: thresholding, brush painting, mask merging.

Run with:
    pytest tests/test_stage1.py -v
"""

import numpy as np
import pytest
from PIL import Image

from pipeline.stage1_bgremove.mask_editor import (
    StrokeLayer,
    apply_threshold,
    checkerboard_background,
    composite_on_checker,
    merge,
    overlay_strokes,
)
from pipeline.stage1_bgremove.remover import (
    apply_alpha_to_image,
    resize_to_max,
)


# ── resize_to_max ────────────────────────────────────────────────────────────

class TestResizeToMax:
    def test_no_resize_when_within_bounds(self):
        img = Image.new("RGB", (800, 600))
        result = resize_to_max(img, max_px=1024)
        assert result.size == (800, 600)

    def test_wide_image_scaled_down(self):
        img = Image.new("RGB", (2048, 1024))
        result = resize_to_max(img, max_px=1024)
        assert result.width == 1024
        assert result.height == 512

    def test_tall_image_scaled_down(self):
        img = Image.new("RGB", (512, 2048))
        result = resize_to_max(img, max_px=1024)
        assert result.height == 1024
        assert result.width == 256

    def test_square_image_at_exact_limit(self):
        img = Image.new("RGB", (1024, 1024))
        result = resize_to_max(img, max_px=1024)
        assert result.size == (1024, 1024)

    def test_aspect_ratio_preserved(self):
        img = Image.new("RGB", (3000, 2000))
        result = resize_to_max(img, max_px=1024)
        original_ratio = 3000 / 2000
        result_ratio = result.width / result.height
        assert abs(original_ratio - result_ratio) < 0.01


# ── apply_threshold ──────────────────────────────────────────────────────────

class TestApplyThreshold:
    def test_above_threshold_becomes_one(self):
        alpha = np.array([[0.6, 0.8], [0.9, 1.0]], dtype=np.float32)
        result = apply_threshold(alpha, threshold=0.5)
        assert np.all(result == 1.0)

    def test_below_threshold_becomes_zero(self):
        alpha = np.array([[0.1, 0.2], [0.3, 0.49]], dtype=np.float32)
        result = apply_threshold(alpha, threshold=0.5)
        assert np.all(result == 0.0)

    def test_exact_threshold_is_foreground(self):
        alpha = np.array([[0.5]], dtype=np.float32)
        result = apply_threshold(alpha, threshold=0.5)
        assert result[0, 0] == 1.0

    def test_output_dtype_is_float32(self):
        alpha = np.random.rand(10, 10).astype(np.float32)
        result = apply_threshold(alpha, threshold=0.5)
        assert result.dtype == np.float32

    def test_output_only_contains_zero_and_one(self):
        alpha = np.random.rand(50, 50).astype(np.float32)
        result = apply_threshold(alpha, threshold=0.5)
        unique = np.unique(result)
        assert set(unique.tolist()).issubset({0.0, 1.0})


# ── StrokeLayer ──────────────────────────────────────────────────────────────

class TestStrokeLayer:
    def test_initial_state_is_empty(self):
        s = StrokeLayer(100, 100)
        assert s.is_empty()
        assert s.stroke_count() == 0

    def test_fg_paint_sets_pixels(self):
        s = StrokeLayer(100, 100)
        s.paint(50, 50, radius=5, mode="fg")
        assert s.fg_strokes[50, 50] == True
        assert s.bg_strokes[50, 50] == False

    def test_bg_paint_sets_pixels(self):
        s = StrokeLayer(100, 100)
        s.paint(50, 50, radius=5, mode="bg")
        assert s.bg_strokes[50, 50] == True
        assert s.fg_strokes[50, 50] == False

    def test_fg_cancels_bg_at_same_location(self):
        s = StrokeLayer(100, 100)
        s.paint(50, 50, radius=5, mode="bg")
        s.paint(50, 50, radius=5, mode="fg")  # fg painted over bg
        assert s.fg_strokes[50, 50] == True
        assert s.bg_strokes[50, 50] == False

    def test_bg_cancels_fg_at_same_location(self):
        s = StrokeLayer(100, 100)
        s.paint(50, 50, radius=5, mode="fg")
        s.paint(50, 50, radius=5, mode="bg")  # bg painted over fg
        assert s.bg_strokes[50, 50] == True
        assert s.fg_strokes[50, 50] == False

    def test_clear_removes_all_strokes(self):
        s = StrokeLayer(100, 100)
        s.paint(50, 50, radius=10, mode="fg")
        s.paint(20, 20, radius=5, mode="bg")
        s.clear()
        assert s.is_empty()

    def test_paint_respects_image_boundary(self):
        """Painting near edge should not raise an IndexError."""
        s = StrokeLayer(100, 100)
        s.paint(0, 0, radius=20, mode="fg")   # top-left corner
        s.paint(99, 99, radius=20, mode="bg")  # bottom-right corner

    def test_stroke_count_increases_with_painting(self):
        s = StrokeLayer(100, 100)
        s.paint(50, 50, radius=10, mode="fg")
        assert s.stroke_count() > 0

    def test_circle_radius_is_respected(self):
        """Pixels at distance > radius should NOT be painted."""
        s = StrokeLayer(200, 200)
        s.paint(100, 100, radius=10, mode="fg")
        # Pixel exactly at radius+1 distance should not be painted
        # Distance from (100,100) to (100, 112) is 12 > 10
        assert s.fg_strokes[100, 112] == False
        # Pixel within radius should be painted
        assert s.fg_strokes[100, 108] == True


# ── merge ────────────────────────────────────────────────────────────────────

class TestMerge:
    def test_fg_strokes_override_model_background(self):
        """Model says remove, user says keep → keep wins."""
        alpha = np.zeros((100, 100), dtype=np.float32)  # all background
        strokes = StrokeLayer(100, 100)
        strokes.paint(50, 50, radius=5, mode="fg")

        result = merge(alpha, strokes, threshold=0.5)
        assert result[50, 50] == 1.0

    def test_bg_strokes_override_model_foreground(self):
        """Model says keep, user says remove → remove wins."""
        alpha = np.ones((100, 100), dtype=np.float32)  # all foreground
        strokes = StrokeLayer(100, 100)
        strokes.paint(50, 50, radius=5, mode="bg")

        result = merge(alpha, strokes, threshold=0.5)
        assert result[50, 50] == 0.0

    def test_no_strokes_uses_threshold_only(self):
        alpha = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
        strokes = StrokeLayer(2, 2)  # empty

        result = merge(alpha, strokes, threshold=0.5)
        expected = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_feather_produces_non_binary_values(self):
        """With feathering, edge pixels should have intermediate values."""
        # Create a clear foreground/background split
        alpha = np.zeros((100, 100), dtype=np.float32)
        alpha[:, 50:] = 1.0  # right half is foreground
        strokes = StrokeLayer(100, 100)

        result = merge(alpha, strokes, threshold=0.5, feather_px=3)
        # The boundary column should have intermediate values after feathering
        boundary_col = result[:, 49]  # just left of the boundary
        assert np.any((boundary_col > 0.0) & (boundary_col < 1.0))

    def test_output_clipped_to_zero_one(self):
        alpha = np.random.rand(50, 50).astype(np.float32)
        strokes = StrokeLayer(50, 50)
        result = merge(alpha, strokes, threshold=0.3, feather_px=2)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ── apply_alpha_to_image ─────────────────────────────────────────────────────

class TestApplyAlphaToImage:
    def test_output_is_rgba(self):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        alpha = np.ones((100, 100), dtype=np.float32)
        result = apply_alpha_to_image(img, alpha)
        assert result.mode == "RGBA"

    def test_full_alpha_is_opaque(self):
        img = Image.new("RGB", (10, 10), color=(100, 150, 200))
        alpha = np.ones((10, 10), dtype=np.float32)
        result = apply_alpha_to_image(img, alpha)
        arr = np.array(result)
        assert np.all(arr[:, :, 3] == 255)

    def test_zero_alpha_is_transparent(self):
        img = Image.new("RGB", (10, 10), color=(100, 150, 200))
        alpha = np.zeros((10, 10), dtype=np.float32)
        result = apply_alpha_to_image(img, alpha)
        arr = np.array(result)
        assert np.all(arr[:, :, 3] == 0)

    def test_rgb_channels_preserved(self):
        img = Image.new("RGB", (10, 10), color=(100, 150, 200))
        alpha = np.ones((10, 10), dtype=np.float32)
        result = apply_alpha_to_image(img, alpha)
        arr = np.array(result)
        assert np.all(arr[:, :, 0] == 100)
        assert np.all(arr[:, :, 1] == 150)
        assert np.all(arr[:, :, 2] == 200)


# ── checkerboard_background ───────────────────────────────────────────────────

class TestCheckerboard:
    def test_output_size(self):
        bg = checkerboard_background(200, 150)
        assert bg.size == (200, 150)

    def test_output_mode(self):
        bg = checkerboard_background(100, 100)
        assert bg.mode == "RGB"

    def test_has_two_distinct_values(self):
        bg = checkerboard_background(64, 64, tile=16)
        arr = np.array(bg)[:, :, 0]  # red channel is enough
        unique_vals = np.unique(arr)
        assert len(unique_vals) == 2
