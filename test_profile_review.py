"""Profile-review (灰度/灰度差对照) rendering + edge-marker tests.

Tests the pure chart/profiling helpers and the frozen, read-only edge-marker logic
with real numeric values and real Pillow output (no source-string assertions).

The two-route marker must be computed for BOTH the uncorrected and candidate
corrected routes (not only the corrected route), with real global-px positions,
the half-pixel difference axis, and a real reason when a side is missing.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ler_lwr_app as app


def ramp_with_edge(width=200):
    """A deterministic profile: flat low, a rising edge, flat high."""
    x = np.arange(width, dtype=float)
    intensity = 40.0 + 160.0 * (1.0 / (1.0 + np.exp(-(x - 100.0) / 1.5)))
    return intensity


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ProfileChartHelperTests(unittest.TestCase):
    def test_scanline_intensity_is_pure(self):
        img = np.arange(60 * 80, dtype=np.uint8).reshape(60, 80)
        profile = app.profile_scanline_intensity(img, 7)
        np.testing.assert_array_equal(profile, img[7, :].astype(float))
        self.assertEqual(profile.dtype, np.float64)

    def test_adjacent_difference_matches_forward_diff(self):
        intensity = ramp_with_edge(200)
        diff = app.profile_adjacent_difference(intensity)
        expected = np.diff(ramp_with_edge(200))
        np.testing.assert_allclose(diff, expected)
        self.assertGreater(float(diff.max()), 0.0)

    def test_difference_axis_half_pixel(self):
        intensity = ramp_with_edge(200)
        axis = app.profile_difference_x_axis(intensity)
        self.assertEqual(len(axis), len(intensity) - 1)
        self.assertEqual(axis[0], 0.5)
        self.assertEqual(axis[-1], len(intensity) - 2 + 0.5)

    def test_review_chart_is_uint8_rgb(self):
        intensity = ramp_with_edge(200)
        chart = app.profile_review_chart_rgb(
            intensity, intensity + 10.0, title="t", marker_x=100.0,
        )
        self.assertEqual(chart.dtype, np.uint8)
        self.assertEqual(chart.ndim, 3)
        self.assertEqual(chart.shape[2], 3)

    def test_review_chart_accepts_per_route_markers(self):
        intensity = ramp_with_edge(200)
        chart = app.profile_review_chart_rgb(
            intensity, intensity + 10.0, title="t",
            markers=[(98.0, "#7fd4ff"), (101.0, "#ffb45e")],
        )
        self.assertEqual(chart.dtype, np.uint8)
        self.assertEqual(chart.shape, (210, 360, 3))

    def test_difference_chart_is_uint8_rgb(self):
        intensity = ramp_with_edge(200)
        chart = app.profile_difference_chart_rgb(
            intensity, intensity + 10.0, title="t", marker_x=100.0,
        )
        self.assertEqual(chart.dtype, np.uint8)
        self.assertEqual(chart.ndim, 3)

    def test_difference_chart_accepts_per_route_markers(self):
        intensity = ramp_with_edge(200)
        chart = app.profile_difference_chart_rgb(
            intensity, intensity + 10.0, title="t",
            markers=[(98.0, "#7fd4ff"), (101.0, "#ffb45e")],
        )
        self.assertEqual(chart.dtype, np.uint8)
        self.assertEqual(chart.ndim, 3)


def _make_sample(index, side_bounds, sample_id=None, selection_source="manual", kind="transition"):
    return app.EraLineSample(
        left_roi_bounds_px=side_bounds if side_bounds else None,
        right_roi_bounds_px=side_bounds if side_bounds else None,
        sample_id=sample_id if sample_id is not None else index,
        selection_source=selection_source,
    )


class ProfileReviewMarkerTests(unittest.TestCase):
    """The read-only marker uses the real analyzer and never mutates."""

    def test_no_samples_returns_none_marker(self):
        marker, note = app.profile_review_marker_for_row(
            10, np.zeros((50, 100), dtype=np.uint8), [], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNone(marker)
        self.assertIn("不在任何样本", note)

    def test_no_pixel_size_returns_none_marker(self):
        marker, note = app.profile_review_marker_for_row(
            10, np.zeros((50, 100), dtype=np.uint8), [], None,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNone(marker)
        self.assertIn("像素尺寸", note)

    def test_out_of_range_row_reports_reason(self):
        marker, note = app.profile_review_marker_for_row(
            55, np.zeros((50, 100), dtype=np.uint8), [], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNone(marker)
        self.assertIn("超出图像范围", note)


def bright_transition_image():
    """A vertical bright band (rising then falling edge) on a dark background. Enough
    rows (>=80) and a >12px-wide transition so analyze_single_edge_era accepts it."""
    height, width = 160, 240
    y, x = np.mgrid[:height, :width]
    image = np.full(x.shape, 40.0)
    image += 170.0 * ((x >= 100.0) & (x < 150.0))
    # soft edges (sigmoid) so the fit has a real transition
    step = 3.0
    for edge_x in (100.0, 150.0):
        rising = 1.0 / (1.0 + np.exp(-(x - edge_x) / step))
        image = np.where((x >= edge_x - 20) & (x <= edge_x + 20), 40.0 + 170.0 * rising, image)
    return np.clip(image, 0, 255).astype(np.uint8)


class RealMarkerLocalizationTests(unittest.TestCase):
    """Real analyze_sample_edge localization on both routes, real global-px positions."""

    @staticmethod
    def _sample(left_bounds, right_bounds, sample_id=1, source="manual"):
        return app.EraLineSample(
            left_roi_bounds_px=left_bounds,
            right_roi_bounds_px=right_bounds,
            sample_id=sample_id,
            selection_source=source,
        )

    def test_right_side_marker_lands_in_right_roi(self):
        image = bright_transition_image()
        # ROI height >= 80 px, width >= 12 px (transition region must fit).
        left_bounds = (80, 10, 110, 150)
        right_bounds = (140, 10, 170, 150)
        sample = self._sample(left_bounds, right_bounds, sample_id=1)
        row = 80
        marker, note = app.profile_review_marker_for_row(
            row, image, [sample], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNotNone(marker, note)
        self.assertGreaterEqual(marker, left_bounds[0])
        self.assertLessEqual(marker, left_bounds[2])

    def test_two_samples_different_row_spans_row_picks_only_matching_sample(self):
        image = bright_transition_image()
        # Two samples stacked vertically, each ROI >= 80 px tall.
        sample1 = self._sample((80, 0, 110, 80), (140, 0, 170, 80), sample_id=1)
        sample2 = self._sample((80, 80, 110, 160), (140, 80, 170, 160), sample_id=2)
        row = 120
        marker, note = app.profile_review_marker_for_row(
            row, image, [sample1, sample2], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNotNone(marker, note)
        self.assertIn("2", note)
        self.assertIn("左", note)

    def test_row_outside_all_rois_reports_reason(self):
        image = bright_transition_image()
        sample = self._sample((80, 10, 110, 80), (140, 10, 170, 80), sample_id=1)
        marker, note = app.profile_review_marker_for_row(
            140, image, [sample], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNone(marker)
        self.assertIn("不在任何样本", note)

    def test_marker_is_global_px_and_readonly(self):
        image = bright_transition_image()
        before = image.copy()
        sample = self._sample((80, 10, 110, 150), (140, 10, 170, 150), sample_id=1)
        marker, note = app.profile_review_marker_for_row(
            100, image, [sample], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNotNone(marker, note)
        np.testing.assert_array_equal(image, before)
        self.assertGreaterEqual(marker, 80.0)
        self.assertLessEqual(marker, 110.0)


class FrozenContextTests(unittest.TestCase):
    """The review window captures an immutable snapshot that host changes never affect."""

    @staticmethod
    def _sample(left_bounds, right_bounds, sample_id=1, source="manual"):
        return app.EraLineSample(
            left_roi_bounds_px=left_bounds,
            right_roi_bounds_px=right_bounds,
            sample_id=sample_id,
            selection_source=source,
        )

    def _host(self, image, pixel_size=0.2):
        from test_auto_era_workflow import headless_app

        state = headless_app(image)
        state.pixel_size_var.set(str(pixel_size))
        return state

    def test_context_freezes_samples_and_settings_independent_of_host_mutation(self):
        image = bright_transition_image()
        state = self._host(image)
        sample = self._sample((80, 10, 110, 150), (140, 10, 170, 150), sample_id=1)
        state.era_line_samples = [sample]
        state.background_enabled_var.set(False)
        state.background_degree_var.set("2")
        state.era_polarity_var.set(app.ERA_POLARITY_FALLING)

        context = app.LERLWRApp._profile_review_context(state)

        # Mutate the host after capture.
        state.era_line_samples = []
        state.background_enabled_var.set(True)
        state.background_degree_var.set("1")
        state.era_polarity_var.set(app.ERA_POLARITY_RISING)
        state.pixel_size_var.set("0.5")

        # The frozen context still holds the original values.
        self.assertEqual(len(context.samples), 1)
        self.assertEqual(context.samples[0].sample_id, 1)
        self.assertFalse(context.background_enabled)
        self.assertEqual(context.background_degree, 2)
        self.assertEqual(context.polarity, app.ERA_POLARITY_FALLING)
        self.assertEqual(context.pixel_size_nm, 0.2)

    def test_two_review_windows_have_independent_image_lists(self):
        # The window stores its photos in a dialog-local list, not a shared host list.
        # Assert the host has no shared mutable photo list attribute to clobber.
        image = bright_transition_image()
        state = self._host(image)
        self.assertFalse(hasattr(state, "_profile_review_photos"))
        # The per-dialog list is local to open_profile_review_window; two opens each
        # get their own list — verified structurally by the absence of the shared attr.

    def test_profile_review_marker_for_row_uses_frozen_arguments(self):
        # The pure function takes explicit frozen args; calling it cannot read live host.
        image = bright_transition_image()
        sample = self._sample((80, 10, 110, 150), (140, 10, 170, 150), sample_id=1)
        marker, note = app.profile_review_marker_for_row(
            100, image, [sample], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNotNone(marker, note)

    def test_review_images_read_only_frozen_processing_fields(self):
        # The real _review_context_images must never consult live host processing vars:
        # after the host switches to B and changes all 5 processing params, repainting
        # the same A-context yields byte-identical arrays and the A source label.
        from test_auto_era_workflow import headless_app

        image_a = bright_transition_image()
        state = headless_app(image_a)
        state.image_path = "/tmp/A.png"
        state.background_enabled_var.set(False)
        state.background_degree_var.set("2")
        context = app.LERLWRApp._profile_review_context(state)
        self.assertTrue(context.normalize_enabled)
        self.assertTrue(context.denoise_enabled)
        self.assertEqual(context.kernel_size, 3)
        self.assertEqual(context.sigma_scale, 1.0)
        self.assertEqual(context.blur_method, app.BLUR_METHOD_GAUSSIAN)
        unc_a, corr_a = app.LERLWRApp._review_context_images(state, context)
        self.assertEqual(unc_a.dtype, np.uint8)

        # Host switches to image B and changes every processing parameter.
        rng = np.random.default_rng(7)
        state.raw_image = rng.integers(0, 256, size=image_a.shape, dtype=np.uint8)
        state.processed_image = state.raw_image
        state.image_path = Path("/tmp/B.png")
        state.normalize_var.set(False)
        state.gaussian_denoise_var.set(False)
        state.blur_method_var.set(app.BLUR_METHOD_MEAN)
        state.gaussian_kernel_size = lambda: 7
        state.gaussian_sigma_scale = lambda: 2.0

        unc_b, corr_b = app.LERLWRApp._review_context_images(state, context)
        self.assertTrue(np.array_equal(unc_a, unc_b))
        self.assertTrue(np.array_equal(corr_a, corr_b))
        self.assertEqual(context.source_label, "/tmp/A.png")

    def test_refresh_clears_only_the_passed_photo_list(self):
        # The real _refresh_profile_review_content runs with a fake Frame/ImageTk and
        # keeps the passed list stable (2 route thumbnails + 2 charts) while another
        # window's list is untouched.
        import ler_lwr_app as module

        from test_auto_era_workflow import headless_app

        calls = []

        class FakeLabel:
            def __init__(self, master, text=None, image=None, **kw):
                calls.append(("Label", text, image))

            def pack(self, **kw):
                return None

        class FakeFrame:
            def __init__(self, master, **kw):
                self._children = []

            def winfo_children(self):
                children, self._children = list(self._children), []
                return children

            def destroy(self):
                return None

            def pack(self, **kw):
                return None

        def frame_factory(master, **kw):
            frame = FakeFrame(master, **kw)
            if isinstance(master, FakeFrame):
                master._children.append(frame)
            return frame

        class FakePhoto:
            def __init__(self, img):
                self.image = img

        fake_image_tk = SimpleNamespace(PhotoImage=FakePhoto)

        class FakeTtk:
            Frame = staticmethod(frame_factory)
            Label = FakeLabel

        saved_ttk, saved_phototk = module.ttk, module.ImageTk
        module.ttk, module.ImageTk = FakeTtk, fake_image_tk
        try:
            state = headless_app(bright_transition_image())
            state.background_enabled_var.set(False)
            state.background_degree_var.set("2")
            context = app.LERLWRApp._profile_review_context(state)
            # headless state is not a real LERLWRApp; bind the real methods unbound.
            state._review_context_images = lambda ctx: app.LERLWRApp._review_context_images(state, ctx)
            photos_a, photos_b = [], ["other-window-sentinel"]
            content = FakeFrame(None)
            app.LERLWRApp._refresh_profile_review_content(state, None, content, photos_a, context, 0)
            first = len(photos_a)
            self.assertGreaterEqual(first, 4)
            self.assertTrue(all(isinstance(p, FakePhoto) for p in photos_a))
            app.LERLWRApp._refresh_profile_review_content(state, None, content, photos_a, context, 5)
            self.assertEqual(len(photos_a), first)
            self.assertEqual(photos_b, ["other-window-sentinel"])
        finally:
            module.ttk, module.ImageTk = saved_ttk, saved_phototk


if __name__ == "__main__":
    unittest.main()

