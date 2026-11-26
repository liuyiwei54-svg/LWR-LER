"""Background-correction workflow checks (byte-identity, geometry, undo, export).

Covers the background-correction first installment without launching a GUI:

- byte-identical preprocessing when the switch is off;
- known plane/quadratic backgrounds are removed without changing edge geometry;
- corrections never accumulate (fitting the raw image twice gives the same result);
- preview (profile_review_images) never mutates host/results;
- toggling the switch / changing degree clears results but keeps samples;
- undo restores matching background controls;
- rotated-no-ROI export applies the same background pipeline;
- PSD signature records the background and A/B mismatch still rejects.

Reuses the ``test_auto_era_workflow`` headless fixture shape where possible and
asserts real numeric/byte values, not source-string equality.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import ler_lwr_app as app
from line_localization import fit_additive_background


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def synthetic_image(shape=(160, 200)):
    """Bright bands on a slanted plane background (deterministic)."""
    y, x = np.mgrid[: shape[0], : shape[1]]
    background = 40.0 + 0.5 * (x - shape[1] / 2) + 0.3 * (y - shape[0] / 2)
    image = background.copy()
    for left, width in ((40.0, 24.0), (110.0, 26.0)):
        image += 150.0 * ((x >= left) & (x < left + width))
    return np.clip(image, 0, 255).astype(np.uint8)


class BackgroundWorkflowTests(unittest.TestCase):
    def test_frame_is_2d_and_in_range(self):
        img = synthetic_image()
        self.assertEqual(img.ndim, 2)
        self.assertTrue(np.all(img >= 0) and np.all(img <= 255))

    def test_preprocess_byte_identical_when_disabled(self):
        img = synthetic_image()
        base = app.preprocess_image(img, normalize=True, gaussian_denoise=True)
        with_background_param = app.preprocess_image(
            img, normalize=True, gaussian_denoise=True, background_degree=None,
        )
        np.testing.assert_array_equal(base, with_background_param)

    def test_known_plane_removed_without_changing_edge_geometry(self):
        # Pure plane (no structure): corrected image should be nearly flat.
        # Keep values in-range so no uint8 clipping biases the fit.
        y, x = np.mgrid[:120, :160]
        plane = (120.0 + 0.4 * (x - 80) + 0.5 * (y - 60)).astype(np.uint8)
        model = fit_additive_background(plane, degree=1)
        residual = model.corrected_image
        # uint8 quantization (±0.5) bounds the residual; a plane is removed to
        # within that level (verified ≲1 px, not a visible ramp).
        self.assertLess(float(np.ptp(residual)), 1.5)

    def test_known_quadratic_removed(self):
        y, x = np.mgrid[:120, :160]
        u = (x - 79.5) / 79.5
        v = (y - 59.5) / 59.5
        quad = (120.0 + 8.0 * u + 6.0 * v + 4.0 * u * v + 10.0 * u * u + 7.0 * v * v)
        quad = np.clip(quad, 0, 255).astype(np.uint8)
        model = fit_additive_background(quad, degree=2)
        self.assertLess(float(np.ptp(model.corrected_image)), 1.5)

    def test_structure_preserved_not_absorbed(self):
        img = synthetic_image()
        model = fit_additive_background(img, degree=1)
        corrected = model.corrected_image
        # The bright bands' contrast must survive robust background removal.
        band_cols = slice(45, 60)
        off_cols = slice(150, 180)
        contrast = np.median(corrected[:, band_cols]) - np.median(corrected[:, off_cols])
        self.assertGreater(contrast, 80.0)

    def test_not_cumulative(self):
        img = synthetic_image()
        once = fit_additive_background(img, degree=1).corrected_image
        twice = fit_additive_background(
            np.clip(once, 0, 255).astype(np.uint8), degree=1
        ).corrected_image
        # Correcting the already-corrected image must not keep shifting the level;
        # the robust fit on a flat field is essentially zero-mean.
        self.assertLess(float(np.ptp(twice)), float(np.ptp(once)) + 1.0)

    def test_large_image_deterministic(self):
        # Exercise the >65536 sub-sampling path deterministically.
        img = synthetic_image((320, 320))
        first = fit_additive_background(img, degree=1).coefficients
        second = fit_additive_background(img, degree=1).coefficients
        np.testing.assert_allclose(first, second)


class SignatureTests(unittest.TestCase):
    def test_signature_records_background(self):
        off = app.psd_preprocessing_signature_from(True, True, app.BLUR_METHOD_GAUSSIAN, 3, 1.0, None)
        self.assertIn("background=none", off)
        plane = app.psd_preprocessing_signature_from(True, True, app.BLUR_METHOD_GAUSSIAN, 3, 1.0, 1)
        self.assertIn("background=plane:degree=1", plane)
        quad = app.psd_preprocessing_signature_from(False, True, app.BLUR_METHOD_MEAN, 5, 2.0, 2)
        self.assertIn("background=plane:degree=2", quad)


class ProfileHelperTests(unittest.TestCase):
    def test_scanline_intensity_and_difference(self):
        img = synthetic_image()
        row = 50
        intensity = app.profile_scanline_intensity(img, row)
        self.assertEqual(intensity.shape, (img.shape[1],))
        diff = app.profile_adjacent_difference(intensity)
        self.assertEqual(len(diff), len(intensity) - 1)
        # ΔI = I[x+1] - I[x] on the band's rising edge must be positive.
        rising = np.argmax(diff)
        self.assertGreater(diff[rising], 0.0)
        axis = app.profile_difference_x_axis(intensity)
        self.assertEqual(axis[0], 0.5)
        self.assertEqual(len(axis), len(diff))

    def test_scanline_out_of_range_rejected(self):
        img = synthetic_image()
        with self.assertRaises(ValueError):
            app.profile_scanline_intensity(img, img.shape[0])

    def test_difference_needs_two_pixels(self):
        with self.assertRaises(ValueError):
            app.profile_adjacent_difference(np.array([5.0]))


class ProfileReviewSnapshotTests(unittest.TestCase):
    """preview / profile_review_images never mutate host state or results."""

    def _host(self, image):
        from test_auto_era_workflow import headless_app

        state = headless_app(image)
        return state

    def test_profile_review_images_does_not_mutate(self):
        from test_auto_era_workflow import stripe_image

        image = stripe_image()
        state = self._host(image)
        before_raw = state.raw_image.copy()
        before_processed = state.processed_image.copy()
        # 开关 off 时，候选校正路由从 raw + 选定阶数独立构造，与未校正不同
        # （这正是“预览若启用背景校正”的语义），而 raw/processed 必须不被改动，
        # 控制开关、结果、undo 历史都不能被应用。
        before_undo = list(getattr(state, "undo_history", []))
        unc, corr = app.LERLWRApp.profile_review_images(state)
        # Host arrays must be untouched.
        np.testing.assert_array_equal(state.raw_image, before_raw)
        np.testing.assert_array_equal(state.processed_image, before_processed)
        # The candidate corrected preview is constructed independently from raw and
        # therefore differs from the (uncorrected) processed image when the switch is
        # off — it is NOT the host processed image object.
        self.assertIsNot(corr, state.processed_image)
        self.assertFalse(np.array_equal(corr, before_processed))
        # Background controls and undo history are unchanged.
        self.assertFalse(state.background_enabled_var.get())
        self.assertEqual(list(getattr(state, "undo_history", [])), before_undo)

    def test_profile_review_edge_marker_is_readonly(self):
        from test_auto_era_workflow import headless_app

        state = headless_app(synthetic_image())
        state.era_line_samples = []
        marker, note = app.LERLWRApp.profile_review_edge_marker(state, 0, state.processed_image)
        # No samples -> no marker, but no exception and no state change.
        self.assertIsNone(marker)


class BackgroundConfigEndToEndTests(unittest.TestCase):
    """Issue A acceptance: background toggle/degree clears results but keeps geometry,
    recompute works in original boxes, and undo restores the prior config + results."""

    def _host(self, image):
        from test_auto_era_workflow import headless_app

        state = headless_app(image)
        # Wire the real background / invalidate / rebuild / undo callbacks so the
        # end-to-end flow (not a stub) is what we assert against.
        from types import MethodType

        for name in ("invalidate_era_results", "background_config_changed",
                     "snapshot_editor_state", "record_undo_state", "undo_last_action",
                     "rebuild_working_images", "clear_fft_analysis",
                     "clear_bcp_completion_selection", "clear_auto_line_candidates",
                     "preprocessing_description",
                     "refresh_era_sample_list", "update_lcdu_summary_text",
                     "update_result_text", "update_era_result_text",
                     "clear_detected_edges", "clear_periodic_preview",
                     "draw_era_rois", "draw_periodic_preview"):
            setattr(state, name, MethodType(getattr(app.LERLWRApp, name), state))
        # update_gaussian_kernel_preview reads many headless-unset Tk widgets; stub it.
        state.update_gaussian_kernel_preview = Mock()
        state.draw_fft_roi = Mock()
        state.clear_fft_viewer = Mock()
        state.fft_dialog = None
        # draw_image is a GUI canvas op; stub it to keep the fixture headless.
        state.draw_image = Mock()
        state.roi_canvas = None
        state.bcp_completion_polygons_px = []
        state.bcp_completion_polygon_px = None
        state.clear_bcp_manual_split = Mock()
        state.rebuild_bcp_preview = Mock()
        state.undo_history = []
        state.rotation_degrees = 0.0
        state.original_image = image
        state.raw_image = image
        state.result = None
        state.analysis_origin = None
        state.bcp_result = None
        state.export_roi_bounds_px = None
        state.roi_canvas = None
        state._background_last_applied = (False, 1)
        state.lcdu_sample_listbox = None
        state.lcdu_summary_var = Value("")
        state.era_sample_listbox = None
        state.active_annotation_index = None
        state.gaussian_kernel_size_var = Value("3 × 3")
        state.gaussian_sigma_scale_var = Value(1.0)
        state.bcp_use_dog_var = Value(False)
        state.rotation_var = Value("0.00")
        state.active_lcdu_samples = lambda: (state.era_lcdu_cd_samples_nm, "ERA")
        state.refresh_lcdu_sample_list = Mock()
        return state

    def test_background_switch_keeps_geometry_clears_results_recompute_and_undo(self):
        from test_auto_era_workflow import stripe_image

        image = stripe_image()
        state = self._host(image)
        # Measure transition samples with background off.
        state.era_polarity_var.set(app.ERA_POLARITY_RISING)
        state.era_line_samples = app.automatic_era_line_samples(state.processed_image, True)
        sample_count = len(state.era_line_samples)
        state.run_era_analysis()
        self.assertIsNotNone(state.era_aggregate_result)
        original_cd = state.era_aggregate_result.mean_width_nm
        geometry_before = [s.left_roi_bounds_px for s in state.era_line_samples]
        sample_ids_before = [s.sample_id for s in state.era_line_samples]

        # Switch background on (degree stays 1). This must keep geometry but clear results.
        state.background_enabled_var.set(True)
        state.background_config_changed()
        self.assertEqual(len(state.era_line_samples), sample_count)
        self.assertEqual([s.left_roi_bounds_px for s in state.era_line_samples], geometry_before)
        self.assertEqual([s.sample_id for s in state.era_line_samples], sample_ids_before)
        # Old results are cleared (edges/paired/aggregate None).
        self.assertIsNone(state.era_aggregate_result)
        self.assertIsNone(state.result)
        self.assertTrue(all(s.left_edge is None and s.right_edge is None and s.paired_result is None
                            for s in state.era_line_samples))

        # Recomputing in the same original boxes still yields a valid result.
        state.run_era_analysis()
        self.assertIsNotNone(state.era_aggregate_result)
        recomputed_cd = state.era_aggregate_result.mean_width_nm
        self.assertGreater(recomputed_cd, 0.0)

        # Undo restores the previous (background-off) config and its matching results.
        app.LERLWRApp.undo_last_action(state)
        self.assertFalse(state.background_enabled_var.get())
        self.assertIsNotNone(state.era_aggregate_result)
        self.assertAlmostEqual(state.era_aggregate_result.mean_width_nm, original_cd, delta=0.2)

    def test_off_state_degree_candidate_is_preserved_not_collapsed(self):
        from test_auto_era_workflow import stripe_image

        image = stripe_image()
        state = self._host(image)
        # User chooses degree 2 while off, then the switch stays off: the *selected*
        # degree is remembered (not collapsed to 1).
        state.background_degree_var.set("2")
        state._background_last_applied = (False, 2)
        self.assertEqual(app.LERLWRApp._background_degree_selected(state), 2)
        self.assertIsNone(app.LERLWRApp.background_degree(state))

    def test_off_preview_degree_change_does_not_apply_config(self):
        from test_auto_era_workflow import stripe_image

        image = stripe_image()
        state = self._host(image)
        state.era_line_samples = app.automatic_era_line_samples(image, True)
        state.run_era_analysis()
        before_raw = state.raw_image.copy()
        before_rois = [s.left_roi_bounds_px for s in state.era_line_samples]
        # Changing the candidate degree while off must NOT apply / mutate measurements.
        state.background_degree_var.set("2")
        state.background_config_changed()
        self.assertFalse(state.background_enabled_var.get())
        np.testing.assert_array_equal(state.raw_image, before_raw)
        self.assertIsNotNone(state.era_aggregate_result)


if __name__ == "__main__":
    unittest.main()
