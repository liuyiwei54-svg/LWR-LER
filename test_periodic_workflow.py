"""Headless end-to-end checks for user-labeled trench/platform metrology."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import ler_lwr_app as app
from test_auto_era_workflow import headless_app
from test_auto_era_workflow import Value


def periodic_image(noise=0.0):
    y, x = np.mgrid[:180, :325]
    image = np.full(x.shape, 30.0)
    for center in (25.25, 60.25, 115.25, 150.25, 205.25, 240.25, 295.25):
        image += 170 * np.exp(-0.5 * ((x - center - np.sin(y / 18)) / 2.2) ** 2)
    if noise:
        image += np.random.default_rng(42).normal(0, noise, image.shape)
    return image


def periodic_state(image=None):
    state = headless_app(periodic_image() if image is None else image)
    state.auto_line_mode_var.set(app.AUTO_MODE_PERIODIC)
    state.preview_periodic_lines()
    state.periodic_reference_kind_var.set("槽")
    return state


def sem_double_peak_image():
    # SEM boundary peaks with an elevated interior: the valley remains above the
    # global foreground threshold, so threshold runs incorrectly merge each pair.
    period = np.array([40, 40, 49, 70, 119, 182, 218, 203, 155, 136,
                       158, 198, 206, 167, 92, 51, 39, 38, 40, 40,
                       40, 40, 40, 40, 40, 40, 40, 40], dtype=float)
    profile = np.concatenate((np.full(20, 40.0), np.tile(period, 6), np.full(20, 40.0)))
    return np.tile(profile, (180, 1))


class PeriodicWorkflowTests(unittest.TestCase):
    def test_sem_double_peak_body_is_two_separate_measured_bands(self):
        image = sem_double_peak_image()
        state = periodic_state(image)
        self.assertEqual(len(state.periodic_bands), 12)
        state.recognize_era_lines(measure=True)
        self.assertEqual(len(state.era_line_samples), 6)
        self.assertEqual(state.era_aggregate_result.line_count, 6)
        for index, sample in enumerate(state.era_line_samples):
            left_peak, valley, right_peak = 26 + 28 * index, 29 + 28 * index, 32 + 28 * index
            left_roi, right_roi = sample.left_roi_bounds_px, sample.right_roi_bounds_px
            self.assertLessEqual(left_roi[2], valley + 1)
            self.assertGreaterEqual(right_roi[0], valley)
            self.assertLess(left_roi[0], left_peak)
            self.assertLess(right_roi[0], right_peak)
            self.assertGreater(right_roi[2], right_peak)
            self.assertTrue(np.all(np.abs(sample.left_edge.edge_px - left_peak) < 1.5))
            self.assertTrue(np.all(np.abs(sample.right_edge.edge_px - right_peak) < 1.5))
            self.assertLess(sample.paired_result.mean_width_nm, 2.0)

    def test_periodic_band_can_be_measured_below_manual_era_minimum_width(self):
        image = np.tile([30., 30., 50., 180., 220., 180., 50., 30., 30.], (180, 1))
        result = app.analyze_bright_band_edge(image, (0, 0, 9, 180), 0.2, reference_band_px=(2.5, 5.5))
        self.assertTrue(np.all(np.abs(result.edge_px - 4.0) < 0.1))
        self.assertEqual(app.MIN_ERA_ROI_WIDTH, 12)

    def test_narrow_roi_edit_is_allowed_only_for_a_periodic_sample(self):
        state = headless_app(np.zeros((180, 50)))
        state.era_line_samples = [app.EraLineSample(edge_profile_kind="bright_band")]
        state.era_pending_roi_target = (0, "left")
        state.roi_canvas = (10, 0, 17, 180)
        self.assertEqual(app.LERLWRApp.era_roi_bounds_from_canvas(state), (10, 0, 17, 180))
        state.era_line_samples[0].edge_profile_kind = "transition"
        with self.assertRaisesRegex(ValueError, "12 px"):
            app.LERLWRApp.era_roi_bounds_from_canvas(state)

    def test_sem_double_peak_positions_keep_inner_center_outer_choices(self):
        image = sem_double_peak_image()
        widths = {}
        for left in ("left", "center", "right"):
            for right in ("left", "center", "right"):
                with self.subTest(left=left, right=right):
                    settings = app.PeriodicLineSettings(1, "trench", "trench", left, right)
                    samples = app.periodic_line_samples(image, settings)
                    self.assertEqual(len(samples), 6)
                    sample = samples[0]
                    a = app.analyze_bright_band_edge(image, sample.left_roi_bounds_px, 0.2, left,
                                                    reference_band_px=sample.left_reference_band_px)
                    b = app.analyze_bright_band_edge(image, sample.right_roi_bounds_px, 0.2, right,
                                                    reference_band_px=sample.right_reference_band_px)
                    widths[left, right] = app.pair_single_edges(a, b, 0.2).mean_width_nm
        self.assertLess(widths["right", "left"], widths["center", "center"])
        self.assertLess(widths["center", "center"], widths["left", "right"])

    def test_measurement_stays_with_preview_when_dark_interior_has_texture(self):
        image = periodic_image()
        y, x = np.mgrid[:180, :325]
        affected = (y >= 50) & (y < 90)
        original_band = np.exp(-0.5 * ((x - 25.25 - np.sin(y / 18)) / 2.2) ** 2)
        image -= np.where(affected, 160.0, 0.0) * original_band
        image += np.where(affected, 60.0, 0.0) * np.exp(-0.5 * ((x - 39.0) / 1.0) ** 2)
        state = periodic_state(image)
        self.assertAlmostEqual(state.periodic_bands[0].center_x_px, 25.25, delta=1.0)
        state.recognize_era_lines(measure=True)
        first = state.era_line_samples[0]
        self.assertIsNotNone(first.paired_result)
        self.assertTrue(np.all(np.abs(first.left_edge.edge_px - 25.25) < 4.0))
        self.assertGreaterEqual(first.left_edge.rejected_rows, 40)
        self.assertAlmostEqual(first.paired_result.mean_width_nm, 7.0, delta=0.02)
        for segment in app.single_edge_segments(first.left_edge, split_missing_rows=True):
            self.assertTrue(np.all(np.diff(segment[:, 1]) == 1))

    def test_sample_generation_uses_preview_bands_without_redetection(self):
        state = periodic_state()
        with patch.object(app, "detect_bright_bands", side_effect=AssertionError("unexpected redetection")):
            state.recognize_era_lines(measure=True)
        self.assertEqual(state.era_aggregate_result.line_count, 3)
        for sample in state.era_line_samples:
            self.assertIsNotNone(sample.left_reference_band_px)
            self.assertIsNotNone(sample.right_reference_band_px)

    def test_actual_left_right_rois_are_local_to_each_bright_band(self):
        state = periodic_state()
        state.recognize_era_lines()
        for sample in state.era_line_samples:
            left_center = np.mean(sample.left_reference_band_px)
            right_center = np.mean(sample.right_reference_band_px)
            dark_middle = 0.5 * (left_center + right_center)
            for side, own_center, other_center in (
                ("left", left_center, right_center), ("right", right_center, left_center),
            ):
                left, _, right, _ = getattr(sample, f"{side}_roi_bounds_px")
                self.assertTrue(left < own_center < right)
                self.assertFalse(left <= other_center < right)
                self.assertFalse(left <= dark_middle < right)
                self.assertLess(right - left, right_center - left_center)

    def test_periodic_sample_rois_exclude_the_other_side_band_and_dark_middle(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        self.assertEqual(len(state.era_line_samples), 3)
        for sample in state.era_line_samples:
            bands = (
                (sample.left_reference_band_px, sample.right_reference_band_px),
                (sample.right_reference_band_px, sample.left_reference_band_px),
            )
            for side, (own_band, other_band) in zip(("left", "right"), bands):
                left, top, right, bottom = getattr(sample, f"{side}_roi_bounds_px")
                self.assertEqual((top, bottom), (0, 180))
                self.assertLessEqual(left, own_band[0])
                self.assertGreaterEqual(right, own_band[1])
                other_span = np.arange(np.ceil(other_band[0]), np.ceil(other_band[1]))
                self.assertFalse(np.any((other_span >= left) & (other_span < right)))
                dark_middle = 0.5 * (np.mean(sample.left_reference_band_px) + np.mean(sample.right_reference_band_px))
                self.assertFalse(left <= dark_middle < right)
            # The ROI never grows beyond the bright body plus a constant 3 px dark
            # margin, and it never exceeds the 12 px geometric minimum by more than
            # the same margin: a wide band's half-height width must not widen the box.
            body_left, body_right = sample.left_roi_bounds_px[0] + 3, sample.left_roi_bounds_px[2] - 3
            self.assertLessEqual(body_right - body_left, 12)
            self.assertLessEqual(sample.left_roi_bounds_px[2] - sample.left_roi_bounds_px[0], 12 + 2 * 3)
            self.assertLessEqual(sample.right_roi_bounds_px[2] - sample.right_roi_bounds_px[0], 12 + 2 * 3)

    def test_selected_band_overlay_draws_the_measured_roi_and_a_separate_band_reference(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        state.show_era_rois = True
        state.periodic_preview_visible_var.set(True)
        state.canvas.create_rectangle.reset_mock()
        app.LERLWRApp.draw_era_rois(state)
        rectangles = [call for call in state.canvas.create_rectangle.call_args_list if call.kwargs.get("tags") == "era_roi"]
        self.assertEqual(len(rectangles), 6)
        for sample, (left, right) in zip(state.era_line_samples, zip(rectangles[::2], rectangles[1::2])):
            self.assertEqual((left.args[0], left.args[2]), (sample.left_roi_bounds_px[0], sample.left_roi_bounds_px[2]))
            self.assertEqual((left.args[1], left.args[3]), (sample.left_roi_bounds_px[1], sample.left_roi_bounds_px[3]))
            self.assertEqual((right.args[0], right.args[2]), (sample.right_roi_bounds_px[0], sample.right_roi_bounds_px[2]))
            self.assertEqual((right.args[1], right.args[3]), (sample.right_roi_bounds_px[1], sample.right_roi_bounds_px[3]))
        references = [call for call in state.canvas.create_rectangle.call_args_list if call.kwargs.get("tags") == "era_band_reference"]
        self.assertEqual(len(references), 6)
        for sample, (left, right) in zip(state.era_line_samples, zip(references[::2], references[1::2])):
            self.assertEqual((left.args[0], left.args[2]), sample.left_reference_band_px)
            self.assertEqual((right.args[0], right.args[2]), sample.right_reference_band_px)
            self.assertLess(left.args[2] - left.args[0], sample.left_roi_bounds_px[2] - sample.left_roi_bounds_px[0])

    def test_wide_flat_and_near_neighbour_periodic_bands_keep_local_rois(self):
        cases = {
            "near_neighbour": [(120.0 + 14.0 * index, 1.0, "gaussian") for index in range(8)],
            "wide_flat": [(60.0 + 40.0 * index, 24.0, "flat") for index in range(6)],
        }
        for name, specs in cases.items():
            with self.subTest(name=name):
                y, x = np.mgrid[:180, :320]
                image = np.full(x.shape, 25.0)
                for center, size, shape in specs:
                    if shape == "flat":
                        image += 175.0 * ((x >= center - size / 2.0) & (x < center + size / 2.0))
                    else:
                        image += 175.0 * np.exp(-0.5 * ((x - center) / size) ** 2)
                state = periodic_state(np.clip(image, 0, 255).astype(np.uint8))
                self.assertEqual(len(state.periodic_bands), len(specs))
                state.recognize_era_lines(measure=True)
                self.assertTrue(state.era_line_samples)
                dark_middles = []
                for sample in state.era_line_samples:
                    left_center = np.mean(sample.left_reference_band_px)
                    right_center = np.mean(sample.right_reference_band_px)
                    dark_middles.append(0.5 * (left_center + right_center))
                for sample in state.era_line_samples:
                    own = (sample.left_reference_band_px, sample.right_reference_band_px)
                    for side, (own_band, other_band) in zip(("left", "right"), (own, own[::-1])):
                        left, _, right, _ = getattr(sample, f"{side}_roi_bounds_px")
                        self.assertLessEqual(left, own_band[0])
                        self.assertGreaterEqual(right, own_band[1])
                        other_span = np.arange(np.ceil(other_band[0]), np.ceil(other_band[1]))
                        self.assertFalse(np.any((other_span >= left) & (other_span < right)))
                        for midpoint in dark_middles:
                            self.assertFalse(left <= midpoint < right)
                        body_width = 5 if name == "near_neighbour" else 24
                        self.assertLessEqual(right - left, body_width + 2 * 3 + 2)
                    self.assertIsNotNone(sample.left_edge)
                    self.assertGreaterEqual(sample.left_edge.rows_px.size, 30)

    def test_physical_labels_require_user_reference(self):
        state = headless_app(periodic_image())
        state.auto_line_mode_var.set(app.AUTO_MODE_PERIODIC)
        with patch.object(app.messagebox, "showwarning") as warning:
            state.recognize_era_lines(measure=True)
        warning.assert_called_once()
        self.assertEqual(len(state.periodic_bands), 7)
        self.assertEqual(state.era_line_samples, [])
        self.assertEqual(state.periodic_reference_kind_var.get(), "请选择")

    def test_oversized_requested_rois_produce_no_end_to_end_sample(self):
        # Explicitly requesting 12 px leaves no safe window at 12 px pitch, so the preview must fail as a
        # whole: no two end-point bands may survive to be paired end-to-end into
        # one 132 px wide "line" that would still produce CD/LCDU samples.
        y, x = np.mgrid[:180, :400]
        image = np.full(x.shape, 25.0)
        for center in (100.0 + 12.0 * index for index in range(12)):
            image += 175.0 * np.exp(-0.5 * ((x - center) / 1.0) ** 2)
        image += np.random.default_rng(7).normal(0, 1.0, image.shape)
        state = headless_app(np.clip(image, 0, 255))
        state.auto_line_mode_var.set(app.AUTO_MODE_PERIODIC)
        detector = app.detect_bright_bands
        with patch.object(app, "detect_bright_bands", side_effect=lambda region: detector(region, minimum_roi_width=12)), patch.object(app.messagebox, "showwarning") as warning:
            state.preview_periodic_lines()
            self.assertEqual(state.periodic_bands, [])
            state.periodic_reference_kind_var.set("槽")
            with patch.object(app.messagebox, "showwarning"):
                state.recognize_era_lines(measure=True)
        warning.assert_called_once()
        self.assertEqual(state.era_line_samples, [])
        self.assertIsNone(state.era_aggregate_result)
        self.assertEqual(state.era_lcdu_cd_samples_nm, [])
        self.assertEqual(state.lcdu_cd_samples_nm, [])

    def test_trench_platform_parities_and_relabeling_reference(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        self.assertEqual([s.interval_index for s in state.era_line_samples], [1, 3, 5])
        self.assertAlmostEqual(state.era_aggregate_result.mean_width_nm, 7.0, delta=0.02)
        state.periodic_target_kind_var.set("平台")
        state.periodic_options_changed()
        self.assertIsNone(state.result)
        with self.assertRaisesRegex(ValueError, "规则已变化"):
            state.run_era_analysis()
        state.recognize_era_lines(measure=True)
        self.assertEqual([s.interval_index for s in state.era_line_samples], [2, 4, 6])
        self.assertAlmostEqual(state.era_aggregate_result.mean_width_nm, 11.0, delta=0.02)
        state.periodic_reference_interval_var.set("2")
        state.periodic_reference_kind_var.set("平台")
        state.periodic_target_kind_var.set("槽")
        state.recognize_era_lines(measure=True)
        self.assertEqual([s.interval_index for s in state.era_line_samples], [1, 3, 5])

    def test_all_nine_left_right_position_combinations(self):
        half_width = 2.2 * np.sqrt(2 * np.log(2))
        shift = {"center": 0.0, "left": -half_width, "right": half_width}
        image = periodic_image()
        for left in ("center", "left", "right"):
            for right in ("center", "left", "right"):
                settings = app.PeriodicLineSettings(1, "trench", "trench", left, right)
                sample = app.periodic_line_samples(image, settings)[0]
                a = app.analyze_bright_band_edge(image, sample.left_roi_bounds_px, 0.2, left)
                b = app.analyze_bright_band_edge(image, sample.right_roi_bounds_px, 0.2, right)
                result = app.pair_single_edges(a, b, 0.2)
                expected = (35 + shift[right] - shift[left]) * 0.2
                self.assertAlmostEqual(result.mean_width_nm, expected, delta=0.04)
                self.assertEqual(a.polynomial_degree, 0)
                self.assertGreaterEqual(len(result.rows_px), 150)

    def test_definition_changes_invalidate_and_do_not_append_lcdu(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        state.periodic_left_position_var.set("亮带右侧")
        state.periodic_right_position_var.set("亮带左侧")
        state.periodic_options_changed()
        self.assertIsNone(state.result)
        self.assertEqual(state.era_lcdu_cd_samples_nm, [])
        state.recognize_era_lines(measure=True)
        self.assertLess(state.era_aggregate_result.mean_width_nm, 6.1)
        state.recognize_era_lines(measure=True)
        self.assertEqual(len(state.era_line_samples), 3)
        self.assertEqual(len(state.era_lcdu_cd_samples_nm), 3)

    def test_preview_labels_and_image_changes_reset_reference(self):
        state = periodic_state()
        state.draw_periodic_preview()
        labels = [call.kwargs.get("text", "") for call in state.canvas.create_text.call_args_list]
        # Only the calibration reference keeps "区N 槽/平台"; the others show the bare number.
        self.assertIn("区1 槽", labels)
        self.assertIn("2", labels)
        self.assertNotIn("区2 平台", labels)
        rows = [call.kwargs.get("font") for call in state.canvas.create_text.call_args_list]
        self.assertTrue(all(font == ("Helvetica", app.PERIODIC_LABEL_FONT_PX) for font in rows))
        self.assertGreaterEqual(len(state.periodic_label_rows[1] + state.periodic_label_rows[2]), len(labels))
        # Labels sit on fixed screen rows inside the ROI top, never above/at the ROI edge.
        placed = [call for call in state.canvas.create_text.call_args_list]
        roi_top = min(band.roi_bounds_px[1] for band in state.periodic_bands)
        tops = [call.args[1] for call in placed if call.args]
        self.assertTrue(tops)
        self.assertTrue(all(y >= roi_top + app.PERIODIC_LABEL_ROW_INSET_PX for y in tops))
        self.assertTrue(set(tops) <= {roi_top + app.PERIODIC_LABEL_ROW_INSET_PX,
                                      roi_top + app.PERIODIC_LABEL_ROW_INSET_PX + app.PERIODIC_LABEL_ROW_GAP_PX})
        self.assertGreaterEqual(app.PERIODIC_LABEL_ROW_GAP_PX, abs(app.PERIODIC_LABEL_FONT_PX) + 8)
        left = min(band.left_half_px for band in state.periodic_bands)
        right = max(band.right_half_px for band in state.periodic_bands)
        for call in placed:
            x = call.args[0]
            half = (state.periodic_label_width(call.kwargs["text"]) + 2 * app.PERIODIC_LABEL_PAD_PX) / 2
            self.assertGreaterEqual(x - half, left - 1e-6)
            self.assertLessEqual(x + half, right + 1e-6)
        self.assertEqual(state.periodic_interval_label(2), "区2 平台")
        self.assertEqual(state.periodic_interval_label(1), "区1 槽")
        state.processed_image = state.processed_image.copy()
        state.preview_periodic_lines()
        self.assertEqual(state.periodic_reference_kind_var.get(), "请选择")

    def test_hover_interval_uses_roi_origin(self):
        state = periodic_state()
        state.periodic_origin_px = (100, 40)
        state.display_scale = 0.5
        first, second = state.periodic_bands[0], state.periodic_bands[1]
        global_x = (first.center_x_px + second.center_x_px) / 2 + 100
        global_y = (first.roi_bounds_px[1] + first.roi_bounds_px[3]) / 2 + 40
        self.assertEqual(state.periodic_interval_at_canvas(global_x * 0.5, global_y * 0.5), 1)
        outside = (second.center_x_px + (state.periodic_bands[2].center_x_px - second.center_x_px) / 2 + 100) * 0.5
        self.assertEqual(state.periodic_interval_at_canvas(outside, global_y * 0.5), 2)
        self.assertIsNone(state.periodic_interval_at_canvas(global_x * 0.5, 0.0))

    def test_reference_interval_outside_band_range_has_no_label(self):
        state = periodic_state()
        interval_count = len(state.periodic_bands) - 1
        for value, expected in (("0", None), ("-1", None), (str(interval_count + 1), None),
                               ("1", 1), (str(interval_count), interval_count)):
            state.periodic_reference_interval_var.set(value)
            self.assertEqual(state.periodic_reference_index(), expected)
        state.periodic_reference_interval_var.set(str(interval_count + 1))
        with patch.object(app, "detect_bright_bands", side_effect=AssertionError("unexpected redetection")):
            state.draw_periodic_preview()
        labels = [call.kwargs.get("text", "") for call in state.canvas.create_text.call_args_list]
        self.assertTrue(all("区" not in label for label in labels))

    def test_labels_stay_inside_narrow_roi_and_small_display(self):
        state = periodic_state()
        state.periodic_origin_px = (0, 0)
        state.display_scale = 1.0
        state.periodic_bands = [
            app.BrightBandCandidate((band.roi_bounds_px[0], 90, band.roi_bounds_px[2], 90 + 40),
                                    band.center_x_px, band.left_half_px, band.right_half_px, band.support_fraction)
            for band in state.periodic_bands
        ]
        state.draw_periodic_preview()
        texts = state.canvas.create_text.call_args_list
        self.assertTrue(texts)
        # Only the first row fits wholly inside the short ROI; one label per row keeps them apart.
        self.assertEqual({call.args[1] for call in texts}, {90 + app.PERIODIC_LABEL_ROW_INSET_PX})
        self.assertEqual(len({call.args[0] for call in texts}), len(texts))
        mid = (state.periodic_bands[0].center_x_px + state.periodic_bands[1].center_x_px) / 2
        self.assertEqual(state.periodic_interval_at_canvas(mid, 100), 1)

    def test_reference_label_is_clamped_inside_narrow_roi(self):
        state = periodic_state()
        state.periodic_reference_interval_var.set("1")
        # Shrink the preview ROI horizontally: the wide "区1 槽" label must be clamped, not cut off.
        state.periodic_bands = [
            app.BrightBandCandidate(band.roi_bounds_px, band.center_x_px,
                                    band.left_half_px, band.right_half_px, band.support_fraction)
            for band in state.periodic_bands
        ]
        state.display_scale = 1.0
        state.draw_periodic_preview()
        texts = [call for call in state.canvas.create_text.call_args_list if "区" in call.kwargs.get("text", "")]
        self.assertEqual(len(texts), 1)
        x = texts[0].args[0]
        width = state.periodic_label_width(texts[0].kwargs["text"]) + 2 * app.PERIODIC_LABEL_PAD_PX
        left = min(band.left_half_px for band in state.periodic_bands)
        right = max(band.right_half_px for band in state.periodic_bands)
        self.assertGreaterEqual(x - width / 2, left - 1e-6)
        self.assertLessEqual(x + width / 2, right + 1e-6)

    def test_analysis_scope_change_invalidates_labels_results_and_lcdu(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        state.auto_line_use_roi_var.set(True)
        state.periodic_scope_changed()
        self.assertEqual(state.periodic_bands, [])
        self.assertEqual(state.periodic_reference_kind_var.get(), "请选择")
        self.assertIsNone(state.result)
        self.assertEqual(state.era_lcdu_cd_samples_nm, [])

    def test_relabeling_new_preview_cannot_reuse_old_region_samples(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        state.auto_line_use_roi_var.set(True)
        state.selected_roi = lambda: (state.processed_image[:, 40:], 40, 0)
        state.preview_periodic_lines()
        state.periodic_reference_kind_var.set("槽")
        with self.assertRaisesRegex(ValueError, "预览或分析范围已变化"):
            state.run_era_analysis()
        self.assertIsNone(state.result)
        self.assertEqual(state.era_lcdu_cd_samples_nm, [])
        state.recognize_era_lines(measure=True)
        self.assertAlmostEqual(state.era_aggregate_result.mean_width_nm, 11.0, delta=0.02)

    def test_changed_calibration_cannot_export_old_nm_values(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        state.pixel_size_var.set("0.4")
        with patch.object(app.messagebox, "showwarning") as warning, patch.object(app.filedialog, "asksaveasfilename") as save:
            state.export_csv()
        warning.assert_called_once()
        save.assert_not_called()
        state.draw_annotations = lambda: None
        app.LERLWRApp.refresh_annotation_labels(state)
        self.assertIsNone(state.result)
        self.assertEqual(state.era_lcdu_cd_samples_nm, [])

    def test_missing_bands_on_some_rows_do_not_measure_noise(self):
        image = periodic_image()
        image[:50] = 30 + np.random.default_rng(23).normal(0, 1.0, size=image[:50].shape)
        state = periodic_state(image)
        state.recognize_era_lines(measure=True)
        self.assertEqual(state.era_aggregate_result.line_count, 3)
        for sample in state.era_line_samples:
            self.assertGreaterEqual(np.min(sample.paired_result.rows_px), 50)

    def test_undo_restores_periodic_controls_from_measured_samples(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        state.active_annotation_index = None
        for name, value in (
            ("bcp_use_dog_var", False), ("normalize_var", True),
            ("gaussian_denoise_var", True), ("gaussian_kernel_size_var", "3 × 3"),
            ("gaussian_sigma_scale_var", 1.0), ("blur_method_var", app.BLUR_METHOD_GAUSSIAN),
            ("rotation_var", "0.00"),
        ):
            setattr(state, name, Value(value))
        state.undo_history = [app.LERLWRApp.snapshot_editor_state(state)]
        state.periodic_target_kind_var.set("平台")
        state.periodic_left_position_var.set("亮带右侧")
        state.periodic_options_changed()
        for name in ("update_gaussian_kernel_preview", "rebuild_working_images", "clear_fft_analysis", "update_result_text"):
            setattr(state, name, lambda *args, **kwargs: None)
        app.LERLWRApp.undo_last_action(state)
        self.assertEqual(state.periodic_target_kind_var.get(), "槽")
        self.assertEqual(state.periodic_reference_kind_var.get(), "槽")
        self.assertEqual(state.periodic_left_position_var.get(), "亮带中心")
        self.assertAlmostEqual(state.era_aggregate_result.mean_width_nm, 7.0, delta=0.02)

    def test_scope_offsets_are_working_image_coordinates(self):
        image = periodic_image()
        settings = app.PeriodicLineSettings()
        local = app.periodic_line_samples(image, settings)
        shifted = app.periodic_line_samples(image, settings, (17, 23))
        for a, b in zip(local, shifted):
            for side in ("left", "right"):
                bounds = getattr(a, f"{side}_roi_bounds_px")
                expected = tuple(value + (17 if i % 2 == 0 else 23) for i, value in enumerate(bounds))
                self.assertEqual(getattr(b, f"{side}_roi_bounds_px"), expected)
                self.assertTrue(np.allclose(np.asarray(getattr(b, f"{side}_reference_band_px")), np.asarray(getattr(a, f"{side}_reference_band_px")) + 17))

    def test_csv_saves_actual_target_pairing_and_definitions(self):
        state = periodic_state()
        state.periodic_left_position_var.set("亮带右侧")
        state.periodic_right_position_var.set("亮带左侧")
        state.recognize_era_lines(measure=True)
        with TemporaryDirectory() as folder:
            target = Path(folder) / "periodic.csv"
            with patch.object(app.filedialog, "asksaveasfilename", return_value=str(target)):
                state.export_csv()
            with target.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertIn(["edge_detector", "周期亮边（半高取点）"], rows)
        self.assertIn(["periodic_bright_band_sample_count", "3"], rows)
        self.assertIn(["era_polynomial_degree", "not_used"], rows)
        index = next(i for i, row in enumerate(rows) if row[:2] == ["era_line_index", "selection_source"])
        for row in rows[index + 1:index + 4]:
            self.assertEqual(row[6:8], ["bright_band", "trench"])
            self.assertEqual(row[-2:], ["right", "left"])

    def test_bad_roi_or_definition_is_not_silently_measured(self):
        image = periodic_image()
        with self.assertRaises(ValueError):
            app.analyze_bright_band_edge(image, (-1, 0, 30, 180), 0.2)
        with self.assertRaises(ValueError):
            app.periodic_line_samples(image, app.PeriodicLineSettings(left_definition="unknown"))
        with self.assertRaises(ValueError):
            app.periodic_line_samples(image, app.PeriodicLineSettings(reference_interval=100))


if __name__ == "__main__":
    unittest.main()
