"""Regression checks for shared sample geometry, unified localization and line Pitch."""
import csv
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import ler_lwr_app as app
from test_auto_era_workflow import Value, headless_app, stripe_image
from test_periodic_workflow import periodic_image, periodic_state


def configured_state(image=None, method=app.LINE_LOCALIZATION_UNIFIED):
    state = headless_app(stripe_image() if image is None else image)
    state.edge_detector_var.set(method)
    state.pitch_confirm_var = Value(False)
    state.era_sample_listbox = None
    return state


class SharedLineWorkflowTests(unittest.TestCase):
    def test_tilt_estimation_precedes_detector_choice_and_preserves_input(self):
        image = stripe_image()
        for angle in (-4.0, 4.0):
            tilted = app.rotate_grayscale_image(image, angle)
            before = tilted.copy()
            correction = app.find_rotation_to_vertical(tilted)
            self.assertAlmostEqual(correction, -angle, delta=.3)
            np.testing.assert_array_equal(tilted, before)

    def test_discovery_keeps_method_and_reuses_identical_boxes(self):
        state = configured_state()
        state.recognize_era_lines()
        self.assertEqual(state.edge_detector_var.get(), app.LINE_LOCALIZATION_UNIFIED)
        bounds = [(s.left_roi_bounds_px, s.right_roi_bounds_px) for s in state.era_line_samples]
        state.run_era_analysis()
        self.assertEqual(state.era_aggregate_result.line_count, 3)
        self.assertEqual(len(state.era_lcdu_cd_samples_nm), 3)
        self.assertEqual(bounds, [(s.left_roi_bounds_px, s.right_roi_bounds_px) for s in state.era_line_samples])
        self.assertEqual(state.result.localization_method, app.LINE_LOCALIZATION_UNIFIED)
        for sample, expected in zip(state.era_line_samples, (5.6, 6.0, 6.4)):
            self.assertAlmostEqual(sample.paired_result.mean_width_nm, expected, delta=0.1)

    def test_dark_line_polarity_and_independent_global_rows(self):
        state = configured_state(255 - stripe_image())
        state.auto_line_polarity_var.set(app.AUTO_LINE_DARK)
        state.recognize_era_lines()
        state.era_line_samples[0].right_roi_bounds_px = tuple(
            v if i % 2 == 0 else (20 if i == 1 else 170)
            for i, v in enumerate(state.era_line_samples[0].right_roi_bounds_px)
        )
        state.run_era_analysis()
        sample = state.era_line_samples[0]
        self.assertEqual(sample.left_edge.polarity, app.ERA_POLARITY_FALLING)
        self.assertEqual(sample.right_edge.polarity, app.ERA_POLARITY_RISING)
        self.assertEqual((min(sample.paired_result.rows_px), max(sample.paired_result.rows_px)), (20, 169))

    def test_gradient_dropout_rows_remain_absent_and_split_overlays(self):
        image = stripe_image()
        image[50:70] = 35
        state = configured_state(image)
        state.recognize_era_lines(measure=True)
        for sample in state.era_line_samples:
            self.assertFalse(np.any((sample.paired_result.rows_px >= 52) & (sample.paired_result.rows_px < 68)))
            self.assertGreaterEqual(len(app.single_edge_segments(sample.left_edge, True)), 2)

    def test_symmetric_double_step_rejects_as_ambiguous(self):
        x = np.arange(40)
        # Two equally strong rising transitions in one claimed single-edge rectangle.
        # The unified localizer rejects the row as ambiguous_peak; with every row
        # ambiguous the whole side fails with a ValueError carrying a diagnostic record.
        profile = 30 + 80 / (1 + np.exp(-(x - 12))) + 80 / (1 + np.exp(-(x - 23)))
        image = np.tile(profile, (180, 1))
        with self.assertRaises(ValueError) as ctx:
            app.analyze_sample_edge(image, (0, 0, 40, 180), 0.2, app.LINE_LOCALIZATION_UNIFIED,
                                    polarity=app.ERA_POLARITY_RISING)
        # The failure is a LocalizationFailure carrying the full per-row diagnostics.
        self.assertIsInstance(ctx.exception, app.LocalizationFailure)
        self.assertIsNotNone(ctx.exception.diagnostics)
        self.assertTrue(all(row.reason == "ambiguous_peak" for row in ctx.exception.diagnostics.rows))

    def test_legacy_detector_strings_are_rejected(self):
        x = np.arange(40)
        profile = 30 + 80 / (1 + np.exp(-(x - 12)))
        image = np.tile(profile, (180, 1))
        for stale in ("Scharr", "Canny", "拉普拉斯", "周期亮边（半高取点）"):
            with self.assertRaises(ValueError):
                app.analyze_sample_edge(image, (0, 0, 40, 180), 0.2, stale, polarity=app.ERA_POLARITY_RISING)

    def test_preview_is_not_exportable_metrology(self):
        state = configured_state()
        state.recognize_era_lines()
        state.run_era_analysis(preview=True)
        self.assertIsNone(state.result)
        self.assertIsNone(state.era_aggregate_result)
        self.assertFalse(state.era_lcdu_cd_samples_nm)
        self.assertTrue(all(s.left_edge is not None and s.paired_result is None for s in state.era_line_samples))
        self.assertFalse(state.show_era_rois)
        self.assertFalse(state.periodic_preview_visible_var.get())

    def test_preview_hides_boxes_and_labels_keeps_geometry_and_export_crop(self):
        state = periodic_state()
        state.recognize_era_lines()
        bounds = [(s.left_roi_bounds_px, s.right_roi_bounds_px) for s in state.era_line_samples]
        state.periodic_preview_visible_var.set(True)
        state.roi_canvas = (10, 15, 270, 165)
        state.canvas.reset_mock()
        state.run_era_analysis(preview=True)
        self.assertIsNone(state.roi_canvas)
        self.assertEqual(state.export_roi_bounds_px, (10, 15, 270, 165))
        self.assertEqual(bounds, [(s.left_roi_bounds_px, s.right_roi_bounds_px) for s in state.era_line_samples])
        state.canvas.create_rectangle.assert_not_called()
        state.canvas.create_text.assert_not_called()
        self.assertTrue(all(s.left_edge is not None and s.right_edge is not None for s in state.era_line_samples))
        state.show_era_rois = True
        state.draw_era_rois()
        self.assertTrue(state.canvas.create_rectangle.called)

    def test_parameter_change_clears_all_traces_and_statistics_keeps_boxes(self):
        state = configured_state()
        state.recognize_era_lines(measure=True)
        bounds = state.era_line_samples[0].left_roi_bounds_px
        state.invalidate_line_parameters()
        self.assertIsNone(state.result)
        self.assertIsNone(state.era_aggregate_result)
        self.assertEqual(state.era_line_samples[0].left_roi_bounds_px, bounds)
        self.assertTrue(all(s.left_edge is None and s.paired_result is None for s in state.era_line_samples))

    def test_repeat_scan_identity_uses_unified_localization(self):
        state = configured_state()
        state.recognize_era_lines(measure=True)
        baseline = app.line_localization_signature(state.result)
        self.assertEqual(baseline, (app.LINE_LOCALIZATION_UNIFIED,))
        # Re-running the same scan reports the same identity; the legacy ERA container
        # marker is never used as the actual algorithm once localization_method is set.
        self.assertEqual(baseline, app.line_localization_signature(replace(state.result)))
        state.run_era_analysis()
        self.assertEqual(app.line_localization_signature(state.result), (app.LINE_LOCALIZATION_UNIFIED,))

    def test_undo_restores_pitch_and_matching_statistics(self):
        state = configured_state()
        state.recognize_era_lines(measure=True)
        state.pitch_confirm_var.set(True)
        state.confirm_line_pitch_adjacency()
        state.active_annotation_index = None
        for name, value in (("bcp_use_dog_var", False), ("normalize_var", True),
                            ("gaussian_denoise_var", True), ("gaussian_kernel_size_var", "3 × 3"),
                            ("gaussian_sigma_scale_var", 1.0), ("blur_method_var", app.BLUR_METHOD_GAUSSIAN),
                            ("rotation_var", "0.00")):
            setattr(state, name, Value(value))
        state.undo_history = [app.LERLWRApp.snapshot_editor_state(state)]
        state.invalidate_line_parameters()
        for name in ("update_gaussian_kernel_preview", "rebuild_working_images", "clear_fft_analysis", "update_result_text"):
            setattr(state, name, Mock())
        app.LERLWRApp.undo_last_action(state)
        self.assertEqual(state.edge_detector_var.get(), app.LINE_LOCALIZATION_UNIFIED)
        self.assertEqual(state.result.localization_method, app.LINE_LOCALIZATION_UNIFIED)
        self.assertEqual(len(app.line_pitch_pairs(state.era_line_samples)), 2)
        self.assertTrue(state.pitch_confirm_var.get())

    def test_unified_localization_never_routes_analysis_to_stale_detectors(self):
        # 统一定位后已移除旧多识别器控件与 stub 回调：analyze_sample_edge 只接受
        # 统一方法或旧 ERA 容器标记，任何 Scharr/Canny/Laplacian/半高 字符串直接拒绝。
        state = configured_state()
        state.recognize_era_lines(measure=True)
        self.assertEqual(state.result.localization_method, app.LINE_LOCALIZATION_UNIFIED)
        self.assertEqual(app.line_localization_signature(state.result), (app.LINE_LOCALIZATION_UNIFIED,))

    def test_manual_completion_inherits_bright_band_identity_and_revokes_pitch(self):
        state = periodic_state()
        state.recognize_era_lines()
        state.pitch_confirm_var = Value(True)
        state.confirm_line_pitch_adjacency()
        app.LERLWRApp.new_era_line_sample(state)
        sample = state.era_line_samples[-1]
        self.assertEqual(sample.edge_profile_kind, "bright_band")
        self.assertEqual(sample.target_kind, "trench")
        self.assertEqual(sample.discovery_signature, state.periodic_preview_signature)
        self.assertIsNone(sample.left_roi_bounds_px)
        self.assertIsNone(sample.right_roi_bounds_px)
        self.assertFalse(state.pitch_confirm_var.get())
        self.assertTrue(all(s.pitch_next_id is None for s in state.era_line_samples))

    def test_single_manual_failure_is_visible_and_other_lines_still_measure(self):
        state = configured_state()
        state.recognize_era_lines()
        state.era_line_samples[0] = replace(state.era_line_samples[0], selection_source="manual", left_roi_bounds_px=(0, 0, 12, 180))
        state.run_era_analysis()
        self.assertEqual(state.era_aggregate_result.line_count, 2)
        self.assertTrue(state.era_line_samples[0].measurement_error)

    def test_automatic_scope_uses_current_then_saved_overall_crop(self):
        state = configured_state()
        state.roi_canvas = (10, 15, 205, 165)
        state.selected_roi = lambda: (state.processed_image[15:165, 10:205], 10, 15)
        state.recognize_era_lines(measure=True)
        self.assertIsNone(state.roi_canvas)
        self.assertEqual(state.export_roi_bounds_px, (10, 15, 205, 165))
        image, x, y = state.automatic_line_region()
        self.assertEqual((image.shape, x, y), ((150, 195), 10, 15))
        self.assertTrue(all(s.left_edge.roi_bounds_px[1::2] == (15, 165) for s in state.era_line_samples))

    def test_edge_editor_never_becomes_overall_export_scope(self):
        state = configured_state()
        state.export_roi_bounds_px = (10, 15, 205, 165)
        state.era_pending_roi_target = (0, "left")
        state.roi_canvas = (20, 30, 40, 150)
        self.assertEqual(state.image_export_roi_bounds(), (10, 15, 205, 165))
        state.export_roi_bounds_px = None
        self.assertIsNone(state.image_export_roi_bounds())

    def test_periodic_all_side_combinations_use_unified_localization(self):
        image = periodic_image()
        for left in ("left", "right", "center"):
            for right in ("left", "right", "center"):
                with self.subTest(left=left, right=right):
                    sample = app.periodic_line_samples(image, app.PeriodicLineSettings(left_definition=left, right_definition=right))[0]
                    edges = [app.analyze_sample_edge(image, getattr(sample, side + "_roi_bounds_px"), .2,
                             app.LINE_LOCALIZATION_UNIFIED,
                             bright_band=True, definition=definition, reference_band=getattr(sample, side + "_reference_band_px"))
                             for side, definition in (("left", left), ("right", right))]
                    result = app.pair_single_edges(*edges, .2)
                    # Low-level pairing keeps the legacy ERA container marker; the app-level
                    # run_era_analysis path stamps the unified localization method.
                    self.assertEqual(result.edge_detector, app.EDGE_DETECTOR_ERA)
                    self.assertGreater(len(result.rows_px), 140)
                    self.assertGreater(result.mean_width_nm, 5)
                    self.assertLess(result.mean_width_nm, 9)


class LinePitchTests(unittest.TestCase):
    def state_with_pitch(self):
        state = configured_state()
        state.recognize_era_lines(measure=True)
        state.pitch_confirm_var.set(True)
        state.confirm_line_pitch_adjacency()
        return state

    def test_pitch_requires_confirmation_and_reports_complete_period(self):
        state = configured_state()
        state.recognize_era_lines(measure=True)
        self.assertEqual(app.line_pitch_pairs(state.era_line_samples), [])
        state.pitch_confirm_var.set(True)
        state.confirm_line_pitch_adjacency()
        pairs = app.line_pitch_pairs(state.era_line_samples)
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            # Center shifts with unequal widths: 64 px + half the 2 px width increase.
            self.assertAlmostEqual(pair.mean_nm, 13, delta=.02)
        self.assertIn("Pitch 总平均", state.result_var.get())

    def test_deletion_failure_missing_half_and_kind_changes_never_bridge_gap(self):
        state = self.state_with_pitch()
        samples = state.era_line_samples
        self.assertEqual(app.line_pitch_pairs([samples[0], samples[2]]), [])
        self.assertEqual(app.line_pitch_pairs([samples[0], replace(samples[1], paired_result=None), samples[2]]), [])
        self.assertEqual(app.line_pitch_pairs([samples[0], replace(samples[1], target_kind="platform"), samples[2]]), [])
        self.assertEqual(app.line_pitch_pairs(samples[:1]), [])
        state.era_line_samples = [samples[0], samples[2]]
        state.confirm_line_pitch_adjacency()
        self.assertFalse(app.line_pitch_pairs(state.era_line_samples))

    def test_periodic_same_kind_full_period_and_missing_interval(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        state.pitch_confirm_var = Value(True)
        state.confirm_line_pitch_adjacency()
        pairs = app.line_pitch_pairs(state.era_line_samples)
        self.assertEqual(len(pairs), 2)
        self.assertAlmostEqual(pairs[0].mean_nm, 18, delta=.02)
        altered = [state.era_line_samples[0], replace(state.era_line_samples[1], interval_index=5)]
        self.assertFalse(app.line_pitch_pairs(altered))

    def test_common_rows_only_csv_and_overlay_use_identical_pitch(self):
        state = self.state_with_pitch()
        second = state.era_line_samples[1].paired_result
        second.rows_px = second.rows_px[:100]
        second.left_px = second.left_px[:100]
        second.right_px = second.right_px[:100]
        pairs = app.line_pitch_pairs(state.era_line_samples)
        self.assertEqual(len(pairs[0].rows_px), 100)
        labels = list(app.line_pitch_annotations(state.era_line_samples))
        self.assertIn(f"{pairs[0].mean_nm:.3f} nm", labels[0][1])
        with TemporaryDirectory() as folder:
            target = Path(folder) / "results.csv"
            with patch.object(app.filedialog, "asksaveasfilename", return_value=str(target)):
                state.export_csv()
            with target.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertIn(["line_pitch_valid_pair_count", "2"], rows)
        self.assertIn(["line_localization_method", app.LINE_LOCALIZATION_UNIFIED], rows)
        self.assertIn(["era_polynomial_degree", "3"], rows)
        self.assertIn(["era_difference_weight_power", "4"], rows)
        self.assertIn(["line_pitch_direction", "working_image_x_after_user_rotation"], rows)
        index = rows.index(["pitch_first_line_index", "pitch_second_line_index", "mean_pitch_nm", "local_along_line_1sigma_nm", "common_valid_row_count"])
        self.assertAlmostEqual(float(rows[index + 1][2]), pairs[0].mean_nm)
        self.assertEqual(rows[index + 1][4], "100")


if __name__ == "__main__":
    unittest.main()
