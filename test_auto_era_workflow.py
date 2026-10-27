"""Headless checks for automatic line selection, ERA metrology and exports."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

import ler_lwr_app as app


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def stripe_image():
    y, x = np.mgrid[:180, :220]
    image = np.full(x.shape, 35.0)
    for left, width in ((22.3, 28), (86.3, 30), (150.3, 32)):
        drift = 1.2 * np.sin(y / 19)
        rising = 1 / (1 + np.exp(np.clip(-(x - left - drift) / 0.9, -60, 60)))
        falling = 1 / (1 + np.exp(np.clip(-(x - left - width - drift) / 0.9, -60, 60)))
        image += 170 * (rising - falling)
    return np.clip(image, 0, 255).astype(np.uint8)


def headless_app(image):
    state = SimpleNamespace(
        processed_image=image, raw_image=image, image_path=None,
        era_line_samples=[], era_aggregate_result=None, era_lcdu_cd_samples_nm=[],
        lcdu_cd_samples_nm=[], era_active_sample_index=None, era_pending_roi_target=None,
        pixel_size_var=Value("0.2"), era_polarity_var=Value(app.ERA_POLARITY_AUTO),
        sigma_multiplier_var=Value(1.0), auto_line_polarity_var=Value(app.AUTO_LINE_BRIGHT),
        auto_line_use_roi_var=Value(False), edge_detector_var=Value(app.EDGE_DETECTOR_ERA),
        auto_line_mode_var=Value(app.AUTO_MODE_SOLID),
        periodic_reference_interval_var=Value("1"), periodic_reference_kind_var=Value("请选择"),
        periodic_target_kind_var=Value("槽"), periodic_left_position_var=Value("亮带中心"),
        periodic_right_position_var=Value("亮带中心"), periodic_bands=[],
        periodic_origin_px=(0, 0), periodic_preview_signature=None, display_scale=1.0,
        periodic_preview_visible_var=Value(True),
        status_var=Value(""), result_var=Value(""), recognition_duration_var=Value(""),
        canvas=Mock(), result=None, analysis_origin=None, bcp_result=None,
        annotations=[], show_era_rois=True, ler_lwr_roi_mode=False,
        bcp_metrics_finalized=False, rotation_degrees=0, fit_left_color="#00e5ff",
        fit_right_color="#ffb000", roi_canvas=None,
    )
    for name in (
        "refresh_era_sample_list", "update_lcdu_summary_text", "clear_auto_line_candidates",
        "clear_detected_edges", "clear_bcp_completion_selection", "update_edge_kernel_preview",
        "record_undo_state", "draw_image",
    ):
        setattr(state, name, Mock())
    state.era_polynomial_degree = lambda: 3
    state.outlier_threshold_px = lambda: 12.0
    state.measurement_pixel_size = lambda: float(state.pixel_size_var.get())
    state.fit_line_width = lambda: 2
    for name in (
        "invalidate_era_results", "run_era_analysis", "recognize_era_lines",
        "active_lcdu_samples", "lcdu_sigma_nm", "update_era_result_text", "export_csv",
        "draw_exported_era_edges",
        "automatic_line_region", "periodic_line_settings", "clear_periodic_preview",
        "periodic_options_changed", "preview_periodic_lines", "draw_periodic_preview",
        "periodic_scope_changed",
    ):
        setattr(state, name, MethodType(getattr(app.LERLWRApp, name), state))
    return state


class AutomaticEraWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.image = stripe_image()
        self.state = headless_app(self.image)

    def test_complete_bright_and_dark_metrology_and_repeat_lcdu(self):
        for bright in (True, False):
            state = headless_app(self.image if bright else 255 - self.image)
            state.era_polarity_var.set(app.ERA_POLARITY_RISING)
            state.era_line_samples = app.automatic_era_line_samples(state.processed_image, bright)
            self.assertEqual(len(state.era_line_samples), 3)
            state.run_era_analysis()
            self.assertEqual(state.era_aggregate_result.line_count, 3)
            for sample, width in zip(state.era_line_samples, (28, 30, 32)):
                self.assertAlmostEqual(sample.paired_result.mean_width_nm, width * 0.2, delta=0.15)
                expected_left = app.ERA_POLARITY_RISING if bright else app.ERA_POLARITY_FALLING
                self.assertEqual(sample.left_edge.polarity, expected_left)
                self.assertGreaterEqual(len(sample.paired_result.rows_px), 150)
            self.assertAlmostEqual(state.lcdu_sigma_nm(), 0.4, delta=0.06)
            state.run_era_analysis()
            self.assertEqual(len(state.era_lcdu_cd_samples_nm), 3)

    def test_preview_replaces_samples_and_requires_no_calibration(self):
        self.state.pixel_size_var.set("")
        self.state.recognize_era_lines()
        self.state.recognize_era_lines()
        self.assertEqual(len(self.state.era_line_samples), 3)
        self.assertIsNone(self.state.result)
        self.assertTrue(self.state.show_era_rois)
        self.assertEqual(self.state.record_undo_state.call_count, 2)

    def test_bad_calibration_or_empty_detection_preserves_existing_samples(self):
        previous = [app.EraLineSample()]
        self.state.era_line_samples = previous
        with patch.object(app.messagebox, "showwarning") as warning:
            self.state.pixel_size_var.set("nan")
            self.state.recognize_era_lines(measure=True)
            self.assertIs(self.state.era_line_samples, previous)
            self.state.processed_image = np.zeros(self.image.shape, dtype=np.uint8)
            self.state.recognize_era_lines()
            self.assertIs(self.state.era_line_samples, previous)
            self.assertEqual(warning.call_count, 2)

    def test_roi_offsets_are_global_and_explicit(self):
        self.state.auto_line_use_roi_var.set(True)
        region = self.image[15:165, 10:205]
        self.state.selected_roi = lambda: (region, 10, 15)
        self.state.recognize_era_lines(measure=True)
        self.assertEqual(self.state.era_aggregate_result.line_count, 3)
        for sample in self.state.era_line_samples:
            self.assertEqual(sample.left_edge.roi_bounds_px[1::2], (15, 165))
            self.assertGreaterEqual(np.min(sample.left_edge.rows_px), 15)
            self.assertLess(np.max(sample.left_edge.rows_px), 165)

    def test_invalid_automatic_sample_is_reported_and_excluded(self):
        self.state.era_line_samples = app.automatic_era_line_samples(self.image)
        self.state.era_line_samples.append(app.EraLineSample(
            (0, 0, 12, 180), (0, 0, 12, 180), selection_source="automatic", line_polarity="bright",
        ))
        self.state.run_era_analysis()
        self.assertEqual(self.state.era_aggregate_result.line_count, 3)
        self.assertTrue(self.state.era_line_samples[-1].measurement_error)
        self.assertIsNone(self.state.era_line_samples[-1].paired_result)
        self.assertIn("失败 1 根", self.state.result_var.get())

    def test_failed_rerun_clears_previous_metrology_and_lcdu(self):
        self.state.recognize_era_lines(measure=True)
        self.assertIsNotNone(self.state.result)
        self.state.processed_image = np.full(self.image.shape, 35, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "均未通过"):
            self.state.run_era_analysis()
        self.assertIsNone(self.state.result)
        self.assertIsNone(self.state.era_aggregate_result)
        self.assertEqual(self.state.era_lcdu_cd_samples_nm, [])

    def test_missing_line_rows_do_not_fit_background_noise(self):
        rng = np.random.default_rng(17)
        image = self.image.astype(float)
        image[:55] = 35 + rng.normal(0, 1.0, size=image[:55].shape)
        state = headless_app(image)
        state.recognize_era_lines(measure=True)
        self.assertEqual(state.era_aggregate_result.line_count, 3)
        for sample in state.era_line_samples:
            self.assertGreaterEqual(np.min(sample.paired_result.rows_px), 55)
            self.assertGreaterEqual(sample.left_edge.rejected_rows, 55)
            self.assertGreaterEqual(sample.right_edge.rejected_rows, 55)

    def test_undo_restores_matching_line_samples_metrology_and_rotation(self):
        state = self.state
        state.recognize_era_lines(measure=True)
        original_cd = state.era_aggregate_result.mean_width_nm
        state.active_annotation_index = None
        state.bcp_use_dog_var = Value(False)
        state.normalize_var = Value(True)
        state.gaussian_denoise_var = Value(True)
        state.gaussian_kernel_size_var = Value("3 × 3")
        state.gaussian_sigma_scale_var = Value(1.0)
        state.blur_method_var = Value(app.BLUR_METHOD_GAUSSIAN)
        state.rotation_var = Value("0.00")
        snapshot = app.LERLWRApp.snapshot_editor_state(state)
        state.undo_history = [snapshot]
        state.era_line_samples.clear()
        state.era_aggregate_result = None
        state.era_lcdu_cd_samples_nm.clear()
        state.rotation_degrees = 9.0
        for name in ("update_gaussian_kernel_preview", "rebuild_working_images", "clear_fft_analysis", "update_result_text"):
            setattr(state, name, Mock())
        app.LERLWRApp.undo_last_action(state)
        self.assertEqual(len(state.era_line_samples), 3)
        self.assertEqual(len(state.era_lcdu_cd_samples_nm), 3)
        self.assertEqual(state.rotation_degrees, 0.0)
        self.assertAlmostEqual(state.era_aggregate_result.mean_width_nm, original_cd)
        self.assertIsNotNone(state.result)

    def test_csv_records_sources_and_all_lines(self):
        self.state.recognize_era_lines(measure=True)
        with TemporaryDirectory() as folder:
            target = Path(folder) / "measurements.csv"
            with patch.object(app.filedialog, "asksaveasfilename", return_value=str(target)):
                self.state.export_csv()
            with target.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertIn(["era_automatic_line_sample_count", "3"], rows)
        self.assertIn(["multi_line_lcdu_sample_count", "3"], rows)
        metadata = next(i for i, row in enumerate(rows) if row[:2] == ["era_line_index", "selection_source"])
        self.assertEqual([row[1] for row in rows[metadata + 1:metadata + 4]], ["automatic"] * 3)
        self.assertTrue(any(row[:2] == ["era_line_index", "mean_cd_nm"] for row in rows))

    def test_export_maps_every_edge_back_to_original_orientation(self):
        self.state.recognize_era_lines(measure=True)
        self.state.rotation_degrees = 3.0
        output = Image.new("RGB", (220, 180))
        with patch.object(app, "rotate_points_about_center", wraps=app.rotate_points_about_center) as mapping:
            self.state.draw_exported_era_edges(ImageDraw.Draw(output), 220, 180)
        self.assertEqual(mapping.call_count, 6)
        for call in mapping.call_args_list:
            self.assertEqual(call.args[1:], (-3.0, 220, 180))
        self.assertIsNotNone(output.getbbox())


if __name__ == "__main__":
    unittest.main()
