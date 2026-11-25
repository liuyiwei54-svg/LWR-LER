"""Headless PNG checks for rectangular exports of the adjusted working view."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import ler_lwr_app as app
from test_auto_era_workflow import Value, headless_app, stripe_image


class ImageExportTests(unittest.TestCase):
    def setUp(self):
        self.image = stripe_image()
        self.state = headless_app(self.image)
        self.state.original_image = self.image
        self.state.normalize_var = Value(False)
        self.state.gaussian_denoise_var = Value(False)
        self.state.blur_method_var = Value(app.BLUR_METHOD_GAUSSIAN)
        self.state.detected_edge_mask = None

    def export(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "export.png"
            with patch.object(app.filedialog, "asksaveasfilename", return_value=str(target)):
                self.state.export_annotated_image()
            with Image.open(target) as image:
                return image.copy()

    def test_no_selection_exports_full_image(self):
        self.assertEqual(self.export().size, (220, 180))

    def test_scaled_reversed_selection_crops_exact_pixels_without_green_frame(self):
        self.state.display_scale = 0.5
        self.state.roi_canvas = (80, 70, 10, 15)
        output = np.asarray(self.export())
        expected = np.repeat(self.image[30:140, 20:160, None], 3, axis=2)
        np.testing.assert_array_equal(output, expected)

    def test_fractional_selection_clips_to_image(self):
        self.state.display_scale = 2.0
        self.state.roi_canvas = (-8, 20.5, 500, 200.5)
        output = np.asarray(self.export())
        expected = np.repeat(self.image[10:101, :, None], 3, axis=2)
        np.testing.assert_array_equal(output, expected)

    def test_hidden_analysis_selection_survives_zoom_and_can_be_cleared(self):
        state = self.state
        state.roi_canvas = (20, 30, 160, 140)
        state.hide_roi_after_successful_analysis()
        self.assertIsNone(state.roi_canvas)
        state.display_scale = 2.0
        self.assertEqual(self.export().size, (140, 110))
        state.clear_roi()
        self.assertEqual(self.export().size, (220, 180))

    def test_new_selection_and_full_image_analysis_replace_saved_selection(self):
        state = self.state
        state.export_roi_bounds_px = (20, 30, 160, 140)
        state.roi_canvas = (30, 40, 110, 160)
        self.assertEqual(self.export().size, (80, 120))
        state.hide_roi_after_successful_analysis()
        self.assertEqual(state.export_roi_bounds_px, (30, 40, 110, 160))
        state.hide_roi_after_successful_analysis()
        self.assertIsNone(state.export_roi_bounds_px)
        self.assertEqual(self.export().size, (220, 180))

    def test_cropped_era_traces_and_annotations_keep_pixel_alignment(self):
        state = self.state
        state.recognize_era_lines(measure=True)
        state.roi_canvas = (15, 20, 125, 155)
        # Use the actual annotation renderer, including length-line end caps.
        state.annotations = [app.MeasurementAnnotation("length", (30, 70, 100, 70), "#ff00ff")]
        for name in ("annotation_image_points", "annotation_polygon", "length_end_caps", "annotation_color"):
            setattr(state, name, getattr(app.LERLWRApp, name).__get__(state))
        state.annotation_label = lambda annotation: "70 px"
        for angle in (0.0, 17.0, -17.0):
            state.rotation_degrees = angle
            expected = Image.fromarray(self.image).convert("RGB")
            draw = ImageDraw.Draw(expected)
            state.draw_exported_annotations(draw, 220, 180, working_orientation=True)
            state.draw_exported_era_edges(draw, 220, 180, working_orientation=True)
            output = np.asarray(self.export())
            np.testing.assert_array_equal(output, np.asarray(expected)[20:155, 15:125])
            self.assertTrue(np.all(output == (255, 0, 255), axis=2).any())

    def test_rotated_selection_is_exact_rectangle_of_current_processed_view(self):
        state = self.state
        state.roi_canvas = (60, 50, 150, 130)
        state.rotation_degrees = 17.0
        state.raw_image = app.rotate_grayscale_image(self.image, 17.0)
        state.processed_image = app.preprocess_image(state.raw_image, True, True, 5, 1.0, app.BLUR_METHOD_GAUSSIAN)
        state.hide_roi_after_successful_analysis()
        output = np.asarray(self.export())
        self.assertEqual(output.shape, (80, 90, 3))
        expected = np.repeat(state.processed_image[50:130, 60:150, None], 3, axis=2)
        np.testing.assert_array_equal(output, expected)

    def test_rotated_selection_keeps_bcp_centroids_and_contours_in_working_view(self):
        state = self.state
        state.rotation_degrees = 17.0
        state.roi_canvas = (60, 50, 150, 130)
        state.bcp_result = SimpleNamespace(
            centers_px=np.array([[80, 80]]),
            contour_segments_px=[np.array([[75, 75, 85, 75]])],
        )
        state.bcp_centroid_color = "#ffff00"
        state.bcp_contour_color = "#00ff00"
        state.bcp_contour_width_var = Value(2)
        state.bcp_reference_display_var = Value(False)
        state.bcp_centroid_layout_overlay_var = Value(False)
        state.bcp_centroid_diameter = lambda: 6
        state.bcp_overlay_line_width = lambda variable: variable.get()
        output = np.asarray(self.export())
        np.testing.assert_array_equal(output[30, 20], (255, 255, 0))
        np.testing.assert_array_equal(output[25, 20], (0, 255, 0))

    def test_cropped_edge_only_overlay_keeps_roi_offset(self):
        state = self.state
        state.rotation_degrees = 17.0
        state.roi_canvas = (20, 30, 160, 140)
        state.detected_edge_origin = (20, 30)
        state.detected_edge_mask = np.zeros((110, 140), dtype=bool)
        state.detected_edge_mask[:, 10] = True
        output = np.asarray(self.export())
        self.assertTrue(np.all(output[:, 10] != np.repeat(self.image[30:140, 30, None], 3, axis=1), axis=1).all())
        np.testing.assert_array_equal(output[:, 11], np.repeat(self.image[30:140, 31, None], 3, axis=1))


if __name__ == "__main__":
    unittest.main()
