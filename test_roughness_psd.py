"""Regression checks for spatial PSD normalization and measurement handoff."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import ler_lwr_app as app
from roughness_psd import repeat_scan_spectrum, roughness_spectrum


class RoughnessSpectrumTests(unittest.TestCase):
    def test_frequency_and_integrated_power(self):
        for size in (127, 128):
            with self.subTest(size=size):
                rows = np.arange(size)
                residual = 2 * np.sin(2 * np.pi * 8 * rows / size)
                spectrum = roughness_spectrum([(rows, residual)], 0.5)
                peak = spectrum.frequency_per_nm[np.argmax(spectrum.density_nm3)]
                self.assertAlmostEqual(peak, 8 / (size * 0.5))
                window = np.hanning(size)
                centered = residual - np.average(residual, weights=window)
                expected_power = np.average(centered**2, weights=window**2)
                long_rms, short_rms = spectrum.band_rms_nm(0.25)
                self.assertAlmostEqual(long_rms**2 + short_rms**2, expected_power)
                self.assertGreater(long_rms, 100 * short_rms)

    def test_separate_runs_and_lines_do_not_create_frequency_components(self):
        rows = np.r_[0:32, 35:67]
        second_rows = np.arange(28)
        residual = np.repeat([2.0, -3.0], 32)
        spectrum = roughness_spectrum([
            (rows, residual), (second_rows, np.ones(28)), (np.arange(8), np.ones(8)),
        ], 1.0)
        np.testing.assert_allclose(spectrum.density_nm3, 0, atol=1e-25)
        self.assertEqual(spectrum.segment_rows, 28)
        self.assertEqual(spectrum.contributing_lines, 2)
        self.assertEqual((spectrum.covered_rows, spectrum.total_rows), (92, 100))
        self.assertEqual(spectrum.window_count, 5)

    def test_sparse_rows_cannot_be_joined_to_make_a_window(self):
        with self.assertRaisesRegex(ValueError, "连续"):
            roughness_spectrum([(np.arange(64) * 2, np.zeros(64))], 1.0)


class PsdHandoffTests(unittest.TestCase):
    def test_sources_keep_original_line_numbers_and_reject_stale_calibration(self):
        fit = SimpleNamespace(edge_detector=app.EDGE_DETECTOR_ERA, pixel_size_nm=0.25)
        host = SimpleNamespace(
            result=fit, era_aggregate_result=object(), measurement_pixel_size=lambda: 0.25,
            era_line_samples=[
                SimpleNamespace(paired_result=None),
                SimpleNamespace(paired_result=fit),
                SimpleNamespace(paired_result=fit),
            ],
        )
        sources = app.LERLWRApp.roughness_measurements(host)
        self.assertEqual(list(sources), ["全部已测线条", "线样本 2", "线样本 3"])
        host.measurement_pixel_size = lambda: 0.30
        self.assertFalse(app.LERLWRApp.roughness_measurements(host))

    def test_only_combined_export_opens_psd_and_only_after_file_exists(self):
        host = Mock()
        host.original_image = np.zeros((12, 12), dtype=np.uint8)
        host.analysis_origin = host.roi_canvas = host.result = host.image_path = None
        host.era_line_samples = []
        host.image_export_roi_bounds.return_value = None
        host.crop_exported_image.side_effect = lambda image: image
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fit.png"
            host.open_roughness_psd.side_effect = lambda target: self.assertTrue(target.is_file())
            with patch.object(app, "preprocess_image", return_value=host.original_image):
                with patch.object(app.filedialog, "asksaveasfilename", side_effect=["", str(path), str(path)]):
                    app.LERLWRApp.export_annotated_image(host, show_psd=True)
                    app.LERLWRApp.export_annotated_image(host)
                    host.open_roughness_psd.assert_not_called()
                    app.LERLWRApp.export_annotated_image(host, show_psd=True)
            host.open_roughness_psd.assert_called_once_with(path)


class RepeatScanTests(unittest.TestCase):
    def test_registration_preserves_shared_roughness_and_estimates_noise(self):
        rng = np.random.default_rng(123)
        rows = np.arange(8192)
        position = 11 + 0.002 * rows + 1.4 * np.sin(2 * np.pi * rows / 32)
        first = position + rng.normal(0, 0.6, len(rows))
        second = position + 10 + rng.normal(0, 0.6, len(rows))
        result = repeat_scan_spectrum(rows, first, rows - 3, second, 1., 3, remove_slope=True)
        measured, noise, corrected = result.rms_nm()
        self.assertAlmostEqual(noise, 0.6, delta=0.02)
        self.assertAlmostEqual(corrected, 1.4 / np.sqrt(2), delta=0.02)
        self.assertGreater(measured, corrected)
        np.testing.assert_array_equal(result.common_rows, rows)

    def test_width_keeps_slope_and_signed_subtraction_is_not_clipped(self):
        rows = np.arange(128)
        width = rows * 0.01
        shared = repeat_scan_spectrum(rows, width, rows, width + 1, 1., 0, remove_slope=False)
        self.assertGreater(shared.rms_nm()[2], 0.1)
        self.assertAlmostEqual(shared.rms_nm()[1], 0.)
        opposite = repeat_scan_spectrum(rows, width, rows, -width, 1., 0, remove_slope=False)
        self.assertTrue(np.any(opposite.corrected_density_nm3 < 0))
        self.assertIsNone(opposite.rms_nm()[2])

    def test_unmatched_rows_are_not_joined(self):
        rows = np.arange(100)
        with self.assertRaisesRegex(ValueError, "共同有效"):
            repeat_scan_spectrum(rows, rows, rows, rows, 1., 90, remove_slope=True)
        with self.assertRaisesRegex(ValueError, "连续"):
            repeat_scan_spectrum(rows, rows, rows[::2], rows[::2], 1., 0, remove_slope=True)
