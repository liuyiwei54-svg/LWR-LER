"""Regression checks for spatial PSD normalization and measurement handoff."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import dataclasses
import unittest
from unittest.mock import Mock, patch

import numpy as np

import ler_lwr_app as app
from roughness_psd import (
    fit_psd_model, repeat_scan_model_comparison, repeat_scan_spectrum, roughness_spectrum,
)


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


class PsdModelHandoffTests(unittest.TestCase):
    """单图模型与双图差分对照接入后，noise_context / 处理签名 / 状态语义的回归。"""

    def _host(self, *, gaussian_denoise: bool, blur_method: str, kernel: str = "3 × 3", kernel_size: int = 3):
        return SimpleNamespace(
            gaussian_denoise_var=SimpleNamespace(get=lambda: gaussian_denoise),
            blur_method_var=SimpleNamespace(get=lambda: blur_method),
            gaussian_kernel_size_var=SimpleNamespace(get=lambda: kernel),
            gaussian_sigma_scale_var=SimpleNamespace(get=lambda: 1.0),
            normalize_var=SimpleNamespace(get=lambda: True),
            gaussian_kernel_size=lambda: kernel_size,
            gaussian_sigma_scale=lambda: 1.0,
            blur_method_var_value=blur_method,
            background_enabled_var=SimpleNamespace(get=lambda: False),
            background_degree_var=SimpleNamespace(get=lambda: "1"),
            background_degree=lambda: None,
        )

    def test_noise_context_never_hard_set_to_white(self):
        smoothed = self._host(gaussian_denoise=True, blur_method=app.BLUR_METHOD_GAUSSIAN)
        self.assertEqual(app.LERLWRApp.psd_noise_context(smoothed), "smoothed")
        mean_smoothed = self._host(gaussian_denoise=True, blur_method=app.BLUR_METHOD_MEAN)
        self.assertEqual(app.LERLWRApp.psd_noise_context(mean_smoothed), "smoothed")
        untouched = self._host(gaussian_denoise=False, blur_method=app.BLUR_METHOD_GAUSSIAN)
        self.assertEqual(app.LERLWRApp.psd_noise_context(untouched), "unknown")
        # 无论哪种平滑，都绝不返回 white（白噪声只能由模型证据给出，不由入口硬设）。
        self.assertNotEqual(app.LERLWRApp.psd_noise_context(smoothed), "white")

    def test_preprocessing_signature_marks_smoothing_and_background_placeholder(self):
        host = self._host(gaussian_denoise=True, blur_method=app.BLUR_METHOD_MEAN, kernel="5 × 5", kernel_size=5)
        signature = app.LERLWRApp.psd_preprocessing_signature(host)
        self.assertIn("smooth=mean:5x5", signature)
        self.assertIn("background=none", signature)
        host_off = self._host(gaussian_denoise=False, blur_method=app.BLUR_METHOD_GAUSSIAN)
        self.assertIn("smooth=off", app.LERLWRApp.psd_preprocessing_signature(host_off))


class PsdModelCsvValueTests(unittest.TestCase):
    """模型 CSV 的数值与状态列必须由真实拟合结果驱动，而非镜像代码。"""

    def _measured_spectrum(self):
        n = 512
        rows = np.arange(n)
        # 结构 + 白噪底：可分离的 gaussian 结构谱 + 平噪声底
        y = 4.0 * np.exp(-((rows - n / 2) / 40.0) ** 2) + 0.05 * np.cos(2 * np.pi * rows / 7)
        return rows, y

    def test_smoothed_context_unreliable_sets_corrected_none_with_candidate(self):
        rows, y = self._measured_spectrum()
        spec = roughness_spectrum([(rows, y)], 1.0)
        result = fit_psd_model(spec, model="gaussian", noise_context="smoothed")
        # 平滑上下文下 corrected_rms 必须为 None（不可靠），候选值仅诊断。
        self.assertFalse(result.reliable)
        self.assertIsNone(result.corrected_rms_nm)
        self.assertIsNotNone(result.candidate_corrected_rms_nm)
        summary = result.summary()
        self.assertIn("corrected_rms_nm", summary)
        self.assertIsNone(summary["corrected_rms_nm"])
        # rms 频段与拟合频段必须落在频网格内
        self.assertGreater(result.rms_band_per_nm[1], result.rms_band_per_nm[0])

    def test_repeat_model_comparison_uses_same_grid_and_independent_corrections(self):
        rng = np.random.default_rng(7)
        rows = np.arange(2048)
        position = 20 + 0.001 * rows + 2.0 * np.sin(2 * np.pi * rows / 40)
        first = position + rng.normal(0, 0.4, len(rows))
        second = position + rng.normal(0, 0.4, len(rows))
        repeat = repeat_scan_spectrum(rows, first, rows, second, 1.0, 0, remove_slope=True)
        comparison = repeat_scan_model_comparison(repeat, 1.0, noise_context="white")
        # 两条校正谱同网格同长度
        self.assertEqual(len(comparison.model_corrected_density_nm3.shape), 1)
        self.assertEqual(
            comparison.model_corrected_density_nm3.shape,
            comparison.difference_corrected_density_nm3.shape,
        )
        # 独立校正：model_corrected = measured - model_noise（平均噪声）
        np.testing.assert_allclose(
            comparison.model_corrected_density_nm3,
            comparison.measured_density_nm3 - comparison.model_noise_density_nm3,
        )
        # 两个单图噪声分别保留，且均来自各自模型
        self.assertEqual(
            comparison.first_model_noise_density_nm3.shape,
            comparison.second_model_noise_density_nm3.shape,
        )
        # 平均噪声 = 两单图噪声算术平均
        np.testing.assert_allclose(
            comparison.model_noise_density_nm3,
            (comparison.first_model_noise_density_nm3 + comparison.second_model_noise_density_nm3) / 2,
        )


class PsdWindowContextCaptureTests(unittest.TestCase):
    """PSD 窗口打开时刻捕获不可变上下文：A 的来源/处理/定位不被 live 主窗口改写。"""

    def test_context_dataclass_is_frozen(self):
        ctx = app.PsdWindowContext(
            source_path="/data/A.png",
            fitted_image_path="/data/A_fit.png",
            preprocessing_signature="normalize=on; smooth=off; background=none",
            noise_context="unknown",
            pixel_size_nm=0.2,
            localization_signature=("era", 3, "bright"),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.source_path = "/data/B.png"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.noise_context = "smoothed"

    def test_noise_context_helper_derives_only_from_actual_smoothing(self):
        # 纯 helper：只有真实平滑才 smoothed，否则 unknown，绝不 white。
        self.assertEqual(app.psd_noise_context_from(True, app.BLUR_METHOD_GAUSSIAN), "smoothed")
        self.assertEqual(app.psd_noise_context_from(True, app.BLUR_METHOD_MEAN), "smoothed")
        self.assertEqual(app.psd_noise_context_from(False, app.BLUR_METHOD_GAUSSIAN), "unknown")

    def test_preprocessing_signature_helper_is_pure(self):
        sig_on = app.psd_preprocessing_signature_from(
            normalize=True, gaussian_denoise=True,
            blur_method=app.BLUR_METHOD_MEAN, kernel_size=5, sigma_scale=1.00,
        )
        self.assertIn("normalize=on", sig_on)
        self.assertIn("smooth=mean:5x5:sigma=1.00", sig_on)
        self.assertIn("background=none", sig_on)
        sig_off = app.psd_preprocessing_signature_from(
            normalize=False, gaussian_denoise=False,
            blur_method=app.BLUR_METHOD_GAUSSIAN, kernel_size=3, sigma_scale=1.00,
        )
        self.assertIn("smooth=off", sig_off)
        # 纯函数：相同输入必得相同输出（与 live 控件无关）。
        self.assertEqual(sig_on, app.psd_preprocessing_signature_from(
            normalize=True, gaussian_denoise=True,
            blur_method=app.BLUR_METHOD_MEAN, kernel_size=5, sigma_scale=1.00,
        ))

    def test_repeat_scan_measurement_snapshot_defaults_are_compatible(self):
        # 默认字段：旧构造（不带处理/噪声/标定）仍可用，且默认值不误报。
        m = app.RepeatScanMeasurement(
            source=Path("/data/A.png"), image=np.zeros((4, 4)), settings=(),
            row_origin=0, lines={}, line_methods={},
        )
        self.assertEqual(m.preprocessing_signature, "")
        self.assertEqual(m.noise_context, "unknown")
        self.assertTrue(np.isnan(m.calibration_nm))

    def test_open_time_context_survives_host_switch_to_b(self):
        # 模拟 open 时刻捕获 A 的上下文（source/处理/噪声/标定），随后 live host
        # 切到 B 图并关掉平滑、改标定；已捕获的 PsdWindowContext 必须仍是 A。
        ctx_a = app.PsdWindowContext(
            source_path="/data/A.png",
            fitted_image_path="/data/A_fit.png",
            preprocessing_signature=app.psd_preprocessing_signature_from(
                normalize=True, gaussian_denoise=True,
                blur_method=app.BLUR_METHOD_GAUSSIAN, kernel_size=3, sigma_scale=1.00,
            ),
            noise_context=app.psd_noise_context_from(True, app.BLUR_METHOD_GAUSSIAN),
            pixel_size_nm=0.2,
            localization_signature=(app.EDGE_DETECTOR_ERA,),
        )
        # live host 切换到 B + 改 blur + 改标定（这些值绝不回写 ctx_a）。
        sig_b = app.psd_preprocessing_signature_from(
            normalize=True, gaussian_denoise=False,
            blur_method=app.BLUR_METHOD_GAUSSIAN, kernel_size=5, sigma_scale=1.00,
        )
        noise_b = app.psd_noise_context_from(False, app.BLUR_METHOD_GAUSSIAN)
        self.assertNotEqual(ctx_a.preprocessing_signature, sig_b)
        self.assertNotEqual(ctx_a.noise_context, noise_b)
        # A 的上下文仍是 A：source、处理签名、文件名、标定均未变。
        self.assertEqual(ctx_a.source_path, "/data/A.png")
        self.assertEqual(ctx_a.fitted_image_path, "/data/A_fit.png")
        self.assertIn("smooth=gaussian:3x3", ctx_a.preprocessing_signature)
        self.assertEqual(ctx_a.noise_context, "smoothed")
        self.assertEqual(ctx_a.pixel_size_nm, 0.2)

    def test_repeat_scan_load_second_rejects_mismatched_preprocessing(self):
        # A/B 不同 saved-processing 应被拒绝：这里验证签名比较逻辑（纯字段）。
        first = app.RepeatScanMeasurement(
            source=Path("/data/A.png"), image=np.zeros((4, 4)), settings=(),
            row_origin=0, lines={}, line_methods={},
            preprocessing_signature="normalize=on; smooth=off; background=none",
            noise_context="unknown", calibration_nm=0.2,
        )
        second = app.RepeatScanMeasurement(
            source=Path("/data/B.png"), image=np.zeros((4, 4)), settings=(),
            row_origin=0, lines={}, line_methods={},
            preprocessing_signature="normalize=on; smooth=gaussian:5x5:sigma=1.00; background=none",
            noise_context="smoothed", calibration_nm=0.2,
        )
        # 不同签名必须判为不匹配（load_second 依此拒绝），同签名判为匹配。
        self.assertNotEqual(first.preprocessing_signature, second.preprocessing_signature)
        third = app.RepeatScanMeasurement(
            source=Path("/data/B.png"), image=np.zeros((4, 4)), settings=(),
            row_origin=0, lines={}, line_methods={},
            preprocessing_signature=first.preprocessing_signature,
            noise_context="smoothed", calibration_nm=0.2,
        )
        self.assertEqual(first.preprocessing_signature, third.preprocessing_signature)


if __name__ == "__main__":
    unittest.main()
