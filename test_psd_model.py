"""Tests for the conservative single-image PSD structure+noise model.

Covers the fit_psd_model / repeat_scan_model_comparison increment on top of the
existing roughness_spectrum / repeat_scan_spectrum API.  Run with the bundled
Python from the lwr:ler directory:

    python -m unittest test_roughness_psd test_psd_model

These tests are deterministic (fixed seeds / closed-form spectra) and never open
a GUI.  They verify unit scaling, the variance identity
(variance == sum(density) * df), parameter recovery, non-clipping of the signed
corrected spectrum, the correlated-noise context flag, and the dual-image
averaging behaviour of repeat_scan_model_comparison.
"""

import math
import unittest

import numpy as np

from roughness_psd import (
    RoughnessSpectrum,
    fit_psd_model,
    repeat_scan_model_comparison,
    repeat_scan_spectrum,
    roughness_spectrum,
)


def _spectrum(density, *, n=512, pixel_size_nm=1.0, frequency=None):
    """Build a RoughnessSpectrum on a consistent positive-frequency grid."""
    if frequency is None:
        frequency = np.fft.rfftfreq(n, d=pixel_size_nm)[1:]
    density = np.asarray(density, dtype=float)
    assert frequency.size == density.size, (frequency.size, density.size)
    step = float(frequency[1] - frequency[0])
    return RoughnessSpectrum(
        frequency, density, step, n, 1, n, n, 1,
    )


def _gaussian_structure(frequency, amplitude, corner, floor):
    return amplitude * np.exp(-(corner * frequency) ** 2) + floor


def _lorentzian_structure(frequency, amplitude, corner, floor):
    return amplitude / (1.0 + (corner * frequency) ** 2) + floor


def _gaussian_correlated_trace(rng, n, correlation_samples, target_std):
    """Gaussian-correlated smooth structure (Gaussian PSD) of a fixed std."""
    half = int(4 * correlation_samples)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / correlation_samples) ** 2)
    kernel /= kernel.sum()
    trace = np.convolve(rng.normal(0.0, 1.0, n), kernel, mode="same")
    trace -= trace.mean()
    trace *= target_std / trace.std()
    return trace


class FitPsdModelDeterministicRecovery(unittest.TestCase):
    """Closed-form spectra must recover their parameters and obey unit scaling."""

    def test_gaussian_parameter_recovery_and_auto_selection(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        amplitude, corner, floor = 10.0, 40.0, 0.5
        density = _gaussian_structure(frequency, amplitude, corner, floor)
        result = fit_psd_model(_spectrum(density), model="gaussian")
        self.assertEqual(result.model_name, "gaussian")
        self.assertAlmostEqual(result.amplitude_nm3, amplitude, delta=0.05 * amplitude)
        self.assertAlmostEqual(result.corner_length_nm, corner, delta=0.08 * corner)
        self.assertAlmostEqual(result.noise_floor_nm3, floor, delta=0.05 * floor)
        # auto must pick gaussian on a gaussian spectrum
        auto = fit_psd_model(_spectrum(density), model="auto")
        self.assertEqual(auto.model_name, "gaussian")

    def test_lorentzian_parameter_recovery_and_auto_selection(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        amplitude, corner, floor = 10.0, 40.0, 0.5
        density = _lorentzian_structure(frequency, amplitude, corner, floor)
        result = fit_psd_model(_spectrum(density), model="lorentzian")
        self.assertEqual(result.model_name, "lorentzian")
        self.assertAlmostEqual(result.amplitude_nm3, amplitude, delta=0.05 * amplitude)
        self.assertAlmostEqual(result.corner_length_nm, corner, delta=0.08 * corner)
        self.assertAlmostEqual(result.noise_floor_nm3, floor, delta=0.05 * floor)
        auto = fit_psd_model(_spectrum(density), model="auto")
        self.assertEqual(auto.model_name, "lorentzian")

    def test_unit_scaling_is_linear_in_density(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = _gaussian_structure(frequency, 10.0, 40.0, 0.5)
        base = fit_psd_model(_spectrum(density), model="gaussian")
        scaled = fit_psd_model(_spectrum(2.0 * density), model="gaussian")
        self.assertAlmostEqual(scaled.amplitude_nm3 / base.amplitude_nm3, 2.0, places=6)
        self.assertAlmostEqual(scaled.noise_floor_nm3 / base.noise_floor_nm3, 2.0, places=6)
        self.assertAlmostEqual(scaled.corner_length_nm / base.corner_length_nm, 1.0, places=6)

    def test_variance_identity(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = _gaussian_structure(frequency, 10.0, 40.0, 0.5)
        spec = _spectrum(density)
        result = fit_psd_model(spec, model="gaussian")
        # variance == sum(density) * df, df == frequency_step_per_nm
        expected = float(np.sum(result.measured_density_nm3)) * spec.frequency_step_per_nm
        self.assertAlmostEqual(result.measured_rms_nm ** 2, expected, places=9)

    def test_summary_has_csv_keys(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        result = fit_psd_model(
            _spectrum(_gaussian_structure(frequency, 10.0, 40.0, 0.5)), model="gaussian",
        )
        summary = result.summary()
        for key in ("status", "model_name", "amplitude_nm3", "corner_length_nm",
                    "noise_floor_nm3", "measured_rms_nm", "corrected_rms_nm",
                    "reliability_reasons"):
            self.assertIn(key, summary)


class FitPsdModelUnidentifiableAndContext(unittest.TestCase):
    def test_flat_spectrum_is_unidentifiable(self):
        frequency = np.fft.rfftfreq(256, d=1.0)[1:]
        result = fit_psd_model(
            _spectrum(np.full_like(frequency, 2.0), frequency=frequency), model="auto",
        )
        self.assertEqual(result.status, "unidentifiable")
        self.assertIsNone(result.corrected_rms_nm)
        self.assertIn("structure_not_supported_by_band", result.reliability_reasons)

    def test_narrow_band_is_unreliable_not_error(self):
        frequency = np.array([1 / 64, 2 / 64, 3 / 64, 4 / 64, 5 / 64, 6 / 64])
        density = 10.0 * np.exp(-(40.0 * frequency) ** 2) + 0.5
        result = fit_psd_model(_spectrum(density, frequency=frequency), model="gaussian")
        self.assertEqual(result.status, "unreliable")
        self.assertIn("narrow_fit_band", result.reliability_reasons)

    def test_too_few_bins_raises(self):
        frequency = np.array([1 / 64, 2 / 64, 3 / 64])
        density = np.array([2.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            fit_psd_model(_spectrum(density, frequency=frequency), model="gaussian")

    def test_correlated_noise_context_is_unreliable(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = _gaussian_structure(frequency, 10.0, 40.0, 0.5)
        for context in ("smoothed", "correlated"):
            result = fit_psd_model(_spectrum(density), model="gaussian", noise_context=context)
            self.assertNotEqual(result.status, "ok")
            self.assertIn("correlated_noise_white_floor_assumption_limited",
                          result.reliability_reasons)

    def test_invalid_noise_context_and_model_raise(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = _gaussian_structure(frequency, 10.0, 40.0, 0.5)
        spec = _spectrum(density)
        with self.assertRaises(ValueError):
            fit_psd_model(spec, noise_context="bogus")
        with self.assertRaises(ValueError):
            fit_psd_model(spec, model="bogus")


class FitPsdModelSignedCorrection(unittest.TestCase):
    def test_signed_corrected_is_not_clipped(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = _gaussian_structure(frequency, 10.0, 40.0, 0.5)
        density[-5:] = 0.1  # a valley below the fitted floor
        result = fit_psd_model(_spectrum(density), model="gaussian")
        self.assertTrue(np.any(result.corrected_density_nm3 < 0.0))
        np.testing.assert_allclose(
            result.corrected_density_nm3,
            result.measured_density_nm3 - result.noise_density_nm3,
        )


class FitPsdModelRandomTrajectory(unittest.TestCase):
    def test_noise_floor_recovered_from_structure_plus_noise(self):
        rng = np.random.default_rng(7)
        n = 8192
        rows = np.arange(n, dtype=float)
        structure = _gaussian_correlated_trace(rng, n, 25.0, 1.5)
        noise = rng.normal(0.0, 0.4, n)
        signal = structure + noise

        spectrum = roughness_spectrum([(rows, signal)], 1.0)
        result = fit_psd_model(spectrum, model="gaussian")

        self.assertNotIn(result.status, ("error", "unidentifiable"))
        # 该轨迹是 Gaussian 相关平滑结构（非单一结构谱+平直白噪底），单拐点
        # 模型拟合残差较大 → unreliable。按验收规则 corrected_rms_nm 必须为
        # None，但 candidate_corrected_rms_nm 仍提供诊断估计。
        self.assertIsNone(result.corrected_rms_nm)
        self.assertIsNotNone(result.candidate_corrected_rms_nm)
        self.assertLess(result.candidate_corrected_rms_nm, result.measured_rms_nm)
        self.assertLess(result.noise_rms_nm, result.measured_rms_nm)
        # the flat white-noise floor is recovered close to the injected std
        self.assertAlmostEqual(result.noise_rms_nm, 0.4, delta=0.12)


class FitPsdModelNonGridBandRecovery(unittest.TestCase):
    """高质量精确 Gaussian/Lorentzian 谱在非网格 B 下应接近真值且可靠，
    不因 128 格点微小拟合误差被 residual_serially_correlated 误判。"""

    @staticmethod
    def _spectrum(density):
        n = 512
        frequency = np.fft.rfftfreq(n, d=1.0)[1:]
        density = np.asarray(density, dtype=float)
        step = float(frequency[1] - frequency[0])
        return RoughnessSpectrum(frequency, density, step, n, 1, n, n, 1)

    def _check(self, model, shape, corner):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = shape(10.0, corner, frequency) + 1.0
        result = fit_psd_model(self._spectrum(density), model=model, noise_context="white")
        self.assertEqual(result.status, "ok", result.reliability_reasons)
        self.assertEqual(result.model_name, model)
        self.assertAlmostEqual(result.corner_length_nm, corner, delta=0.08 * corner)
        self.assertAlmostEqual(result.amplitude_nm3, 10.0, delta=0.05 * 10.0)
        self.assertAlmostEqual(result.noise_floor_nm3, 1.0, delta=0.05 * 1.0)
        self.assertIsNotNone(result.corrected_rms_nm)

    def test_gaussian_non_grid_b(self):
        shape = lambda A, B, f: A * np.exp(-(B * f) ** 2)
        for corner in (20.0, 40.0):
            with self.subTest(corner=corner):
                self._check("gaussian", shape, corner)

    def test_lorentzian_non_grid_b(self):
        shape = lambda A, B, f: A / (1.0 + (B * f) ** 2)
        for corner in (20.0, 40.0):
            with self.subTest(corner=corner):
                self._check("lorentzian", shape, corner)

    def test_wrong_shape_still_not_reliable(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = 10.0 * np.exp(-(20.0 * frequency) ** 2) + 1.0
        result = fit_psd_model(
            self._spectrum(density), model="lorentzian", noise_context="white",
        )
        self.assertNotEqual(result.status, "ok")
        self.assertIsNone(result.corrected_rms_nm)

    def test_obvious_misfit_still_not_reliable(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = 10.0 * np.exp(-(40.0 * frequency) ** 2) + 3.0 * np.exp(-(8.0 * frequency) ** 2) + 0.3
        result = fit_psd_model(self._spectrum(density), model="gaussian", noise_context="white")
        self.assertNotEqual(result.status, "ok")
        self.assertIsNone(result.corrected_rms_nm)

    def test_correlated_input_still_not_reliable(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = 10.0 * np.exp(-(20.0 * frequency) ** 2) + 1.0
        result = fit_psd_model(
            self._spectrum(density), model="gaussian", noise_context="correlated",
        )
        self.assertNotEqual(result.status, "ok")
        self.assertIsNone(result.corrected_rms_nm)


class FitPsdModelCorrectedRmsSemantics(unittest.TestCase):
    """校正 RMS 语义：unreliable/unidentifiable/error 的正式 corrected_rms_nm
    一律 None，诊断估计放 candidate_corrected_rms_nm。"""

    @staticmethod
    def _spectrum(density, frequency=None):
        if frequency is None:
            n = 512
            frequency = np.fft.rfftfreq(n, d=1.0)[1:]
        density = np.asarray(density, dtype=float)
        step = float(frequency[1] - frequency[0])
        return RoughnessSpectrum(frequency, density, step, 512, 1, 512, 512, 1)

    def test_smoothed_context_corrected_none_candidate_not_none(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = 10.0 * np.exp(-(20.0 * frequency) ** 2) + 1.0
        result = fit_psd_model(
            self._spectrum(density), model="gaussian", noise_context="smoothed",
        )
        self.assertEqual(result.status, "unreliable")
        self.assertIsNone(result.corrected_rms_nm)
        self.assertIsNotNone(result.candidate_corrected_rms_nm)
        self.assertGreater(result.candidate_corrected_rms_nm, 0.0)

    def test_unidentifiable_corrected_and_candidate_none(self):
        frequency = np.fft.rfftfreq(256, d=1.0)[1:]
        density = np.full_like(frequency, 2.0)
        result = fit_psd_model(self._spectrum(density, frequency=frequency), model="auto")
        self.assertEqual(result.status, "unidentifiable")
        self.assertIsNone(result.corrected_rms_nm)
        self.assertIsNone(result.candidate_corrected_rms_nm)

    def test_summary_exposes_candidate_corrected(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = 10.0 * np.exp(-(20.0 * frequency) ** 2) + 1.0
        summary = fit_psd_model(
            self._spectrum(density), model="gaussian", noise_context="smoothed",
        ).summary()
        self.assertIn("candidate_corrected_rms_nm", summary)
        self.assertIsNone(summary["corrected_rms_nm"])
        self.assertIsNotNone(summary["candidate_corrected_rms_nm"])


class FitPsdModelInputValidation(unittest.TestCase):
    """输入校验：只接受规范的未校正 PSD，非法输入抛 ValueError（非 IndexError）。"""

    @staticmethod
    def _make(frequency, density, step=None, segment_rows=512):
        frequency = np.asarray(frequency, dtype=float)
        density = np.asarray(density, dtype=float)
        if step is None:
            step = float(frequency[1] - frequency[0]) if frequency.size > 1 else float("nan")
        return RoughnessSpectrum(frequency, density, step, segment_rows, 1,
                                 segment_rows, segment_rows, 1)

    def test_negative_density_rejected(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        density = np.full_like(frequency, 1.0)
        density[10] = -0.5
        with self.assertRaises(ValueError):
            fit_psd_model(self._make(frequency, density), model="gaussian")

    def test_non_1d_rejected(self):
        frequency = np.zeros((4, 1))
        density = np.zeros(4)
        with self.assertRaises(ValueError):
            fit_psd_model(
                RoughnessSpectrum(frequency, density, 1.0, 512, 1, 512, 512, 1),
                model="gaussian",
            )

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            fit_psd_model(self._make(np.array([]), np.array([])), model="gaussian")

    def test_unequal_length_rejected(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        with self.assertRaises(ValueError):
            fit_psd_model(
                RoughnessSpectrum(frequency, np.ones(len(frequency) - 1),
                                  1 / 512, 512, 1, 512, 512, 1),
                model="gaussian",
            )

    def test_non_finite_frequency_rejected(self):
        frequency = np.array([0.0, np.nan, 2 / 512])
        with self.assertRaises(ValueError):
            fit_psd_model(self._make(frequency, np.ones(3)), model="gaussian")

    def test_non_increasing_frequency_rejected(self):
        frequency = np.array([1, 2, 1, 3]) * 0.01
        with self.assertRaises(ValueError):
            fit_psd_model(self._make(frequency, np.ones(4)), model="gaussian")

    def test_non_uniform_frequency_rejected(self):
        frequency = np.array([1, 2, 4, 5]) * 0.01
        with self.assertRaises(ValueError):
            fit_psd_model(self._make(frequency, np.ones(4), step=0.01), model="gaussian")

    def test_non_positive_df_rejected(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        with self.assertRaises(ValueError):
            fit_psd_model(
                RoughnessSpectrum(frequency, np.ones(len(frequency)), 0.0,
                                  512, 1, 512, 512, 1),
                model="gaussian",
            )

    def test_non_finite_df_rejected(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        with self.assertRaises(ValueError):
            fit_psd_model(
                RoughnessSpectrum(frequency, np.ones(len(frequency)), float("nan"),
                                  512, 1, 512, 512, 1),
                model="gaussian",
            )

    def test_inconsistent_df_rejected(self):
        frequency = np.fft.rfftfreq(512, d=1.0)[1:]
        with self.assertRaises(ValueError):
            fit_psd_model(
                RoughnessSpectrum(frequency, np.ones(len(frequency)), 3.0,
                                  512, 1, 512, 512, 1),
                model="gaussian",
            )

    def test_empty_fit_band_raises_value_error_not_index_error(self):
        frequency = np.array([1, 2, 3, 4, 5, 6]) * 0.01
        density = np.ones(6)
        with self.assertRaises(ValueError):
            fit_psd_model(
                self._make(frequency, density, step=0.01), model="gaussian",
                fit_min=0.2, fit_max=0.3,
            )

    def test_empty_integration_band_raises_value_error(self):
        frequency = np.array([1, 2, 3, 4, 5, 6]) * 0.01
        density = np.ones(6)
        with self.assertRaises(ValueError):
            fit_psd_model(
                self._make(frequency, density, step=0.01), model="gaussian",
                integration_min=0.2, integration_max=0.3,
            )


    def setUp(self):
        rng = np.random.default_rng(7)
        self.n = 8192
        rows = np.arange(self.n, dtype=float)
        structure = _gaussian_correlated_trace(rng, self.n, 25.0, 1.5)
        # two scans with DIFFERENT independent noise levels
        self.sigma_first, self.sigma_second = 0.4, 0.8
        first = structure + rng.normal(0.0, self.sigma_first, self.n)
        second = structure + 10.0 + rng.normal(0.0, self.sigma_second, self.n)
        self.repeat = repeat_scan_spectrum(
            rows, first, rows - 3, second, 1.0, 3, remove_slope=True,
        )
        self.comparison = repeat_scan_model_comparison(self.repeat, 1.0, model="auto")

    def test_two_noises_preserved_and_averaged(self):
        first = self.comparison.first_model.noise_rms_nm
        second = self.comparison.second_model.noise_rms_nm
        # the two single-image models recover their own (different) noise levels
        self.assertAlmostEqual(first, self.sigma_first, delta=0.15)
        self.assertAlmostEqual(second, self.sigma_second, delta=0.25)
        self.assertNotAlmostEqual(first, second, places=1)
        # model noise is the average of the two single-image noise spectra
        np.testing.assert_allclose(
            self.comparison.model_noise_density_nm3,
            (self.comparison.first_model_noise_density_nm3
             + self.comparison.second_model_noise_density_nm3) / 2.0,
        )

    def test_no_serial_subtraction(self):
        # both corrections start from the SAME measured spectrum
        np.testing.assert_allclose(
            self.comparison.model_corrected_density_nm3,
            self.comparison.measured_density_nm3 - self.comparison.model_noise_density_nm3,
        )
        np.testing.assert_allclose(
            self.comparison.difference_corrected_density_nm3,
            self.comparison.measured_density_nm3 - self.comparison.difference_noise_density_nm3,
        )

    def test_difference_noise_matches_existing_estimate(self):
        band = self.comparison.band_per_nm
        frequency = self.repeat.measured.frequency_per_nm
        start = int(np.flatnonzero(np.isclose(frequency, band[0]))[0])
        stop = start + len(self.comparison.frequency_per_nm)
        np.testing.assert_allclose(
            self.comparison.difference_noise_density_nm3,
            self.repeat.noise_density_nm3[start:stop],
        )
        np.testing.assert_allclose(
            self.comparison.measured_density_nm3,
            self.repeat.measured.density_nm3[start:stop],
        )

    def test_noise_agreement(self):
        # averaged single-image noise agrees with the difference-based noise
        self.assertEqual(self.comparison.noise_agreement, "consistent")
        expected = math.sqrt((self.sigma_first ** 2 + self.sigma_second ** 2) / 2.0)
        self.assertAlmostEqual(self.comparison.model_noise_rms_nm, expected, delta=0.1)
        self.assertAlmostEqual(self.comparison.difference_noise_rms_nm, expected, delta=0.1)

    def test_mismatched_pixel_size_raises(self):
        with self.assertRaises(ValueError):
            repeat_scan_model_comparison(self.repeat, 2.0, model="auto")


class RepeatScanModelComparisonIntegrationBand(unittest.TestCase):
    """repeat_scan_model_comparison 的汇总 RMS 必须使用与 fit_psd_model 相同的实际
    积分频段（含半 df 边界约定），而不是整个拟合频段；否则 integration_min /
    integration_max 会被忽略。"""

    @staticmethod
    def _make_repeat():
        rng = np.random.default_rng(43)
        y = np.arange(4096, dtype=float)
        first = np.sin(y / 7) + rng.normal(0.0, 0.5, len(y))
        second = np.sin(y / 7) + rng.normal(0.0, 0.8, len(y))
        return repeat_scan_spectrum(y, first, y, second, 1.0, 0, remove_slope=True)

    def test_subband_vs_fullband_measured_rms_differ(self):
        """明示子频段（integration_max=.1）与整个拟合频段的 measured_rms 明显不同，
        且等于按选定积分频段 sum(measured_density)*df 开方。"""
        repeat = self._make_repeat()
        full = repeat_scan_model_comparison(repeat, 1.0, model="auto")
        sub = repeat_scan_model_comparison(repeat, 1.0, model="auto", integration_max=0.1)
        # 子频段 RMS 必须小于全频段 RMS（全频段含更多高频功率）。
        self.assertLess(sub.measured_rms_nm, full.measured_rms_nm)
        # 按选定积分频段的直接验算（半 df 边界约定）。
        f = repeat.measured.frequency_per_nm
        df = repeat.measured.frequency_step_per_nm
        sel = f <= 0.1 + df / 2
        expected = math.sqrt(float(np.sum(sub.measured_density_nm3[sel[:len(sub.measured_density_nm3)]]) * df))
        self.assertAlmostEqual(sub.measured_rms_nm, expected, places=12)
        # 显式回归值（修复后应命中此值，而非整个拟合频段的 0.9757918496）。
        self.assertAlmostEqual(sub.measured_rms_nm, 0.7684593072878628, places=12)

    def test_each_spectrum_variance_identity_in_integration_band(self):
        """measured / model_noise / difference_noise / model_corrected /
        difference_corrected 的方差与 RMS 均在同一积分频段内满足 sum(density)*df。"""
        repeat = self._make_repeat()
        comp = repeat_scan_model_comparison(repeat, 1.0, model="auto", integration_max=0.1)
        df = repeat.measured.frequency_step_per_nm
        f = repeat.measured.frequency_per_nm
        start = int(np.flatnonzero(np.isclose(f, comp.band_per_nm[0]))[0])
        # 用报告的实际积分频段边界切到 measured 频点范围。
        lo, hi = comp.rms_band_per_nm
        full = repeat.measured.density_nm3[start:start + len(comp.frequency_per_nm)]
        freq = repeat.measured.frequency_per_nm[start:start + len(comp.frequency_per_nm)]
        sel = (freq >= lo - 0.5 * df) & (freq <= hi + 0.5 * df)
        # measured_rms vs full measured density 在积分频段内的和。
        self.assertAlmostEqual(
            comp.measured_rms_nm ** 2, float(np.sum(full[sel])) * df, places=12,
        )
        self.assertAlmostEqual(
            comp.model_noise_rms_nm ** 2, float(np.sum(comp.model_noise_density_nm3[sel])) * df, places=12,
        )
        self.assertAlmostEqual(
            comp.difference_noise_rms_nm ** 2, float(np.sum(comp.difference_noise_density_nm3[sel])) * df, places=12,
        )
        # 校正 RMS 若有效，也应等于校正谱在积分频段内的方差开方。
        if comp.model_corrected_rms_nm is not None:
            self.assertAlmostEqual(
                comp.model_corrected_rms_nm ** 2,
                float(np.sum(comp.model_corrected_density_nm3[sel])) * df, places=12,
            )
        if comp.difference_corrected_rms_nm is not None:
            self.assertAlmostEqual(
                comp.difference_corrected_rms_nm ** 2,
                float(np.sum(comp.difference_corrected_density_nm3[sel])) * df, places=12,
            )
        # rms_band_per_nm 必须正是实际积分频段边界（与半 df 约定一致）。
        self.assertAlmostEqual(lo, comp.band_per_nm[0], places=12)
        self.assertLessEqual(hi, 0.1 + 0.5 * df + 1e-12)

    def test_two_corrections_start_from_measured_in_integration_band(self):
        """两条独立扣法各自从同一 measured 出发（在积分频段内），不重复扣噪。"""
        repeat = self._make_repeat()
        comp = repeat_scan_model_comparison(repeat, 1.0, model="auto", integration_max=0.1)
        np.testing.assert_allclose(
            comp.model_corrected_density_nm3 + comp.model_noise_density_nm3,
            comp.measured_density_nm3,
        )
        np.testing.assert_allclose(
            comp.difference_corrected_density_nm3 + comp.difference_noise_density_nm3,
            comp.measured_density_nm3,
        )

    def test_rms_band_matches_model_rms_band(self):
        """汇总积分频段应与两次单图模型的 rms_band_per_nm 一致。"""
        repeat = self._make_repeat()
        comp = repeat_scan_model_comparison(repeat, 1.0, model="auto", integration_max=0.1)
        self.assertAlmostEqual(comp.rms_band_per_nm[0], comp.first_model.rms_band_per_nm[0], places=12)
        self.assertAlmostEqual(comp.rms_band_per_nm[1], comp.first_model.rms_band_per_nm[1], places=12)
        self.assertAlmostEqual(comp.rms_band_per_nm[0], comp.second_model.rms_band_per_nm[0], places=12)
        self.assertAlmostEqual(comp.rms_band_per_nm[1], comp.second_model.rms_band_per_nm[1], places=12)


if __name__ == "__main__":
    unittest.main()
