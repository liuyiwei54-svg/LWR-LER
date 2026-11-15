"""One-sided spatial spectra for already measured SEM roughness traces."""

from dataclasses import dataclass, replace
import math

import numpy as np


MIN_SEGMENT_ROWS = 16
MAX_SEGMENT_ROWS = 256


@dataclass(frozen=True)
class RoughnessSpectrum:
    frequency_per_nm: np.ndarray
    density_nm3: np.ndarray
    frequency_step_per_nm: float
    segment_rows: int
    contributing_lines: int
    covered_rows: int
    total_rows: int
    window_count: int

    def band_rms_nm(self, cutoff_per_nm: float) -> tuple[float, float]:
        low = self.frequency_per_nm < cutoff_per_nm
        variance = self.density_nm3 * self.frequency_step_per_nm
        return math.sqrt(float(variance[low].sum())), math.sqrt(float(variance[~low].sum()))


def roughness_spectrum(
    traces: list[tuple[np.ndarray, np.ndarray]], pixel_size_nm: float,
) -> RoughnessSpectrum:
    """Average Hann periodograms without joining separate lines or bridging gaps."""
    if not math.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
        raise ValueError("PSD 需要有效的像素尺寸标定。")

    runs_by_line = []
    for rows, residuals_nm in traces:
        boundaries = np.flatnonzero(np.diff(rows) != 1) + 1
        runs_by_line.append(np.split(residuals_nm, boundaries))
    longest_by_line = [max(map(len, runs)) for runs in runs_by_line]
    usable_lengths = [length for length in longest_by_line if length >= MIN_SEGMENT_ROWS]
    if not usable_lengths:
        raise ValueError(f"PSD 至少需要 {MIN_SEGMENT_ROWS} 行连续的有效测量点。")

    size = min(MAX_SEGMENT_ROWS, min(usable_lengths))
    window = np.hanning(size)
    normalization = pixel_size_nm / np.dot(window, window)
    window_sum = window.sum()
    power = np.zeros(size // 2 + 1)
    window_count = covered_rows = contributing_lines = 0
    for runs in runs_by_line:
        contributed = False
        for run in runs:
            if len(run) < size:
                continue
            contributed = True
            covered_rows += len(run)
            last_start = len(run) - size
            starts = list(range(0, last_start + 1, size // 2))
            if starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                segment = run[start:start + size]
                centered = segment - np.dot(segment, window) / window_sum
                power += normalization * np.abs(np.fft.rfft(centered * window)) ** 2
                window_count += 1
        contributing_lines += int(contributed)

    positive_end = -1 if size % 2 == 0 else None
    power[1:positive_end] *= 2
    frequency = np.fft.rfftfreq(size, d=pixel_size_nm)
    return RoughnessSpectrum(
        frequency[1:], power[1:] / window_count, float(frequency[1]), size,
        contributing_lines, covered_rows, sum(len(rows) for rows, _ in traces), window_count,
    )


@dataclass(frozen=True)
class RepeatScanSpectrum:
    measured: RoughnessSpectrum
    noise_density_nm3: np.ndarray
    corrected_density_nm3: np.ndarray
    common_rows: np.ndarray
    first_residual_nm: np.ndarray
    second_residual_nm: np.ndarray

    def rms_nm(self) -> tuple[float, float, float | None]:
        step = self.measured.frequency_step_per_nm
        measured, noise, corrected = (
            float(density.sum()) * step
            for density in (self.measured.density_nm3, self.noise_density_nm3, self.corrected_density_nm3)
        )
        return math.sqrt(measured), math.sqrt(noise), math.sqrt(corrected) if corrected > 0 else None


def repeat_scan_spectrum(
    first_rows: np.ndarray, first_px: np.ndarray,
    second_rows: np.ndarray, second_px: np.ndarray,
    pixel_size_nm: float, second_row_offset: int, *, remove_slope: bool,
) -> RepeatScanSpectrum:
    """Compare one physical trace sampled on the same row grid in two scans."""
    rows, first_index, second_index = np.intersect1d(
        first_rows, second_rows + second_row_offset, assume_unique=True, return_indices=True,
    )
    if len(rows) < MIN_SEGMENT_ROWS:
        raise ValueError(f"对齐后至少需要 {MIN_SEGMENT_ROWS} 行共同有效点。")

    residuals = []
    for positions in (first_px[first_index], second_px[second_index]):
        if remove_slope:
            baseline = np.polyval(np.polyfit(rows, positions, 1), rows)
        else:
            baseline = positions.mean()
        residuals.append((positions - baseline) * pixel_size_nm)

    first, second = residuals
    first_psd = roughness_spectrum([(rows, first)], pixel_size_nm)
    second_psd = roughness_spectrum([(rows, second)], pixel_size_nm)
    difference_psd = roughness_spectrum([(rows, first - second)], pixel_size_nm)
    measured = (first_psd.density_nm3 + second_psd.density_nm3) / 2
    noise = difference_psd.density_nm3 / 2
    return RepeatScanSpectrum(
        replace(first_psd, density_nm3=measured), noise, measured - noise, rows, first, second,
    )
