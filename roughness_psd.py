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


# ---------------------------------------------------------------------------
# 单图 PSD 模型估计（2026-11-26 新增）
#
# 独立于主计量：输入是 roughness_spectrum() 已规范化的测得 PSD，只做谱级
# 模型估计，不修改像素、边轨迹、CD/LER/LWR 或 repeat_scan_spectrum 语义，
# 不实现 HHCF。
#
# 概念参考 SMILE 2.3.1 源码：Gaussian.m 的 A*exp(-(f.^2)*B^2)+|C| 结构谱加
# 白噪底形态，以及 NoWhiteNoise.m 的 Lorentzian 型结构谱形态。未移植任何
# SMILE 源码（MATLAB → 本项目的自主 NumPy 实现）；本实现的目标函数、约束、
# 网格搜索、模型检查与可靠性判定均为自行设计，详见
# smile/PSD_model_integration.md。
#
# 单位：PSD 密度 nm^3，频率 cycles/nm，df = 1/(N*d)。
# 噪声 RMS = sqrt(sum(noise_density) * df)（在指定积分频段内），C 是谱密度
# 不是 nm。校正谱 = 测得谱 - 噪声谱，保留符号、不 clip。
# ---------------------------------------------------------------------------

# 可调常量（保守默认；含义与理由见 PSD_model_integration.md）
MIN_MODEL_BAND_BINS = 8            # 低于该频点数 → narrow_fit_band（不可靠）
MIN_MODEL_BAND_BINS_HARD = 4       # 低于该频点数 → 直接报错（无法拟合）
B_GRID_POINTS = 128                # B 的对数网格点数
B_GRID_MARGIN = 4.0                # B 网格超出 [1/f_max, 1/f_min] 的倍数
LOCAL_REFINE_FACTOR = 2.0          # 局部细化子网格半径（以相邻网格比例为单位）
LOCAL_REFINE_STEPS = 3             # 局部对数细化轮数（确定性、有界）
LOCAL_REFINE_POINTS = 16           # 每轮细化子网格点数
STRUCTURE_EVIDENCE_IMPROVEMENT = 0.10   # 相对常数模型的最小 SSE 改善
MIN_STRUCTURE_AMPLITUDE_RATIO = 0.5     # 最小结构幅度 / 噪声底比值
MODEL_CONTRADICTION_LIMIT = 0.30        # 两个模型噪声底 C 的最大相对差异
RESIDUAL_MISMATCH_LIMIT = 0.5           # 残差 RMS / 平均密度 上限
RESIDUAL_CORRELATION_LIMIT = 0.6        # 残差滞后 1 自相关绝对值上限
RESIDUAL_AUTOCORRELATION_TOLERANCE = 2e-3  # 残差 RMS/平均密度 低于此相对幅度时不因自相关单独否决
EDGE_FRACTION = 0.3                     # 谱段末端平坦性检查使用的频段占比
EDGE_DECAY_LIMIT = 0.5                  # 末端频段内谱衰减到该比例以下 → 警告
REPEAT_NOISE_DISAGREEMENT_LIMIT = 0.30  # 双图对照两种噪声估计的最大相对差异

# noise_context 的合法取值（见 fit_psd_model 文档）：
# - "unknown"   ：缺省，无外部噪声信息，仅做自动检查。
# - "white"     ：调用方声明噪声为白噪（独立、不相关），C 可作为可靠噪底。
# - "smoothed"  ：输入经平滑，噪声相关，白噪假设受限。
# - "correlated"：噪声相关，同 "smoothed"。
NOISE_CONTEXT_VALUES = ("unknown", "white", "smoothed", "correlated")


def gaussian_structure_density(amplitude, corner_length_nm, frequency_per_nm):
    """Gaussian 结构谱 A*exp(-(B*f)^2)。A、C 单位 nm^3，B 单位 nm，f 单位 cycles/nm。"""
    return amplitude * np.exp(-(corner_length_nm * frequency_per_nm) ** 2)


def lorentzian_structure_density(amplitude, corner_length_nm, frequency_per_nm):
    """Lorentzian 结构谱 A/(1+(B*f)^2)。A、C 单位 nm^3，B 单位 nm，f 单位 cycles/nm。"""
    return amplitude / (1.0 + (corner_length_nm * frequency_per_nm) ** 2)


PSD_STRUCTURE_MODELS = {
    "gaussian": gaussian_structure_density,
    "lorentzian": lorentzian_structure_density,
}


def _fit_linear_amplitudes(structure_column, measured_density):
    """对 A、C 的精确非负最小二乘：minimize sum((measured - (A*g + C))^2)，A>=0，C>=0。

    两参数情形下最优解只可能落在无约束解或某一条约束边界上，因此枚举四个
    子集（无约束 / A=0 / C=0 / 双方约束）即可得到精确解，不使用迭代优化，
    不存在步长或收敛问题。
    """
    n = len(measured_density)
    candidates = []
    design = np.column_stack((structure_column, np.ones(n)))
    try:
        free, *_ = np.linalg.lstsq(design, measured_density, rcond=None)
        if free[0] >= 0.0 and free[1] >= 0.0:
            candidates.append((float(free[0]), float(free[1])))
    except np.linalg.LinAlgError:
        pass
    candidates.append((0.0, float(np.mean(measured_density))))  # A = 0 → C = 均值
    g2 = float(np.dot(structure_column, structure_column))
    if g2 > 0.0:
        candidates.append((max(0.0, float(np.dot(measured_density, structure_column)) / g2), 0.0))
    best = None
    for amplitude, floor in candidates:
        sse = float(np.sum((measured_density - amplitude * structure_column - floor) ** 2))
        if best is None or sse < best[0]:
            best = (sse, amplitude, floor)
    return best


def _default_b_grid(f_min, f_max, points=B_GRID_POINTS, margin=B_GRID_MARGIN):
    """B 的对数网格。结构谱拐角频率约为 f=1/B，网格覆盖
    [1/(f_max*margin), margin/f_min]，即拐角允许落在频段外 margin 倍处，
    以便检测“拐角在频段外/参数触及边界”的情况。"""
    return np.geomspace(1.0 / (f_max * margin), margin / f_min, points)


def _best_on_local_grid(shape, f_band, s_band, center_b, below_b, above_b):
    """在 [below_b, above_b] 的对数邻域里做确定性局部网格细化，返回该邻域内
    SSE 最小的 (sse, amplitude, floor, b)。邻域由 coarse 网格中最佳 B 两侧
    相邻格点界定，故细化始终有界、落在粗网格内，不改变粗网格语义。"""
    local = np.geomspace(below_b, above_b, LOCAL_REFINE_POINTS)
    best = None
    for b in local:
        sse, amplitude, floor = _fit_linear_amplitudes(shape(1.0, b, f_band), s_band)
        candidate = (sse, amplitude, floor, b)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def _search_b_with_local_refinement(shape, f_band, s_band):
    """粗对数网格穷举 B 后，在最佳 B 邻域内做有界的确定性局部对数细化。

    目的：B 网格只有 128 点时，高质量 Gaussian/Lorentzian 谱的 B 可能落在
    相邻格点之间，导致数点级拟合残差出现缓慢串行结构，被
    residual_serially_correlated 误判为不可靠。受限细化在不依赖扩展网格
    或迭代优化器的前提下把 B 逼近真值，从而压平残差；细化轮数/点数固定、
    邻域由粗网格格点约束，保证有界、确定、可复现。
    """
    b_grid = _default_b_grid(float(f_band[0]), float(f_band[-1]))
    coarse = []
    for b in b_grid:
        sse, amplitude, floor = _fit_linear_amplitudes(shape(1.0, b, f_band), s_band)
        coarse.append((sse, amplitude, floor, b))
    best = min(coarse, key=lambda item: item[0])
    best_index = min(range(len(coarse)), key=lambda i: coarse[i][0])
    below_b = b_grid[max(0, best_index - 1)]
    above_b = b_grid[min(len(b_grid) - 1, best_index + 1)]
    for _ in range(LOCAL_REFINE_STEPS):
        refined = _best_on_local_grid(shape, f_band, s_band, best[3], below_b, above_b)
        if refined[0] < best[0]:
            best = refined
        else:
            break
    return best


def _lag1_autocorrelation(values):
    values = np.asarray(values, dtype=float)
    variance = float(np.dot(values, values))
    if variance <= 0.0 or len(values) < 2:
        return 0.0
    return float(np.dot(values[:-1], values[1:])) / variance


def _edge_decay_drop(frequency_top, density_top):
    """对拟合频段末端 freq 段做 log 密度对频率的线性拟合，返回该频段内
    谱密度的下降倍数（末端/首端）。纯白噪底时约等于 1；持续衰减说明
    结构拐角在频段外或噪声不是白噪声（如平滑相关噪声）。"""
    positive = density_top > 0
    if int(positive.sum()) < 3:
        return None
    f = frequency_top[positive]
    log_density = np.log(density_top[positive])
    design = np.column_stack((f, np.ones_like(f)))
    try:
        coef, *_ = np.linalg.lstsq(design, log_density, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if f[-1] <= f[0]:
        return 1.0
    slope = float(coef[0])
    return float(np.exp(slope * (f[-1] - f[0])))


def _rms_over_band(density, start, stop, frequency_step):
    return math.sqrt(float(np.sum(density[start:stop])) * frequency_step)


@dataclass(frozen=True)
class PsdModelResult:
    """fit_psd_model 的结果（谱级，独立于主计量）。

    status 语义：
    - "ok"：模型检查通过，可作可靠估计显示（corrected_rms_nm 有效）。
    - "unreliable"：给出估计但至少一项检查未通过，UI 必须带警告显示、
      不得当作可靠值；corrected_rms_nm 为 None，diagnostic 的
      candidate_corrected_rms_nm 可作参考。
    - "unidentifiable"：结构/噪声无法分离（谱在频段内平直或结构证据不足），
      corrected_rms_nm 与 candidate_corrected_rms_nm 均为 None，
      noise_floor_nm3 仅为参考。
    - "error"：输入数据异常，各估计量不可用。
    """
    model_name: str
    amplitude_nm3: float
    corner_length_nm: float
    noise_floor_nm3: float
    parameters: dict
    candidates: dict
    status: str
    reliability_reasons: tuple
    reliable: bool
    noise_context: str
    frequency_per_nm: np.ndarray
    measured_density_nm3: np.ndarray
    total_fit_density_nm3: np.ndarray
    structure_density_nm3: np.ndarray
    noise_density_nm3: np.ndarray
    corrected_density_nm3: np.ndarray
    residual_density_nm3: np.ndarray
    fit_band_per_nm: tuple
    rms_band_per_nm: tuple
    measured_rms_nm: float
    noise_rms_nm: float
    structure_rms_nm: float
    total_fit_rms_nm: float
    corrected_rms_nm: float | None
    candidate_corrected_rms_nm: float | None
    corrected_variance_nm2: float
    selected_sse_nm6: float
    white_floor_sse_nm6: float
    structure_evidence_improvement: float

    @property
    def is_reliable(self) -> bool:
        return self.status == "ok"

    def summary(self) -> dict:
        """CSV/UI 用的扁平摘要。None 的 corrected_rms_nm 保留 None；
        candidate_corrected_rms_nm 仅供诊断，UI 不得当可靠值。"""
        return {
            "status": self.status,
            "reliable": self.reliable,
            "model_name": self.model_name,
            "amplitude_nm3": self.amplitude_nm3,
            "corner_length_nm": self.corner_length_nm,
            "noise_floor_nm3": self.noise_floor_nm3,
            "measured_rms_nm": self.measured_rms_nm,
            "noise_rms_nm": self.noise_rms_nm,
            "structure_rms_nm": self.structure_rms_nm,
            "total_fit_rms_nm": self.total_fit_rms_nm,
            "corrected_rms_nm": self.corrected_rms_nm,
            "candidate_corrected_rms_nm": self.candidate_corrected_rms_nm,
            "fit_band_low_per_nm": self.fit_band_per_nm[0],
            "fit_band_high_per_nm": self.fit_band_per_nm[1],
            "rms_band_low_per_nm": self.rms_band_per_nm[0],
            "rms_band_high_per_nm": self.rms_band_per_nm[1],
            "noise_context": self.noise_context,
            "reliability_reasons": ";".join(self.reliability_reasons),
            "candidate_models": ",".join(sorted(self.candidates)),
        }


def _error_psd_model_result(frequency, density, noise_context, message):
    nan = math.nan
    n = len(density)
    return PsdModelResult(
        model_name="", amplitude_nm3=nan, corner_length_nm=nan, noise_floor_nm3=nan,
        parameters={}, candidates={}, status="error", reliability_reasons=(message,),
        reliable=False, noise_context=noise_context,
        frequency_per_nm=frequency, measured_density_nm3=density,
        total_fit_density_nm3=np.full(n, nan), structure_density_nm3=np.full(n, nan),
        noise_density_nm3=np.full(n, nan), corrected_density_nm3=np.full(n, nan),
        residual_density_nm3=np.full(n, nan),
        fit_band_per_nm=(float(frequency[0]), float(frequency[-1])),
        rms_band_per_nm=(float(frequency[0]), float(frequency[-1])),
        measured_rms_nm=nan, noise_rms_nm=nan, structure_rms_nm=nan,
        total_fit_rms_nm=nan, corrected_rms_nm=None, candidate_corrected_rms_nm=None,
        corrected_variance_nm2=nan,
        selected_sse_nm6=nan, white_floor_sse_nm6=nan,
        structure_evidence_improvement=nan,
    )


def fit_psd_model(
    spectrum: RoughnessSpectrum,
    model: str = "auto",
    fit_min: float | None = None,
    fit_max: float | None = None,
    integration_min: float | None = None,
    integration_max: float | None = None,
    noise_context: str = "unknown",
) -> PsdModelResult:
    """对规范化测得 PSD 做保守的单图模型估计（结构谱 + 白噪底）。

    模型：S(f) = A*structure(B, f) + C，A>=0，C>=0，B>0。
    structure 取 gaussian（A*exp(-(Bf)^2)）或 lorentzian（A/(1+(Bf)^2)）。
    model="auto" 同时拟合两个模型并按拟合频段内 SSE 取较小者（噪声底 C 明显
    矛盾时标不可靠），不冒充唯一真实模型；也可显式指定 "gaussian"/"lorentzian"。

    算法（目标函数与边界明示，无迭代优化）：
    - B 在对数网格上搜索（见 _default_b_grid）。
    - 每个 B 下 A、C 用精确两参数非负最小二乘求解（见 _fit_linear_amplitudes）。
    - 目标函数：拟合频段内 sum((measured - (A*structure + C))^2)，线性密度
      单位（nm^6），无权重。输入密度不做任何平滑。

    参数（单位 cycles/nm）：
    - fit_min / fit_max：拟合频段 (f_lo, f_hi)，缺省为测得谱全频段。
    - integration_min / integration_max：积分频段，缺省等于拟合频段，必须是
      拟合频段子集；measured/noise/structure/corrected RMS 均在该频段内按
      sqrt(sum(density)*df) 计算，方差 = sum(density)*df。
    - noise_context：外部噪声信息。传入 "smoothed"/"correlated" 时不把白噪底
      估计当作可靠（status="unreliable" 并附原因）；"white" 显式声明白噪；
      "unknown"（缺省）仅依赖自动检查。

    返回 PsdModelResult：包含 f（frequency_per_nm）、measured、total_fit、
    structure、noise、corrected（= measured - noise，保留符号、不 clip）、
    参数、实际拟合/积分频段、残差、status/reliability_reasons/reliable 与
    积分 RMS。corrected_variance <= 0 或 status="unidentifiable" 时
    corrected_rms_nm 为 None。非有限输入返回 status="error"，band/model/
    noise_context 非法时抛 ValueError。
    """
    frequency = spectrum.frequency_per_nm
    density = spectrum.density_nm3
    frequency_step = spectrum.frequency_step_per_nm
    # 输入校验：只接受规范的未校正 PSD。
    frequency = np.asarray(frequency, dtype=float)
    density = np.asarray(density, dtype=float)
    if frequency.ndim != 1 or density.ndim != 1:
        raise ValueError("PSD 频率与密度必须是一维数组。")
    if frequency.size != density.size or frequency.size < 2:
        raise ValueError("PSD 频点不足，无法做模型估计。")
    if frequency.size == 0 or density.size == 0:
        raise ValueError("PSD 频率与密度不能为空。")
    if not np.all(np.isfinite(frequency)):
        raise ValueError("PSD 频率包含非有限值。")
    if not np.all(np.isfinite(density)):
        return _error_psd_model_result(frequency, density, noise_context, "non_finite_density")
    if np.any(density < 0.0):
        raise ValueError("测得 PSD 密度不能为负（疑似已扣噪谱，请传入规范的未校正 PSD）。")
    if not np.all(np.diff(frequency) > 0.0):
        raise ValueError("PSD 频率必须严格递增。")
    if not math.isfinite(frequency_step) or frequency_step <= 0.0:
        raise ValueError("PSD 频率步长 df 必须为正有限值。")
    if frequency.size > 1 and not np.allclose(
        np.diff(frequency), frequency_step, rtol=1e-6, atol=0.0,
    ):
        raise ValueError("PSD 频率必须均匀（等间距），且与 df 一致。")
    if not math.isfinite(float(spectrum.segment_rows)) or spectrum.segment_rows <= 0:
        raise ValueError("PSD 段长 segment_rows 必须为正。")
    models = ("gaussian", "lorentzian") if model == "auto" else (model,)

    if model not in ("auto", "gaussian", "lorentzian"):
        raise ValueError(f"未知 PSD 结构谱模型：{model!r}（可选 auto/gaussian/lorentzian）。")
    if noise_context not in NOISE_CONTEXT_VALUES:
        raise ValueError(
            f"未知 noise_context：{noise_context!r}（可选 {', '.join(NOISE_CONTEXT_VALUES)}）。"
        )

    if fit_min is None:
        fit_min = float(frequency[0])
    if fit_max is None:
        fit_max = float(frequency[-1])
    if not (math.isfinite(fit_min) and math.isfinite(fit_max) and fit_min < fit_max):
        raise ValueError("拟合频段必须是 fit_min < fit_max 的有效频率（cycles/nm）。")
    tolerance = 0.5 * frequency_step
    if fit_min < frequency[0] - tolerance or fit_max > frequency[-1] + tolerance:
        raise ValueError("拟合频段超出测得 PSD 频率范围。")
    band_mask = (frequency >= fit_min - tolerance) & (frequency <= fit_max + tolerance)
    band_indexes = np.flatnonzero(band_mask)
    if band_indexes.size == 0:
        raise ValueError("拟合频段内没有频点。")  # 明确 ValueError，而非下游 IndexError
    start, stop = int(band_indexes[0]), int(band_indexes[-1]) + 1
    f_band = frequency[start:stop]
    s_band = density[start:stop]
    if s_band.size < MIN_MODEL_BAND_BINS_HARD:
        raise ValueError(f"拟合频段内至少需要 {MIN_MODEL_BAND_BINS_HARD} 个频点。")

    if integration_min is None:
        integration_min = float(f_band[0])
    if integration_max is None:
        integration_max = float(f_band[-1])
    if not (math.isfinite(integration_min) and math.isfinite(integration_max)
            and integration_min <= integration_max):
        raise ValueError("积分频段必须是有效频率区间。")
    if integration_min < f_band[0] - tolerance or integration_max > f_band[-1] + tolerance:
        raise ValueError("积分频段必须落在拟合频段内。")
    r_mask = (f_band >= integration_min - tolerance) & (f_band <= integration_max + tolerance)
    r_indexes = np.flatnonzero(r_mask)
    if r_indexes.size == 0:
        raise ValueError("积分频段内没有频点。")  # 明确 ValueError，而非 IndexError
    r_start, r_stop = int(r_indexes[0]), int(r_indexes[-1]) + 1

    # 每个模型：对数网格搜 B（含有界局部细化），每个 B 下对 A、C 精确非负最小二乘。
    candidates = {}
    for name in models:
        shape = PSD_STRUCTURE_MODELS[name]
        sse, amplitude, floor, b = _search_b_with_local_refinement(shape, f_band, s_band)
        candidates[name] = {
            "amplitude_nm3": amplitude, "corner_length_nm": b,
            "noise_floor_nm3": floor, "sse_nm6": sse,
        }
    selected_name = min(candidates, key=lambda name: candidates[name]["sse_nm6"])
    selected = candidates[selected_name]
    amplitude = selected["amplitude_nm3"]
    corner = selected["corner_length_nm"]
    floor = selected["noise_floor_nm3"]

    # 拟合/模型检查（保守：宁可不可靠/不可识别，不硬报确定校正 RMS）。
    white_floor = float(np.mean(s_band))
    white_sse = float(np.sum((s_band - white_floor) ** 2))
    improvement = (white_sse - selected["sse_nm6"]) / white_sse if white_sse > 0 else 0.0
    structure_density = PSD_STRUCTURE_MODELS[selected_name](amplitude, corner, f_band)
    total_fit_density = structure_density + floor
    noise_density = np.full_like(f_band, floor)
    corrected_density = s_band - noise_density  # 保留符号，不 clip
    residual_density = s_band - total_fit_density

    reasons: list[str] = []
    if amplitude <= MIN_STRUCTURE_AMPLITUDE_RATIO * floor or improvement < STRUCTURE_EVIDENCE_IMPROVEMENT:
        status = "unidentifiable"
        reasons.append("structure_not_supported_by_band")
    else:
        status = "ok"
        if s_band.size < MIN_MODEL_BAND_BINS:
            reasons.append("narrow_fit_band")
        # 边界判定用原始粗网格（与 _default_b_grid 同参数），细化不改粗网格语义。
        coarse_grid = _default_b_grid(float(f_band[0]), float(f_band[-1]))
        if corner <= coarse_grid[0] or corner >= coarse_grid[-1]:
            reasons.append("corner_parameter_at_grid_boundary")
        knee = 1.0 / corner
        if not (f_band[0] <= knee <= f_band[-1]):
            reasons.append("structure_corner_outside_fit_band")
        mean_density = float(np.mean(s_band))
        residual_rms = float(np.sqrt(np.mean(residual_density ** 2)))
        if mean_density > 0.0:
            relative_residual = residual_rms / mean_density
            if relative_residual > RESIDUAL_MISMATCH_LIMIT:
                reasons.append("residual_mismatch")
        # 自相关判据仅在残差幅度仍显著时应用：微小（纯网格/数值舍入）残差的
        # 缓慢串行结构不代表真实失配，单独据此否决会误杀高质量精确谱。判定
        # 用相对幅度：残差 RMS / 平均密度 低于 RESIDUAL_AUTOCORRELATION_TOLERANCE
        # 时忽略自相关；残差本身为 0（完美拟合）时也不否决。
        residual_is_negligible = (
            residual_rms <= 0.0
            or (mean_density > 0.0
                and (residual_rms / mean_density) < RESIDUAL_AUTOCORRELATION_TOLERANCE)
        )
        if (not residual_is_negligible
                and abs(_lag1_autocorrelation(residual_density)) > RESIDUAL_CORRELATION_LIMIT):
            reasons.append("residual_serially_correlated")
        top_bins = max(3, int(math.ceil(EDGE_FRACTION * s_band.size)))
        if top_bins >= 3:
            drop = _edge_decay_drop(f_band[-top_bins:], s_band[-top_bins:])
            if drop is not None and drop < EDGE_DECAY_LIMIT:
                reasons.append("spectrum_decaying_at_band_edge")
        if len(candidates) >= 2:
            floors = [candidate["noise_floor_nm3"] for candidate in candidates.values()]
            max_floor = max(floors)
            if max_floor > 0.0:
                spread = max(abs(x - y) for i, x in enumerate(floors) for y in floors[i + 1:])
                if spread / max_floor > MODEL_CONTRADICTION_LIMIT:
                    reasons.append("model_noise_contradiction")
        if noise_context in ("smoothed", "correlated"):
            reasons.append("correlated_noise_white_floor_assumption_limited")
        if reasons:
            status = "unreliable"

    measured_rms = _rms_over_band(s_band, r_start, r_stop, frequency_step)
    noise_rms = _rms_over_band(noise_density, r_start, r_stop, frequency_step)
    structure_rms = _rms_over_band(structure_density, r_start, r_stop, frequency_step)
    total_fit_rms = _rms_over_band(total_fit_density, r_start, r_stop, frequency_step)
    corrected_variance = float(np.sum(corrected_density[r_start:r_stop])) * frequency_step

    # 候选估计（诊断用）：只要数学上可计算（方差 > 0）就给出，不承载可靠语义。
    candidate_corrected_rms = math.sqrt(corrected_variance) if corrected_variance > 0.0 else None

    # 正式 corrected_rms_nm：仅 status == "ok" 时有效；unreliable/unidentifiable/
    # error 一律 None，绝不把不可靠估计冒充确定真值。
    if corrected_variance <= 0.0:
        if status != "unidentifiable":
            reasons.append("corrected_variance_nonpositive")
            status = "unreliable"
        corrected_rms = None
    elif status == "ok":
        corrected_rms = math.sqrt(corrected_variance)
    else:
        corrected_rms = None

    return PsdModelResult(
        model_name=selected_name, amplitude_nm3=amplitude, corner_length_nm=corner,
        noise_floor_nm3=floor,
        parameters={
            "amplitude_nm3": amplitude, "corner_length_nm": corner,
            "noise_floor_nm3": floor,
        },
        candidates=candidates, status=status, reliability_reasons=tuple(reasons),
        reliable=(status == "ok"), noise_context=noise_context,
        frequency_per_nm=f_band, measured_density_nm3=s_band,
        total_fit_density_nm3=total_fit_density, structure_density_nm3=structure_density,
        noise_density_nm3=noise_density, corrected_density_nm3=corrected_density,
        residual_density_nm3=residual_density,
        fit_band_per_nm=(float(f_band[0]), float(f_band[-1])),
        rms_band_per_nm=(float(f_band[r_start]), float(f_band[r_stop - 1])),
        measured_rms_nm=measured_rms, noise_rms_nm=noise_rms,
        structure_rms_nm=structure_rms, total_fit_rms_nm=total_fit_rms,
        corrected_rms_nm=corrected_rms, candidate_corrected_rms_nm=candidate_corrected_rms,
        corrected_variance_nm2=corrected_variance,
        selected_sse_nm6=selected["sse_nm6"], white_floor_sse_nm6=white_sse,
        structure_evidence_improvement=improvement,
    )


@dataclass(frozen=True)
class RepeatScanModelComparison:
    """双图对照：同一测得谱上，单图模型平均噪声与现有差值谱/2 的并列对照。

    不先扣单图再扣双图：model_corrected = measured - model_noise，
    difference_corrected = measured - difference_noise，两条校正谱从同一
    measured 出发、彼此独立，本结果不提供也不计算任何“双重扣除”量。

    保留两次不同的单图噪声：first_model.noise_density_nm3 与
    second_model.noise_density_nm3 分别保留，model_noise_density_nm3 是它们的
    平均（对应“平均单次噪声”的模型估计）。
    """
    first_psd: RoughnessSpectrum
    second_psd: RoughnessSpectrum
    first_model: PsdModelResult
    second_model: PsdModelResult
    frequency_per_nm: np.ndarray
    band_per_nm: tuple
    rms_band_per_nm: tuple
    measured_density_nm3: np.ndarray
    first_model_noise_density_nm3: np.ndarray
    second_model_noise_density_nm3: np.ndarray
    model_noise_density_nm3: np.ndarray
    difference_noise_density_nm3: np.ndarray
    model_corrected_density_nm3: np.ndarray
    difference_corrected_density_nm3: np.ndarray
    measured_rms_nm: float
    model_noise_rms_nm: float
    difference_noise_rms_nm: float
    model_corrected_rms_nm: float | None
    difference_corrected_rms_nm: float | None
    noise_agreement: str
    status: str
    status_reasons: tuple
    reliable: bool
    method_note: str

    @property
    def is_reliable(self) -> bool:
        return self.status == "ok"

    def summary(self) -> dict:
        return {
            "status": self.status,
            "reliable": self.reliable,
            "measured_rms_nm": self.measured_rms_nm,
            "model_noise_rms_nm": self.model_noise_rms_nm,
            "difference_noise_rms_nm": self.difference_noise_rms_nm,
            "model_corrected_rms_nm": self.model_corrected_rms_nm,
            "difference_corrected_rms_nm": self.difference_corrected_rms_nm,
            "rms_band_low_per_nm": self.rms_band_per_nm[0],
            "rms_band_high_per_nm": self.rms_band_per_nm[1],
            "noise_agreement": self.noise_agreement,
            "status_reasons": ";".join(self.status_reasons),
        }


def repeat_scan_model_comparison(
    repeat_result: RepeatScanSpectrum,
    pixel_size_nm: float,
    *,
    model: str = "auto",
    fit_min: float | None = None,
    fit_max: float | None = None,
    integration_min: float | None = None,
    integration_max: float | None = None,
    noise_context: str = "unknown",
    disagreement_limit: float = REPEAT_NOISE_DISAGREEMENT_LIMIT,
) -> RepeatScanModelComparison:
    """在共同有效行、同窗长、同 Hann 窗、同频率网格上做单图模型与双图差值的对照。

    第一/第二 PSD 依据 repeat_result.common_rows / first_residual_nm /
    second_residual_nm 用 roughness_spectrum 重算（与 repeat_scan_spectrum 完全
    相同的流程与像素标定，确保同窗长、同 Hann、同频率网格），再分别用
    fit_psd_model 拟合。单图模型噪声取两个模型噪声谱的平均（对应“平均单次
    噪声”的模型估计），与 repeat_result.noise_density_nm3（现有差值谱/2）对照。

    不串行扣噪：model_corrected = measured - model_noise，
    difference_corrected = measured - difference_noise，均从同一 measured 出发、
    彼此独立，不再相减。两次不同的单图噪声在结果中分别保留。

    前提校验：重算的两次 PSD 必须与 repeat_result.measured 同频率网格、且
    (first+second)/2 与 repeat_result.measured.density_nm3 一致，否则抛
    ValueError（说明传入的 pixel_size_nm 或数据状态与建 repeat 时不一致）。

    状态：两模型均可靠且两种噪声估计一致 → "ok"；否则 "unreliable" 并给原因。
    """
    if not math.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
        raise ValueError("PSD 需要有效的像素尺寸标定。")

    first_psd = roughness_spectrum(
        [(repeat_result.common_rows, repeat_result.first_residual_nm)], pixel_size_nm,
    )
    second_psd = roughness_spectrum(
        [(repeat_result.common_rows, repeat_result.second_residual_nm)], pixel_size_nm,
    )
    frequency = repeat_result.measured.frequency_per_nm
    if not (np.allclose(first_psd.frequency_per_nm, frequency)
            and np.allclose(second_psd.frequency_per_nm, frequency)):
        raise ValueError("重算的单图 PSD 与双图结果频率网格不一致（像素标定或数据状态异常）。")
    if not np.allclose((first_psd.density_nm3 + second_psd.density_nm3) / 2,
                       repeat_result.measured.density_nm3, rtol=1e-9):
        raise ValueError("双图测得谱与两次单图测得谱平均不一致（数据状态异常）。")

    first_model = fit_psd_model(
        first_psd, model=model, fit_min=fit_min, fit_max=fit_max,
        integration_min=integration_min, integration_max=integration_max,
        noise_context=noise_context,
    )
    second_model = fit_psd_model(
        second_psd, model=model, fit_min=fit_min, fit_max=fit_max,
        integration_min=integration_min, integration_max=integration_max,
        noise_context=noise_context,
    )

    f_band = first_model.frequency_per_nm
    if not np.allclose(f_band, second_model.frequency_per_nm):
        raise ValueError("两次单图模型不在同一拟合频段上（数据状态异常）。")
    start = int(np.flatnonzero(np.isclose(frequency, f_band[0]))[0])
    stop = start + len(f_band)
    if not np.allclose(frequency[start:stop], f_band):
        raise ValueError("单图模型频段与双图结果不对齐（数据状态异常）。")

    measured = repeat_result.measured.density_nm3[start:stop]
    difference_noise = repeat_result.noise_density_nm3[start:stop]
    difference_corrected = repeat_result.corrected_density_nm3[start:stop]
    first_noise = first_model.noise_density_nm3
    second_noise = second_model.noise_density_nm3
    model_noise = (first_noise + second_noise) / 2.0
    model_corrected = measured - model_noise
    frequency_step = repeat_result.measured.frequency_step_per_nm
    band = (float(f_band[0]), float(f_band[-1]))

    # 实际积分频段：与 fit_psd_model 相同的半 df 边界约定（tolerance = df/2），
    # measured / model_noise / difference_noise / model_corrected /
    # difference_corrected 的方差与 RMS 都必须在该频段内计算，而不是整个拟合
    # 频段。Integration 参数已传给两次 fit_psd_model，这里必须用同一频段汇总，
    # 否则 integration_min/integration_max 会被忽略而得到整个拟合频段的 RMS。
    tolerance = 0.5 * frequency_step
    if integration_min is None:
        integration_min = float(f_band[0])
    if integration_max is None:
        integration_max = float(f_band[-1])
    r_mask = (f_band >= integration_min - tolerance) & (f_band <= integration_max + tolerance)
    r_indexes = np.flatnonzero(r_mask)
    if r_indexes.size == 0:
        raise ValueError("积分频段内没有频点。")
    r_start, r_stop = int(r_indexes[0]), int(r_indexes[-1]) + 1
    rms_band = (float(f_band[r_start]), float(f_band[r_stop - 1]))

    measured_rms = _rms_over_band(measured, r_start, r_stop, frequency_step)
    model_noise_rms = _rms_over_band(model_noise, r_start, r_stop, frequency_step)
    difference_noise_rms = _rms_over_band(difference_noise, r_start, r_stop, frequency_step)
    model_variance = float(np.sum(model_corrected[r_start:r_stop])) * frequency_step
    difference_variance = float(np.sum(difference_corrected[r_start:r_stop])) * frequency_step

    reasons: list[str] = []
    status = "ok"
    for label, single in (("first", first_model), ("second", second_model)):
        if single.status != "ok":
            status = "unreliable"
            reasons.append(
                f"{label}_model_{single.status}:{','.join(single.reliability_reasons) or 'unspecified'}"
            )
    larger_noise = max(model_noise_rms, difference_noise_rms)
    if larger_noise <= 0.0:
        noise_agreement = "consistent"
    else:
        relative_difference = abs(model_noise_rms - difference_noise_rms) / larger_noise
        noise_agreement = "consistent" if relative_difference <= disagreement_limit else "disagrees"
    if noise_agreement == "disagrees":
        status = "unreliable"
        reasons.append("single_image_vs_difference_noise_disagree")

    # model_corrected_rms_nm 依赖“两个源模型均可靠”，且自身 status 必须 ok；
    # 任一源模型不可靠（其 corrected_rms_nm 已是 None）则平均模型噪声不可作
    # 可靠扣除，model_corrected_rms_nm 一并置 None。差分校正 RMS 是独立估计，
    # 不因单图模型不可靠被连带抹掉——它只受自身非正方差约束。
    source_models_reliable = (
        first_model.status == "ok" and second_model.status == "ok"
    )
    if model_variance <= 0:
        status = "unreliable"
        reasons.append("model_corrected_variance_nonpositive")
        model_corrected_rms = None
    else:
        model_corrected_rms = math.sqrt(model_variance)
    if not source_models_reliable:
        model_corrected_rms = None

    difference_corrected_rms = math.sqrt(difference_variance) if difference_variance > 0 else None
    if difference_variance <= 0:
        status = "unreliable"
        reasons.append("difference_corrected_variance_nonpositive")

    return RepeatScanModelComparison(
        first_psd=first_psd, second_psd=second_psd,
        first_model=first_model, second_model=second_model,
        frequency_per_nm=f_band, band_per_nm=band, rms_band_per_nm=rms_band,
        measured_density_nm3=measured,
        first_model_noise_density_nm3=first_noise,
        second_model_noise_density_nm3=second_noise,
        model_noise_density_nm3=model_noise,
        difference_noise_density_nm3=difference_noise,
        model_corrected_density_nm3=model_corrected,
        difference_corrected_density_nm3=difference_corrected,
        measured_rms_nm=measured_rms,
        model_noise_rms_nm=model_noise_rms,
        difference_noise_rms_nm=difference_noise_rms,
        model_corrected_rms_nm=model_corrected_rms,
        difference_corrected_rms_nm=difference_corrected_rms,
        noise_agreement=noise_agreement,
        status=status, status_reasons=tuple(reasons), reliable=(status == "ok"),
        method_note=(
            "单图模型噪声与双图差值噪声是同一随机噪声的两种独立估计途径，均从同一"
            "测得谱出发；model_corrected = measured - model_noise，"
            "difference_corrected = measured - difference_noise，两条校正谱彼此"
            "独立，不得再相减（不重复扣噪）。"
        ),
    )
