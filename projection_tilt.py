"""Projection profile under a small candidate tilt (base layer, phase 1).

Given a 2-D image, ``prepare_projection`` builds a cache of per-row real FFTs of
horizontally reflected-padded rows.  ``mean_projection`` then shifts every
candidate row horizontally by the tilt-induced amount

    s(y, r) = (y - center) * tan(deg2rad(r))

(where ``r`` is the extra tilt correction in degrees, ``center = (H - 1) / 2``
and ``s`` has one entry per *row*), applies the shift as a phase ramp in the
Fourier domain, averages the rows and takes one inverse transform.

Phase 2 adds :func:`refine_projection_tilt`, a narrow-range (about ±0.5°)
refinement of an initial tilt correction.  It rotates the source once with
Pillow (BICUBIC, F mode), takes the largest common valid centre rectangle,
finds at least three isolated same-polarity bright peaks in the r = 0 mean
projection, then searches r in ±0.5° for the angle that minimises the median
per-peak width (variance-weighted smoothed centroid over a fixed per-peak
window), relative to each peak's own r = 0 width.  No gate, polynomial or
vertical sin-scaling model is introduced; the base phase formula above is
unchanged.

The two stages are combined as ``correction = initial + residual``: the caller
supplies the initial (coarse) correction, this module only estimates a small
residual, and the ``refined`` boolean says whether a residual was accepted.
Every accepted angle is a numerical estimate on this module's own grid and
consensus tolerance; it is not a measurement-accuracy guarantee, and it says
nothing about real SEM frames under other noise, focus or texture conditions.

Only NumPy and Pillow are used; nothing here touches the UI or modifies the
source image in place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

MAX_ROW_SAMPLES = 128
_PAD_EXTRA = 16
MAX_RESIDUAL_LIMIT_DEGREES = 45.0

# --- phase 2 (narrow-range refinement) -----------------------------------
REFINE_MAX_RESIDUAL_DEGREES = 0.5
REFINE_COARSE_STEP = 0.01
REFINE_FINE_STEP = 0.002
REFINE_FINE_HALF_WIDTH = 0.01
REFINE_MIN_PEAKS = 3
REFINE_AGREEMENT_FRACTION = 2.0 / 3.0
# Independent per-peak voting tolerance (degrees).  Deliberately NOT tied to
# REFINE_FINE_STEP: the fine grid is a numerical search resolution, not an
# accuracy statement, and the agreement window must not silently follow it.
REFINE_PEAK_CONSENSUS_DEGREES = 0.02
REFINE_SCORE_REL_GAIN = 1e-5
REFINE_EDGE_MARGIN_DEGREES = 0.5
ROTATE_BOUNDARY_PX = 3
MIN_ROI_SIDE = 32
PEAK_COVERAGE_PERCENTILE = 50.0
PEAK_MIN_PROMINENCE_FRACTION = 0.12
PEAK_SMOOTH_HALF_WIDTH = 2
PEAK_MIN_SPAN_FRACTION = 0.35
LOST_PEAK_RATIO = 0.05


@dataclass(frozen=True)
class PreparedProjection:
    """Cached per-row spectra for one image and one maximum residual angle."""

    spectra: np.ndarray        # complex, shape (rows, N)
    n: int                     # N = W + 2P
    width: int                 # W (original columns)
    pad: int                   # P
    freq: np.ndarray           # rfftfreq(N), shape (N // 2 + 1,)
    rowdy: np.ndarray          # selected_y - center, shape (rows,)
    selected_y: np.ndarray     # original row indices, shape (rows,)
    max_residual_degrees: float
    shape: tuple               # original (H, W)


def _validate_image(image):
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {arr.shape}")
    arr = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("image contains non-finite values")
    return arr


def _validate_angle(value, name):
    try:
        deg = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite scalar, got {value!r}")
    if not np.isfinite(deg):
        raise ValueError(f"{name} must be finite, got {deg!r}")
    return deg


def _select_rows(h):
    count = min(h, MAX_ROW_SAMPLES)
    if count >= h:
        return np.arange(h, dtype=np.int64)
    return np.unique(np.linspace(0, h - 1, count).round().astype(np.int64))


def prepare_projection(image, max_residual_degrees=0.5):
    """Return a :class:`PreparedProjection` cache for ``image``.

    ``max_residual_degrees`` bounds the residual tilt correction that
    :func:`mean_projection` may be asked to apply (degrees).  It must lie in
    ``0 .. 45`` so that the reflect padding stays ``<= H`` wide and the
    candidate shifts ``dy * tan(deg2rad(r))`` used by :func:`mean_projection`
    and :func:`refine_projection_tilt` cannot explode.
    """
    arr = _validate_image(image)
    maxr = _validate_angle(max_residual_degrees, "max_residual_degrees")
    if maxr < 0 or maxr > MAX_RESIDUAL_LIMIT_DEGREES:
        raise ValueError(
            f"max_residual_degrees must lie in 0..{MAX_RESIDUAL_LIMIT_DEGREES:g}, "
            f"got {maxr!r}"
        )

    h, w = arr.shape
    if h < 1 or w < 1:
        raise ValueError(f"image must not be empty, got shape {arr.shape}")

    center = (h - 1) / 2.0
    pad = int(np.ceil(center * np.tan(np.deg2rad(maxr)))) + _PAD_EXTRA
    n = w + 2 * pad

    selected_y = _select_rows(h)
    rowdy = selected_y.astype(np.float64) - center

    padded = np.pad(arr[selected_y], ((0, 0), (pad, pad)), mode="reflect")
    spectra = np.fft.rfft(padded, axis=1)
    freq = np.fft.rfftfreq(n)

    return PreparedProjection(
        spectra=spectra,
        n=n,
        width=w,
        pad=pad,
        freq=freq,
        rowdy=rowdy,
        selected_y=selected_y,
        max_residual_degrees=maxr,
        shape=(h, w),
    )


def mean_projection(prepared, correction_degrees):
    """Row arithmetic mean of ``prepared`` rows after shifting each row by the
    candidate tilt correction ``correction_degrees``.

    Returns a 1-D profile of length ``prepared.width``.
    """
    if not isinstance(prepared, PreparedProjection):
        raise ValueError("prepared must come from prepare_projection()")
    r = _validate_angle(correction_degrees, "correction_degrees")
    if abs(r) > prepared.max_residual_degrees + 1e-12:
        raise ValueError(
            f"correction_degrees {r!r} exceeds prepared range "
            f"±{prepared.max_residual_degrees!r}"
        )

    shift = prepared.rowdy * np.tan(np.deg2rad(r))
    phase = np.exp(-2j * np.pi * shift[:, None] * prepared.freq[None, :])
    shifted = prepared.spectra * phase
    profile = np.fft.irfft(shifted.mean(axis=0), n=prepared.n)
    p = prepared.pad
    return profile[p:p + prepared.width].copy()


# =========================================================================
# Phase 2: narrow-range refinement of an initial tilt correction.
# =========================================================================


@dataclass(frozen=True)
class ProjectionTiltResult:
    """Outcome of :func:`refine_projection_tilt`.

    ``correction_degrees`` is ALWAYS ``initial_degrees + residual``, where
    ``initial_degrees`` is the caller's initial correction (degrees) and the
    residual is the small angle this module found.  The boolean field
    ``refined`` (not a numeric offset) says whether a residual was accepted:
    when it is False the residual is 0.0, so ``correction_degrees`` equals
    ``initial_degrees`` exactly and ``reason`` states why.
    ``angle_samples`` / ``score_samples`` are the two same-length numeric
    tuples over every angle actually scored, in evaluation order:
    ``angle_samples`` holds the residual angles ``r`` (degrees) and
    ``score_samples`` the matching median relative peak width.  These are
    numerical search-precision samples only, not an accuracy claim.
    A peak lost at ``r`` records ``score_samples[i] = inf``; both tuples are
    empty when nothing was scored (e.g. refused before scoring).
    """

    initial_degrees: float
    correction_degrees: float
    refined: bool
    reason: str
    peak_count: int
    score_before: float | None
    score_after: float | None
    angle_samples: tuple = ()
    score_samples: tuple = ()


@dataclass(frozen=True)
class _Rotation:
    """Output of the single source rotation, cropped to a common rectangle."""

    image: np.ndarray
    roi: tuple        # (x0, y0, x1, y1) inclusive in the rotated image
    origin_x: int
    origin_y: int


@dataclass(frozen=True)
class _Peak:
    index: int        # position of the local maximum in the ROI profile
    window: slice     # fixed, non-overlapping evaluation window (ROI columns)


def _refuse(initial, reason, peaks=0, samples=None):
    """Unrefined result that keeps the caller's initial correction.

    ``samples`` is the already-evaluated ``{angle: (value, ...)}`` mapping (or
    None); only its angles and numeric scores are kept, as two tuples of the
    same length.  ``score_before`` / ``score_after`` stay None: nothing was
    accepted.
    """
    angle_samples, score_samples = _split_samples(samples)
    return ProjectionTiltResult(
        initial_degrees=initial,
        correction_degrees=initial,
        refined=False,
        reason=reason,
        peak_count=peaks,
        score_before=None,
        score_after=None,
        angle_samples=angle_samples,
        score_samples=score_samples,
    )


def _rotate_source(item, degrees):
    """Rotate ``item`` once (PIL, BICUBIC, float32) and crop a common centre ROI.

    ``item`` is an image path or a 2-D ndarray.  The source is never modified.
    Returns ``(_Rotation, None)`` or ``(None, chinese_reason)``.
    """
    try:
        if isinstance(item, (str, bytes)):
            with Image.open(item) as handle:
                pil = handle.convert("F")
        else:
            arr = np.asarray(item)
            if arr.ndim != 2:
                return None, f"图像必须是二维灰度图，实际形状 {arr.shape}"
            if arr.size == 0:
                return None, f"图像为空，实际形状 {arr.shape}"
            # 16-bit SEM frames rotate exactly in their own integer mode; an
            # ``Image.fromarray(float32, mode="F")`` round-trip would re-quantize
            # them.  8-bit frames keep the established float path (PIL clamps
            # integers to 0..255, which is what the projection owns).
            if arr.dtype == np.uint16:
                pil = Image.fromarray(np.ascontiguousarray(arr)).convert("F")
            else:
                if arr.dtype != np.float32:
                    arr = np.asarray(arr, dtype=np.float32)
                pil = Image.fromarray(arr, mode="F")
    except (OSError, ValueError, TypeError) as exc:
        return None, f"无法读取图像：{exc}"

    h, w = int(pil.height), int(pil.width)
    if degrees != 0.0:
        pil = pil.rotate(degrees, resample=Image.BICUBIC)
    rot = np.asarray(pil, dtype=np.float64)
    hr, wr = rot.shape

    u = (w - 1) / 2.0 - ROTATE_BOUNDARY_PX
    v = (h - 1) / 2.0 - ROTATE_BOUNDARY_PX
    c = abs(np.cos(np.deg2rad(degrees)))
    s = abs(np.sin(np.deg2rad(degrees)))
    # Half extents of the rotated half-width u and half-height v both inside the
    # source rectangle (t is the same scale in both axes, so no aspect change):
    # U = t(u c + v s) <= u  and  V = t(u s + v c) <= v, with separate denominators.
    denom_u = c * u + s * v
    denom_v = s * u + c * v
    if u <= 1.0 or v <= 1.0 or denom_u <= 1e-12 or denom_v <= 1e-12:
        return None, "图像在初始旋转后没有可用的共同有效矩形（图像太小）"
    t = min(1.0, u / denom_u, v / denom_v)
    half_w = int(np.floor(t * u))
    half_h = int(np.floor(t * v))
    if 2 * half_h + 1 < MIN_ROI_SIDE or 2 * half_w + 1 < MIN_ROI_SIDE:
        return None, f"初始旋转后的共同有效区域过小（{2 * half_w + 1}×{2 * half_h + 1}）"

    cx, cy = (wr - 1) / 2.0, (hr - 1) / 2.0
    x0, x1 = int(round(cx - half_w)), int(round(cx + half_w))
    y0, y1 = int(round(cy - half_h)), int(round(cy + half_h))
    if y1 > hr - 1:
        y0, y1 = y0 - (y1 - (hr - 1)), 0 + half_h
    if x1 > wr - 1:
        x0, x1 = x0 - (x1 - (wr - 1)), 0 + half_w
    if y0 < 0 or x0 < 0 or y1 < 0 or x1 < 0:
        return None, "共同有效矩形超出旋转后图像范围"
    roi = (x0, y0, x1, y1)
    return _Rotation(image=rot, roi=roi, origin_x=x0, origin_y=y0), None


def _smooth(values, half_width=PEAK_SMOOTH_HALF_WIDTH):
    """Plain moving average (edge-normalised) for peak detection only."""
    if half_width <= 0 or values.size <= 1:
        return values.copy()
    kernel = np.ones(2 * half_width + 1, dtype=np.float64)
    num = np.convolve(values, kernel, mode="same")
    den = np.convolve(np.ones(values.size, dtype=np.float64), kernel, mode="same")
    return num / den


def _profile_scale(profile):
    return max(float(np.max(profile) - np.min(profile)), float(np.abs(profile).max()) * 1e-6, 1e-12)


def _select_peaks(profile, margin):
    """Pick >= ``REFINE_MIN_PEAKS`` isolated same-polarity peaks.

    Returns ``(peaks, polarity, None)``; the caller must score the polarity-
    flipped profile (``polarity * profile``) so every peak is a positive
    excursion.  The whole profile is flipped once when the dark lines
    dominate, so only one polarity is ever accepted: no bright and dark peaks
    are mixed in one score.  Noise/reflection columns are excluded by
    ``margin`` and by each peak's half-width; windows never overlap and never
    reach the image edge.

    Returns ``(peaks, polarity, None)`` or ``(None, 0, chinese_reason)``.
    """
    p = np.asarray(profile, dtype=np.float64)
    inner = p[margin:p.size - margin]
    if inner.size < REFINE_MIN_PEAKS * 2:
        return None, 0, "两侧边距扣掉后有效列太少"
    inner = inner - np.median(inner)
    if float(np.max(inner) - np.min(inner)) <= 0:
        return None, 0, "投影剖面没有对比度"
    polarity = 1
    if -(np.min(inner)) > np.max(inner):
        inner = -inner
        polarity = -1

    lowest = float(np.min(inner))
    highest = float(np.max(inner))
    background = float(np.percentile(inner, PEAK_COVERAGE_PERCENTILE))
    if highest - lowest <= 1e-12 or highest - background <= 1e-12:
        return None, 0, "投影剖面没有可用的正峰（背景或对比度不足）"
    limit = background + PEAK_MIN_PROMINENCE_FRACTION * (highest - background)

    # Candidate centres: raw local maxima that clear the height gate.  A
    # moving average only rejects plateau/noise blips; it never invents height,
    # so a broad smear stays below the gate.
    radius = max(4, int(round(inner.size / (REFINE_MIN_PEAKS * 8))))
    smooth = _smooth(inner, 1)
    is_max = np.zeros(inner.size, dtype=bool)
    is_max[1:-1] = (inner[1:-1] >= inner[:-2]) & (inner[1:-1] >= inner[2:])
    candidates = np.flatnonzero(is_max)
    candidates = candidates[
        (inner[candidates] >= limit) & (smooth[candidates] >= limit)
    ]
    # Span gate: a real line carries a steep rise/fall, so the height range
    # inside its own window is a sizeable part of its height.  A ripple on a
    # broad smear (typical after a large initial rotation) is a local maximum
    # whose window stays flat, and must not become a scoring peak.
    span_half = max(PEAK_SMOOTH_HALF_WIDTH, radius // 2)
    if candidates.size:
        kept_cands = []
        for idx in candidates.tolist():
            lo = max(0, idx - span_half)
            hi_i = min(inner.size, idx + span_half + 1)
            if float(inner[lo:hi_i].max() - inner[lo:hi_i].min()) < (
                PEAK_MIN_SPAN_FRACTION * float(inner[idx])
            ):
                continue
            kept_cands.append(idx)
        candidates = np.array(kept_cands, dtype=np.intp)
    # Minimum separation by the fixed peak radius (greedy NMS), then restore
    # profile order so windows are assigned left to right.
    keep = []
    for idx in candidates.tolist():
        if any(abs(idx - other) <= radius for other in keep):
            continue
        keep.append(idx)
    seen = sorted(keep)
    if len(seen) < REFINE_MIN_PEAKS:
        return None, 0, (
            f"投影剖面中只找到 {len(seen)} 个显著同极性峰（需要 >= {REFINE_MIN_PEAKS}）"
        )

    # Fixed, non-overlapping window per peak: half-width from the peak gap, so
    # neighbouring peaks can never leak into one window or shrink it per angle.
    half = max(PEAK_SMOOTH_HALF_WIDTH, radius // 2)
    half = int(min(half, (inner.size // len(seen) - 1) // 2))
    if half < PEAK_SMOOTH_HALF_WIDTH:
        return None, 0, "峰太密，无法给每个峰分配固定的不重叠窗口"
    kept = []
    for idx in sorted(seen):
        if kept and idx - kept[-1] < 2 * half + 2:
            continue
        kept.append(idx)
    peaks = []
    for idx in kept:
        start, stop = idx - half, idx + half + 1
        if start < 0 or stop > inner.size:      # drop, never crop/anchor at edge
            continue
        peaks.append(_Peak(index=margin + idx, window=slice(margin + start, margin + stop)))
    if len(peaks) < REFINE_MIN_PEAKS:
        return None, 0, f"可用的完整峰窗口只有 {len(peaks)} 个（需要 >= {REFINE_MIN_PEAKS}）"
    return peaks, polarity, None


def _peak_width(profile, peak):
    """Local-endpoint-linear-baseline weighted centroid variance of one peak."""
    seg = np.asarray(profile[peak.window], dtype=np.float64)
    n = seg.size
    if n < 5:
        return float("nan"), float("nan")
    edge = max(1, n // 8)
    left = float(np.mean(seg[:edge]))
    right = float(np.mean(seg[-1 - edge + 1:]))
    baseline = left + (right - left) * np.arange(n) / (n - 1.0)
    positive = seg - baseline
    positive[positive <= 0] = 0.0
    total = float(positive.sum())
    if not np.isfinite(total) or total <= 1e-9:
        return float("nan"), float("nan")
    x = np.arange(n, dtype=np.float64)
    centre = float((positive * x).sum() / total)
    variance = float((positive * (x - centre) ** 2).sum() / total)
    return variance, centre


def _profiles_for(prepared, angles):
    return {float(a): mean_projection(prepared, float(a)) for a in angles}


def refine_projection_tilt(image, initial_correction_degrees):
    """Refine ``initial_correction_degrees`` by about ±0.5° using projection widths.

    ``image`` is a file path or a 2-D ndarray (never modified).  The returned
    :class:`ProjectionTiltResult` reports the total correction
    ``correction_degrees = initial_degrees + residual`` and whether a residual
    was accepted in the boolean ``refined``; whenever the estimate is not
    reliable the initial value is returned unchanged (``refined`` False) with
    a concrete Chinese ``reason``.  An accepted residual is the best value on
    this module's search grid and peak-consensus tolerance, which is
    numerical precision — not a guarantee of measurement accuracy.
    """
    try:
        initial = _validate_angle(initial_correction_degrees, "initial_correction_degrees")
    except ValueError as exc:
        return _refuse(float("nan"), f"初始角度无效：{exc}")
    if abs(initial) > MAX_RESIDUAL_LIMIT_DEGREES:
        return _refuse(initial, f"初始角度 {initial:g}° 超出支持的 ±{MAX_RESIDUAL_LIMIT_DEGREES:g}° 范围")

    rotation, problem = _rotate_source(image, initial)
    if rotation is None:
        return _refuse(initial, problem)
    x0, y0, x1, y1 = rotation.roi
    region = rotation.image[y0:y1 + 1, x0:x1 + 1]
    region_h, region_w = region.shape
    margin = int(np.ceil((region_h - 1) / 2.0 * np.tan(np.deg2rad(REFINE_EDGE_MARGIN_DEGREES)))) + 4
    if region_w <= 2 * margin + 2 * REFINE_MIN_PEAKS:
        return _refuse(initial, f"共同有效区域宽度 {region_w} 扣除 {2 * margin} 列边距后不足")

    prepared = prepare_projection(region, REFINE_MAX_RESIDUAL_DEGREES)
    base = mean_projection(prepared, 0.0)
    peaks, polarity, problem = _select_peaks(base, margin)
    if peaks is None:
        return _refuse(initial, problem)
    sign = float(polarity)

    # Per-peak r = 0 reference widths (same fixed window as every candidate).
    r0 = [_peak_width(sign * base, pk)[0] for pk in peaks]
    if any(not np.isfinite(v) or v <= 1e-9 for v in r0):
        return _refuse(initial, "r=0 基线投影中至少一个峰缺少可测量的宽度（峰出界或对比度不足）", len(peaks))

    evaluated = {}

    def score_at(angle):
        """Median relative width over peaks that are still measurable.

        A peak that leaves its fixed window or collapses (signal lost) can
        never make a candidate look *better*: its slot keeps a real ``inf``
        ratio (the list keeps one entry per peak, in peak order), it is left
        out of the median, and the candidate's aggregate score becomes ``inf``
        so the candidate is refused outright.
        """
        angle = round(float(angle), 6)
        if angle in evaluated:
            return evaluated[angle]
        profile = mean_projection(prepared, angle)
        oriented = sign * profile
        # Fixed identity: one ratio per peak, in peak order, for every angle.
        # A lost peak keeps its slot as ``inf`` (never removed), so index k
        # always refers to the same peak and the list never shrinks.
        ratios, lost = [], 0
        for pk, ref in zip(peaks, r0):
            var = _peak_width(oriented, pk)[0]
            if not np.isfinite(var) or var < LOST_PEAK_RATIO * ref:
                lost += 1
                ratios.append(float("inf"))
                continue
            ratios.append(float(var / ref))
        measurable = [r for r in ratios if np.isfinite(r)]
        value = float(np.median(measurable)) if measurable else float("inf")
        # A lost peak is an artifact, never an improvement.
        if lost:
            value = float("inf")
        evaluated[angle] = (value, tuple(ratios), lost)
        return evaluated[angle]

    scored_angles, scored_values = [], []

    def record(angle):
        """Evaluate once and append the residual angle with its numeric score."""
        value = score_at(angle)[0]
        scored_angles.append(round(float(angle), 6))
        scored_values.append(float(value))

    coarse_angles = _angle_grid(-REFINE_MAX_RESIDUAL_DEGREES, REFINE_MAX_RESIDUAL_DEGREES,
                                REFINE_COARSE_STEP)
    for a in coarse_angles:
        record(a)
    coarse_best = min(coarse_angles, key=lambda a: evaluated[a][0])

    if abs(coarse_best) >= REFINE_MAX_RESIDUAL_DEGREES - 1e-9:
        return _refuse(initial, f"最优残差落在 ±{REFINE_MAX_RESIDUAL_DEGREES:g}° 搜索边界，不可靠",
                       len(peaks), _pairs(scored_angles, scored_values))

    fine_angles = _angle_grid(coarse_best - REFINE_FINE_HALF_WIDTH,
                              coarse_best + REFINE_FINE_HALF_WIDTH, REFINE_FINE_STEP)
    for a in fine_angles:
        record(a)
    candidates = [a for a in fine_angles if abs(a) < REFINE_MAX_RESIDUAL_DEGREES - 1e-9]
    best = min(candidates, key=lambda a: evaluated[a][0])
    best_value, best_ratios, best_lost = evaluated[best]
    before_value = evaluated[0.0][0]
    samples = _pairs(scored_angles, scored_values)

    if not np.isfinite(best_value):
        return _refuse(initial, f"最优角 {best:+.3f}° 处所有峰都失去信号，判为不可靠",
                       len(peaks), samples)
    if before_value < 0.02:
        return _refuse(initial, "r=0 剖面各峰已经很窄（残差接近 0°），无需强改",
                       len(peaks), samples)
    if best_lost:
        return _refuse(initial, f"最优角 {best:+.3f}° 处有 {best_lost} 个峰几乎消失"
                                "（峰出界而非变窄），判为不可靠", len(peaks), samples)
    if not best_value < before_value - REFINE_SCORE_REL_GAIN * max(before_value, 1e-12):
        return _refuse(initial, f"最优角 {best:+.3f}° 相对 r=0 的宽度比没有实质改善",
                       len(peaks), samples)

    # Every peak's own best angle.  Voting only uses the fixed peak identity:
    # candidates whose aggregate score is non-finite (any lost peak) are
    # excluded BEFORE indexing, as are non-finite per-peak ratios, so a peak
    # that vanished for one candidate can never shift another peak's ratios.
    own = []
    for k in range(len(r0)):
        scored = []
        for a in evaluated:
            aggregate, ratios_k, lost_k = evaluated[a]
            if lost_k or not np.isfinite(aggregate):
                continue
            if k >= len(ratios_k):
                continue
            value = ratios_k[k]
            if not np.isfinite(value):
                continue
            if abs(a) <= abs(best) + REFINE_FINE_HALF_WIDTH + REFINE_FINE_STEP:
                scored.append((value, a))
        if scored:
            own.append(min(scored)[1])
    needed = max(2, int(np.ceil(REFINE_AGREEMENT_FRACTION * len(peaks))))
    agree = sum(1 for a in own if abs(a - best) <= REFINE_PEAK_CONSENSUS_DEGREES + 1e-9)
    if len(own) < needed or agree < needed:
        return _refuse(initial, f"各峰自身最优角不一致（{agree}/{len(peaks)} 与聚合角 {best:+.3f}° 相符）",
                       len(peaks), samples)

    return ProjectionTiltResult(
        initial_degrees=initial,
        correction_degrees=initial + best,
        refined=True,
        reason=f"在 {len(peaks)} 个同极性峰上聚合一致（{agree}/{len(peaks)}）",
        peak_count=len(peaks),
        score_before=float(before_value),
        score_after=float(best_value),
        angle_samples=tuple(scored_angles),
        score_samples=tuple(scored_values),
    )


def _angle_grid(start, stop, step):
    """Angles from ``start`` to ``stop`` inclusive, rounded to 6 decimals."""
    count = int(np.floor((stop - start) / step + 1e-9))
    grid = [round(start + step * i, 6) for i in range(count + 1)]
    if abs(grid[-1] - stop) > 1e-9:
        grid.append(round(stop, 6))
    return grid


def _split_samples(pairs):
    """Split ``(angles, scores)`` pure-number tuples into the result fields.

    Refusal passes only samples already evaluated as two parallel tuples (or
    None); nothing is ever stored as a mixed ``(angle, score)`` tuple.
    """
    if not pairs:
        return tuple(), tuple()
    angles, scores = pairs
    angles = tuple(float(a) for a in angles)
    scores = tuple(float(v) for v in scores)
    if len(angles) != len(scores):
        raise ValueError(f"angle/score sample count mismatch: {len(angles)} != {len(scores)}")
    return angles, scores


def _pairs(angles, scores):
    return tuple(float(a) for a in angles), tuple(float(v) for v in scores)
