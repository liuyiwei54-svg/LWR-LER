"""Unified grayscale-profile line localization with per-row diagnostics.

This module implements the single line-measurement pipeline for SEM profiles
(design: unified localization, phase 1).  It contains:

- one per-row transition localizer: the ERA-style polarity-weighted difference
  centroid guides a locally normalized polynomial half-height fit;
- one per-row bright-band localizer: independent left/right local-baseline
  *polynomial* half-height crossings (degree 2/3/4, default 3) with the periodic
  peak/valley identity policy.  This replaces the historical linear half-height
  interpolation with the ERA two-flank polynomial fit, never delegating to the
  discovery module's linear helpers;
- a bounded, neighbor-guided re-localization stage: neighbouring valid rows may
  only guide one restricted re-read of the real pixels inside the same ROI and
  the same local fitting-window semantics.  The position definition, model,
  contrast thresholds and jump gate are never changed, externally masked /
  low-contrast rows are never retried, and rows that still fail remain missing
  (no zero filling, no interpolation, no median-as-observation);
- per-row diagnostics: candidate, final position, valid mask, stage
  (primary / relocalized / missing) and failure reason, plus the bright-band's
  two half-height crossings and center;
- missing-row statistics: valid counts, coverage, longest consecutive gap and
  reason histograms, plus left/right common-row pairing statistics;
- a conservative low-order (first/second degree) two-dimensional additive
  background model, fitted from the original working pixels on a single
  normalized global coordinate basis so corrections never accumulate and never
  absorb periodic real structure.  The regional mean is restored so intensity
  levels and the local half-height baselines keep their meaning.

Only NumPy is used; this module never touches the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_PRIMARY = "primary"
STAGE_RELOCALIZED = "relocalized"
STAGE_MISSING = "missing"

REASON_NONE = ""
REASON_LOW_CONTRAST = "low_contrast"
REASON_MISSING_FLANK = "missing_flank"
REASON_AMBIGUOUS = "ambiguous_peak"
REASON_JUMP_REJECTED = "jump_rejected"
REASON_UNRESOLVED = "unresolved"

# Bounded recovery limits: neighbours may guide only rows with at least
# RECOVERY_MIN_NEIGHBORS valid neighbours inside RECOVERY_WINDOW_ROWS, and the
# retry must pass the same jump gate as the primary pass.
RECOVERY_WINDOW_ROWS = 15
RECOVERY_MIN_NEIGHBORS = 2

LOCALIZATION_METHOD_LABEL = "统一灰度剖面定位"
BACKGROUND_DEGREE_LABELS = {1: "一阶（平面）", 2: "二阶（曲面）"}
BACKGROUND_REGION_FULL = "全图"
BACKGROUND_REGION_ROI = "整体 ROI"

# ERA difference-weighting power, preserved from the original application.
DIFFERENCE_POWER = 4
# Supported bright-band polynomial degrees (the ERA fit supports 2/3/4 only).
BRIGHT_BAND_DEGREES = (2, 3, 4)
DEFAULT_BRIGHT_BAND_DEGREE = 3

# Bright-band peak-identity ambiguity.  When several eligible base peaks sit at
# nearly equal distance from the reference center (tolerance below) AND their
# contrasts are comparable (rival contrast reaches the ratio below of the
# selected peak's contrast), the preview reference does not identify a unique
# target, so the row is rejected as ``ambiguous_peak`` instead of silently
# keeping the geometrically first peak.  A narrow reference that actually points
# at one known peak keeps resolving that single peak.
#  - AMBIGUITY_DISTANCE_TOLERANCE_PX: two peaks are "nearly equidistant" when
#    their |center - reference_center| distances differ by at most this many px.
#  - AMBIGUITY_CONTRAST_RATIO: a rival is "comparably strong" when its contrast
#    reaches this fraction of the nearest peak's contrast.
AMBIGUITY_DISTANCE_TOLERANCE_PX = 1.5
AMBIGUITY_CONTRAST_RATIO = 0.50

# Transition-validity thresholds (baked in, identical for the primary pass and the
# neighbor-guided retry).  A directed transition is only trusted when a single,
# well-separated significant difference peak dominates the local window.
#  - TRANSSITION_NOISE_FACTOR: a difference peak must exceed this many times the
#    robust window noise to count as a real transition group.
#  - TRANSSITION_RELATIVE_FLOOR: a second peak is only "comparable" (and therefore
#    ambiguous) when it reaches this fraction of the strongest peak's height.
#  - TRANSSITION_GROUP_GAP_PX: difference peaks closer than this many pixels are
#    folded into the same transition group (noise ripple around one step).
#  - TRANSSITION_VALLEY_FRACTION: two peaks are only "two transitions" when the
#    minimum between them drops to this fraction of the weaker peak (a real
#    valley separating two physical steps, not a shoulder).
#  - TRANSSITION_ENDPOINT_NOISE_FACTOR: pure-noise windows are rejected when the
#    endpoint contrast is below this many times the robust window noise.
TRANSSITION_NOISE_FACTOR = 3.0
TRANSSITION_RELATIVE_FLOOR = 0.30
TRANSSITION_GROUP_GAP_PX = 4
TRANSSITION_VALLEY_FRACTION = 0.50
TRANSSITION_ENDPOINT_NOISE_FACTOR = 3.0
# Two-physical-step discrimination (the double-transition criterion).  A window
# is reported as holding two distinct physical steps (ambiguous) only when the
# *candidate root* itself — the difference-weighted centroid that would become
# the emitted edge position — lies on a sustained flat plateau *between* two
# well-separated net-confirmed step candidates.  Every feature is read from the
# same current-row's original gray level: no neighbour rows, no hidden spatial
# smoothing, and the final ERA difference-weighted-centroid polynomial
# half-height fit input is unchanged.
#
# Why the multiple-step test is needed here at all: a single ~80-amplitude step
# has a first-difference peak of ~N samples; two equal steps side by side form a
# flat middle plateau whose midpoint is NOT a physical edge.  The localizer must
# therefore reject the between-steps centroid instead of emitting it.
#
# The failure mode this design replaces (the 8-sigma fix): the previous test
# required each candidate step's difference peak to exceed 8σ of the *difference*
# noise (``TRANSSITION_STEP_STRENGTH_FACTOR``).  For a genuinely resolvable
# double step the two real peaks (~20 each in a 64 px profile) often fall below
# that 8σ threshold (≈22.6 at σ=2), so the test silently declared the window a
# single transition and emitted the false between-steps midpoint (~17.5).  No
# single *peak-strength* threshold separates a real step from noise riding a wide
# slow ramp (both span ~2–8σ), so a strength gate is the wrong tool and has been
# removed.  Instead the discriminator tests the *candidate root's* local support:
#   - ``_sustained_step_contrast``: a candidate diff-peak keeps a genuine net
#     signed level change (drops single noise spikes in flat regions).
#   - ``TRANSSITION_MIN_STEP_SEPARATION_PX``: the two strongest confirmed steps
#     must be at least this far apart to be two physical steps rather than one
#     step's noise ripple.
#   - the difference-weighted centroid (the same root the ERA fit would emit)
#     must sit strictly between the two steps AND its local first-difference
#     magnitude (robust median over ``TRANSSITION_ROOT_FLAT_WINDOW_PX`` profiles)
#     must be at most ``TRANSSITION_ROOT_FLAT_FRACTION`` of the window's peak
#     difference.  A real double step's centroid lands on the flat plateau
#     (local slope ≈ 0.10–0.19 of peak), while a wide single's centroid lands on
#     a still-rising ramp (≈ 0.38–0.87), so the flat-plateau test separates them
#     where every peak-strength and between-peak-median threshold fails.
#  - TRANSSITION_STEP_CONFIRM_WINDOW_PX: samples on each side of the step used
#    for the robust local level.
#  - TRANSSITION_STEP_CONFIRM_NET_FACTOR: the net signed contrast must exceed
#    this many times the gray-domain window noise to count as a step.
#  - TRANSSITION_MIN_STEP_SEPARATION_PX: minimum profile separation between two
#    confirmed step candidates for them to count as two physical steps.
#  - TRANSSITION_ROOT_FLAT_WINDOW_PX: half-width (profiles) around the candidate
#    root used for the robust local-slope estimate.
#  - TRANSSITION_ROOT_FLAT_FRACTION: the candidate root is "on a plateau" when
#    its local first-difference magnitude is at most this fraction of the
#    window's peak first difference.
#  - TRANSSITION_MIN_STEP_NET_FRACTION: a second, independent ambiguity arm.
#    When two well-separated net-confirmed steps EACH carry at least this
#    fraction of the window's endpoint contrast, they are two genuinely
#    substantial physical steps (a double step), not two noise ripples riding a
#    single wide ramp.  A real double step's weaker step keeps ~0.28-0.40 of the
#    endpoint contrast, whereas noise ripples on a wide ramp keep only ~0.10-0.26,
#    so this arm catches noisy double steps (whose plateau's local slope is
#    noise-dominated and can exceed the root-flat threshold) without rejecting
#    wide single transitions.
TRANSSITION_STEP_CONFIRM_WINDOW_PX = 5
TRANSSITION_STEP_CONFIRM_NET_FACTOR = 3.0
TRANSSITION_MIN_STEP_SEPARATION_PX = 7
TRANSSITION_ROOT_FLAT_WINDOW_PX = 3
TRANSSITION_ROOT_FLAT_FRACTION = 0.30
TRANSSITION_MIN_STEP_NET_FRACTION = 0.28

# Supported transition polynomial degrees (ERA manual single-edge fits 2/3/4).
TRANSITION_DEGREES = (2, 3, 4)

# Minimum number of valid rows for a single edge, and the historical
# 60-percent row support gate, both preserved from the pre-unification code.
MIN_EDGE_VALID_ROWS = 30
MIN_EDGE_VALID_FRACTION = 0.6


def _nan() -> float:
    return float("nan")


# ---------------------------------------------------------------------------
# Row-level transition localizer (ERA single-edge)
# ---------------------------------------------------------------------------

def _robust_window_noise(values: np.ndarray) -> float:
    """Robust white-noise sigma from the local window's second differences.

    ``median(|second difference|) / (sqrt(6) * 0.6745)`` recovers the additive
    white-noise standard deviation of a profile whose local second difference is
    otherwise flat (the same estimator used by ``_flank_noise``).  A clean flat
    region or a noiseless step gives noise near zero; a pure-noise window gives
    its true sigma.
    """
    if values.size < 5:
        return 0.0
    second = np.diff(values, n=2)
    if second.size == 0:
        return 0.0
    return float(np.median(np.abs(second)) / (math.sqrt(6.0) * 0.6744897501960817))


def _endpoint_contrast(values: np.ndarray) -> float:
    """Robust level difference across a local window (outer-sample medians)."""
    if values.size < 4:
        return 0.0
    count = max(2, min(5, values.size // 4))
    left = float(np.median(values[:count]))
    right = float(np.median(values[-count:]))
    return abs(right - left)


def _transition_group_positions(local_difference: np.ndarray, noise: float) -> list[tuple[int, float]]:
    """Representative ``(index, strength)`` positions of significant separated
    transition groups in a positive first-difference window.

    A local maximum of the positive difference counts as a group only when it
    exceeds the noise gate and the dominant-window relative floor; maxima closer
    than ``TRANSSITION_GROUP_GAP_PX`` are folded into the strongest one.  No
    derivative kernel or spatial smoothing is applied: peaks are read directly
    off the existing first-difference vector.
    """
    d = np.asarray(local_difference, dtype=float)
    n = len(d)
    if n < 3:
        return []
    peak_floor = max(TRANSSITION_NOISE_FACTOR * noise, 0.08 * float(np.max(d)), 1e-12)
    # Local maxima with plateau handling: a plateau of equal values counts once,
    # at its argmax (the flat-top's representative position).
    maxima: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and d[j + 1] == d[i]:
            j += 1
        left = d[i - 1] if i > 0 else -math.inf
        right = d[j + 1] if j + 1 < n else -math.inf
        if d[i] > left and d[i] > right and d[i] >= peak_floor:
            maxima.append(i + int(np.argmax(d[i:j + 1])))
        i = j + 1
    if not maxima:
        return []
    # Fold near-by maxima into distinct transition groups.
    groups: list[tuple[int, float]] = []
    for idx in maxima:
        strength = float(d[idx])
        if groups and idx - groups[-1][0] <= TRANSSITION_GROUP_GAP_PX:
            if strength > groups[-1][1]:
                groups[-1] = (idx, strength)
            continue
        groups.append((idx, strength))
    return groups


def _sustained_step_contrast(
    values: np.ndarray,
    diff_index: int,
    noise: float,
    polarity_sign: int,
) -> float:
    """Net signed level contrast confirmed at a first-difference peak, or 0.0.

    ``diff_index`` is a *profile* difference index: the candidate step sits
    between profile index ``diff_index`` and ``diff_index + 1``.  The robust
    local level is the median of ``TRANSSITION_STEP_CONFIRM_WINDOW_PX`` samples
    on each side; the step is real when the far side stays above the near side
    by at least ``TRANSSITION_STEP_CONFIRM_NET_FACTOR`` times the gray-domain
    window noise in the ``polarity_sign`` direction.  Returns that net contrast
    (non-negative) when confirmed, else 0.0.  A single noise spike in a flat
    region averages to ~0 and is not confirmed; a real step or a point on a wide
    monotonic ramp both keep a genuine net contrast and are separated downstream
    by the difference-peak strength test.
    """
    n = len(values)
    sign = 1 if polarity_sign >= 0 else -1
    count = TRANSSITION_STEP_CONFIRM_WINDOW_PX
    lo = int(diff_index)
    if lo < 0 or lo + 1 >= n:
        return 0.0
    left = values[max(0, lo - count + 1): lo + 1]
    right = values[lo + 1: min(n, lo + 1 + count)]
    if left.size < 3 or right.size < 3:
        return 0.0
    net_signed = sign * (float(np.median(right)) - float(np.median(left)))
    if net_signed < TRANSSITION_STEP_CONFIRM_NET_FACTOR * max(noise, 1e-12):
        return 0.0
    return float(net_signed)


def _multiple_transitions(
    values: np.ndarray,
    local_difference: np.ndarray,
    diff_start: int,
    noise: float,
    polarity_sign: int,
) -> bool:
    """True only when the window holds two distinct physical steps.

    A window is ambiguous (holds two transitions) only when, in addition to two
    well-separated net-confirmed step candidates, the *candidate root* (the
    difference-weighted centroid that the ERA fit would otherwise emit) sits on
    a sustained flat plateau strictly between those two steps.  This is the
    robust replacement for the removed 8-sigma peak-strength gate: a real double
    step's false midpoint lies on the flat between-steps plateau, whereas a wide
    single transition's centroid lies on a still-rising ramp.  No spatial
    smoothing, neighbour-row information, or hidden filtering is used; the
    polynomial-fit input is unchanged.
    """
    groups = _transition_group_positions(local_difference, noise)
    strongest: list[tuple[int, float]] = []
    for group_index, strength in groups:
        step_index = diff_start + group_index
        contrast = _sustained_step_contrast(values, step_index, noise, polarity_sign)
        if contrast > 0.0:
            strongest.append((step_index, contrast))
    strongest.sort(key=lambda item: item[1], reverse=True)
    if len(strongest) < 2:
        return False
    left_i = min(strongest[0][0], strongest[1][0])
    right_i = max(strongest[0][0], strongest[1][0])
    if right_i - left_i < TRANSSITION_MIN_STEP_SEPARATION_PX:
        return False
    # Arm B first (mutually independent of the root test): two well-separated
    # steps that EACH carry a substantial fraction of the window's endpoint
    # contrast are two real physical steps.  This catches noisy double steps
    # whose plateau slope is noise-dominated (and so can exceed the root-flat
    # threshold below); a wide single's noise ripples keep only a small fraction
    # of the endpoint contrast and are not promoted.
    window_value_span = values[diff_start: diff_start + len(local_difference) + 1]
    endurance = _endpoint_contrast(window_value_span)
    if endurance > 1e-9:
        weaker_frac = strongest[1][1] / endurance
        if weaker_frac >= TRANSSITION_MIN_STEP_NET_FRACTION:
            return True
    # Arm A: candidate root — the difference-weighted centroid (identical
    # weighting to the downstream ERA fit) — must sit on a sustained flat
    # plateau strictly between the two steps (a real double step's false
    # midpoint), rather than on a still-rising ramp (a wide single).
    peak = float(np.max(local_difference))
    if peak <= 1e-12:
        return False
    normalized = local_difference / peak
    weights = normalized ** DIFFERENCE_POWER
    if float(np.sum(weights)) <= 0.0:
        return False
    x_centers = np.arange(diff_start, diff_start + len(local_difference), dtype=float) + 0.5
    root = float(np.dot(weights, x_centers) / np.sum(weights))
    # The root must sit strictly inside the plateau between the two steps.
    if not (left_i < root < right_i):
        return False
    # Local-slope (robust median |first difference| around the root) normalized
    # by the window peak difference: a plateau read is at most a small fraction.
    ci = int(round(root - 0.5))
    lo = max(0, ci - TRANSSITION_ROOT_FLAT_WINDOW_PX)
    hi = min(len(values), ci + TRANSSITION_ROOT_FLAT_WINDOW_PX + 1)
    if hi - lo < 2:
        return False
    local_slope = float(np.median(np.abs(np.diff(values[lo:hi]))))
    return local_slope <= TRANSSITION_ROOT_FLAT_FRACTION * peak


def transition_row_position(
    profile: np.ndarray,
    reference_x: float,
    polarity_sign: int,
    polynomial_degree: int,
    fit_bounds: tuple[int, int] | None = None,
) -> tuple[float, str]:
    """Locate one directed gray-level transition at its normalized 0.5 crossing.

    This is the original ERA localization (difference-weighted centroid guides
    a locally normalized polynomial fit).  ``fit_bounds`` optionally overrides
    the automatically selected local fit window, which is the only lever used
    by the bounded neighbor-guided re-localization.

    ``fit_bounds`` is a half-open ``(start, end)`` window in *pixel* (profile
    index) coordinates.  The difference window is clamped to the width of the
    first-difference array so the right boundary, narrow windows and the
    width-1 difference length never produce a length mismatch.
    """
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1:
        return _nan(), REASON_MISSING_FLANK
    width = len(values)
    if width < 3 or not np.all(np.isfinite(values)):
        return _nan(), REASON_MISSING_FLANK
    if not isinstance(polynomial_degree, (int, np.integer)) or int(polynomial_degree) not in TRANSITION_DEGREES:
        return _nan(), REASON_MISSING_FLANK
    polynomial_degree = int(polynomial_degree)
    if not math.isfinite(float(polarity_sign)) or polarity_sign == 0:
        return _nan(), REASON_MISSING_FLANK
    polarity_sign = 1 if polarity_sign > 0 else -1
    if fit_bounds is not None:
        try:
            fb_left = int(fit_bounds[0])
            fb_right = int(fit_bounds[1])
        except (TypeError, ValueError, IndexError):
            return _nan(), REASON_MISSING_FLANK
        if len(fit_bounds) != 2 or not (math.isfinite(fb_left) and math.isfinite(fb_right)):
            return _nan(), REASON_MISSING_FLANK
    difference = polarity_sign * np.diff(values)  # length width - 1
    if fit_bounds is None:
        half_search = max(5, min(30, width // 3))
        start = max(0, int(math.floor(reference_x - half_search)))
        end = min(width - 1, int(math.ceil(reference_x + half_search)))
    else:
        start = max(0, fb_left)
        end = min(width, fb_right)
    diff_end = min(end, width - 1)
    if diff_end <= start + 1:
        return _nan(), REASON_MISSING_FLANK
    local_difference = np.maximum(difference[start:diff_end], 0.0)
    if not len(local_difference) or float(np.max(local_difference)) <= 1e-12:
        return _nan(), REASON_MISSING_FLANK
    # Validity gates applied before the difference-weighted centroid.  A pure-noise
    # window (endpoint contrast below the robust noise gate) must not produce a
    # random polynomial half-height on a real trajectory, and two comparable
    # valley-separated transitions are a false edge between two steps.
    window_values = values[start:min(end + 1, width)]
    window_noise = _robust_window_noise(values[start:min(diff_end + 1, width)])
    if _endpoint_contrast(window_values) <= TRANSSITION_ENDPOINT_NOISE_FACTOR * max(window_noise, 1e-12):
        return _nan(), REASON_LOW_CONTRAST
    if _multiple_transitions(values, local_difference, start, window_noise, polarity_sign):
        return _nan(), REASON_AMBIGUOUS
    normalized_difference = local_difference / float(np.max(local_difference))
    weights = normalized_difference ** DIFFERENCE_POWER
    x_centers = np.arange(start, diff_end, dtype=float) + 0.5
    weighted_center = float(np.dot(weights, x_centers) / np.sum(weights))
    peak_center = float(start + int(np.argmax(local_difference)) + 0.5)
    if fit_bounds is None:
        half_fit = max(3, min(15, int(math.ceil(abs(peak_center - weighted_center))) + 3))
        fit_start = max(0, int(math.floor(weighted_center - half_fit)))
        fit_end = min(width, int(math.ceil(weighted_center + half_fit)) + 1)
    else:
        fit_start, fit_end = start, end
    if fit_end <= fit_start + polynomial_degree:
        return _nan(), REASON_MISSING_FLANK
    x_values = np.arange(fit_start, fit_end, dtype=float)
    fit_values = values[fit_start:fit_end]
    if len(x_values) <= polynomial_degree or float(np.ptp(fit_values)) <= 1e-12:
        return _nan(), REASON_MISSING_FLANK
    normalized_values = (fit_values - float(np.min(fit_values))) / float(np.ptp(fit_values))
    coefficients = np.polyfit(x_values, normalized_values, polynomial_degree)
    roots = np.roots(np.array([*coefficients[:-1], coefficients[-1] - 0.5], dtype=float))
    candidates = []
    derivative = np.polyder(coefficients)
    for root in roots:
        if abs(float(root.imag)) > 1e-7:
            continue
        position = float(root.real)
        slope = float(np.polyval(derivative, position))
        if fit_start <= position <= fit_end - 1 and slope * polarity_sign > 0:
            candidates.append(position)
    if not candidates:
        return _nan(), REASON_AMBIGUOUS
    return min(candidates, key=lambda position: abs(position - weighted_center)), REASON_NONE


# ---------------------------------------------------------------------------
# Row-level bright-band localizer (ERA polynomial two-flank half height)
# ---------------------------------------------------------------------------

def _flank_noise(values: np.ndarray, endpoint_count: int) -> float:
    """Estimate baseline noise from second differences near both endpoints."""
    count = max(endpoint_count, min(6, max(3, values.size // 10)))
    count = min(count, values.size // 2)
    if count < 3:
        return 0.0
    left_difference = np.diff(values[:count])
    right_difference = np.diff(values[-count:])
    if not (np.any(left_difference[:-1] * left_difference[1:] < 0)
            or np.any(right_difference[:-1] * right_difference[1:] < 0)):
        return 0.0
    second = np.concatenate(
        (np.diff(values[:count], n=2), np.diff(values[-count:], n=2))
    )
    return float(np.median(np.abs(second)) / (math.sqrt(6.0) * 0.6744897501960817))


def _bright_band_search_bounds(size: int, reference_band_px: tuple[float, float]) -> tuple[int, int]:
    """Bound peak identity by the preview band's own center and width."""
    left, right = map(float, reference_band_px)
    if not (math.isfinite(left) and math.isfinite(right) and 0 <= left < right < size):
        raise ValueError("reference band must lie inside the bright-band ROI.")
    center = 0.5 * (left + right)
    radius = max(2.0, 0.75 * (right - left))
    return max(0, math.ceil(center - radius)), min(size, math.floor(center + radius) + 1)


def _peak_basins(values: np.ndarray, noise: float) -> list[tuple[int, int, int, int]]:
    """Significant local maxima and the valley walls belonging to each one.

    Prominence is evaluated against both sides before constructing any ROI; a low
    global threshold must never merge two SEM boundary peaks.
    """
    size = len(values)
    if size < 5:
        return []
    span = float(np.ptp(values))
    prominence_floor = max(3.0 * noise, 0.08 * span, 1e-12)
    changes = np.flatnonzero(np.diff(values) != 0) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, size]
    peaks = []
    for start, stop in zip(starts[1:-1], stops[1:-1]):
        top = float(values[start])
        if top <= values[start - 1] or top <= values[stop]:
            continue
        higher_left = np.flatnonzero(values[:start] > top)
        higher_right = np.flatnonzero(values[stop:] > top)
        lo = int(higher_left[-1]) + 1 if higher_left.size else 0
        hi = stop + int(higher_right[0]) if higher_right.size else size
        left_floor = float(np.min(values[lo:start]))
        right_floor = float(np.min(values[stop:hi]))
        if top - max(left_floor, right_floor) > prominence_floor:
            peaks.append((int(start), int(stop)))
    if not peaks:
        return []
    walls = []
    floor_tolerance = max(2.0 * noise, 0.01 * span, 1e-12)
    for (_, previous_stop), (next_start, _) in zip(peaks, peaks[1:]):
        gap = values[previous_stop:next_start]
        minimum = float(np.min(gap))
        low = np.flatnonzero(gap <= minimum + floor_tolerance)
        walls.append(previous_stop + int(round(0.5 * (low[0] + low[-1]))))
    return [
        (start, stop, 0 if index == 0 else walls[index - 1],
         size - 1 if index == len(peaks) - 1 else walls[index])
        for index, (start, stop) in enumerate(peaks)
    ]


def _polynomial_flank_crossing(
    values: np.ndarray,
    a: int,
    b: int,
    level: float,
    direction: int,
    degree: int,
) -> float | None:
    """ERA polynomial half-height crossing on one flank ``[a, b)``.

    ``direction`` is +1 (rising) or -1 (falling).  The difference-weighted
    centroid centers a narrow local window (±3/+4 px) for the polynomial fit,
    and the root whose slope matches ``direction`` nearest that centroid is the
    crossing of the ``level`` (the flank's own local-baseline half height).
    """
    if b - a <= degree:
        return None
    differences = np.maximum(direction * np.diff(values[a:b]), 0.0)
    if not len(differences) or float(np.max(differences)) <= 1e-12:
        return None
    weights = (differences / float(np.max(differences))) ** DIFFERENCE_POWER
    x_centers = np.arange(a, b - 1, dtype=float) + 0.5
    center = float(np.dot(weights, x_centers) / np.sum(weights))
    fit_start = max(a, int(math.floor(center)) - 3)
    fit_stop = min(b, int(math.ceil(center)) + 4)
    if fit_stop - fit_start <= degree:
        return None
    x = np.arange(fit_start, fit_stop, dtype=float) - center
    y = values[fit_start:fit_stop] - level
    if float(np.ptp(y)) <= 1e-12:
        return None
    coefficients = np.polyfit(x, y, degree)
    derivative = np.polyder(coefficients)
    candidates = []
    for root in np.roots(coefficients):
        if abs(float(root.imag)) > 1e-7:
            continue
        r = float(root.real)
        position = r + center
        slope = float(np.polyval(derivative, r))
        if fit_start <= position <= fit_stop - 1 and slope * direction > 0:
            candidates.append(position)
    if not candidates:
        return None
    return min(candidates, key=lambda position: abs(position - center))


def _basin_contrast(
    values: np.ndarray,
    start: int,
    stop: int,
    left_wall: int,
    right_wall: int,
    noise: float,
    span: float,
) -> tuple[float, float, list[float]] | None:
    """Local half-height floors and contrast for one basin, or None if a flank is empty.

    ``values[start:stop]`` is the flat/heavy top of one base peak; its two local
    dark floors are the median near-minimum inside each neighbouring valley.
    Returns ``(contrast, top, floors)`` where ``contrast = top - max(floors)``.
    """
    if left_wall >= start or right_wall <= stop - 1:
        return None
    top = float(np.max(values[start:stop]))
    tolerance = max(2.0 * noise, 0.01 * span, 1e-12)
    floors: list[float] = []
    for side in (values[left_wall:start], values[stop:right_wall + 1]):
        if not len(side):
            return None
        low = float(np.min(side))
        floors.append(float(np.median(side[side <= low + tolerance])))
    return top - max(floors), top, floors


def _polynomial_bright_band_crossings(
    values: np.ndarray,
    reference_local: tuple[float, float],
    degree: int,
) -> tuple[float | None, float | None, float | None, str]:
    """Return ``(left_half, right_half, contrast, reason)`` for one bright band.

    On failure the three numeric results are ``None`` and ``reason`` names the
    cause (``missing_flank`` or ``ambiguous_peak``).  Selects the preview peak
    inside the reference band, uses its own two local dark floors for the
    half-height targets, refines each flank with a ``degree`` polynomial, and
    applies the periodic center/width identity policy.

    When several eligible base peaks sit at nearly equal distance from the
    reference center (``AMBIGUITY_DISTANCE_TOLERANCE_PX``) with comparable
    contrast (``AMBIGUITY_CONTRAST_RATIO``), the preview reference does not
    identify a unique target, so the row is rejected as ``ambiguous_peak``
    instead of silently keeping the geometrically first peak.
    """
    values = np.asarray(values, dtype=float)
    width = len(values)
    if width < 5 or not np.all(np.isfinite(values)):
        return None, None, None, REASON_MISSING_FLANK
    try:
        search_left, search_right = _bright_band_search_bounds(width, reference_local)
    except ValueError:
        return None, None, None, REASON_MISSING_FLANK
    reference_center = 0.5 * (reference_local[0] + reference_local[1])
    reference_width = reference_local[1] - reference_local[0]
    endpoint_count = max(1, min(3, width // 5))
    noise = _flank_noise(values, endpoint_count)
    basins = _peak_basins(values, noise)
    eligible = [b for b in basins if b[0] < search_right and b[1] > search_left]
    if not eligible:
        return None, None, None, REASON_MISSING_FLANK
    basin = min(eligible, key=lambda b: abs(0.5 * (b[0] + b[1] - 1) - reference_center))
    start, stop, left_wall, right_wall = basin
    if start <= left_wall or stop - 1 >= right_wall:
        return None, None, None, REASON_MISSING_FLANK
    span = float(np.ptp(values))
    cont = _basin_contrast(values, start, stop, left_wall, right_wall, noise, span)
    if cont is None:
        return None, None, None, REASON_MISSING_FLANK
    contrast, top, floors = cont
    if contrast <= max(3.0 * noise, 1e-12):
        return None, None, None, REASON_MISSING_FLANK
    # Peak-identity ambiguity: keep this row only when no rival eligible base
    # peak is both nearly equidistant from the reference center and comparably
    # strong.  Otherwise the reference is symmetric around two equal bands and
    # the target identity is genuinely ambiguous.
    center = 0.5 * (start + stop - 1)
    best_distance = abs(center - reference_center)
    for rival in eligible:
        if rival == basin:
            continue
        r_start, r_stop, r_left_wall, r_right_wall = rival
        if r_start <= r_left_wall or r_stop - 1 >= r_right_wall:
            continue
        r_cont = _basin_contrast(values, r_start, r_stop, r_left_wall, r_right_wall, noise, span)
        if r_cont is None:
            continue
        r_contrast = r_cont[0]
        r_distance = abs(0.5 * (r_start + r_stop - 1) - reference_center)
        if abs(r_distance - best_distance) <= AMBIGUITY_DISTANCE_TOLERANCE_PX \
                and r_contrast >= AMBIGUITY_CONTRAST_RATIO * contrast:
            return None, None, None, REASON_AMBIGUOUS
    left_level = floors[0] + 0.5 * (top - floors[0])
    right_level = floors[1] + 0.5 * (top - floors[1])
    left = _polynomial_flank_crossing(values, left_wall, start + 1, left_level, +1, degree)
    right = _polynomial_flank_crossing(values, stop - 1, right_wall + 1, right_level, -1, degree)
    if left is None or right is None:
        return None, None, None, REASON_MISSING_FLANK
    if not 0 < left < right < width - 1:
        return None, None, None, REASON_MISSING_FLANK
    if abs(0.5 * (left + right) - reference_center) > max(2.0, 0.75 * reference_width):
        return None, None, None, REASON_MISSING_FLANK
    if right - left > max(3.0 * reference_width, reference_width + 4.0):
        return None, None, None, REASON_MISSING_FLANK
    return float(left), float(right), contrast, REASON_NONE


def bright_band_crossings(
    profile: np.ndarray,
    reference_local: tuple[float, float],
    polynomial_degree: int = DEFAULT_BRIGHT_BAND_DEGREE,
) -> tuple[float, float, float] | None:
    """One bright-band row's ``(left_half, right_half, contrast)`` or None.

    The two crossings are the ERA polynomial local-baseline half heights; the
    center is their arithmetic midpoint.  A symmetric reference over two equal
    bands (ambiguous peak identity) and any other failure both return None.
    """
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or polynomial_degree not in BRIGHT_BAND_DEGREES:
        return None
    left, right, contrast, _reason = _polynomial_bright_band_crossings(
        values, reference_local, polynomial_degree
    )
    if left is None:
        return None
    return float(left), float(right), contrast


def bright_band_row_position(
    profile: np.ndarray,
    definition: str,
    reference_local: tuple[float, float],
    position_function=None,
    crossing_function=None,
    polynomial_degree: int = DEFAULT_BRIGHT_BAND_DEGREE,
) -> tuple[float, str]:
    """ERA-style per-row bright-band polynomial local half height.

    ``definition`` is one of ``"left"``, ``"right"`` or ``"center"`` with
    independent meanings: left/right are the two polynomial half-height
    crossings, center their arithmetic midpoint.  ``position_function`` and
    ``crossing_function`` are accepted for backward compatibility but are not
    used: this module owns its polynomial localizer and never delegates to the
    discovery module's linear half-height interpolation.  On failure the reason
    is the cause (``missing_flank`` or ``ambiguous_peak``) rather than a generic
    placeholder.
    """
    if definition not in ("center", "left", "right"):
        return _nan(), REASON_AMBIGUOUS
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or polynomial_degree not in BRIGHT_BAND_DEGREES:
        return _nan(), REASON_MISSING_FLANK
    left, right, _, reason = _polynomial_bright_band_crossings(
        values, reference_local, polynomial_degree
    )
    if left is None:
        return _nan(), reason
    position = left if definition == "left" else right if definition == "right" else 0.5 * (left + right)
    return float(position), REASON_NONE


# ---------------------------------------------------------------------------
# Diagnostics containers
# ---------------------------------------------------------------------------

@dataclass
class RowLocalization:
    """Per-row localization record (local coordinates inside the ROI)."""

    row_px: int
    candidate_px: float
    final_px: float
    stage: str
    reason: str
    recovered_from_px: float = float("nan")
    # Bright-band auxiliary crossings (local x). NaN for transition rows.
    band_left_px: float = float("nan")
    band_right_px: float = float("nan")
    band_center_px: float = float("nan")


@dataclass
class EdgeLocalization:
    """One edge's full row-level localization, including the recovery pass."""

    kind: str
    roi_bounds_px: tuple[int, int, int, int]
    height: int
    reference_x_px: float = float("nan")
    reference_band_local: tuple[float, float] | None = None
    fit_degree: int = 0
    position_definition: str = "center"
    polarity: str = ""
    contrast_threshold: float = float("nan")
    max_jump_px: float = 12.0
    rows: list[RowLocalization] = field(default_factory=list)

    # Derived views -----------------------------------------------------
    @property
    def candidates_px(self) -> np.ndarray:
        return np.asarray([row.candidate_px for row in self.rows], dtype=float)

    @property
    def positions_px(self) -> np.ndarray:
        return np.asarray([row.final_px for row in self.rows], dtype=float)

    @property
    def valid(self) -> np.ndarray:
        return np.asarray([row.stage != STAGE_MISSING for row in self.rows], dtype=bool)

    @property
    def stages(self) -> list[str]:
        return [row.stage for row in self.rows]

    @property
    def reasons(self) -> list[str]:
        return [row.reason for row in self.rows]

    def row_records(self) -> list[RowLocalization]:
        return list(self.rows)

    def valid_local_rows(self) -> np.ndarray:
        return np.asarray([row.row_px for row in self.rows if row.stage != STAGE_MISSING], dtype=float)

    def valid_positions(self) -> np.ndarray:
        return np.asarray([row.final_px for row in self.rows if row.stage != STAGE_MISSING], dtype=float)


@dataclass
class LocalizationStats:
    """Missing-row statistics for one edge (row counts are local to the ROI)."""

    total_rows: int = 0
    primary_rows: int = 0
    recovered_rows: int = 0
    missing_rows: int = 0
    coverage_fraction: float = float("nan")
    longest_gap_rows: int = 0
    gap_runs: list[tuple[int, int]] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)


def localization_stats(diag: EdgeLocalization) -> LocalizationStats:
    stats = LocalizationStats()
    stats.total_rows = diag.height
    stats.primary_rows = sum(1 for row in diag.rows if row.stage == STAGE_PRIMARY)
    stats.recovered_rows = sum(1 for row in diag.rows if row.stage == STAGE_RELOCALIZED)
    stats.missing_rows = sum(1 for row in diag.rows if row.stage == STAGE_MISSING)
    stats.coverage_fraction = (
        (stats.primary_rows + stats.recovered_rows) / diag.height if diag.height else float("nan")
    )
    reasons: dict[str, int] = {}
    for row in diag.rows:
        if row.stage == STAGE_MISSING and row.reason:
            reasons[row.reason] = reasons.get(row.reason, 0) + 1
    stats.reasons = reasons
    stats.gap_runs = gap_runs(diag.valid)
    stats.longest_gap_rows = max([0, *[end - start + 1 for start, end in stats.gap_runs]])
    return stats


@dataclass
class PairLocalizationStats:
    """Common-row pairing statistics between a left and a right edge."""

    left_total_rows: int = 0
    right_total_rows: int = 0
    left_valid_rows: int = 0
    right_valid_rows: int = 0
    overlap_rows: int = 0
    common_valid_rows: int = 0
    coverage_fraction: float = float("nan")
    longest_common_gap_rows: int = 0


def pair_localization_stats(left_diag: EdgeLocalization | None, right_diag: EdgeLocalization | None) -> PairLocalizationStats:
    stats = PairLocalizationStats()
    if left_diag is not None:
        stats.left_total_rows = left_diag.height
        stats.left_valid_rows = int(left_diag.valid.sum())
    if right_diag is not None:
        stats.right_total_rows = right_diag.height
        stats.right_valid_rows = int(right_diag.valid.sum())
    if left_diag is None or right_diag is None:
        return stats
    left_origin = left_diag.roi_bounds_px[1]
    right_origin = right_diag.roi_bounds_px[1]
    left_rows = set(int(row) + left_origin for row in left_diag.valid_local_rows())
    right_rows = set(int(row) + right_origin for row in right_diag.valid_local_rows())
    common = left_rows & right_rows
    stats.common_valid_rows = len(common)
    overlap_top = max(left_origin, right_origin)
    overlap_bottom = min(left_origin + left_diag.height, right_origin + right_diag.height)
    stats.overlap_rows = max(0, overlap_bottom - overlap_top)
    if stats.overlap_rows:
        stats.coverage_fraction = len(common) / stats.overlap_rows
        longest_gap = 0
        run = 0
        for row in range(overlap_top, overlap_bottom):
            if row in common:
                run = 0
            else:
                run += 1
                longest_gap = max(longest_gap, run)
        stats.longest_common_gap_rows = longest_gap
    return stats


def failure_reason_summary(stats: LocalizationStats | None, pair: PairLocalizationStats | None = None) -> str:
    """One compact Chinese summary line for reports and CSV."""
    if stats is None:
        return ""
    parts = [
        f"总行 {stats.total_rows}",
        f"首轮 {stats.primary_rows}",
        f"恢复 {stats.recovered_rows}",
        f"失败 {stats.missing_rows}",
        f"覆盖率 {stats.coverage_fraction:.1%}" if math.isfinite(stats.coverage_fraction) else "覆盖率 -",
        f"最长缺口 {stats.longest_gap_rows} 行",
    ]
    for reason, count in sorted(stats.reasons.items()):
        parts.append(f"{reason}={count}")
    if pair is not None:
        parts.append(f"共同有效 {pair.common_valid_rows}/{pair.overlap_rows}")
    return "；".join(parts)


# ---------------------------------------------------------------------------
# Unified per-edge drivers
# ---------------------------------------------------------------------------

def _rolling_median(trace: np.ndarray, window: int = 9) -> np.ndarray:
    radius = window // 2
    padded = np.pad(trace, (radius, radius), mode="edge")
    return np.array([np.median(padded[index : index + window]) for index in range(len(trace))])


def _outlier_gate(trace: np.ndarray, max_jump_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Historical jump gate: observed values within ``max_jump_px`` of the local
    9-row median of the filled trace.  The filled trace is only used for the
    gate and never becomes an observation."""
    observed = np.isfinite(trace)
    if not observed.any():
        return np.zeros(len(trace)), np.zeros(len(trace), dtype=bool)
    indices = np.arange(len(trace))
    filled = np.interp(indices, indices[observed], trace[observed])
    baseline = _rolling_median(filled, window=9)
    valid = observed & (np.abs(filled - baseline) <= max_jump_px)
    return baseline, valid


def _neighbor_guided_seed(
    diag_rows: list[RowLocalization],
    index: int,
    valid_positions: dict[int, float],
) -> float | None:
    """Median of the nearest valid positions on each side inside the recovery
    window; requires RECOVERY_MIN_NEIGHBORS valid neighbours total."""
    neighbors = []
    for offset in range(1, RECOVERY_WINDOW_ROWS + 1):
        if len(neighbors) >= RECOVERY_MIN_NEIGHBORS and (
            (index - offset < 0 or index - offset not in valid_positions)
            and (index + offset >= len(diag_rows) or index + offset not in valid_positions)
        ):
            break
        below, above = index - offset, index + offset
        if below >= 0 and below in valid_positions:
            neighbors.append(valid_positions[below])
        if len(neighbors) >= RECOVERY_MIN_NEIGHBORS:
            break
        if above < len(diag_rows) and above in valid_positions:
            neighbors.append(valid_positions[above])
        if len(neighbors) >= RECOVERY_MIN_NEIGHBORS:
            break
    if len(neighbors) < RECOVERY_MIN_NEIGHBORS:
        return None
    return float(np.median(neighbors))


def localize_transition_edge(
    region: np.ndarray,
    roi_bounds_px: tuple[int, int, int, int],
    reference_x: float,
    polarity_sign: int,
    polynomial_degree: int,
    polarity_label: str,
    max_jump_px: float = 12.0,
    row_mask: np.ndarray | None = None,
    relocalize: bool = True,
) -> EdgeLocalization:
    """Unified transition-edge localization with per-row diagnostics.

    ``row_mask`` is a selection-only mask (automatic discovery contrast policy);
    masked rows are reported as low-contrast and are never primary candidates and
    never retried by the recovery pass.
    """
    numeric = np.asarray(region, dtype=float)
    if numeric.ndim != 2:
        raise ValueError("region must be a 2-D grayscale array.")
    height, width = numeric.shape
    if row_mask is None:
        row_mask = np.ones(height, dtype=bool)
    else:
        row_mask = np.asarray(row_mask, dtype=bool)
        if row_mask.shape != (height,):
            raise ValueError("row_mask must have one boolean value per ROI row.")
    diag = EdgeLocalization(
        kind="transition",
        roi_bounds_px=roi_bounds_px,
        height=height,
        reference_x_px=reference_x,
        fit_degree=polynomial_degree,
        polarity=polarity_label,
        max_jump_px=max_jump_px,
    )
    rows: list[RowLocalization] = []
    for row_index in range(height):
        if not row_mask[row_index]:
            rows.append(RowLocalization(row_index, _nan(), _nan(), STAGE_MISSING, REASON_LOW_CONTRAST))
            continue
        candidate, reason = transition_row_position(
            numeric[row_index], reference_x, polarity_sign, polynomial_degree
        )
        rows.append(RowLocalization(
            row_index, candidate, candidate,
            STAGE_PRIMARY if reason == REASON_NONE else STAGE_MISSING, reason,
        ))
    diag.rows = rows
    if relocalize:
        _relocalize_transition_rows(diag, numeric, polarity_sign, polynomial_degree)
    else:
        baseline, valid = _outlier_gate(diag.positions_px, max_jump_px)
        for row_index, row in enumerate(diag.rows):
            if row.stage != STAGE_MISSING and not valid[row_index]:
                row.stage = STAGE_MISSING
                row.final_px = _nan()
                row.reason = REASON_JUMP_REJECTED
    return diag


def _relocalize_transition_rows(
    diag: EdgeLocalization,
    region: np.ndarray,
    polarity_sign: int,
    polynomial_degree: int,
) -> None:
    """One bounded neighbor-guided re-read of real pixels for failed rows.

    Externally masked / low-contrast rows and jump-rejected rows are never
    retried.  The retry re-anchors the *same* automatic search window at the
    neighbor median and re-runs the identical contrast / polynomial fit; it does
    not use a tighter window, so a genuine large fluctuation is still found and
    then correctly rejected by the same jump gate instead of being replaced by
    the neighbor median.
    """
    max_jump_px = diag.max_jump_px
    primary_trace = np.asarray(
        [row.final_px if row.stage == STAGE_PRIMARY else _nan() for row in diag.rows], dtype=float
    )
    baseline, valid = _outlier_gate(primary_trace, max_jump_px)
    for index, row in enumerate(diag.rows):
        if row.stage == STAGE_PRIMARY and not valid[index]:
            row.stage = STAGE_MISSING
            row.reason = REASON_JUMP_REJECTED
            row.final_px = _nan()
    # Frozen seed set: recovered rows do not expand the search.
    valid_positions = {row.row_px: row.final_px for row in diag.rows if row.stage != STAGE_MISSING}
    for index in range(diag.height):
        row = diag.rows[index]
        if row.stage != STAGE_MISSING:
            continue
        if row.reason in (REASON_LOW_CONTRAST, REASON_JUMP_REJECTED):
            continue
        seed = _neighbor_guided_seed(diag.rows, index, valid_positions)
        if seed is None or not math.isfinite(seed):
            row.reason = row.reason or REASON_UNRESOLVED
            continue
        retry, retry_reason = transition_row_position(
            region[index], seed, polarity_sign, polynomial_degree
        )
        if retry_reason != REASON_NONE or not math.isfinite(retry):
            row.reason = row.reason or REASON_UNRESOLVED
            continue
        if abs(retry - baseline[index]) > max_jump_px:
            row.reason = REASON_JUMP_REJECTED
            continue
        row.final_px = retry
        row.stage = STAGE_RELOCALIZED
        row.reason = REASON_NONE
        row.recovered_from_px = seed


def localize_bright_band_edge(
    region: np.ndarray,
    roi_bounds_px: tuple[int, int, int, int],
    definition: str,
    reference_local: tuple[float, float],
    position_function=None,
    crossing_function=None,
    search_bounds_function=None,
    max_jump_px: float = 12.0,
    contrast_threshold: float | None = None,
    relocalize: bool = True,
    polynomial_degree: int = DEFAULT_BRIGHT_BAND_DEGREE,
) -> EdgeLocalization:
    """Unified bright-band localization with per-row diagnostics.

    The position is the ERA polynomial local-baseline half height for the chosen
    ``definition``.  ``position_function`` / ``crossing_function`` /
    ``search_bounds_function`` are accepted for backward compatibility only; the
    polynomial localizer is self-contained and they are not used.  Rows below
    the per-row contrast gate (default: 20 % of the ROI's 75th percentile row
    contrast) are low-contrast and never retried.
    """
    if definition not in ("center", "left", "right"):
        raise ValueError("definition must be exactly 'center', 'left' or 'right'.")
    if polynomial_degree not in BRIGHT_BAND_DEGREES:
        raise ValueError("统一拟合仅支持 2、3 或 4 次多项式。")
    numeric = np.asarray(region, dtype=float)
    if numeric.ndim != 2:
        raise ValueError("region must be a 2-D grayscale array.")
    height, width = numeric.shape
    try:
        search_start, search_stop = _bright_band_search_bounds(width, reference_local)
    except ValueError as exc:
        raise ValueError("亮带 ROI 未包含有效的预览亮边；请重新预览或调整选区。") from exc
    flank = max(1, min(3, width // 5))
    backgrounds = np.maximum(np.median(numeric[:, :flank], axis=1), np.median(numeric[:, -flank:], axis=1))
    contrasts = np.max(numeric[:, search_start:search_stop], axis=1) - backgrounds
    if contrast_threshold is None:
        contrast_threshold = max(1e-9, 0.2 * float(np.percentile(contrasts, 75)))
    diag = EdgeLocalization(
        kind="bright_band",
        roi_bounds_px=roi_bounds_px,
        height=height,
        reference_band_local=reference_local,
        fit_degree=polynomial_degree,
        position_definition=definition,
        polarity=f"bright_band_{definition}",
        contrast_threshold=contrast_threshold,
        max_jump_px=max_jump_px,
    )
    rows: list[RowLocalization] = []
    for row_index, (profile, contrast) in enumerate(zip(numeric, contrasts)):
        if contrast <= contrast_threshold:
            rows.append(RowLocalization(row_index, _nan(), _nan(), STAGE_MISSING, REASON_LOW_CONTRAST))
            continue
        left, right, _, reason = _polynomial_bright_band_crossings(
            profile, reference_local, polynomial_degree
        )
        if left is None:
            rows.append(RowLocalization(row_index, _nan(), _nan(), STAGE_MISSING, reason))
            continue
        position = left if definition == "left" else right if definition == "right" else 0.5 * (left + right)
        rows.append(RowLocalization(
            row_index, position, position, STAGE_PRIMARY, REASON_NONE,
            band_left_px=left, band_right_px=right, band_center_px=0.5 * (left + right),
        ))
    diag.rows = rows
    if relocalize:
        _relocalize_bright_band_rows(diag, numeric, contrasts, contrast_threshold, polynomial_degree)
    else:
        baseline, valid = _outlier_gate(diag.positions_px, max_jump_px)
        for row_index, row in enumerate(diag.rows):
            if row.stage != STAGE_MISSING and not valid[row_index]:
                row.stage = STAGE_MISSING
                row.final_px = _nan()
                row.reason = REASON_JUMP_REJECTED
    return diag


def _relocalize_bright_band_rows(
    diag: EdgeLocalization,
    region: np.ndarray,
    contrasts: np.ndarray,
    contrast_threshold: float,
    polynomial_degree: int,
) -> None:
    """One bounded neighbor-guided re-read for failed bright-band rows.

    The seed is the neighbor-verified band *center* (independent of the chosen
    left/right/center definition), and the retry re-anchors the reference band
    around that center inside the ROI with the same identity policy.  Recovered
    rows are never used as seeds (no chaining), and the same contrast / jump
    gates as the primary pass apply.
    """
    max_jump_px = diag.max_jump_px
    primary_trace = np.asarray(
        [row.final_px if row.stage == STAGE_PRIMARY else _nan() for row in diag.rows], dtype=float
    )
    baseline, valid = _outlier_gate(primary_trace, max_jump_px)
    for index, row in enumerate(diag.rows):
        if row.stage == STAGE_PRIMARY and not valid[index]:
            row.stage = STAGE_MISSING
            row.reason = REASON_JUMP_REJECTED
            row.final_px = _nan()
    reference_local = diag.reference_band_local
    if reference_local is None:
        return
    reference_width = max(1.0, float(reference_local[1] - reference_local[0]))
    # Frozen seed set of verified band CENTERS (definition-independent).
    center_positions = {
        row.row_px: row.band_center_px
        for row in diag.rows
        if row.stage != STAGE_MISSING and math.isfinite(row.band_center_px)
    }
    for index in range(diag.height):
        row = diag.rows[index]
        if row.stage != STAGE_MISSING:
            continue
        if row.reason in (REASON_LOW_CONTRAST, REASON_JUMP_REJECTED):
            continue
        if contrasts[index] <= contrast_threshold:
            row.reason = REASON_LOW_CONTRAST
            continue
        seed = _neighbor_guided_seed(diag.rows, index, center_positions)
        if seed is None or not math.isfinite(seed):
            row.reason = row.reason or REASON_UNRESOLVED
            continue
        local_left = max(0.0, seed - reference_width / 2.0)
        local_right = min(float(region.shape[1]), seed + reference_width / 2.0)
        if local_right - local_left < 1.0:
            row.reason = row.reason or REASON_UNRESOLVED
            continue
        crossings = _polynomial_bright_band_crossings(region[index], (local_left, local_right), polynomial_degree)
        left, right, _, retry_reason = crossings
        if left is None:
            # The retry uses the same ambiguity policy as the primary pass; a
            # genuine ambiguous re-read keeps the more informative reason.
            row.reason = retry_reason if retry_reason == REASON_AMBIGUOUS else (row.reason or REASON_UNRESOLVED)
            continue
        definition = diag.position_definition
        retry = left if definition == "left" else right if definition == "right" else 0.5 * (left + right)
        if abs(retry - baseline[index]) > max_jump_px:
            row.reason = REASON_JUMP_REJECTED
            continue
        row.final_px = retry
        row.band_left_px = left
        row.band_right_px = right
        row.band_center_px = 0.5 * (left + right)
        row.stage = STAGE_RELOCALIZED
        row.reason = REASON_NONE
        row.recovered_from_px = seed


def valid_row_threshold(height: int) -> int:
    """Historical acceptance gate: max(30, 60 % of the ROI rows)."""
    return max(MIN_EDGE_VALID_ROWS, int(math.ceil(height * MIN_EDGE_VALID_FRACTION)))


# ---------------------------------------------------------------------------
# Background correction
# ---------------------------------------------------------------------------

@dataclass
class BackgroundModel:
    """Conservative low-order two-dimensional additive background.

    ``coefficients`` are expressed in the normalized global coordinate basis:
    ``u = (column - (width-1)/2) / max((width-1)/2, 1)`` and
    ``v = (row - (height-1)/2) / max((height-1)/2, 1)``, with monomial order
    ``[1, u, v, u*v, u^2, v^2]`` for degree 2.
    """

    degree: int
    region: tuple[int, int, int, int] | None
    coefficients: np.ndarray
    background: np.ndarray
    corrected_image: np.ndarray
    fit_mean: float
    fit_std: float

    @property
    def label(self) -> str:
        degree = BACKGROUND_DEGREE_LABELS.get(self.degree, f"{self.degree} 阶")
        region = BACKGROUND_REGION_FULL if self.region is None else BACKGROUND_REGION_ROI
        return f"{degree}·{region}"


def polynomial_features(rows: np.ndarray, columns: np.ndarray, degree: int) -> np.ndarray:
    """Monomial design matrix for a two-dimensional polynomial up to degree 1 or 2.

    ``columns`` is the horizontal (x) coordinate, ``rows`` the vertical (y)
    coordinate; both are expected to already be normalized global coordinates.
    """
    if degree == 1:
        return np.column_stack((np.ones(len(rows)), columns, rows))
    if degree == 2:
        return np.column_stack((np.ones(len(rows)), columns, rows, columns * rows, columns ** 2, rows ** 2))
    raise ValueError("背景模型仅支持一阶或二阶多项式。")


def _robust_background_coefficients(features: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Tukey-biweight iteratively reweighted least squares (pure NumPy).

    Down-weights pixels whose residual from the smooth low-order model is large,
    so the background does not absorb bright/dark periodic structure the way a
    plain whole-image least-squares fit would.  Falls back to plain least
    squares when the robust fit degenerates (e.g. a noiseless exact background).
    """
    num_terms = features.shape[1]
    coefficients, *_ = np.linalg.lstsq(features, values, rcond=None)
    for _ in range(6):
        residual = values - features @ coefficients
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        sigma = 1.4826 * mad
        if sigma <= 1e-12:
            break
        r = (residual - med) / (4.685 * sigma)
        mask = np.abs(r) < 1.0
        if int(mask.sum()) < num_terms:
            break
        weights = np.where(mask, (1.0 - r * r) ** 2, 0.0)
        wf = features * weights[:, None]
        wv = values * weights
        coefficients, *_ = np.linalg.lstsq(wf, wv, rcond=None)
    return coefficients


# Deterministic fit-grid sub-sampling budget: the fit is keyed on at most this
# many points so large working images never build a full-pixel design matrix.
# NOTE: this is a point *budget*, not a claim that sub-sampling leaves results
# identical.  On noisy input, sub-sampling decimates the effective support and
# can change the fitted polynomial coefficients; the first/last index of each
# axis stays anchored so the extremes and the global (normalized) coordinate
# mapping are preserved, but the resulting fit is an approximation of the exact
# full-resolution fit.  See `_background_grid_indices` for the exact budget.
BACKGROUND_FIT_MAX_SAMPLES = 65536
BACKGROUND_TERMS = {1: (0, 0), 2: (0, 0)}
# monomial exponent pairs in polynomial_features order:
#   degree 1: [1, u, v]           -> (0,0) (1,0) (0,1)
#   degree 2: [1, u, v, u*v, u², v²] -> (0,0) (1,0) (0,1) (1,1) (2,0) (0,2)
_BACKGROUND_MONOMIAL_EXPONENTS = {
    1: ((0, 0), (1, 0), (0, 1)),
    2: ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)),
}


def _background_grid_indices(
    height: int, width: int, region: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic whole-region grid row/column indices for the background fit.

    At most ``BACKGROUND_FIT_MAX_SAMPLES`` points are used.  The grid is spread
    uniformly across ``region``, and the first and last index of each axis is
    always included so the polynomial's extremes are anchored, while the global
    (normalized) coordinate mapping is preserved exactly.  The same grid is used
    every call, so the fit is deterministic.  The returned point count is an
    approximation of the full-resolution support (not a lossless reduction):
    decimating a noisy field can change the fitted coefficients.
    """
    left, top, right, bottom = region
    region_height = bottom - top
    region_width = right - left
    count = region_height * region_width
    if count <= BACKGROUND_FIT_MAX_SAMPLES:
        rows, cols = np.indices((region_height, region_width))
        return rows.ravel(), cols.ravel()
    # Choose a uniform stride whose rectangular grid stays strictly within the
    # budget.  Start from ceil(sqrt(count / budget)) and grow stride until the
    # resulting (row_positions × col_positions) product is at most the budget;
    # each axis keeps both its first and last index anchored.
    stride = int(math.ceil(math.sqrt(count / BACKGROUND_FIT_MAX_SAMPLES)))
    while True:
        row_positions = np.unique(
            np.concatenate((np.arange(0, region_height, stride), [region_height - 1]))
        )
        col_positions = np.unique(
            np.concatenate((np.arange(0, region_width, stride), [region_width - 1]))
        )
        if len(row_positions) * len(col_positions) <= BACKGROUND_FIT_MAX_SAMPLES:
            break
        stride += 1
    rows, cols = np.meshgrid(row_positions, col_positions, indexing="ij")
    return rows.ravel(), cols.ravel()


def _evaluate_background_polynomial(
    u: np.ndarray, v: np.ndarray, coefficients: np.ndarray, degree: int
) -> np.ndarray:
    """Broadcast per-term polynomial evaluation (never builds an N×k matrix).

    ``u`` and ``v`` are normalized global coordinate grids of matching shape;
    each monomial is computed element-wise and summed with its coefficient, so
    the full-image background is produced in O(pixels) memory instead of
    allocating a full-pixel design matrix.
    """
    background = coefficients[0] * np.ones_like(u)
    for term_index, (uexp, vexp) in enumerate(_BACKGROUND_MONOMIAL_EXPONENTS[degree][1:], start=1):
        background = background + coefficients[term_index] * (u ** uexp) * (v ** vexp)
    return background


def fit_additive_background(
    image: np.ndarray,
    degree: int = 1,
    region: tuple[int, int, int, int] | None = None,
) -> BackgroundModel:
    """Fit a low-order additive background from the original working pixels.

    The fit and the full-image evaluation share one numerically stable
    *normalized global* coordinate basis (never the ROI-local coordinates), so a
    non-zero-offset ROI is handled correctly.  The fit is robust against bright
    or dark structure, the background is evaluated over the full image, and the
    regional mean is restored so the corrected image keeps its intensity level.
    Corrections are never accumulated: the caller must always pass the
    uncorrected image.

    Memory: the fit runs on a deterministic uniform grid of at most
    ``BACKGROUND_FIT_MAX_SAMPLES`` points (first/last index of each axis
    anchored), and the full-image background is produced by broadcast per-term
    polynomial evaluation rather than a full-pixel design matrix, so large
    working images stay bounded.  Sub-sampling is an approximation: on noisy
    input the decimated support can change the fitted coefficients relative to
    an exact full-resolution fit.
    """
    if degree not in (1, 2):
        raise ValueError("背景模型仅支持一阶或二阶多项式。")
    numeric = np.asarray(image, dtype=float)
    if numeric.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array.")
    if not np.all(np.isfinite(numeric)):
        raise ValueError("image must not contain NaN or infinite values.")
    height, width = numeric.shape
    if region is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        left, top, right, bottom = region
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("背景拟合区域超出图像范围。")
    sample = numeric[top:bottom, left:right]
    min_terms = (degree + 1) * (degree + 2) // 2
    if sample.size < min_terms:
        raise ValueError("背景拟合区域像素过少。")
    # Normalized global coordinate basis (consistent between fit and evaluation).
    row_center = (height - 1) / 2.0
    col_center = (width - 1) / 2.0
    row_scale = max((height - 1) / 2.0, 1.0)
    col_scale = max((width - 1) / 2.0, 1.0)
    sample_rows, sample_columns = _background_grid_indices(
        height, width, (left, top, right, bottom)
    )
    sample_rows_norm = (sample_rows.astype(float) + top - row_center) / row_scale
    sample_columns_norm = (sample_columns.astype(float) + left - col_center) / col_scale
    features = polynomial_features(sample_rows_norm, sample_columns_norm, degree)
    values = sample[sample_rows, sample_columns].astype(float)
    coefficients = _robust_background_coefficients(features, values)
    full_v, full_u = np.indices(numeric.shape)
    full_v = (full_v.astype(float) - row_center) / row_scale
    full_u = (full_u.astype(float) - col_center) / col_scale
    background = _evaluate_background_polynomial(full_u, full_v, coefficients, degree)
    fit_mean = float(np.mean(background[top:bottom, left:right]))
    fit_std = float(np.std(background[top:bottom, left:right]))
    corrected = numeric - (background - fit_mean)
    return BackgroundModel(degree, region, coefficients, background, corrected, fit_mean, fit_std)


# ---------------------------------------------------------------------------
# Row-index helpers shared by the curve review UI and CSV
# ---------------------------------------------------------------------------

def missing_row_indices(diag: EdgeLocalization) -> list[int]:
    """Local row indices of every missing row, in order."""
    return [row.row_px for row in diag.rows if row.stage == STAGE_MISSING]


def gap_runs(valid: Iterable[bool]) -> list[tuple[int, int]]:
    """Consecutive missing-row runs as (start, end) local indices (inclusive)."""
    runs: list[tuple[int, int]] = []
    start = None
    last_index = -1
    for index, ok in enumerate(valid):
        last_index = index
        if not ok and start is None:
            start = index
        elif ok and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, last_index))
    return runs
