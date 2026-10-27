"""Automatic near-vertical SEM line discovery for the opt-in ERA workflow.

The module proposes one pair of single-edge ROI rectangles per complete stripe.
It is discovery-only: it never mutates the supplied image, never computes
roughness, and adds no spatial smoothing.  Each returned ROI is intended to be
passed unchanged to the existing ``analyze_single_edge_era`` measurement.

Only complete, near-vertical lines are supported.  Horizontal or branched
topology, lattice clues, expected line counts, and hand-provided diameters are
deliberately not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# A line must be present on at least this fraction of the image rows.
MIN_SUPPORT_FRACTION = 0.60
# Contrast is considered usable only when it rises above the measured row noise.
SUPPORT_NOISE_FACTOR = 3.0
SUPPORT_AMPLITUDE_FRACTION = 0.20
# Edges are measured between the 10 % and 90 % points of their local transition.
TRANSITION_START_FRACTION = 0.10
TRANSITION_END_FRACTION = 0.90
# Real line edges are compact; a smooth illumination hump is not a stripe.
MAX_TRANSITION_FRACTION = 0.25
MIN_TRANSITION_ALLOWANCE_PX = 4.0
# Two single-edge rectangles need room between the paired opposite edges.
MIN_LINE_WIDTH_FACTOR = 2

__all__ = ["AutoLineCandidate", "detect_line_candidates"]


@dataclass(frozen=True)
class AutoLineCandidate:
    """One complete near-vertical line and its two single-edge rectangles.

    All coordinates are local to the supplied 2-D image.  ROI tuples are
    half-open ``(left, top, right, bottom)`` pixel bounds.  ``center_x_px`` and
    ``width_px`` are the two edge-midpoint positions and their separation.
    """

    left_roi_bounds_px: tuple[int, int, int, int]
    right_roi_bounds_px: tuple[int, int, int, int]
    center_x_px: float
    width_px: float
    support_fraction: float


@dataclass(frozen=True)
class _Edge:
    position: float
    start: int
    end: int
    rising: bool


@dataclass(frozen=True)
class _Band:
    left: _Edge
    right: _Edge
    amplitude: float
    support_fraction: float


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return resolved


def _validate_image(
    image: np.ndarray,
    minimum_roi_width: int,
    minimum_height: int,
) -> tuple[np.ndarray, int, int]:
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a 2-D NumPy array.")
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array.")
    if image.dtype.kind not in "buif":
        raise ValueError("image must have a real numeric dtype.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    try:
        work = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("image must contain real numeric pixels.") from exc
    if not np.all(np.isfinite(work)):
        raise ValueError("image must not contain NaN or infinite values.")
    height, width = work.shape
    if height < minimum_height:
        raise ValueError("image height is smaller than minimum_height.")
    if width < 2 * minimum_roi_width + 1:
        raise ValueError("image width is too small for two minimum-width ROIs.")
    return work, height, width


def _robust_pixel_noise(image: np.ndarray) -> float:
    """Estimate per-pixel white-noise sigma from adjacent-pixel differences."""
    differences = np.abs(np.diff(image, axis=1))
    if not differences.size:
        return 0.0
    return float(np.median(differences) / (math.sqrt(2.0) * 0.6744897501960817))


def _otsu_threshold(values: np.ndarray) -> float | None:
    """Return a NumPy-only Otsu threshold, using the middle of a flat optimum."""
    low = float(np.min(values))
    high = float(np.max(values))
    if not high > low:
        return None
    scaled = (values - low) / (high - low) * 255.0
    bins = np.clip(scaled.astype(np.intp), 0, 255)
    histogram = np.bincount(bins, minlength=256).astype(np.float64)
    total = float(np.sum(histogram))
    if total <= 0.0:
        return None
    cumulative = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256, dtype=np.float64))
    total_mean = float(cumulative_mean[-1])
    weight_low = cumulative[:-1]
    weight_high = total - weight_low
    valid = (weight_low > 0.0) & (weight_high > 0.0)
    between = np.zeros(255, dtype=np.float64)
    mean_low = np.divide(
        cumulative_mean[:-1], weight_low, out=np.zeros(255), where=weight_low > 0
    )
    mean_high = np.divide(
        total_mean - cumulative_mean[:-1],
        weight_high,
        out=np.zeros(255),
        where=weight_high > 0,
    )
    between[valid] = (
        weight_low[valid] * weight_high[valid] * (mean_low[valid] - mean_high[valid]) ** 2
    )
    best = float(np.max(between))
    if best <= 0.0:
        return None
    tolerance = max(1e-12, best * 1e-9)
    plateau = np.flatnonzero(between >= best - tolerance)
    index = int(plateau[len(plateau) // 2])
    return low + (index + 0.5) * (high - low) / 255.0


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _find_crossing(
    profile: np.ndarray,
    start: int,
    stop: int,
    level: float,
    rising: bool,
) -> float | None:
    """Find the strongest profile crossing of ``level`` in a search interval."""
    start = int(max(0, start))
    stop = int(min(len(profile) - 1, stop))
    if stop <= start:
        return None
    left_values = profile[start:stop]
    right_values = profile[start + 1 : stop + 1]
    differences = right_values - left_values
    if rising:
        hit = (left_values <= level) & (right_values > level)
    else:
        hit = (left_values >= level) & (right_values < level)
    hit &= np.abs(differences) > 1e-12
    indices = np.flatnonzero(hit)
    if not len(indices):
        return None
    best = indices[int(np.argmax(np.abs(differences[indices])))]
    fraction = float((level - left_values[best]) / differences[best])
    fraction = min(max(fraction, 0.0), 1.0)
    return float(start + int(best) + fraction)


def _transition_bounds(
    profile: np.ndarray,
    edge: float,
    low: float,
    high: float,
    search: int,
    rising: bool,
) -> tuple[int, int] | None:
    """Return the background-side and line-side 10 %/90 % transition pixels."""
    amplitude = high - low
    if not amplitude > 0.0:
        return None
    low_level = low + TRANSITION_START_FRACTION * amplitude
    high_level = low + TRANSITION_END_FRACTION * amplitude
    index = int(math.floor(edge))
    search_start = max(0, index - search)
    search_stop = min(len(profile) - 1, index + search)
    segment = profile[search_start : search_stop + 1]
    low_positions = np.flatnonzero(segment <= low_level) + search_start
    high_positions = np.flatnonzero(segment >= high_level) + search_start
    if rising:
        starts = low_positions[low_positions <= index]
        ends = high_positions[high_positions >= index]
    else:
        starts = high_positions[high_positions <= index]
        ends = low_positions[low_positions >= index]
    if not len(starts) or not len(ends):
        return None
    return int(starts[-1]), int(ends[0])


def _line_profile(image: np.ndarray, bright_lines: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return row-centered signed image and its robust column median profile."""
    signed = image if bright_lines else -image
    signed = signed - np.median(signed, axis=1, keepdims=True)
    return signed, np.median(signed, axis=0)


def _band_edges(
    profile: np.ndarray,
    run: tuple[int, int],
    previous_stop: int,
    next_start: int,
    minimum_roi_width: int,
    noise: float,
) -> tuple[list[_Edge], _Band | None]:
    """Measure the edge pair of one mask run and any detectable border edge."""
    start, stop = run
    width = stop - start
    # Even unmeasurably narrow bands must bound neighboring single-edge ROIs.
    background_span = max(6, int(round(1.5 * width)))
    left_window = max(previous_stop, start - background_span)
    right_window = min(next_start, stop + background_span)
    quarter = max(1, width // 4)
    if stop - quarter > start + quarter:
        high = float(np.median(profile[start + quarter : stop - quarter]))
    else:
        high = float(np.median(profile[start:stop]))
    low_left = (
        float(np.median(profile[left_window:start])) if start > left_window else None
    )
    low_right = (
        float(np.median(profile[stop:right_window])) if right_window > stop else None
    )

    edges: list[_Edge] = []
    left_edge = None
    right_edge = None
    if start > 0 and low_left is not None:
        amplitude = high - low_left
        if amplitude > max(SUPPORT_NOISE_FACTOR * noise, 1e-9):
            level = 0.5 * (high + low_left)
            position = _find_crossing(
                profile,
                max(left_window, start - 2 * width),
                min(stop, start + 2 * width),
                level,
                rising=True,
            )
            if position is not None:
                bounds = _transition_bounds(
                    profile, position, low_left, high, max(4, width), rising=True
                )
                if bounds is not None:
                    left_edge = _Edge(position, bounds[0], bounds[1], True)
                    edges.append(left_edge)
    if stop < len(profile) and low_right is not None:
        amplitude = high - low_right
        if amplitude > max(SUPPORT_NOISE_FACTOR * noise, 1e-9):
            level = 0.5 * (high + low_right)
            position = _find_crossing(
                profile,
                max(start, stop - 2 * width),
                min(right_window, stop + 2 * width),
                level,
                rising=False,
            )
            if position is not None:
                bounds = _transition_bounds(
                    profile, position, low_right, high, max(4, width), rising=False
                )
                if bounds is not None:
                    right_edge = _Edge(position, bounds[0], bounds[1], False)
                    edges.append(right_edge)

    if left_edge is None or right_edge is None or low_left is None or low_right is None:
        return edges, None
    amplitude = min(high - low_left, high - low_right)
    if amplitude <= max(SUPPORT_NOISE_FACTOR * noise, 1e-9):
        return edges, None
    return edges, _Band(left_edge, right_edge, amplitude, 0.0)


def _support_fraction(
    signed: np.ndarray,
    band: _Band,
    mid: int,
    width: float,
    noise: float,
) -> float:
    core_half = max(1, int(round(0.25 * width)))
    core_start = max(0, mid - core_half)
    core_stop = min(signed.shape[1], mid + core_half)
    if core_stop <= core_start:
        return 0.0
    core_values = np.median(signed[:, core_start:core_stop], axis=1)
    flank_width = max(2, int(round(width / 4)))
    left_stop = max(0, band.left.start)
    right_start = min(signed.shape[1], band.right.end + 1)
    if left_stop < 1 or right_start >= signed.shape[1]:
        return 0.0
    left_background = np.median(signed[:, max(0, left_stop - flank_width):left_stop], axis=1)
    right_background = np.median(signed[:, right_start:min(signed.shape[1], right_start + flank_width)], axis=1)
    contrast = np.minimum(core_values - left_background, core_values - right_background)
    threshold = max(
        SUPPORT_NOISE_FACTOR * noise,
        SUPPORT_AMPLITUDE_FRACTION * band.amplitude,
    )
    return float(np.mean(contrast >= threshold))


def _candidate_rois(
    band: _Band,
    edges: list[_Edge],
    height: int,
    width: int,
    minimum_roi_width: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Bound the two edge rectangles by adjacent edge midpoints."""
    mid = int(math.floor(0.5 * (band.left.position + band.right.position)))
    if mid <= band.left.end or mid >= band.right.start:
        return None

    previous_edges = [edge for edge in edges if edge.position < band.left.position - 1e-9]
    next_edges = [edge for edge in edges if edge.position > band.right.position + 1e-9]
    previous = previous_edges[-1] if previous_edges else None
    following = next_edges[0] if next_edges else None

    left_low = 0
    if previous is not None:
        previous_midpoint = int(math.floor(0.5 * (previous.position + band.left.position)))
        if previous_midpoint <= previous.end:
            return None
        left_low = max(0, previous_midpoint)
    left_high = min(mid - minimum_roi_width, band.left.start - 2)
    if left_low > left_high:
        return None
    target_left = int(math.floor(2.0 * band.left.position - mid))
    outer_left = min(max(target_left, left_low), left_high)

    right_high = width
    if following is not None:
        next_midpoint = int(math.floor(0.5 * (band.right.position + following.position)))
        if next_midpoint > following.start:
            return None
        right_high = next_midpoint
    right_low = max(mid + minimum_roi_width, band.right.end + 3)
    if right_low > right_high:
        return None
    target_right = int(math.ceil(2.0 * band.right.position - mid))
    outer_right = max(min(target_right, right_high), right_low)

    left_bounds = (outer_left, 0, mid, height)
    right_bounds = (mid, 0, outer_right, height)
    if mid - outer_left < minimum_roi_width or outer_right - mid < minimum_roi_width:
        return None
    return left_bounds, right_bounds


def detect_line_candidates(
    image: np.ndarray,
    bright_lines: bool = True,
    minimum_roi_width: int = 12,
    minimum_height: int = 80,
) -> list[AutoLineCandidate]:
    """Detect complete near-vertical SEM lines and return paired edge ROIs.

    ``bright_lines=True`` expects bright stripes on a darker background;
    ``False`` is the same analysis after sign reversal for dark stripes on a
    bright background.  Candidates are ordered left to right.  The function
    returns an empty list when no sustained, separable line is present, and
    raises ``ValueError`` for invalid arrays or an image smaller than the
    requested ROI limits.
    """
    resolved_width = _positive_integer(minimum_roi_width, "minimum_roi_width")
    resolved_height = _positive_integer(minimum_height, "minimum_height")
    image_work, height, width = _validate_image(
        image, resolved_width, resolved_height
    )
    if not isinstance(bright_lines, (bool, np.bool_)):
        raise ValueError("bright_lines must be a boolean.")
    signed, profile = _line_profile(image_work, bool(bright_lines))
    noise = _robust_pixel_noise(image_work)
    profile_span = float(np.ptp(profile))
    if profile_span <= 1e-12 or profile_span < SUPPORT_NOISE_FACTOR * noise:
        return []

    threshold = _otsu_threshold(profile)
    if threshold is None:
        return []
    runs = _mask_runs(profile > threshold)

    edges: list[_Edge] = []
    bands: list[_Band] = []
    for index, run in enumerate(runs):
        previous_stop = runs[index - 1][1] if index > 0 else 0
        next_start = runs[index + 1][0] if index + 1 < len(runs) else width
        run_edges, band = _band_edges(
            profile,
            run,
            previous_stop,
            next_start,
            resolved_width,
            noise,
        )
        edges.extend(run_edges)
        start, stop = run
        if start > 0 and not any(edge.rising for edge in run_edges):
            edges.append(_Edge(start - 0.5, start - 1, start, True))
        if stop < width and not any(not edge.rising for edge in run_edges):
            edges.append(_Edge(stop - 0.5, stop - 1, stop, False))
        if band is None:
            continue
        separation = band.right.position - band.left.position
        if separation + 0.5 < resolved_width * MIN_LINE_WIDTH_FACTOR:
            continue
        allowed_transition = max(
            MIN_TRANSITION_ALLOWANCE_PX, MAX_TRANSITION_FRACTION * separation
        )
        if (
            band.left.end - band.left.start + 1 > allowed_transition
            or band.right.end - band.right.start + 1 > allowed_transition
        ):
            continue
        mid = int(math.floor(0.5 * (band.left.position + band.right.position)))
        if mid <= band.left.end or mid >= band.right.start:
            continue
        support = _support_fraction(signed, band, mid, separation, noise)
        if support < MIN_SUPPORT_FRACTION:
            continue
        bands.append(_Band(band.left, band.right, band.amplitude, support))

    edges.sort(key=lambda edge: edge.position)
    candidates: list[AutoLineCandidate] = []
    for band in bands:
        rois = _candidate_rois(band, edges, height, width, resolved_width)
        if rois is None:
            continue
        left_bounds, right_bounds = rois
        candidates.append(
            AutoLineCandidate(
                left_roi_bounds_px=left_bounds,
                right_roi_bounds_px=right_bounds,
                center_x_px=0.5 * (band.left.position + band.right.position),
                width_px=band.right.position - band.left.position,
                support_fraction=band.support_fraction,
            )
        )
    candidates.sort(key=lambda candidate: candidate.center_x_px)
    return candidates
