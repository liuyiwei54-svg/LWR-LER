"""Compact pure-NumPy detection of periodic near-vertical bright edges.

The module works on bright *edges* of a periodic SEM line/space pattern.  A
bright band is a narrow physical edge (typically only 2-10 px wide), so the
old 24 px bright-body width rule deliberately does not apply.  Detection is
discovery-only: the supplied image is never modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BrightBandCandidate",
    "bright_band_position",
    "bright_band_search_bounds",
    "detect_bright_bands",
    "select_periodic_pairs",
]

# A physical band must be visible on this fraction of the image rows.
MIN_SUPPORT_FRACTION = 0.60
SUPPORT_NOISE_FACTOR = 3.0
SUPPORT_AMPLITUDE_FRACTION = 0.20
MIN_PEAK_PROMINENCE_FRACTION = 0.08
# Local dark background kept on each side of a bright band inside its own ROI.
# Constant, not proportional to the band width, so wide flat bands do not pull
# their share of the dark gap into the measurement rectangle.
ROI_DARK_MARGIN_PX = 3
# Default minimum measurement rectangle kept inside one band's own two walls.
DEFAULT_MINIMUM_ROI_WIDTH_PX = 5


@dataclass(frozen=True)
class BrightBandCandidate:
    """One complete near-vertical bright band and its local image ROI.

    ``roi_bounds_px`` is a half-open ``(left, top, right, bottom)`` rectangle
    local to the supplied image.  ``center_x_px`` is the midpoint of the two
    half-height crossings and ``left_half_px``/``right_half_px`` are those two
    subpixel x positions (not widths or signed offsets).
    ``band_left_px``/``band_right_px`` are the half-open integer columns of the
    threshold run that identifies the band's bright body; they bound the small
    dark margin kept inside the ROI and are reported so diagnostics can tell a
    real bright body from dark pixels enclosed by a geometric minimum width.
    """

    roi_bounds_px: tuple[int, int, int, int]
    center_x_px: float
    left_half_px: float
    right_half_px: float
    support_fraction: float
    band_left_px: int | None = None
    band_right_px: int | None = None


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
    if width < minimum_roi_width:
        raise ValueError("image width is smaller than minimum_roi_width.")
    return work, height, width


def _robust_noise(values: np.ndarray) -> float:
    """Estimate white-noise sigma from adjacent-pixel differences."""
    differences = np.abs(np.diff(values, axis=-1))
    if not differences.size:
        return 0.0
    return float(np.median(differences) / (math.sqrt(2.0) * 0.6744897501960817))


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
        # A monotonic SEM flank may be curved. Its second derivative is signal,
        # not evidence of random noise, especially in a narrow single-peak ROI.
        return 0.0
    second = np.concatenate(
        (np.diff(values[:count], n=2), np.diff(values[-count:], n=2))
    )
    return float(
        np.median(np.abs(second)) / (math.sqrt(6.0) * 0.6744897501960817)
    )


def _centered_profile(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signed = image - np.median(image, axis=1, keepdims=True)
    return signed, np.median(signed, axis=0)


def bright_band_search_bounds(size: int, reference_band_px: tuple[float, float]) -> tuple[int, int]:
    """Bound peak identity by the preview band's own center and width."""
    left, right = map(float, reference_band_px)
    if not (math.isfinite(left) and math.isfinite(right) and 0 <= left < right < size):
        raise ValueError("reference band must lie inside the bright-band ROI.")
    center = 0.5 * (left + right)
    radius = max(2.0, 0.75 * (right - left))
    return max(0, math.ceil(center - radius)), min(size, math.floor(center + radius) + 1)


def _peak_basins(values: np.ndarray, noise: float) -> list[tuple[int, int, int, int]]:
    """Significant local maxima and the valley walls belonging to each one.

    Prominence is evaluated against both sides before constructing any ROI.
    A low global threshold must never merge two SEM boundary peaks.
    """
    size = len(values)
    if size < 5:
        return []
    span = float(np.ptp(values))
    prominence_floor = max(3.0 * noise, MIN_PEAK_PROMINENCE_FRACTION * span, 1e-12)
    changes = np.flatnonzero(np.diff(values) != 0) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, size]
    peaks = []
    for start, stop in zip(starts[1:-1], stops[1:-1]):
        top = float(values[start])
        if top <= values[start - 1] or top <= values[stop]:
            continue
        # Stop at a higher peak: a tiny ripple atop a bright flat band has only
        # its local dip as prominence, rather than the full exterior contrast.
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
        # A dark plateau is split centrally. A sharp interior valley may be a
        # single baseline pixel shared by the two neighboring measurement ROIs.
        walls.append(previous_stop + int(round(0.5 * (low[0] + low[-1]))))
    return [
        (start, stop, 0 if index == 0 else walls[index - 1],
         size - 1 if index == len(peaks) - 1 else walls[index])
        for index, (start, stop) in enumerate(peaks)
    ]


def _basin_crossings(
    values: np.ndarray, basin: tuple[int, int, int, int], noise: float,
    fraction: float = 0.5,
) -> tuple[float, float, float] | None:
    """Use the selected peak's separate left/right local dark floors."""
    start, stop, left_wall, right_wall = basin
    if start <= left_wall or stop - 1 >= right_wall:
        return None
    top = float(np.max(values[start:stop]))
    tolerance = max(2.0 * noise, 0.01 * float(np.ptp(values)), 1e-12)
    floors = []
    for side in (values[left_wall:start], values[stop:right_wall + 1]):
        low = float(np.min(side))
        floors.append(float(np.median(side[side <= low + tolerance])))
    contrast = top - max(floors)
    if contrast <= max(SUPPORT_NOISE_FACTOR * noise, 1e-12):
        return None
    left_level = floors[0] + fraction * (top - floors[0])
    right_level = floors[1] + fraction * (top - floors[1])
    left_crossing = right_crossing = None
    for index in range(start - 1, left_wall - 1, -1):
        before, after = values[index:index + 2]
        if before <= left_level < after:
            left_crossing = index + (left_level - before) / (after - before)
            break
    for index in range(stop - 1, right_wall):
        before, after = values[index:index + 2]
        if before >= right_level > after:
            right_crossing = index + (right_level - before) / (after - before)
            break
    if left_crossing is None or right_crossing is None:
        return None
    if not 0 < left_crossing < right_crossing < len(values) - 1:
        return None
    return float(left_crossing), float(right_crossing), contrast


def _referenced_crossings(
    values: np.ndarray, reference_band_px: tuple[float, float] | None,
) -> tuple[float, float, float] | None:
    endpoint_count = max(1, min(3, len(values) // 5))
    noise = _flank_noise(values, endpoint_count)
    basins = _peak_basins(values, noise)
    if reference_band_px is None:
        if len(basins) != 1:
            return None
        basin = basins[0]
    else:
        search_left, search_right = bright_band_search_bounds(len(values), reference_band_px)
        reference_center = 0.5 * sum(reference_band_px)
        eligible = [b for b in basins if b[0] < search_right and b[1] > search_left]
        if not eligible:
            return None
        basin = min(eligible, key=lambda b: abs(0.5 * (b[0] + b[1] - 1) - reference_center))
    crossings = _basin_crossings(values, basin, noise)
    if crossings is None or reference_band_px is None:
        return crossings
    left, right, contrast = crossings
    reference_width = reference_band_px[1] - reference_band_px[0]
    if abs(0.5 * (left + right) - reference_center) > max(2.0, 0.75 * reference_width):
        return None
    if right - left > max(3.0 * reference_width, reference_width + 4.0):
        return None
    return left, right, contrast


def bright_band_position(
    profile: np.ndarray, definition: str = "center",
    reference_band_px: tuple[float, float] | None = None,
) -> float:
    """Left/right local half height, or their midpoint, for one bright peak.

    Unanchored multi-peak profiles are ambiguous. Anchored extraction selects
    only the preview peak and stops each crossing at its own adjacent valley.
    """
    if definition not in ("center", "left", "right"):
        raise ValueError("definition must be exactly 'center', 'left' or 'right'.")
    if not isinstance(profile, np.ndarray) or profile.ndim != 1:
        raise ValueError("profile must be a one-dimensional NumPy array.")
    try:
        values = np.asarray(profile, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("profile must contain real numeric values.") from exc
    if len(values) < 5 or not np.all(np.isfinite(values)):
        return np.nan
    crossings = _referenced_crossings(values, reference_band_px)
    if crossings is None:
        return np.nan
    left, right, _ = crossings
    return left if definition == "left" else right if definition == "right" else 0.5 * (left + right)


def detect_bright_bands(
    image: np.ndarray, minimum_roi_width: int = DEFAULT_MINIMUM_ROI_WIDTH_PX,
    minimum_height: int = 80,
) -> list[BrightBandCandidate]:
    """Discover local SEM boundary peaks, not globally connected bright bodies.

    Each peak has its own two valley baselines, reference half heights, and local
    measurement ROI. No spatial filtering is added. An otherwise valid band
    that cannot fit a safe requested ROI invalidates the whole sequence, since
    silently removing an interior band would change trench/platform parity.
    """
    roi_width = _positive_integer(minimum_roi_width, "minimum_roi_width")
    roi_height = _positive_integer(minimum_height, "minimum_height")
    work, height, width = _validate_image(image, roi_width, roi_height)
    signed, profile = _centered_profile(work)
    # A dense pattern's horizontal differences contain real edge contrast.
    # The smaller orthogonal estimate avoids treating that signal as noise.
    noise = min(_robust_noise(work), _robust_noise(work.T))
    if float(np.ptp(profile)) <= max(3.0 * noise, 1e-12):
        return []
    # The column median removes row noise for discovery; retain the original
    # image noise threshold to reject texture without smoothing measurement data.
    candidates = []
    for basin in _peak_basins(profile, noise):
        peak_start, peak_stop, left_wall, right_wall = basin
        crossings = _basin_crossings(profile, basin, noise)
        body_crossings = _basin_crossings(profile, basin, noise, 0.10)
        if crossings is None or body_crossings is None:
            continue
        left_half, right_half, reference_contrast = crossings
        center = 0.5 * (left_half + right_half)
        body_left = max(left_wall + 1, int(math.floor(body_crossings[0])) + 1)
        body_right = min(right_wall, int(math.ceil(body_crossings[1])))
        roi_left = max(left_wall, body_left - ROI_DARK_MARGIN_PX)
        roi_right = min(right_wall + 1, body_right + ROI_DARK_MARGIN_PX)
        safe_width = max(0, right_wall - left_wall - 1)
        # Validate physical support before failing the sequence for ROI geometry.
        reference = (left_half - roi_left, right_half - roi_left)
        region = signed[:, roi_left:roi_right]
        threshold = max(3.0 * noise, SUPPORT_AMPLITUDE_FRACTION * reference_contrast)
        supported = 0
        for row in region:
            row_crossings = _referenced_crossings(row, reference) if len(row) >= 5 else None
            supported += row_crossings is not None and row_crossings[2] >= threshold
        support = supported / height
        if support < MIN_SUPPORT_FRACTION:
            continue
        if safe_width < roi_width:
            return []
        if roi_right - roi_left < roi_width:
            roi_left = max(left_wall + 1, min(int(round(center - roi_width / 2)), right_wall - roi_width))
            roi_right = roi_left + roi_width
        if not (roi_left < left_half < right_half < roi_right):
            return []
        candidates.append(BrightBandCandidate(
            (roi_left, 0, roi_right, height), center, left_half, right_half, support,
            body_left, body_right,
        ))
    return candidates


def select_periodic_pairs(
    bands: list[BrightBandCandidate],
    reference_interval: int = 1,
    reference_kind: str = "trench",
    target_kind: str = "trench",
) -> list[tuple[int, int]]:
    """Select adjacent band pairs with the requested physical interval parity.

    One-based interval ``i`` lies between zero-based bands ``i - 1`` and ``i``.
    The user supplies the reference interval and its physical kind.  Same-kind
    targets keep the reference parity; the other kind uses the opposite parity.
    Band brightness/width and non-adjacent distances are never used.
    """
    if reference_kind not in ("trench", "platform"):
        raise ValueError("reference_kind must be exactly 'trench' or 'platform'.")
    if target_kind not in ("trench", "platform"):
        raise ValueError("target_kind must be exactly 'trench' or 'platform'.")
    if len(bands) < 2:
        return []
    if isinstance(reference_interval, (bool, np.bool_)) or not isinstance(
        reference_interval, (int, np.integer)
    ):
        raise ValueError("reference_interval must be a positive integer.")
    resolved = int(reference_interval)
    if not 1 <= resolved <= len(bands) - 1:
        raise ValueError("reference_interval must index an adjacent band interval.")
    same_kind = reference_kind == target_kind
    pairs: list[tuple[int, int]] = []
    for interval in range(1, len(bands)):
        parity_matches = interval % 2 == resolved % 2
        if parity_matches == same_kind:
            pairs.append((interval - 1, interval))
    return pairs
