"""Focused tests for periodic bright-edge discovery."""

from __future__ import annotations

import unittest

import numpy as np

from periodic_line_detection import (
    BrightBandCandidate,
    bright_band_position,
    detect_bright_bands,
    select_periodic_pairs,
)

HEIGHT = 240
WIDTH = 620


def render_bands(
    specs,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    background: float = 25.0,
    amplitude: float = 175.0,
    noise_sigma: float = 2.0,
    meander_px: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Render narrow Gaussian, flat, or asymmetric bright bands."""
    rng = np.random.default_rng(seed)
    rows = np.arange(height, dtype=float)[:, None]
    columns = np.arange(width, dtype=float)[None, :]
    shift = meander_px * np.sin(2.0 * np.pi * rows / 70.0)
    image = np.full((height, width), background, dtype=float)
    for center, size, shape in specs:
        if shape == "gaussian":
            image += amplitude * np.exp(-0.5 * ((columns - (center + shift)) / size) ** 2)
        elif shape == "flat":
            image += amplitude * (
                (columns >= center - size / 2.0 + shift)
                & (columns < center + size / 2.0 + shift)
            )
        else:
            left = amplitude * np.exp(-0.5 * ((columns - (center + shift)) / size) ** 2)
            right = amplitude * np.exp(
                -0.5 * ((columns - (center + shift)) / (2.5 * size)) ** 2
            )
            image += np.maximum(left, right)
    image += rng.normal(0.0, noise_sigma, size=(height, width))
    return np.clip(image, 0.0, 255.0)


def dummy_bands(count: int) -> list[BrightBandCandidate]:
    return [
        BrightBandCandidate((index, 0, index + 1, 1), float(index), float(index), float(index + 1), 1.0)
        for index in range(count)
    ]


class DetectBrightBandsTests(unittest.TestCase):
    def assert_candidate_shape(self, candidate: BrightBandCandidate, image: np.ndarray) -> None:
        self.assertIsInstance(candidate, BrightBandCandidate)
        height, width = image.shape
        left, top, right, bottom = candidate.roi_bounds_px
        self.assertEqual((top, bottom), (0, height))
        self.assertGreaterEqual(left, 0)
        self.assertLessEqual(right, width)
        self.assertGreaterEqual(right - left, 5)
        self.assertLess(left, candidate.left_half_px)
        self.assertLess(candidate.left_half_px, candidate.center_x_px)
        self.assertLess(candidate.center_x_px, candidate.right_half_px)
        self.assertLess(candidate.right_half_px, right)
        self.assertAlmostEqual(
            candidate.center_x_px,
            0.5 * (candidate.left_half_px + candidate.right_half_px),
        )
        self.assertGreaterEqual(candidate.support_fraction, 0.60)
        self.assertLessEqual(candidate.support_fraction, 1.0)

    def test_periodic_narrow_gaussian_edges_with_unequal_spacings(self) -> None:
        centers = [50, 120, 220, 290, 390, 460, 560]
        image = render_bands(
            [(center, 1.4, "gaussian") for center in centers],
            meander_px=2.0,
            noise_sigma=2.5,
            seed=5,
        )
        candidates = detect_bright_bands(image)

        self.assertEqual(len(candidates), len(centers))
        self.assertEqual([c.center_x_px for c in candidates], sorted(c.center_x_px for c in candidates))
        for index, candidate in enumerate(candidates):
            self.assert_candidate_shape(candidate, image)
            self.assertAlmostEqual(candidate.center_x_px, centers[index], delta=3.0)
            for other_index, other_center in enumerate(centers):
                if other_index != index:
                    left, _, right, _ = candidate.roi_bounds_px
                    self.assertFalse(left <= other_center < right)
            band_width = candidate.right_half_px - candidate.left_half_px
            self.assertLess(band_width, 24.0)

    def test_flat_asymmetric_and_narrow_bands_are_counted(self) -> None:
        specs = [
            (60, 3.0, "flat"),
            (180, 1.0, "gaussian"),
            (300, 4.0, "asymmetric"),
            (430, 5.0, "flat"),
            (540, 1.5, "gaussian"),
        ]
        image = render_bands(specs, meander_px=1.5, seed=9)
        candidates = detect_bright_bands(image)

        self.assertEqual(len(candidates), len(specs))
        for (center, _, _), candidate in zip(specs, candidates):
            self.assert_candidate_shape(candidate, image)
            self.assertAlmostEqual(candidate.center_x_px, center, delta=3.5)
            self.assertLess(candidate.right_half_px - candidate.left_half_px, 24.0)

    def test_near_neighbour_rois_exclude_the_dark_midpoint(self) -> None:
        # pitch 14 with a ~2.5 px half-height width: the 12 px minimum width must
        # stay inside the dark gap instead of swallowing the neighbouring midpoint.
        centers = [100.0 + 14.0 * index for index in range(20)]
        image = render_bands([(center, 1.0, "gaussian") for center in centers], noise_sigma=1.0)
        candidates = detect_bright_bands(image)

        self.assertEqual(len(candidates), len(centers))
        # Every 12 px minimum window must stay inside the ~11 px dark gaps, so no
        # ROI may reach the dark midpoint of the neighbouring interval.
        drawn = []
        for index, candidate in enumerate(candidates):
            self.assert_candidate_shape(candidate, image)
            left, _, right, _ = candidate.roi_bounds_px
            drawn.append((left, right))
        for index in range(len(candidates) - 1):
            self.assertLessEqual(drawn[index][1], drawn[index + 1][0])
        for index, candidate in enumerate(candidates):
            left, right = drawn[index]
            for other_index, other in enumerate(candidates):
                if other_index == index:
                    continue
                other_left, other_right = drawn[other_index]
                midpoint = 0.5 * (other.center_x_px + candidate.center_x_px)
                self.assertFalse(left <= midpoint < right)
                if abs(other_index - index) > 1:
                    self.assertFalse(left < other_right and other_left < right)
                if other_index == index - 1:
                    self.assertLessEqual(other.center_x_px, left)
                if other_index == index + 1:
                    self.assertLessEqual(right, other.center_x_px)

    def test_wide_flat_bands_keep_a_small_symmetric_margin(self) -> None:
        # A 24 px flat top must not inflate the dark margin to half of its width.
        for pitch in (50.0, 46.0, 44.0, 40.0):
            with self.subTest(pitch=pitch):
                centers = [100.0 + pitch * index for index in range(8)]
                image = render_bands([(center, 24.0, "flat") for center in centers], noise_sigma=1.0)
                candidates = detect_bright_bands(image)
                self.assertEqual(len(candidates), len(centers))
                for index, candidate in enumerate(candidates):
                    left, _, right, _ = candidate.roi_bounds_px
                    self.assertAlmostEqual(candidate.right_half_px - candidate.left_half_px, 24.0, delta=2.0)
                    self.assertGreaterEqual(right - left, 12)
                    # Margin is measured from the bright body, not from the
                    # half-height width, so a wide band keeps at most 3 dark px and
                    # only the 12 px minimum extension may add more.
                    self.assertIn(candidate.band_left_px - left, range(0, 4))
                    self.assertIn(right - candidate.band_right_px, range(0, 4))
                    self.assertLessEqual(
                        right - candidate.band_right_px,
                        candidate.band_left_px - left + max(0, 12 - (candidate.band_right_px - candidate.band_left_px)),
                    )
                    if index + 1 < len(candidates):
                        midpoint = 0.5 * (candidate.center_x_px + candidates[index + 1].center_x_px)
                        self.assertFalse(left <= midpoint < right)
                        self.assertLess(right, candidates[index + 1].roi_bounds_px[0])
                self.assertAlmostEqual(candidates[0].center_x_px, centers[0], delta=3.0)

    def test_asymmetric_bands_keep_each_band_inside_its_own_roi(self) -> None:
        for pitch, size in ((50.0, 6.0), (40.0, 1.5), (36.0, 2.0)):
            with self.subTest(pitch=pitch, size=size):
                centers = [100.0 + pitch * index for index in range(8)]
                image = render_bands([(center, size, "asymmetric") for center in centers], noise_sigma=1.0)
                candidates = detect_bright_bands(image)
                self.assertEqual(len(candidates), len(centers))
                for index, candidate in enumerate(candidates):
                    left, _, right, _ = candidate.roi_bounds_px
                    # A constant 3 px dark margin around the bright body, never a
                    # margin proportional to the (asymmetric) half-height width.
                    self.assertGreaterEqual(right - left, 12)
                    self.assertLessEqual(left, candidate.band_left_px - 3)
                    self.assertGreaterEqual(right, candidate.band_right_px + 3)
                    self.assertLessEqual(left, candidate.band_left_px)
                    self.assertGreaterEqual(right, candidate.band_right_px)
                    self.assertLessEqual(left, candidate.left_half_px)
                    self.assertGreaterEqual(right, candidate.right_half_px)
                    self.assertLessEqual(right - left, (candidate.band_right_px - candidate.band_left_px) + 2 * 3)
                    for other_index, other in enumerate(candidates):
                        if abs(other_index - index) > 1:
                            midpoint = 0.5 * (other.center_x_px + candidate.center_x_px)
                            self.assertFalse(left <= midpoint < right)

    def test_rejected_when_no_safe_span_can_hold_the_minimum_roi(self) -> None:
        # pitch 12 leaves ~10 px between bright bodies: a 12 px measurement window
        # necessarily crosses a dark-interior midpoint.  Every band there is a
        # verified bright edge, so the detection must fail as a whole instead of
        # returning only the two outermost centres (100 and 232), which the pair
        # selector would then mis-pair as one 132 px wide "line".
        centers = [100.0 + 12.0 * index for index in range(12)]
        image = render_bands([(center, 1.0, "gaussian") for center in centers], noise_sigma=1.0)
        candidates = detect_bright_bands(image, minimum_roi_width=12)
        self.assertEqual(candidates, [])
        self.assertEqual(select_periodic_pairs(candidates, 1, "trench", "trench"), [])
        narrow = detect_bright_bands(image, minimum_roi_width=10)
        self.assertEqual(len(narrow), len(centers))
        self.assertEqual([round(candidate.center_x_px) for candidate in narrow], [int(center) for center in centers])
        for index in range(len(narrow) - 1):
            self.assertLessEqual(narrow[index].roi_bounds_px[2], narrow[index + 1].roi_bounds_px[0])
            # Local valley background may be shared; the opposite bright peak
            # must never enter this band's actual measurement rectangle.
            self.assertLess(narrow[index].roi_bounds_px[2], narrow[index + 1].center_x_px)
            self.assertGreater(narrow[index + 1].roi_bounds_px[0], narrow[index].center_x_px)

    def test_noise_or_boundary_runs_never_fail_the_whole_detection(self) -> None:
        # A real periodic group plus one unqualified low-contrast run and one band
        # clipped by the image edge must return the real bands, not an empty list:
        # only verified bright bands can make the whole detection fail.
        specs = [(60.0 + 40.0 * index, 2.0, "gaussian") for index in range(5)]
        image = render_bands(specs, width=400, noise_sigma=1.0)
        image[:, 330:335] += 12.0
        image[:, -1] += 120.0
        candidates = detect_bright_bands(image)
        self.assertEqual(len(candidates), len(specs))
        for (center, _, _), candidate in zip(specs, candidates):
            self.assertAlmostEqual(candidate.center_x_px, center, delta=3.0)

    def test_partial_bands_at_horizontal_boundaries_are_excluded(self) -> None:
        local_width = 300
        image = render_bands(
            [(0, 2.0, "gaussian"), (150, 2.0, "gaussian"), (local_width - 1, 2.0, "gaussian")],
            width=local_width,
            seed=3,
        )
        candidates = detect_bright_bands(image)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].center_x_px, 150.0, delta=2.0)
        left, _, right, _ = candidates[0].roi_bounds_px
        self.assertGreater(left, 0)
        self.assertLess(right, local_width)

    def test_uniform_and_noise_images_return_empty(self) -> None:
        self.assertEqual(detect_bright_bands(np.full((160, 300), 128.0)), [])
        noise = np.random.default_rng(4).normal(100.0, 8.0, size=(160, 300))
        self.assertEqual(detect_bright_bands(noise), [])

    def test_repeat_is_deterministic_and_input_is_unchanged(self) -> None:
        image = render_bands(
            [(80, 1.5, "gaussian"), (240, 3.0, "flat"), (430, 2.0, "asymmetric")],
            meander_px=2.0,
            seed=11,
        )
        original = image.copy()
        first = detect_bright_bands(image)
        second = detect_bright_bands(image)

        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_invalid_images_and_limits_raise_value_error(self) -> None:
        valid = np.zeros((100, 60), dtype=np.float64)
        for bad in (
            [[0.0] * 60] * 100,
            np.zeros(100),
            np.zeros((0, 60)),
            np.full((100, 60), np.nan),
            np.full((79, 60), 1.0),
            np.full((100, 4), 1.0),
            np.zeros((100, 60), dtype=object),
        ):
            with self.assertRaises(ValueError):
                detect_bright_bands(bad)
        for kwargs in (
            {"minimum_roi_width": 0},
            {"minimum_roi_width": True},
            {"minimum_height": 0},
        ):
            with self.assertRaises(ValueError):
                detect_bright_bands(valid, **kwargs)


class BrightBandPositionTests(unittest.TestCase):
    def test_elevated_valley_is_two_bands_and_never_one_wide_crossing(self) -> None:
        for valley in (136.0, 160.0, 180.0):
            with self.subTest(valley=valley):
                profile = np.array([40, 40, 49, 70, 119, 182, 218, 203,
                                    max(155, valley), valley, max(158, valley),
                                    198, 206, 167, 92, 51, 39, 38, 40], dtype=float)
                self.assertTrue(np.isnan(bright_band_position(profile)))
                bands = detect_bright_bands(np.tile(profile, (180, 1)))
                self.assertEqual(len(bands), 2)
                self.assertLess(bands[0].right_half_px, 9)
                self.assertGreater(bands[1].left_half_px, 9)
                self.assertLessEqual(bands[0].roi_bounds_px[2], 10)
                self.assertGreaterEqual(bands[1].roi_bounds_px[0], 9)
                for band, expected in zip(bands, (6, 12)):
                    point = bright_band_position(profile, reference_band_px=(band.left_half_px, band.right_half_px))
                    self.assertAlmostEqual(point, expected, delta=1.5)

    def test_flat_plateau_positions_and_definition_order(self) -> None:
        profile = np.full(80, 10.0)
        profile[30:34] = 110.0

        left = bright_band_position(profile, "left")
        center = bright_band_position(profile, "center")
        right = bright_band_position(profile, "right")

        self.assertAlmostEqual(left, 29.5)
        self.assertAlmostEqual(center, 31.5)
        self.assertAlmostEqual(right, 33.5)
        self.assertLess(left, center)
        self.assertLess(center, right)

    def test_asymmetric_peak_uses_half_height_midpoint(self) -> None:
        x = np.arange(81, dtype=float)
        profile = np.interp(x, [0, 10, 20, 60, 80], [10, 10, 100, 10, 10])

        left = bright_band_position(profile, "left")
        center = bright_band_position(profile, "center")
        right = bright_band_position(profile, "right")

        self.assertAlmostEqual(left, 15.0)
        self.assertAlmostEqual(right, 40.0)
        self.assertAlmostEqual(center, 27.5)
        self.assertNotAlmostEqual(center, float(np.argmax(profile)))
        self.assertLess(left, center)
        self.assertLess(center, right)

    def test_no_contrast_returns_nan(self) -> None:
        self.assertTrue(np.isnan(bright_band_position(np.full(80, 12.0))))
        self.assertTrue(
            np.isnan(bright_band_position(np.full(80, 12.0), "left"))
        )

    def test_single_sided_or_clipped_peaks_return_nan(self) -> None:
        ramp = np.linspace(0.0, 100.0, 60)
        x = np.arange(60, dtype=float)
        clipped_left = 20.0 + 100.0 * np.exp(-0.5 * (x / 2.0) ** 2)
        clipped_right = clipped_left[::-1]
        for profile in (ramp, ramp[::-1], clipped_left, clipped_right):
            self.assertTrue(np.isnan(bright_band_position(profile)))

    def test_multiple_comparable_peaks_return_nan(self) -> None:
        x = np.arange(101, dtype=float)
        profile = 30.0 + 100.0 * np.exp(-0.5 * ((x - 25.0) / 2.0) ** 2)
        profile += 100.0 * np.exp(-0.5 * ((x - 75.0) / 2.0) ** 2)
        self.assertTrue(np.isnan(bright_band_position(profile)))

    def test_nonfinite_and_invalid_arguments(self) -> None:
        profile = 10.0 + 100.0 * np.exp(-0.5 * ((np.arange(80) - 40.0) / 2.0) ** 2)
        with self.assertRaises(ValueError):
            bright_band_position(profile, "middle")
        with self.assertRaises(ValueError):
            bright_band_position(profile.reshape(8, 10))
        with self.assertRaises(ValueError):
            bright_band_position([0.0, 1.0, 2.0])
        bad = profile.copy()
        bad[5] = np.inf
        self.assertTrue(np.isnan(bright_band_position(bad)))


class SelectPeriodicPairsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bands = dummy_bands(7)

    def test_same_kind_keeps_reference_parity(self) -> None:
        pairs = select_periodic_pairs(self.bands, 2, "trench", "trench")
        self.assertEqual(pairs, [(1, 2), (3, 4), (5, 6)])
        pairs = select_periodic_pairs(self.bands, 3, "platform", "platform")
        self.assertEqual(pairs, [(0, 1), (2, 3), (4, 5)])

    def test_other_kind_uses_opposite_parity(self) -> None:
        pairs = select_periodic_pairs(self.bands, 2, "trench", "platform")
        self.assertEqual(pairs, [(0, 1), (2, 3), (4, 5)])
        pairs = select_periodic_pairs(self.bands, 1, "platform", "trench")
        self.assertEqual(pairs, [(1, 2), (3, 4), (5, 6)])

    def test_two_bands_and_small_or_empty_lists(self) -> None:
        pair_bands = dummy_bands(2)
        self.assertEqual(select_periodic_pairs(pair_bands, 1, "trench", "trench"), [(0, 1)])
        self.assertEqual(select_periodic_pairs(pair_bands, 1, "trench", "platform"), [])
        self.assertEqual(select_periodic_pairs([], 1), [])
        self.assertEqual(select_periodic_pairs(dummy_bands(1), 1), [])

    def test_invalid_reference_index_or_kind_raises(self) -> None:
        for reference in (0, -1, len(self.bands), 2.0, True):
            with self.assertRaises(ValueError):
                select_periodic_pairs(self.bands, reference, "trench", "trench")
        for reference_kind, target_kind in (
            ("line", "trench"),
            ("trench", "line"),
            ("Trench", "platform"),
        ):
            with self.assertRaises(ValueError):
                select_periodic_pairs(self.bands, 1, reference_kind, target_kind)

    def test_input_list_and_bands_are_untouched(self) -> None:
        bands = dummy_bands(7)
        snapshot = list(bands)
        select_periodic_pairs(bands, 2, "trench", "platform")
        self.assertEqual(bands, snapshot)
        self.assertEqual([id(band) for band in bands], [id(band) for band in snapshot])


if __name__ == "__main__":
    unittest.main()
