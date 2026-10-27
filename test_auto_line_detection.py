"""Focused tests for automatic near-vertical line discovery."""

from __future__ import annotations

import unittest

import numpy as np

from auto_line_detection import AutoLineCandidate, detect_line_candidates

HEIGHT = 260
WIDTH = 520


def make_stripe_image(
    stripes,
    height: int = HEIGHT,
    width: int = WIDTH,
    background: float = 40.0,
    amplitude: float = 170.0,
    softness: float = 1.5,
    noise_sigma: float = 2.0,
    meander_px: float = 0.0,
    period: float = 90.0,
    tilt_px: float = 0.0,
    illumination: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Render smooth near-vertical stripes with optional roughness and noise."""
    rng = np.random.default_rng(seed)
    rows = np.arange(height, dtype=float)[:, None]
    columns = np.arange(width, dtype=float)[None, :]
    shift = meander_px * np.sin(2.0 * np.pi * rows / period)
    shift = shift + tilt_px * (rows / max(height - 1, 1) - 0.5)
    image = np.full((height, width), background, dtype=float)
    for left, right in stripes:
        left_profile = 0.5 * (1.0 + np.tanh((columns - (left + shift)) / softness))
        right_profile = 0.5 * (1.0 + np.tanh(((right + shift) - columns) / softness))
        image += amplitude * left_profile * right_profile
    if illumination:
        image *= 1.0 + illumination * (columns / max(width - 1, 1) - 0.5)
    if noise_sigma:
        image += rng.normal(0.0, noise_sigma, size=(height, width))
    return np.clip(image, 0.0, 255.0)


class AutoLineDetectionTests(unittest.TestCase):
    def assert_candidate_geometry(
        self,
        candidate: AutoLineCandidate,
        width: int = WIDTH,
        height: int = HEIGHT,
        minimum_roi_width: int = 12,
        minimum_height: int = 80,
    ) -> None:
        self.assertIsInstance(candidate, AutoLineCandidate)
        left_roi = candidate.left_roi_bounds_px
        right_roi = candidate.right_roi_bounds_px
        self.assertEqual(left_roi[2], right_roi[0])
        self.assertEqual(left_roi[1], 0)
        self.assertEqual(right_roi[1], 0)
        self.assertEqual(left_roi[3], height)
        self.assertEqual(right_roi[3], height)
        for bounds in (left_roi, right_roi):
            left, top, right, bottom = bounds
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, width)
            self.assertLessEqual(bottom, height)
            self.assertGreaterEqual(right - left, minimum_roi_width)
            self.assertGreaterEqual(bottom - top, minimum_height)
        self.assertGreater(candidate.width_px, 0.0)
        self.assertGreaterEqual(candidate.support_fraction, 0.60)
        self.assertLessEqual(candidate.support_fraction, 1.0)

    def test_multiple_isolated_bright_stripes(self) -> None:
        stripes = ((60, 100), (220, 265), (400, 440))
        image = make_stripe_image(
            stripes,
            meander_px=4.0,
            tilt_px=2.0,
            illumination=0.08,
            noise_sigma=2.5,
            seed=1,
        )
        candidates = detect_line_candidates(image)

        self.assertEqual(len(candidates), len(stripes))
        centers = [candidate.center_x_px for candidate in candidates]
        self.assertEqual(centers, sorted(centers))
        for candidate, (stripe_left, stripe_right) in zip(candidates, stripes):
            self.assertAlmostEqual(
                candidate.center_x_px, (stripe_left + stripe_right) / 2.0, delta=4.0
            )
            self.assertAlmostEqual(
                candidate.width_px, stripe_right - stripe_left, delta=6.0
            )
            self.assert_candidate_geometry(candidate)
            left_roi = candidate.left_roi_bounds_px
            right_roi = candidate.right_roi_bounds_px
            self.assertLessEqual(left_roi[0], stripe_left - 1)
            self.assertGreaterEqual(left_roi[2], stripe_left + 1)
            self.assertLessEqual(left_roi[2], stripe_right - 2)
            self.assertLessEqual(right_roi[0], stripe_right - 1)
            self.assertGreaterEqual(right_roi[2], stripe_right + 2)
            for other_left, other_right in stripes:
                if (other_left, other_right) == (stripe_left, stripe_right):
                    continue
                for edge in (other_left, other_right):
                    self.assertFalse(left_roi[0] <= edge < left_roi[2])
                    self.assertFalse(right_roi[0] <= edge < right_roi[2])

    def test_dark_stripes_via_sign_reversal_match_bright_count(self) -> None:
        stripes = ((100, 140), (300, 345))
        bright = make_stripe_image(stripes, meander_px=3.0, noise_sigma=2.0, seed=3)
        dark = 255.0 - bright

        bright_candidates = detect_line_candidates(bright, bright_lines=True)
        dark_candidates = detect_line_candidates(dark, bright_lines=False)

        self.assertEqual(len(bright_candidates), len(stripes))
        self.assertEqual(len(dark_candidates), len(stripes))
        for bright_candidate, dark_candidate in zip(bright_candidates, dark_candidates):
            self.assertAlmostEqual(
                bright_candidate.center_x_px, dark_candidate.center_x_px, delta=1.0
            )
            self.assertAlmostEqual(
                bright_candidate.width_px, dark_candidate.width_px, delta=1.0
            )
            self.assert_candidate_geometry(dark_candidate)
        # With the opposite polarity selected, the bright gap is itself a band.
        bright_gaps = detect_line_candidates(dark, bright_lines=True)
        self.assertEqual(len(bright_gaps), 1)
        self.assertAlmostEqual(bright_gaps[0].center_x_px, 220.0, delta=3.0)

    def test_single_stripe(self) -> None:
        image = make_stripe_image(
            ((200, 240),), meander_px=2.0, noise_sigma=2.0, seed=2
        )
        candidates = detect_line_candidates(image)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertAlmostEqual(candidate.center_x_px, 220.0, delta=2.0)
        self.assertAlmostEqual(candidate.width_px, 40.0, delta=3.0)
        self.assertGreater(candidate.support_fraction, 0.9)
        self.assert_candidate_geometry(candidate)

    def test_bright_lines_can_occupy_most_of_the_image(self) -> None:
        image = make_stripe_image(((30, 190),), width=220, noise_sigma=1.0)
        candidates = detect_line_candidates(image)
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].support_fraction, 0.9)

    def test_rejected_narrow_band_still_limits_neighbor_rois(self) -> None:
        stripes = ((100, 140), (151, 159), (240, 280))
        image = make_stripe_image(stripes, softness=0.8, noise_sigma=0.0)
        candidates = detect_line_candidates(image)
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            for bounds in (candidate.left_roi_bounds_px, candidate.right_roi_bounds_px):
                self.assertFalse(bounds[0] <= 151 < bounds[2])
                self.assertFalse(bounds[0] <= 159 < bounds[2])

    def test_uniform_images_return_empty(self) -> None:
        for value, dtype in ((0, np.uint8), (128, np.uint8), (128.0, np.float64), (200, np.int16)):
            image = np.full((HEIGHT, WIDTH), value, dtype=dtype)
            self.assertEqual(detect_line_candidates(image), [])

    def test_pure_random_noise_returns_empty(self) -> None:
        for seed in range(3):
            rng = np.random.default_rng(seed)
            image = rng.normal(128.0, 40.0, size=(HEIGHT, WIDTH))
            self.assertEqual(detect_line_candidates(image), [])
        for seed in range(2):
            rng = np.random.default_rng(100 + seed)
            image = rng.integers(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8)
            self.assertEqual(detect_line_candidates(image), [])

    def test_partial_border_stripes_are_excluded(self) -> None:
        image = make_stripe_image(
            ((0, 40), (240, 280), (480, 520)), seed=4, noise_sigma=2.0
        )
        candidates = detect_line_candidates(image)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].center_x_px, 260.0, delta=2.0)
        self.assert_candidate_geometry(candidates[0])

    def test_fully_contained_near_edge_stripe_is_retained(self) -> None:
        image = make_stripe_image(((6, 46),), seed=5, noise_sigma=1.5)
        candidates = detect_line_candidates(image)

        self.assertEqual(len(candidates), 1)
        self.assert_candidate_geometry(candidates[0])
        self.assertGreaterEqual(candidates[0].left_roi_bounds_px[0], 0)

    def test_mild_tilt_noise_and_illumination(self) -> None:
        stripes = ((90, 130), (300, 345))
        image = make_stripe_image(
            stripes,
            meander_px=5.0,
            tilt_px=4.0,
            illumination=0.20,
            noise_sigma=3.0,
            seed=6,
        )
        candidates = detect_line_candidates(image)

        self.assertEqual(len(candidates), len(stripes))
        for candidate, (stripe_left, stripe_right) in zip(candidates, stripes):
            self.assertAlmostEqual(
                candidate.center_x_px, (stripe_left + stripe_right) / 2.0, delta=5.0
            )
            self.assertAlmostEqual(
                candidate.width_px, stripe_right - stripe_left, delta=6.0
            )
            self.assertGreaterEqual(candidate.support_fraction, 0.7)
            self.assert_candidate_geometry(candidate)

    def test_smooth_illumination_hump_is_not_a_stripe(self) -> None:
        rows = np.arange(HEIGHT, dtype=float)[:, None]
        columns = np.arange(WIDTH, dtype=float)[None, :]
        image = 40.0 + 80.0 * np.exp(-((columns - 260.0) / 120.0) ** 2) + 0.0 * rows
        self.assertEqual(detect_line_candidates(image), [])

    def test_roi_geometry_and_minimum_width_parameter(self) -> None:
        stripes = ((100, 140), (300, 340))
        image = make_stripe_image(stripes, seed=7, noise_sigma=2.0)
        candidates = detect_line_candidates(image, minimum_roi_width=20)

        self.assertEqual(len(candidates), len(stripes))
        for candidate in candidates:
            self.assert_candidate_geometry(
                candidate, minimum_roi_width=20, minimum_height=80
            )

        narrow = make_stripe_image(((200, 216),), seed=8, noise_sigma=1.5)
        self.assertEqual(detect_line_candidates(narrow), [])

    def test_malformed_inputs_raise_value_error(self) -> None:
        invalid_images = [
            None,
            [],
            "image",
            np.arange(HEIGHT, dtype=float),
            np.zeros((HEIGHT, WIDTH, 1)),
            np.zeros((0, WIDTH)),
            np.full((HEIGHT, WIDTH), np.nan),
            np.full((HEIGHT, WIDTH), np.inf),
            np.full((HEIGHT, WIDTH), -np.inf),
            np.zeros((HEIGHT, WIDTH), dtype=complex),
            np.full((HEIGHT, WIDTH), None, dtype=object),
            np.zeros((79, WIDTH)),
            np.zeros((HEIGHT, 20)),
        ]
        for image in invalid_images:
            with self.subTest(image=repr(type(image))):
                with self.assertRaises(ValueError):
                    detect_line_candidates(image)

        valid = np.full((HEIGHT, WIDTH), 128.0)
        for kwargs in (
            {"minimum_roi_width": 0},
            {"minimum_roi_width": -3},
            {"minimum_roi_width": 12.5},
            {"minimum_roi_width": True},
            {"minimum_height": 0},
            {"minimum_height": -80},
            {"minimum_height": 80.5},
            {"minimum_height": False},
            {"bright_lines": 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    detect_line_candidates(valid, **kwargs)
        with self.assertRaises(ValueError):
            detect_line_candidates(valid, minimum_height=HEIGHT + 1)

    def test_repeat_determinism_and_input_unchanged(self) -> None:
        image = make_stripe_image(
            ((80, 120), (300, 340)),
            meander_px=3.0,
            tilt_px=1.0,
            illumination=0.05,
            noise_sigma=2.0,
            seed=9,
        )
        original = image.copy()
        image.setflags(write=False)

        first = detect_line_candidates(image)
        second = detect_line_candidates(image)

        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 1)

    def test_existing_era_measures_discovered_rois_unchanged(self) -> None:
        import ler_lwr_app as app

        stripes = ((80, 124), (260, 296), (400, 440))
        image = make_stripe_image(
            stripes,
            meander_px=3.0,
            tilt_px=2.0,
            illumination=0.06,
            noise_sigma=1.0,
            seed=10,
        )
        candidates = detect_line_candidates(image)
        self.assertEqual(len(candidates), len(stripes))

        for candidate, (stripe_left, stripe_right) in zip(candidates, stripes):
            left_edge = app.analyze_single_edge_era(
                image,
                candidate.left_roi_bounds_px,
                1.0,
                polarity=app.ERA_POLARITY_AUTO,
            )
            right_edge = app.analyze_single_edge_era(
                image,
                candidate.right_roi_bounds_px,
                1.0,
                polarity=app.ERA_POLARITY_AUTO,
            )
            paired = app.pair_single_edges(left_edge, right_edge, 1.0)
            self.assertEqual(left_edge.polarity, app.ERA_POLARITY_RISING)
            self.assertEqual(right_edge.polarity, app.ERA_POLARITY_FALLING)
            self.assertAlmostEqual(
                paired.mean_width_nm, stripe_right - stripe_left, delta=1.5
            )
            self.assertGreaterEqual(len(paired.rows_px), 30)


if __name__ == "__main__":
    unittest.main()
