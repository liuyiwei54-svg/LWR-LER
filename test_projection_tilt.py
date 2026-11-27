"""Base numerics for projection tilt (phase 1) and the phase-2 refinement.

Phase 1: ``prepare_projection`` cache + shifted row mean.
Phase 2: ``refine_projection_tilt`` accuracy is measured as the REAL angle
error ``abs(correction_degrees + phi)`` against the analytic truth ``-phi``.
The peak-centroid column of a symmetric analytic stripe set does not drift with
tilt, so its mean absolute deviation is kept only as an independent geometric
invariant — never as evidence of tilt accuracy.
"""

import unittest

import numpy as np

from projection_tilt import (
    MAX_ROW_SAMPLES,
    ProjectionTiltResult,
    PreparedProjection,
    REFINE_MAX_RESIDUAL_DEGREES,
    mean_projection,
    prepare_projection,
    refine_projection_tilt,
)


def _centroid_moment(profile):
    """Centroid x and area-normalized second central moment of a profile."""
    p = np.asarray(profile, dtype=np.float64)
    p = p - p.min()
    total = p.sum()
    x = np.arange(p.size, dtype=np.float64)
    c = float((x * p).sum() / total)
    m2 = float((p * (x - c) ** 2).sum() / total)
    return c, m2


class ProjectionTiltBaseTests(unittest.TestCase):
    def test_non_square_zero_angle_is_row_mean(self):
        h, w = 151, 211
        rng = np.random.default_rng(7)
        img = rng.normal(100.0, 20.0, size=(h, w))
        original = img.copy()

        prepared = prepare_projection(img, 0.5)
        profile = mean_projection(prepared, 0.0)

        self.assertIsInstance(prepared, PreparedProjection)
        self.assertEqual(profile.shape, (w,))
        self.assertEqual(prepared.n, w + 2 * prepared.pad)
        self.assertEqual(prepared.freq.shape[0], prepared.n // 2 + 1)
        self.assertLessEqual(prepared.selected_y.size, MAX_ROW_SAMPLES)
        self.assertEqual(prepared.rowdy.shape, (prepared.selected_y.size,))
        self.assertEqual(prepared.spectra.shape[0], prepared.selected_y.size)
        # original y indices kept
        self.assertEqual(prepared.selected_y[0], 0)
        self.assertEqual(prepared.selected_y[-1], h - 1)

        expected = img[prepared.selected_y].mean(axis=0)
        np.testing.assert_allclose(profile, expected, atol=1e-9)
        np.testing.assert_array_equal(img, original)

    def test_analytic_gaussian_stripes_sharpens_at_true_tilt(self):
        h, w = 800, 400
        cy = (h - 1) / 2.0
        x0, sigma, amp, base, period = 200.0, 3.0, 500.0, 400.0, 97.0
        phi = 0.147
        y = np.arange(h, dtype=np.float64)
        x = np.arange(w, dtype=np.float64)
        xcenter = x0 + (y - cy) * np.tan(np.deg2rad(phi))
        img = base + amp * np.exp(
            -0.5 * ((x[None, :] - xcenter[:, None]) / sigma) ** 2
        )
        img = np.clip(img, 0, 65535)

        prepared = prepare_projection(img, 0.5)
        c0, m0 = _centroid_moment(mean_projection(prepared, 0.0))

        # The stripes lean: x(y) = x0 + dy*tan(phi), so the correction that
        # removes the lean is r = -phi (shift s = dy*tan(r)).
        correct = -phi
        c_true, m_true = _centroid_moment(mean_projection(prepared, correct))
        c_wrong, m_wrong = _centroid_moment(mean_projection(prepared, -correct))

        # Correct correction gives the narrower averaged peak, and its own
        # centroid second moment shrinks well below the r = 0 profile.
        self.assertLess(m_true, m0)
        self.assertLess(m_true, m_wrong)
        self.assertGreater(m_wrong, m0)
        # Peak position does not drift between the two corrections.
        self.assertAlmostEqual(c_true, c_wrong, delta=1.5)
        self.assertAlmostEqual(c_true, c0, delta=1.5)
        # The peak centre stays on the analytic x0.
        self.assertAlmostEqual(c_true, x0, delta=1.0)

    def test_invalid_inputs_and_source_untouched(self):
        rng = np.random.default_rng(3)
        img = rng.normal(size=(40, 33))
        original = img.copy()

        with self.assertRaises(ValueError):
            prepare_projection(img, -0.1)
        with self.assertRaises(ValueError):
            prepare_projection(img, float("nan"))
        with self.assertRaises(ValueError):
            prepare_projection(np.zeros((1, 2, 3)))
        with self.assertRaises(ValueError):
            prepare_projection(np.zeros((0, 5)))

        bad = img.copy()
        bad[3, 4] = np.nan
        with self.assertRaises(ValueError):
            prepare_projection(bad)
        bad = img.copy()
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            prepare_projection(bad)

        prepared = prepare_projection(img, 0.3)
        with self.assertRaises(ValueError):
            mean_projection(prepared, 0.31)
        with self.assertRaises(ValueError):
            mean_projection(prepared, float("inf"))
        with self.assertRaises(ValueError):
            mean_projection(np.zeros(33, dtype=np.float64), 0.0)

        # Boundary of the prepared range is accepted.
        profile = mean_projection(prepared, 0.3)
        self.assertEqual(profile.shape, (33,))
        self.assertTrue(np.all(np.isfinite(profile)))
        np.testing.assert_array_equal(img, original)



H, W = 800, 600
CY = (H - 1) / 2.0
X0S = (90, 190, 290, 390, 490)


def _stripes(phi_deg, invert=False):
    """Independent analytic truth: 5 Gaussian bright stripes leaning by phi."""
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    img = np.full((H, W), 20.0, dtype=np.float64)
    for x0 in X0S:
        centre = x0 + (y - CY) * np.tan(np.deg2rad(phi_deg))
        img = img + 100.0 * np.exp(-0.5 * ((x - centre) / 3.0) ** 2)
    if invert:
        img = img.max() + img.min() - img
    return img


def _stripes_extra(phi_deg, rng, noise=0.0, ramp=0.0, amplitudes=(100.0, 60.0, 130.0, 80.0, 110.0),
                   sigmas=(3.0, 2.4, 3.6, 2.8, 3.2), offsets=X0S):
    """Unequal-amplitude stripes + optional row ramp + optional Gaussian noise."""
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]
    img = np.full((H, W), 20.0, dtype=np.float64)
    for x0, amp, sigma in zip(offsets, amplitudes, sigmas):
        centre = x0 + (y - CY) * np.tan(np.deg2rad(phi_deg))
        img = img + amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    img = img + ramp * y
    if noise > 0:
        img = img + rng.normal(0.0, noise, size=img.shape)
    return img


def _antiparallel_image(phi_left, phi_right):
    """Left half leans by phi_left, right half by phi_right (signed truth per half)."""
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    img = np.full((H, W), 20.0, dtype=np.float64)
    for x0 in X0S:
        lean = phi_left if x0 < W / 2 else phi_right
        centre = x0 + (y - CY) * np.tan(np.deg2rad(lean))
        img = img + 100.0 * np.exp(-0.5 * ((x - centre) / 3.0) ** 2)
    return img


def _mean_abs_dev_error(img, phi_true, correction):
    """Independent geometric invariant: per-stripe peak centroid deviation (px).

    The stripe set is symmetric and the rotation is about the image centre, so
    the averaged peak centroid does not move with tilt; a non-zero value would
    flag a broken projection/geometry, but a ~0 value says nothing about the
    tilt angle being right.
    """
    prepared = prepare_projection(img, 2.0)
    profile = mean_projection(prepared, correction)
    errs = []
    for x0 in X0S:
        lo, hi = int(x0 - 15), int(x0 + 16)
        seg = profile[lo:hi]
        wts = np.clip(seg - min(float(seg.min()), float(np.median(profile))), 0.0, None)
        if wts.sum() <= 0:
            errs.append(float("inf"))
            continue
        idx = np.arange(lo, hi, dtype=np.float64)
        errs.append(abs(float((wts * idx).sum() / wts.sum()) - x0))
    return float(np.mean(errs))


def _angle_error(result, phi):
    """True error in degrees: correction should equal -phi."""
    return abs(float(result.correction_degrees) + phi)


def _report(tag, phi, initial, res):
    print(f"[{tag}] phi={phi!r} initial={initial!r} refined={res.refined!r} "
          f"correction={res.correction_degrees!r} reason={res.reason!r} "
          f"peaks={res.peak_count} before={res.score_before!r} after={res.score_after!r}")
    return res


def _assert_samples_are_parallel_numbers(case, res):
    case.assertIsInstance(res.angle_samples, tuple)
    case.assertIsInstance(res.score_samples, tuple)
    case.assertEqual(len(res.angle_samples), len(res.score_samples))
    for angle in res.angle_samples:
        case.assertIsInstance(angle, float)
    for score in res.score_samples:
        case.assertIsInstance(score, float)


class ProductionRefineTests(unittest.TestCase):
    def test_phi_correction_halves_angle_error(self):
        cases = (0.037, -0.037, 0.147, -0.147, 0.41, -0.41)
        for phi in cases:
            for invert in (False, True):
                tag = f"phi={phi:+g} invert={invert}"
                img = _stripes(phi, invert=invert)
                original = img.copy()
                res = _report(tag, phi, 0.0, refine_projection_tilt(img, 0.0))
                np.testing.assert_array_equal(img, original)
                self.assertTrue(res.refined, f"{tag}: refined False, reason={res.reason!r}")
                self.assertLessEqual(abs(res.correction_degrees - (-phi)), 0.02,
                                     f"{tag}: correction={res.correction_degrees!r}")
                # Real angular error: |correction + phi| against the analytic -phi.
                e0 = abs(0.0 + phi)
                e1 = _angle_error(res, phi)
                print(f"    angle error initial={e0:.6f}deg refined={e1:.6f}deg "
                      f"ratio={e1 / e0:.3f}")
                self.assertLessEqual(e1, 0.5 * e0,
                                     f"{tag}: angle error {e1:.6f} not <= half of {e0:.6f}")

    def test_centroid_does_not_drift_is_only_a_geometric_invariant(self):
        # Symmetric stripes: the averaged peak centre is ~x0 at ANY correction,
        # so this invariant cannot and must not stand in for tilt accuracy.
        img = _stripes(0.147)
        for correction in (0.0, -0.147, 0.147, -0.4):
            self.assertLessEqual(_mean_abs_dev_error(img, 0.147, correction), 1.0)
        res = refine_projection_tilt(img, 0.0)
        self.assertLessEqual(_mean_abs_dev_error(img, 0.147, res.correction_degrees), 1.0)
        # Accuracy is only the angle criterion.
        self.assertLessEqual(_angle_error(res, 0.147), 0.02)

    def test_exact_vertical_keeps_zero(self):
        img = _stripes(0.0)
        res = _report("vertical", 0.0, 0.0, refine_projection_tilt(img, 0.0))
        self.assertLessEqual(abs(res.correction_degrees), 1e-12)

    def test_blank_and_single_stripe_refuse(self):
        blank = np.full((H, W), 20.0, dtype=np.float64)
        res = _report("blank", None, 0.0, refine_projection_tilt(blank, 0.0))
        self.assertIsInstance(res, ProjectionTiltResult)
        self.assertFalse(res.refined)
        self.assertEqual(res.correction_degrees, 0.0)
        self.assertTrue(res.reason.strip())
        self.assertEqual(res.angle_samples, ())
        self.assertEqual(res.score_samples, ())
        self.assertIsNone(res.score_before)
        self.assertIsNone(res.score_after)

        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        centre = 290 + (y - CY) * np.tan(np.deg2rad(0.147))
        one = 20.0 + 100.0 * np.exp(-0.5 * ((x - centre) / 3.0) ** 2)
        res = _report("single", 0.147, 0.0, refine_projection_tilt(one, 0.0))
        self.assertFalse(res.refined)
        self.assertEqual(res.correction_degrees, 0.0)
        self.assertTrue(res.reason.strip())
        _assert_samples_are_parallel_numbers(self, res)

    def test_large_phi_refuses_with_initial(self):
        for phi, initial in ((-7.147, 7.0), (7.147, -7.0)):
            img = _stripes(phi)
            res = _report(f"large phi={phi:+g}", phi, initial, refine_projection_tilt(img, initial))
            self.assertLessEqual(abs(res.correction_degrees - (-phi)), 0.03,
                                 f"phi={phi}: correction={res.correction_degrees!r} reason={res.reason!r}")

    def test_reported_samples_are_parallel_numeric_tuples(self):
        img = _stripes(0.147)
        res = refine_projection_tilt(img, 0.0)
        _assert_samples_are_parallel_numbers(self, res)
        self.assertTrue(res.angle_samples)
        self.assertEqual(len(res.angle_samples), len(res.score_samples))
        for angle, score in zip(res.angle_samples, res.score_samples):
            if np.isfinite(score):
                self.assertGreaterEqual(score, 0.0)
        self.assertTrue(any(abs(a) < 1e-12 for a in res.angle_samples))
        self.assertTrue(all(abs(a) <= REFINE_MAX_RESIDUAL_DEGREES + 1e-9
                            for a in res.angle_samples))
        # The refined residual is the reported correction minus the initial one.
        best_angle = min(zip(res.score_samples, res.angle_samples))[1]
        self.assertAlmostEqual(res.correction_degrees, best_angle, delta=1e-6)
        self.assertIsNotNone(res.score_before)
        self.assertIsNotNone(res.score_after)
        self.assertLess(res.score_after, res.score_before)

    def test_refusal_after_scoring_keeps_parallel_samples(self):
        # Blank inside a valid-size image refuses at peak selection; make a
        # smoothing single stripe wide enough to reach the scoring stage.
        img = _stripes(3.0)
        res = refine_projection_tilt(img, 0.0)
        self.assertFalse(res.refined)
        self.assertEqual(res.correction_degrees, 0.0)
        self.assertEqual(res.reason.strip(), res.reason.strip())
        _assert_samples_are_parallel_numbers(self, res)
        self.assertTrue(res.reason.strip())
        if res.angle_samples:
            self.assertGreater(len(res.angle_samples), 0)
            self.assertEqual(len(res.angle_samples), len(res.score_samples))

    def test_invalid_initial_is_refused_not_raised(self):
        img = _stripes(0.147)
        for bad in (float("nan"), float("inf"), 90.0, -46.0):
            res = refine_projection_tilt(img, bad)
            self.assertFalse(res.refined, bad)
            self.assertTrue(res.reason.strip())
        self.assertTrue(np.isnan(refine_projection_tilt(img, float("nan")).initial_degrees))


class RefineBoundaryCaseTests(unittest.TestCase):
    """Unequal-amplitude stripes (+ row ramp), and left/right antiparallel halves."""

    def test_unequal_amplitudes_with_row_ramp_stays_near_truth(self):
        # Amplitudes 180..390 (ratio 2.2:1) plus a linear row ramp: the analytic
        # lean is still recoverable, so the REQUIRED outcome is a refined result
        # within 0.02 deg of the truth (measured: err 0.001 for both polarities).
        for phi in (0.147, -0.147):
            rng = np.random.default_rng(20261127)
            img = _stripes_extra(phi, rng, noise=0.0, ramp=6.0,
                                 amplitudes=(390.0, 180.0, 300.0, 240.0, 330.0))
            original = img.copy()
            res = _report(f"unequal+ramp phi={phi:+g}", phi, 0.0,
                          refine_projection_tilt(img, 0.0))
            np.testing.assert_array_equal(img, original)
            self.assertTrue(res.refined, f"phi={phi:+g}: reason={res.reason!r}")
            self.assertLessEqual(_angle_error(res, phi), 0.02,
                                 f"phi={phi:+g}: correction={res.correction_degrees!r}")

    def test_small_noise_uniform_path_is_consistently_refined(self):
        # The measured NOISE EDGE for this stripe set is sigma ~= 0.5 counts: below
        # it the same unequal-amplitude + ramp image refines on every seed, above
        # it (sigma 1.0+) the per-peak agreement gate starts refusing.  Both
        # branches are honest; only a refined-but-wrong answer would be a failure.
        phi = 0.147
        amplitudes = (390.0, 180.0, 300.0, 240.0, 330.0)
        refined = 0
        for seed in range(4):
            img = _stripes_extra(phi, np.random.default_rng(seed), noise=0.5, ramp=6.0,
                                 amplitudes=amplitudes)
            res = refine_projection_tilt(img, 0.0)
            if res.refined:
                refined += 1
                self.assertLessEqual(_angle_error(res, phi), 0.01,
                                     f"seed {seed}: refined but inaccurate "
                                     f"({res.correction_degrees!r}, {res.reason!r})")
                self.assertAlmostEqual(res.correction_degrees, -phi, delta=0.02)
            else:
                self.assertEqual(res.correction_degrees, 0.0)
                self.assertTrue(res.reason.strip())
        self.assertEqual(refined, 4, "sigma=0.5 noise refines on every seed (measured)")

    def test_moderate_additive_noise_is_honest(self):
        # ~2 counts of additive Gaussian noise is the measured reliability edge:
        # results may refine accurately or refuse, but a refined answer must
        # never be wrong.  Counts are intensity units; this fixture makes no
        # claim about an equivalent sub-pixel position error.
        phi = 0.147
        outcomes = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            img = _stripes_extra(phi, rng, noise=2.0, ramp=6.0,
                                 amplitudes=(390.0, 180.0, 300.0, 240.0, 330.0))
            res = refine_projection_tilt(img, 0.0)
            err = _angle_error(res, phi)
            outcomes.append((seed, res.refined, round(res.correction_degrees, 4),
                             round(err, 4), res.reason))
            if res.refined:
                self.assertLessEqual(res.correction_degrees, 0.0,
                                     f"seed {seed}: refined angle has the wrong sign/reason={res.reason}")
                self.assertLessEqual(err, 0.01, f"seed {seed}: refined but inaccurate ({err:.4f} deg)")
            else:
                self.assertEqual(res.correction_degrees, 0.0)
                self.assertTrue(res.reason.strip())
                self.assertNotIn("精修", res.reason)
        print("    noise outcomes (seed, refined, correction, err, reason):")
        for row in outcomes:
            print(f"      {row}")
        self.assertEqual(len(outcomes), 6)

    def test_antiparallel_halves_are_not_reported_as_refined_accuracy(self):
        # Left half leans +s, right half -s: no single rotation aligns both halves,
        # so the tool must NOT claim a refined accuracy (measured: refused with the
        # per-peak-agreement reason for both s = 0.15 and s = 0.30).
        for s in (0.15, 0.30):
            img = _antiparallel_image(s, -s)
            res = _report(f"antiparallel {s:g}", s, 0.0, refine_projection_tilt(img, 0.0))
            per_half = [abs(res.correction_degrees + lean) for lean in (s, -s)]
            print(f"    correction={res.correction_degrees:.4f} refined={res.refined} "
                  f"errors vs halves={per_half} reason={res.reason}")
            if res.refined:
                self.assertGreaterEqual(
                    max(per_half), 0.1,
                    "antiparallel halves cannot both be corrected by one rotation; "
                    "a refined result must not appear to align both",
                )
                self.assertIn("不一致", res.reason)
            else:
                self.assertEqual(res.correction_degrees, 0.0)
                self.assertIn("不一致", res.reason)


if __name__ == "__main__":
    unittest.main()
