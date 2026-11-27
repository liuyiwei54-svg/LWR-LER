"""Production wiring for the two-stage tilt estimate (structure tensor → projection).

Everything here drives the REAL code: the real structure-tensor coarse stage, the
real ``refine_projection_tilt`` module, the real ``auto_align_roi`` window builder
and its real callbacks.  Only Tk *display* facilities are faked (shared doubles
from ``test_bridge_review_fixes``), plus the ROI accessor so no canvas drag is
required.  No algorithm is mocked, and no angle tolerance is relaxed.

Measured behaviour this suite locks in (7-arg analytic stripes, 800×600, 5 bands,
unequal amplitudes):
* float input: coarse ±0.1° grid lands within 0.03° and the refine stage halves
  the error (e.g. lean +0.370° → init -0.400° → refined -0.372°, err 0.030→0.002).
* uint8 input (the app's real dtype): the refine stage ACCEPTS on these clean
  analytic stripes — coarse is a 0.1° multiple, the refined error is <= 0.02°
  and smaller than the coarse error.  Quantisation is not by itself a refusal
  trigger; a refusal must instead be honest (initial kept exactly, reason
  stated, no "refined" claim in the text).
"""

import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import ler_lwr_app as app
from projection_tilt import ProjectionTiltResult, refine_projection_tilt

# Shared REAL value containers and the shared fake-Tk doubles (display only).
from test_auto_era_workflow import Value, headless_app
from test_bridge_review_fixes import (
    _FakePhoto,
    _FakeTkMixin,
    _FakeToplevel,
    _FakeVar,
    _FakeWidget,
    _fake_ttk_module,
    _fake_tk_module,
)
from test_projection_tilt import H, W, CY, X0S, _stripes_extra


def tilted_stripes(lean_degrees, dtype=np.float32):
    """5 unequal-amplitude stripes rotated by PIL (the app's own rotation call)."""
    upright = _stripes_extra(0.0, np.random.default_rng(0), 0.0, 0.0)
    return app.rotate_grayscale_image(upright.astype(dtype), lean_degrees)


def single_stripe_image(height=220, width=320):
    _y, x = np.mgrid[:height, :width]
    return (20.0 + 120.0 * np.exp(-0.5 * ((x - width / 2.0) / 3.0) ** 2)).astype(np.float32)


def blank_image(height=200, width=300):
    return np.full((height, width), 30.0, dtype=np.uint8)


class TwoStageEstimateTests(unittest.TestCase):
    """Real coarse tensor → real projection module → float wrapper."""

    def test_real_estimate_improves_a_non_tenth_multiple_angle(self):
        for lean in (0.214, -0.214, 0.370, -0.370):
            tilted = tilted_stripes(lean)
            before_pixels = tilted.copy()
            estimate = app.estimate_rotation_to_vertical(tilted)
            self.assertIsInstance(estimate, ProjectionTiltResult)
            # Coarse stage is a 0.1° grid; it must already be within 0.05°.
            self.assertLessEqual(abs(estimate.initial_degrees + lean), 0.05,
                                 f"lean={lean:+g}: coarse={estimate.initial_degrees!r}")
            # Non-.1-multiple truth: only the refine stage can reach it.
            self.assertTrue(estimate.refined, f"lean={lean:+g}: reason={estimate.reason!r}")
            error_initial = abs(estimate.initial_degrees + lean)
            error_final = abs(estimate.correction_degrees + lean)
            self.assertLessEqual(error_final, 0.02, f"lean={lean:+g}: {error_final!r}")
            self.assertLess(error_final, error_initial,
                            f"lean={lean:+g}: {error_final!r} is not better than {error_initial!r}")
            self.assertLessEqual(error_final, 0.5 * error_initial)
            np.testing.assert_array_equal(tilted, before_pixels)
            # The float wrapper reports exactly the dataclass correction.
            self.assertAlmostEqual(app.find_rotation_to_vertical(tilted),
                                   float(estimate.correction_degrees), places=12)

    def test_tensor_initial_and_final_are_reported_separately(self):
        lean = 0.370
        tilted = tilted_stripes(lean)
        coarse = app.find_structure_tensor_rotation(tilted)
        estimate = app.estimate_rotation_to_vertical(tilted)
        self.assertAlmostEqual(estimate.initial_degrees, coarse, places=12)
        self.assertNotAlmostEqual(estimate.correction_degrees, estimate.initial_degrees, places=6)
        # initial_degrees stays the coarse angle even after refinement.
        self.assertAlmostEqual(estimate.initial_degrees, -0.4, places=9)
        self.assertAlmostEqual(estimate.correction_degrees, -0.372, places=3)

    def test_uint8_frames_refine_past_the_coarse_grid(self):
        """The app's real uint8 dtype on clean stripes: refine accepts and is accurate.

        Coarse is a 0.1° multiple, so only the refine stage can reach these
        non-.1-multiple truths.  Required: accepted, error <= 0.02° in the
        correct direction, strictly better than the coarse error, and at least
        one input genuinely refined.
        """
        outcomes = []
        refined_count = 0
        for lean in (0.214, -0.214, 0.370, -0.370):
            tilted = tilted_stripes(lean, dtype=np.uint8)
            self.assertEqual(tilted.dtype, np.uint8)
            estimate = app.estimate_rotation_to_vertical(tilted)
            error_initial = abs(estimate.initial_degrees + lean)
            error_final = abs(estimate.correction_degrees + lean)
            outcomes.append((lean, estimate.refined, round(estimate.initial_degrees, 4),
                             round(estimate.correction_degrees, 4), round(error_final, 4)))
            # Coarse stage: a 0.1° grid step, so these truths are off-grid.
            self.assertLessEqual(error_initial, 0.05,
                                 f"lean={lean:+g}: coarse={estimate.initial_degrees!r}")
            self.assertGreater(error_initial, 0.0, f"lean={lean:+g}: truth is not on the 0.1° grid")
            if estimate.refined:
                refined_count += 1
                self.assertLessEqual(error_final, 0.02,
                                     f"lean={lean:+g}: accepted {estimate.correction_degrees!r}")
                self.assertLess(error_final, error_initial,
                                f"lean={lean:+g}: {error_final!r} not better than {error_initial!r}")
                text = app.describe_tilt_estimate(estimate)
                self.assertIn("精修", text)
                self.assertNotIn("保留初估", text)
            else:
                # A refusal must keep the coarse angle EXACTLY and state why.
                self.assertEqual(estimate.correction_degrees, estimate.initial_degrees)
                self.assertTrue(estimate.reason.strip())
                text = app.describe_tilt_estimate(estimate)
                self.assertTrue(text.startswith("保留初估"), text)
                self.assertIn(estimate.reason, text)
                self.assertNotIn("精修", text)
        print("    uint8 outcomes (lean, refined, initial, correction, err):")
        for row in outcomes:
            print(f"      {row}")
        self.assertGreaterEqual(refined_count, 1, f"no uint8 input was refined: {outcomes}")

    def test_module_refines_float_and_quantized_input_identically(self):
        """Same image, same coarse start: uint8 is not a refusal trigger."""
        lean = 0.370
        float_source = tilted_stripes(lean, dtype=np.float32)
        uint8_source = tilted_stripes(lean, dtype=np.uint8)
        float_result = refine_projection_tilt(float_source, -0.4)
        uint8_result = refine_projection_tilt(uint8_source, -0.4)
        self.assertTrue(float_result.refined, float_result.reason)
        self.assertAlmostEqual(float_result.correction_degrees, -0.372, places=3)
        self.assertTrue(uint8_result.refined, uint8_result.reason)
        self.assertLessEqual(abs(uint8_result.correction_degrees + lean), 0.02,
                             uint8_result.correction_degrees)
        self.assertAlmostEqual(uint8_result.initial_degrees, -0.4, places=12)

    def test_quantized_noisy_frames_are_either_accurate_or_honest(self):
        """Independent analytic uint8 truth (500-count banded stripes, sigma 2 counts).

        Kernel-averaged stripes stay within a 0..255 count range, a per-pixel
        Gaussian sigma of 2 counts is added, then the frame is rounded and
        clipped to uint8: no wrap-around.  The initial call is the real coarse
        estimator.  An accepted result must be accurate (correct sign, error
        <= 0.02°); a refusal must keep that coarse initial exactly with its
        reason.  At least one of the three seeds must refine.
        """
        phi = 0.147
        amplitudes = (60.0, 130.0, 80.0, 100.0, 110.0)
        sigmas = (3.0, 2.4, 3.6, 2.8, 3.2)
        baseline = 40.0
        y = np.arange(H, dtype=np.float64)[:, None]
        x = np.arange(W, dtype=np.float64)[None, :]
        outcomes = []
        accepted_count = 0
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            img = np.full((H, W), baseline, dtype=np.float64)
            # One single true tilt for every stripe: centre = x0 + (y-CY)*tan(phi).
            for x0, amp, sigma in zip(X0S, amplitudes, sigmas):
                centre = x0 + (y - CY) * np.tan(np.deg2rad(phi))
                img = img + amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
            img = img + rng.normal(0.0, 2.0, size=img.shape)
            frame = np.clip(np.rint(img), 0, 255).astype(np.uint8)
            initial = app.find_structure_tensor_rotation(frame)
            estimate = app.estimate_rotation_to_vertical(frame)
            self.assertEqual(estimate.initial_degrees, initial)
            error_initial = abs(initial + phi)
            error_final = abs(estimate.correction_degrees + phi)
            outcomes.append((seed, estimate.refined, round(initial, 4),
                             round(estimate.correction_degrees, 4), round(error_final, 4)))
            # The coarse stage is a 0.1° grid: it cannot hit 0.147° exactly.
            self.assertGreater(error_initial, 0.0, f"seed {seed}: coarse on the grid?")
            if estimate.refined:
                accepted_count += 1
                # Sign: a refused/refined value must lean against the truth.
                self.assertLess(estimate.correction_degrees, 0.0,
                                f"seed {seed}: sign, {estimate.correction_degrees!r}")
                self.assertLessEqual(error_final, 0.02,
                                     f"seed {seed}: accepted but inaccurate "
                                     f"({estimate.correction_degrees!r}, {estimate.reason!r})")
            else:
                self.assertEqual(estimate.correction_degrees, initial)
                self.assertTrue(estimate.reason.strip())
                text = app.describe_tilt_estimate(estimate)
                self.assertTrue(text.startswith("保留初估"), text)
                self.assertIn(estimate.reason, text)
        print("    uint8 sigma=2 outcomes (seed, refined, initial, correction, err):")
        for row in outcomes:
            print(f"      {row}")
        self.assertGreaterEqual(accepted_count, 1, f"no seed refined: {outcomes}")

    def test_blank_roi_still_raises_the_friendly_value_error(self):
        with self.assertRaises(ValueError) as caught:
            app.estimate_rotation_to_vertical(blank_image())
        self.assertIn("对比度不足", str(caught.exception))
        with self.assertRaises(ValueError):
            app.find_rotation_to_vertical(blank_image())

    def test_few_peaks_reaches_the_ui_text_as_a_fallback_reason(self):
        estimate = app.estimate_rotation_to_vertical(single_stripe_image())
        self.assertFalse(estimate.refined)
        self.assertIn("只找到 1 个", estimate.reason)
        text = app.describe_tilt_estimate(estimate)
        self.assertIn("保留初估", text)
        self.assertIn("只找到 1 个", text)


class TracedVar(_FakeVar):
    """Fake StringVar that really runs ``trace_add("write", ...)`` callbacks."""

    def __init__(self, master=None, value=""):
        super().__init__(master, value)
        self.traces: dict = {}

    def trace_add(self, mode, callback):
        self.traces.setdefault(mode, []).append(callback)
        return f"{mode}-{len(self.traces[mode])}"

    def set(self, value):
        super().set(value)
        for callback in self.traces.get("write", ()):
            callback(None, None, None)


def traced_tk(tk_fake):
    """Return a fake ``tk`` module whose ``StringVar`` really fires write traces."""
    tk_fake.StringVar = TracedVar
    return tk_fake


class FakePhotoImage(_FakePhoto):
    """Stand-in for the patched ``app.ImageTk.PhotoImage``; records each thumbnail."""

    instances: list = []

    def __init__(self, image=None, master=None):
        super().__init__(image, master)
        FakePhotoImage.instances.append(self)


class _RecordingWidget(_FakeWidget):
    """``_FakeWidget`` that appends every instance to one per-test list."""

    def __init__(self, container, master=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        container.append(self)


_WIDGET_CONSTRUCTORS = (
    "Frame", "Canvas", "Label", "Entry", "Button", "Listbox", "Text", "Menu",
    "Scrollbar", "Notebook", "Labelframe", "Checkbutton",
)


def _fake_tk_module_recording(container):
    """A fake ``tk`` module whose widgets record themselves into ``container``.

    ``Toplevel`` / ``StringVar`` and friends keep the shared doubles; only the
    plain widget constructors are routed to :class:`_RecordingWidget`.
    """
    tk_fake = _fake_tk_module()
    for name in _WIDGET_CONSTRUCTORS:
        tk_fake.__dict__[name] = (
            lambda master=None, *a, _name=name, **k: _RecordingWidget(container, master, *a, **k)
        )
    return tk_fake


def _recording_ttk_module(container):
    """Fake ``ttk`` whose widget classes all record into ``container``."""
    ttk_fake = _fake_ttk_module()
    for name in ("Frame", "Label", "Button", "Entry", "Notebook",
                 "Labelframe", "Checkbutton", "Scrollbar"):
        if hasattr(ttk_fake, name):
            ttk_fake.__dict__[name] = (
                lambda master=None, *a, **k: _RecordingWidget(container, master, *a, **k)
            )
    return ttk_fake


class AutoAlignRoiWindowTests(_FakeTkMixin, unittest.TestCase):
    """The REAL ``auto_align_roi`` builder + callbacks (display faked only)."""

    def setUp(self):
        _FakeToplevel.instances = []
        FakePhotoImage.instances = []
        self.image = tilted_stripes(0.214, dtype=np.float32)
        self.host = headless_app(self.image.copy())
        self.host.image_path = Path("/tmp/tilt.png")
        self.host.rotation_degrees = 0.0
        self.host.rotation_var = Value("0.000")
        # Real production methods under test, bound onto the headless host.
        for name in ("auto_align_roi", "selected_ler_lwr_region", "set_rotation_degrees"):
            setattr(self.host, name, MethodType(getattr(app.LERLWRApp, name), self.host))
        self.host.record_undo_state = Mock()
        self.host.rebuild_working_images = Mock()
        self.host.draw_image = Mock()
        self.host.current_view_center = lambda: (0.0, 0.0)
        self.host.restore_view_center = lambda center: None
        self.host.reset_zoom_control = Mock()
        self.host.clear_fft_analysis = Mock()
        self.host.clear_bcp_completion_selection = Mock()
        self.host.discard_bcp_reference_circles = Mock()
        self.host.clear_era_line_samples = Mock()
        self.host.clear_auto_line_candidates = Mock()
        self.host.clear_annotations = Mock()
        self.host._background_last_applied = (False, None)
        # The ROI accessor is the one thing faked: no canvas drag in a headless run.
        self.host.selected_ler_lwr_region = lambda: (self.host.processed_image.astype(np.float32), 0, 0, False)
        # Per-test widget recording: the production window builds its real
        # widgets through ``tk`` / ``ttk`` and every one lands in this test's list.
        # Nothing is patched at import time; each patch is undone by addCleanup.
        self.widgets: list = []
        self.install_fake_tk()
        tk_fake = traced_tk(_fake_tk_module_recording(self.widgets))
        ttk_fake = _recording_ttk_module(self.widgets)
        for attr, fake in (("tk", tk_fake), ("ttk", ttk_fake)):
            patcher = patch.object(app, attr, fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.host.tk = tk_fake
        self.host.ttk = ttk_fake
        # Production resolves ``ImageTk`` from module scope (``app.ImageTk``), NOT
        # from the host.  install_fake_tk has just replaced that attribute, so the
        # recording fake is written onto that exact object and removed afterwards.
        image_tk = app.ImageTk
        self.addCleanup(setattr, image_tk, "PhotoImage", image_tk.PhotoImage)
        image_tk.PhotoImage = FakePhotoImage

    def _open(self):
        with patch.object(app.messagebox, "showwarning") as showwarning, \
             patch.object(app.messagebox, "showinfo") as showinfo:
            self.host.auto_align_roi()
        self.assertEqual(showwarning.call_count, 0, showwarning.call_args_list)
        self.assertEqual(showinfo.call_count, 0)
        self.assertEqual(len(_FakeToplevel.instances), 1)
        return _FakeToplevel.instances[0]

    def _widgets(self, window):
        """Find the real entry / result label / buttons the window built."""
        widgets = [child for child in self.widgets]
        entry = next(w for w in widgets
                     if type(w.options.get("textvariable", None)) is TracedVar)
        var = entry.options["textvariable"]
        labels = [w for w in widgets if isinstance(w.text, str)]
        buttons = {w.text: w for w in widgets if isinstance(w.options.get("command"), object) and w.text}
        return entry, var, labels, buttons

    def test_window_shows_the_two_stage_result_and_keeps_three_decimals(self):
        window = self._open()
        entry, var, labels, buttons = self._widgets(window)
        texts = [w.text for w in labels if isinstance(w.text, str)]
        result_text = next(t for t in texts if "初估" in t)
        self.assertIn("精修", result_text)
        self.assertIn("5 条亮带", result_text)
        # The entry really carries the refined angle, not a 0.1° grid value.
        self.assertRegex(str(var.get()), r"^-0\.2\d{2}$")
        self.assertEqual(len(str(var.get()).split(".")[1]), 3)
        self.assertAlmostEqual(float(var.get()), -0.214, places=3)
        # A thumbnail preview was drawn once without touching pixels.
        preview_labels = [w for w in self.widgets if w.image is not None]
        self.assertEqual(len(preview_labels), 1)
        self.assertIsInstance(preview_labels[0].image.image, Image.Image)
        thumbnail = preview_labels[0].image.image
        self.assertLessEqual(max(thumbnail.size), 620)
        self.assertEqual(self.host.set_rotation_degrees.__self__ if False else self.host.rotation_degrees, 0.0)

    def test_adjust_keeps_three_decimals_and_never_applies(self):
        window = self._open()
        entry, var, labels, buttons = self._widgets(window)
        start = float(var.get())
        buttons["+0.1°"].options["command"]()
        self.assertAlmostEqual(float(var.get()), start + 0.1, places=9)
        self.assertEqual(len(str(var.get()).split(".")[1]), 3)
        self.assertIn("0.114", str(var.get()))
        buttons["-0.1°"].options["command"]()
        self.assertAlmostEqual(float(var.get()), start, places=9)
        self.assertIn("0.214", str(var.get()))
        # Preview only: the rotation was never applied.
        self.assertEqual(self.host.rotation_degrees, 0.0)
        self.assertEqual(self.host.record_undo_state.call_count, 0)
        self.assertEqual(self.host.rebuild_working_images.call_count, 0)

    def test_cancel_does_not_touch_the_rotation(self):
        window = self._open()
        entry, var, labels, buttons = self._widgets(window)
        buttons["取消"].options["command"]()
        self.assertEqual(self.host.rotation_degrees, 0.0)
        self.assertEqual(self.host.rotation_var.get(), "0.000")
        self.assertEqual(self.host.record_undo_state.call_count, 0)

    def test_apply_sets_the_entry_float_through_the_real_setter(self):
        window = self._open()
        entry, var, labels, buttons = self._widgets(window)
        value = float(var.get())
        with patch.object(app.messagebox, "showwarning") as showwarning:
            buttons["应用倾角校正"].options["command"]()
        self.assertEqual(showwarning.call_count, 0)
        self.assertAlmostEqual(self.host.rotation_degrees, value, places=9)
        self.assertEqual(self.host.rotation_var.get(), f"{value:.3f}")
        self.assertEqual(self.host.record_undo_state.call_count, 1)
        self.assertEqual(self.host.rebuild_working_images.call_count, 1)

    def test_refused_estimate_is_reported_as_kept_initial(self):
        # Single stripe: the tool refuses, so the window must say "保留初估".
        self.host.processed_image = single_stripe_image()
        self.host.selected_ler_lwr_region = (
            lambda: (self.host.processed_image.astype(np.float32), 0, 0, False))
        window = self._open()
        entry, var, labels, buttons = self._widgets(window)
        texts = [w.text for w in labels if isinstance(w.text, str)]
        result_text = next(t for t in texts if "初估" in t)
        self.assertIn("保留初估", result_text)
        self.assertIn("只找到 1 个", result_text)
        self.assertNotIn("精修", result_text)


if __name__ == "__main__":
    unittest.main()
