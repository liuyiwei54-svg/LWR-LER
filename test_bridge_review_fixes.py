"""Regression tests for the four supervisor-found UI/parameter bugs.

Covers, with REAL localization (no patching out the data fetch):
1. ``analyze_bright_band_edge`` now honours a caller-supplied 2/3/4 ``degree`` and
   records it in ``polynomial_degree``/diagnostics/extra_metadata (direct call still
   defaults to 3).
2. ``open_profile_review_window`` accepts an optional frozen ``context`` and never
   reads live ``raw_image``/``processed_image`` when one is supplied.
3. ``profile_review_marker_for_row`` / ``_refresh_profile_review_content`` honour a
   structured (0-based sample position, side) target, so the 2nd sample's right side
   (different x, different row span) is the one actually localized — not the first.
4. ``era_edge_position`` is a thin wrapper over ``transition_row_position`` and no
   longer accepts a double equal-height step as a valid edge.
"""

import functools
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import ler_lwr_app as app
import line_diagnostics as ldiag

# Real value-container doubles from the shared headless harness (same as other suites).
from test_auto_era_workflow import Value, headless_app, stripe_image


def smooth_band_image(height=160, width=240, left_edge=100.0, right_edge=150.0):
    """A bright vertical band with sigmoid flanks on a dark background."""
    y, x = np.mgrid[:height, :width]
    image = np.full(x.shape, 40.0)
    image += 170.0 * ((x >= left_edge) & (x < right_edge))
    step = 3.0
    for edge_x in (left_edge, right_edge):
        rising = 1.0 / (1.0 + np.exp(-(x - edge_x) / step))
        image = np.where((x >= edge_x - 25) & (x <= edge_x + 25), 40.0 + 170.0 * rising, image)
    return np.clip(image, 0, 255).astype(np.uint8)


def two_flank_band_image(height=160, width=240, left_edge=100.0, right_edge=150.0):
    """Bright stripe = rising sigmoid at ``left_edge`` minus rising sigmoid at ``right_edge``.

    Every physical flank is a real ~2 px ascent/descent, so each side carries an
    unambiguous bright->dark / dark->bright transition for the real ERA localization.
    """
    _y, x = np.mgrid[:height, :width]
    rise = 1.0 / (1.0 + np.exp(-(x - left_edge) / 2.0))
    fall = 1.0 / (1.0 + np.exp(-(x - right_edge) / 2.0))
    return np.clip(30.0 + 200.0 * (rise - fall), 0, 255).astype(np.uint8)


def _box_bounds(left, top, right, bottom):
    return (left, top, right, bottom)


class BrightBandDegreeTests(unittest.TestCase):
    """Issue 1: bright-band localization honours the user degree, default 3."""

    def _image(self):
        x = np.arange(60.0)
        img = np.tile(30 + 160 * np.exp(-((x - 30) / 4) ** 2), (100, 1))
        return np.clip(img, 0, 255).astype(np.uint8)

    def test_bright_band_degree_2_3_4_recorded(self):
        img = self._image()
        bounds = (15, 0, 46, 100)
        reference = (26.0, 34.0)
        for degree in (2, 3, 4):
            edge = app.analyze_sample_edge(
                img, bounds, 0.2, app.LINE_LOCALIZATION_UNIFIED,
                bright_band=True, definition="left",
                reference_band=reference, degree=degree,
            )
            self.assertEqual(edge.polynomial_degree, degree)
            self.assertEqual(int(edge.extra_metadata.get("fit_degree")), degree)
            self.assertEqual(edge.extra_metadata.get("definition"), "left")

    def test_direct_call_default_is_3(self):
        img = self._image()
        edge = app.analyze_bright_band_edge(
            img, (15, 0, 46, 100), 0.2, definition="left", reference_band_px=(26.0, 34.0),
        )
        self.assertEqual(edge.polynomial_degree, 3)

    def test_direct_call_explicit_degree_keyword(self):
        img = self._image()
        edge = app.analyze_bright_band_edge(
            img, (15, 0, 46, 100), 0.2, definition="left",
            reference_band_px=(26.0, 34.0), degree=4,
        )
        self.assertEqual(edge.polynomial_degree, 4)

    def test_invalid_degree_rejected(self):
        img = self._image()
        with self.assertRaises(ValueError):
            app.analyze_bright_band_edge(
                img, (15, 0, 46, 100), 0.2, definition="left",
                reference_band_px=(26.0, 34.0), degree=5,
            )


class EraEdgePositionThinWrapperTests(unittest.TestCase):
    """Issue 4: era_edge_position delegates to transition_row_position."""

    def test_clean_transition_returns_finite_position(self):
        prof = 1.0 / (1.0 + np.exp(-(np.arange(80) - 40.0) / 2.0))
        pos = app.era_edge_position(prof, 40.0, 1, 3)
        self.assertTrue(np.isfinite(pos))
        self.assertGreaterEqual(pos, 35.0)
        self.assertLessEqual(pos, 45.0)

    def test_double_equal_height_step_is_not_accepted(self):
        # Rise at 20, fall at 50: an equal-height plateau (the old 17.5 midpoint bug).
        prof = np.zeros(100)
        prof[20:50] = 1.0
        pos = app.era_edge_position(prof, 35.0, 1, 3)
        self.assertTrue(np.isnan(pos))


class StructuredTargetMarkerTests(unittest.TestCase):
    """Issue 3: the structured (0-based position, side) target picks the RIGHT side."""

    def _sample(self, sample_id, left_bounds, right_bounds, source="manual"):
        return app.EraLineSample(
            left_roi_bounds_px=left_bounds,
            right_roi_bounds_px=right_bounds,
            sample_id=sample_id,
            selection_source=source,
        )

    def test_two_samples_same_top_right_side_second_sample(self):
        image = smooth_band_image()
        # Both samples share top row 10; second sample's right ROI is shifted in x and
        # has a different bottom, so a "top-row reverse lookup" would mis-target.
        s1 = self._sample(1, (80, 10, 110, 90), (140, 10, 170, 90))
        s2 = self._sample(2, (80, 10, 110, 150), (160, 10, 190, 150))
        samples = [s1, s2]
        # Second sample, RIGHT side, at a row inside its wider span.
        row = 130
        marker, note = app.profile_review_marker_for_row(
            row, image, samples, 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
            selected_sample=1, selected_side="right",
        )
        self.assertIsNotNone(marker, note)
        self.assertIn("2", note)
        self.assertIn("右", note)
        # The right side of sample 2 is at x ~160..190 — clearly distinct from s1 right
        # (~140..170) and s2 left (~80..110).
        self.assertGreaterEqual(marker, 160.0)
        self.assertLessEqual(marker, 190.0)

    def test_first_sample_left_is_not_confused(self):
        image = smooth_band_image()
        s1 = self._sample(1, (80, 10, 110, 90), (140, 10, 170, 90))
        s2 = self._sample(2, (80, 10, 110, 150), (160, 10, 190, 150))
        marker, note = app.profile_review_marker_for_row(
            50, image, [s1, s2], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
            selected_sample=0, selected_side="left",
        )
        self.assertIsNotNone(marker, note)
        self.assertIn("1", note)
        self.assertIn("左", note)
        self.assertGreaterEqual(marker, 80.0)
        self.assertLessEqual(marker, 110.0)

    def test_out_of_bounds_selected_sample_reports_reason(self):
        image = smooth_band_image()
        s1 = self._sample(1, (80, 10, 110, 90), (140, 10, 170, 90))
        marker, note = app.profile_review_marker_for_row(
            50, image, [s1], 0.2, selected_sample=5, selected_side="left",
        )
        self.assertIsNone(marker)
        self.assertIn("越界", note)

    def test_no_selection_keeps_first_match_behaviour(self):
        # Backward compatibility: no selected target -> first sample/side containing row.
        image = smooth_band_image()
        s1 = self._sample(1, (80, 10, 110, 90), (140, 10, 170, 90))
        s2 = self._sample(2, (80, 10, 110, 150), (160, 10, 190, 150))
        marker, note = app.profile_review_marker_for_row(
            50, image, [s1, s2], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
        )
        self.assertIsNotNone(marker, note)
        self.assertIn("1", note)
        self.assertIn("左", note)


class _FakePhoto:
    """Stand-in for ``ImageTk.PhotoImage`` recording the PIL image it wrapped."""

    def __init__(self, image, master=None):
        self.image = image
        self.master = master


class _FakePhoto:
    """Stand-in for ``ImageTk.PhotoImage`` recording the PIL image it wrapped."""

    def __init__(self, image, master=None):
        self.image = image
        self.master = master


class _FakeWidget:
    """Generic Tk widget double.

    Beyond ``pack``/``configure`` it really implements the small protocol the production
    windows use to drive their own controls: ``textvariable``-backed ``get``/``set``,
    Combobox ``values``/``current``/``set`` and ``bind`` (so the real
    ``<<ComboboxSelected>>`` / ``<Return>`` callbacks can be invoked by a test).
    """

    def __init__(self, master=None, *args, **kwargs):
        self.master = master
        self.options = dict(kwargs)
        self.text = kwargs.get("text")
        self.textvariable = kwargs.get("textvariable")
        self.image = kwargs.get("image")
        self.values = list(kwargs.get("values") or ())
        self.bindings: dict = {}
        self._index = -1
        self._value = ""

    def pack(self, *args, **kwargs):
        return None

    def pack_forget(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def grid_forget(self, *args, **kwargs):
        return None

    def configure(self, *args, **kwargs):
        self.options.update(kwargs)
        if "values" in kwargs:
            self.values = list(kwargs["values"])
        if "image" in kwargs:
            self.image = kwargs["image"]
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "textvariable" in kwargs:
            self.textvariable = kwargs["textvariable"]
        return None

    config = configure

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func
        return None

    def destroy(self):
        return None

    def columnconfigure(self, *args, **kwargs):
        return None

    def rowconfigure(self, *args, **kwargs):
        return None

    def yview(self, *args, **kwargs):
        return None

    def xview(self, *args, **kwargs):
        return None

    # read-only geometry / focus helpers queried by production code
    def winfo_children(self):
        return ()

    def winfo_width(self):
        return 800

    def winfo_height(self):
        return 600

    def winfo_reqwidth(self):
        return 800

    def winfo_reqheight(self):
        return 600

    def winfo_screenheight(self):
        return 900

    def focus_set(self, *args, **kwargs):
        return None

    def focus_force(self, *args, **kwargs):
        return None

    def see(self, *args, **kwargs):
        return None

    def after(self, *args, **kwargs):
        return None

    def update_idletasks(self, *args, **kwargs):
        return None

    # text / var / combobox / entry protocol
    def get(self):
        if self.textvariable is not None:
            return self.textvariable.get()
        return self._value

    def set(self, value):
        if self.textvariable is not None:
            self.textvariable.set(value)
        if value in self.values:
            self._index = self.values.index(value)
        self._value = value
        return None

    def current(self, index=None):
        if index is None:
            return self._index
        self._index = int(index)
        self._value = self.values[self._index] if self.values else ""
        if self.textvariable is not None:
            self.textvariable.set(self._value)
        return None


class _FakeToplevel(_FakeWidget):
    instances: list = []

    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.titles: list = []
        _FakeToplevel.instances.append(self)

    def title(self, value=None):
        if value is None:
            return self.titles[-1] if self.titles else ""
        self.titles.append(value)
        return None

    def transient(self, *args, **kwargs):
        return None

    def minsize(self, *args, **kwargs):
        return None

    def geometry(self, value=None):
        return None

    def resizable(self, *args, **kwargs):
        return None

    def attributes(self, *args, **kwargs):
        return None

    def grab_set(self, *args, **kwargs):
        return None

    def wait_window(self, *args, **kwargs):
        return None

    def deiconify(self, *args, **kwargs):
        return None


class _FakeVar:
    def __init__(self, master=None, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeTreeview(_FakeWidget):
    """Treeview double with real selection semantics (drives ``<<TreeviewSelect>>``)."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows: dict = {}
        self.order: list = []
        self._selected: tuple = ()
        _FakeTreeview.instances.append(self)

    def heading(self, *args, **kwargs):
        return None

    def column(self, *args, **kwargs):
        return None

    def delete(self, *ids):
        self.rows, self.order = {}, []

    def insert(self, parent, index, values=(), iid=None):
        self.rows[str(iid)] = tuple(values)
        self.order.append(str(iid))

    def get_children(self):
        return tuple(self.order)

    def selection(self):
        return self._selected

    def selection_set(self, ids):
        self._selected = (ids,) if isinstance(ids, str) else tuple(ids)
        handler = self.bindings.get("<<TreeviewSelect>>")
        if handler is not None:
            handler(None)


class _FakeScrollFrame(_FakeWidget):
    """Stand-in for ``VerticalScrollFrame``: window container exposing ``.content``."""

    def __init__(self, parent=None, padding=0):
        super().__init__(parent, padding=padding)
        self.content = _FakeWidget(parent)


_AXIS_CONSTANTS = (
    "BOTH", "X", "Y", "LEFT", "RIGHT", "TOP", "BOTTOM", "END", "VERTICAL", "HORIZONTAL",
    "NW", "N", "S", "E", "W", "WORD", "HIDDEN", "DISABLED", "NORMAL", "TRUE", "FALSE",
    "OPEN", "ANCHOR", "CENTER", "BROWSE", "SINGLE", "EXTENDED", "MULTIPLE",
)


def _fake_tk_module():
    names = {name: name.lower() for name in _AXIS_CONSTANTS}
    return SimpleNamespace(
        Toplevel=_FakeToplevel, Frame=_FakeWidget, Canvas=_FakeWidget, Label=_FakeWidget,
        Entry=_FakeWidget, Button=_FakeWidget, StringVar=_FakeVar, IntVar=_FakeVar,
        DoubleVar=_FakeVar, BooleanVar=_FakeVar, Listbox=_FakeWidget, Text=_FakeWidget,
        Menu=_FakeWidget, Scrollbar=_FakeWidget, **names,
    )


def _fake_ttk_module():
    return SimpleNamespace(
        Frame=_FakeWidget, Label=_FakeWidget, Button=_FakeWidget, Entry=_FakeWidget,
        Combobox=_FakeWidget, Treeview=_FakeTreeview, Scrollbar=_FakeWidget,
        Notebook=_FakeWidget, Labelframe=_FakeWidget, Checkbutton=_FakeWidget,
    )


class _FakeTkMixin:
    """Install the fake Tk layer for the duration of one test.

    Only Tk widgets / PhotoImage / the scroll container are replaced.  ``app.tk``,
    ``app.ttk``, ``app.ImageTk`` and ``ldiag``'s copies are patched in the modules where
    the production methods resolve them, and the host namespace carries the same fakes.
    Every production method, ``profile_review_marker_for_row`` and the real
    image/preprocess path run unpatched.
    """

    def install_fake_tk(self):
        tk_fake = _fake_tk_module()
        ttk_fake = _fake_ttk_module()
        image_tk_fake = SimpleNamespace(PhotoImage=_FakePhoto)
        patchers = [
            patch.object(app, "tk", tk_fake),
            patch.object(app, "ttk", ttk_fake),
            patch.object(app, "ImageTk", image_tk_fake),
            patch.object(app, "VerticalScrollFrame", _FakeScrollFrame),
            patch.object(ldiag, "tk", tk_fake),
            patch.object(ldiag, "ttk", ttk_fake),
            patch.object(ldiag, "ImageTk", image_tk_fake),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.host.tk = tk_fake
        self.host.ttk = ttk_fake
        self.host.ImageTk = image_tk_fake
        return tk_fake, ttk_fake


class _SpyRefreshMixin:
    """Wrap the REAL ``_refresh_profile_review_content`` to record its arguments.

    The wrapper calls straight through (``functools.wraps`` keeps name/signature), so
    production rendering, the real ``profile_review_marker_for_row`` call and the real
    image/preprocess path still execute — only the call arguments are observed.
    """

    def install_refresh_spy(self, host):
        records: list = []
        original = app.LERLWRApp._refresh_profile_review_content

        @functools.wraps(original)
        def spy(inner_self, dialog, content, photos, context, row,
                selected_sample=None, selected_side=None):
            records.append({
                "context": context, "row": row,
                "selected_sample": selected_sample, "selected_side": selected_side,
                "dialog": dialog,
            })
            return original(
                inner_self, dialog, content, photos, context, row,
                selected_sample=selected_sample, selected_side=selected_side,
            )

        host._refresh_profile_review_content = MethodType(spy, host)
        return records


TWO_SAMPLE_ROIS = (
    # sample 1: rows 10..89; sample 2: rows 10..149 (same top row, different bottom).
    # The right ROI must START just outside the bright band (x=150 rising sigmoid), else
    # the column profile is flat inside the band and the real localization rejects it.
    dict(left=(80, 10, 110, 90), right=(140, 10, 170, 90)),
    dict(left=(80, 10, 110, 150), right=(158, 10, 185, 150)),
)


class _ReviewWindowHarness(_FakeTkMixin, _SpyRefreshMixin, unittest.TestCase):
    """Shared host/image/samples for the real ``open_profile_review_window`` tests."""

    def setUp(self):
        _FakeToplevel.instances = []
        _FakeTreeview.instances = []
        self.image = two_flank_band_image()
        self.host = headless_app(self.image.copy())
        self.host.image_path = Path("/tmp/A.png")
        self.host.winfo_screenheight = lambda: 900
        # Real production helpers the review window calls (NOT patched): the two-route
        # image construction from the frozen context and the whole-window refresh.
        for name in (
            "_review_context_images", "_refresh_profile_review_content",
            # Real source-info chain the diagnostic window freezes at open time.
            "line_diagnostics_source_info", "is_same_live_source",
            "psd_preprocessing_signature", "psd_noise_context",
            # Real local-context capture: the diagnostic window freezes A through it.
            "_profile_review_context", "open_profile_review_window",
        ):
            setattr(self.host, name, MethodType(getattr(app.LERLWRApp, name), self.host))
        self.install_fake_tk()

    def _host_with_samples(self):
        samples = [
            app.EraLineSample(
                left_roi_bounds_px=rois["left"], right_roi_bounds_px=rois["right"], sample_id=index + 1,
            )
            for index, rois in enumerate(TWO_SAMPLE_ROIS)
        ]
        self.host.era_line_samples = samples
        return samples

    def _clear_live_view(self):
        """Main view switched to B / cleared: no live raw, no processed, no samples."""
        self.host.raw_image = None
        self.host.processed_image = None
        self.host.image_path = None
        self.host.era_line_samples = []

    def _forbid_live_context(self):
        calls: list = []

        def forbidden():
            calls.append(1)
            raise AssertionError("live _profile_review_context() must not run with a frozen context")

        self.host._profile_review_context = forbidden
        return calls

    def _open(self, frozen, **preset):
        return app.LERLWRApp.open_profile_review_window(self.host, context=frozen, **preset)

    def install_widget_capture(self, host):
        """Record the row entry + sample dropdown the production window really built.

        A ``ttk.Entry`` with a ``textvariable`` is the scan-row field; a ``Combobox``
        whose values start with ``样本`` is the structured sample/side selector.
        """
        captured: dict = {}
        original_entry = _FakeWidget.__init__

        def recording_entry(self, master=None, *args, **kwargs):
            original_entry(self, master, *args, **kwargs)
            if kwargs.get("textvariable") is not None and "row_entry" not in captured:
                captured["row_entry"] = self
            if self.values and str(self.values[0]).startswith("样本") and "sample_selector" not in captured:
                captured["sample_selector"] = self

        patcher = patch.object(_FakeWidget, "__init__", recording_entry)
        patcher.start()
        self.addCleanup(patcher.stop)
        return captured


class ReviewWindowCallbackTests(_ReviewWindowHarness):
    """Issue 2, real wiring: production window builder fed a frozen A context."""

    def test_frozen_context_builds_with_both_live_images_none_and_never_reads_live(self):
        self._host_with_samples()
        frozen = app.LERLWRApp._profile_review_context(self.host)
        self.assertIsNotNone(frozen.image)
        self._clear_live_view()
        live_calls = self._forbid_live_context()
        records = self.install_refresh_spy(self.host)

        with patch.object(app.messagebox, "showinfo") as showinfo, \
             patch.object(app.messagebox, "showwarning") as showwarning, \
             patch.object(app.messagebox, "showerror") as showerror:
            self._open(frozen, preset_sample_index=1, preset_side="right", preset_row=130)

        # The old guard ("请先打开一张 SEM 图像") must NOT fire: raw/processed are None but
        # the frozen context carries A, so the window is really built.
        showinfo.assert_not_called()
        showwarning.assert_not_called()
        showerror.assert_not_called()
        self.assertEqual(live_calls, [])
        self.assertEqual(len(_FakeToplevel.instances), 1)
        self.assertEqual(_FakeToplevel.instances[0].titles, ["灰度 / 灰度差对照"])
        # The production refresh really ran, more than the opening row-0 render.
        self.assertGreaterEqual(len(records), 2)
        for record in records:
            self.assertIs(record["context"], frozen)
            np.testing.assert_array_equal(record["context"].image, self.image)
            self.assertEqual(record["context"].source_label, "/tmp/A.png")
            self.assertEqual([snapshot.sample_id for snapshot in record["context"].samples], [1, 2])
            self.assertEqual(record["context"].pixel_size_nm, 0.2)
        # Preset target reached the real refresh as (0-based sample 1 → 样本 2, right, row 130).
        last = records[-1]
        self.assertEqual((last["selected_sample"], last["selected_side"], last["row"]), (1, "right", 130))
        self.assertIn((1, "right", 130), [(r["selected_sample"], r["selected_side"], r["row"]) for r in records])
        # A render without a target happened first (row 0, no fake localization claim).
        self.assertEqual((records[0]["selected_sample"], records[0]["selected_side"], records[0]["row"]), (None, None, 0))
        # The window's own photo list was populated with real PIL images (no shared list).
        self.assertIsNot(records[-1]["dialog"], None)

    def test_preset_second_sample_right_row130_is_localized_first_sample_rejected(self):
        """Second sample's right side at row 130 is allowed; first sample's is not."""
        self._host_with_samples()
        frozen = app.LERLWRApp._profile_review_context(self.host)
        self._clear_live_view()
        records = self.install_refresh_spy(self.host)
        self._open(frozen, preset_sample_index=1, preset_side="right", preset_row=130)
        # records[0] is the untargeted opening render; only the preset render is targeted.
        self.assertGreaterEqual(len(records), 2)
        for record in records[1:]:
            self.assertEqual((record["selected_sample"], record["selected_side"]), (1, "right"))
        # Real helper: sample 2 right at row 130 localizes on its x=158..185 ROI flank.
        marker, note = app.profile_review_marker_for_row(
            130, frozen.image, list(frozen.samples), frozen.pixel_size_nm,
            polarity=frozen.polarity, polynomial_degree=frozen.polynomial_degree,
            max_jump_px=frozen.max_jump_px, selected_sample=1, selected_side="right",
        )
        self.assertIsNotNone(marker, note)
        self.assertIn("样本 2 右边", note)
        self.assertGreaterEqual(marker, 158.0)
        self.assertLessEqual(marker, 185.0)
        # Sample 1's right ROI ends at row 89 → row 130 is outside, no fabricated point.
        marker1, note1 = app.profile_review_marker_for_row(
            130, frozen.image, list(frozen.samples), frozen.pixel_size_nm,
            polarity=frozen.polarity, polynomial_degree=frozen.polynomial_degree,
            max_jump_px=frozen.max_jump_px, selected_sample=0, selected_side="right",
        )
        self.assertIsNone(marker1, note1)
        self.assertIn("样本 1 右边", note1)

    def test_dropdown_callback_picks_first_sample_then_row_130_is_rejected(self):
        """Drive the REAL ``<<ComboboxSelected>>`` + ``<Return>`` handlers of the window."""
        self._host_with_samples()
        frozen = app.LERLWRApp._profile_review_context(self.host)
        self._clear_live_view()
        records = self.install_refresh_spy(self.host)
        captured = self.install_widget_capture(self.host)
        self._open(frozen, preset_sample_index=1, preset_side="right", preset_row=130)

        selector = captured["sample_selector"]
        row_entry = captured["row_entry"]
        self.assertEqual(selector.values, [
            "样本 1 左边（行 10…89）", "样本 1 右边（行 10…89）",
            "样本 2 左边（行 10…149）", "样本 2 右边（行 10…149）",
        ])
        self.assertEqual(selector.current(), 3)  # preset selected "样本 2 右边"
        self.assertEqual(row_entry.get(), "130")

        # Choosing "样本 1 右边" (index 1) via the real binding jumps to its top row 10 and
        # localizes sample 0 / right — the second sample is NOT picked by top-row lookup.
        selector.current(1)
        selector.bindings["<<ComboboxSelected>>"](None)
        self.assertEqual((records[-1]["selected_sample"], records[-1]["selected_side"], records[-1]["row"]), (0, "right", 10))
        self.assertEqual(row_entry.get(), "10")

        # Row 130 is outside that side's real span → real range check warns, no refresh.
        row_entry.set("130")
        before = len(records)
        with patch.object(app.messagebox, "showwarning") as showwarning, \
             patch.object(app.messagebox, "showinfo") as showinfo:
            row_entry.bindings["<Return>"](None)
        showwarning.assert_called_once()
        self.assertIn("10..89", str(showwarning.call_args))
        showinfo.assert_not_called()
        self.assertEqual(len(records), before)

        # Same top row, different bottom: row 89 is the LAST row of sample 1's right side
        # (rows 10..89) while sample 2's right side really covers 10..149.  The real entry
        # handler accepts both without hitting the range check, and both localize.
        for prompt_row in ("89", "149"):
            selector.current(3)
            selector.bindings["<<ComboboxSelected>>"](None)
            row_entry.set(prompt_row)
            with patch.object(app.messagebox, "showwarning") as showwarning:
                row_entry.bindings["<Return>"](None)
            showwarning.assert_not_called()
            self.assertEqual((records[-1]["selected_sample"], records[-1]["selected_side"], records[-1]["row"]), (1, "right", int(prompt_row)))
        marker_top, note_top = app.profile_review_marker_for_row(
            89, frozen.image, list(frozen.samples), frozen.pixel_size_nm,
            polarity=frozen.polarity, polynomial_degree=frozen.polynomial_degree,
            max_jump_px=frozen.max_jump_px, selected_sample=0, selected_side="right",
        )
        self.assertIsNotNone(marker_top, note_top)
        # One row past each side's real range is rejected, not silently localized.
        for index, out_row in ((0, 90), (1, 150)):
            past, past_note = app.profile_review_marker_for_row(
                out_row, frozen.image, list(frozen.samples), frozen.pixel_size_nm,
                polarity=frozen.polarity, polynomial_degree=frozen.polynomial_degree,
                max_jump_px=frozen.max_jump_px, selected_sample=index, selected_side="right",
            )
            self.assertIsNone(past, (index, out_row))
            self.assertIn("该行无有效定位", past_note)

        # Back to "样本 2 右边" then row 130 through the real entry handler → accepted.
        selector.current(3)
        selector.bindings["<<ComboboxSelected>>"](None)
        row_entry.set("130")
        with patch.object(app.messagebox, "showwarning") as showwarning:
            row_entry.bindings["<Return>"](None)
        showwarning.assert_not_called()
        self.assertEqual((records[-1]["selected_sample"], records[-1]["selected_side"], records[-1]["row"]), (1, "right", 130))

    def test_without_selecting_anything_row130_localizes_the_first_matching_sample(self):
        """No dropdown choice: historical first-match behaviour, still the real builder."""
        self._host_with_samples()
        frozen = app.LERLWRApp._profile_review_context(self.host)
        self._clear_live_view()
        records = self.install_refresh_spy(self.host)
        captured = self.install_widget_capture(self.host)
        self._open(frozen, preset_row=130)
        self.assertEqual(captured["sample_selector"].current(), -1)  # nothing preselected
        self.assertEqual([r["selected_sample"] for r in records], [None] * len(records))
        self.assertEqual(records[-1]["row"], 130)
        marker, note = app.profile_review_marker_for_row(
            130, frozen.image, list(frozen.samples), frozen.pixel_size_nm,
            polarity=frozen.polarity, polynomial_degree=frozen.polynomial_degree,
            max_jump_px=frozen.max_jump_px,
        )
        self.assertIsNotNone(marker, note)
        self.assertIn("样本 2 左边", note)  # sample 1 right ends at 89; first hit is s2 left


class LineDiagnosticsBridgeTests(_ReviewWindowHarness):
    """Issue 2 end-to-end: real ``open_line_diagnostics_window`` → ``on_review`` → review."""

    def setUp(self):
        super().setUp()
        self.host.era_polarity_var.set(app.ERA_POLARITY_AUTO)
        self.host.restore_view_center = Mock()

    def _real_diagnostic_samples(self):
        """One completed sample A (both edges localized by the real ``run_era_analysis``)."""
        sample = app.automatic_era_line_samples(stripe_image(), True)[0]
        self.host.processed_image = stripe_image()
        self.host.raw_image = stripe_image()
        self.host.era_line_samples = [sample]
        self.host.run_era_analysis()
        return self.host.era_line_samples[0]

    def _image_b(self):
        return 255 - self.host.raw_image

    def test_on_review_from_open_diagnostics_passes_frozen_a_not_live_b(self):
        sample = self._real_diagnostic_samples()
        self.assertIsNotNone(sample.right_edge)
        captured: dict = {}
        real_builder = app.build_line_diagnostics_ui

        def capturing_builder(*args, **kwargs):
            captured["on_review"] = kwargs["on_review"]
            captured["snapshot"] = kwargs["source_snapshot"]
            return real_builder(*args, **kwargs)

        frozen_a_image = self.host.raw_image.copy()
        with patch.object(app, "build_line_diagnostics_ui", side_effect=capturing_builder):
            app.LERLWRApp.open_line_diagnostics_window(self.host)
        self.assertIn("on_review", captured)
        self.assertEqual(len(_FakeToplevel.instances), 1)
        self.assertEqual(_FakeToplevel.instances[0].titles, ["逐行定位诊断"])

        # Main view switches to B and the parameters change.
        b_image = self._image_b()
        self.host.raw_image = b_image
        self.host.processed_image = b_image
        self.host.image_path = Path("/tmp/B.png")
        self.host.pixel_size_var.set("0.5")
        self.host.era_polarity_var.set(app.ERA_POLARITY_FALLING)
        self.host.era_line_samples = []
        records = self.install_refresh_spy(self.host)

        with patch.object(app.messagebox, "showinfo") as showinfo:
            captured["on_review"](0, "right", 130)
        showinfo.assert_not_called()
        self.assertEqual(len(_FakeToplevel.instances), 2)
        self.assertEqual(_FakeToplevel.instances[1].titles, ["灰度 / 灰度差对照"])
        self.assertTrue(records)
        for record in records:
            context = record["context"]
            np.testing.assert_array_equal(context.image, frozen_a_image)  # NOT B (B = 255 - A)
            self.assertNotEqual(context.source_label, "/tmp/B.png")
            self.assertEqual(context.pixel_size_nm, 0.2)   # frozen, not live 0.5
            self.assertEqual(context.polarity, app.ERA_POLARITY_AUTO)  # frozen, not live falling
            self.assertEqual([snapshot.sample_id for snapshot in context.samples], [sample.sample_id])
            self.assertEqual(context.samples[0].right_roi_bounds_px, sample.right_roi_bounds_px)
        self.assertEqual((records[-1]["selected_sample"], records[-1]["selected_side"]), (0, "right"))
        # Row 130 is inside A's real right ROI span and really localizes on A.
        top, bottom = sample.right_roi_bounds_px[1], sample.right_roi_bounds_px[3]
        self.assertTrue(top <= 130 < bottom)
        marker, note = app.profile_review_marker_for_row(
            130, frozen_a_image, [context.samples[0]], 0.2,
            polarity=app.ERA_POLARITY_AUTO, polynomial_degree=3, max_jump_px=12.0,
            selected_sample=0, selected_side="right",
        )
        self.assertIsNotNone(marker, note)

    def test_review_from_old_diagnostics_after_main_view_cleared_still_shows_a(self):
        self._real_diagnostic_samples()
        captured: dict = {}
        real_builder = app.build_line_diagnostics_ui

        def capturing_builder(*args, **kwargs):
            captured["on_review"] = kwargs["on_review"]
            return real_builder(*args, **kwargs)

        frozen_a_image = self.host.raw_image.copy()
        with patch.object(app, "build_line_diagnostics_ui", side_effect=capturing_builder):
            app.LERLWRApp.open_line_diagnostics_window(self.host)
        self.host.raw_image = None
        self.host.processed_image = None
        self.host.image_path = None
        self.host.era_line_samples = []
        records = self.install_refresh_spy(self.host)
        with patch.object(app.messagebox, "showinfo") as showinfo:
            captured["on_review"](0, "left", 50)
        showinfo.assert_not_called()
        self.assertTrue(records)
        np.testing.assert_array_equal(records[-1]["context"].image, frozen_a_image)
        self.assertEqual((records[-1]["selected_sample"], records[-1]["selected_side"], records[-1]["row"]), (0, "left", 50))


if __name__ == "__main__":
    unittest.main()
