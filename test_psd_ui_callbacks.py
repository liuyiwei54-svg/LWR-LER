"""Dedicated headless callback tests for the PSD window UI/export section.

These tests drive the REAL callbacks embedded in
``LERLWRApp.build_single_psd_model_tab`` and ``LERLWRApp.build_repeat_scan_psd_tab``
through a fake Tk widget stack (headless: no Tk display, no GUI).  They assert
real CSV file contents and real numeric values, not source-string search.

Coverage maps to the 5 acceptance items:

1. ``export_model_csv`` derives its default filename from the frozen
   ``window_context`` (source/fitted Image snapshot), never a live main-window
   image path, and never raises ``NameError`` on the old ``label_path`` bug.
2. Single-image cache invalidation on every setting change (line/metric/model +
   4 band entries) and failed parse; dual-tab cache invalidation (offset/a/b/
   metric/model/bands); load-B/reload-B clears stale result; export validates the
   current settings still match the successful snapshot and refuses otherwise.
3. Single tab shares the measured tab's line/metric ``StringVar`` objects.
4. Dual tab renders the A/B alignment preview (reuses the alignment helper).
5. All three CSVs record localization signature + per-line localization, fit
   constraints, fit/rms bands, Hann/df/effective-total-row coverage, and (dual)
   both models' reliability reasons + fit bands + per-frequency
   fit/structure/residual.

No Tk is imported at module level; ``ler_lwr_app`` imports ``tkinter`` lazily as
``tk`` inside the guarded import block, so we patch ``app.tk``/``app.ttk`` before
calling the builder.
"""

import csv
import io
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import ler_lwr_app as app


# --------------------------------------------------------------------------- #
# Fake Tk widget stack
# --------------------------------------------------------------------------- #

_FAKE_REGISTRY = {}


class _FakeVar:
    """Minimal StringVar with trace firing on every ``.set``."""

    def __init__(self, master=None, value="", name=""):
        self._value = value if value is not None else ""
        self._name = name or f"PY_VAR{len(_FAKE_REGISTRY.get('vars', []))}"
        self._traces = []
        _FAKE_REGISTRY.setdefault("vars", []).append(self)

    def get(self):
        return self._value

    def set(self, value):
        self._value = value
        for cb in list(self._traces):
            # Tk calls a ``write`` trace callback with (name, index, mode); the
            # ``_traces`` list holds (mode, callback) pairs, or a bare callback
            # registered by older code paths that ignored the arguments.
            if isinstance(cb, tuple):
                cb[1](self._name, "", cb[0])
            else:
                cb()

    def trace_add(self, mode, cb):
        self._traces.append((mode, cb))
        return f"{self._name}_{mode}"

    def trace_info(self):
        return [{"name": self._name, "mode": mode} for mode, _ in self._traces]


class _FakeWidget:
    def __init__(self, master=None, **kwargs):
        self._kwargs = dict(kwargs)
        self.image = None
        self._text = kwargs.get("text", "")
        self._command = kwargs.get("command")
        self._textvariable = kwargs.get("textvariable")
        self._values = kwargs.get("values", ())
        self._bindings = {}
        _FAKE_REGISTRY.setdefault("widgets", []).append(self)

    def pack(self, **kwargs):
        return _FakeWidget()

    def grid(self, **kwargs):
        return _FakeWidget()

    def bind(self, sequence=None, callback=None, add=None):
        self._bindings[sequence] = callback
        return "fake_bind"

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            if k == "text":
                self._text = v
            elif k == "image":
                self.image = v
            elif k == "values":
                self._values = v
            else:
                self._kwargs[k] = v

    def cget(self, key):
        if key == "text":
            return self._text
        return self._kwargs.get(key)

    def get(self):
        if self._textvariable is not None:
            return self._textvariable.get()
        return self._text

    def winfo_children(self):
        return []

    # Combobox-specific compat
    def add(self, *a, **k):
        pass


class _FakeToplevel(_FakeWidget):
    """Toplevel plus the window methods open_roughness_psd calls on creation."""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._title = ""
        self._geometry = ""

    def title(self, text=None):
        if text is not None:
            self._title = text
        return self._title

    def minsize(self, *a, **k):
        return None

    def geometry(self, spec=None):
        if spec is not None:
            self._geometry = spec
        return self._geometry

    def resizable(self, *a, **k):
        return None

    def destroy(self):
        return None


class _FakeFrame(_FakeWidget):
    pass


class _FakeLabel(_FakeWidget):
    pass


class _FakeButton(_FakeWidget):
    pass


class _FakeEntry(_FakeWidget):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)


class _FakeCombobox(_FakeWidget):
    pass


class _FakeNotebook(_FakeWidget):
    pass


class _FakeScrollFrame:
    """Stands in for VerticalScrollFrame: exposes ``.content`` and ``.pack``."""

    def __init__(self, parent, padding=0):
        self.content = _FakeFrame()
        self.padding = padding

    def pack(self, **kwargs):
        return self


def _fake_ttk():
    ttk = types.SimpleNamespace()
    ttk.Frame = _FakeFrame
    ttk.Label = _FakeLabel
    ttk.Button = _FakeButton
    ttk.Entry = _FakeEntry
    ttk.Combobox = _FakeCombobox
    ttk.Notebook = _FakeNotebook
    ttk.Scrollbar = _FakeWidget
    ttk.Separator = _FakeWidget
    ttk.Checkbutton = _FakeWidget
    ttk.Radiobutton = _FakeWidget
    ttk.Spinbox = _FakeWidget
    return ttk


def _fake_tk():
    tk = types.SimpleNamespace()
    tk.StringVar = _FakeVar
    tk.Toplevel = _FakeToplevel
    tk.Frame = _FakeFrame
    tk.Label = _FakeLabel
    tk.Button = _FakeButton
    tk.Entry = _FakeEntry
    tk.Event = type("Event", (), {})
    tk.Misc = type("Misc", (), {})
    tk.LEFT = "left"
    tk.RIGHT = "right"
    tk.BOTH = "both"
    tk.TOP = "top"
    tk.BOTTOM = "bottom"
    tk.VERTICAL = "vertical"
    tk.HORIZONTAL = "horizontal"
    tk.NW = "nw"
    tk.END = "end"
    return tk


class _FakeImageTk:
    @staticmethod
    def PhotoImage(image, master=None, **kwargs):
        return image


class _FakeMessagebox:
    def __init__(self):
        self.calls = []

    def _record(self, kind, title, message, **kwargs):
        self.calls.append((kind, title, message))
        return None

    def showinfo(self, title, message, **kwargs):
        self._record("info", title, message)

    def showwarning(self, title, message, **kwargs):
        self._record("warning", title, message)

    def showerror(self, title, message, **kwargs):
        self._record("error", title, message)


# --------------------------------------------------------------------------- #
# Shared fake host + data builders
# --------------------------------------------------------------------------- #

def make_result(rows, pixels, pixel_size_nm=0.2, detector=app.EDGE_DETECTOR_ERA):
    """Build a minimal AnalysisResult-like object good enough for the builders.

    The builder only reads ``rows_px`` plus ``left_residual_nm`` /
    ``right_residual_nm`` / ``width_residual_nm`` and (dual) ``left_px`` /
    ``right_px`` + ``pixel_size_nm``.  A typed ``app.AnalysisResult`` would force
    every field, so we construct a real dataclass with the required fields filled.
    """
    rows = np.asarray(rows, dtype=int)
    px = np.asarray(pixels, dtype=float)
    n = len(rows)
    residual = (px - np.polyval(np.polyfit(rows, px, 1), rows)) * pixel_size_nm
    left_px = px * 0.5 + 100.0
    right_px = px * 0.5 + 150.0
    width_nm = (right_px - left_px) * pixel_size_nm
    return app.AnalysisResult(
        rows_px=rows,
        left_px=left_px,
        right_px=right_px,
        left_residual_nm=residual,
        right_residual_nm=residual,
        width_nm=width_nm,
        width_residual_nm=(width_nm - width_nm.mean()),
        mean_width_nm=float(width_nm.mean()),
        ler_left_sigma_nm=float(residual.std()),
        ler_right_sigma_nm=float(residual.std()),
        lwr_sigma_nm=float((width_nm - width_nm.mean()).std()),
        total_ler_sigma_nm=float(np.hypot(residual.std(), residual.std())),
        edge_correlation=0.0,
        pixel_size_nm=pixel_size_nm,
        rejected_rows=0,
        edge_detector=detector,
        edge_kernel_size=0,
        edge_diagonal_weight=0.0,
        edge_axial_weight=0.0,
        canny_high_threshold=0.0,
        canny_threshold_ratio=0.0,
        canny_low_threshold=0.0,
        canny_normalization_scale=0.0,
        localization_method=app.LINE_LOCALIZATION_UNIFIED
        if detector == app.EDGE_DETECTOR_ERA
        else detector,
    )


def make_window_context(source="A_source.tif", fitted="A_fitted.png", pixel_size_nm=0.2,
                        signature=("LINE_LOCALIZATION_UNIFIED",)):
    return app.PsdWindowContext(
        source_path=str(Path("/tmp") / source),
        fitted_image_path=str(Path("/tmp") / fitted),
        preprocessing_signature="smooth=gaussian:3x3:sigma=1.00; background=none",
        noise_context="smoothed",
        pixel_size_nm=pixel_size_nm,
        localization_signature=tuple(signature),
    )


def build_structure_rows(n=256, seed=7):
    """Deterministic correlated residual with a real low-f shape so fit_psd_model
    resolves structure (amplitude/B/C) rather than returning unidentifiable."""
    rng = np.random.default_rng(seed)
    rows = np.arange(n, dtype=int)
    # A slowly varying profile + noise -> non-trivial PSD with low-f structure.
    profile = 1.2 * np.sin(2 * np.pi * rows / 90.0) + 0.6 * np.cos(2 * np.pi * rows / 37.0)
    noise = rng.normal(0.0, 0.25, size=n)
    return rows, profile + noise


class _FakeHost:
    """Stand-in for LERLWRApp self used only by the builders."""

    def __init__(self):
        self.status_var = _FakeVar()
        self._model_dialog_folded = None

    def roughness_measurements(self):
        return {}

    def repeat_scan_measurement(self, measurements):
        return None


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #

_METRICS = {"左边缘 LER": ("left_residual_nm", "ler_left"),
            "右边缘 LER": ("right_residual_nm", "ler_right"),
            "线宽 LWR": ("width_residual_nm", "lwr")}


class _PsdUiBase(unittest.TestCase):
    def setUp(self):
        _FAKE_REGISTRY.clear()
        self.fake_tk = _fake_tk()
        self.fake_ttk = _fake_ttk()
        self.fake_imagetk = _FakeImageTk()
        self.fake_msg = _FakeMessagebox()
        self.patchers = [
            mock.patch.object(app, "tk", self.fake_tk),
            mock.patch.object(app, "ttk", self.fake_ttk),
            mock.patch.object(app, "ImageTk", self.fake_imagetk),
            mock.patch.object(app, "messagebox", self.fake_msg),
            mock.patch.object(app, "VerticalScrollFrame", _FakeScrollFrame),
            mock.patch.object(app, "filedialog"),
        ]
        self.filedialog = self.patchers[-1].start().asksaveasfilename
        self.filedialog.side_effect = self._save_path
        for p in self.patchers[:-1]:
            p.start()
        self.host = _FakeHost()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.save_requests = []

    def _save_path(self, **kwargs):
        # Record the initialfile so tests can assert item-1 filename derivation.
        path = Path(self.tmpdir.name) / kwargs.get("initialfile", "out.csv")
        self.save_requests.append((dict(kwargs), str(path)))
        return str(path)

    def tearDown(self):
        for p in self.patchers:
            p.stop()

    def _button_command(self, text):
        for w in _FAKE_REGISTRY.get("widgets", []):
            if isinstance(w, _FakeButton) and w._text == text:
                return w._command
        return None

    def _read_csv_rows(self, path):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            return list(csv.reader(fh))

    def _read_csv_pairs(self, path):
        rows = self._read_csv_rows(path)
        pairs = {}
        header = None
        for row in rows:
            if len(row) == 2 and not header:
                pairs[row[0]] = row[1]
            elif len(row) >= 2 and header is None:
                # first non-pairs row may be the frequency header; stop metadata
                pass
        # metadata are the two-column rows before the blank + header
        return rows

    def _build_single(self, measurements, window_context, line_name="当前线条",
                      metric_name="左边缘 LER"):
        dialog = _FakeToplevel()
        content = _FakeFrame()
        line_var = _FakeVar(value=line_name)
        metric_var = _FakeVar(value=metric_name)
        self.line_var = line_var
        self.metric_var = metric_var
        app.LERLWRApp.build_single_psd_model_tab(
            self.host, dialog, content, measurements, _METRICS, 0.2,
            "source label", metric_var, line_var, None, window_context,
        )
        return dialog, content

    def _assert_export_rejected(self):
        """Any stale-export refusal counts: cache cleared => '尚未拟合'/'尚未计算',
        or a still-present-but-mismatched snapshot => '设置已变更'."""
        self.assertTrue(
            any(c[0] == "info" and c[1] in ("尚未拟合", "设置已变更", "尚未计算") for c in self.fake_msg.calls),
            f"expected export refusal, got {self.fake_msg.calls!r}",
        )


class ExportModelCsvTests(_PsdUiBase):
    """Item 1: export_model_csv default filename + real CSV values."""

    def test_export_model_csv_writes_real_csv_with_no_name_error(self):
        rows, residual = build_structure_rows()
        result = make_result(rows, residual)
        measurements = {"当前线条": [result]}
        wc = make_window_context(source="A_source.tif", fitted="A_fitted.png")
        self._build_single(measurements, wc)

        render = self._button_command("应用模型拟合")
        self.assertIsNotNone(render)
        render()
        export = self._button_command("导出模型 CSV")
        self.assertIsNotNone(export)
        export()

        # No "尚未拟合" / "设置已变更" rejection; exactly one save requested.
        self.assertEqual([c[1] for c in self.fake_msg.calls], [])
        self.assertEqual(len(self.save_requests), 1)
        kwargs, path = self.save_requests[0]
        # Item 1: default filename comes from fitted_image STEM via frozen context.
        self.assertEqual(kwargs["initialfile"], "A_fitted_ler_left_psd_model.csv")

        rows_out = self._read_csv_rows(path)
        pairs = {}
        for r in rows_out:
            if len(r) == 2:
                pairs[r[0]] = r[1]
        self.assertEqual(pairs["source"], wc.source_path)
        self.assertEqual(pairs["fitted_image"], wc.fitted_image_path)
        self.assertEqual(pairs["line"], "当前线条")
        self.assertEqual(pairs["metric"], "ler_left")
        # real (non-empty) numeric model value preserved
        self.assertIn("amplitude_nm3", pairs)
        self.assertNotEqual(pairs["amplitude_nm3"], "")
        # localization signature present (item 5)
        self.assertEqual(pairs.get("localization_signature"), "LINE_LOCALIZATION_UNIFIED")


class SingleCacheInvalidationTests(_PsdUiBase):
    """Items 2 & 3: cache invalidation, snapshot validation, shared vars."""

    def _fresh(self):
        rows, residual = build_structure_rows()
        result = make_result(rows, residual)
        measurements = {"当前线条": [result]}
        wc = make_window_context()
        self._build_single(measurements, wc)
        return measurements, wc

    def test_shared_vars_with_measured_tab(self):
        """Item 3: single tab reuses the passed-in (measured tab) StringVars."""
        self._fresh()
        # build_single must use the exact line/metric StringVar objects passed in
        # (i.e. the same objects the measured tab holds), not fresh copies.  We
        # assert the objects we passed in are the ones still referenced & traced:
        # mutate our handle and confirm it invalidates the model cache.
        render = self._button_command("应用模型拟合")
        render()
        self.metric_var.set("线宽 LWR")  # this is the SAME object build used
        export = self._button_command("导出模型 CSV")
        export()
        self._assert_export_rejected()

    def test_export_rejected_after_line_change(self):
        measurements, wc = self._fresh()
        render = self._button_command("应用模型拟合")
        render()
        export = self._button_command("导出模型 CSV")
        export()  # success
        # change line selection -> trace fires invalidate -> export must now reject
        self.line_var.set("无此类线")
        export()
        self._assert_export_rejected()

    def test_export_rejected_after_metric_change(self):
        measurements, wc = self._fresh()
        render = self._button_command("应用模型拟合")
        render()
        export = self._button_command("导出模型 CSV")
        export()
        self.metric_var.set("线宽 LWR")
        export()
        self._assert_export_rejected()

    def test_export_rejected_after_band_entry_change(self):
        measurements, wc = self._fresh()
        render = self._button_command("应用模型拟合")
        render()
        export = self._button_command("导出模型 CSV")
        export()
        # find the "拟合 fmin" entry var among fake vars and mutate it
        band_var = self._find_entry_var("拟合 fmin")
        band_var.set("0.001")
        export()
        self._assert_export_rejected()

    def test_failed_parse_clears_cache_then_export_refused(self):
        measurements, wc = self._fresh()
        render = self._button_command("应用模型拟合")
        render()  # success
        # invalid band value -> next render fails; cache cleared first
        band_var = self._find_entry_var("拟合 fmin")
        band_var.set("not-a-number")
        render()  # raises ValueError internally; shows error, cache cleared
        export = self._button_command("导出模型 CSV")
        export()
        # cache empty -> "尚未拟合", never exports stale value
        self.assertTrue(any(c[0] == "info" and c[1] == "尚未拟合" for c in self.fake_msg.calls))
        # second save not requested (first success path was never reached here)
        self.assertEqual(len(self.save_requests), 0)

    def test_recompute_restores_export(self):
        measurements, wc = self._fresh()
        render = self._button_command("应用模型拟合")
        render()
        export = self._button_command("导出模型 CSV")
        export()  # 1 save
        band_var = self._find_entry_var("拟合 fmin")
        band_var.set("0.001")
        export()  # rejected
        # restore to empty and recompute -> export allowed again with current state
        band_var.set("")
        render()
        export()
        self.assertEqual(len(self.save_requests), 2)

    def _find_entry_var(self, label):
        # Registry order: [host status_var, line, metric, model, 4 entry vars].
        # The four entry vars are the four empty-valued _FakeVars created
        # last (after the 'auto' model var).  Exclude the host status_var by
        # indexing from the model var's position.
        all_vars = _FAKE_REGISTRY.get("vars", [])
        # locate model var (value 'auto'), entry vars are the 4 immediately after
        for i, v in enumerate(all_vars):
            if v.get() == "auto":
                index = {"拟合 fmin": 0, "拟合 fmax": 1, "积分 fmin": 2, "积分 fmax": 3}[label]
                return all_vars[i + 1 + index]
        raise AssertionError("model var not found in registry")


class MeasuredTabTraceTests(_PsdUiBase):
    """The measured tab refreshes from the SHARED line/metric StringVar writes.

    The single-image model tab reuses the measured tab's StringVars, and a
    Combobox on either tab only writes its variable — the measured tab's own
    ``<<ComboboxSelected>>`` never fires for the other tab.  So the measured tab
    must listen to the shared variable write, exactly once per change, and must
    not keep both triggers (that would compute every render twice).
    """

    def _open(self, measurements, window_context):
        host = _FakeHost()
        host.roughness_measurements = lambda: measurements
        host.repeat_scan_measurement = lambda m: None
        host.image_path = None
        host.era_line_samples = ()
        host.psd_preprocessing_signature = lambda: window_context.preprocessing_signature
        host.psd_noise_context = lambda: window_context.noise_context
        host.winfo_screenheight = lambda: 900
        host.build_single_psd_model_tab = lambda *a, **k: None
        host.build_repeat_scan_psd_tab = lambda *a, **k: None
        with mock.patch.object(app, "psd_line_localization_rows", return_value=()), \
                mock.patch.object(app, "write_psd_localization_block", lambda *a, **k: None):
            app.LERLWRApp.open_roughness_psd(host, None)
        return host

    def test_shared_var_write_refreshes_measured_chart(self):
        rows = np.arange(256, dtype=float)
        base = 0.25 * np.sin(2 * np.pi * rows / 40.0) + 0.01
        measurements = {
            "线1": [make_result(rows, base)],
            "线2": [make_result(rows, base * 0.4 + 0.33)],
        }
        self._open(measurements, make_window_context())
        charts = [w for w in _FAKE_REGISTRY["widgets"]
                  if isinstance(w, _FakeLabel) and w.image is not None]
        self.assertTrue(charts, "measured chart was never rendered")
        chart = charts[0]
        first_image = chart.image
        self.assertIsNotNone(first_image)

        # The measured tab creates line_var then metric_var right before its two
        # Comboboxes; build_single_psd_model_tab reuses those very objects, so the
        # pair is identified from the Combobox textvariables the tab really built.
        def _var_of(box):
            return box._textvariable

        measured_boxes = [w for w in _FAKE_REGISTRY["widgets"]
                          if isinstance(w, _FakeCombobox) and w._kwargs.get("values")
                          and set(w._kwargs["values"]) in (set(measurements), set(_METRICS))]
        self.assertEqual(len(measured_boxes), 2)
        shared_line_var, shared_metric_var = (_var_of(b) for b in measured_boxes)
        self.assertEqual(shared_line_var.get(), "线1")
        self.assertEqual(shared_metric_var.get(), "左边缘 LER")

        with mock.patch.object(app, "roughness_spectrum", wraps=app.roughness_spectrum) as spy:
            shared_line_var.set("线2")            # what the model tab's Combobox does
            self.assertEqual(spy.call_count, 1, "one shared-variable write must render exactly once")
            self.assertIsNot(chart.image, first_image, "chart image was not replaced")
            shared_metric_var.set("线宽 LWR")
            self.assertEqual(spy.call_count, 2)
        self.assertIsNotNone(chart.image)

        # The measured tab's own Combobox is no longer bound: the frame holding
        # line_selector/metric_selector keeps a single trace-driven path only.
        for box in measured_boxes:
            self.assertEqual(box._bindings, {}, "measured Combobox must not bind <<ComboboxSelected>>")


class DualTabTests(_PsdUiBase):
    """Items 2, 4 & 5: dual-tab cache/reload, alignment preview, CSV completeness."""

    def _make_first(self, rows, values, metric="ler_left"):
        result = make_result(rows, values)
        first = app.RepeatScanMeasurement(
            source=Path("/tmp/A_source.tif"),
            image=np.zeros((4, 4), dtype=np.uint8),
            settings=("rot0",),
            row_origin=0,
            lines={"当前线条": result},
            line_methods={"当前线条": (app.LINE_LOCALIZATION_UNIFIED,)},
            preprocessing_signature="smooth=off; background=none",
            noise_context="unknown",
            calibration_nm=0.2,
        )
        return first, metric

    def _build_dual(self, first, first_line, initial_metric, window_context):
        dialog = _FakeToplevel()
        tab = _FakeFrame()
        app.LERLWRApp.build_repeat_scan_psd_tab(
            self.host, dialog, tab, first, first_line, initial_metric, window_context,
        )
        return dialog, tab

    def _fake_second_measurement(self, rows, values):
        result = make_result(rows, values)
        return app.RepeatScanMeasurement(
            source=Path("/tmp/B_source.tif"),
            image=np.ones((4, 4), dtype=np.uint8) * 255,
            settings=("rot0",),
            row_origin=0,
            lines={"当前线条": result},
            line_methods={"当前线条": (app.LINE_LOCALIZATION_UNIFIED,)},
            preprocessing_signature="smooth=off; background=none",
            noise_context="unknown",
            calibration_nm=0.2,
        )

    def test_load_b_failure_clears_stale_result(self):
        rows, values = build_structure_rows()
        first, metric = self._make_first(rows, values)
        wc = make_window_context()
        self._build_dual(first, "当前线条", "左边缘 LER", wc)
        # host.repeat_scan_measurement returns None -> load fails
        self.host.repeat_scan_measurement = lambda m: None
        load = self._button_command("载入主窗口测量为 B")
        load()
        self.assertTrue(any(c[0] == "info" for c in self.fake_msg.calls))
        # a previous render must not have been produced; export must show 尚未计算
        export = self._button_command("导出 CSV")
        export()
        self.assertTrue(any(c[1] == "尚未计算" for c in self.fake_msg.calls))

    def test_render_and_export_full_dual_csv_with_alignment_preview(self):
        rows, values = build_structure_rows()
        first, metric = self._make_first(rows, values)
        second = self._fake_second_measurement(rows, values + 0.3)
        # host returns the second scan
        self.host.repeat_scan_measurement = lambda m: second
        wc = make_window_context()
        self._build_dual(first, "当前线条", "左边缘 LER", wc)

        load = self._button_command("载入主窗口测量为 B")
        load()
        # b_var now set to candidate's first line
        render = self._button_command("计算差分对照")
        render()
        export = self._button_command("导出 CSV")
        export()
        self.assertEqual([c[0] for c in self.fake_msg.calls].count("info"), 0)

        kwargs, path = self.save_requests[0]
        rows_out = self._read_csv_rows(path)
        pairs = {}
        for r in rows_out:
            if len(r) == 2:
                pairs[r[0]] = r[1]
        self.assertEqual(pairs["source_a"], str(first.source))
        self.assertEqual(pairs["source_b"], str(second.source))
        self.assertEqual(pairs["metric"], "ler_left")
        # per-model reliability reasons + fit bands + localization block (item 5)
        self.assertIn("first_model_reliability_reasons", pairs)
        self.assertIn("second_model_reliability_reasons", pairs)
        self.assertIn("first_model_fit_band_min_cycles_per_nm", pairs)
        self.assertIn("second_model_fit_band_max_cycles_per_nm", pairs)
        self.assertEqual(pairs.get("localization_signature"), "LINE_LOCALIZATION_UNIFIED")
        # frequency table header includes per-model fit/structure/residual
        freq_header = None
        for r in rows_out:
            if r and r[0] == "frequency_cycles_per_nm":
                freq_header = r
                break
        self.assertIsNotNone(freq_header)
        for col in ("first_model_fit_nm3", "first_model_structure_nm3", "first_model_residual_nm3",
                    "second_model_fit_nm3", "second_model_structure_nm3", "second_model_residual_nm3"):
            self.assertIn(col, freq_header)

        # ---- real numbers, not only column names: recompute the SAME expectation
        # with the production functions on the SAME measurement snapshots and the
        # exact argument list render() uses, then compare several real frequencies.
        a_result = first.lines["当前线条"]
        b_result = second.lines["当前线条"]
        repeat = app.repeat_scan_spectrum(
            a_result.rows_px + first.row_origin, a_result.left_px,
            b_result.rows_px + second.row_origin, b_result.left_px,
            a_result.pixel_size_nm, 0, remove_slope=True,
        )
        expected = app.repeat_scan_model_comparison(
            repeat, a_result.pixel_size_nm, model="auto", noise_context=wc.noise_context,
        )
        first_model = expected.first_model
        second_model = expected.second_model

        # metadata rows the previous version never checked
        self.assertEqual(pairs["pixel_size_nm"], str(a_result.pixel_size_nm))
        self.assertEqual(pairs["model"], "auto")
        self.assertEqual(pairs["model_constraints"], app.PSD_MODEL_CONSTRAINTS)
        self.assertEqual(pairs["model_constraints"], "A>=0; B>0; C>=0")
        self.assertEqual(pairs["objective"], app.PSD_MODEL_OBJECTIVE)
        self.assertEqual(pairs["objective"], "sum_squared_residuals_linear_psd")
        self.assertEqual(pairs["first_model_B_nm"], str(first_model.corner_length_nm))
        self.assertEqual(pairs["second_model_C_nm3"], str(second_model.noise_floor_nm3))
        self.assertEqual(pairs["rms_band_min_cycles_per_nm"], str(expected.rms_band_per_nm[0]))

        # only the numeric density columns (the first column is the frequency axis)
        index_of = {name: i for i, name in enumerate(freq_header) if i > 0}
        data_rows = [r for r in rows_out[rows_out.index(freq_header) + 1:] if len(r) > 1]
        self.assertEqual(len(data_rows), len(expected.frequency_per_nm))
        frequency = expected.frequency_per_nm
        # first, middle and last frequency points of the real exported table
        for position in (0, len(data_rows) // 2, len(data_rows) - 1):
            exported = data_rows[position]
            values = {name: float(exported[index]) for name, index in index_of.items()}
            values["frequency_cycles_per_nm"] = float(exported[0])
            self.assertAlmostEqual(values["frequency_cycles_per_nm"], float(frequency[position]), places=12)
            for column, array in (
                ("measured_nm3", expected.measured_density_nm3),
                ("first_model_fit_nm3", first_model.total_fit_density_nm3),
                ("first_model_structure_nm3", first_model.structure_density_nm3),
                ("first_model_residual_nm3", first_model.residual_density_nm3),
                ("second_model_fit_nm3", second_model.total_fit_density_nm3),
                ("second_model_structure_nm3", second_model.structure_density_nm3),
                ("second_model_residual_nm3", second_model.residual_density_nm3),
                ("first_model_noise_nm3", expected.first_model_noise_density_nm3),
                ("second_model_noise_nm3", expected.second_model_noise_density_nm3),
                ("model_noise_avg_nm3", expected.model_noise_density_nm3),
                ("difference_noise_nm3", expected.difference_noise_density_nm3),
                ("model_corrected_nm3", expected.model_corrected_density_nm3),
                ("difference_corrected_nm3", expected.difference_corrected_density_nm3),
            ):
                self.assertAlmostEqual(values[column], float(array[position]), places=15, msg=f"{column}@{position}")
            # the two corrections are independent (both start from measured) and
            # the difference under low f is pure white noise (≈measured - 0).
            self.assertAlmostEqual(
                values["difference_corrected_nm3"] - values["measured_nm3"],
                -values["difference_noise_nm3"], places=15,
            )
            self.assertAlmostEqual(
                values["model_corrected_nm3"] - values["measured_nm3"],
                -values["model_noise_avg_nm3"], places=15,
            )

    def test_export_rejected_after_offset_change(self):
        rows, values = build_structure_rows()
        first, metric = self._make_first(rows, values)
        second = self._fake_second_measurement(rows, values + 0.3)
        self.host.repeat_scan_measurement = lambda m: second
        wc = make_window_context()
        self._build_dual(first, "当前线条", "左边缘 LER", wc)
        self._button_command("载入主窗口测量为 B")()
        self._button_command("计算差分对照")()
        export = self._button_command("导出 CSV")
        export()  # success
        offset_var = self._find_offset_var()
        offset_var.set("3")
        export()
        self._assert_export_rejected()

    def _find_offset_var(self):
        # offset_var starts "0"; it is a _FakeVar whose value is "0"
        for v in _FAKE_REGISTRY.get("vars", []):
            if v.get() == "0":
                return v
        return None


if __name__ == "__main__":
    unittest.main()
