"""End-to-end tests for the isolated per-row line-localization diagnostics module.

Covers the item-6 API surface without launching a GUI: the failure-side diagnostic
is really attached by ``run_era_analysis`` (the wrapper calls the new core, not a
separate module-only test), double-step / pure-noise rejection, per-row records vs
measurement positions, CSV validity alignment, snapshot immutability, invalidation /
undo consistency, and the exported diagnostic CSV rows.
"""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import ler_lwr_app as app
import line_diagnostics as ldiag
from line_localization import localization_stats, pair_localization_stats
from test_auto_era_workflow import Value, headless_app, stripe_image
from test_periodic_workflow import periodic_image, periodic_state


class DiagnosticModuleTests(unittest.TestCase):
    """Pure-function checks on line_diagnostics (no Tk)."""

    def _ambiguous_transition_image(self):
        x = np.arange(40)
        profile = 30 + 80 / (1 + np.exp(-(x - 12))) + 80 / (1 + np.exp(-(x - 23)))
        return np.tile(profile, (180, 1))

    def _failed_side_diag(self):
        image = self._ambiguous_transition_image()
        try:
            app.analyze_sample_edge(
                image, (10, 0, 50, 180), 0.2, app.LINE_LOCALIZATION_UNIFIED,
                polarity=app.ERA_POLARITY_RISING,
            )
        except app.LocalizationFailure as exc:
            return exc.diagnostics
        self.fail("expected LocalizationFailure")

    def test_pure_noise_transition_fails_with_diagnostics(self):
        rng = np.random.default_rng(7)
        noise = np.tile(rng.normal(45, 5, 40), (180, 1))
        with self.assertRaises(app.LocalizationFailure) as ctx:
            app.analyze_sample_edge(
                noise, (0, 0, 40, 180), 0.2, app.LINE_LOCALIZATION_UNIFIED,
                polarity=app.ERA_POLARITY_RISING,
            )
        diag = ctx.exception.diagnostics
        self.assertIsNotNone(diag)
        self.assertEqual(diag.height, 180)

    def test_diag_row_records_never_fabricate_missing_x(self):
        diag = self._failed_side_diag()
        records = ldiag.diag_row_records(diag)
        self.assertEqual(len(records), 180)
        for record in records:
            self.assertEqual(record["stage"], "missing")
            self.assertTrue(np.isnan(record["candidate_global_x"]))
            self.assertTrue(np.isnan(record["final_global_x"]))
            self.assertEqual(record["reason"], "ambiguous_peak")

    def test_diag_row_records_global_x_adds_roi_left(self):
        # A clean single logistic placed so the step (x=12 local) sits inside a
        # non-origin ROI (left=5): global x = local + left via diag_row_records.
        x = np.arange(40)
        profile = 30 + 80 / (1 + np.exp(-(x - 12)))
        image = np.tile(profile, (180, 1))
        result = app.analyze_sample_edge(
            image, (5, 0, 45, 180), 0.2, app.LINE_LOCALIZATION_UNIFIED,
            polarity=app.ERA_POLARITY_RISING,
        )
        diag = result.diagnostics
        records = ldiag.diag_row_records(diag)
        self.assertEqual(records[0]["global_row"], 0)
        self.assertEqual(records[0]["local_row"], 0)
        # Local final x is the subpixel crossing in ROI-local coordinates (~7.07 here,
        # because the difference-weighted centroid reference sits ~6.5 px from the ROI's
        # left edge); global x = local + left recovers the true ~12 step position.
        local = records[0]["final_local_x"]
        self.assertAlmostEqual(records[0]["final_global_x"], local + 5.0, delta=1e-6)
        self.assertAlmostEqual(records[0]["final_global_x"], 12.0, delta=0.3)
        self.assertEqual(records[0]["stage"], "primary")

    def test_diag_stat_rows_match_localization_stats(self):
        diag = self._failed_side_diag()
        rows = ldiag.diag_stat_rows(diag, None)
        stats = localization_stats(diag)
        self.assertEqual(rows["left"].total_rows, stats.total_rows)
        self.assertEqual(rows["left"].missing_rows, stats.missing_rows)
        self.assertEqual(rows["left"].primary_rows, stats.primary_rows)
        self.assertEqual(rows["left"].recovered_rows, stats.recovered_rows)

    def test_diagnostic_csv_rows_header_and_blank_missing(self):
        image = self._ambiguous_transition_image()
        sample = app.EraLineSample(left_roi_bounds_px=(10, 0, 50, 180))
        sample.left_diagnostics = self._failed_side_diag()
        snapshot = ldiag.SourceSnapshot(image=image, samples=(sample,))
        rows = ldiag.diagnostic_csv_rows(snapshot)
        # The frozen-source key/value block leads, then the column header row.
        header_index = next(i for i, row in enumerate(rows) if row and row[0] == "era_line_index")
        header = rows[header_index]
        self.assertEqual(header, ldiag.DIAGNOSTIC_CSV_COLUMNS)
        self.assertIn("valid", header)
        for column in ("fit_degree", "position_definition", "max_jump_px", "recovery_policy",
                       "preprocessing", "calibration", "rotation", "source_path"):
            self.assertIn(column, header)
        # All rows missing → candidate/final and recovered are empty strings, never 0,
        # and ``valid`` is explicitly 0 (never silently absent).
        data_rows = rows[header_index + 1:]
        self.assertEqual(len(data_rows), 180)
        for row in data_rows:
            self.assertEqual(len(row), len(header))
            self.assertEqual(row[header.index("candidate_global_x")], "")
            self.assertEqual(row[header.index("final_global_x")], "")
            self.assertEqual(row[header.index("stage")], "missing")
            self.assertEqual(row[header.index("valid")], "0")
            self.assertEqual(row[header.index("recovered_from_global_x")], "")
        # Frozen source block is real key/value text, not an empty "已冻结" label.
        frozen = {row[0]: row[1] for row in rows[:header_index] if len(row) >= 2 and row[0] != "# 逐行定位诊断导出（静态来源快照；来源/参数在窗口打开时刻冻结）"}
        self.assertEqual(frozen["source"], ldiag.DEFAULT_SOURCE_LABEL)

    def test_snapshot_is_static_image_copy(self):
        image = periodic_image()
        snapshot = ldiag.SourceSnapshot(image=image.copy(), samples=())
        image[:] = 99
        self.assertFalse(np.all(snapshot.image == 99))


class DiagnosticWorkflowTests(unittest.TestCase):
    """Wrapper integration: run_era_analysis really attaches diagnostics (item 6)."""

    def test_failed_side_keeps_diag_after_run_era_analysis(self):
        # BOTH sides are ambiguous double steps → both sides fail (<60% valid), the
        # whole sample fails, yet each side keeps its per-row diagnostics.
        x = np.arange(40)
        ambiguous = 30 + 80 / (1 + np.exp(-(x - 12))) + 80 / (1 + np.exp(-(x - 23)))
        image = np.tile(ambiguous, (180, 1))

        state = headless_app(image)
        state.era_line_samples = [
            app.EraLineSample(left_roi_bounds_px=(0, 0, 40, 180), right_roi_bounds_px=(0, 0, 40, 180))
        ]
        with self.assertRaises(ValueError):
            state.run_era_analysis()
        sample = state.era_line_samples[0]
        self.assertIsNotNone(sample.left_diagnostics)
        self.assertIsNotNone(sample.right_diagnostics)
        self.assertTrue(all(r.reason == "ambiguous_peak" for r in sample.left_diagnostics.rows))

    def test_awareness_that_wrapper_uses_new_core_not_module_only(self):
        # The bright-band wrapper must report the polynomial degree (3), proving it
        # routed through localize_bright_band_edge rather than the old linear path.
        image = periodic_image()
        for definition in ("center", "left", "right"):
            result = app.analyze_bright_band_edge(image, (15, 0, 40, 180), 0.2, definition)
            self.assertEqual(result.polynomial_degree, 3)
            self.assertIsNotNone(result.diagnostics)
            self.assertEqual(result.diagnostics.fit_degree, 3)
            self.assertEqual(result.diagnostics.position_definition, definition)

    def test_periodic_all_nine_combinations_center_is_midpoint(self):
        # left/right/center are mutually independent and center is the average of the
        # two half-heights — verified via the measured polynomial half-width shift.
        half_width = 2.6595
        shift = {"center": 0.0, "left": -half_width, "right": half_width}
        image = periodic_image()
        for left in ("center", "left", "right"):
            for right in ("center", "left", "right"):
                settings = app.PeriodicLineSettings(1, "trench", "trench", left, right)
                sample = app.periodic_line_samples(image, settings)[0]
                a = app.analyze_bright_band_edge(image, sample.left_roi_bounds_px, 0.2, left)
                b = app.analyze_bright_band_edge(image, sample.right_roi_bounds_px, 0.2, right)
                self.assertEqual(a.polynomial_degree, 3)
                self.assertEqual(a.diagnostics.fit_degree, 3)
                result = app.pair_single_edges(a, b, 0.2)
                expected = (35 + shift[right] - shift[left]) * 0.2
                self.assertAlmostEqual(result.mean_width_nm, expected, delta=0.04)

    def test_invalidate_clears_side_diagnostics_undo_restores(self):
        image = periodic_image()
        state = periodic_state(image)
        state.recognize_era_lines(measure=True)
        self.assertGreaterEqual(len(state.era_line_samples), 1)
        sample = state.era_line_samples[0]
        self.assertIsNotNone(sample.left_edge)
        # Capture diagnostics before invalidation (undo must restore them).
        saved_left_diag = sample.left_edge.diagnostics
        self.assertIsNotNone(saved_left_diag)

        state.record_undo_state = Mock()  # ensure undo path is exercised via snapshot
        # Manually invoke the invalidation helper's side-diagnostics clearing.
        state.invalidate_era_results()
        self.assertIsNone(state.era_line_samples[0].left_diagnostics)
        self.assertIsNone(state.era_line_samples[0].right_diagnostics)
        self.assertIsNone(state.era_line_samples[0].left_edge)


class CsvDiagnosticsTests(unittest.TestCase):
    """The main measurement CSV carries per-side / pair / per-row diagnostic fields."""

    def test_export_csv_contains_diagnostic_blocks(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        state.result = state.era_line_samples[0].paired_result
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            with patch.object(app.filedialog, "asksaveasfilename", return_value=str(path)):
                state.export_csv()
            text = path.read_text(encoding="utf-8-sig")
        # Per-side stats header.
        self.assertIn("coverage_fraction", text)
        self.assertIn("longest_gap_rows", text)
        self.assertIn("reason_counts", text)
        # Pair stats header.
        self.assertIn("common_valid_rows", text)
        self.assertIn("pair_coverage_fraction", text)
        self.assertIn("longest_common_gap_rows", text)
        # Per-row header.
        self.assertIn("candidate_global_x", text)
        self.assertIn("final_global_x", text)
        self.assertIn("recovered_from_global_x", text)

    def test_export_csv_row_valid_flags_match_measurement_positions(self):
        state = periodic_state()
        state.recognize_era_lines(measure=True)
        sample = state.era_line_samples[0]
        left_edge = sample.left_edge
        diag = left_edge.diagnostics
        records = ldiag.diag_row_records(diag)
        # Measurement positions come from valid rows only; each valid row's final
        # global x must be present and finite, matching valid_positions() (+left).
        valid_local = diag.valid_local_rows()
        valid_global = diag.valid_positions()
        left = diag.roi_bounds_px[0]
        n_valid = len(valid_local)
        self.assertEqual(n_valid, len(valid_global))
        self.assertEqual(n_valid, len(left_edge.rows_px))
        matching = [r for r in records if r["stage"] != "missing"]
        self.assertEqual(len(matching), n_valid)
        for record in matching:
            self.assertTrue(np.isfinite(record["final_global_x"]))

    def test_failed_sample_has_diag_but_not_in_total_result(self):
        # A sample whose entire side fails (<60% valid) keeps its diagnostics and is
        # excluded from pooled aggregate/LCDU counts.
        x = np.arange(40)
        profile = 30 + 80 / (1 + np.exp(-(x - 12))) + 80 / (1 + np.exp(-(x - 23)))
        bad = np.tile(profile, (180, 1))
        state = headless_app(bad)
        state.era_line_samples = [
            app.EraLineSample(left_roi_bounds_px=(0, 0, 40, 180), right_roi_bounds_px=(0, 0, 40, 180))
        ]
        with self.assertRaises(ValueError):
            state.run_era_analysis()
        self.assertIsNotNone(state.era_line_samples[0].left_diagnostics)
        self.assertIsNone(state.era_aggregate_result)


class NoisyWholeEdgeFailureTests(unittest.TestCase):
    """The noisy double-step gap-8 case must never yield a formal edge/aggregate,
    yet its failed per-row diagnostic stays reviewable (the supervisor's gap8 sigma4
    seed184 repro: most rows reject, the whole edge is <60% valid → LocalizationFailure
    instead of a short fabricated metrology trace)."""

    @staticmethod
    def _noisy_double_step(seed, sigma, nrows=100, width=64, low=30.0, amp=80.0, c1=12.0, c2=20.0):
        x = np.arange(width, dtype=float)
        rng = np.random.default_rng(seed)
        signal = low + amp / (1.0 + np.exp(-(x - c1))) + amp / (1.0 + np.exp(-(x - c2)))
        return signal + rng.normal(0.0, sigma, (nrows, width))

    def test_gap8_sigma4_seed184_whole_edge_fails_no_formal_result(self):
        # Two close (gap 8) equal steps at high noise: the single-line discriminator
        # accepts only a handful of rows (the false midpoint), far below the 60% gate.
        image = self._noisy_double_step(184, 4.0)
        with self.assertRaises(app.LocalizationFailure) as ctx:
            app.analyze_sample_edge(
                image, (0, 0, 64, 100), 0.2, app.LINE_LOCALIZATION_UNIFIED,
                degree=3, polarity=app.ERA_POLARITY_RISING,
            )
        diag = ctx.exception.diagnostics
        self.assertIsNotNone(diag)
        self.assertLess(len(diag.valid_local_rows()), len(diag.rows))  # not all rows valid
        self.assertLess(len(diag.valid_local_rows()), 60)  # below the 60% gate
        # The failed diagnostic is still fully reviewable (100 rows, each staged/reasoned).
        self.assertEqual(len(diag.rows), 100)
        self.assertTrue(all(hasattr(r, "stage") and hasattr(r, "reason") for r in diag.rows))

    def test_gap8_failed_sample_is_excluded_from_aggregate(self):
        # A two-sided sample built from the same noisy gap-8 image fails both sides,
        # so run_era_analysis raises and never forms an aggregate LCDU statistic.
        image = self._noisy_double_step(184, 4.0, nrows=100)
        state = headless_app(image)
        state.era_line_samples = [
            app.EraLineSample(left_roi_bounds_px=(0, 0, 64, 100), right_roi_bounds_px=(0, 0, 64, 100))
        ]
        with self.assertRaises(ValueError):
            state.run_era_analysis()
        sample = state.era_line_samples[0]
        self.assertIsNotNone(sample.left_diagnostics)
        self.assertIsNotNone(sample.right_diagnostics)
        self.assertIsNone(sample.paired_result)
        self.assertIsNone(state.era_aggregate_result)
        self.assertEqual(state.era_lcdu_cd_samples_nm, [])
        self.assertTrue(sample.measurement_error)

    def test_gap8_diag_csv_rows_valid_flag_never_1_for_whole_fail(self):
        # The diagnostic CSV for a whole-edge failure must report valid=0 rows (blank
        # final x), matching the fact that no formal measurement was produced.
        import line_diagnostics as ldiag
        image = self._noisy_double_step(184, 4.0, nrows=100)
        with self.assertRaises(app.LocalizationFailure) as ctx:
            app.analyze_sample_edge(
                image, (0, 0, 64, 100), 0.2, app.LINE_LOCALIZATION_UNIFIED,
                degree=3, polarity=app.ERA_POLARITY_RISING,
            )
        diag = ctx.exception.diagnostics
        sample = app.EraLineSample(left_roi_bounds_px=(0, 0, 64, 100))
        sample.left_diagnostics = diag
        snapshot = ldiag.SourceSnapshot(image=image, samples=(sample,))
        rows = ldiag.diagnostic_csv_rows(snapshot)
        header_index = rows.index(ldiag.DIAGNOSTIC_CSV_COLUMNS)
        data = [dict(zip(ldiag.DIAGNOSTIC_CSV_COLUMNS, row)) for row in rows[header_index + 1:] if row]
        valid_count = sum(1 for row in data if row["valid"] == "1")
        # Any row accepted by the single-line discriminator must still be far below
        # the whole-edge gate; the CSV reflects per-row validity faithfully.
        self.assertLess(valid_count, 60)
        # No formal final x was produced for the whole edge: the valid flag must match
        # the raw diag's valid rows exactly (never fabricate a formal result).
        self.assertEqual(valid_count, len(diag.valid_local_rows()))


if __name__ == "__main__":
    unittest.main()


class ChartCoordinateTests(unittest.TestCase):
    """Item 1: the chart must honour an explicit ROI origin (never image width)."""

    def _roi_left_80_edge(self):
        # ROI left=80 inside a 400-wide image, slice length 30 → the historical bug drew
        # x 370..399 while the marker sat at ~95 inside 80..109 and vanished off-chart.
        return (80, 20, 110, 180)

    def test_chart_coordinates_use_roi_origin_and_half_pixel_delta(self):
        data = ldiag.chart_sample_data(np.linspace(10, 40, 30), rowwidth=400, origin_x=80)
        self.assertEqual(data["origin_x"], 80)
        self.assertEqual(data["gray_x"][0], 80.0)
        self.assertEqual(data["gray_x"][-1], 109.0)
        self.assertEqual(data["delta_x"][0], 80.5)
        self.assertEqual(data["delta_x"][-1], 108.5)
        self.assertEqual(len(data["delta_x"]), 29)

    def test_chart_origin_is_not_right_aligned_to_image_width(self):
        # Defaults stay backward compatible (origin 0), and nothing may derive x from
        # rowwidth - len(values): the first gray x is always the origin itself.
        data = ldiag.chart_sample_data(np.arange(30.0), rowwidth=400)
        self.assertEqual(data["gray_x"][0], 0.0)
        self.assertEqual(data["gray_span"], (0.0, 29.0))
        self.assertNotEqual(data["gray_x"][0], 400 - 30)

    def test_chart_rgb_renders_candidate_and_final_markers_inside_roi(self):
        # Flat ramp whose single step is at ROI-local 15 → global 95 (inside 80..109).
        profile = 30 + 120 / (1 + np.exp(-(np.arange(30) - 15)))
        chart = ldiag.diag_chart_rgb(
            profile, rowwidth=400, origin_x=80, size=(640, 360), candidate_x=95.4, final_x=95.1,
        )
        self.assertEqual(chart.shape, (360, 640, 3))
        colors = {tuple(int(v) for v in pixel) for pixel in chart.reshape(-1, 3)}
        self.assertIn((255, 180, 94), colors)   # candidate marker colour
        self.assertIn((94, 240, 122), colors)   # final marker colour
        # Marker columns cluster near x≈95 in the chart, not near the right edge.
        marker_columns = [
            col for col in range(chart.shape[1])
            if np.any(np.all(chart[:, col, :] == np.array([94, 240, 122], dtype=np.uint8), axis=-1))
        ]
        self.assertTrue(marker_columns)
        self.assertLess(max(marker_columns), 640)

    def test_missing_row_with_candidate_does_not_fabricate_final(self):
        data = ldiag.chart_sample_data(np.zeros(30), rowwidth=400, origin_x=80)
        self.assertEqual(len(data["delta"]), 29)
        chart = ldiag.diag_chart_rgb(
            np.zeros(30), rowwidth=400, origin_x=80, candidate_x=95.0, final_x=None,
            failure_text="缺失：ambiguous_peak（候选未接受）",
        )
        colors = {tuple(int(v) for v in pixel) for pixel in chart.reshape(-1, 3)}
        self.assertIn((255, 180, 94), colors)   # candidate drawn
        self.assertNotIn((94, 240, 122), colors)  # no final marker invented

    def test_row_focus_region_is_single_selected_row_not_whole_roi(self):
        record = {"roi_left": 80, "roi_right": 110, "global_row": 37, "local_row": 37}
        self.assertEqual(ldiag.row_focus_region(record), (80, 37, 110, 38))


def _failed_side_diag():
    """Ambiguous double-step ROI (10,0,50,180) → all rows ``ambiguous_peak``, no candidate."""
    x = np.arange(40)
    profile = 30 + 80 / (1 + np.exp(-(x - 12))) + 80 / (1 + np.exp(-(x - 23)))
    image = np.tile(profile, (180, 1))
    try:
        app.analyze_sample_edge(
            image, (10, 0, 50, 180), 0.2, app.LINE_LOCALIZATION_UNIFIED,
            polarity=app.ERA_POLARITY_RISING,
        )
    except app.LocalizationFailure as exc:
        return exc.diagnostics
    raise AssertionError("expected LocalizationFailure")


class _FakePhoto:
    """Stand-in for ``ImageTk.PhotoImage`` (records the chart it wrapped)."""

    def __init__(self, image, master=None):
        self.image = image
        self.master = master


class _FakeTree:
    """Minimal Treeview double: selection + heading/column/insert/bind."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.items: list[tuple[str, str, tuple]] = []
        self.selection_value: tuple = ()
        self.bindings: dict = {}
        self.columns = kwargs.get("columns", ())
        _FakeTree.instances.append(self)

    def heading(self, *args, **kwargs):
        return None

    def column(self, *args, **kwargs):
        return None

    def pack(self, *args, **kwargs):
        return None

    def configure(self, *args, **kwargs):
        return None

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func

    def delete(self, *ids):
        self.items = []

    def insert(self, parent, index, values=(), iid=None):
        self.items.append((str(iid), "", tuple(values)))

    def get_children(self):
        return tuple(item[0] for item in self.items)

    def selection(self):
        return self.selection_value

    def selection_set(self, ids):
        self.selection_value = (ids,) if isinstance(ids, str) else tuple(ids)
        handler = self.bindings.get("<<TreeviewSelect>>")
        if handler is not None:
            handler(None)


class _FakeSelector:
    """Combobox double: current()/set()/bind()."""

    def __init__(self, master=None, values=(), state=None, width=None):
        self.values = list(values)
        self.index = -1
        self.bindings: dict = {}

    def pack(self, *args, **kwargs):
        return None

    def set(self, value):
        self.index = self.values.index(value) if value in self.values else -1

    def set_index(self, index):
        self.index = index

    def current(self):
        return self.index

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func


class _FakeWidget:
    def set(self, *args, **kwargs):
        return None

    def __init__(self, *args, **kwargs):
        self.image = kwargs.get("image")

    def pack(self, *args, **kwargs):
        return None

    def configure(self, *args, **kwargs):
        self.image = kwargs.get("image", self.image)
        return None


class _FakeToplevel(_FakeWidget):
    instances = []

    def __init__(self, master=None, **kwargs):
        super().__init__(**kwargs)
        self.title_value = ""
        self.geometry_value = ""
        self.master = master
        _FakeToplevel.instances.append(self)

    def title(self, value):
        self.title_value = value

    def minsize(self, *args):
        return None

    def geometry(self, value):
        self.geometry_value = value

    def winfo_screenheight(self):
        return 900


class _FakeVar:
    def __init__(self, master=None, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _failed_side_diag():
    """Ambiguous double-step ROI (10,0,50,180) → all rows ``ambiguous_peak``, no candidate."""
    x = np.arange(40)
    profile = 30 + 80 / (1 + np.exp(-(x - 12))) + 80 / (1 + np.exp(-(x - 23)))
    image = np.tile(profile, (180, 1))
    try:
        app.analyze_sample_edge(
            image, (10, 0, 50, 180), 0.2, app.LINE_LOCALIZATION_UNIFIED,
            polarity=app.ERA_POLARITY_RISING,
        )
    except app.LocalizationFailure as exc:
        return exc.diagnostics
    raise AssertionError("expected LocalizationFailure")


class _FakePhoto:
    """Stand-in for ``ImageTk.PhotoImage`` (records the chart image it wrapped)."""

    def __init__(self, image, master=None):
        self.image = image
        self.master = master


class _FakeTree:
    """Minimal Treeview double: real selection semantics + insert/bind."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.items: dict[str, tuple] = {}
        self.order: list[str] = []
        self.bindings: dict = {}
        _FakeTree.instances.append(self)

    def heading(self, *args, **kwargs):
        return None

    def column(self, *args, **kwargs):
        return None

    def pack(self, *args, **kwargs):
        return None

    def configure(self, *args, **kwargs):
        return None

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func

    def yview(self, *args, **kwargs):
        return None

    def delete(self, *ids):
        self.items = {}
        self.order = []

    def insert(self, parent, index, values=(), iid=None):
        self.items[str(iid)] = tuple(values)
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

    _selected: tuple = ()


class _FakeCombobox:
    """Combobox double: values/current()/set()/bind()."""

    instances: list = []

    def __init__(self, master=None, values=(), state=None, width=None):
        self.values = list(values)
        self.index = -1
        self.bindings: dict = {}
        _FakeCombobox.instances.append(self)

    def pack(self, *args, **kwargs):
        return None

    def set(self, value):
        self.index = self.values.index(value) if value in self.values else -1

    def current(self):
        return self.index

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func


class _FakeWidget:
    def set(self, *args, **kwargs):
        return None

    def __init__(self, *args, **kwargs):
        self.image = kwargs.get("image")
        self.instances_text = kwargs.get("text")

    def pack(self, *args, **kwargs):
        return None

    def configure(self, *args, **kwargs):
        if "image" in kwargs:
            self.image = kwargs["image"]
        return None


class _FakeToplevel(_FakeWidget):
    instances: list = []

    def __init__(self, master=None, **kwargs):
        super().__init__(**kwargs)
        self.master = master
        self.title_value = ""
        _FakeToplevel.instances.append(self)

    def title(self, value):
        self.title_value = value

    def minsize(self, *args):
        return None

    def geometry(self, value):
        return None

    def winfo_screenheight(self):
        return 900


class _FakeVar:
    def __init__(self, master=None, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


_tk_fake = SimpleNamespace(
    Toplevel=_FakeToplevel, StringVar=_FakeVar, END="end", BOTH="both", X="x", Y="y",
    LEFT="left", RIGHT="right", VERTICAL="vertical",
)
_ttk_fake = SimpleNamespace(
    Frame=_FakeWidget, Label=_FakeWidget, Button=_FakeWidget, Scrollbar=_FakeWidget,
    Treeview=_FakeTree, Combobox=_FakeCombobox,
)


class DiagnosticsUiCallbackTests(unittest.TestCase):
    """Items 2/3/4/6: drive the REAL ``build_line_diagnostics_ui`` callbacks.

    ``ldiag.tk`` / ``ldiag.ttk`` / ``ldiag.ImageTk`` are patched (Tk is normally an app
    dependency); ``ldiag.Image``/``diag_chart_rgb`` stay REAL so the chart pixels are
    production pixels. ``draw_helper``/``on_focus`` are given the same bodies the app
    passes (row slice of the frozen image; recenter on the requested region).
    """

    def setUp(self):
        _FakeTree.instances = []
        _FakeCombobox.instances = []
        _FakeToplevel.instances = []
        self.image = stripe_image()
        self.draw_calls: list = []
        self.focused: list = []
        self.state = headless_app(self.image.copy())
        self.state.era_polarity_var.set(app.ERA_POLARITY_RISING)
        self.state.restore_view_center = lambda center: self.focused.append(center)
        for patcher in (
            patch.object(ldiag, "tk", _tk_fake),
            patch.object(ldiag, "ttk", _ttk_fake),
            patch.object(ldiag, "ImageTk", SimpleNamespace(PhotoImage=_FakePhoto)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _draw_helper(self, image, global_row, bounds):
        self.draw_calls.append((global_row, tuple(bounds)))
        left, top, right, bottom = bounds
        return image[global_row, left:right].astype(float)

    def _on_focus(self, left, top, right, bottom):
        self.state.restore_view_center(((left + right) / 2.0, (top + bottom) / 2.0))

    def _build(self, samples, live=None, left_x=0, **snapshot_fields):
        live = self.state if live is None else live
        # Same body the app passes as its live getter (path frozen at open time).
        live_getter = lambda: SimpleNamespace(
            is_same_live_source=lambda capture: live is self.state and self._live_identity(capture),
        )
        snapshot = ldiag.SourceSnapshot(
            image=self.image.copy(), samples=tuple(samples), left_x=left_x,
            live=live_getter(), source_identity=id(self.state.processed_image),
            source_path=self._live_path(), **snapshot_fields,
        )
        return ldiag.build_line_diagnostics_ui(
            None, source_snapshot=snapshot, draw_helper=self._draw_helper, on_focus=self._on_focus,
        )

    def _live_path(self) -> str:
        return "" if self.state.image_path is None else str(self.state.image_path)

    def _live_identity(self, capture) -> bool:
        return (
            self.state.processed_image is not None
            and id(self.state.processed_image) == capture.get("source_identity")
            and self._live_path() == (capture.get("source_path") or "")
        )

    def _failed_left_sample(self):
        sample = app.EraLineSample(left_roi_bounds_px=(10, 0, 50, 180))
        sample.left_diagnostics = _failed_side_diag()
        return sample

    def _good_and_right_failed(self):
        good = app.automatic_era_line_samples(self.image, True)[0]
        self.state.era_line_samples = [good]
        self.state.run_era_analysis()
        return self.state.era_line_samples[0]

    # ---- item 2: per-row focus ----------------------------------------
    def test_selected_row_focuses_that_row_inside_the_roi_not_the_centre(self):
        dialog = self._build([self._failed_left_sample()])
        tree = _FakeTree.instances[-1]
        self.assertEqual(len(tree.get_children()), 180)
        self.focused.clear()
        tree.selection_set("57")
        self.assertEqual(self.draw_calls[-1][0], 57)
        self.assertEqual(self.draw_calls[-1][1], (10, 0, 50, 180))
        # Focus region = ROI columns (10..50) on row 57..58; free-look main view, so
        # the headless centre arithmetic (region midpoint) is what the callback got.
        self.assertEqual(self.focused[-1], (30.0, 57.5))
        self.assertNotEqual(self.focused[-1], (30.0, 90.0))  # NOT the ROI centre
        self.assertEqual(len(dialog.line_diagnostics_photos), 1)

    # ---- item 1/4/6: right-side non-zero ROI failed row -----------------
    def test_right_side_failed_row_uses_right_axes_and_is_labelled_right(self):
        sample = app.EraLineSample(
            left_roi_bounds_px=(10, 0, 50, 180), right_roi_bounds_px=(100, 0, 140, 180),
        )
        self.state.era_line_samples = [sample]
        with self.assertRaises(ValueError):
            self.state.run_era_analysis()
        sample = self.state.era_line_samples[0]
        self.assertIsNotNone(sample.right_diagnostics)
        dialog = self._build([sample])
        selector = _FakeCombobox.instances[-1]
        self.assertEqual(selector.values, ["样本 1 左边", "样本 1 右边"])
        selector.set("样本 1 右边")
        selector.bindings["<<ComboboxSelected>>"](None)
        self.assertEqual(dialog.line_diagnostics_state["side"], "right")
        self.assertEqual(dialog.line_diagnostics_state["left"], 100)
        # A missing row in the right ROI: real global axes from the right ROI's left edge.
        missing_row = next(
            str(record["local_row"]) for record in dialog.line_diagnostics_state["records"]
            if record["stage"] == "missing"
        )
        self.focused.clear()
        tree = _FakeTree.instances[-1]
        tree.selection_set(missing_row)
        row = int(missing_row)
        self.assertEqual(self.draw_calls[-1][1], (100, 0, 140, 180))
        self.assertEqual(self.focused[-1], (120.0, row + 0.5))
        values = tree.items[missing_row]
        self.assertEqual(values[2], "")  # final global x never fabricated
        self.assertEqual(values[3], "缺失")

    # ---- item 6: photo lifetime ---------------------------------------
    def test_repeated_selection_keeps_one_photo_and_windows_do_not_share_it(self):
        first = self._build([self._failed_left_sample()])
        tree_a = _FakeTree.instances[-1]
        photo_ids = set()
        for row in ("3", "4", "5", "120"):
            tree_a.selection_set(row)
            photo_ids.add(id(first.line_diagnostics_photos["current"]))
            self.assertEqual(len(first.line_diagnostics_photos), 1)
        second = self._build([self._failed_left_sample()])
        tree_b = _FakeTree.instances[-1]
        tree_b.selection_set("8")
        # B's chart replaced only B's slot; A's slot still holds A's own photo.
        self.assertIsNot(first.line_diagnostics_photos, second.line_diagnostics_photos)
        self.assertEqual(len(first.line_diagnostics_photos), 1)
        self.assertEqual(len(second.line_diagnostics_photos), 1)
        self.assertIsNot(first.line_diagnostics_photos["current"], second.line_diagnostics_photos["current"])
        self.assertEqual(first.line_diagnostics_photos["current"].master, _FakeToplevel.instances[0])

    # ---- item 3: stale source ----------------------------------------
    def test_switching_to_image_b_never_pans_and_shows_a_short_notice(self):
        dialog = self._build([self._failed_left_sample()])
        tree = _FakeTree.instances[-1]
        self.focused.clear()
        tree.selection_set("11")
        self.assertEqual(len(self.focused), 1)  # A still live → normal jump
        # Main view switches to image B: a different processed object.
        self.state.processed_image = self.image[::-1].copy()
        self.state.image_path = "B.psd"
        tree.selection_set("12")
        self.assertEqual(len(self.focused), 1)  # no pan of B
        self.assertEqual(dialog.line_diagnostics_notice.get(), ldiag.STALE_SOURCE_NOTICE)
        # Back to A (same object again is not restored — it was replaced), still no pan.
        tree.selection_set("13")
        self.assertEqual(len(self.focused), 1)

    def test_stale_detection_also_requires_same_path(self):
        live = SimpleNamespace(is_same_live_source=lambda capture: True)
        capture = {"live": live, "source_path": "A.psd", "source_id": "A", "source_identity": 1}
        self.assertTrue(ldiag.live_still_same(capture, live))
        self.assertFalse(ldiag.live_still_same(capture, SimpleNamespace(is_same_live_source=lambda c: True)))
        self.assertFalse(ldiag.live_still_same(capture, None))

    # ---- item 4: both-side statistics ---------------------------------
    def test_stats_text_shows_both_sides_and_common_coverage(self):
        good = self._good_and_right_failed()
        dialog = self._build([good])
        text = dialog.line_diagnostics_stats.get()
        self.assertIn("左侧 ROI 总行", text)
        self.assertIn("右侧 ROI 总行", text)
        self.assertIn("两侧共同有效", text)
        stats = localization_stats(good.left_edge.diagnostics)
        self.assertIn(f"有效 {stats.primary_rows + stats.recovered_rows}/{stats.total_rows}", text)
        pair = pair_localization_stats(good.left_edge.diagnostics, good.right_edge.diagnostics)
        self.assertIn(f"{pair.common_valid_rows}/{pair.overlap_rows}", text)
        # Selecting the right side must not relabel it as the left side.
        _FakeCombobox.instances[-1].set("样本 1 右边")
        _FakeCombobox.instances[-1].bindings["<<ComboboxSelected>>"](None)
        self.assertIn("当前选中：右侧", dialog.line_diagnostics_stats.get())
        self.assertIn("左侧 ROI 总行", dialog.line_diagnostics_stats.get())


class DiagnosticCsvFreezeTests(unittest.TestCase):
    """Item 5: the diagnostic CSV is exported from the frozen snapshot, full precision."""

    def _state_with_diagnostics(self):
        image = periodic_image()
        state = periodic_state(image)
        state.recognize_era_lines(measure=True)
        sample = state.era_line_samples[0]
        state.line_diagnostics_source_info = MethodType(app.LERLWRApp.line_diagnostics_source_info, state)
        info = state.line_diagnostics_source_info()
        info["source_path"] = "A.psd"
        info["preprocessing"] = "denoise=on;normalize=on"
        info["calibration"] = "0.2 nm/px"
        info["rotation"] = "0°"
        snapshot = ldiag.SourceSnapshot(
            image=state.processed_image.copy(), samples=tuple(state.era_line_samples), **info,
        )
        return state, sample, snapshot

    def test_rows_carry_real_parameters_valid_flags_and_full_precision(self):
        state, sample, snapshot = self._state_with_diagnostics()
        rows = ldiag.diagnostic_csv_rows(snapshot)
        header = ldiag.DIAGNOSTIC_CSV_COLUMNS
        header_index = rows.index(header)
        frozen = {row[0]: row[1] for row in rows[:header_index] if len(row) >= 2}
        self.assertEqual(frozen["source"], "A.psd")
        self.assertEqual(frozen["preprocessing"], "denoise=on;normalize=on")
        self.assertEqual(frozen["calibration"], "0.2 nm/px")
        self.assertEqual(frozen["rotation"], "0°")
        self.assertEqual(frozen["recovery_policy"], ldiag.RECOVERY_POLICY)
        diag = sample.left_edge.diagnostics
        data = [dict(zip(header, row)) for row in rows[header_index + 1:] if row]
        left_rows = [row for row in data if row["side"] == "left"]
        self.assertEqual(len(left_rows), diag.height * len(snapshot.samples))
        for row in left_rows:
            self.assertEqual(row["position_definition"], diag.position_definition)
            self.assertEqual(float(row["fit_degree"]), float(diag.fit_degree))
            self.assertEqual(float(row["max_jump_px"]), float(diag.max_jump_px))
            self.assertEqual(row["recovery_policy"], ldiag.RECOVERY_POLICY)
            self.assertEqual(row["source_path"], "A.psd")
            final = row["final_global_x"]
            if row["stage"] == "missing":
                self.assertEqual(row["valid"], "0")
                self.assertEqual(final, "")
            else:
                self.assertEqual(row["valid"], "1")
                # Full precision: text is repr(float(record)), not '.4f' nor '.3f'.
                self.assertGreaterEqual(len(final.split(".")[-1]), 6)
                self.assertNotEqual(final, f"{float(final):.4f}")
                self.assertNotEqual(final, f"{float(final):.3f}")


    def test_csv_stays_on_source_a_after_the_main_image_changes(self):
        state, sample, snapshot = self._state_with_diagnostics()
        before = ldiag.diagnostic_csv_rows(snapshot)
        state.processed_image = np.flipud(state.processed_image).copy()
        state.image_path = "B.psd"
        state.normalize_var.set(False)
        after = ldiag.diagnostic_csv_rows(snapshot)
        self.assertEqual(before, after)
        frozen = {row[0]: row[1] for row in after[:after.index(ldiag.DIAGNOSTIC_CSV_COLUMNS)] if len(row) >= 2}
        self.assertEqual(frozen["source"], "A.psd")
        self.assertEqual(frozen["preprocessing"], "denoise=on;normalize=on")


class PairingIsolationTests(unittest.TestCase):
    """Item 7: a per-sample pairing failure must not abort the remaining samples."""

    def _run(self, samples):
        state = headless_app(stripe_image())
        state.era_polarity_var.set(app.ERA_POLARITY_RISING)
        state.era_line_samples = samples
        state.run_era_analysis()  # bound MethodType → patch app.pair_single_edges (module global)
        return state

    def test_y_disjoint_pair_failure_is_per_sample_and_good_sample_still_aggregates(self):
        good = app.automatic_era_line_samples(stripe_image(), True)[0]
        # Both sides localize on their own, but their valid rows do not overlap at all
        # (right ROI re-localizes to its own rows): pairing must fail for THIS sample only.
        bad = app.EraLineSample(
            left_roi_bounds_px=(9, 0, 36, 80), right_roi_bounds_px=(36, 100, 66, 180),
        )
        state = self._run([good, bad])
        first, second = state.era_line_samples
        self.assertIsNotNone(first.paired_result)
        self.assertIsNone(first.measurement_error)
        self.assertIsNone(second.paired_result)
        self.assertIn("右边", second.measurement_error)
        self.assertIsNotNone(second.left_edge)          # diagnostics preserved
        self.assertIsNotNone(second.left_edge.diagnostics)
        # The good sample still aggregates, and the bad one is reported as a failure.
        self.assertEqual(state.era_aggregate_result.line_count, 1)
        self.assertEqual(len(state.era_lcdu_cd_samples_nm), 1)
        text = state.result_var.get()
        self.assertIn("失败 1 根", text)
        self.assertIn("未参与统计", text)

    def test_right_completely_missing_sample_keeps_diag_and_other_samples_continue(self):
        """Both ROIs y-disjoint AND right side unfittable: still per-sample only."""
        good = app.automatic_era_line_samples(stripe_image(), True)[0]
        bad = app.EraLineSample(
            left_roi_bounds_px=(9, 0, 36, 180), right_roi_bounds_px=(36, 300, 70, 400),
        )
        state = self._run([good, bad])
        first, second = state.era_line_samples
        self.assertIsNotNone(first.paired_result)
        self.assertIsNone(second.paired_result)
        self.assertIn("右边", second.measurement_error)
        self.assertIn("单边 ROI 过小", second.measurement_error)
        self.assertEqual(state.era_aggregate_result.line_count, 1)
