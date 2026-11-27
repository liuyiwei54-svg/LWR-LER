"""主图下方「灰度差曲线」面板：行范围平均 + 水平差分 + 物理对齐测试。

覆盖四条不变量：
1. 纯数据：算术平均（float64）跨行 → ``np.diff`` 水平差分，半像素 x 轴；平均 ≠
   单行 ≠ 多行中位数（粗定位原始逻辑不动）。
2. 真实面板构造 + 真实回调（Return / 应用按钮 / 范围校验），仅伪造显示设施
   （Tk 控件 / PhotoImage），不替换被测方法本身。
3. 物理对齐：数据点屏幕 x 由主/图 canvas 的真实 ``winfo_rootx()``、``display_scale``
   与 ``canvasx(0)`` 计算（生产代码不使用不存在的 ``.rootx`` 属性），非全宽拉伸；
   含横向刻度文字；零轴、图像右界（x = width）、离屏裁剪与空白右侧；滚动/缩放/
   重设大小经真实回调重绘。
4. 生命周期/缓存：源对象或范围变化失效；显示只读、不污染图像或测量状态。
"""

from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import ler_lwr_app as app


class FakeCanvas:
    """Minimal canvas recording drawn items plus width/height.

    与真实 Tk Canvas 一致：**没有** ``.rootx`` 属性，屏幕坐标只通过
    ``winfo_rootx()`` 取得。
    """

    def __init__(self, width, height, root_x=0.0):
        self._width = width
        self._height = height
        self._root_x = root_x
        self.items = []
        self.bindings = {}

    def winfo_width(self):
        return self._width

    def winfo_height(self):
        return self._height

    def winfo_rootx(self):
        return self._root_x

    def set_root_x(self, value):
        self._root_x = value

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func

    def delete(self, _tag="all"):
        self.items = []

    def create_image(self, x, y, image=None, anchor=None):
        self.items.append(("image", x, y, image))

    def create_line(self, *coords, **kwargs):
        # 与真实 Tk 一致：create_line 至少 4 个坐标（两个点）且坐标个数为偶数，
        # 否则 ``canvas create line`` 抛 TclError("wrong # coordinates")。
        if len(coords) < 4:
            raise app.tk.TclError(
                f"wrong # coordinates: expected at least 4, got {len(coords)}")
        if len(coords) % 2:
            raise app.tk.TclError(
                f"wrong # coordinates: expected an even number, got {len(coords)}")
        self.items.append(("line", np.asarray(coords, dtype=float), kwargs))

    def create_oval(self, x0, y0, x1, y1, **kwargs):
        self.items.append(("oval", float(x0), float(y0), float(x1), float(y1), kwargs))

    def create_text(self, x, y, text="", **kwargs):
        self.items.append(("text", float(x), float(y), text, kwargs))


class FakeEntry:
    def __init__(self, _parent=None, textvariable=None, width=8):
        self.textvariable = textvariable
        self.bindings = {}

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func

    def pack(self, **_kwargs):
        pass

    def grid(self, **_kwargs):
        pass


class FakeButton:
    def __init__(self, _parent=None, text="", command=None):
        self.text = text
        self.command = command
        self.bindings = {}

    def pack(self, **_kwargs):
        pass

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func


class FakeCombobox(FakeEntry):
    """``ttk.Combobox`` stand-in: real ``StringVar`` semantics + widget-event binding.

    只替换显示设施：值仍由真实 ``textvariable``（生产构造的 ``StringVar``）承载，
    ``<<ComboboxSelected>>`` 绑定的是生产回调本身，测试通过设置变量并调用该绑定来
    模拟用户在下拉中选择，不做任何 mock 绘图目标。
    """

    def __init__(self, _parent=None, textvariable=None, values=(), state=None, width=8, **_kwargs):
        super().__init__(_parent, textvariable, width)
        self.textvariable = textvariable
        self.values = tuple(values)
        self.state = state
        self.bindings = {}


class FakeEntryVar:
    def __init__(self, master=None, value="", name=None):
        # ``master`` 可能被静默替换成 binding 里的 value，因此支持两种既有调用写法。
        if name is None and isinstance(master, str) and not isinstance(value, str):
            master, value = value, master
        if value is None:
            value = ""
        self._value = value
        self._trace = []

    def trace_add(self, mode, callback):
        # 真实 ``StringVar.trace_add``：写入后立即回调（fake 无事件循环）。
        self._trace.append((mode, callback))
        return f"trace#{len(self._trace)}"

    def get(self):
        return self._value

    def set(self, value):
        self._value = value
        for _mode, callback in self._trace:
            callback("*", "", "")


class FakePhoto:
    def __init__(self, pil_image):
        self.pil = pil_image


class FakeTkCanvas(FakeCanvas):
    """Canvas fake that records bindings (``<Configure>``) and created lines/text."""

    def __init__(self, width=640, height=160, root_x=0.0, **_kwargs):
        super().__init__(width, height, root_x=root_x)
        self.parent = None

    def pack(self, **_kwargs):
        pass


class FakeTkChild:
    """Frame / LabelFrame / PanedWindow stand-in recording children and pane adds."""

    def __init__(self, master=None, **_kwargs):
        self.master = master
        if master is not None and hasattr(master, "children"):
            master.children.append(self)
        self.children = []
        self.panes = []
        self.adds = 0

    def pack(self, **_kwargs):
        pass

    def grid(self, **_kwargs):
        pass

    def pack_propagate(self, _flag):
        pass

    def add(self, child, **_kwargs):
        self.adds += 1
        self.panes.append(child)


class FakeMainCanvas:
    """Stand-in for the real main SEM canvas: ``winfo_rootx`` / ``canvasx(0)`` / ``xview``.

    刻意不提供 ``.rootx``：真实 Tk Canvas 没有该属性，生产代码若写 ``main.rootx``
    在真实界面会 ``AttributeError``，fake 必须同样暴露失败而不是掩盖。
    """

    def __init__(self, root_x=0.0, canvasx_zero=0.0, mapped=True):
        self._root_x = root_x
        self._canvasx_zero = canvasx_zero
        self._mapped = mapped
        self.bindings = {}
        self.xscrollcommand = None
        self.configure_calls = []

    def winfo_rootx(self):
        return self._root_x

    def set_root_x(self, value):
        self._root_x = value

    def winfo_ismapped(self):
        return self._mapped

    def canvasx(self, x):
        return self._canvasx_zero + x

    def bind(self, sequence, func=None, add=None):
        self.bindings[sequence] = func

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)
        # 真实 Tk：``canvas.configure(xscrollcommand=fn)`` 等价于设置选项值。
        if "xscrollcommand" in kwargs:
            self.xscrollcommand = kwargs["xscrollcommand"]


def line_items(canvas):
    """Return every drawn line item as ``(coords_array, kwargs)``."""
    return [it[1:] for it in canvas.items if it[0] == "line"]


def text_items(canvas):
    """Return every drawn text item as ``(x, y, text, kwargs)``."""
    return [it[1:] for it in canvas.items if it[0] == "text"]


def oval_items(canvas):
    """Return every drawn oval item as ``(x0, y0, x1, y1, kwargs)``."""
    return [it[1:] for it in canvas.items if it[0] == "oval"]


def curve_dots(canvas):
    """Single-point curve markers (ovals drawn in the curve colour)."""
    return [oval for oval in oval_items(canvas) if oval[4].get("fill") == "#7fd4ff"]


def curve_segments(canvas):
    return [np.asarray(coords, dtype=float) for coords, kw in line_items(canvas)
            if kw.get("fill") == "#7fd4ff"]


def build_curve_state(image, y_start=None, y_end=None, width=640, height=160,
                      main=None, display_scale=1.0, graph_root_x=0.0, main_root_x=None,
                      gray_difference_mode_var=None):
    """Construct a namespace bound to the real production curve methods."""
    state = SimpleNamespace(
        processed_image=image,
        raw_image=image,
        original_image=image,
        gray_difference_y_start=y_start,
        gray_difference_y_end=y_end,
        gray_difference_photo=None,
        gray_difference_canvas=FakeCanvas(width, height, root_x=graph_root_x),
        gray_difference_start_entry_var=FakeEntryVar(str(y_start) if y_start is not None else ""),
        gray_difference_end_entry_var=FakeEntryVar(str(y_end) if y_end is not None else ""),
        gray_difference_status_var=FakeEntryVar(""),
        gray_difference_mode_var=gray_difference_mode_var,
        gray_difference_error=None,
        _gray_diff_redraw_pending=False,
        _gray_diff_cache_key=None,
        _gray_diff_cached_mean=None,
        _gray_diff_cache_source=None,
        canvas=main,
        display_scale=display_scale,
        main_root_x=main_root_x,
    )
    for name in (
        "gray_difference_resolved_range",
        "gray_difference_cached_mean",
        "invalidate_gray_difference_cache",
        "gray_difference_view_range",
        "update_gray_difference_curve",
        "_schedule_gray_difference_redraw",
        "_cancel_gray_difference_redraw",
        "_clear_gray_difference",
        "_gray_difference_main_root_x",
        "_gray_difference_xscroll",
        "gray_difference_widget_root_x",
        "gray_difference_nice_tick_step",
        "gray_difference_mode",
        "gray_difference_mode_value",
        "gray_difference_mode_changed",
    ):
        # 静态方法保持原样（绑定成 MethodType 会多传 self）。
        descriptor = app.LERLWRApp.__dict__[name]
        if isinstance(descriptor, staticmethod):
            setattr(state, name, descriptor.__func__)
        else:
            setattr(state, name, MethodType(descriptor, state))
    state.after_idle = lambda fn: fn()
    state.after_cancel = lambda _token: None
    state.image_horizontal_scrollbar = None
    return state


def run_configure_bindings(bindings):
    for sequence, func in list(bindings.items()):
        if func is not None:
            func(SimpleNamespace(widget=None, width=0, height=0, type="Configure"))


def make_image(height=32, width=64, pattern="ramp"):
    """A heterogeneous fixture whose rows differ in edge position/slope/offset.

    ``step_left`` rows place a hard edge at x==12; ``step_right`` at x==width-16;
    ``ramp`` rows add a per-row offset so a mean across rows differs from any
    single row and from a per-column median.
    """
    image = np.zeros((height, width), dtype=np.uint8)
    columns = np.arange(width, dtype=int)
    for row in range(height):
        if pattern == "ramp":
            image[row, :] = np.clip(columns * 3 + row * 2, 0, 255)
        elif pattern == "flat_ramp":
            image[row, :] = np.clip(columns * 3, 0, 255)
        elif pattern == "step_left":
            # 斜坡 + 台阶：均值剖面仍保留差分，不会因常数段被压平。
            image[row, :] = np.clip(40 + columns * 2, 0, 255) + np.where(columns >= 12, 80, 0)
        elif pattern == "step_right":
            image[row, :] = np.where(columns >= width - 16, 120, 40)
        elif pattern == "soft_step":
            image[row, :] = np.clip(40 + columns * 2, 0, 255)
        elif pattern == "falling":
            image[row, :] = np.clip(255 - columns * 3 - row * 2, 0, 255)
        elif pattern == "hetero_mix":
            # Even rows rise at x=12, odd rows rise at x=width-16 → the averaged
            # profile carries two rising steps (a single row carries only one).
            image[row, :] = (
                np.where(columns >= 12, 140, 40)
                if row % 2 == 0
                else np.where(columns >= width - 16, 140, 40)
            )
        else:
            image[row, :] = np.clip(columns + row * 7, 0, 255)
    return image


class PureMeanDifferenceFunctionTests(unittest.TestCase):
    def test_mean_diff_equals_np_diff_of_float_mean(self):
        image = make_image(height=8, width=32, pattern="ramp")
        mean = app.profile_mean_intensity(image, 1, 6)
        self.assertEqual(mean.dtype, np.float64)
        self.assertEqual(mean.shape, (32,))
        np.testing.assert_allclose(mean, image[1:6, :].astype(float).mean(axis=0))
        diff = app.profile_adjacent_difference(mean)
        np.testing.assert_allclose(diff, np.diff(image[1:6, :].astype(float).mean(axis=0)))

    def test_mean_differs_from_single_row_and_median(self):
        # 偶行左台阶、奇行右台阶：多行均值 ≠ 任一行；奇数行数的每列中位数也 ≠ 均值
        # （三行 = 左台阶/右台阶/左台阶，中位数落在左台阶，均值是两者折中）。
        image = make_image(height=8, width=32, pattern="hetero_mix")
        mean = app.profile_mean_intensity(image, 0, 3)
        single = app.profile_scanline_intensity(image, 0)
        column_median = np.median(image[0:3, :].astype(float), axis=0)
        self.assertFalse(np.allclose(mean, single),
                         "mean across two offset rows must differ from one row")
        self.assertFalse(np.allclose(mean, column_median),
                         "arithmetic mean must differ from per-column median")
        # The heterogeneous average has two negative edges (both step directions).
        diff = app.profile_adjacent_difference(mean)
        self.assertTrue(np.any(diff != 0), "averaged profile must carry the edges")
        self.assertFalse(np.allclose(np.median(image[0:3, :].astype(float), axis=0), mean))

    def test_negative_edge_signed_float(self):
        image = make_image(height=4, width=24, pattern="falling")
        mean = app.profile_mean_intensity(image, 0, 4)
        diff = app.profile_adjacent_difference(mean)
        self.assertEqual(diff.dtype, np.float64)
        self.assertTrue(np.all(diff <= 0))

    def test_half_pixel_x_axis(self):
        x = app.profile_difference_x_axis(np.zeros(12, dtype=float))
        np.testing.assert_array_equal(x, np.arange(11, dtype=float) + 0.5)

    def test_mean_rejects_bad_range(self):
        image = make_image(height=6, width=8)
        with self.assertRaises(ValueError):
            app.profile_mean_intensity(image, 4, 3)
        with self.assertRaises(ValueError):
            app.profile_mean_intensity(image, -1, 3)
        with self.assertRaises(ValueError):
            app.profile_mean_intensity(image, 0, 7)


class InlineMeanCurvePanelTests(unittest.TestCase):
    def setUp(self):
        self.patch_show = patch(
            "tkinter.PhotoImage.__init__", new=lambda self, image=None, **kw: None
        )

    def test_defaults_to_full_height_and_draws_without_click(self):
        image = make_image(height=16, width=40, pattern="ramp")
        state = build_curve_state(image, y_start=None, y_end=None,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        # Absorb tkinter PhotoImage creation in the (unused) photo path.
        state.update_gray_difference_curve()
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end), (0, 16))
        self.assertEqual(state.gray_difference_start_entry_var.get(), "0")
        self.assertEqual(state.gray_difference_end_entry_var.get(), "16")
        self.assertIn("平均行 0..16", state.gray_difference_status_var.get())
        self.assertIn("共 16 行", state.gray_difference_status_var.get())
        # A visible mean curve is drawn (native line items), no photo path used.
        self.assertTrue(any(it[0] == "line" for it in state.gray_difference_canvas.items))

    def test_range_average_diff_matches_independent_expectation(self):
        image = make_image(height=16, width=48, pattern="ramp")
        state = build_curve_state(image, y_start=2, y_end=6,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.update_gray_difference_curve()
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end), (2, 6))
        independent = np.diff(image[2:6, :].astype(float).mean(axis=0))
        cached = state._gray_diff_cached_mean
        np.testing.assert_allclose(app.profile_adjacent_difference(cached), independent)
        self.assertIn("共 4 行", state.gray_difference_status_var.get())

    def test_view_range_rejects_invalid_and_keeps_range(self):
        image = make_image(height=16, width=40, pattern="ramp")
        state = build_curve_state(image, y_start=2, y_end=6,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.update_gray_difference_curve()
        for raw_start, raw_end in (("999", "6"), ("2", "1.5"), ("abc", "6"),
                                   ("2", "20"), ("2", "1"), ("-3", "6")):
            state.gray_difference_start_entry_var.set(raw_start)
            state.gray_difference_end_entry_var.set(raw_end)
            state.gray_difference_view_range()
            self.assertEqual(state.gray_difference_y_start, 2, (raw_start, raw_end))
            self.assertEqual(state.gray_difference_y_end, 6, (raw_start, raw_end))
            self.assertEqual(state.gray_difference_canvas.items, [], (raw_start, raw_end))
            # Error caption persists; range unchanged.
            self.assertTrue(state.gray_difference_error)
        # Valid range restores display.
        state.gray_difference_start_entry_var.set("3")
        state.gray_difference_end_entry_var.set("8")
        state.gray_difference_view_range()
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end), (3, 8))
        self.assertTrue(any(it[0] == "line" for it in state.gray_difference_canvas.items))

    def test_empty_both_resets_to_full_height(self):
        image = make_image(height=16, width=40, pattern="ramp")
        state = build_curve_state(image, y_start=2, y_end=6,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.update_gray_difference_curve()
        state.gray_difference_start_entry_var.set("")
        state.gray_difference_end_entry_var.set("")
        state.gray_difference_view_range()
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end), (0, 16))

    def test_single_sided_empty_is_error(self):
        image = make_image(height=16, width=40, pattern="ramp")
        state = build_curve_state(image, y_start=2, y_end=6,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.update_gray_difference_curve()
        state.gray_difference_start_entry_var.set("3")
        state.gray_difference_end_entry_var.set("")
        state.gray_difference_view_range()
        self.assertTrue(state.gray_difference_error)
        self.assertEqual(state.gray_difference_canvas.items, [])
        self.assertEqual(state.gray_difference_y_start, 2)  # unchanged

    def test_no_image_clears_and_reports(self):
        state = build_curve_state(None, y_start=0, y_end=5)
        state.update_gray_difference_curve()
        self.assertEqual(state.gray_difference_canvas.items, [])
        self.assertEqual(state.gray_difference_start_entry_var.get(), "")
        self.assertIn("未载入图像", state.gray_difference_status_var.get())

    def test_display_is_read_only(self):
        image = make_image(height=16, width=40, pattern="ramp")
        before = image.copy()
        state = build_curve_state(image, y_start=2, y_end=6,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.result = object()          # arbitrary measurement sentinel
        state.roi_canvas = (1.0, 2.0, 3.0, 4.0)
        state.era_line_samples = []
        state.update_gray_difference_curve()
        state.gray_difference_start_entry_var.set("3")
        state.gray_difference_view_range()
        np.testing.assert_array_equal(state.processed_image, before)
        self.assertIsNotNone(state.result)
        self.assertEqual(state.roi_canvas, (1.0, 2.0, 3.0, 4.0))


class AlignmentTransformTests(unittest.TestCase):
    """屏幕 x 变换：只用真实 Tk API（``winfo_rootx``），多种 scale / 滚动 / 根偏移。"""

    def _screen_x(self, state, x_img):
        main = state.canvas
        return (main.winfo_rootx() + x_img * state.display_scale
                - main.canvasx(0) - state.gray_difference_canvas.winfo_rootx())

    def test_production_never_uses_nonexistent_canvas_rootx(self):
        # 真实 Tk Canvas 没有 rootx；fake 也不提供，因此用属性访问的生产代码会失败。
        self.assertFalse(hasattr(FakeCanvas(10, 10), "rootx"))
        self.assertFalse(hasattr(FakeMainCanvas(), "rootx"))
        image = make_image(height=16, width=64, pattern="step_left")
        state = build_curve_state(image, y_start=0, y_end=16,
                                  main=FakeMainCanvas(root_x=100.0), display_scale=1.0,
                                  graph_root_x=20.0)
        state.update_gray_difference_curve()
        self.assertTrue(curve_segments(state.gray_difference_canvas))

    def test_curve_points_align_at_multiple_scales_and_offsets(self):
        image = make_image(height=16, width=64, pattern="step_left")
        col_centers = np.arange(len(app.profile_adjacent_difference(
            image.astype(float).mean(axis=0)))) + 0.5
        for scale, main_root, canvasx_zero, graph_root, cw in (
            (1.0, 100.0, 50.0, 20.0, 640),
            (2.0, 96.0, 0.0, 12.0, 640),
            (0.5, 0.0, 0.0, 0.0, 200),
            (1.5, 240.0, 30.0, 90.0, 400),
        ):
            with self.subTest(scale=scale, main_root=main_root, graph_root=graph_root):
                main = FakeMainCanvas(root_x=main_root, canvasx_zero=canvasx_zero)
                state = build_curve_state(image, y_start=0, y_end=16, main=main,
                                          display_scale=scale, graph_root_x=graph_root,
                                          width=cw, height=160,
                                          gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
                state.update_gray_difference_curve()
                segments = curve_segments(state.gray_difference_canvas)
                self.assertTrue(segments)
                xs = [float(p[0]) for seg in segments for p in seg.reshape(-1, 2)]
                first_x = min(xs)
                last_x = max(xs)
                expect_first = self._screen_x(state, col_centers[0])
                expect_last = self._screen_x(state, col_centers[-1])
                if expect_first >= 0.0:
                    self.assertAlmostEqual(first_x, expect_first, places=6)
                if expect_last <= cw:
                    self.assertAlmostEqual(last_x, expect_last, places=6)
                # 图像右边界竖线在 x = width（不是 width + 0.5）。
                right_x = self._screen_x(state, float(image.shape[1]))
                border = [coords for coords, kw in line_items(state.gray_difference_canvas)
                          if kw.get("fill") == "#3a3a3a"]
                if 0.0 <= right_x <= cw:
                    self.assertTrue(border)
                    self.assertAlmostEqual(float(border[0][0]), right_x, places=6)
                else:
                    self.assertFalse(border)

    def test_curve_y_encodes_signed_difference(self):
        # 偶行左台阶、奇行右台阶：均值剖面两处上升台阶，y 必须把高度差画出来。
        # 面板默认宽 640 覆盖整幅 64px 图，曲线不被视口裁断成多段。
        image = make_image(height=16, width=64, pattern="hetero_mix")
        width = image.shape[1]
        main = FakeMainCanvas(root_x=0.0, canvasx_zero=0.0)
        state = build_curve_state(image, y_start=0, y_end=16, main=main, display_scale=1.0,
                                  gray_difference_mode_var=FakeEntryVar(
                                      value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        state.update_gray_difference_curve()
        diff = app.profile_adjacent_difference(image.astype(float).mean(axis=0))
        seg = curve_segments(state.gray_difference_canvas)[0].reshape(-1, 2)
        self.assertEqual(len(seg), len(diff))
        rising = int(np.argmax(diff))
        rising_right = width - 16 - 1
        # 右台阶后紧跟的平段（差分为 0）用于验证“台阶点画得更高”。
        plateau = int(np.flatnonzero(diff[rising_right + 1:] == 0)[0]) + rising_right + 1
        # 偶行 40→140 @12、奇行 40→140 @width-16：均值在左右各出现 +50 上升台阶。
        self.assertAlmostEqual(diff[rising], 50.0)
        self.assertAlmostEqual(diff[rising_right], 50.0)
        y_at = {int(round(float(x) - 0.5)): float(y) for x, y in seg}
        self.assertLess(y_at[rising], y_at[plateau],
                        "step must be drawn higher than the flat level after it")
        self.assertLess(y_at[rising_right], y_at[plateau])
        # 零轴虚线：数据全为 0 的列画在下方，台阶峰值画在上方（图形方向一致）。
        zero_lines = [coords for coords, kw in line_items(state.gray_difference_canvas)
                      if kw.get("fill") == "#555555"]
        self.assertTrue(zero_lines)
        zero_y = float(zero_lines[0][1])
        self.assertLess(y_at[rising], y_at[plateau])
        self.assertLessEqual(y_at[rising], zero_y)
        self.assertGreaterEqual(y_at[plateau], zero_y)

    def test_horizontal_ticks_map_through_same_transform(self):
        image = make_image(height=16, width=400, pattern="step_left")
        main = FakeMainCanvas(root_x=120.0, canvasx_zero=0.0)
        state = build_curve_state(image, y_start=0, y_end=16, main=main,
                                  display_scale=1.0, graph_root_x=20.0, width=640, height=160,
                                  gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        state.update_gray_difference_curve()
        texts = text_items(state.gray_difference_canvas)
        self.assertTrue(texts, "must draw readable tick labels")
        tick_texts = [t for t in texts if t[2].lstrip("-").isdigit()]
        self.assertGreaterEqual(len(tick_texts), 3)
        xs = [t[0] for t in tick_texts]
        self.assertEqual(xs, sorted(xs))
        for item in tick_texts:
            x, _y, label, _kw = item
            self.assertAlmostEqual(x, self._screen_x(state, float(label)), places=6)
        # 刻度间距 ≈ 100 屏幕 px（1/2/5 步长选取），不密集重叠。
        base = float(app.LERLWRApp.gray_difference_nice_tick_step(state.display_scale))
        offsets = sorted(float(t[2]) % base for t in tick_texts)
        self.assertTrue(all(off < 1e-9 for off in offsets), (base, offsets))
        gaps = np.diff(xs)
        self.assertTrue(np.all(gaps >= 60.0), gaps)
        self.assertTrue(any("ΔI" in t[2] for t in texts), "vertical ΔI unit label required")

    def test_horizontal_scroll_callback_moves_curve(self):
        image = make_image(height=16, width=400, pattern="step_left")
        main = FakeMainCanvas(root_x=100.0, canvasx_zero=0.0)
        state = build_curve_state(image, y_start=0, y_end=16, main=main,
                                  display_scale=1.0, graph_root_x=30.0, width=300, height=160,
                                  gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        queued = []
        state.after_idle = lambda fn: queued.append(fn) or f"after#{len(queued)}"
        bar = Mock()
        state.image_horizontal_scrollbar = bar
        state._gray_difference_xscroll(0.0, 0.5)
        bar.set.assert_called_once_with(0.0, 0.5)
        self.assertEqual(len(queued), 1)
        queued.pop()()
        def first_point_x():
            return float(curve_segments(state.gray_difference_canvas)[0].reshape(-1, 2)[0][0])

        first_before = first_point_x()
        # 真实滚动：canvasx(0) 变为 100，再跑一次已排队的回调。
        main._canvasx_zero = 100.0
        state._gray_difference_xscroll(0.25, 0.75)
        self.assertEqual(len(queued), 1)
        queued.pop()()
        # 向右滚动 100 屏幕 px：origin 变为 30 − 100 = −70，曲线整体左移 100。图像左
        # 边缘外的点（x_img < 0，即 x+0.5 < 0 的半像素列）已全部离屏，首点是被视口
        # 左边界 0 裁剪后的第一个可见点。
        first_after = first_point_x()
        self.assertAlmostEqual(first_after, 0.0, places=6)
        # 像素刻度也必须同样左移 100 px。
        tick_before = [t[0] for t in text_items(state.gray_difference_canvas)
                       if t[2].lstrip("-").isdigit()]
        self.assertTrue(tick_before)
        for tx in tick_before:
            self.assertAlmostEqual(tx % 100.0, 70.0, places=6)
        main._canvasx_zero = 0.0
        state.update_gray_difference_curve()
        tick_unscrolled = [t[0] for t in text_items(state.gray_difference_canvas)
                           if t[2].lstrip("-").isdigit()]
        shifted = sorted(set(round(x, 6) for x in tick_unscrolled))
        expected = sorted(set(round(x - 100.0, 6) for x in shifted))
        self.assertTrue(set(round(x, 6) for x in tick_before) & set(expected))

    def test_offscreen_points_clipped_blank_right_preserved(self):
        image = make_image(height=16, width=200, pattern="step_left")
        main = FakeMainCanvas(root_x=0.0, canvasx_zero=0.0)
        state = build_curve_state(image, y_start=0, y_end=16, main=main,
                                  display_scale=1.0, width=80, height=160,
                                  gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        state.update_gray_difference_curve()
        segments = curve_segments(state.gray_difference_canvas)
        self.assertTrue(segments)
        all_x = [float(p[0]) for seg in segments for p in seg.reshape(-1, 2)]
        self.assertLessEqual(max(all_x), 80.0)
        # 图像本身宽于视口：图像右界竖线不可见，右侧不补画任何线。
        self.assertFalse([c for c, kw in line_items(state.gray_difference_canvas)
                          if kw.get("fill") == "#3a3a3a"])
        for text in text_items(state.gray_difference_canvas):
            self.assertLessEqual(text[0], 80.0)
        # 图像宽于视口：像素刻度只落在可见视口内。
        tick_labels = [t for t in text_items(state.gray_difference_canvas)
                       if t[2].lstrip("-").isdigit()]
        self.assertTrue(tick_labels)
        self.assertTrue(all(t[0] <= 80.0 for t in tick_labels))

    def test_gap_between_image_and_blank_right_is_not_bridged(self):
        # 图像 40px、面板 200px：曲线不得延伸到 x>40 的空白区。
        image = make_image(height=16, width=40, pattern="step_left")
        main = FakeMainCanvas(root_x=0.0, canvasx_zero=0.0)
        state = build_curve_state(image, y_start=0, y_end=16, main=main,
                                  display_scale=1.0, width=200, height=160,
                                  gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        state.update_gray_difference_curve()
        xs = [float(p[0]) for seg in curve_segments(state.gray_difference_canvas)
              for p in seg.reshape(-1, 2)]
        self.assertTrue(xs)
        # 差分长度 n = width-1，末点 x = (n-1)+0.5 = width-1.5。
        self.assertAlmostEqual(max(xs), self._screen_x(state, image.shape[1] - 1.5), places=6)
        self.assertLess(max(xs), 200.0)

    def test_curve_preserves_signed_edges_and_half_pixel_sampling(self):
        # 已知台阶在 x=12（0-based 列），差分峰值落在 diff[12]=+80；
        # 数据点画在半像素 (idx+0.5) 处，符号必须体现在 canvas y 上。
        image = make_image(height=16, width=64, pattern="step_left")
        main = FakeMainCanvas(root_x=0.0, canvasx_zero=0.0)
        state = build_curve_state(image, y_start=0, y_end=16, main=main, display_scale=1.0,
                                  gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        state.update_gray_difference_curve()
        seg = curve_segments(state.gray_difference_canvas)[0].reshape(-1, 2)
        diff = app.profile_adjacent_difference(image.astype(float).mean(axis=0))
        self.assertEqual(len(seg), len(diff))
        self.assertAlmostEqual(diff[11], 82.0)
        self.assertAlmostEqual(float(seg[0][0]), 0.5, places=6)
        self.assertAlmostEqual(float(seg[11][0]), 11.5, places=6)
        self.assertLess(float(seg[11][1]), float(seg[12][1]))

    def test_no_left_plot_margin_shifts_data(self):
        image = make_image(height=16, width=64, pattern="step_left")
        main = FakeMainCanvas(root_x=0.0, canvasx_zero=0.0)
        state = build_curve_state(image, y_start=0, y_end=16, main=main, display_scale=1.0,
                                  gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        state.update_gray_difference_curve()
        first_x = min(float(p[0]) for seg in curve_segments(state.gray_difference_canvas)
                      for p in seg.reshape(-1, 2))
        self.assertAlmostEqual(first_x, 0.5, places=3)

    def test_unlaid_out_widgets_do_not_invent_alignment(self):
        image = make_image(height=16, width=64, pattern="step_left")
        # 未映射的主图：不画曲线，也不报错，状态文字说明。
        state = build_curve_state(image, y_start=0, y_end=16,
                                  main=FakeMainCanvas(root_x=0.0, mapped=False))
        state.update_gray_difference_curve()
        self.assertFalse(curve_segments(state.gray_difference_canvas))
        self.assertIn("主图未布局", state.gray_difference_status_var.get())

        class NoRootX(FakeCanvas):
            winfo_rootx = None

        state2 = build_curve_state(image, y_start=0, y_end=16, main=FakeMainCanvas())
        state2.gray_difference_canvas = NoRootX(640, 160)
        state2.update_gray_difference_curve()
        self.assertFalse(curve_segments(state2.gray_difference_canvas))

        class RaisingRootX(FakeCanvas):
            def winfo_rootx(self):
                raise app.tk.TclError("not laid out")

        state3 = build_curve_state(image, y_start=0, y_end=16, main=FakeMainCanvas())
        state3.gray_difference_canvas = RaisingRootX(640, 160)
        state3.update_gray_difference_curve()
        self.assertFalse(curve_segments(state3.gray_difference_canvas))

    def test_tiny_viewport_is_safe(self):
        image = make_image(height=16, width=64, pattern="step_left")
        main = FakeMainCanvas(root_x=0.0, canvasx_zero=0.0)
        state = build_curve_state(image, y_start=0, y_end=16, main=main,
                                  display_scale=1.0, width=5, height=4)
        state.update_gray_difference_curve()
        for coords, _kw in line_items(state.gray_difference_canvas):
            self.assertTrue(np.all(np.isfinite(coords)))


class CachingAndLifecycleTests(unittest.TestCase):
    def test_cache_reused_across_redraw_same_source_and_range(self):
        image = make_image(height=16, width=40, pattern="ramp")
        state = build_curve_state(image, y_start=0, y_end=16,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        with patch.object(app, "profile_mean_intensity", wraps=app.profile_mean_intensity) as sp:
            state.update_gray_difference_curve()
            state.update_gray_difference_curve()
            self.assertEqual(sp.call_count, 1, "same source+range must reuse cached mean")

    def test_cache_invalidates_on_range_change_no_stale(self):
        image = make_image(height=16, width=40, pattern="ramp")
        state = build_curve_state(image, y_start=0, y_end=16,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.update_gray_difference_curve()
        state.gray_difference_y_start = 4
        state.gray_difference_y_end = 12
        with patch.object(app, "profile_mean_intensity", wraps=app.profile_mean_intensity) as sp:
            state.update_gray_difference_curve()
            self.assertEqual(sp.call_count, 1, "range change must recompute")
            np.testing.assert_allclose(state._gray_diff_cached_mean,
                                       image[4:12, :].astype(float).mean(axis=0))

    def test_cache_invalidates_on_source_switch_no_stale(self):
        image_a = make_image(height=16, width=40, pattern="ramp")
        state = build_curve_state(image_a, y_start=0, y_end=16,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.update_gray_difference_curve()
        # 新源数组（id 与旧数组不同），不改缓存键也必须重算。
        image_b = make_image(height=16, width=40, pattern="falling")
        state.processed_image = image_b
        with patch.object(app, "profile_mean_intensity", wraps=app.profile_mean_intensity) as sp:
            state.update_gray_difference_curve()
            self.assertEqual(sp.call_count, 1)
            np.testing.assert_allclose(state._gray_diff_cached_mean,
                                       image_b.astype(float).mean(axis=0))
            self.assertIs(state._gray_diff_cache_source, image_b)

    def test_cache_rejects_recycled_id(self):
        image_a = make_image(height=16, width=40, pattern="ramp")
        state = build_curve_state(image_a, y_start=0, y_end=16,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.update_gray_difference_curve()
        mean_a = state._gray_diff_cached_mean.copy()
        # 模拟源数组被回收、id() 被另一数组复用：键相同但源对象不同。
        lookalike = state.processed_image
        state.processed_image = image_a.copy()
        state._gray_diff_cache_key = (id(state.processed_image), 0, 16)
        state._gray_diff_cache_source = lookalike
        got = state.gray_difference_cached_mean(0, 16)
        np.testing.assert_allclose(got, mean_a)  # 数值相同但必须重算而非旧对象
        self.assertIs(state._gray_diff_cache_source, state.processed_image)

    def test_rebuild_invalidation_clears_cache(self):
        state = SimpleNamespace(processed_image=None,
                                _gray_diff_cache_key=(1, 0, 1),
                                _gray_diff_cached_mean=np.zeros(3),
                                _gray_diff_cache_source=np.zeros(3))
        state.invalidate_gray_difference_cache = MethodType(
            app.LERLWRApp.invalidate_gray_difference_cache, state)
        state.invalidate_gray_difference_cache()
        self.assertIsNone(state._gray_diff_cache_key)
        self.assertIsNone(state._gray_diff_cached_mean)
        self.assertIsNone(state._gray_diff_cache_source)


class RealPanelConstructionTests(unittest.TestCase):
    """Build the real panel/split and wire the real main-canvas callbacks.

    只伪造 Tk 控件/图片设施（Frame/Entry/Button/Canvas/StringVar/PhotoImage）；
    面板构建、范围解析、平均+差分、绘图、滚动/Configure 回调全部是生产代码。
    """

    def _make_state(self, image, canvas_width=420, graph_root_x=0.0, main_root_x=None):
        state = SimpleNamespace(
            processed_image=image, raw_image=image, original_image=image,
            gray_difference_y_start=None, gray_difference_y_end=None,
            gray_difference_photo=None, gray_difference_canvas=None,
            gray_difference_start_entry_var=None, gray_difference_end_entry_var=None,
            gray_difference_status_var=None, gray_difference_mode_var=None,
            gray_difference_error=None,
            _gray_diff_redraw_pending=False, _gray_diff_cache_key=None,
            _gray_diff_cached_mean=None, _gray_diff_cache_source=None,
            _gray_diff_after_token=None, canvas=None, display_scale=1.0,
            image_horizontal_scrollbar=None, main_root_x=main_root_x,
        )
        for name in (
            "gray_difference_resolved_range", "gray_difference_cached_mean",
            "invalidate_gray_difference_cache", "gray_difference_view_range",
            "update_gray_difference_curve",
            "_schedule_gray_difference_redraw", "_cancel_gray_difference_redraw",
            "_clear_gray_difference", "_build_gray_difference_split",
            "build_gray_difference_panel", "_gray_difference_main_root_x",
            "_gray_difference_xscroll", "gray_difference_widget_root_x",
            "gray_difference_nice_tick_step", "gray_difference_mode",
            "gray_difference_mode_value", "gray_difference_mode_changed",
        ):
            # 静态方法保持原样（绑定成 MethodType 会多传 self）。
            descriptor = app.LERLWRApp.__dict__[name]
            if isinstance(descriptor, staticmethod):
                setattr(state, name, descriptor.__func__)
            else:
                setattr(state, name, MethodType(descriptor, state))
        state.idle_queue = []
        state.cancelled = []
        state.after_idle = lambda fn: state.idle_queue.append(fn) or f"after#{len(state.idle_queue)}"
        state.after_cancel = lambda token: state.cancelled.append(token)
        state._graph_root_x = graph_root_x
        return state

    def _run_panel(self, image, canvas_width=420, graph_root_x=0.0, main=None,
                   main_root_x=None):
        state = self._make_state(image, canvas_width, graph_root_x, main_root_x)
        state.canvas = main if main is not None else FakeMainCanvas(
            root_x=100.0 if state.main_root_x is None else state.main_root_x,
            canvasx_zero=0.0)
        created = {}

        def make_canvas(parent=None, **kwargs):
            if "canvas" not in created:
                canvas = FakeTkCanvas(canvas_width, kwargs.get("height", 160),
                                      root_x=state._graph_root_x)
                canvas.parent = parent
                created["canvas"] = canvas
            return created["canvas"]

        def make_fake_entry(parent=None, **kwargs):
            created.setdefault("entry_count", 0)
            created["entry_count"] += 1
            entry = FakeEntry(parent, **kwargs)
            created.setdefault("entries", []).append(entry)
            return entry

        def make_fake_button(parent=None, **kwargs):
            created.setdefault("buttons", []).append(FakeButton(parent, **kwargs))
            created["button"] = created["buttons"][-1]
            return created["button"]

        def make_fake_combobox(parent=None, **kwargs):
            box = FakeCombobox(parent, **kwargs)
            created["combobox"] = box
            if kwargs.get("textvariable") is state.gray_difference_mode_var:
                created["mode_selector"] = box
            return box

        with patch.object(app.ttk, "Frame", FakeTkChild), \
                patch.object(app.ttk, "LabelFrame", FakeTkChild), \
                patch.object(app.ttk, "PanedWindow", FakeTkChild), \
                patch.object(app.ttk, "Label", FakeTkChild), \
                patch.object(app.ttk, "Entry", make_fake_entry), \
                patch.object(app.ttk, "Combobox", make_fake_combobox), \
                patch.object(app.ttk, "Button", make_fake_button), \
                patch.object(app.tk, "Canvas", make_canvas), \
                patch.object(app.tk, "StringVar", FakeEntryVar), \
                patch.object(app.ImageTk, "PhotoImage", FakePhoto):
            frame = FakeTkChild()
            builds = []
            image_pane = state._build_gray_difference_split(
                frame, lambda parent: (builds.append(parent), state.build_gray_difference_panel(parent)))
        self.assertIsInstance(image_pane, FakeTkChild)
        self.assertEqual(len(builds), 1)
        difference_pane = builds[0]
        self.assertIsNot(difference_pane, image_pane)
        self.assertEqual(len(image_pane.master.panes), 2)
        self.assertIn(difference_pane, image_pane.master.panes)
        # 主 canvas 用真实生产接线（与 _build_ui 相同）挂到同一个面板状态上。
        # 面板构建期（<Configure> 后）主图已布局，故构建前就把 main 放进 state。
        main = state.canvas
        main.configure(xscrollcommand=state._gray_difference_xscroll)
        main.bind("<Configure>", lambda _event: state._schedule_gray_difference_redraw(), add="+")
        state._schedule_gray_difference_redraw()
        state.idle_queue.pop()()
        state.image_horizontal_scrollbar = Mock()
        return state, created, image_pane, difference_pane

    def test_split_uses_panes_query_only_and_distinct_parents(self):
        image = make_image(height=16, width=48, pattern="step_left")
        with patch.object(app.ttk.PanedWindow, "panes") as panes:
            panes.side_effect = lambda self: []
            state, created, _image_pane, difference_pane = self._run_panel(image)
        panes.assert_not_called()
        self.assertIs(created["canvas"].parent, difference_pane)
        self.assertIn("<Configure>", created["canvas"].bindings)
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end), (0, 16))
        self.assertTrue(any(it[0] == "line" for it in created["canvas"].items))
        self.assertTrue(text_items(created["canvas"]), "tick labels drawn on first build")

    def test_entry_return_and_apply_button_drive_real_curve(self):
        image = make_image(height=16, width=48, pattern="ramp")
        state, created, _image_pane, _difference_pane = self._run_panel(image)
        canvas = created["canvas"]
        self.assertEqual(state.gray_difference_start_entry_var.get(), "0")
        self.assertEqual(state.gray_difference_end_entry_var.get(), "16")
        start_entry, end_entry = created["entries"][0], created["entries"][1]
        state.gray_difference_start_entry_var.set("3")
        state.gray_difference_end_entry_var.set("10")
        start_entry.bindings["<Return>"](None)
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end), (3, 10))
        np.testing.assert_allclose(state._gray_diff_cached_mean,
                                   image[3:10, :].astype(float).mean(axis=0))
        state.gray_difference_start_entry_var.set("4")
        state.gray_difference_end_entry_var.set("9")
        end_entry.bindings["<Return>"](None)
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end), (4, 9))
        # 真实「应用」按钮 + 非法输入：清画布并报错，范围不变。
        state.gray_difference_start_entry_var.set("abc")
        created["buttons"][0].command()
        self.assertEqual(state.gray_difference_y_start, 4)
        self.assertEqual(canvas.items, [])
        self.assertIn("范围", state.gray_difference_status_var.get())
        state.gray_difference_start_entry_var.set("2")
        state.gray_difference_end_entry_var.set("6")
        created["buttons"][0].command()
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end), (2, 6))
        self.assertTrue(curve_segments(canvas))

    def test_invalid_range_survives_main_configure_and_resize(self):
        image = make_image(height=16, width=40, pattern="ramp")
        state, created, _image_pane, _difference_pane = self._run_panel(image)
        canvas = created["canvas"]
        state.gray_difference_start_entry_var.set("999")
        state.gray_difference_end_entry_var.set("6")
        created["buttons"][0].command()
        self.assertEqual(canvas.items, [])
        self.assertTrue(state.gray_difference_error)
        # 主 canvas 与图 canvas 的真实 <Configure> 回调：各排一次 idle，错误保持。
        state.canvas.bindings["<Configure>"](None)
        canvas.bindings["<Configure>"](None)
        self.assertEqual(len(state.idle_queue), 1)
        state.idle_queue.pop()()
        self.assertEqual(canvas.items, [])
        self.assertTrue(state.gray_difference_error)
        # 改成合法范围后恢复。
        state.gray_difference_start_entry_var.set("1")
        state.gray_difference_end_entry_var.set("5")
        created["buttons"][0].command()
        self.assertIsNone(state.gray_difference_error)
        self.assertTrue(curve_segments(canvas))

    def test_main_scroll_and_zoom_callbacks_realign_curve(self):
        image = make_image(height=16, width=400, pattern="step_left")
        main = FakeMainCanvas(root_x=100.0, canvasx_zero=0.0)
        state, created, _image_pane, _difference_pane = self._run_panel(
            image, canvas_width=300, graph_root_x=40.0, main=main)
        canvas = created["canvas"]
        # 本测试验证差值半像素映射（首点 x=0.5）：显式选择差值，不依赖默认值。
        selector = created["mode_selector"]
        state.gray_difference_mode_var.set(app.GRAY_DIFFERENCE_MODE_DIFFERENCE)
        selector.bindings["<<ComboboxSelected>>"](SimpleNamespace(widget=selector))

        def first_x():
            xs = [float(p[0]) for seg in curve_segments(canvas) for p in seg.reshape(-1, 2)]
            return min(xs)

        baseline = first_x()
        # 水平滚动：生产 xscrollcommand → 排一次 idle → 执行后曲线左移 canvasx 差值。
        main.xscrollcommand(0.2, 0.7)
        state.image_horizontal_scrollbar.set.assert_called_once_with(0.2, 0.7)
        main._canvasx_zero = 50.0
        state.idle_queue.pop()()
        self.assertAlmostEqual(first_x() - baseline, -50.0, places=6)
        # 缩放：display_scale 变化后主 canvas <Configure> 触发真实重排。
        state.display_scale = 0.5
        main.bindings["<Configure>"](None)
        state.idle_queue.pop()()
        expected = 100.0 + 0.5 * 0.5 - 50.0 - 40.0
        self.assertAlmostEqual(first_x(), expected, places=6)
        main._root_x = 150.0
        # 根偏移变化（窗口移动/分栏拖动）同样经 Configure 重绘。
        canvas.bindings["<Configure>"](None)
        state.idle_queue.pop()()
        self.assertAlmostEqual(first_x(), expected + 50.0, places=6)

    def test_rebuild_source_swap_through_production_invalidation(self):
        image_a = make_image(height=16, width=40, pattern="ramp")
        state, created, _image_pane, _difference_pane = self._run_panel(image_a)
        canvas = created["canvas"]
        cached_a = state._gray_diff_cached_mean.copy()
        image_b = make_image(height=16, width=40, pattern="falling")
        state.processed_image = image_b
        state.invalidate_gray_difference_cache()
        state.update_gray_difference_curve()
        self.assertIsNot(state._gray_diff_cached_mean, None)
        np.testing.assert_allclose(state._gray_diff_cached_mean,
                                   image_b.astype(float).mean(axis=0))
        self.assertFalse(np.allclose(state._gray_diff_cached_mean, cached_a))
        # 缓存源引用随新源更新（不残留旧源强引用）。
        self.assertIs(state._gray_diff_cache_source, image_b)

    def test_dog_preview_never_reaches_panel_curve(self):
        image = make_image(height=16, width=48, pattern="step_left")
        state = build_curve_state(image, y_start=0, y_end=16,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0),
                                  gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        state.bcp_preview_image = 255 - image
        state.bcp_use_dog_var = FakeEntryVar(True)
        state.update_gray_difference_curve()
        expected = app.profile_adjacent_difference(image.astype(float).mean(axis=0))
        np.testing.assert_allclose(app.profile_adjacent_difference(state._gray_diff_cached_mean),
                                   expected)
        np.testing.assert_array_equal(state.processed_image, image)


class ModeSwitchTests(RealPanelConstructionTests):
    """下拉切换 平均灰度 ⇄ 平均灰度差：真实 panel 构建 + 真实回调 + 真实数据/坐标。

    fake 只替换 Tk 显示设施（Frame/LabelFrame/PanedWindow/Label/Entry/Combobox/
    Button/Canvas/StringVar）；面板构建、模式解析、行平均缓存、差分、绘图坐标与
    回调接线全部走生产代码，绘图目标不做任何 mock。
    """

    def _run_panel(self, image, **kwargs):
        return super()._run_panel(image, **kwargs)

    def _select(self, state, created, label):
        """Simulate the user picking ``label`` in the readonly Combobox."""
        selector = created["mode_selector"]
        self.assertIn(label, selector.values)
        self.assertEqual(selector.state, "readonly")
        state.gray_difference_mode_var.set(label)
        selector.bindings["<<ComboboxSelected>>"](SimpleNamespace(widget=selector))

    def _points(self, canvas):
        # 曲线点 = 折线段数据点 + 单点标记（oval 中心）；单列灰度只有一个点，
        # 生产代码用 create_oval 画（Tk create_line 不接受单点）。
        points = [np.asarray(seg, dtype=float).reshape(-1, 2) for seg in curve_segments(canvas)]
        for x0, y0, x1, y1, _kw in curve_dots(canvas):
            points.append(np.asarray([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]]))
        return points

    def _all_points(self, canvas):
        pts = [p for seg in self._points(canvas) for p in seg]
        return pts

    def _texts(self, canvas):
        return [item[2] for item in text_items(canvas)]

    def test_panel_builds_combobox_defaults_to_intensity(self):
        image = make_image(height=16, width=48, pattern="step_left")
        state, created, _image_pane, difference_pane = self._run_panel(image)
        selector = created["mode_selector"]
        self.assertEqual(selector.values, (app.GRAY_DIFFERENCE_MODE_INTENSITY,
                                           app.GRAY_DIFFERENCE_MODE_DIFFERENCE))
        labels = (app.GRAY_DIFFERENCE_MODE_INTENSITY, app.GRAY_DIFFERENCE_MODE_DIFFERENCE)
        self.assertEqual(selector.values, labels)
        # 新打开/首次构建：无已有选择 → 默认「平均灰度」，仍可用下拉切到差值。
        self.assertEqual(state.gray_difference_mode_var.get(), app.GRAY_DIFFERENCE_MODE_INTENSITY)
        self.assertEqual(state.gray_difference_mode(), app.GRAY_DIFFERENCE_MODE_INTENSITY)
        self.assertIn("<<ComboboxSelected>>", selector.bindings)
        # 默认绘制真实 I：长度 width，状态说明标注“平均灰度”。
        expected = image.astype(float).mean(axis=0)
        pts = self._all_points(created["canvas"])
        self.assertEqual(len(pts), len(expected))
        self.assertIn("平均行 0..16", state.gray_difference_status_var.get())
        self.assertIn("平均灰度｜", state.gray_difference_status_var.get())
        self.assertIn("I 点数 48", state.gray_difference_status_var.get())
        # 下拉切到差值仍正确。
        self._select(state, created, app.GRAY_DIFFERENCE_MODE_DIFFERENCE)
        diff = np.diff(image.astype(float).mean(axis=0))
        pts_diff = self._all_points(created["canvas"])
        self.assertEqual(len(pts_diff), len(diff))
        self.assertIn("平均灰度差", state.gray_difference_status_var.get())
        self.assertIn("ΔI 点数 47", state.gray_difference_status_var.get())

    def test_combobox_callback_switches_real_data_and_coords(self):
        image = make_image(height=16, width=48, pattern='hetero_mix')
        main = FakeMainCanvas(root_x=100.0, canvasx_zero=0.0)
        # 先显式选差值再验证“默认已非差值”：切到灰度/切回差值的回调都要真实执行。
        state, created, _image_pane, _difference_pane = self._run_panel(
            image, canvas_width=420, graph_root_x=40.0, main=main)
        canvas = created["canvas"]
        mean = image.astype(float).mean(axis=0)
        diff = np.diff(mean)
        origin = 100.0 - 40.0
        # 初始（默认平均灰度）：长度 width，x = 列整数。
        pts_initial = self._all_points(canvas)
        self.assertEqual(len(pts_initial), image.shape[1])
        self.assertEqual(state.gray_difference_mode(), app.GRAY_DIFFERENCE_MODE_INTENSITY)
        # 下拉切到差值：x = idx + 0.5，y 越低值越大；计数 width−1。
        self._select(state, created, app.GRAY_DIFFERENCE_MODE_DIFFERENCE)
        pts_diff = self._all_points(canvas)
        self.assertEqual(len(pts_diff), image.shape[1] - 1)
        for idx in (0, 5, 20, len(diff) - 1):
            hit = [p for p in pts_diff if abs(p[0] - (origin + idx + 0.5)) < 1e-9]
            self.assertEqual(len(hit), 1, idx)
        self._assert_y_order(pts_diff, diff)
        self._assert_ticks(canvas, "ΔI")
        # 切到平均灰度：真实回调 → 清旧曲线后绘 I，长度 width，x = 列整数。
        self._select(state, created, app.GRAY_DIFFERENCE_MODE_INTENSITY)
        self.assertEqual(state.gray_difference_mode(), app.GRAY_DIFFERENCE_MODE_INTENSITY)
        self.assertIsNone(state.gray_difference_error)
        self.assertIn("平均灰度｜", state.gray_difference_status_var.get())
        pts_int = self._all_points(canvas)
        self.assertEqual(len(pts_int), image.shape[1])
        self.assertEqual(len(self._points(canvas)), 1, "single mode drawn, no overlay")
        xs = sorted(p[0] for p in pts_int)
        np.testing.assert_allclose(xs, [origin + x for x in range(image.shape[1])])
        self._assert_y_order(pts_int, mean)
        self._assert_ticks(canvas, "I")
        caption = state.gray_difference_status_var.get()
        self.assertIn("平均灰度｜", caption)
        self.assertIn("I 点数 48", caption)
        self.assertIn("列坐标", caption)
        # 切回差值：坐标/长度恢复 width−1。
        self._select(state, created, app.GRAY_DIFFERENCE_MODE_DIFFERENCE)
        pts_back = self._all_points(canvas)
        self.assertEqual(len(pts_back), image.shape[1] - 1)
        self._assert_y_order(pts_back, diff)

    def _assert_y_order(self, points, values):
        """数据值越大 ⇒ 屏幕 y 越小（纵轴自适应，方向正确）。"""
        pairs = list(zip(values, [p[1] for p in points]))
        for vai, yai in pairs:
            for vbi, ybi in pairs:
                if vai > vbi + 1e-9:
                    self.assertLess(yai, ybi, (vai, vbi))

    def _assert_ticks(self, canvas, unit):
        """纵轴单位标注使用当前模式的单位，且不残留另一模式的标签。"""
        texts = self._texts(canvas)
        self.assertTrue(any(t.startswith(f"{unit} ") for t in texts), texts)
        other = "I" if unit == "ΔI" else "ΔI"
        self.assertFalse(any(t.startswith(f"{other} ") for t in texts), texts)

    def test_switch_reuses_cached_mean_without_re_averaging(self):
        image = make_image(height=16, width=48, pattern="ramp")
        state, created, _image_pane, _difference_pane = self._run_panel(image)
        y_start, y_end = state.gray_difference_y_start, state.gray_difference_y_end
        cached = state._gray_diff_cached_mean
        calls = []
        real_mean = app.profile_mean_intensity

        def counting_mean(image_arg, ys, ye):
            calls.append((ys, ye))
            return real_mean(image_arg, ys, ye)

        with patch.object(app, "profile_mean_intensity", counting_mean):
            self._select(state, created, app.GRAY_DIFFERENCE_MODE_INTENSITY)
            self._select(state, created, app.GRAY_DIFFERENCE_MODE_DIFFERENCE)
            self._select(state, created, app.GRAY_DIFFERENCE_MODE_INTENSITY)
            state.display_scale = 0.5
            state.canvas.bindings["<Configure>"](None)
            state.idle_queue.pop()()
        self.assertEqual(calls, [], "模式切换/缩放不得重新做整图平均")
        self.assertIs(state._gray_diff_cached_mean, cached)
        self.assertEqual((state.gray_difference_y_start, state.gray_difference_y_end),
                         (y_start, y_end))

    def test_mode_switch_keeps_no_image_and_range_error_states(self):
        # 模式回调在未载图时安全：清空画布并保留“未载入图像”。
        state = build_curve_state(None, y_start=0, y_end=5)
        state.gray_difference_mode_var = FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_INTENSITY)
        state.gray_difference_mode_changed()
        self.assertEqual(state.gray_difference_canvas.items, [])
        self.assertIn("未载入图像", state.gray_difference_status_var.get())
        # 非法范围错误态：切模式只清画布并保留错误文字，不偷偷画曲线。
        image = make_image(height=16, width=40, pattern="ramp")
        state, created, _image_pane, _difference_pane = self._run_panel(image)
        state.gray_difference_start_entry_var.set("999")
        state.gray_difference_end_entry_var.set("6")
        created["buttons"][0].command()
        error_text = state.gray_difference_status_var.get()
        for label in (app.GRAY_DIFFERENCE_MODE_INTENSITY, app.GRAY_DIFFERENCE_MODE_DIFFERENCE):
            self._select(state, created, label)
            self.assertEqual(created["canvas"].items, [])
            self.assertEqual(state.gray_difference_status_var.get(), error_text)
        # 新图/预处理后 invalidate + 重绘仍按当前模式画对应长度。
        state.gray_difference_start_entry_var.set("")
        state.gray_difference_end_entry_var.set("")
        created["buttons"][0].command()
        self.assertIsNone(state.gray_difference_error)
        self.assertIsNotNone(state._gray_diff_cached_mean)
        self.assertEqual(len(self._all_points(created["canvas"])), 40 - 1)

    def test_mode_offsets_and_scroll_keep_production_mapping(self):
        image = make_image(height=16, width=200, pattern="ramp")
        # display_scale = 0.5：200 px 工作图宽 100 px，滚到 canvasx(0)=80 后
        # 右侧数据点仍在视口内，用于验证滚动后整条曲线按同一映射平移。
        main = FakeMainCanvas(root_x=120.0, canvasx_zero=30.0)
        state, created, _image_pane, _difference_pane = self._run_panel(
            image, canvas_width=1000, graph_root_x=50.0, main=main)
        state.display_scale = 0.5
        state.update_gray_difference_curve()
        canvas = created["canvas"]
        mean = image.astype(float).mean(axis=0)

        def xs_at(origin, scale, x_of, count, right_limit=float("inf")):
            # 生产只保留屏幕 x 落在面板视口 [0, 面板宽] 的数据点，并在可见带内把
            # 首尾点 clamp 到 0 / 面板宽（工作图 x = width 之外不连线）。
            xs = [origin + x_of(i) * scale for i in range(count)]
            visible = [x for x in xs if 0.0 <= x <= right_limit]
            clamped = [min(max(x, 0.0), float(wig)) for x in xs
                       if -1.0 <= x <= float(wig) + 1.0]
            return {round(x, 6) for x in visible}, {round(x, 6) for x in clamped}

        wig = float(created["canvas"]._width)

        for label, count, x_of in (
            (app.GRAY_DIFFERENCE_MODE_INTENSITY, len(mean), lambda i: float(i)),
            (app.GRAY_DIFFERENCE_MODE_DIFFERENCE, len(mean) - 1, lambda i: i + 0.5),
        ):
            # 每个子场景都从同一初始几何开始（构造后/上一轮复位后的状态）。
            self._select(state, created, label)
            drawn = {round(float(p[0]), 6) for p in self._all_points(canvas)}
            self.assertEqual(drawn, xs_at(40.0, 0.5, x_of, count)[1], label)
            # 水平滚动 → 生产 xscrollcommand → idle 重绘 → 整体左移 canvasx 差值。
            main._canvasx_zero = 80.0
            main.xscrollcommand(0.2, 0.7)
            state.idle_queue.pop()()
            self.assertEqual({round(float(p[0]), 6) for p in self._all_points(canvas)},
                             xs_at(-10.0, 0.5, x_of, count)[1], label)
            # 缩放 + 主图/图表 root 偏移变化（生产 <Configure> 合并重绘）；图像右界
            # 随之右移，灰度模式的最高实际列仍可落在界内。
            state.display_scale = 1.0
            main._root_x = 140.0
            canvas.set_root_x(20.0)
            canvas.bindings["<Configure>"](None)
            state.idle_queue.pop()()
            self.assertEqual({round(float(p[0]), 6) for p in self._all_points(canvas)},
                             xs_at(40.0, 1.0, x_of, count)[1], label)
            # 复位到与生产一致的初始几何，供下一次模式循环复用同一坐标基准。
            main._root_x = 120.0
            main._canvasx_zero = 30.0
            state.display_scale = 0.5
            canvas.set_root_x(50.0)
            state.update_gray_difference_curve()

    def test_difference_mode_narrow_image_still_errors(self):
        image = make_image(height=8, width=1, pattern="ramp")
        state = build_curve_state(image, y_start=0, y_end=8,
                                  gray_difference_mode_var=FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE),
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        state.update_gray_difference_curve()
        self.assertEqual(state.gray_difference_canvas.items, [])
        self.assertIn("过窄", state.gray_difference_status_var.get())
        # 灰度模式对单列图有明确的有限结果：一根列值，不报“过窄”。
        state.gray_difference_mode_var.set(app.GRAY_DIFFERENCE_MODE_INTENSITY)
        state.gray_difference_mode_changed()
        self.assertNotIn("过窄", state.gray_difference_status_var.get())
        self.assertIn("I 点数 1", state.gray_difference_status_var.get())
        self.assertEqual(len(self._all_points(state.gray_difference_canvas)), 1)
        # 单点不能用 create_line 画（FakeCanvas 已按真实 Tk 拒绝 <4 坐标）：生产代码
        # 必须落到合法原子图元上，且能看到该点本身。
        canvas = state.gray_difference_canvas
        self.assertFalse(curve_segments(canvas))
        dots = curve_dots(canvas)
        self.assertEqual(len(dots), 1)
        x0, y0, x1, y1, _kw = dots[0]
        self.assertAlmostEqual((x0 + x1) / 2.0, 0.0, places=6)  # 灰度 sample_x(0) = 0
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)
        self.assertTrue(all(np.isfinite(v) for v in (x0, y0, x1, y1)))

    def test_intensity_clipped_run_of_one_point_uses_legal_primitive(self):
        # 切到平均灰度后放大到视口内只容得下首列一个数据点、其余各列全部离屏：可见
        # 片段长度=1，仍必须画出该点，且只能走合法原子图元（FakeCanvas.create_line 已
        # 按真实 Tk 拒绝 <4 坐标，能走到断言即证明没有非法折线）。
        #
        # 视口只容一列是由**真实几何**推出的，不是假设：屏幕 x = origin + x*scale，
        # origin = 主图 winfo_rootx() − 图 canvas winfo_rootx() − canvasx(0)
        # （= 100.0 − 0.0 − 0.0）；可见条件是 origin + x*scale <= canvas.winfo_width()，
        # 故 scale 取 (canvas 宽 − origin) 向下取整到 100 的倍数：420 → 300 时首列
        # 100、第二列 400 仍在 420 视口内（两点折线），因此这里取 300 之后再加一档
        # 100，即 400：首列 100 可见，第二列 500 已离屏。
        image = make_image(height=8, width=301, pattern="ramp")
        state, created, _image_pane, _difference_pane = self._run_panel(image)
        canvas = state.gray_difference_canvas
        mode_before = state.gray_difference_mode_var.get()
        origin = (state.canvas.winfo_rootx() - canvas.winfo_rootx()
                  - state.canvas.canvasx(0))
        viewport = float(canvas.winfo_width())
        scale = float(np.ceil((viewport - origin) / 100.0) * 100)
        self.assertGreater(origin, 0.0)
        self.assertLessEqual(origin, viewport)
        self.assertEqual((origin, scale), (100.0, 400.0))
        self.assertGreater(origin + scale, viewport)  # 第二列离屏
        try:
            # 先切模式、再改 display_scale 后直接重绘：模式回调若在放缩前触发，测到的
            # 就是 scale=1 的整幅折线，而不是被裁成单点的可见片段。
            self._select(state, created, app.GRAY_DIFFERENCE_MODE_INTENSITY)
            state.gray_difference_y_start = 0
            state.gray_difference_y_end = 8
            state.display_scale = scale
            state.update_gray_difference_curve()

            for coords, _kw in line_items(canvas):
                self.assertGreaterEqual(len(coords), 4)
                self.assertEqual(len(coords) % 2, 0)
            dots = curve_dots(canvas)
            self.assertEqual(len(dots), 1)
            self.assertFalse(curve_segments(canvas))
            x0, y0, x1, y1, _kw = dots[0]
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            # 唯一可见列 x=0 → 屏幕 x = origin(100.0)；小点半径 1 px，故 99..101。
            self.assertAlmostEqual(cx, origin, places=6)
            self.assertAlmostEqual(x0, origin - 1.0, places=6)
            self.assertAlmostEqual(x1, origin + 1.0, places=6)
            self.assertGreater(x0, 0.0)
            self.assertLess(x1, viewport)
            # 纵坐标由真实数据独立推导：行平均剖面首列（pattern="ramp")→ 与生产同一
            # 映射（5% 纵向边距 + 顶部刻度带 / 底部单位标注带）。
            profile = image.astype(float).mean(axis=0)
            low, high = float(profile.min()), float(profile.max())
            pad = (high - low) * 0.05
            low, high = low - pad, high + pad
            top = 14.0 + 4.0
            bottom = max(top + 1.0, canvas.winfo_height() - 12.0 - 2.0)
            expected_y = bottom - (profile[0] - low) * (bottom - top) / (high - low)
            self.assertTrue(np.isfinite(cy))
            self.assertAlmostEqual(cy, expected_y, places=6)
            self.assertGreater(cy, top)
            self.assertLess(cy, bottom)
        finally:
            # 画完恢复正常缩放，避免影响同一实例后续检查。
            state.display_scale = 1.0
            state.gray_difference_mode_var.set(mode_before)

    def test_unconfigured_modevar_defaults_to_intensity(self):
        image = make_image(height=16, width=32, pattern="ramp")
        state = build_curve_state(image, y_start=None, y_end=None,
                                  main=FakeMainCanvas(root_x=0.0, canvasx_zero=0.0))
        # None / 非法值 fallback 与面板默认值一致：都是「平均灰度」。
        self.assertIsNone(state.gray_difference_mode_var)
        self.assertEqual(state.gray_difference_mode(), app.GRAY_DIFFERENCE_MODE_INTENSITY)
        self.assertEqual(app.LERLWRApp.gray_difference_mode_value(None),
                         app.GRAY_DIFFERENCE_MODE_INTENSITY)
        self.assertEqual(app.LERLWRApp.gray_difference_mode_value(FakeEntryVar("bogus")),
                         app.GRAY_DIFFERENCE_MODE_INTENSITY)
        # 合法选择原样保留（不被 fallback 覆盖）。
        self.assertEqual(app.LERLWRApp.gray_difference_mode_value(
            FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_DIFFERENCE)),
            app.GRAY_DIFFERENCE_MODE_DIFFERENCE)
        self.assertEqual(app.LERLWRApp.gray_difference_mode_value(
            FakeEntryVar(value=app.GRAY_DIFFERENCE_MODE_INTENSITY)),
            app.GRAY_DIFFERENCE_MODE_INTENSITY)
        state.update_gray_difference_curve()
        self.assertEqual(len(self._all_points(state.gray_difference_canvas)), 32)
        self.assertIn("平均灰度｜", state.gray_difference_status_var.get())



if __name__ == "__main__":
    unittest.main()
