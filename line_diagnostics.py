"""逐行定位诊断（line localization row diagnostics）— isolated, app-agnostic UI.

本模块提供「逐行定位诊断」窗口的核心构建逻辑，与主程序（``ler_lwr_app``）
解耦：它只 import ``line_localization`` 与第三方库（tkinter/Pillow/NumPy），
不 import ``ler_lwr_app``，因此不会造成循环依赖。主程序通过传入
``source_snapshot``（静态来源快照）、``draw_helper``（取灰度曲线/灰度差曲线
的纯函数）与 ``on_focus``（安全回调，跳到主 SEM 对应区域）来复用本模块。

设计要点：
- 来源快照在窗口打开时冻结（图像/样本/控件的拷贝），此后主图切到图 B 也不会
  让旧窗口去读或跳图 B。
- 列表针对单个样本 + 侧别显示每一行：global 行 / 候选 global x / 最终 global x
  首轮或恢复或缺失 / 失败原因；missing 不伪造 x。
- statistics 全部来自 ``line_localization.stats``，不另造。
- 选行时显示该 ROI 该行的真实灰度与 ΔI 曲线（global x 整数与半像素 x+0.5）、
  候选/最终标记或失败文字，并通过 ``on_focus`` 回调跳主 SEM 对应区域（只读，
  不改 ROI/采样/统计）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

# Tk/Pillow 在主模块顶部导入（本应用本来就依赖 Tk），便于测试对 ``ldiag.tk`` /
# ``ldiag.ttk`` / ``ldiag.ImageTk`` 打补丁，驱动真实 UI 回调；不新增任何依赖。
try:  # 无 Tk 的环境（纯核心测试）仍可 import 本模块的纯函数
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk, ImageDraw
except Exception:  # pragma: no cover - 仅在无 Tk/Pillow 的退化环境
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    Image = ImageTk = ImageDraw = None  # type: ignore[assignment]

from line_localization import (
    REASON_AMBIGUOUS,
    REASON_JUMP_REJECTED,
    REASON_LOW_CONTRAST,
    REASON_MISSING_FLANK,
    REASON_UNRESOLVED,
    RECOVERY_MIN_NEIGHBORS,
    RECOVERY_WINDOW_ROWS,
    STAGE_MISSING,
    STAGE_PRIMARY,
    STAGE_RELOCALIZED,
    EdgeLocalization,
    RowLocalization,
    failure_reason_summary,
    gap_runs,
    localization_stats,
    pair_localization_stats,
)

STAGE_LABELS = {
    STAGE_PRIMARY: "首轮",
    STAGE_RELOCALIZED: "恢复",
    STAGE_MISSING: "缺失",
}

REASON_LABELS = {
    REASON_LOW_CONTRAST: "低对比",
    REASON_MISSING_FLANK: "缺少侧翼",
    REASON_AMBIGUOUS: "歧义峰",
    REASON_JUMP_REJECTED: "跳变剔除",
    REASON_UNRESOLVED: "未解析",
    "": "—",
}


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage)


def reason_label(reason: str) -> str:
    return REASON_LABELS.get(reason, reason or "—")


DEFAULT_SOURCE_LABEL = "来源未记录"
CHART_CANDIDATE_COLOR = "#ffb45e"
CHART_FINAL_COLOR = "#5ef07a"
STALE_SOURCE_NOTICE = "来源已变更：仅显示打开时的快照，不跳转主图。"


@dataclass(frozen=True)
class SourceSnapshot:
    """Static, frozen capture of everything the diagnostic window may read.

    Opening the window copies the current image / samples / controls into this
    snapshot; the window then reads only the snapshot, so switching the main
    image to figure B never makes an old window read or jump to B.

    追加的来源字段（默认值保持向后兼容）只由主应用在打开时刻填写：真实源图路径、
    库名、唯一 id / python ``id()`` 身份、处理签名、标定、旋转、处理方式、载入时间、
    PSD 噪声上下文。诊断 CSV 只从这些冻结字段导出，绝不回读主窗口的 live 状态。
    """

    image: np.ndarray | None
    samples: tuple
    left_x: int = 0
    top_y: int = 0
    # Opaque live handle captured at open time (validated by the app callback).
    live: object | None = None
    source_path: str = ""
    source_name: str = ""
    source_id: str = ""
    source_identity: int | None = None
    preprocessing: str = ""
    calibration: str = ""
    rotation: str = ""
    image_size: str = ""
    processing: str = ""
    method: str = ""
    captured_at: str = ""
    noise_context: str = ""
    extra: tuple = field(default_factory=tuple)

    def source_label(self) -> str:
        path = self.source_path or self.source_name
        return path or DEFAULT_SOURCE_LABEL


def row_reason(row: RowLocalization) -> str:
    """Prefer the jump-rejection / ambiguity / missing-flank cause over a blank
    primary reason; a recovered row always reports '—' because recovery succeeded."""
    if row.stage == STAGE_RELOCALIZED:
        return ""
    if row.stage == STAGE_PRIMARY:
        return ""
    return row.reason or REASON_UNRESOLVED


def diag_row_records(diag: EdgeLocalization, *, left_x: int = 0) -> list[dict]:
    """One dict per ROI row with local + global coordinates and per-row identity.

    ``candidate_global_x`` / ``final_global_x`` are nan for missing rows (never a
    fabricated x).  ``recovered_from_global_x`` is the neighbor seed in global x
    for relocalized rows, else nan.
    """
    left, top, _right, _bottom = diag.roi_bounds_px
    origin = int(left) + int(left_x)
    records = []
    for row in diag.rows:
        candidate_gx = row.candidate_px + origin if math.isfinite(row.candidate_px) else float("nan")
        final_gx = row.final_px + origin if math.isfinite(row.final_px) else float("nan")
        recovered_gx = row.recovered_from_px + origin if math.isfinite(row.recovered_from_px) else float("nan")
        records.append({
            "global_row": int(row.row_px) + top,
            "local_row": int(row.row_px),
            "candidate_local_x": np.nan if not math.isfinite(row.candidate_px) else float(row.candidate_px),
            "final_local_x": np.nan if not math.isfinite(row.final_px) else float(row.final_px),
            "candidate_global_x": candidate_gx,
            "final_global_x": final_gx,
            "stage": row.stage,
            "stage_label": stage_label(row.stage),
            "reason": row_reason(row),
            "reason_label": reason_label(row_reason(row)),
            "recovered_from_global_x": recovered_gx,
        })
    return records


def diag_stat_rows(left_diag: EdgeLocalization | None, right_diag: EdgeLocalization | None) -> dict:
    """Reuse ``line_localization`` statistics verbatim (no separate re-derivation)."""
    left_stats = localization_stats(left_diag) if left_diag is not None else None
    right_stats = localization_stats(right_diag) if right_diag is not None else None
    pair_stats = pair_localization_stats(left_diag, right_diag) if (left_diag is not None and right_diag is not None) else None
    return {
        "left": left_stats,
        "right": right_stats,
        "pair": pair_stats,
        "left_summary": failure_reason_summary(left_stats, None) if left_stats is not None else "",
        "right_summary": failure_reason_summary(right_stats, None) if right_stats is not None else "",
        "pair_summary": failure_reason_summary(left_stats, pair_stats) if pair_stats is not None else "",
    }


def side_diagnostics(sample, side: str):
    """Resolve one side's ``EdgeLocalization`` (failed-side record first, else the edge)."""
    if side not in ("left", "right"):
        raise ValueError("side 只能是 left 或 right。")
    diag = getattr(sample, f"{side}_diagnostics", None)
    if diag is not None:
        return diag
    edge = getattr(sample, f"{side}_edge", None)
    if edge is not None and getattr(edge, "diagnostics", None) is not None:
        return edge.diagnostics
    return None


def _roi_origin_x(diag: EdgeLocalization, left_x: int) -> int:
    """Explicit ROI x origin: the ROI's real left edge plus any crop offset.

    Never derive the origin from the image width (the historical bug drew a
    right-aligned ROI as if it were ``rowwidth - len(intensity)``); if a diag
    carries no ROI bounds, fall back to the caller-supplied ``left_x``.
    """
    bounds = getattr(diag, "roi_bounds_px", None)
    if bounds:
        return int(bounds[0]) + int(left_x)
    return int(left_x)


def chart_sample_data(
    intensity: np.ndarray,
    *,
    rowwidth: int,
    origin_x: int = 0,
) -> dict:
    """Pure coordinates for the chart: ROI-origin-relative, never image-width-derived.

    ``gray_x[i] = origin_x + i`` (global x of pixel i) and ``delta_x[j] = origin_x + j + 0.5``
    (ΔI is the neighbour difference centred on the half-pixel j+0.5).  ``origin_x`` is the
    ROI's real left edge, NOT ``rowwidth - len(intensity)``.
    """
    values = np.asarray(intensity, dtype=float).ravel()
    diff = np.diff(values) if len(values) >= 2 else np.array([], dtype=float)
    origin = int(origin_x)
    return {
        "rowwidth": int(rowwidth),
        "origin_x": origin,
        "gray": values,
        "gray_x": np.arange(len(values), dtype=float) + origin,
        "delta": diff,
        "delta_x": np.arange(len(diff), dtype=float) + 0.5 + origin,
        "gray_span": (float(origin), float(origin + max(len(values) - 1, 0))),
        "delta_span": (float(origin) + 0.5, float(origin) + max(len(diff), 0) - 0.5) if len(diff) else (0.0, 0.0),
    }


def diag_chart_rgb(
    intensity: np.ndarray,
    *,
    rowwidth: int,
    title: str = "",
    size: tuple[int, int] = (640, 360),
    candidate_x: float | None = None,
    final_x: float | None = None,
    failure_text: str | None = None,
    origin_x: int = 0,
) -> np.ndarray:
    """Draw one ROI row's real grayscale 与 ΔI (x+0.5) curves with markers.

    ``origin_x`` is the ROI's real left edge (default 0 keeps old callers valid):
    gray x = ``origin_x + arange(n)`` and ΔI x = ``origin_x + 0.5 + arange(n-1)``.
    Candidate and final markers use different colours and labels; a missing row
    with only a candidate is drawn in the candidate colour and labelled
    「未接受」— a final marker is never fabricated for a NaN final x.
    Pure function → uint8 RGB.
    """
    values = np.asarray(intensity, dtype=float).ravel()
    view_width, view_height = size
    canvas = Image.new("RGB", size, "#181818")
    draw = ImageDraw.Draw(canvas)
    foreground = "#d9d9d9"
    grid = "#4a4a4a"
    split_y = view_height // 2
    data = chart_sample_data(values, rowwidth=rowwidth, origin_x=origin_x)
    candidate_finite = candidate_x is not None and math.isfinite(float(candidate_x))
    final_finite = final_x is not None and math.isfinite(float(final_x))
    markers = []
    if candidate_finite:
        markers.append(("候选", float(candidate_x), CHART_CANDIDATE_COLOR))
    if final_finite:
        markers.append(("最终", float(final_x), CHART_FINAL_COLOR))

    def render_pane(top: int, bottom: int, y_values, x_values, color, label, y_label):
        draw.rectangle((0, top, view_width, bottom), fill="#181818")
        left, right = 46, view_width - 12
        inner_top, inner_bottom = top + 20, bottom - 30
        finite = y_values[np.isfinite(y_values)]
        low, high = (np.percentile(finite, (1, 99)) if len(finite) else (0.0, 1.0))
        if high <= low:
            high = low + 1.0
        if len(x_values):
            x_low, x_high = float(x_values[0]), float(x_values[-1])
        else:
            x_low, x_high = float(data["origin_x"]), float(data["origin_x"]) + 1.0
        if math.isclose(x_low, x_high):
            x_high = x_low + 1.0
        for fraction in (0.0, 0.5, 1.0):
            gx = round(left + (right - left) * fraction)
            gy = round(inner_top + (inner_bottom - inner_top) * fraction)
            draw.line((gx, inner_top, gx, inner_bottom), fill=grid)
            draw.line((left, gy, right, gy), fill=grid)
        draw.rectangle((left, inner_top, right, inner_bottom), outline=foreground)
        for name, marker_x, marker_color in markers:
            if x_low <= marker_x <= x_high:
                m = round(left + (right - left) * (marker_x - x_low) / (x_high - x_low))
                draw.line((m, inner_top, m, inner_bottom), fill=marker_color, width=2)
                draw.text((min(m + 2, right - 60), inner_top + 2), f"{name} {marker_x:.2f}", fill=marker_color)
        points = []
        for idx, value in enumerate(y_values):
            if not np.isfinite(value):
                if len(points) >= 2:
                    draw.line(points, fill=color, width=2)
                points = []
                continue
            px = left + (float(x_values[idx]) - x_low) * (right - left) / (x_high - x_low)
            py = inner_bottom - (float(value) - low) * (inner_bottom - inner_top) / (high - low)
            points.append((round(px), round(np.clip(py, inner_top, inner_bottom))))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        draw.text((left + 5, top + 2), label, fill=color)
        draw.text((4, inner_top - 1), y_label, fill=foreground)
        draw.text((left, inner_bottom + 2), f"{low:.2g}", fill="#aaaaaa")
        draw.text((left + 3, inner_top + 2), f"{high:.2g}", fill="#aaaaaa")
        draw.text((left, inner_bottom + 14), f"{x_low:.2f}", fill="#aaaaaa")
        draw.text((right - 40, inner_bottom + 14), f"{x_high:.2f}", fill="#aaaaaa")

    if len(values):
        render_pane(0, split_y, data["gray"], data["gray_x"], "#7fd4ff", "灰度 I", "I")
        if len(data["delta"]):
            render_pane(split_y, view_height, data["delta"], data["delta_x"], "#ffd27f", "邻像素差 ΔI", "ΔI")
    if title:
        draw.text((4, 2), title, fill=foreground)
    if failure_text is not None:
        draw.text((view_width // 2 - 40, view_height // 2), failure_text, fill="#ff6b6b")
    if candidate_finite and not final_finite:
        draw.text((46, split_y - 14), "候选未接受（无最终位置）", fill=CHART_CANDIDATE_COLOR)
    return np.asarray(canvas, dtype=np.uint8)


def row_focus_region(record: dict) -> tuple[int, int, int, int]:
    """Focus region for one selected global row: the ROI's real columns (left,right) at
    that single global row (``top = global_row``, ``bottom = global_row + 1``)."""
    return (int(record["roi_left"]), int(record["global_row"]), int(record["roi_right"]), int(record["global_row"]) + 1)


def focus_capture(snapshot: SourceSnapshot) -> dict:
    """Frozen identity of the window's source, validated before any main-view jump."""
    return {
        "live": snapshot.live,
        "source_path": snapshot.source_path,
        "source_id": snapshot.source_id,
        "source_identity": snapshot.source_identity,
    }


def live_still_same(capture: dict, live) -> bool:
    """True only while the main app still holds the exact same image/processing object."""
    return (
        capture.get("live") is not None
        and live is not None
        and live is capture.get("live")
        and getattr(live, "is_same_live_source", lambda _c: False)(capture)
    )


def diag_stats_text(left_diag, right_diag, side: str, left_index: int = 1, right_index: int = 1) -> str:
    """Both sides' ``line_localization`` statistics plus the common-valid coverage.

    The same two diags always fill the LEFT and RIGHT slots (a selected right side is
    shown as '右侧'), never relabelled into the left slot.  Every count is printed with
    its denominator; all numbers come from ``localization_stats`` /
    ``pair_localization_stats`` with unchanged definitions.
    """
    stats = diag_stat_rows(left_diag, right_diag)
    parts = [f"样本 {left_index}；当前选中：{'左侧' if side == 'left' else '右侧'}"]
    for name, label, item in (("left", "左侧", stats["left"]), ("right", "右侧", stats["right"])):
        if item is None:
            parts.append(f"{label}：无诊断记录")
            continue
        reason_parts = "，".join(f"{reason_label(k)}={v}" for k, v in sorted(item.reasons.items()))
        parts.append(
            f"{label} ROI 总行 {item.total_rows}；有效 {item.primary_rows + item.recovered_rows}/{item.total_rows}"
            f"（首轮 {item.primary_rows}，恢复 {item.recovered_rows}，缺失 {item.missing_rows}，"
            f"覆盖 {item.coverage_fraction:.1%}）；最长缺口 {item.longest_gap_rows}/{item.total_rows} 行"
            + (f"；{reason_parts}" if reason_parts else "")
        )
    pair = stats["pair"]
    if pair is None:
        parts.append("两侧共同有效：两侧诊断不全，无法配对")
    elif not pair.overlap_rows:
        parts.append(f"两侧共同有效 0（纵向重叠 0/{pair.overlap_rows} 行）")
    else:
        parts.append(
            f"两侧共同有效 {pair.common_valid_rows}/{pair.overlap_rows} 行（覆盖 {pair.coverage_fraction:.1%}；"
            f"左有效 {pair.left_valid_rows}/{pair.left_total_rows}，右有效 {pair.right_valid_rows}/{pair.right_total_rows}；"
            f"最长共同缺口 {pair.longest_common_gap_rows} 行）"
        )
    return "\n".join(parts)


def _csv_value(value) -> str:
    """Full-precision float repr; blank for missing/non-finite (never 0, never truncated)."""
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return repr(number)


def _sample_side_index(sample) -> tuple[int, int]:
    return (int(getattr(sample, "era_line_index", 0) or 0), int(getattr(sample, "era_line_index", 0) or 0))


def build_line_diagnostics_ui(
    parent,
    *,
    source_snapshot: SourceSnapshot,
    draw_helper: Callable[..., np.ndarray],
    on_focus: Callable[..., object],
    live_getter: Callable[[], object] | None = None,
    scroll_factory: Callable[..., object] | None = None,
    on_review: Callable[..., object] | None = None,
) -> object:
    """Build and show the读-only「逐行定位诊断」窗口 as a ``Toplevel`` of ``parent``.

    ``draw_helper(image, global_row, bounds)`` 返回该 ROI 该行的真实灰度一维数组（供
    ``diag_chart_rgb`` 使用），须为纯函数。``on_focus(left, top, right, bottom)``
    为主程序提供的安全回调：仅平移主 SEM 视图到该目标区域（选中行的单行区域），
    不改任何状态。

    ``live_getter``（可选）返回主应用当前的 live 来源句柄；每次跳转前都用
    ``live_still_same`` 验证它仍与打开时刻捕获的同一对象相同——主图 A 切到 B 后，
    旧窗口只看快照、绝不平移 B，并给出一条简短提示。

    ``scroll_factory``（可选）返回一个带 ``.content`` 的可滚动容器（主程序传来的
    ``VerticalScrollFrame``；本模块不 import 应用，避免循环依赖）。当提供时，整窗
    内容放进该滚动容器，使树 + 统计 + 图表在最小高度下仍可两轴滚动。缺省为
    ``None``，退回到旧的直接 ``ttk.Frame`` 布局，保持既有调用/测试兼容。

    ``on_review``（可选）回调，选中行后提供一个「查看校正前后对照」按钮：主程序
    据此打开灰度/灰度差对照窗口并预设到该样本该侧该全局行。回调签名为
    ``on_review(sample_index, side, global_row)``，缺省时不显示该按钮（旧行为）。
    """
    image = source_snapshot.image
    samples = list(source_snapshot.samples)
    if image is None:
        from tkinter import messagebox
        messagebox.showinfo("暂无图像", "请先打开一张 SEM 图像。", parent=parent)
        return None

    dialog = tk.Toplevel(parent)
    dialog.title("逐行定位诊断")
    dialog.minsize(860, 560)
    try:
        screen_height = int(parent.winfo_screenheight())
    except Exception:  # 无 Tk 环境（测试替身）时用安全默认高度
        screen_height = 1000
    dialog.geometry(f"980x{min(760, max(560, screen_height - 100))}")
    # Only the CURRENT chart's PhotoImage is kept per window (no unbounded growth, no
    # shared global host list). Switching samples reuses this one slot.
    photos: dict[str, object] = {"current": None}
    focus = focus_capture(source_snapshot)
    live_getter = live_getter or (lambda: source_snapshot.live)

    # Whole-window two-axis scrolling when the host supplies a scroll_factory
    # (its .content is the real parent of every widget below); otherwise keep the
    # plain frame layout for backward compatibility with old callers/tests.
    if scroll_factory is not None:
        root = scroll_factory(dialog, padding=12)
        root_pack = root  # container frame (already packed inside the factory's scrollable)
        root.pack(fill=tk.BOTH, expand=True)
        content_root = root.content
    else:
        root = ttk.Frame(dialog, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        content_root = root
        root_pack = None

    ttk.Label(
        content_root,
        text="逐行定位诊断：列表显示每行候选/最终 global x 与首轮/恢复/缺失及原因；选行后显示该 ROI 该行真实灰度与 ΔI 曲线，并通过只读回调跳转主 SEM 对应行（不改 ROI/采样/统计）。",
        foreground="#666666",
        wraplength=940,
    ).pack(anchor="w")
    ttk.Label(
        content_root,
        text=f"来源（打开时冻结）：{source_snapshot.source_label()}"
             + (f"　|　处理：{source_snapshot.preprocessing}" if source_snapshot.preprocessing else "")
             + (f"　|　标定：{source_snapshot.calibration}" if source_snapshot.calibration else "")
             + (f"　|　旋转：{source_snapshot.rotation}" if source_snapshot.rotation else ""),
        foreground="#a05000",
        wraplength=940,
    ).pack(anchor="w")

    # ---- sample + side selector --------------------------------------
    control_row = ttk.Frame(content_root)
    control_row.pack(fill=tk.X, pady=(8, 0))
    ttk.Label(control_row, text="样本 / 侧别").pack(side=tk.LEFT)

    options: list[str] = []
    option_targets: list[tuple[int, str, EdgeLocalization]] = []
    option_origins: list[int] = []
    for index, sample in enumerate(samples, start=1):
        for side in ("left", "right"):
            diag = side_diagnostics(sample, side)
            bounds = getattr(sample, f"{side}_roi_bounds_px")
            if bounds is None or diag is None:
                continue
            label = f"样本 {index} {'左' if side == 'left' else '右'}边"
            options.append(label)
            option_targets.append((index - 1, side, diag))
            option_origins.append(_roi_origin_x(diag, source_snapshot.left_x))

    selector = ttk.Combobox(control_row, values=options, state="readonly", width=40)
    selector.pack(side=tk.LEFT, padx=(6, 0))

    content = ttk.Frame(content_root)
    content.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    scroll = ttk.Frame(content)
    scroll.pack(fill=tk.BOTH, expand=True)
    tree = ttk.Treeview(
        scroll, columns=("grow", "candx", "finx", "stage", "reason"), show="headings", height=12
    )
    for col, text, w in (
        ("grow", "global 行", 70),
        ("candx", "候选 global x", 110),
        ("finx", "最终 global x", 110),
        ("stage", "阶段", 70),
        ("reason", "失败原因", 130),
    ):
        tree.heading(col, text=text)
        tree.column(col, width=w, anchor="w")
    vsb = ttk.Scrollbar(scroll, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)

    stats_var = tk.StringVar(dialog, value="")
    ttk.Label(content_root, textvariable=stats_var, foreground="#0a7d3c", wraplength=940).pack(anchor="w", pady=(6, 0))
    notice_var = tk.StringVar(dialog, value="")
    notice_label = ttk.Label(content_root, textvariable=notice_var, foreground="#b00020", wraplength=940)
    notice_label.pack(anchor="w")

    # Diagnostic CSV export (static source snapshot), available even without any
    # successful metrology result.
    def _export_diagnostic_csv() -> None:
        import csv
        from tkinter import filedialog, messagebox

        path = filedialog.asksaveasfilename(
            parent=dialog,
            title="导出逐行定位诊断 CSV",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                for row in diagnostic_csv_rows(source_snapshot):
                    writer.writerow(row)
            messagebox.showinfo("导出完成", f"已导出诊断 CSV：\n{path}", parent=dialog)
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc), parent=dialog)

    export_button = ttk.Button(content_root, text="导出诊断 CSV", command=_export_diagnostic_csv)
    export_button.pack(anchor="e", pady=(6, 0))

    # Optional review callback: jump to the host's profile review window preset to
    # this sample/side/global row (correct前 vs 后 gray/ΔI comparison). Only shown
    # when the host passes on_review; old callers/tests keep the original layout.
    if on_review is not None:
        review_button = ttk.Button(
            content_root,
            text="查看校正前后对照",
            command=lambda: on_review(state["sample_idx"], state["side"], _last_global_row()),
        )
        review_button.pack(anchor="w", pady=(4, 0))

    chart_label = ttk.Label(content_root)
    chart_label.pack(anchor="w", pady=(6, 0))

    state: dict[str, object] = {"records": [], "sample_idx": 0, "side": "left", "bounds": None, "left": 0, "right": 0, "_global_row": None}

    def _last_global_row() -> int | None:
        return state.get("_global_row")

    def fmt_x(value) -> str:
        return "" if not math.isfinite(value) else f"{value:.3f}"

    def _record_to_values(record: dict) -> tuple:
        return (
            str(record["global_row"]),
            fmt_x(record["candidate_global_x"]),
            fmt_x(record["final_global_x"]),
            record["stage_label"],
            record["reason_label"],
        )

    def _focus(left: int, top: int, right: int, bottom: int) -> bool:
        """Validated jump: same live source only, else show the snapshot and say so."""
        live = live_getter()
        if not live_still_same(focus, live):
            notice_var.set(STALE_SOURCE_NOTICE)
            return False
        notice_var.set("")
        on_focus(left, top, right, bottom)
        return True

    def on_select(_event=None) -> None:
        selection = tree.selection()
        if not selection:
            return
        local_row = int(selection[0])
        records = state["records"]
        record = next((r for r in records if r["local_row"] == local_row), None)
        if record is None:
            return
        global_row = record["global_row"]
        state["_global_row"] = global_row
        bounds = state["bounds"]
        intensity = draw_helper(image, global_row, bounds)
        candidate_x = record["candidate_global_x"] if math.isfinite(record["candidate_global_x"]) else None
        final_x = record["final_global_x"] if math.isfinite(record["final_global_x"]) else None
        failure_text = None
        if record["stage"] == STAGE_MISSING:
            failure_text = f"缺失：{record['reason_label']}"
            if candidate_x is not None:
                failure_text += "（候选未接受）"
        option_index = next((i for i, t in enumerate(option_targets) if t[0] == state["sample_idx"] and t[1] == state["side"]), 0)
        origin_x = option_origins[option_index]
        chart = diag_chart_rgb(
            intensity, rowwidth=image.shape[1], origin_x=origin_x,
            title=f"全局行 {global_row}（ROI x {state['left']}..{state['right'] - 1}）",
            candidate_x=candidate_x, final_x=final_x, failure_text=failure_text,
        )
        photo = ImageTk.PhotoImage(Image.fromarray(chart), master=dialog)
        photos["current"] = photo  # replace: the old chart's PhotoImage is released here
        chart_label.configure(image=photo)
        _focus(state["left"], global_row, state["right"], global_row + 1)

    def apply(option_index: int | None = None) -> None:
        if option_index is None:
            option_index = selector.current()
        if option_index < 0 or option_index >= len(option_targets):
            return
        sample_idx, side, diag = option_targets[option_index]
        sample = samples[sample_idx]
        left_diag = side_diagnostics(sample, "left")
        right_diag = side_diagnostics(sample, "right")
        left_no, right_no = _sample_side_index(sample)
        stats_var.set(diag_stats_text(left_diag, right_diag, side, left_no or sample_idx + 1, right_no or sample_idx + 1))

        origin_x = option_origins[option_index]
        records = diag_row_records(diag, left_x=source_snapshot.left_x)
        # Column extent from the ROI rectangle itself (never a row count or image width).
        roi_right = int(diag.roi_bounds_px[2]) + int(source_snapshot.left_x)
        if roi_right <= origin_x:
            roi_right = origin_x + len(diag.rows)
        for record in records:
            record["roi_left"] = origin_x
            record["roi_right"] = roi_right
        state.update(
            records=records, sample_idx=sample_idx, side=side,
            bounds=getattr(sample, f"{side}_roi_bounds_px"),
            left=origin_x, right=roi_right,
        )
        tree.delete(*tree.get_children())
        for record in records:
            tree.insert("", tk.END, values=_record_to_values(record), iid=str(record["local_row"]))
        # Default select the first row so the chart/focus populate immediately.
        if records:
            tree.selection_set(str(records[0]["local_row"]))

    tree.bind("<<TreeviewSelect>>", on_select)
    if options:
        selector.set(options[0])
        selector.bind("<<ComboboxSelected>>", lambda _e: apply())
        apply(0)
    else:
        ttk.Label(content_root, text="暂无诊断记录：请先完成一次测量（失败侧诊断也会在此列出）。").pack(anchor="w", pady=10)

    # Expose the window state for callbacks/tests (view-only, not a Tk dependency).
    dialog.line_diagnostics_photos = photos
    dialog.line_diagnostics_state = state
    dialog.line_diagnostics_stats = stats_var
    dialog.line_diagnostics_notice = notice_var
    return dialog


DIAGNOSTIC_CSV_COLUMNS = [
    "era_line_index", "side", "global_row", "local_row",
    "candidate_global_x", "final_global_x", "valid", "stage", "reason",
    "recovered_from_global_x",
    "fit_degree", "position_definition", "max_jump_px", "recovery_policy",
    "source_path", "source_name", "source_id", "preprocessing", "calibration", "rotation",
    "image_size", "processing", "line_localization_method", "noise_context", "captured_at",
]

RECOVERY_POLICY = (
    f"neighbor_guided_window_{RECOVERY_WINDOW_ROWS}_rows_min_{RECOVERY_MIN_NEIGHBORS}"
)


def frozen_source_block(snapshot: SourceSnapshot) -> list[list[str]]:
    """Key/value header lines describing the FROZEN source (never live app state)."""
    return [
        ["# 逐行定位诊断导出（静态来源快照；来源/参数在窗口打开时刻冻结）"],
        ["source", snapshot.source_label()],
        ["source_name", snapshot.source_name],
        ["source_id", snapshot.source_id],
        ["preprocessing", snapshot.preprocessing],
        ["calibration", snapshot.calibration],
        ["rotation", snapshot.rotation],
        ["image_size", snapshot.image_size],
        ["processing", snapshot.processing],
        ["line_localization_method", snapshot.method],
        ["recovery_policy", RECOVERY_POLICY],
        ["noise_context", snapshot.noise_context],
        ["captured_at", snapshot.captured_at],
        *[[key, str(value)] for key, value in snapshot.extra],
    ]


def diagnostic_csv_rows(
    source_snapshot: SourceSnapshot,
) -> list[list[str]]:
    """Static-source diagnostic CSV rows (pure, testable, no Tk).

    Coordinates are written at full float repr (no fixed-point truncation); missing or
    non-finite values stay blank (never 0).  ``valid`` is 1/0 and always agrees with the
    per-row stage: ``valid == 0`` => blank final x, ``valid == 1`` => finite final x.
    Every parameter comes from the stored ``EdgeLocalization`` of the frozen snapshot.
    """
    rows: list[list[str]] = []
    rows.extend(frozen_source_block(source_snapshot))
    rows.append([])
    rows.append(list(DIAGNOSTIC_CSV_COLUMNS))
    for index, sample in enumerate(source_snapshot.samples, start=1):
        line_index = int(getattr(sample, "era_line_index", 0) or 0) or index
        for side in ("left", "right"):
            diag = side_diagnostics(sample, side)
            if diag is None:
                continue
            for record in diag_row_records(diag, left_x=source_snapshot.left_x):
                final_gx = record["final_global_x"]
                valid = record["stage"] != STAGE_MISSING and bool(math.isfinite(final_gx))
                rows.append([
                    str(line_index),
                    side,
                    str(record["global_row"]),
                    str(record["local_row"]),
                    _csv_value(record["candidate_global_x"]),
                    _csv_value(final_gx) if valid else "",
                    "1" if valid else "0",
                    record["stage"],
                    record["reason"],
                    _csv_value(record["recovered_from_global_x"]),
                    str(getattr(diag, "fit_degree", "")),
                    str(getattr(diag, "position_definition", "")),
                    _csv_value(getattr(diag, "max_jump_px", float("nan"))),
                    RECOVERY_POLICY,
                    source_snapshot.source_path,
                    source_snapshot.source_name,
                    source_snapshot.source_id,
                    source_snapshot.preprocessing,
                    source_snapshot.calibration,
                    source_snapshot.rotation,
                    source_snapshot.image_size,
                    source_snapshot.processing,
                    source_snapshot.method,
                    source_snapshot.noise_context,
                    source_snapshot.captured_at,
                ])
    return rows
