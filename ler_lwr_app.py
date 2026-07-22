"""Interactive SEM line-edge roughness (LER) and line-width roughness (LWR) tool.

The program intentionally reports standard deviation by default.  The user-selected
sigma multiplier only changes the displayed and exported multiplied metrics.
"""

from __future__ import annotations

import csv
import fcntl
import math
import re
import sys
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, scrolledtext, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

VENDOR_DIRECTORY = Path(__file__).with_name("vendor")
if VENDOR_DIRECTORY.exists():
    sys.path.insert(0, str(VENDOR_DIRECTORY))

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    AppBase = TkinterDnD.Tk
    DND_AVAILABLE = True
except ImportError:
    AppBase = tk.Tk
    DND_FILES = "DND_Files"
    DND_AVAILABLE = False


MAX_VIEW_WIDTH = 980
MAX_VIEW_HEIGHT = 620
ZOOM_MIN = 0.25
ZOOM_MAX = 3.0
ZOOM_STEP = 0.10
MIN_ROI_WIDTH = 24
MIN_ROI_HEIGHT = 80
ROI_HANDLE_SIZE = 5
ROI_HANDLE_HIT = 10
ANNOTATION_HANDLE_HIT = 10
ANNOTATION_COLORS = {
    "length": "#ffde59",
    "rectangle": "#ff6b6b",
    "circle": "#a78bfa",
    "hexagon": "#38bdf8",
}
MAX_UNDO_STEPS = 30
OUTLIER_LEVELS = {
    "严格（8 px）": 8.0,
    "标准（12 px，默认）": 12.0,
    "宽松（16 px）": 16.0,
}
DEFAULT_OUTLIER_LEVEL = "标准（12 px，默认）"
LOCK_PATH = Path(tempfile.gettempdir()) / "sem_ler_lwr_app.lock"


@dataclass
class AnalysisResult:
    rows_px: np.ndarray
    left_px: np.ndarray
    right_px: np.ndarray
    left_residual_nm: np.ndarray
    right_residual_nm: np.ndarray
    width_nm: np.ndarray
    width_residual_nm: np.ndarray
    mean_width_nm: float
    ler_left_sigma_nm: float
    ler_right_sigma_nm: float
    lwr_sigma_nm: float
    total_ler_sigma_nm: float
    edge_correlation: float
    pixel_size_nm: float
    rejected_rows: int


@dataclass
class MeasurementAnnotation:
    """One editable general-purpose measurement drawn in working-image pixels."""

    kind: str
    bounds_px: tuple[float, float, float, float]
    color: str | None = None


@dataclass
class EditorState:
    roi_canvas: tuple[float, float, float, float] | None
    annotations: list[MeasurementAnnotation]
    active_annotation_index: int | None
    rotation_degrees: float
    preprocess_enabled: bool
    result: AnalysisResult | None
    analysis_origin: tuple[int, int] | None


def read_sem_pixel_size_nm(image: Image.Image) -> float | None:
    """Read FEI/Thermo Fisher SEM pixel width from TIFF private metadata.

    Verios TIFF files store their acquisition metadata in TIFF tag 34682.  The
    `PixelWidth` field is expressed in metres, unlike the generic TIFF DPI tags.
    """
    metadata = str(getattr(image, "tag_v2", {}).get(34682, ""))
    matched = re.search(r"(?m)^PixelWidth=([0-9.eE+-]+)\s*$", metadata)
    if not matched:
        return None
    pixel_size_nm = float(matched.group(1)) * 1e9
    return pixel_size_nm if 0 < pixel_size_nm < 1e5 else None


def format_tiff_metadata(image: Image.Image, source_path: Path) -> str:
    """Build a metadata view for TIFF and ordinary image formats."""
    tags = getattr(image, "tag_v2", {})
    lines = [
        f"文件：{source_path}",
        f"格式：{image.format or '未知'}",
        f"像素：{image.width} × {image.height}",
        f"颜色模式：{image.mode}",
        f"压缩：{image.info.get('compression', '未知')}",
        "",
        "[通用 TIFF 信息]",
    ]
    tag_names = {
        256: "ImageWidth",
        257: "ImageLength",
        258: "BitsPerSample",
        259: "Compression",
        262: "PhotometricInterpretation",
        282: "XResolution",
        283: "YResolution",
        296: "ResolutionUnit",
        305: "Software",
        306: "DateTime",
    }
    for tag_id, name in tag_names.items():
        if tag_id in tags:
            lines.append(f"{name} = {tags[tag_id]}")

    sem_metadata = str(tags.get(34682, "")).strip()
    if sem_metadata:
        lines.extend(["", "[SEM 采集元数据（原始）]", sem_metadata])
    else:
        lines.extend(["", "[SEM 采集元数据]", "该图像不含 FEI/Thermo Fisher 的 SEM 像素尺寸元数据。"])
        if image.info:
            lines.append("")
            lines.append("[图像文件元数据]")
            for key, value in image.info.items():
                summary = f"{len(value)} bytes" if isinstance(value, bytes) else str(value)
                lines.append(f"{key} = {summary}")
    return "\n".join(lines)


def normalize_and_denoise(image: np.ndarray) -> np.ndarray:
    """Apply robust contrast normalization and a light 3×3 median denoise.

    The original image is never modified.  Percentile normalization makes the
    preview legible while avoiding the influence of isolated bright SEM pixels.
    """
    low, high = np.percentile(image.astype(float), (1.0, 99.0))
    if high <= low:
        normalized = np.zeros(image.shape, dtype=np.uint8)
    else:
        normalized = np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(normalized).filter(ImageFilter.MedianFilter(size=3)))


def rotate_grayscale_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a grayscale image without changing its pixel dimensions."""
    fill = int(np.median(image))
    rotated = Image.fromarray(image).rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )
    return np.asarray(rotated)


def rotate_points_about_center(
    points: list[tuple[float, float]],
    angle_degrees: float,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Rotate image-coordinate points the same way PIL rotates same-size images."""
    if not points or abs(angle_degrees) < 1e-12:
        return points
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    rotated_points = []
    for x, y in points:
        dx = x - center_x
        dy = y - center_y
        rotated_points.append((center_x + dx * cos_a + dy * sin_a, center_y - dx * sin_a + dy * cos_a))
    return rotated_points


def estimate_line_tilt_degrees(image: np.ndarray) -> float:
    """Estimate the signed deviation of the dominant line direction from vertical.

    The local image gradient is normal to a line edge.  The structure tensor
    aggregates all strong gradients in an ROI, producing a robust orientation
    modulo 180° even though the two line edges have opposite contrast signs.
    """
    image_float = image.astype(float)
    gradient_y, gradient_x = np.gradient(image_float)
    tensor_xx = float(np.sum(gradient_x * gradient_x))
    tensor_yy = float(np.sum(gradient_y * gradient_y))
    tensor_xy = float(np.sum(gradient_x * gradient_y))
    if tensor_xx + tensor_yy < 1e-9:
        raise ValueError("选区对比度不足，无法识别线条倾角。")
    normal_angle = 0.5 * math.atan2(2 * tensor_xy, tensor_xx - tensor_yy)
    return math.degrees(normal_angle)


def find_rotation_to_vertical(image: np.ndarray) -> float:
    """Find the image rotation that makes the dominant ROI direction vertical."""
    height, width = image.shape
    scale = min(1.0, 360.0 / max(height, width))
    if scale < 1.0:
        preview = np.asarray(
            Image.fromarray(image).resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
        )
    else:
        preview = image

    def score(candidate_image: np.ndarray, angle: float) -> float:
        rotated = rotate_grayscale_image(candidate_image, angle)
        border = max(4, int(min(rotated.shape) * 0.08))
        interior = rotated[border:-border, border:-border]
        return abs(estimate_line_tilt_degrees(interior))

    coarse_angles = np.arange(-45.0, 46.0, 1.0)
    coarse_best = min(coarse_angles, key=lambda angle: score(preview, angle))
    fine_angles = np.arange(coarse_best - 3.0, coarse_best + 3.01, 0.1)
    return float(min(fine_angles, key=lambda angle: score(image, angle)))


def acquire_single_instance_lock():
    """Return an exclusive lock handle, or None when another app is active."""
    handle = open(LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def show_already_running_message() -> None:
    """Show a concise macOS dialog without creating a persistent second window."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("SEM LER / LWR", "测量窗口已打开；请使用现有窗口。")
    root.destroy()


def parabola_peak(values: np.ndarray, index: int) -> float:
    """Return a subpixel peak location for a local maximum."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, center, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float(index) + float(np.clip(offset, -0.5, 0.5))


def smooth_profiles(image: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """Apply a small Gaussian filter along each horizontal profile."""
    radius = max(1, math.ceil(3 * sigma_px))
    positions = np.arange(-radius, radius + 1)
    kernel = np.exp(-(positions**2) / (2 * sigma_px**2))
    kernel /= kernel.sum()
    padded = np.pad(image.astype(float), ((0, 0), (radius, radius)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)


def rolling_median(trace: np.ndarray, window: int = 9) -> np.ndarray:
    """Return a short local-median baseline without an extra SciPy dependency."""
    radius = window // 2
    padded = np.pad(trace, (radius, radius), mode="edge")
    return np.array([np.median(padded[index : index + window]) for index in range(len(trace))])


def replace_outliers(trace: np.ndarray, max_jump_px: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated trace failures without smoothing genuine roughness."""
    baseline = rolling_median(trace, window=9)
    valid = np.abs(trace - baseline) <= max_jump_px
    repaired = trace.copy()
    repaired[~valid] = baseline[~valid]
    return repaired, valid


def locate_edges(image: np.ndarray, max_jump_px: float = 12.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate the two edges of one vertical line in a grayscale ROI.

    The input ROI must contain one line and some background on both sides.
    For every horizontal scanline, the largest gradient magnitude in its left and
    right halves is refined by a three-point parabolic fit.
    """
    height, width = image.shape
    margin = max(3, int(width * 0.05))
    midpoint = width // 2
    if midpoint - margin < 3 or width - midpoint - margin < 3:
        raise ValueError("分析区域太窄；请框选一条线及其两侧背景。")

    # SEM noise can make a one-pixel derivative stronger than a physical edge.
    # A 3-pixel Gaussian is only used for locating the edge; output coordinates
    # remain subpixel positions rather than a smoothed edge trace.
    smoothed = smooth_profiles(image, sigma_px=3.0)
    gradient = np.abs(np.gradient(smoothed, axis=1))
    reference_profile = np.median(gradient, axis=0)
    left_reference = margin + int(np.argmax(reference_profile[margin:midpoint]))
    right_reference = midpoint + int(np.argmax(reference_profile[midpoint : width - margin]))
    search_radius = max(12, min(30, int(width * 0.05)))
    left = np.empty(height, dtype=float)
    right = np.empty(height, dtype=float)

    for row in range(height):
        left_start = max(1, left_reference - search_radius)
        left_end = min(width - 1, left_reference + search_radius + 1)
        right_start = max(1, right_reference - search_radius)
        right_end = min(width - 1, right_reference + search_radius + 1)
        left_index = left_start + int(np.argmax(gradient[row, left_start:left_end]))
        right_index = right_start + int(np.argmax(gradient[row, right_start:right_end]))
        left[row] = parabola_peak(gradient[row], left_index)
        right[row] = parabola_peak(gradient[row], right_index)

    left, left_valid = replace_outliers(left, max_jump_px)
    right, right_valid = replace_outliers(right, max_jump_px)
    valid = left_valid & right_valid & (right > left)
    return left, right, valid


def analyze_roi(image: np.ndarray, pixel_size_nm: float, max_jump_px: float = 12.0) -> AnalysisResult:
    """Calculate standard RMS LER and LWR for one vertical SEM line ROI.

    A linear baseline is removed from each edge (LER), while only the average
    width is removed from the width trace (LWR).  This follows the NIST/CD-SEM
    convention: LER uses N-2 degrees of freedom and LWR uses N-1.
    """
    if pixel_size_nm <= 0:
        raise ValueError("像素尺寸必须大于 0 nm/pixel。")
    height, width = image.shape
    if width < MIN_ROI_WIDTH or height < MIN_ROI_HEIGHT:
        raise ValueError("分析区域过小；请至少包含 80 个沿线方向的像素。")

    left, right, valid = locate_edges(image, max_jump_px)
    rows = np.arange(height, dtype=float)
    if valid.sum() < max(30, height * 0.6):
        raise ValueError("有效边缘点太少；请检查 ROI 是否只包含一条线。")

    valid_rows = rows[valid]
    valid_left = left[valid]
    valid_right = right[valid]
    left_fit = np.polyval(np.polyfit(valid_rows, valid_left, 1), valid_rows)
    right_fit = np.polyval(np.polyfit(valid_rows, valid_right, 1), valid_rows)
    left_residual_nm = (valid_left - left_fit) * pixel_size_nm
    right_residual_nm = (valid_right - right_fit) * pixel_size_nm
    width_nm = (valid_right - valid_left) * pixel_size_nm
    width_residual_nm = width_nm - np.mean(width_nm)
    sample_count = len(valid_rows)
    ler_left_sigma_nm = math.sqrt(float(np.dot(left_residual_nm, left_residual_nm)) / (sample_count - 2))
    ler_right_sigma_nm = math.sqrt(float(np.dot(right_residual_nm, right_residual_nm)) / (sample_count - 2))
    lwr_sigma_nm = math.sqrt(float(np.dot(width_residual_nm, width_residual_nm)) / (sample_count - 1))
    total_ler_sigma_nm = math.hypot(ler_left_sigma_nm, ler_right_sigma_nm)
    if ler_left_sigma_nm > 0 and ler_right_sigma_nm > 0:
        edge_correlation = float(np.corrcoef(left_residual_nm, right_residual_nm)[0, 1])
    else:
        edge_correlation = float("nan")

    return AnalysisResult(
        rows_px=valid_rows,
        left_px=valid_left,
        right_px=valid_right,
        left_residual_nm=left_residual_nm,
        right_residual_nm=right_residual_nm,
        width_nm=width_nm,
        width_residual_nm=width_residual_nm,
        mean_width_nm=float(np.mean(width_nm)),
        ler_left_sigma_nm=ler_left_sigma_nm,
        ler_right_sigma_nm=ler_right_sigma_nm,
        lwr_sigma_nm=lwr_sigma_nm,
        total_ler_sigma_nm=total_ler_sigma_nm,
        edge_correlation=edge_correlation,
        pixel_size_nm=pixel_size_nm,
        rejected_rows=int((~valid).sum()),
    )


class LERLWRApp(AppBase):
    def __init__(self) -> None:
        super().__init__()
        self.title("SEM measure（LER/LWR + 尺寸标注）")
        self.minsize(1100, 780)
        self.image_path: Path | None = None
        self.original_image: np.ndarray | None = None
        self.raw_image: np.ndarray | None = None
        self.processed_image: np.ndarray | None = None
        self.display_image: ImageTk.PhotoImage | None = None
        self.base_scale = 1.0
        self.display_scale = 1.0
        self.zoom_factor = 1.0
        self.roi_canvas: tuple[float, float, float, float] | None = None
        self.roi_rectangle: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.roi_drag_mode: str | None = None
        self.roi_drag_anchor: tuple[float, float] | None = None
        self.roi_start_bounds: tuple[float, float, float, float] | None = None
        self.roi_was_changed = False
        self.annotations: list[MeasurementAnnotation] = []
        self.active_annotation_index: int | None = None
        self.annotation_drag_mode: str | None = None
        self.annotation_drag_anchor: tuple[float, float] | None = None
        self.annotation_start_bounds: tuple[float, float, float, float] | None = None
        self.annotation_was_changed = False
        self.undo_history: list[EditorState] = []
        self.pending_undo_state: EditorState | None = None
        self.space_held = False
        self.panning = False
        self.result: AnalysisResult | None = None
        self.analysis_origin: tuple[int, int] | None = None
        self.lcdu_cd_samples_nm: list[float] = []
        self.lcdu_sample_listbox: tk.Listbox | None = None
        self.metadata_text = "尚未导入 TIFF 文件。"
        self.rotation_degrees = 0.0

        self.pixel_size_var = tk.StringVar(value="")
        self.metadata_var = tk.StringVar(value="尚未读取 TIFF 元数据")
        self.preprocess_var = tk.BooleanVar(value=True)
        self.sigma_multiplier_var = tk.DoubleVar(value=1.0)
        self.outlier_level_var = tk.StringVar(value=DEFAULT_OUTLIER_LEVEL)
        self.lcdu_summary_var = tk.StringVar(value="LCDU 样本：0 条线")
        self.rotation_var = tk.StringVar(value="0.00")
        self.measurement_tool_var = tk.StringVar(value="ROI（LER/LWR）")
        self.measurement_color = ANNOTATION_COLORS["length"]
        self.zoom_slider_var = tk.DoubleVar(value=1.0)
        self.zoom_percent_var = tk.StringVar(value="100%")
        self.status_var = tk.StringVar(value="打开一张俯视 SEM 图，然后框选一条线及两侧背景。")
        self.result_var = tk.StringVar(value="尚未分析")

        self._build_ui()
        self.pixel_size_var.trace_add("write", self.refresh_annotation_labels)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)
        file_menu = tk.Menu(controls, tearoff=False)
        file_menu.add_command(label="打开 SEM 图像", command=self.open_image)
        file_menu.add_command(label="查看 TIFF 元数据", command=self.show_metadata)
        file_menu.add_separator()
        outlier_menu = tk.Menu(file_menu, tearoff=False)
        for label in OUTLIER_LEVELS:
            outlier_menu.add_radiobutton(
                label=label,
                variable=self.outlier_level_var,
                value=label,
                command=self.change_outlier_level,
            )
        file_menu.add_cascade(label="异常点剔除", menu=outlier_menu)
        file_menu.add_separator()
        file_menu.add_command(label="导出 CSV", command=self.export_csv)
        file_menu.add_command(label="导出标注/拟合图片", command=self.export_annotated_image)
        ttk.Menubutton(controls, text="菜单 ▾", menu=file_menu).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(controls, text="像素尺寸 (nm/pixel):").grid(row=0, column=1, sticky="e")
        ttk.Entry(controls, textvariable=self.pixel_size_var, width=10).grid(row=0, column=2, padx=(4, 12))
        ttk.Label(controls, textvariable=self.metadata_var, foreground="#426b2d").grid(row=0, column=3, padx=(0, 12), sticky="w")
        ttk.Checkbutton(
            controls,
            text="归一化 + 3×3 去噪",
            variable=self.preprocess_var,
            command=self.refresh_preprocessing,
        ).grid(row=0, column=4, padx=(0, 12))
        ttk.Button(controls, text="自动校正 ROI 倾角", command=self.auto_align_roi).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(controls, text="重置角度", command=self.reset_rotation).grid(row=0, column=6, padx=(0, 12))
        ttk.Button(controls, text="分析选区", command=self.run_analysis).grid(row=0, column=7, padx=(0, 12))
        ttk.Label(controls, text="测量工具").grid(row=0, column=8, sticky="e")
        measurement_tool = ttk.Combobox(
            controls,
            textvariable=self.measurement_tool_var,
            state="readonly",
            values=("ROI（LER/LWR）", "长度", "矩形", "圆形", "正六边形"),
            width=13,
        )
        measurement_tool.grid(row=0, column=9, padx=(4, 8))
        measurement_tool.bind("<<ComboboxSelected>>", self.change_measurement_tool)
        self.color_button = tk.Button(controls, text="线条颜色", command=self.choose_measurement_color, relief=tk.GROOVE)
        self.color_button.grid(row=0, column=10, padx=(0, 8))
        self.update_measurement_color_button()

        multiplier = ttk.LabelFrame(controls, text="显示/导出倍数", padding=(8, 2))
        multiplier.grid(row=0, column=11, padx=(8, 0), sticky="ew")
        ttk.Scale(
            multiplier,
            from_=0.5,
            to=6.0,
            orient=tk.HORIZONTAL,
            variable=self.sigma_multiplier_var,
            command=lambda _value: self.update_displayed_results(),
            length=180,
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(multiplier, textvariable=self.sigma_multiplier_var, width=4).grid(row=0, column=1)

        content = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        image_frame = ttk.LabelFrame(content, text="图像：倾斜线先框选 ROI，再点击“自动校正 ROI 倾角”", padding=5)
        result_frame = ttk.LabelFrame(content, text="测量结果", padding=12)
        content.add(image_frame, weight=4)
        content.add(result_frame, weight=2)

        zoom_controls = ttk.Frame(image_frame, padding=(4, 6))
        zoom_controls.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(zoom_controls, text="−", width=3, command=lambda: self.step_zoom(-ZOOM_STEP)).pack(side=tk.LEFT)
        ttk.Scale(
            zoom_controls,
            from_=ZOOM_MIN,
            to=ZOOM_MAX,
            orient=tk.HORIZONTAL,
            variable=self.zoom_slider_var,
            command=self.zoom_from_slider,
            length=150,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(zoom_controls, text="+", width=3, command=lambda: self.step_zoom(ZOOM_STEP)).pack(side=tk.LEFT)
        ttk.Label(zoom_controls, textvariable=self.zoom_percent_var, width=6).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(zoom_controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Label(zoom_controls, text="旋转").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(zoom_controls, text="−1°", width=5, command=lambda: self.rotate_manually(-1.0)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(zoom_controls, text="−0.1°", width=6, command=lambda: self.rotate_manually(-0.1)).pack(side=tk.LEFT, padx=(0, 6))
        rotation_entry = ttk.Entry(zoom_controls, textvariable=self.rotation_var, width=8)
        rotation_entry.pack(side=tk.LEFT, padx=(0, 3))
        rotation_entry.bind("<Return>", lambda _event: self.apply_manual_rotation_value())
        ttk.Label(zoom_controls, text="°").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(zoom_controls, text="应用", command=self.apply_manual_rotation_value).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(zoom_controls, text="+0.1°", width=6, command=lambda: self.rotate_manually(0.1)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(zoom_controls, text="+1°", width=5, command=lambda: self.rotate_manually(1.0)).pack(side=tk.LEFT)

        ttk.Separator(zoom_controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Label(zoom_controls, text="标注管理").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_controls, text="删除标注", command=self.delete_active_annotation).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(zoom_controls, text="清空标注", command=self.clear_annotations).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(image_frame, background="#202020", highlightthickness=0, width=780, height=620, takefocus=True)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.start_canvas_action)
        self.canvas.bind("<B1-Motion>", self.move_canvas_action)
        self.canvas.bind("<ButtonRelease-1>", self.finish_canvas_action)
        self.canvas.bind("<MouseWheel>", self.zoom_with_space_wheel)
        self.canvas.bind("<Button-4>", self.zoom_with_space_wheel)
        self.canvas.bind("<Button-5>", self.zoom_with_space_wheel)
        self.canvas.bind("<Enter>", lambda _event: self.canvas.focus_set())
        self.canvas.bind("<Motion>", self.update_canvas_cursor)
        self.canvas.bind("<BackSpace>", self.clear_selection)
        self.canvas.bind("<Delete>", self.clear_selection)
        self.bind_all("<KeyPress-space>", self.start_pan_mode)
        self.bind_all("<KeyRelease-space>", self.end_pan_mode)
        self.bind_all("<Command-z>", self.undo_last_action)
        self.bind_all("<Control-z>", self.undo_last_action)
        if DND_AVAILABLE:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self.drop_file)
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self.drop_file)

        ttk.Label(result_frame, textvariable=self.result_var, justify=tk.LEFT, font=("Menlo", 13)).pack(anchor="nw", fill=tk.X)
        ttk.Separator(result_frame).pack(fill=tk.X, pady=14)
        lcdu_controls = ttk.LabelFrame(result_frame, text="多线 LCDU", padding=(8, 6))
        lcdu_controls.pack(anchor="nw", fill=tk.X, pady=(0, 12))
        ttk.Label(lcdu_controls, textvariable=self.lcdu_summary_var, justify=tk.LEFT).pack(anchor="w", fill=tk.X)
        lcdu_buttons = ttk.Frame(lcdu_controls)
        lcdu_buttons.pack(anchor="w", pady=(6, 0))
        ttk.Button(lcdu_buttons, text="加入当前线 CD", command=self.add_lcdu_sample).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(lcdu_buttons, text="删除选中样本", command=self.delete_selected_lcdu_sample).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(lcdu_buttons, text="清空 LCDU 样本", command=self.clear_lcdu_samples).pack(side=tk.LEFT)
        self.lcdu_sample_listbox = tk.Listbox(lcdu_controls, height=5, exportselection=False)
        self.lcdu_sample_listbox.pack(fill=tk.X, pady=(8, 0))
        self.lcdu_sample_listbox.bind("<Delete>", self.delete_selected_lcdu_sample)
        self.lcdu_sample_listbox.bind("<BackSpace>", self.delete_selected_lcdu_sample)
        hint = (
            "使用说明\n"
            "1. 点击导入，或从 Finder 直接拖入 TIFF。\n"
            "2. TIFF 会自动读取 SEM 像素尺寸。\n"
            "3. 倾斜图案：先框选，再点“自动校正 ROI 倾角”。\n"
            "4. 可用底部手动旋转微调角度，再重新框选分析。\n"
            "5. 默认输出 1σ；拖动滑块可改为 kσ。\n\n"
            "通用测量：在底部工具中选择长度、矩形、圆形或正六边形后拖动创建。\n"
            "点击已有图形可移动或调整；尺寸会按 nm/pixel 自动标注。\n"
            "未填写像素尺寸时，通用测量暂以 px 显示。\n\n"
            "底部缩放条：拖动滑块或点 − / + 调整大小。\n"
            "按住空格 + 鼠标滚轮：以鼠标位置缩放。\n"
            "按住空格 + 左键拖动：平移图像。\n\n"
            "框选后可拖动绿色点调整范围；拖框内或中心点可移动选区。\n"
            "选中通用标注时 Backspace 删除标注，否则删除 ROI。\n\n"
            "默认使用 1%–99% 归一化和 3×3 中值去噪。\n"
            "如需比较原始图，可取消勾选预处理。\n\n"
            "本版假设线条沿竖直方向。\n"
            "LER 对左右边分别线性去趋势；LWR 只减去平均线宽。\n"
            "LCDU 由多条线的平均 CD 样本计算。\n"
            "LCDU 样本列表可选中单条删除。\n"
            "ER 的行业标准记号是 LER；总边缘 RMS 不等于 LWR。"
        )
        ttk.Label(result_frame, text=hint, justify=tk.LEFT, wraplength=290).pack(anchor="nw")
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(12, 6)).pack(side=tk.BOTTOM, fill=tk.X)

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择 SEM 图像",
            filetypes=[("常用图像", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.gif"), ("所有文件", "*.*")],
        )
        if not filename:
            return
        self.load_image(Path(filename))

    def drop_file(self, event: tk.Event) -> str:
        paths = self.tk.splitlist(event.data)
        if not paths:
            return "break"
        if len(paths) > 1:
            self.status_var.set("一次拖入多个文件；已加载第一个文件。")
        self.load_image(Path(paths[0]))
        return "break"

    def load_image(self, path: Path) -> None:
        if self.image_path is not None and path != self.image_path:
            replace = messagebox.askyesno(
                "替换当前图像？",
                "当前窗口已加载一张 SEM 图。导入新图会替换当前结果，不会新开窗口。\n\n是否继续？",
            )
            if not replace:
                return
        try:
            with Image.open(path) as source:
                metadata_pixel_size = read_sem_pixel_size_nm(source)
                self.metadata_text = format_tiff_metadata(source, path)
                image = source.convert("L")
                self.original_image = np.asarray(image)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法读取", f"无法读取图像：\n{exc}")
            return
        self.image_path = path
        self.undo_history.clear()
        self.pending_undo_state = None
        self.rotation_degrees = 0.0
        self.rotation_var.set(f"{self.rotation_degrees:.2f}")
        self.reset_zoom_control()
        self.rebuild_working_images()
        if metadata_pixel_size is None:
            self.pixel_size_var.set("")
            self.metadata_var.set("无 SEM 像素尺寸；请手动输入")
        else:
            self.pixel_size_var.set(f"{metadata_pixel_size:.8g}")
            self.metadata_var.set(f"TIFF 元数据：{metadata_pixel_size:.6g} nm/pixel")
        self.result = None
        self.analysis_origin = None
        self.lcdu_cd_samples_nm.clear()
        self.update_lcdu_summary_text()
        self.roi_canvas = None
        self.annotations.clear()
        self.active_annotation_index = None
        self.draw_image(reset_view=True)
        self.result_var.set("请框选一条线及两侧背景，然后点击“分析选区”。")
        self.status_var.set(f"已打开并完成预处理：{self.image_path.name}  ({self.raw_image.shape[1]} × {self.raw_image.shape[0]} px)")

    def snapshot_editor_state(self) -> EditorState:
        annotations = [MeasurementAnnotation(item.kind, item.bounds_px, item.color) for item in self.annotations]
        return EditorState(
            self.roi_canvas,
            annotations,
            self.active_annotation_index,
            self.rotation_degrees,
            self.preprocess_var.get(),
            self.result,
            self.analysis_origin,
        )

    def record_undo_state(self) -> None:
        if self.raw_image is None:
            return
        self.undo_history.append(self.snapshot_editor_state())
        del self.undo_history[:-MAX_UNDO_STEPS]

    def begin_undoable_edit(self) -> None:
        self.pending_undo_state = self.snapshot_editor_state() if self.raw_image is not None else None

    def commit_undoable_edit(self) -> None:
        if self.pending_undo_state is not None:
            self.undo_history.append(self.pending_undo_state)
            del self.undo_history[:-MAX_UNDO_STEPS]
        self.pending_undo_state = None

    def undo_last_action(self, _event: tk.Event | None = None) -> str:
        if not self.undo_history:
            self.status_var.set("没有可撤销的操作。")
            return "break"
        state = self.undo_history.pop()
        self.roi_canvas = state.roi_canvas
        self.annotations = [MeasurementAnnotation(item.kind, item.bounds_px, item.color) for item in state.annotations]
        self.active_annotation_index = state.active_annotation_index
        self.rotation_degrees = state.rotation_degrees
        self.rotation_var.set(f"{self.rotation_degrees:.2f}")
        self.preprocess_var.set(state.preprocess_enabled)
        self.result = state.result
        self.analysis_origin = state.analysis_origin
        self.rebuild_working_images()
        self.draw_image()
        if self.result is None:
            self.result_var.set("已撤销上一步操作。")
        else:
            self.update_result_text()
        self.status_var.set("已撤销上一步操作。")
        return "break"

    def rebuild_working_images(self) -> None:
        if self.original_image is None:
            return
        self.raw_image = rotate_grayscale_image(self.original_image, self.rotation_degrees)
        self.processed_image = normalize_and_denoise(self.raw_image)

    def auto_align_roi(self) -> None:
        try:
            roi, _, _ = self.selected_roi()
            correction = find_rotation_to_vertical(roi)
            if abs(correction) < 0.05:
                self.status_var.set("当前 ROI 已接近竖直，无需旋转。")
                return
            self.record_undo_state()
            view_center = self.current_view_center()
            self.rotation_degrees += correction
            self.rotation_var.set(f"{self.rotation_degrees:.2f}")
            self.rebuild_working_images()
            self.result = None
            self.analysis_origin = None
            self.roi_canvas = None
            self.clear_annotations(redraw=False, record_history=False)
            self.draw_image()
            self.restore_view_center(view_center)
            self.result_var.set("倾角已校正；请在校正后的图像上重新框选 ROI。")
            self.status_var.set(f"已校正 {correction:+.2f}°；累计旋转 {self.rotation_degrees:+.2f}°。")
        except ValueError as exc:
            messagebox.showwarning("无法校正倾角", str(exc))

    def reset_rotation(self) -> None:
        if self.original_image is None or abs(self.rotation_degrees) < 1e-12:
            return
        self.record_undo_state()
        self.rotation_degrees = 0.0
        self.rotation_var.set(f"{self.rotation_degrees:.2f}")
        self.reset_zoom_control()
        self.rebuild_working_images()
        self.result = None
        self.analysis_origin = None
        self.roi_canvas = None
        self.clear_annotations(redraw=False, record_history=False)
        self.draw_image(reset_view=True)
        self.result_var.set("已恢复原始图像方向；请重新框选 ROI。")
        self.status_var.set("已重置倾角校正。")

    def rotate_manually(self, delta_degrees: float) -> None:
        self.set_rotation_degrees(self.rotation_degrees + delta_degrees)

    def apply_manual_rotation_value(self) -> None:
        try:
            angle = float(self.rotation_var.get())
        except ValueError:
            messagebox.showwarning("角度无效", "请输入数字角度，例如 1.5 或 -0.3。")
            self.rotation_var.set(f"{self.rotation_degrees:.2f}")
            return
        self.set_rotation_degrees(angle)

    def set_rotation_degrees(self, angle_degrees: float) -> None:
        if self.original_image is None:
            self.rotation_var.set(f"{self.rotation_degrees:.2f}")
            return
        if math.isclose(angle_degrees, self.rotation_degrees, abs_tol=1e-9):
            self.rotation_var.set(f"{self.rotation_degrees:.2f}")
            return
        self.record_undo_state()
        view_center = self.current_view_center()
        self.rotation_degrees = angle_degrees
        self.rotation_var.set(f"{self.rotation_degrees:.2f}")
        self.rebuild_working_images()
        self.result = None
        self.analysis_origin = None
        self.roi_canvas = None
        self.clear_annotations(redraw=False, record_history=False)
        self.draw_image()
        self.restore_view_center(view_center)
        self.result_var.set("图像角度已手动调整；请重新框选 ROI 并分析。")
        self.status_var.set(f"当前累计旋转角度：{self.rotation_degrees:+.2f}°。")

    def show_metadata(self) -> None:
        window = tk.Toplevel(self)
        window.title("TIFF / SEM 元数据")
        window.geometry("820x650")
        viewer = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Menlo", 11))
        viewer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        viewer.insert("1.0", self.metadata_text)
        viewer.configure(state=tk.DISABLED)

    def refresh_preprocessing(self) -> None:
        if self.raw_image is None:
            return
        self.result = None
        self.analysis_origin = None
        self.draw_image()
        mode = "归一化 + 3×3 去噪" if self.preprocess_var.get() else "原始灰度图"
        self.result_var.set("处理方式已切换；请重新点击“分析选区”。")
        self.status_var.set(f"当前使用：{mode}")

    def draw_image(self, reset_view: bool = False) -> None:
        if self.raw_image is None:
            return
        image = self.processed_image if self.preprocess_var.get() else self.raw_image
        if image is None:
            return
        height, width = image.shape
        old_scale = self.display_scale
        roi_image_coordinates = None
        if self.roi_canvas is not None and old_scale > 0:
            roi_image_coordinates = tuple(value / old_scale for value in self.roi_canvas)
        self.base_scale = min(MAX_VIEW_WIDTH / width, MAX_VIEW_HEIGHT / height, 1.0)
        self.display_scale = self.base_scale * self.zoom_factor
        display_size = (max(1, round(width * self.display_scale)), max(1, round(height * self.display_scale)))
        preview = Image.fromarray(image).resize(display_size, Image.Resampling.LANCZOS)
        self.display_image = ImageTk.PhotoImage(preview)
        self.canvas.configure(scrollregion=(0, 0, *display_size))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.display_image, anchor=tk.NW)
        if roi_image_coordinates is not None:
            self.roi_canvas = tuple(value * self.display_scale for value in roi_image_coordinates)
        else:
            self.roi_rectangle = None
        self.draw_roi()
        self.render_analysis_overlay()
        self.draw_annotations()
        if reset_view:
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)

    def display_bounds(self) -> tuple[float, float]:
        if self.raw_image is None:
            return 0.0, 0.0
        return self.raw_image.shape[1] * self.display_scale, self.raw_image.shape[0] * self.display_scale

    def normalized_roi(self, bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        display_width, display_height = self.display_bounds()
        x0, y0, x1, y1 = bounds
        x0 = float(np.clip(x0, 0, display_width))
        x1 = float(np.clip(x1, 0, display_width))
        y0 = float(np.clip(y0, 0, display_height))
        y1 = float(np.clip(y1, 0, display_height))
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def draw_roi(self) -> None:
        self.canvas.delete("roi")
        self.roi_rectangle = None
        if self.roi_canvas is None:
            return
        self.roi_canvas = self.normalized_roi(self.roi_canvas)
        x0, y0, x1, y1 = self.roi_canvas
        self.roi_rectangle = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00ff66", width=2, tags=("roi",))
        handles = self.roi_handle_positions()
        for name, (x, y) in handles.items():
            fill = "#00ff66" if name != "move" else "#1aff8c"
            self.canvas.create_rectangle(
                x - ROI_HANDLE_SIZE,
                y - ROI_HANDLE_SIZE,
                x + ROI_HANDLE_SIZE,
                y + ROI_HANDLE_SIZE,
                fill=fill,
                outline="white",
                tags=("roi", f"roi_handle_{name}"),
            )

    def roi_handle_positions(self) -> dict[str, tuple[float, float]]:
        if self.roi_canvas is None:
            return {}
        x0, y0, x1, y1 = self.roi_canvas
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        return {
            "nw": (x0, y0),
            "n": (mid_x, y0),
            "ne": (x1, y0),
            "w": (x0, mid_y),
            "move": (mid_x, mid_y),
            "e": (x1, mid_y),
            "sw": (x0, y1),
            "s": (mid_x, y1),
            "se": (x1, y1),
        }

    def roi_hit_test(self, x: float, y: float) -> str | None:
        if self.roi_canvas is None:
            return None
        x0, y0, x1, y1 = self.roi_canvas
        for name, (handle_x, handle_y) in self.roi_handle_positions().items():
            if abs(x - handle_x) <= ROI_HANDLE_HIT and abs(y - handle_y) <= ROI_HANDLE_HIT:
                return name
        if x0 <= x <= x1 and y0 <= y <= y1:
            return "move"
        return None

    def update_canvas_cursor(self, event: tk.Event) -> None:
        if self.space_held:
            self.canvas.configure(cursor="fleur")
            return
        if self.annotation_at(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)) is not None:
            self.canvas.configure(cursor="fleur")
            return
        hit = self.roi_hit_test(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        cursors = {
            "nw": "crosshair",
            "se": "crosshair",
            "ne": "crosshair",
            "sw": "crosshair",
            "n": "sb_v_double_arrow",
            "s": "sb_v_double_arrow",
            "w": "sb_h_double_arrow",
            "e": "sb_h_double_arrow",
            "move": "fleur",
        }
        self.canvas.configure(cursor=cursors.get(hit, ""))

    def change_measurement_tool(self, _event: tk.Event | None = None) -> None:
        tool = self.measurement_tool_var.get()
        self.status_var.set(f"当前工具：{tool}。拖动可创建；点击已有标注可移动或调整。")

    def change_outlier_level(self, _event: tk.Event | None = None) -> None:
        if self.result is None:
            self.status_var.set(f"异常点剔除：{self.outlier_level_var.get()}。请点击“分析选区”应用。")
            return
        self.result = None
        self.analysis_origin = None
        self.draw_image()
        self.result_var.set("剔除档位已更改；请重新点击“分析选区”。")
        self.status_var.set("已更改异常点剔除设置，旧分析结果已清除。")

    def update_measurement_color_button(self) -> None:
        self.color_button.configure(background=self.measurement_color, activebackground=self.measurement_color)

    def choose_measurement_color(self) -> None:
        color = colorchooser.askcolor(color=self.measurement_color, parent=self, title="选择测量线颜色")[1]
        if color is None:
            return
        self.measurement_color = color
        self.update_measurement_color_button()
        if self.active_annotation_index is not None:
            annotation = self.annotations[self.active_annotation_index]
            if annotation.kind == "length":
                self.record_undo_state()
                annotation.color = color
                self.draw_annotations()
                self.status_var.set("已更新选中长度测量线的颜色。")
                return
        self.status_var.set("已选择长度测量线颜色；新建长度线会使用此颜色。")

    def selected_annotation_kind(self) -> str | None:
        return {
            "长度": "length",
            "矩形": "rectangle",
            "圆形": "circle",
            "正六边形": "hexagon",
        }.get(self.measurement_tool_var.get())

    def canvas_to_image_point(self, x: float, y: float) -> tuple[float, float]:
        return x / self.display_scale, y / self.display_scale

    def normalized_annotation_bounds(self, bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if self.raw_image is None:
            return bounds
        x0, y0, x1, y1 = bounds
        width, height = self.raw_image.shape[1], self.raw_image.shape[0]
        return (
            float(np.clip(x0, 0, width)),
            float(np.clip(y0, 0, height)),
            float(np.clip(x1, 0, width)),
            float(np.clip(y1, 0, height)),
        )

    def regularize_circle_bounds(self, bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = bounds
        delta_x, delta_y = x1 - x0, y1 - y0
        size = max(abs(delta_x), abs(delta_y))
        if size == 0:
            return bounds
        end_x = x0 + math.copysign(size, delta_x if delta_x else 1)
        end_y = y0 + math.copysign(size, delta_y if delta_y else 1)
        return x0, y0, end_x, end_y

    def annotation_display_bounds(self, annotation: MeasurementAnnotation) -> tuple[float, float, float, float]:
        return tuple(value * self.display_scale for value in annotation.bounds_px)

    def annotation_label(self, annotation: MeasurementAnnotation) -> str:
        x0, y0, x1, y1 = annotation.bounds_px
        pixel_size = self.measurement_pixel_size()
        unit = "nm" if pixel_size is not None else "px"
        scale = pixel_size if pixel_size is not None else 1.0
        width = abs(x1 - x0) * scale
        height = abs(y1 - y0) * scale
        if annotation.kind == "length":
            return f"L={math.hypot(x1 - x0, y1 - y0) * scale:.3f} {unit}"
        if annotation.kind == "rectangle":
            return f"W={width:.3f}, H={height:.3f} {unit}"
        if annotation.kind == "circle":
            return f"Ø={min(width, height):.3f} {unit}"
        radius = min(abs(x1 - x0) / 2, abs(y1 - y0) / math.sqrt(3)) * scale
        return f"S={radius:.3f}, D={2 * radius:.3f} {unit}"

    def annotation_color(self, annotation: MeasurementAnnotation) -> str:
        return annotation.color or ANNOTATION_COLORS[annotation.kind]

    def measurement_pixel_size(self) -> float | None:
        try:
            pixel_size = float(self.pixel_size_var.get())
        except ValueError:
            return None
        return pixel_size if pixel_size > 0 else None

    def refresh_annotation_labels(self, *_args: str) -> None:
        if self.raw_image is not None:
            self.draw_annotations()

    def annotation_polygon(self, annotation: MeasurementAnnotation, scale: float | None = None) -> list[tuple[float, float]]:
        scale = self.display_scale if scale is None else scale
        x0, y0, x1, y1 = annotation.bounds_px
        if annotation.kind == "rectangle":
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        elif annotation.kind == "circle":
            return []
        elif annotation.kind == "hexagon":
            center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
            radius = min(abs(x1 - x0) / 2, abs(y1 - y0) / math.sqrt(3))
            points = [
                (center_x - radius, center_y),
                (center_x - radius / 2, center_y - math.sqrt(3) * radius / 2),
                (center_x + radius / 2, center_y - math.sqrt(3) * radius / 2),
                (center_x + radius, center_y),
                (center_x + radius / 2, center_y + math.sqrt(3) * radius / 2),
                (center_x - radius / 2, center_y + math.sqrt(3) * radius / 2),
                (center_x - radius, center_y),
            ]
        else:
            points = [(x0, y0), (x1, y1)]
        return [(x * scale, y * scale) for x, y in points]

    def length_end_caps(self, annotation: MeasurementAnnotation, scale: float, cap_half_length: float) -> list[tuple[float, float, float, float]]:
        x0, y0, x1, y1 = (value * scale for value in annotation.bounds_px)
        line_length = math.hypot(x1 - x0, y1 - y0)
        if line_length == 0:
            return []
        cap_half_length = min(cap_half_length, line_length * 0.2)
        offset_x = -(y1 - y0) * cap_half_length / line_length
        offset_y = (x1 - x0) * cap_half_length / line_length
        return [
            (x0 - offset_x, y0 - offset_y, x0 + offset_x, y0 + offset_y),
            (x1 - offset_x, y1 - offset_y, x1 + offset_x, y1 + offset_y),
        ]

    def draw_annotations(self) -> None:
        self.canvas.delete("annotation")
        for index, annotation in enumerate(self.annotations):
            color = self.annotation_color(annotation)
            x0, y0, x1, y1 = self.annotation_display_bounds(annotation)
            if annotation.kind == "circle":
                self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=2, tags="annotation")
            else:
                points = self.annotation_polygon(annotation)
                self.canvas.create_line(*[value for point in points for value in point], fill=color, width=2, tags="annotation")
                if annotation.kind == "length":
                    for cap in self.length_end_caps(annotation, self.display_scale, 6):
                        self.canvas.create_line(*cap, fill=color, width=2, tags="annotation")
            label_x, label_y = min(x0, x1) + 5, min(y0, y1) - 7
            self.canvas.create_text(
                label_x,
                label_y,
                anchor=tk.SW,
                text=self.annotation_label(annotation),
                fill=color,
                font=("Menlo", 11, "bold"),
                tags="annotation",
            )
            if index == self.active_annotation_index:
                self.draw_annotation_handles(annotation)

    def draw_annotation_handles(self, annotation: MeasurementAnnotation) -> None:
        x0, y0, x1, y1 = self.annotation_display_bounds(annotation)
        if annotation.kind == "length":
            handles = [(x0, y0), (x1, y1)]
        else:
            handles = [(x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]
        for x, y in handles:
            self.canvas.create_rectangle(
                x - ROI_HANDLE_SIZE,
                y - ROI_HANDLE_SIZE,
                x + ROI_HANDLE_SIZE,
                y + ROI_HANDLE_SIZE,
                fill="#ffffff",
                outline="#202020",
                tags="annotation",
            )

    def annotation_at(self, canvas_x: float, canvas_y: float) -> tuple[int, str] | None:
        for index in range(len(self.annotations) - 1, -1, -1):
            mode = self.annotation_hit_test(self.annotations[index], canvas_x, canvas_y)
            if mode is not None:
                return index, mode
        return None

    def annotation_hit_test(self, annotation: MeasurementAnnotation, x: float, y: float) -> str | None:
        x0, y0, x1, y1 = self.annotation_display_bounds(annotation)
        if annotation.kind == "length":
            if math.hypot(x - x0, y - y0) <= ANNOTATION_HANDLE_HIT:
                return "start"
            if math.hypot(x - x1, y - y1) <= ANNOTATION_HANDLE_HIT:
                return "end"
            length = math.hypot(x1 - x0, y1 - y0)
            if length and abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / length <= 6:
                return "move"
            return None
        handles = {
            "nw": (x0, y0),
            "ne": (x1, y0),
            "sw": (x0, y1),
            "se": (x1, y1),
            "move": ((x0 + x1) / 2, (y0 + y1) / 2),
        }
        for name, (handle_x, handle_y) in handles.items():
            if abs(x - handle_x) <= ANNOTATION_HANDLE_HIT and abs(y - handle_y) <= ANNOTATION_HANDLE_HIT:
                return name
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if annotation.kind == "circle":
            radius_x, radius_y = max((right - left) / 2, 1), max((bottom - top) / 2, 1)
            return "move" if ((x - (left + right) / 2) / radius_x) ** 2 + ((y - (top + bottom) / 2) / radius_y) ** 2 <= 1 else None
        return "move" if left <= x <= right and top <= y <= bottom else None

    def start_canvas_action(self, event: tk.Event) -> None:
        if self.raw_image is None or self.space_held:
            self.start_roi(event)
            return
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        existing = self.annotation_at(canvas_x, canvas_y)
        if existing is not None:
            self.begin_undoable_edit()
            self.annotation_was_changed = False
            self.active_annotation_index, self.annotation_drag_mode = existing
            self.annotation_drag_anchor = self.canvas_to_image_point(canvas_x, canvas_y)
            self.annotation_start_bounds = self.annotations[self.active_annotation_index].bounds_px
            self.draw_annotations()
            return
        kind = self.selected_annotation_kind()
        if kind is None:
            self.active_annotation_index = None
            self.begin_undoable_edit()
            self.start_roi(event)
            return
        point = self.canvas_to_image_point(canvas_x, canvas_y)
        self.begin_undoable_edit()
        self.annotation_was_changed = False
        color = self.measurement_color if kind == "length" else None
        self.annotations.append(MeasurementAnnotation(kind, (*point, *point), color))
        self.active_annotation_index = len(self.annotations) - 1
        self.annotation_drag_mode = "create"
        self.annotation_drag_anchor = point
        self.annotation_start_bounds = self.annotations[-1].bounds_px
        self.draw_annotations()

    def move_canvas_action(self, event: tk.Event) -> None:
        if self.panning:
            self.move_roi(event)
            return
        if self.annotation_drag_mode is None or self.active_annotation_index is None:
            self.move_roi(event)
            return
        point = self.canvas_to_image_point(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.update_annotation_from_drag(point)
        self.annotation_was_changed = True
        self.draw_annotations()

    def finish_canvas_action(self, event: tk.Event) -> None:
        if self.panning:
            self.finish_roi(event)
            return
        if self.annotation_drag_mode is None or self.active_annotation_index is None:
            roi_was_changed = self.roi_was_changed
            self.finish_roi(event)
            if roi_was_changed:
                self.commit_undoable_edit()
            else:
                self.pending_undo_state = None
            return
        point = self.canvas_to_image_point(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.update_annotation_from_drag(point)
        self.annotation_drag_mode = None
        self.annotation_drag_anchor = None
        self.annotation_start_bounds = None
        if self.annotation_was_changed:
            self.commit_undoable_edit()
        else:
            self.pending_undo_state = None
        self.draw_annotations()
        self.status_var.set("通用测量标注已更新；可继续拖动控制点调整尺寸。")

    def update_annotation_from_drag(self, point: tuple[float, float]) -> None:
        if self.active_annotation_index is None or self.annotation_drag_mode is None or self.annotation_start_bounds is None:
            return
        annotation = self.annotations[self.active_annotation_index]
        x0, y0, x1, y1 = self.annotation_start_bounds
        x, y = point
        mode = self.annotation_drag_mode
        if mode == "create":
            bounds = (x0, y0, x, y)
        elif mode == "move" and self.annotation_drag_anchor is not None:
            anchor_x, anchor_y = self.annotation_drag_anchor
            bounds = (x0 + x - anchor_x, y0 + y - anchor_y, x1 + x - anchor_x, y1 + y - anchor_y)
        elif mode == "start":
            bounds = (x, y, x1, y1)
        elif mode == "end":
            bounds = (x0, y0, x, y)
        elif mode == "nw":
            bounds = (x, y, x1, y1)
        elif mode == "ne":
            bounds = (x0, y, x, y1)
        elif mode == "sw":
            bounds = (x, y0, x1, y)
        else:
            bounds = (x0, y0, x, y)
        if annotation.kind == "circle":
            bounds = self.regularize_circle_bounds(bounds)
        annotation.bounds_px = self.normalized_annotation_bounds(bounds)

    def delete_active_annotation(self, _event: tk.Event | None = None) -> str:
        if self.active_annotation_index is None:
            self.status_var.set("请先点击一个通用测量标注。")
            return "break"
        self.record_undo_state()
        self.annotations.pop(self.active_annotation_index)
        self.active_annotation_index = None
        self.draw_annotations()
        self.status_var.set("已删除选中标注。")
        return "break"

    def clear_annotations(self, redraw: bool = True, record_history: bool = True) -> None:
        if record_history and self.annotations:
            self.record_undo_state()
        self.annotations.clear()
        self.active_annotation_index = None
        self.annotation_drag_mode = None
        self.annotation_drag_anchor = None
        self.annotation_start_bounds = None
        if redraw and self.raw_image is not None:
            self.draw_annotations()
        if redraw:
            self.status_var.set("已清空所有通用测量标注。")

    def clear_selection(self, _event: tk.Event | None = None) -> str:
        if self.active_annotation_index is not None:
            return self.delete_active_annotation()
        return self.clear_roi()

    def update_roi_from_drag(self, x: float, y: float) -> None:
        if self.roi_drag_mode is None or self.roi_drag_anchor is None:
            return
        if self.roi_drag_mode == "create":
            ax, ay = self.roi_drag_anchor
            self.roi_canvas = self.normalized_roi((ax, ay, x, y))
            return
        if self.roi_start_bounds is None:
            return
        x0, y0, x1, y1 = self.roi_start_bounds
        if self.roi_drag_mode == "move":
            ax, ay = self.roi_drag_anchor
            dx, dy = x - ax, y - ay
            width, height = x1 - x0, y1 - y0
            display_width, display_height = self.display_bounds()
            new_x0 = float(np.clip(x0 + dx, 0, max(0.0, display_width - width)))
            new_y0 = float(np.clip(y0 + dy, 0, max(0.0, display_height - height)))
            self.roi_canvas = (new_x0, new_y0, new_x0 + width, new_y0 + height)
            return
        if self.roi_drag_mode == "nw":
            self.roi_canvas = self.normalized_roi((x, y, x1, y1))
        elif self.roi_drag_mode == "n":
            self.roi_canvas = self.normalized_roi((x0, y, x1, y1))
        elif self.roi_drag_mode == "ne":
            self.roi_canvas = self.normalized_roi((x0, y, x, y1))
        elif self.roi_drag_mode == "w":
            self.roi_canvas = self.normalized_roi((x, y0, x1, y1))
        elif self.roi_drag_mode == "e":
            self.roi_canvas = self.normalized_roi((x0, y0, x, y1))
        elif self.roi_drag_mode == "sw":
            self.roi_canvas = self.normalized_roi((x, y0, x1, y))
        elif self.roi_drag_mode == "s":
            self.roi_canvas = self.normalized_roi((x0, y0, x1, y))
        elif self.roi_drag_mode == "se":
            self.roi_canvas = self.normalized_roi((x0, y0, x, y))

    def current_view_center(self) -> tuple[float, float]:
        """Return the image coordinate currently at the center of the canvas."""
        return (
            self.canvas.canvasx(self.canvas.winfo_width() / 2) / self.display_scale,
            self.canvas.canvasy(self.canvas.winfo_height() / 2) / self.display_scale,
        )

    def restore_view_center(self, image_center: tuple[float, float]) -> None:
        """Keep the same image region visible after a redraw or rotation."""
        if self.raw_image is None:
            return
        self.update_idletasks()
        content_width = self.raw_image.shape[1] * self.display_scale
        content_height = self.raw_image.shape[0] * self.display_scale
        view_width = self.canvas.winfo_width()
        view_height = self.canvas.winfo_height()
        if content_width > view_width:
            position = (image_center[0] * self.display_scale - view_width / 2) / (content_width - view_width)
            self.canvas.xview_moveto(float(np.clip(position, 0, 1)))
        if content_height > view_height:
            position = (image_center[1] * self.display_scale - view_height / 2) / (content_height - view_height)
            self.canvas.yview_moveto(float(np.clip(position, 0, 1)))

    def reset_zoom_control(self) -> None:
        self.zoom_factor = 1.0
        self.zoom_slider_var.set(1.0)
        self.zoom_percent_var.set("100%")

    def step_zoom(self, change: float) -> None:
        self.apply_zoom(self.zoom_factor + change)

    def zoom_from_slider(self, value: str) -> None:
        self.apply_zoom(float(value))

    def zoom_with_space_wheel(self, event: tk.Event) -> str | None:
        if not self.space_held:
            return None
        delta = getattr(event, "delta", 0)
        if not delta:
            delta = 1 if getattr(event, "num", 0) == 4 else -1
        factor = 1.2 if delta > 0 else 1 / 1.2
        return self.apply_zoom(self.zoom_factor * factor, event.x, event.y)

    def apply_zoom(self, requested_zoom: float, view_x: float | None = None, view_y: float | None = None) -> str | None:
        if self.raw_image is None:
            return "break"
        new_zoom = float(np.clip(requested_zoom, ZOOM_MIN, ZOOM_MAX))
        if math.isclose(new_zoom, self.zoom_factor):
            return "break"
        if view_x is None:
            view_x = self.canvas.winfo_width() / 2
        if view_y is None:
            view_y = self.canvas.winfo_height() / 2
        image_x = self.canvas.canvasx(view_x) / self.display_scale
        image_y = self.canvas.canvasy(view_y) / self.display_scale
        self.zoom_factor = new_zoom
        self.zoom_slider_var.set(new_zoom)
        self.zoom_percent_var.set(f"{new_zoom * 100:.0f}%")
        self.draw_image()
        self.update_idletasks()
        content_width = self.raw_image.shape[1] * self.display_scale
        content_height = self.raw_image.shape[0] * self.display_scale
        view_width = self.canvas.winfo_width()
        view_height = self.canvas.winfo_height()
        if content_width > view_width:
            position = (image_x * self.display_scale - view_x) / (content_width - view_width)
            self.canvas.xview_moveto(float(np.clip(position, 0, 1)))
        if content_height > view_height:
            position = (image_y * self.display_scale - view_y) / (content_height - view_height)
            self.canvas.yview_moveto(float(np.clip(position, 0, 1)))
        self.status_var.set(f"图像缩放：{self.zoom_factor * 100:.0f}%")

    def start_pan_mode(self, _event: tk.Event) -> None:
        self.space_held = True
        self.canvas.configure(cursor="fleur")

    def end_pan_mode(self, _event: tk.Event) -> None:
        self.space_held = False
        if not self.panning:
            self.canvas.configure(cursor="")

    def start_roi(self, event: tk.Event) -> None:
        if self.raw_image is None:
            return
        self.canvas.focus_set()
        if self.space_held:
            self.panning = True
            self.canvas.scan_mark(event.x, event.y)
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        hit = self.roi_hit_test(x, y)
        self.roi_drag_mode = hit or "create"
        self.roi_drag_anchor = (x, y)
        self.roi_start_bounds = self.roi_canvas
        self.roi_was_changed = False
        if self.roi_drag_mode == "create":
            self.drag_start = (x, y)
            self.roi_canvas = (x, y, x, y)
            self.draw_roi()

    def move_roi(self, event: tk.Event) -> None:
        if self.panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            return
        if self.roi_drag_mode is not None:
            self.update_roi_from_drag(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            self.roi_was_changed = True
            self.draw_roi()

    def finish_roi(self, event: tk.Event) -> None:
        if self.panning:
            self.panning = False
            if not self.space_held:
                self.canvas.configure(cursor="")
            return
        if self.roi_drag_mode is None:
            return
        self.update_roi_from_drag(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.drag_start = None
        self.roi_drag_mode = None
        self.roi_drag_anchor = None
        self.roi_start_bounds = None
        self.draw_roi()
        if self.roi_was_changed:
            self.result = None
            self.analysis_origin = None
            self.canvas.delete("edge")
            self.result_var.set("选区已更新；请重新点击“分析选区”。")
        self.status_var.set("选区已确定。输入像素尺寸后点击“分析选区”。")
        self.roi_was_changed = False

    def clear_roi(self, _event: tk.Event | None = None) -> str:
        if self.roi_canvas is None:
            return "break"
        self.record_undo_state()
        self.roi_canvas = None
        self.roi_rectangle = None
        self.drag_start = None
        self.roi_drag_mode = None
        self.result = None
        self.analysis_origin = None
        self.canvas.delete("roi")
        self.canvas.delete("edge")
        self.result_var.set("选区已删除；请重新框选 ROI。")
        self.status_var.set("已删除选区。")
        return "break"

    def selected_roi(self) -> tuple[np.ndarray, int, int]:
        if self.raw_image is None or self.roi_canvas is None:
            raise ValueError("请先打开图像并拖动鼠标框选分析区域。")
        x0, y0, x1, y1 = self.roi_canvas
        left = max(0, math.floor(x0 / self.display_scale))
        top = max(0, math.floor(y0 / self.display_scale))
        right = min(self.raw_image.shape[1], math.ceil(x1 / self.display_scale))
        bottom = min(self.raw_image.shape[0], math.ceil(y1 / self.display_scale))
        if right - left < MIN_ROI_WIDTH or bottom - top < MIN_ROI_HEIGHT:
            raise ValueError("选区过小；请框选更长的一段线及两侧背景。")
        analysis_image = self.processed_image if self.preprocess_var.get() else self.raw_image
        if analysis_image is None:
            raise ValueError("图像未准备完成。")
        return analysis_image[top:bottom, left:right], left, top

    def lcdu_sigma_nm(self) -> float | None:
        sample_count = len(self.lcdu_cd_samples_nm)
        if sample_count < 2:
            return None
        samples = np.array(self.lcdu_cd_samples_nm, dtype=float)
        residual = samples - np.mean(samples)
        return math.sqrt(float(np.dot(residual, residual)) / (sample_count - 1))

    def update_lcdu_summary_text(self) -> None:
        self.refresh_lcdu_sample_list()
        sample_count = len(self.lcdu_cd_samples_nm)
        sigma = self.lcdu_sigma_nm()
        if sigma is None:
            self.lcdu_summary_var.set(f"LCDU 样本：{sample_count} 条线；至少加入 2 条线后计算。")
            return
        multiplier = self.sigma_multiplier_var.get()
        self.lcdu_summary_var.set(
            f"LCDU 样本：{sample_count} 条线\n"
            f"LCDU：{multiplier * sigma:.3f} nm  ({multiplier:.1f}σ)，原始 σ={sigma:.3f} nm"
        )

    def refresh_lcdu_sample_list(self) -> None:
        if self.lcdu_sample_listbox is None:
            return
        selected = self.lcdu_sample_listbox.curselection()
        selected_index = selected[0] if selected else None
        self.lcdu_sample_listbox.delete(0, tk.END)
        for index, mean_cd_nm in enumerate(self.lcdu_cd_samples_nm, start=1):
            self.lcdu_sample_listbox.insert(tk.END, f"{index:02d}. CD = {mean_cd_nm:.3f} nm")
        if selected_index is not None and self.lcdu_cd_samples_nm:
            next_index = min(selected_index, len(self.lcdu_cd_samples_nm) - 1)
            self.lcdu_sample_listbox.selection_set(next_index)
            self.lcdu_sample_listbox.activate(next_index)

    def add_lcdu_sample(self) -> None:
        if self.result is None:
            messagebox.showinfo("暂无 CD", "请先分析一条线，再加入 LCDU 样本。")
            return
        self.lcdu_cd_samples_nm.append(self.result.mean_width_nm)
        self.update_lcdu_summary_text()
        self.status_var.set(f"已加入当前线 CD={self.result.mean_width_nm:.3f} nm；LCDU 样本数 {len(self.lcdu_cd_samples_nm)}。")

    def clear_lcdu_samples(self) -> None:
        self.lcdu_cd_samples_nm.clear()
        self.update_lcdu_summary_text()
        self.status_var.set("已清空 LCDU 样本。")

    def delete_selected_lcdu_sample(self, _event: tk.Event | None = None) -> str:
        if self.lcdu_sample_listbox is None:
            return "break"
        selected = self.lcdu_sample_listbox.curselection()
        if not selected:
            self.status_var.set("请先在 LCDU 样本列表中选中一条。")
            return "break"
        index = selected[0]
        removed_cd = self.lcdu_cd_samples_nm.pop(index)
        self.update_lcdu_summary_text()
        self.status_var.set(f"已删除第 {index + 1} 条 LCDU 样本：CD={removed_cd:.3f} nm。")
        return "break"

    def run_analysis(self) -> None:
        try:
            pixel_size = float(self.pixel_size_var.get())
            roi, left_offset, top_offset = self.selected_roi()
            self.result = analyze_roi(roi, pixel_size, self.outlier_threshold_px())
            self.analysis_origin = (left_offset, top_offset)
            self.draw_image()
            self.update_result_text()
            self.status_var.set("分析完成。可调整 σ 倍数，或导出轨迹和结果。")
        except ValueError as exc:
            messagebox.showwarning("无法分析", str(exc))

    def outlier_threshold_px(self) -> float:
        return OUTLIER_LEVELS.get(self.outlier_level_var.get(), OUTLIER_LEVELS[DEFAULT_OUTLIER_LEVEL])

    def render_analysis_overlay(self) -> None:
        self.canvas.delete("edge")
        if self.result is None or self.analysis_origin is None:
            return
        result = self.result
        left_offset, top_offset = self.analysis_origin
        scale = self.display_scale
        points_left = []
        points_right = []
        for row, left, right in zip(result.rows_px, result.left_px, result.right_px):
            points_left.extend([(left + left_offset) * scale, (row + top_offset) * scale])
            points_right.extend([(right + left_offset) * scale, (row + top_offset) * scale])
        if len(points_left) >= 4:
            self.canvas.create_line(*points_left, fill="#00e5ff", width=1, tags="edge")
        if len(points_right) >= 4:
            self.canvas.create_line(*points_right, fill="#ffb000", width=1, tags="edge")

    def update_result_text(self) -> None:
        if self.result is None:
            return
        multiplier = self.sigma_multiplier_var.get()
        result = self.result
        self.result_var.set(
            f"输出：{multiplier:.1f}σ\n\n"
            f"左 ER（LER_L）  {multiplier * result.ler_left_sigma_nm:.3f} nm\n"
            f"右 ER（LER_R）  {multiplier * result.ler_right_sigma_nm:.3f} nm\n"
            f"宽度 LWR        {multiplier * result.lwr_sigma_nm:.3f} nm\n"
            f"\n"
            f"原始 σ\n"
            f"左 ER（LER_L）  {result.ler_left_sigma_nm:.3f} nm\n"
            f"右 ER（LER_R）  {result.ler_right_sigma_nm:.3f} nm\n"
            f"LWR             {result.lwr_sigma_nm:.3f} nm\n"
            f"总边缘 RMS      {result.total_ler_sigma_nm:.3f} nm\n"
            f"边缘相关系数 ρ  {result.edge_correlation:.3f}\n\n"
            f"平均线宽  {result.mean_width_nm:.3f} nm\n"
            f"有效点数  {len(result.rows_px)}\n"
            f"剔除点数  {result.rejected_rows}\n"
            f"剔除阈值  {self.outlier_threshold_px():.0f} px"
        )

    def update_displayed_results(self) -> None:
        self.update_result_text()
        self.update_lcdu_summary_text()

    def export_annotated_image(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("暂无图像", "请先打开一张 SEM 图像。")
            return
        image = normalize_and_denoise(self.original_image) if self.preprocess_var.get() else self.original_image
        target = filedialog.asksaveasfilename(
            title="保存拟合图片",
            defaultextension=".png",
            initialfile=f"{self.image_path.stem if self.image_path else 'sem'}_annotated.png",
            filetypes=[("PNG 图片", "*.png")],
        )
        if not target:
            return

        height, width = image.shape
        output = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(output)
        if self.roi_canvas is not None and self.display_scale > 0:
            x0, y0, x1, y1 = (value / self.display_scale for value in self.roi_canvas)
            roi_points = rotate_points_about_center(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                -self.rotation_degrees,
                width,
                height,
            )
            draw.line(roi_points, fill=(0, 255, 102), width=2)

        self.draw_exported_annotations(draw, width, height)
        if self.result is not None and self.analysis_origin is not None:
            result = self.result
            left_offset, top_offset = self.analysis_origin
            left_points_rotated = [
                (float(left + left_offset), float(row + top_offset))
                for row, left in zip(result.rows_px, result.left_px)
            ]
            right_points_rotated = [
                (float(right + left_offset), float(row + top_offset))
                for row, right in zip(result.rows_px, result.right_px)
            ]
            left_points = rotate_points_about_center(left_points_rotated, -self.rotation_degrees, width, height)
            right_points = rotate_points_about_center(right_points_rotated, -self.rotation_degrees, width, height)
            if len(left_points) >= 2:
                draw.line(left_points, fill=(0, 229, 255), width=2)
            if len(right_points) >= 2:
                draw.line(right_points, fill=(255, 176, 0), width=2)

        try:
            output.save(target, "PNG")
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.status_var.set(f"已导出拟合图片：{target}")

    def annotation_image_points(self, annotation: MeasurementAnnotation) -> list[tuple[float, float]]:
        if annotation.kind != "circle":
            return self.annotation_polygon(annotation, scale=1.0)
        x0, y0, x1, y1 = annotation.bounds_px
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        radius_x, radius_y = abs(x1 - x0) / 2, abs(y1 - y0) / 2
        return [
            (center_x + radius_x * math.cos(angle), center_y + radius_y * math.sin(angle))
            for angle in np.linspace(0, 2 * math.pi, 41)
        ]

    def draw_exported_annotations(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        for annotation in self.annotations:
            points = rotate_points_about_center(
                self.annotation_image_points(annotation),
                -self.rotation_degrees,
                width,
                height,
            )
            if len(points) >= 2:
                draw.line(points, fill=self.annotation_color(annotation), width=2)
            if annotation.kind == "length":
                for cap in self.length_end_caps(annotation, 1.0, 6):
                    cap_points = rotate_points_about_center(
                        [(cap[0], cap[1]), (cap[2], cap[3])],
                        -self.rotation_degrees,
                        width,
                        height,
                    )
                    draw.line(cap_points, fill=self.annotation_color(annotation), width=2)
            x0, y0, _x1, _y1 = annotation.bounds_px
            label_point = rotate_points_about_center([(x0, y0)], -self.rotation_degrees, width, height)[0]
            draw.text((label_point[0] + 4, label_point[1] - 14), self.annotation_label(annotation), fill=self.annotation_color(annotation))

    def export_csv(self) -> None:
        if self.result is None and not self.annotations:
            messagebox.showinfo("暂无结果", "请先完成 LER/LWR 分析或添加通用测量标注。")
            return
        target = filedialog.asksaveasfilename(
            title="保存测量结果",
            defaultextension=".csv",
            initialfile=f"{self.image_path.stem if self.image_path else 'sem'}_measurements.csv",
            filetypes=[("CSV 文件", "*.csv")],
        )
        if not target:
            return
        multiplier = self.sigma_multiplier_var.get()
        try:
            with open(target, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source", str(self.image_path) if self.image_path else ""])
                pixel_size = self.measurement_pixel_size()
                writer.writerow(["pixel_size_nm_per_px", "" if pixel_size is None else pixel_size])
                writer.writerow(["sigma_multiplier", multiplier])
                writer.writerow(["outlier_rejection_threshold_px", self.outlier_threshold_px()])
                if self.result is not None:
                    result = self.result
                    writer.writerow(["ler_left_sigma_nm", result.ler_left_sigma_nm])
                    writer.writerow(["ler_right_sigma_nm", result.ler_right_sigma_nm])
                    writer.writerow(["lwr_sigma_nm", result.lwr_sigma_nm])
                    writer.writerow(["total_ler_sigma_nm", result.total_ler_sigma_nm])
                    writer.writerow(["edge_correlation_rho", result.edge_correlation])
                    writer.writerow(["ler_left_display_nm", multiplier * result.ler_left_sigma_nm])
                    writer.writerow(["ler_right_display_nm", multiplier * result.ler_right_sigma_nm])
                    writer.writerow(["lwr_display_nm", multiplier * result.lwr_sigma_nm])
                    writer.writerow(["mean_width_nm", result.mean_width_nm])
                lcdu_sigma = self.lcdu_sigma_nm()
                writer.writerow(["multi_line_lcdu_sample_count", len(self.lcdu_cd_samples_nm)])
                writer.writerow(["multi_line_lcdu_sigma_nm", "" if lcdu_sigma is None else lcdu_sigma])
                writer.writerow(["multi_line_lcdu_display_nm", "" if lcdu_sigma is None else multiplier * lcdu_sigma])
                writer.writerow([])
                writer.writerow(["multi_line_lcdu_sample_index", "mean_cd_nm"])
                for index, mean_cd_nm in enumerate(self.lcdu_cd_samples_nm, start=1):
                    writer.writerow([index, mean_cd_nm])
                writer.writerow([])
                writer.writerow(["annotation_index", "type", "x0_px", "y0_px", "x1_px", "y1_px", "display_label"])
                for index, annotation in enumerate(self.annotations, start=1):
                    writer.writerow([index, annotation.kind, *annotation.bounds_px, self.annotation_label(annotation)])
                if self.result is not None:
                    result = self.result
                    writer.writerow([])
                    writer.writerow(["row_px", "left_edge_px", "right_edge_px", "left_residual_nm", "right_residual_nm", "width_nm", "width_residual_nm"])
                    for row in zip(
                        result.rows_px,
                        result.left_px,
                        result.right_px,
                        result.left_residual_nm,
                        result.right_residual_nm,
                        result.width_nm,
                        result.width_residual_nm,
                    ):
                        writer.writerow(row)
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.status_var.set(f"已导出：{target}")


if __name__ == "__main__":
    instance_lock = acquire_single_instance_lock()
    if instance_lock is None:
        show_already_running_message()
    else:
        LERLWRApp().mainloop()
        instance_lock.close()
