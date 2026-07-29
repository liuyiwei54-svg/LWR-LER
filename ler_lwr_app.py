"""Interactive SEM LER/LWR and perpendicular BCP dot-array measurement tool.

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
import time
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
MIN_ERA_ROI_WIDTH = 12
MIN_ERA_ROI_HEIGHT = 80
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
INFO_BAR_SEARCH_START = 0.72
INFO_BAR_MIN_HEIGHT_PX = 24
GAUSSIAN_KERNEL_SIZES = (3, 4, 5, 7, 9, 11)
FIRST_ORDER_KERNEL_SIZES = (3, 5, 7, 9, 11)
GAUSSIAN_SIGMA_SCALE_MIN = 0.4
GAUSSIAN_SIGMA_SCALE_MAX = 2.5
BLUR_METHOD_GAUSSIAN = "高斯模糊（默认）"
BLUR_METHOD_MEAN = "均值模糊"
EDGE_DETECTOR_SCHARR = "Scharr（默认）"
EDGE_DETECTOR_CANNY = "Canny（Sobel）"
EDGE_DETECTOR_LAPLACIAN = "拉普拉斯"
EDGE_DETECTOR_ERA = "单边测量（ERA）"
ERA_POLARITY_AUTO = "自动"
ERA_POLARITY_RISING = "上升沿（暗→亮）"
ERA_POLARITY_FALLING = "下降沿（亮→暗）"
ERA_POLARITIES = (ERA_POLARITY_AUTO, ERA_POLARITY_RISING, ERA_POLARITY_FALLING)
ERA_DEFAULT_POLYNOMIAL_DEGREE = 3
ERA_DIFFERENCE_POWER = 4
VERTICAL_EDGE_SUPPORT_WINDOW_PX = 41
VERTICAL_EDGE_SUPPORT_MIN_POINTS = 24
CD_RAY_STEP_PX = 0.25
PITCH_DIRECTION_LABELS = ("水平（0°）", "+60°", "−60°")
PITCH_DIRECTION_ANGLES = (0.0, 60.0, 120.0)
PITCH_DIRECTION_TOLERANCE_DEGREES = 15.0
GRAIN_BOUNDARY_ORIENTATION_THRESHOLD_DEGREES = 8.0
MIN_GRAIN_DOTS = 3
GRAIN_OVERLAY_ALPHA = 72
GRAIN_OVERLAY_MAX_SIDE_PX = 360
CENTROID_LAYOUT_OUTLINE_COLOR = "#c638ff"
CENTROID_LAYOUT_ANCHOR_COLOR = "#ff3748"
GRAIN_OVERLAY_COLORS = np.asarray(
    (
        (239, 111, 108), (78, 161, 255), (255, 190, 80), (106, 203, 138),
        (181, 126, 220), (65, 196, 193), (240, 139, 190), (170, 185, 95),
    ),
    dtype=np.uint8,
)
TRIANGULATION_DISPLAY_ALL = "全部方向"
TRIANGULATION_DISPLAY_MODES = (TRIANGULATION_DISPLAY_ALL, "主方向", "主方向 +60°", "主方向 −60°")
BCP_LOCAL_PITCH_NEIGHBORS = 3
BCP_MUTUAL_NEIGHBORS = 6
BCP_MAX_LOCAL_LINK_FACTOR = 1.45
BCP_BLOCKING_CORRIDOR_FACTOR = 0.35
BCP_BLOCKING_PROJECTION_MARGIN = 0.12
MANUAL_BCP_SPLIT_LINE_WIDTH_PX = 3
DEFAULT_BCP_CONTOUR_COLOR = "#39ff8e"
DEFAULT_BCP_TRIANGULATION_COLOR = "#4db8ff"
DEFAULT_BCP_OVERLAY_LINE_WIDTH = "2"
DEFAULT_BCP_CENTROID_COLOR = "#ffe066"
DEFAULT_BCP_CENTROID_DIAMETER = "6"
DEFAULT_FIT_LEFT_COLOR = "#00e5ff"
DEFAULT_FIT_RIGHT_COLOR = "#ffb000"
DEFAULT_FIT_LINE_WIDTH = "2"


class VerticalScrollFrame(ttk.Frame):
    """A two-axis scrollable frame for compact application windows."""

    def __init__(self, parent: tk.Misc, padding: int | tuple[int, int, int, int] = 0) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vertical_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.horizontal_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.content = ttk.Frame(self.canvas, padding=padding)
        self.content_window = self.canvas.create_window((0, 0), window=self.content, anchor=tk.NW)
        self.canvas.configure(
            xscrollcommand=self.horizontal_scrollbar.set,
            yscrollcommand=self.vertical_scrollbar.set,
        )
        self.vertical_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.horizontal_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.content.bind("<Configure>", self.update_scroll_region)
        self.canvas.bind("<Configure>", self.resize_content)

    def update_scroll_region(self, _event: tk.Event | None = None) -> None:
        required_width = max(self.canvas.winfo_width(), self.content.winfo_reqwidth())
        self.canvas.itemconfigure(self.content_window, width=required_width)
        self.canvas.configure(scrollregion=(0, 0, required_width, self.content.winfo_reqheight()))

    def resize_content(self, _event: tk.Event) -> None:
        self.update_scroll_region()

    def scroll_wheel(self, event: tk.Event) -> str:
        delta = getattr(event, "delta", 0)
        if delta:
            direction = -1 if delta > 0 else 1
        else:
            direction = -1 if getattr(event, "num", 0) == 4 else 1
        if event.state & 0x0001:
            self.canvas.xview_scroll(direction, "units")
        else:
            self.canvas.yview_scroll(direction, "units")
        return "break"


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
    edge_detector: str
    edge_kernel_size: int
    edge_diagonal_weight: float
    edge_axial_weight: float
    canny_high_threshold: float
    canny_threshold_ratio: float
    canny_low_threshold: float
    canny_normalization_scale: float


@dataclass
class SingleEdgeResult:
    """One independently fitted edge, expressed in working-image coordinates."""

    rows_px: np.ndarray
    edge_px: np.ndarray
    residual_nm: np.ndarray
    ler_sigma_nm: float
    rejected_rows: int
    roi_bounds_px: tuple[int, int, int, int]
    polarity: str
    polynomial_degree: int


@dataclass
class EraLineSample:
    """User-defined left/right single-edge ROIs for one physical line."""

    left_roi_bounds_px: tuple[int, int, int, int] | None = None
    right_roi_bounds_px: tuple[int, int, int, int] | None = None
    left_edge: SingleEdgeResult | None = None
    right_edge: SingleEdgeResult | None = None
    paired_result: AnalysisResult | None = None


@dataclass
class EraAggregateResult:
    """Pooled roughness statistics after independently measuring several lines."""

    line_count: int
    point_count: int
    mean_width_nm: float
    ler_left_sigma_nm: float
    ler_right_sigma_nm: float
    lwr_sigma_nm: float
    total_ler_sigma_nm: float
    edge_correlation: float


@dataclass
class BCPAnalysisResult:
    """Detected perpendicular BCP dots and grain-boundary segments in image pixels."""

    centers_px: np.ndarray
    equivalent_diameters_px: np.ndarray
    major_axes_px: np.ndarray
    minor_axes_px: np.ndarray
    angles_degrees: np.ndarray
    areas_px: np.ndarray
    contour_segments_px: list[np.ndarray]
    components_px: list[np.ndarray]
    directional_cds_px: np.ndarray
    cd_means_px: np.ndarray
    triangulation_segments_px: np.ndarray
    pitch_values_by_direction_px: dict[str, np.ndarray]
    boundary_segments_px: np.ndarray
    boundary_dot_indices: np.ndarray
    lattice_spacing_px: float
    pixel_size_nm: float


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
    normalize_enabled: bool
    gaussian_denoise_enabled: bool
    gaussian_kernel_size: str
    gaussian_sigma_scale: float
    blur_method: str
    bcp_dog_enabled: bool
    result: AnalysisResult | None
    analysis_origin: tuple[int, int] | None
    bcp_result: BCPAnalysisResult | None
    bcp_metrics_finalized: bool


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


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Apply 1%–99% intensity normalization without changing the source image."""
    low, high = np.percentile(image.astype(float), (1.0, 99.0))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def gaussian_kernel(size: int, sigma_scale: float = 1.0) -> np.ndarray:
    """Return a normalized Gaussian kernel, preserving binomial weights at default strength."""
    if size not in GAUSSIAN_KERNEL_SIZES:
        raise ValueError("高斯卷积核仅支持 3×3、4×4、5×5、7×7、9×9 或 11×11。")
    row = np.array([math.comb(size - 1, index) for index in range(size)], dtype=float)
    if math.isclose(sigma_scale, 1.0, abs_tol=1e-9):
        kernel = np.outer(row, row)
        return kernel / kernel.sum()
    base_sigma = 0.5 + 0.175 * (size - 1)
    sigma = base_sigma * float(np.clip(sigma_scale, GAUSSIAN_SIGMA_SCALE_MIN, GAUSSIAN_SIGMA_SCALE_MAX))
    positions = np.arange(size, dtype=float) - (size - 1) / 2
    row = np.exp(-(positions**2) / (2 * sigma**2))
    kernel = np.outer(row, row)
    return kernel / kernel.sum()


def gaussian_kernel_preview(size: int, sigma_scale: float = 1.0) -> str:
    """Format the selected integer convolution weights for the compact UI preview."""
    if math.isclose(sigma_scale, 1.0, abs_tol=1e-9):
        row = [math.comb(size - 1, index) for index in range(size)]
        denominator = sum(row) ** 2
        lines = ["  ".join(f"{left * right:>2}" for right in row) for left in row]
        return f"{size}×{size} 核（标准 ÷{denominator}）\n" + "\n".join(lines)
    kernel = gaussian_kernel(size, sigma_scale)
    lines = ["  ".join(f"{weight:.3f}" for weight in row) for row in kernel]
    return f"{size}×{size} 核（σ×{sigma_scale:.2f}）\n" + "\n".join(lines)


def mean_kernel_preview(size: int) -> str:
    """Format an equal-weight mean-blur kernel for the compact UI preview."""
    value = 1.0 / (size * size)
    lines = ["  ".join(f"{value:.3f}" for _column in range(size)) for _row in range(size)]
    return f"{size}×{size} 均值核（每项 ÷{size * size}）\n" + "\n".join(lines)


def gaussian_convolve(image: np.ndarray, size: int, sigma_scale: float = 1.0) -> np.ndarray:
    """Apply an edge-preserving-size two-dimensional Gaussian convolution."""
    top_pad, bottom_pad = size // 2, size - 1 - size // 2
    padded = np.pad(image.astype(float), ((top_pad, bottom_pad), (top_pad, bottom_pad)), mode="edge")
    convolved = np.zeros(image.shape, dtype=float)
    for row, weights in enumerate(gaussian_kernel(size, sigma_scale)):
        for column, weight in enumerate(weights):
            convolved += weight * padded[row : row + image.shape[0], column : column + image.shape[1]]
    return np.clip(convolved, 0, 255).astype(np.uint8)


def mean_convolve(image: np.ndarray, size: int) -> np.ndarray:
    """Apply an equal-weight mean blur with the selected square kernel size."""
    top_pad, bottom_pad = size // 2, size - 1 - size // 2
    padded = np.pad(image.astype(float), ((top_pad, bottom_pad), (top_pad, bottom_pad)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    summed = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return np.clip(summed / (size * size), 0, 255).astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    normalize: bool = True,
    gaussian_denoise: bool = True,
    gaussian_size: int = 3,
    gaussian_sigma_scale: float = 1.0,
    blur_method: str = BLUR_METHOD_GAUSSIAN,
) -> np.ndarray:
    """Build the selected display and measurement image from raw grayscale data."""
    processed = normalize_intensity(image) if normalize else image.copy()
    if not gaussian_denoise:
        return processed
    if blur_method == BLUR_METHOD_MEAN:
        return mean_convolve(processed, gaussian_size)
    return gaussian_convolve(processed, gaussian_size, gaussian_sigma_scale)


def normalize_and_denoise(image: np.ndarray) -> np.ndarray:
    """Keep the former public helper available with the new 3×3 default."""
    return preprocess_image(image, normalize=True, gaussian_denoise=True, gaussian_size=3)


def remove_bottom_information_bar(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Crop a sustained dark SEM instrument-information bar from the image bottom."""
    height = image.shape[0]
    start = int(height * INFO_BAR_SEARCH_START)
    if height - start < INFO_BAR_MIN_HEIGHT_PX:
        return image, 0
    row_means = image.astype(float).mean(axis=1)
    for row in range(start, height - INFO_BAR_MIN_HEIGHT_PX):
        preceding = row_means[max(0, row - 40) : row]
        following = row_means[row : row + INFO_BAR_MIN_HEIGHT_PX]
        if len(preceding) and np.median(following) < np.median(preceding) - 25.0:
            return image[:row].copy(), height - row
    return image, 0


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


def rolling_median(trace: np.ndarray, window: int = 9) -> np.ndarray:
    """Return a short local-median baseline without an extra SciPy dependency."""
    radius = window // 2
    padded = np.pad(trace, (radius, radius), mode="edge")
    return np.array([np.median(padded[index : index + window]) for index in range(len(trace))])


def replace_outliers(trace: np.ndarray, max_jump_px: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated trace failures without smoothing genuine roughness."""
    observed = np.isfinite(trace)
    if not observed.any():
        return np.zeros_like(trace), np.zeros(trace.shape, dtype=bool)
    indices = np.arange(len(trace))
    filled = np.interp(indices, indices[observed], trace[observed])
    baseline = rolling_median(filled, window=9)
    valid = observed & (np.abs(filled - baseline) <= max_jump_px)
    repaired = filled.copy()
    repaired[~valid] = baseline[~valid]
    return repaired, valid


def convolve_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a small odd square kernel without adding a SciPy dependency."""
    radius = kernel.shape[0] // 2
    padded = np.pad(image.astype(float), radius, mode="edge")
    output = np.zeros(image.shape, dtype=float)
    for row in range(kernel.shape[0]):
        for column in range(kernel.shape[1]):
            output += kernel[row, column] * padded[row : row + image.shape[0], column : column + image.shape[1]]
    return output


def first_order_kernels(
    detector: str,
    diagonal_weight: float,
    axial_weight: float,
    kernel_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an adjustable 3×3 or extended separable first-order kernel."""
    if kernel_size not in FIRST_ORDER_KERNEL_SIZES:
        raise ValueError("一阶边缘卷积核仅支持 3×3、5×5、7×7、9×9 或 11×11。")
    if detector == EDGE_DETECTOR_CANNY:
        diagonal_weight, axial_weight = max(diagonal_weight, 0.01), max(axial_weight, 0.01)
    if kernel_size == 3:
        kernel_x = np.array(
            ((diagonal_weight, 0.0, -diagonal_weight), (axial_weight, 0.0, -axial_weight), (diagonal_weight, 0.0, -diagonal_weight)),
            dtype=float,
        )
        return kernel_x, kernel_x.T

    smoothing = np.array([math.comb(kernel_size - 1, index) for index in range(kernel_size)], dtype=float)
    center_index = kernel_size // 2
    smoothing *= diagonal_weight
    smoothing[center_index] *= axial_weight / (2.0 * diagonal_weight) if diagonal_weight > 0 else axial_weight
    derivative_base = np.array([math.comb(kernel_size - 3, index) for index in range(kernel_size - 2)], dtype=float)
    derivative = np.convolve(np.array((1.0, 0.0, -1.0)), derivative_base)
    kernel_x = np.outer(smoothing, derivative)
    return kernel_x, kernel_x.T


def non_maximum_suppression(magnitude: np.ndarray, gradient_x: np.ndarray, gradient_y: np.ndarray) -> np.ndarray:
    """Keep only local gradient maxima along the quantized gradient direction."""
    angle = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0
    padded = np.pad(magnitude, 1, mode="constant")
    output = np.zeros_like(magnitude)
    for row in range(magnitude.shape[0]):
        for column in range(magnitude.shape[1]):
            value = magnitude[row, column]
            direction = angle[row, column]
            if direction < 22.5 or direction >= 157.5:
                before, after = padded[row + 1, column], padded[row + 1, column + 2]
            elif direction < 67.5:
                before, after = padded[row, column + 2], padded[row + 2, column]
            elif direction < 112.5:
                before, after = padded[row, column + 1], padded[row + 2, column + 1]
            else:
                before, after = padded[row, column], padded[row + 2, column + 2]
            if value >= before and value >= after:
                output[row, column] = value
    return output


def vertically_supported_edge_mask(
    edge_mask: np.ndarray,
    window_px: int = VERTICAL_EDGE_SUPPORT_WINDOW_PX,
    min_points: int = VERTICAL_EDGE_SUPPORT_MIN_POINTS,
) -> np.ndarray:
    """Keep existing edge pixels with sustained support along a near-vertical path."""
    if window_px < 1 or window_px % 2 == 0:
        raise ValueError("竖向边缘支持窗口必须是正奇数。")
    if min_points < 1:
        raise ValueError("竖向边缘支持点数必须为正数。")
    horizontal_radius = 1
    padded_columns = np.pad(edge_mask.astype(np.int16), ((0, 0), (horizontal_radius, horizontal_radius)))
    local_columns = sum(
        padded_columns[:, offset : offset + edge_mask.shape[1]]
        for offset in range(horizontal_radius * 2 + 1)
    )
    vertical_radius = window_px // 2
    padded_rows = np.pad(local_columns, ((vertical_radius, vertical_radius), (0, 0)))
    cumulative = np.vstack((np.zeros((1, edge_mask.shape[1]), dtype=np.int32), np.cumsum(padded_rows, axis=0)))
    support = cumulative[window_px:] - cumulative[:-window_px]
    return edge_mask & (support >= min_points)


def robust_normalize_response(response: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale edge responses by their positive 99th percentile, limiting outlier influence."""
    positive = response[response > 0]
    if not len(positive):
        return np.zeros(response.shape, dtype=float), 0.0
    scale = float(np.percentile(positive, 99))
    return np.clip(response / max(scale, 1e-12), 0.0, 1.0), scale


def hysteresis_edges(response: np.ndarray, high_threshold: float, threshold_ratio: float) -> tuple[np.ndarray, float, float]:
    """Keep weak edges connected to strong edges after robust response normalization."""
    normalized, scale = robust_normalize_response(response)
    low_threshold = high_threshold / threshold_ratio
    strong = normalized >= high_threshold
    weak = normalized >= low_threshold
    accepted = strong.copy()
    stack = list(zip(*np.nonzero(strong)))
    while stack:
        row, column = stack.pop()
        for next_row in range(max(0, row - 1), min(response.shape[0], row + 2)):
            for next_column in range(max(0, column - 1), min(response.shape[1], column + 2)):
                if weak[next_row, next_column] and not accepted[next_row, next_column]:
                    accepted[next_row, next_column] = True
                    stack.append((next_row, next_column))
    return accepted, low_threshold, scale


def edge_response(
    image: np.ndarray,
    detector: str,
    diagonal_weight: float,
    axial_weight: float,
    kernel_size: int,
    canny_high_threshold: float,
    canny_threshold_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build candidate and subpixel-refinement responses for a selected edge detector."""
    if detector == EDGE_DETECTOR_LAPLACIAN:
        center = -4.0 * axial_weight - 4.0 * diagonal_weight
        kernel = np.array(((diagonal_weight, axial_weight, diagonal_weight), (axial_weight, center, axial_weight), (diagonal_weight, axial_weight, diagonal_weight)))
        response = np.abs(convolve_kernel(image, kernel))
        return response, response, response > 0, float("nan"), float("nan")
    gradient_x, gradient_y = (
        convolve_kernel(image, kernel)
        for kernel in first_order_kernels(detector, diagonal_weight, axial_weight, kernel_size)
    )
    magnitude = np.hypot(gradient_x, gradient_y)
    suppressed = non_maximum_suppression(magnitude, gradient_x, gradient_y)
    if detector == EDGE_DETECTOR_CANNY:
        connected, low_threshold, normalization_scale = hysteresis_edges(suppressed, canny_high_threshold, canny_threshold_ratio)
        return suppressed * connected, magnitude, connected, normalization_scale, low_threshold
    return suppressed, magnitude, suppressed > 0, float("nan"), float("nan")


def detect_full_image_edges(
    image: np.ndarray,
    detector: str,
    diagonal_weight: float,
    axial_weight: float,
    canny_high_threshold: float = 0.125,
    canny_threshold_ratio: float = 2.5,
    require_vertical_continuity: bool = False,
    kernel_size: int = 3,
) -> np.ndarray:
    """Return edges for the supplied image region, with optional short-noise removal."""
    if detector == EDGE_DETECTOR_LAPLACIAN:
        raise ValueError("拉普拉斯使用单线 LER/LWR 分析，不提供单独的边缘叠加。")
    response, _refinement, candidate_mask, _scale, _low = edge_response(
        image,
        detector,
        diagonal_weight,
        axial_weight,
        kernel_size,
        canny_high_threshold,
        canny_threshold_ratio,
    )
    if detector == EDGE_DETECTOR_CANNY:
        edge_mask = candidate_mask & (response > 0)
    else:
        positive = response[response > 0]
        if not len(positive):
            return np.zeros(image.shape, dtype=bool)
        percentile = max(55, 75 - 5 * ((kernel_size - 3) // 2))
        threshold = max(float(np.percentile(positive, percentile)), float(np.max(positive)) * 0.12)
        edge_mask = response >= threshold
    if require_vertical_continuity:
        return vertically_supported_edge_mask(edge_mask)
    return edge_mask.astype(bool)


def locate_edges(
    image: np.ndarray,
    max_jump_px: float = 12.0,
    detector: str = EDGE_DETECTOR_SCHARR,
    diagonal_weight: float = 3.0,
    axial_weight: float = 10.0,
    canny_high_threshold: float = 0.125,
    canny_threshold_ratio: float = 2.5,
    kernel_size: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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

    candidates, refinement, candidate_mask, normalization_scale, low_threshold = edge_response(
        image,
        detector,
        diagonal_weight,
        axial_weight,
        kernel_size,
        canny_high_threshold,
        canny_threshold_ratio,
    )
    reference_profile = np.median(candidates, axis=0)
    left_reference = margin + int(np.argmax(reference_profile[margin:midpoint]))
    right_reference = midpoint + int(np.argmax(reference_profile[midpoint : width - margin]))
    search_radius = max(12, min(30, int(width * 0.05)))
    left = np.full(height, np.nan, dtype=float)
    right = np.full(height, np.nan, dtype=float)

    for row in range(height):
        left_start = max(1, left_reference - search_radius)
        left_end = min(width - 1, left_reference + search_radius + 1)
        right_start = max(1, right_reference - search_radius)
        right_end = min(width - 1, right_reference + search_radius + 1)
        left_mask = candidate_mask[row, left_start:left_end]
        right_mask = candidate_mask[row, right_start:right_end]
        if left_mask.any():
            left_values = np.where(left_mask, candidates[row, left_start:left_end], -np.inf)
            left_index = left_start + int(np.argmax(left_values))
            left[row] = parabola_peak(refinement[row], left_index)
        if right_mask.any():
            right_values = np.where(right_mask, candidates[row, right_start:right_end], -np.inf)
            right_index = right_start + int(np.argmax(right_values))
            right[row] = parabola_peak(refinement[row], right_index)

    left, left_valid = replace_outliers(left, max_jump_px)
    right, right_valid = replace_outliers(right, max_jump_px)
    valid = left_valid & right_valid & (right > left)
    return left, right, valid, normalization_scale, low_threshold


def analyze_roi(
    image: np.ndarray,
    pixel_size_nm: float,
    max_jump_px: float = 12.0,
    detector: str = EDGE_DETECTOR_SCHARR,
    diagonal_weight: float = 3.0,
    axial_weight: float = 10.0,
    canny_high_threshold: float = 0.125,
    canny_threshold_ratio: float = 2.5,
    kernel_size: int = 3,
) -> AnalysisResult:
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

    left, right, valid, canny_normalization_scale, canny_low_threshold = locate_edges(
        image,
        max_jump_px,
        detector,
        diagonal_weight,
        axial_weight,
        canny_high_threshold,
        canny_threshold_ratio,
        kernel_size,
    )
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
        edge_detector=detector,
        edge_kernel_size=kernel_size,
        edge_diagonal_weight=diagonal_weight,
        edge_axial_weight=axial_weight,
        canny_high_threshold=canny_high_threshold,
        canny_threshold_ratio=canny_threshold_ratio,
        canny_low_threshold=canny_low_threshold,
        canny_normalization_scale=canny_normalization_scale,
    )


def era_polarity_sign(image: np.ndarray, polarity: str) -> tuple[int, str]:
    """Resolve the expected intensity transition of one manually selected edge ROI."""
    if polarity == ERA_POLARITY_RISING:
        return 1, polarity
    if polarity == ERA_POLARITY_FALLING:
        return -1, polarity
    profile_difference = np.diff(np.median(image.astype(float), axis=0))
    if not len(profile_difference) or np.max(np.abs(profile_difference)) <= 1e-12:
        raise ValueError("单边 ROI 的灰度变化不足，无法判断边缘方向。")
    sign = 1 if abs(float(np.max(profile_difference))) >= abs(float(np.min(profile_difference))) else -1
    return sign, ERA_POLARITY_RISING if sign > 0 else ERA_POLARITY_FALLING


def era_edge_position(
    profile: np.ndarray,
    reference_x: float,
    polarity_sign: int,
    polynomial_degree: int,
) -> float:
    """Fit one scanline transition and return its normalized 0.5 crossing in pixels."""
    width = len(profile)
    difference = polarity_sign * np.diff(profile.astype(float))
    half_search = max(5, min(30, width // 3))
    start = max(0, int(math.floor(reference_x - half_search)))
    end = min(width - 1, int(math.ceil(reference_x + half_search)))
    local_difference = np.maximum(difference[start:end], 0.0)
    if not len(local_difference) or float(np.max(local_difference)) <= 1e-12:
        return float("nan")
    normalized_difference = local_difference / float(np.max(local_difference))
    weights = normalized_difference**ERA_DIFFERENCE_POWER
    x_centers = np.arange(start, end, dtype=float) + 0.5
    weighted_center = float(np.dot(weights, x_centers) / np.sum(weights))
    peak_center = float(start + int(np.argmax(local_difference)) + 0.5)
    half_fit = max(3, min(15, int(math.ceil(abs(peak_center - weighted_center))) + 3))
    fit_start = max(0, int(math.floor(weighted_center - half_fit)))
    fit_end = min(width, int(math.ceil(weighted_center + half_fit)) + 1)
    x_values = np.arange(fit_start, fit_end, dtype=float)
    values = profile[fit_start:fit_end].astype(float)
    if len(x_values) <= polynomial_degree or float(np.ptp(values)) <= 1e-12:
        return float("nan")
    normalized_values = (values - float(np.min(values))) / float(np.ptp(values))
    coefficients = np.polyfit(x_values, normalized_values, polynomial_degree)
    roots = np.roots(np.array([*coefficients[:-1], coefficients[-1] - 0.5], dtype=float))
    candidates = []
    derivative = np.polyder(coefficients)
    for root in roots:
        if abs(float(root.imag)) > 1e-7:
            continue
        position = float(root.real)
        slope = float(np.polyval(derivative, position))
        if fit_start <= position <= fit_end - 1 and slope * polarity_sign > 0:
            candidates.append(position)
    if not candidates:
        return float("nan")
    return min(candidates, key=lambda position: abs(position - weighted_center))


def analyze_single_edge_era(
    image: np.ndarray,
    roi_bounds_px: tuple[int, int, int, int],
    pixel_size_nm: float,
    polynomial_degree: int = ERA_DEFAULT_POLYNOMIAL_DEGREE,
    polarity: str = ERA_POLARITY_AUTO,
    max_jump_px: float = 12.0,
) -> SingleEdgeResult:
    """Independently locate one SEM edge with an ERA-style profile fit.

    The input image is already the application's selected preprocessed image.  This
    method deliberately applies no additional spatial filter or image smoothing.
    """
    if pixel_size_nm <= 0:
        raise ValueError("像素尺寸必须大于 0 nm/pixel。")
    if polynomial_degree not in (2, 3, 4):
        raise ValueError("单边拟合仅支持 2、3 或 4 次多项式。")
    left, top, right, bottom = roi_bounds_px
    roi = image[top:bottom, left:right]
    height, width = roi.shape
    if width < MIN_ERA_ROI_WIDTH or height < MIN_ERA_ROI_HEIGHT:
        raise ValueError("单边 ROI 过小；请至少包含 80 个沿线方向像素和约 12 px 横向灰度过渡区域。")
    polarity_sign, resolved_polarity = era_polarity_sign(roi, polarity)
    reference_difference = polarity_sign * np.diff(np.median(roi.astype(float), axis=0))
    reference_x = float(np.argmax(reference_difference) + 0.5)
    trace = np.array(
        [era_edge_position(roi[row], reference_x, polarity_sign, polynomial_degree) for row in range(height)],
        dtype=float,
    )
    repaired_trace, valid = replace_outliers(trace, max_jump_px)
    if int(valid.sum()) < max(30, int(height * 0.6)):
        raise ValueError("单边有效拟合点太少；请让 ROI 只覆盖一条边，并包含边缘两侧的灰度过渡。")
    valid_rows_local = np.arange(height, dtype=float)[valid]
    valid_trace_local = repaired_trace[valid]
    baseline = np.polyval(np.polyfit(valid_rows_local, valid_trace_local, 1), valid_rows_local)
    residual_nm = (valid_trace_local - baseline) * pixel_size_nm
    ler_sigma_nm = math.sqrt(float(np.dot(residual_nm, residual_nm)) / (len(valid_rows_local) - 2))
    return SingleEdgeResult(
        rows_px=valid_rows_local + top,
        edge_px=valid_trace_local + left,
        residual_nm=residual_nm,
        ler_sigma_nm=ler_sigma_nm,
        rejected_rows=int((~valid).sum()),
        roi_bounds_px=roi_bounds_px,
        polarity=resolved_polarity,
        polynomial_degree=polynomial_degree,
    )


def pair_single_edges(left_edge: SingleEdgeResult, right_edge: SingleEdgeResult, pixel_size_nm: float) -> AnalysisResult:
    """Pair only user-supplied left/right traces on their common global scanlines."""
    left_values = {int(row): value for row, value in zip(left_edge.rows_px, left_edge.edge_px)}
    right_values = {int(row): value for row, value in zip(right_edge.rows_px, right_edge.edge_px)}
    rows = np.array(sorted(set(left_values).intersection(right_values)), dtype=float)
    left = np.array([left_values[int(row)] for row in rows], dtype=float)
    right = np.array([right_values[int(row)] for row in rows], dtype=float)
    valid = right > left
    rows, left, right = rows[valid], left[valid], right[valid]
    if len(rows) < 30:
        raise ValueError("左右单边 ROI 的共同有效行不足；请检查两边是否属于同一根线。")
    left_fit = np.polyval(np.polyfit(rows, left, 1), rows)
    right_fit = np.polyval(np.polyfit(rows, right, 1), rows)
    left_residual_nm = (left - left_fit) * pixel_size_nm
    right_residual_nm = (right - right_fit) * pixel_size_nm
    width_nm = (right - left) * pixel_size_nm
    width_residual_nm = width_nm - np.mean(width_nm)
    count = len(rows)
    left_sigma = math.sqrt(float(np.dot(left_residual_nm, left_residual_nm)) / (count - 2))
    right_sigma = math.sqrt(float(np.dot(right_residual_nm, right_residual_nm)) / (count - 2))
    lwr_sigma = math.sqrt(float(np.dot(width_residual_nm, width_residual_nm)) / (count - 1))
    correlation = float(np.corrcoef(left_residual_nm, right_residual_nm)[0, 1]) if left_sigma > 0 and right_sigma > 0 else float("nan")
    return AnalysisResult(
        rows_px=rows,
        left_px=left,
        right_px=right,
        left_residual_nm=left_residual_nm,
        right_residual_nm=right_residual_nm,
        width_nm=width_nm,
        width_residual_nm=width_residual_nm,
        mean_width_nm=float(np.mean(width_nm)),
        ler_left_sigma_nm=left_sigma,
        ler_right_sigma_nm=right_sigma,
        lwr_sigma_nm=lwr_sigma,
        total_ler_sigma_nm=math.hypot(left_sigma, right_sigma),
        edge_correlation=correlation,
        pixel_size_nm=pixel_size_nm,
        rejected_rows=left_edge.rejected_rows + right_edge.rejected_rows + int((~valid).sum()),
        edge_detector=EDGE_DETECTOR_ERA,
        edge_kernel_size=0,
        edge_diagonal_weight=float("nan"),
        edge_axial_weight=float("nan"),
        canny_high_threshold=float("nan"),
        canny_threshold_ratio=float("nan"),
        canny_low_threshold=float("nan"),
        canny_normalization_scale=float("nan"),
    )


def aggregate_era_line_results(results: list[AnalysisResult]) -> EraAggregateResult:
    """Pool independently detrended line residuals without averaging sigmas."""
    if not results:
        raise ValueError("请至少完成一根线的左右单边 ROI。")
    left_rss = sum(float(np.dot(result.left_residual_nm, result.left_residual_nm)) for result in results)
    right_rss = sum(float(np.dot(result.right_residual_nm, result.right_residual_nm)) for result in results)
    width_rss = sum(float(np.dot(result.width_residual_nm, result.width_residual_nm)) for result in results)
    left_dof = sum(len(result.rows_px) - 2 for result in results)
    right_dof = sum(len(result.rows_px) - 2 for result in results)
    width_dof = sum(len(result.rows_px) - 1 for result in results)
    left_sigma = math.sqrt(left_rss / left_dof)
    right_sigma = math.sqrt(right_rss / right_dof)
    lwr_sigma = math.sqrt(width_rss / width_dof)
    left_residuals = np.concatenate([result.left_residual_nm for result in results])
    right_residuals = np.concatenate([result.right_residual_nm for result in results])
    correlation = float(np.corrcoef(left_residuals, right_residuals)[0, 1]) if left_sigma > 0 and right_sigma > 0 else float("nan")
    return EraAggregateResult(
        line_count=len(results),
        point_count=sum(len(result.rows_px) for result in results),
        mean_width_nm=float(np.mean([result.mean_width_nm for result in results])),
        ler_left_sigma_nm=left_sigma,
        ler_right_sigma_nm=right_sigma,
        lwr_sigma_nm=lwr_sigma,
        total_ler_sigma_nm=math.hypot(left_sigma, right_sigma),
        edge_correlation=correlation,
    )


def connected_pixel_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    """Return binary connected components without adding a SciPy dependency."""
    if connectivity not in (4, 8):
        raise ValueError("连通性只能为 4 或 8。")
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    offsets = ((-1, 0), (0, -1), (0, 1), (1, 0)) if connectivity == 4 else (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        points = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for offset_y, offset_x in offsets:
                neighbor_y, neighbor_x = y + offset_y, x + offset_x
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and mask[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))
        components.append(np.asarray(points, dtype=float))
    return components


def lattice_support_score(dots: list[tuple[float, float, float, float, float, float, float]]) -> float:
    """Score whether candidate centers form a repeated point lattice rather than edge noise."""
    if len(dots) < 4:
        return 0.0
    centers = np.asarray(dots, dtype=float)[:, :2]
    nearest = []
    for index, center in enumerate(centers):
        distances = np.hypot(*(centers - center).T)
        distances[index] = np.inf
        nearest.append(float(np.min(distances)))
    spacing = float(np.median(nearest))
    if spacing <= 0:
        return 0.0
    support = 0
    for index, center in enumerate(centers):
        distances = np.hypot(*(centers - center).T)
        support += int(np.count_nonzero((distances >= spacing * 0.55) & (distances <= spacing * 1.6)) >= 2)
    return support / (1.0 + float(np.std(nearest) / spacing))


def reference_bcp_polarity(image: np.ndarray, references: list[tuple[float, float, float]]) -> float:
    """Infer whether manually marked BCP dots appear bright or dark."""
    contrasts = []
    for center_x, center_y, radius in references:
        top, bottom = max(0, int(center_y - radius)), min(image.shape[0], int(center_y + radius + 1))
        left, right = max(0, int(center_x - radius)), min(image.shape[1], int(center_x + radius + 1))
        yy, xx = np.indices((bottom - top, right - left), dtype=float)
        distances = np.hypot(xx + left - center_x, yy + top - center_y)
        core = image[top:bottom, left:right][distances <= radius * 0.35]
        ring = image[top:bottom, left:right][(distances >= radius * 0.65) & (distances <= radius)]
        if len(core) and len(ring):
            contrasts.append(float(np.mean(core, dtype=float) - np.mean(ring, dtype=float)))
    return 1.0 if not contrasts or np.median(contrasts) >= 0 else -1.0


def bcp_response(
    image: np.ndarray,
    polarity: float,
    diameter_px: float,
    use_dog_contrast: bool = True,
) -> np.ndarray:
    """Return either the optional DoG contrast response or the original intensity."""
    if not use_dog_contrast:
        return polarity * image.astype(float)
    small_radius = float(np.clip(diameter_px * 0.08, 1.0, 3.0))
    large_radius = float(np.clip(diameter_px * 0.65, 4.0, max(12, min(image.shape) / 50)))
    small = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=small_radius)), dtype=float)
    large = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=large_radius)), dtype=float)
    return polarity * (small - large)


def bcp_reference_diameter(references: list[tuple[float, float, float]], expected_diameter_px: float | None) -> float:
    """Choose the explicit expected diameter, otherwise the median manual-circle diameter."""
    if expected_diameter_px is not None:
        return expected_diameter_px
    if references:
        return float(np.median([radius * 2 for _x, _y, radius in references]))
    raise ValueError("自动识别需要输入圆柱大致直径，或至少标示 3 个样本圆柱。")


def bcp_minimum_component_area(
    references: list[tuple[float, float, float]],
    area_fraction: float,
    diameter_px: float,
) -> float:
    """Convert the user-selected sample-area fraction to a pixel-area threshold."""
    if area_fraction <= 0:
        return 0.0
    return float(area_fraction * bcp_reference_area(references, diameter_px))


def bcp_reference_area(references: list[tuple[float, float, float]], diameter_px: float) -> float:
    """Return the manual-sample or expected-cylinder reference area in pixels."""
    sample_areas = [math.pi * radius**2 for _x, _y, radius in references]
    return float(np.median(sample_areas)) if sample_areas else math.pi * (diameter_px / 2) ** 2


def fill_bcp_internal_holes(mask: np.ndarray, maximum_hole_area_px: float) -> np.ndarray:
    """Fill small background islands enclosed by a foreground BCP region."""
    if maximum_hole_area_px <= 0:
        return mask
    filled = mask.copy()
    height, width = mask.shape
    for hole in connected_pixel_components(~mask, connectivity=4):
        rows, columns = hole.astype(int).T
        touches_image_edge = (
            np.any(rows == 0) or np.any(rows == height - 1)
            or np.any(columns == 0) or np.any(columns == width - 1)
        )
        if not touches_image_edge and len(hole) <= maximum_hole_area_px:
            filled[rows, columns] = True
    return filled


def retain_bcp_components(mask: np.ndarray, minimum_area_px: float) -> np.ndarray:
    """Keep only 8-connected binary foreground components meeting the area limit."""
    if minimum_area_px <= 0:
        return mask
    retained = np.zeros_like(mask, dtype=bool)
    for component in connected_pixel_components(mask):
        if len(component) >= minimum_area_px:
            rows, columns = component.astype(int).T
            retained[rows, columns] = True
    return retained


def adaptive_bcp_foreground(
    response: np.ndarray,
    horizontal_sections: int,
    vertical_sections: int,
) -> tuple[np.ndarray, float]:
    """Threshold every user-defined two-dimensional image section independently."""
    section_width = max(1, math.ceil(response.shape[1] / horizontal_sections))
    section_height = max(1, math.ceil(response.shape[0] / vertical_sections))
    foreground = np.zeros(response.shape, dtype=bool)
    thresholds = []
    for top in range(0, response.shape[0], section_height):
        for left in range(0, response.shape[1], section_width):
            section = response[top : top + section_height, left : left + section_width]
            baseline = float(np.median(section))
            high = float(np.percentile(section, 95))
            threshold = baseline + 0.28 * max(high - baseline, 0.0)
            foreground[top : top + section.shape[0], left : left + section.shape[1]] = section >= threshold
            thresholds.append(threshold)
    return foreground, float(np.median(thresholds))


def global_bcp_foreground(response: np.ndarray) -> tuple[np.ndarray, float]:
    """Build one whole-image foreground mask when local splitting is disabled."""
    baseline = float(np.median(response))
    high = float(np.percentile(response, 95))
    threshold = baseline + 0.28 * max(high - baseline, 0.0)
    return response >= threshold, threshold


def bcp_foreground(
    response: np.ndarray,
    horizontal_sections: int,
    vertical_sections: int,
    use_local_segmentation: bool,
) -> tuple[np.ndarray, float]:
    """Choose the optional local or global binary segmentation path."""
    if use_local_segmentation:
        return adaptive_bcp_foreground(response, horizontal_sections, vertical_sections)
    return global_bcp_foreground(response)


def fit_bcp_component(component: np.ndarray) -> tuple[float, float, float, float, float, float, float] | None:
    """Fit binary connected-component geometry without using internal intensity peaks."""
    if len(component) < 6:
        return None
    y, x = component[:, 0], component[:, 1]
    center_x, center_y = float(np.mean(x)), float(np.mean(y))
    dx, dy = x - center_x, y - center_y
    covariance = np.array([
        [float(np.mean(dx * dx)), float(np.mean(dx * dy))],
        [float(np.mean(dx * dy)), float(np.mean(dy * dy))],
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minor_axis, major_axis = 4.0 * np.sqrt(np.maximum(eigenvalues, 0.0))
    if minor_axis < 2.0 or major_axis / minor_axis > 3.0:
        return None
    angle = math.degrees(math.atan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    area = float(len(component))
    return center_x, center_y, 2.0 * math.sqrt(area / math.pi), major_axis, minor_axis, angle, area


def component_contour_segments(component: np.ndarray) -> np.ndarray:
    """Trace the pixel-edge contour of one binary connected component."""
    pixels = {tuple(point.astype(int)) for point in component}
    segments = []
    for y, x in pixels:
        if (y - 1, x) not in pixels:
            segments.append((x - 0.5, y - 0.5, x + 0.5, y - 0.5))
        if (y + 1, x) not in pixels:
            segments.append((x - 0.5, y + 0.5, x + 0.5, y + 0.5))
        if (y, x - 1) not in pixels:
            segments.append((x - 0.5, y - 0.5, x - 0.5, y + 0.5))
        if (y, x + 1) not in pixels:
            segments.append((x + 0.5, y - 0.5, x + 0.5, y + 0.5))
    return np.asarray(segments, dtype=float).reshape(-1, 4)


def bcp_dots_from_foreground(
    foreground: np.ndarray,
    diameter_px: float,
    minimum_area_px: float,
) -> tuple[list[tuple[float, float, float, float, float, float, float]], list[np.ndarray], list[np.ndarray]]:
    """Fit every eligible connected foreground component and trace its contour."""
    lower, upper = diameter_px * 0.4, diameter_px * 1.8
    dots, contours, components = [], [], []
    for component in connected_pixel_components(foreground):
        dot = fit_bcp_component(component)
        if dot is not None and minimum_area_px <= dot[6] and lower <= dot[2] <= upper:
            dots.append(dot)
            contours.append(component_contour_segments(component))
            components.append(component)
    return dots, contours, components


def connected_component_bcp_dots(
    response: np.ndarray,
    diameter_px: float,
    minimum_area_px: float,
    maximum_hole_area_px: float,
    local_horizontal_sections: int,
    local_vertical_sections: int,
    use_local_segmentation: bool,
) -> tuple[list[tuple[float, float, float, float, float, float, float]], list[np.ndarray], list[np.ndarray]]:
    """Identify one cylinder per binary connected region and trace its contour."""
    foreground, _threshold = bcp_foreground(
        response,
        local_horizontal_sections,
        local_vertical_sections,
        use_local_segmentation,
    )
    foreground = fill_bcp_internal_holes(foreground, maximum_hole_area_px)
    foreground = retain_bcp_components(foreground, minimum_area_px)
    return bcp_dots_from_foreground(foreground, diameter_px, minimum_area_px)


def bcp_candidate_contrast_score(
    image: np.ndarray,
    dots: list[tuple[float, float, float, float, float, float, float]],
    polarity: float,
    diameter_px: float,
) -> float:
    """Score whether candidate centers have the requested bright or dark core contrast."""
    radius = diameter_px / 2
    contrasts = []
    for center_x, center_y, *_values in dots:
        top, bottom = max(0, int(center_y - radius)), min(image.shape[0], int(center_y + radius + 1))
        left, right = max(0, int(center_x - radius)), min(image.shape[1], int(center_x + radius + 1))
        yy, xx = np.indices((bottom - top, right - left), dtype=float)
        distances = np.hypot(xx + left - center_x, yy + top - center_y)
        core = image[top:bottom, left:right][distances <= radius * 0.35]
        ring = image[top:bottom, left:right][(distances >= radius * 0.65) & (distances <= radius)]
        if len(core) and len(ring):
            contrasts.append(polarity * float(np.mean(core) - np.mean(ring)))
    return max(0.0, float(np.median(contrasts))) if contrasts else 0.0


def automatic_bcp_polarity(
    image: np.ndarray,
    diameter_px: float,
    minimum_area_px: float,
    maximum_hole_area_px: float,
    local_horizontal_sections: int,
    local_vertical_sections: int,
    use_local_segmentation: bool,
    use_dog_contrast: bool = True,
) -> float:
    """Choose polarity from the selected binary-region and contour-detection result."""
    scores = {}
    for polarity in (1.0, -1.0):
        response = bcp_response(image, polarity, diameter_px, use_dog_contrast)
        dots, _contours, _components = connected_component_bcp_dots(
            response,
            diameter_px,
            minimum_area_px,
            maximum_hole_area_px,
            local_horizontal_sections,
            local_vertical_sections,
            use_local_segmentation,
        )
        scores[polarity] = bcp_candidate_contrast_score(image, dots, polarity, diameter_px) * lattice_support_score(dots)
    return max(scores, key=scores.get)


def connected_region_bcp_dots(
    image: np.ndarray,
    references: list[tuple[float, float, float]],
    expected_diameter_px: float | None,
    min_area_fraction: float,
    internal_hole_fraction: float,
    local_horizontal_sections: int,
    local_vertical_sections: int,
    use_local_segmentation: bool,
    use_dog_contrast: bool = True,
) -> tuple[list[tuple[float, float, float, float, float, float, float]], list[np.ndarray], list[np.ndarray]]:
    """Detect BCP dots from thresholded connected regions and their contours."""
    diameter = bcp_reference_diameter(references, expected_diameter_px)
    minimum_area_px = bcp_minimum_component_area(references, min_area_fraction, diameter)
    maximum_hole_area_px = internal_hole_fraction * bcp_reference_area(references, diameter)
    polarity = reference_bcp_polarity(image, references) if references else automatic_bcp_polarity(
        image,
        diameter,
        minimum_area_px,
        maximum_hole_area_px,
        local_horizontal_sections,
        local_vertical_sections,
        use_local_segmentation,
        use_dog_contrast,
    )
    response = bcp_response(image, polarity, diameter, use_dog_contrast)
    return connected_component_bcp_dots(
        response,
        diameter,
        minimum_area_px,
        maximum_hole_area_px,
        local_horizontal_sections,
        local_vertical_sections,
        use_local_segmentation,
    )


def polygon_pixel_mask(shape: tuple[int, int], polygon_px: list[tuple[float, float]]) -> np.ndarray:
    """Rasterize an image-coordinate polygon into a boolean inclusion mask."""
    mask = Image.new("1", (shape[1], shape[0]), 0)
    ImageDraw.Draw(mask).polygon([(round(x), round(y)) for x, y in polygon_px], fill=1)
    return np.asarray(mask, dtype=bool)


def line_pixel_mask(
    shape: tuple[int, int],
    start_px: tuple[float, float],
    end_px: tuple[float, float],
    width_px: int = MANUAL_BCP_SPLIT_LINE_WIDTH_PX,
) -> np.ndarray:
    """Rasterize a user-drawn cut line into a foreground-pixel removal mask."""
    mask = Image.new("1", (shape[1], shape[0]), 0)
    ImageDraw.Draw(mask).line([start_px, end_px], fill=1, width=width_px)
    return np.asarray(mask, dtype=bool)


def polygon_bcp_foreground(response: np.ndarray, selection_mask: np.ndarray) -> np.ndarray:
    """Threshold the response using only pixels inside a manually drawn polygon."""
    values = response[selection_mask]
    if not len(values):
        raise ValueError("局部补漏区域没有有效像素。")
    baseline = float(np.median(values))
    high = float(np.percentile(values, 99))
    threshold = baseline + 0.28 * max(high - baseline, 0.0)
    return (response >= threshold) & selection_mask


def component_intersects_mask(component: np.ndarray, mask: np.ndarray) -> bool:
    """Return whether a component has any actual foreground pixel inside a mask."""
    rows, columns = component.astype(int).T
    return bool(np.any(mask[rows, columns]))


def components_share_pixels(first: np.ndarray, second: np.ndarray) -> bool:
    """Test overlap from component pixels, never from fitted centers or peak locations."""
    smaller, larger = (first, second) if len(first) <= len(second) else (second, first)
    larger_pixels = {tuple(point.astype(int)) for point in larger}
    return any(tuple(point.astype(int)) in larger_pixels for point in smaller)


def local_bcp_completion_candidates(
    image: np.ndarray,
    selection_mask: np.ndarray,
    references: list[tuple[float, float, float]],
    expected_diameter_px: float | None,
    min_area_fraction: float,
    internal_hole_fraction: float,
    use_dog_contrast: bool,
) -> tuple[list[tuple[float, float, float, float, float, float, float]], list[np.ndarray], list[np.ndarray]]:
    """Find contour candidates from one manually selected local threshold region."""
    diameter = bcp_reference_diameter(references, expected_diameter_px)
    minimum_area_px = bcp_minimum_component_area(references, min_area_fraction, diameter)
    maximum_hole_area_px = internal_hole_fraction * bcp_reference_area(references, diameter)
    polarity = reference_bcp_polarity(image, references) if references else automatic_bcp_polarity(
        image, diameter, minimum_area_px, maximum_hole_area_px, 1, 1, False, use_dog_contrast
    )
    foreground = polygon_bcp_foreground(bcp_response(image, polarity, diameter, use_dog_contrast), selection_mask)
    foreground = fill_bcp_internal_holes(foreground, maximum_hole_area_px)
    foreground = retain_bcp_components(foreground, minimum_area_px)
    return bcp_dots_from_foreground(foreground, diameter, minimum_area_px)


def bcp_result_dimension_scale(result: BCPAnalysisResult) -> float:
    """Recover the existing result's manual-reference calibration scale."""
    if not result.components_px:
        return 1.0
    raw_diameters = np.asarray([2.0 * math.sqrt(len(component) / math.pi) for component in result.components_px])
    valid = raw_diameters > 0
    return float(np.median(result.equivalent_diameters_px[valid] / raw_diameters[valid])) if np.any(valid) else 1.0


def bcp_layout_diameter_px(result: BCPAnalysisResult) -> float:
    """Return the measured mean CD as the displayed virtual-cylinder diameter."""
    return float(np.mean(result.cd_means_px))


def split_bcp_component(
    result: BCPAnalysisResult,
    cut_mask: np.ndarray,
    minimum_area_px: float,
) -> tuple[BCPAnalysisResult, int]:
    """Replace one manually cut connected region with its valid disconnected pieces."""
    touched = [index for index, component in enumerate(result.components_px) if component_intersects_mask(component, cut_mask)]
    if len(touched) != 1:
        raise ValueError("分割线应只穿过一个粘连轮廓。")
    component_index = touched[0]
    component = result.components_px[component_index]
    rows, columns = component.astype(int).T
    remaining = np.zeros(cut_mask.shape, dtype=bool)
    keep = ~cut_mask[rows, columns]
    remaining[rows[keep], columns[keep]] = True
    pieces = [piece for piece in connected_pixel_components(remaining) if len(piece) >= minimum_area_px]
    dots = [fit_bcp_component(piece) for piece in pieces]
    valid = [(dot, piece) for dot, piece in zip(dots, pieces) if dot is not None]
    if len(valid) < 2:
        raise ValueError("分割后没有得到至少两个有效区域；请让分割线完整穿过两柱的粘连处。")
    split_dots, split_components = zip(*valid)
    return replace_bcp_component(result, component_index, list(split_dots), list(split_components))


def replace_bcp_component(
    result: BCPAnalysisResult,
    component_index: int,
    split_dots: list[tuple[float, float, float, float, float, float, float]],
    split_components: list[np.ndarray],
) -> tuple[BCPAnalysisResult, int]:
    """Rebuild result geometry after replacing one component with manual split pieces."""
    keep = np.arange(len(result.components_px)) != component_index
    scale = bcp_result_dimension_scale(result)
    dots = np.asarray(split_dots, dtype=float)
    dots[:, 2:5] *= scale
    dots[:, 6] *= scale**2
    directional_cds = np.asarray([component_directional_cds(component) for component in split_components]) * scale
    centers = np.vstack((result.centers_px[keep], dots[:, :2]))
    _orientations, spacing, _neighbors = local_hexagonal_orientations(centers)
    updated = BCPAnalysisResult(
        centers_px=centers,
        equivalent_diameters_px=np.concatenate((result.equivalent_diameters_px[keep], dots[:, 2])),
        major_axes_px=np.concatenate((result.major_axes_px[keep], dots[:, 3])),
        minor_axes_px=np.concatenate((result.minor_axes_px[keep], dots[:, 4])),
        angles_degrees=np.concatenate((result.angles_degrees[keep], dots[:, 5])),
        areas_px=np.concatenate((result.areas_px[keep], dots[:, 6])),
        contour_segments_px=[contour for index, contour in enumerate(result.contour_segments_px) if keep[index]] + [component_contour_segments(component) for component in split_components],
        components_px=[component for index, component in enumerate(result.components_px) if keep[index]] + split_components,
        directional_cds_px=np.vstack((result.directional_cds_px[keep], directional_cds)),
        cd_means_px=np.concatenate((result.cd_means_px[keep], np.mean(directional_cds, axis=1))),
        triangulation_segments_px=result.triangulation_segments_px,
        pitch_values_by_direction_px=result.pitch_values_by_direction_px,
        boundary_segments_px=result.boundary_segments_px,
        boundary_dot_indices=result.boundary_dot_indices,
        lattice_spacing_px=spacing,
        pixel_size_nm=result.pixel_size_nm,
    )
    return updated, len(split_components)


def append_bcp_completion(
    result: BCPAnalysisResult,
    candidates: tuple[list[tuple[float, float, float, float, float, float, float]], list[np.ndarray], list[np.ndarray]],
    selection_mask: np.ndarray,
) -> tuple[BCPAnalysisResult, int]:
    """Add local components without moving, replacing, or re-deduplicating old ones."""
    dots, contours, components = candidates
    accepted = [
        index for index, component in enumerate(components)
        if component_intersects_mask(component, selection_mask)
        and not any(components_share_pixels(component, existing) for existing in result.components_px)
    ]
    if not accepted:
        return result, 0
    dot_array = np.asarray([dots[index] for index in accepted], dtype=float)
    scale = bcp_result_dimension_scale(result)
    dot_array[:, 2:5] *= scale
    dot_array[:, 6] *= scale**2
    directional_cds = np.asarray([component_directional_cds(components[index]) for index in accepted]) * scale
    centers = np.vstack((result.centers_px, dot_array[:, :2]))
    _orientations, spacing, _neighbors = local_hexagonal_orientations(centers)
    updated = BCPAnalysisResult(
        centers_px=centers,
        equivalent_diameters_px=np.concatenate((result.equivalent_diameters_px, dot_array[:, 2])),
        major_axes_px=np.concatenate((result.major_axes_px, dot_array[:, 3])),
        minor_axes_px=np.concatenate((result.minor_axes_px, dot_array[:, 4])),
        angles_degrees=np.concatenate((result.angles_degrees, dot_array[:, 5])),
        areas_px=np.concatenate((result.areas_px, dot_array[:, 6])),
        contour_segments_px=result.contour_segments_px + [contours[index] for index in accepted],
        components_px=result.components_px + [components[index] for index in accepted],
        directional_cds_px=np.vstack((result.directional_cds_px, directional_cds)),
        cd_means_px=np.concatenate((result.cd_means_px, np.mean(directional_cds, axis=1))),
        triangulation_segments_px=result.triangulation_segments_px,
        pitch_values_by_direction_px=result.pitch_values_by_direction_px,
        boundary_segments_px=result.boundary_segments_px,
        boundary_dot_indices=result.boundary_dot_indices,
        lattice_spacing_px=spacing,
        pixel_size_nm=result.pixel_size_nm,
    )
    return updated, len(accepted)


def local_hexagonal_orientations(centers_px: np.ndarray) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """Estimate local BCP lattice orientation modulo 60 degrees from nearby dots."""
    count = len(centers_px)
    nearest_distances = np.empty(count, dtype=float)
    for index, center in enumerate(centers_px):
        distances = np.hypot(*(centers_px - center).T)
        distances[index] = np.inf
        nearest_distances[index] = np.min(distances)
    spacing = float(np.median(nearest_distances))
    neighbor_radius = spacing * 1.45
    orientations = np.full(count, np.nan)
    neighbors: list[np.ndarray] = []
    for index, center in enumerate(centers_px):
        offsets = centers_px - center
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        nearby = np.flatnonzero((distances > 0) & (distances <= neighbor_radius))
        neighbors.append(nearby)
        if len(nearby) < 3:
            continue
        bond_angles = np.arctan2(offsets[nearby, 1], offsets[nearby, 0])
        order = np.mean(np.exp(6j * bond_angles))
        if abs(order) >= 0.45:
            orientations[index] = math.degrees(np.angle(order) / 6.0) % 60.0
    return orientations, spacing, neighbors


def hexagonal_angle_difference(first: float, second: float) -> float:
    """Return the smallest orientation difference for a sixfold-symmetric lattice."""
    return abs((first - second + 30.0) % 60.0 - 30.0)


def component_ray_origin(component: np.ndarray) -> tuple[float, float]:
    """Return the binary centroid, or the nearest foreground pixel when it lies outside."""
    center_x, center_y = float(np.mean(component[:, 1])), float(np.mean(component[:, 0]))
    pixels = {tuple(point.astype(int)) for point in component}
    if (int(math.floor(center_y + 0.5)), int(math.floor(center_x + 0.5))) in pixels:
        return center_x, center_y
    distances = (component[:, 1] - center_x) ** 2 + (component[:, 0] - center_y) ** 2
    row, column = component[int(np.argmin(distances))]
    return float(column), float(row)


def ray_exit_distance(component: np.ndarray, origin: tuple[float, float], angle_degrees: float) -> float:
    """Measure one binary-region radius by ray marching until the foreground ends."""
    pixels = {tuple(point.astype(int)) for point in component}
    angle = math.radians(angle_degrees)
    step_x, step_y = math.cos(angle) * CD_RAY_STEP_PX, math.sin(angle) * CD_RAY_STEP_PX
    max_distance = math.hypot(np.ptp(component[:, 1]), np.ptp(component[:, 0])) + 4.0
    distance = 0.0
    while distance <= max_distance:
        x = origin[0] + step_x * (distance / CD_RAY_STEP_PX)
        y = origin[1] + step_y * (distance / CD_RAY_STEP_PX)
        if (int(math.floor(y + 0.5)), int(math.floor(x + 0.5))) not in pixels:
            return max(0.0, distance - CD_RAY_STEP_PX / 2)
        distance += CD_RAY_STEP_PX
    return max_distance


def component_directional_cds(component: np.ndarray) -> np.ndarray:
    """Measure CD on the horizontal, +60°, and −60° axes through the region centroid."""
    origin = component_ray_origin(component)
    return np.asarray([
        ray_exit_distance(component, origin, angle) + ray_exit_distance(component, origin, angle + 180.0)
        for angle in (0.0, 60.0, 120.0)
    ])


def circumcircle(points: np.ndarray, triangle: tuple[int, int, int]) -> tuple[float, float, float] | None:
    """Return a triangle circumcircle as center x/y and squared radius."""
    first, second, third = points[list(triangle)]
    ax, ay = first
    bx, by = second
    cx, cy = third
    denominator = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denominator) < 1e-10:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    center_x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / denominator
    center_y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / denominator
    return center_x, center_y, (center_x - ax) ** 2 + (center_y - ay) ** 2


def delaunay_edge_indices(points: np.ndarray) -> np.ndarray:
    """Build Delaunay neighbor edges with Bowyer-Watson triangulation."""
    if len(points) < 3:
        return np.empty((0, 2), dtype=int)
    lower, upper = np.min(points, axis=0), np.max(points, axis=0)
    center, span = (lower + upper) / 2, max(float(np.max(upper - lower)), 1.0)
    supertriangle = np.asarray([
        (center[0] - 20 * span, center[1] - 10 * span),
        (center[0], center[1] + 20 * span),
        (center[0] + 20 * span, center[1] - 10 * span),
    ])
    all_points = np.vstack((points, supertriangle))
    triangles = [(len(points), len(points) + 1, len(points) + 2)]
    for point_index, point in enumerate(points):
        bad = []
        for triangle in triangles:
            circle = circumcircle(all_points, triangle)
            if circle is not None and (point[0] - circle[0]) ** 2 + (point[1] - circle[1]) ** 2 <= circle[2] * (1 + 1e-10):
                bad.append(triangle)
        boundary: dict[tuple[int, int], int] = {}
        for triangle in bad:
            for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                edge = tuple(sorted(edge))
                boundary[edge] = boundary.get(edge, 0) + 1
        triangles = [triangle for triangle in triangles if triangle not in bad]
        triangles.extend((first, second, point_index) for (first, second), count in boundary.items() if count == 1)
    edges = {
        tuple(sorted(edge))
        for triangle in triangles
        if all(index < len(points) for index in triangle)
        for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
    }
    return np.asarray(sorted(edges), dtype=int).reshape(-1, 2)


def nearest_center_indices(centers_px: np.ndarray, count: int) -> list[np.ndarray]:
    """Return each center's closest other-center indices."""
    neighbors = []
    for index, center in enumerate(centers_px):
        distances = np.hypot(*(centers_px - center).T)
        distances[index] = np.inf
        neighbors.append(np.argsort(distances)[:min(count, len(centers_px) - 1)])
    return neighbors


def local_pitch_estimates(centers_px: np.ndarray) -> np.ndarray:
    """Estimate local Pitch from each center's nearest lattice neighbors."""
    nearby = nearest_center_indices(centers_px, BCP_LOCAL_PITCH_NEIGHBORS)
    pitches = []
    for index, indices in enumerate(nearby):
        distances = np.hypot(*(centers_px[indices] - centers_px[index]).T)
        pitches.append(float(np.median(distances)) if len(distances) else float("nan"))
    return np.asarray(pitches, dtype=float)


def link_has_intermediate_center(
    centers_px: np.ndarray,
    first: int,
    second: int,
    local_pitch: float,
) -> bool:
    """Reject a link when another pillar center lies in its physical corridor."""
    start, end = centers_px[first], centers_px[second]
    vector = end - start
    squared_length = float(np.dot(vector, vector))
    if squared_length == 0:
        return True
    offsets = centers_px - start
    projection = offsets @ vector / squared_length
    perpendicular = np.abs(offsets[:, 0] * vector[1] - offsets[:, 1] * vector[0]) / math.sqrt(squared_length)
    between = (projection >= BCP_BLOCKING_PROJECTION_MARGIN) & (projection <= 1.0 - BCP_BLOCKING_PROJECTION_MARGIN)
    between[first] = False
    between[second] = False
    return bool(np.any(between & (perpendicular <= local_pitch * BCP_BLOCKING_CORRIDOR_FACTOR)))


def retain_local_bcp_edges(centers_px: np.ndarray, candidate_edges: np.ndarray) -> np.ndarray:
    """Keep only mutually local, unblocked Delaunay links for BCP metrology."""
    local_pitches = local_pitch_estimates(centers_px)
    mutual_neighbors = nearest_center_indices(centers_px, BCP_MUTUAL_NEIGHBORS)
    retained = []
    for first, second in candidate_edges:
        local_pitch = float(np.median((local_pitches[first], local_pitches[second])))
        length = float(np.hypot(*(centers_px[second] - centers_px[first])))
        if not np.isfinite(local_pitch) or length > local_pitch * BCP_MAX_LOCAL_LINK_FACTOR:
            continue
        if second not in mutual_neighbors[first] or first not in mutual_neighbors[second]:
            continue
        if link_has_intermediate_center(centers_px, int(first), int(second), local_pitch):
            continue
        retained.append((int(first), int(second)))
    return np.asarray(retained, dtype=int).reshape(-1, 2)


def triangulation_primary_axis_degrees(segments_px: np.ndarray) -> float:
    """Infer the first of the three sixfold axes from retained neighbor links."""
    if not len(segments_px):
        return 0.0
    offsets = segments_px[:, 2:4] - segments_px[:, :2]
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    order = np.mean(np.exp(6j * angles))
    return math.degrees(np.angle(order) / 6.0) % 60.0 if abs(order) > 1e-10 else 0.0


def filter_triangulation_display_segments(
    segments_px: np.ndarray,
    display_mode: str,
    primary_axis_degrees: float | None = None,
) -> np.ndarray:
    """Return all links or only one automatically aligned hexagonal direction."""
    if display_mode == TRIANGULATION_DISPLAY_ALL or not len(segments_px):
        return segments_px
    try:
        direction_index = TRIANGULATION_DISPLAY_MODES.index(display_mode) - 1
    except ValueError:
        return segments_px
    primary_axis = triangulation_primary_axis_degrees(segments_px) if primary_axis_degrees is None else primary_axis_degrees
    target_axis = (primary_axis + 60.0 * direction_index) % 180.0
    offsets = segments_px[:, 2:4] - segments_px[:, :2]
    angles = np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0])) % 180.0
    differences = np.abs((angles - target_axis + 90.0) % 180.0 - 90.0)
    return segments_px[differences <= PITCH_DIRECTION_TOLERANCE_DEGREES]


def pitch_values_by_direction(centers_px: np.ndarray, edges: np.ndarray) -> dict[str, np.ndarray]:
    """Group Delaunay edge lengths into the three hexagonal lattice directions."""
    grouped = {label: [] for label in PITCH_DIRECTION_LABELS}
    for first, second in edges:
        offset = centers_px[second] - centers_px[first]
        angle = math.degrees(math.atan2(offset[1], offset[0])) % 180.0
        differences = [abs((angle - reference + 90.0) % 180.0 - 90.0) for reference in PITCH_DIRECTION_ANGLES]
        direction = int(np.argmin(differences))
        if differences[direction] <= PITCH_DIRECTION_TOLERANCE_DEGREES:
            grouped[PITCH_DIRECTION_LABELS[direction]].append(float(np.hypot(*offset)))
    return {label: np.asarray(values, dtype=float) for label, values in grouped.items()}


def bcp_grain_labels(centers_px: np.ndarray) -> np.ndarray:
    """Group locally connected pillars whose sixfold orientations agree."""
    count = len(centers_px)
    labels = np.full(count, -1, dtype=int)
    if count < MIN_GRAIN_DOTS:
        return labels
    edges = retain_local_bcp_edges(centers_px, delaunay_edge_indices(centers_px))
    orientations, _spacing, _neighbors = local_hexagonal_orientations(centers_px)
    parents = np.arange(count)

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    for first, second in edges:
        if not (np.isfinite(orientations[first]) and np.isfinite(orientations[second])):
            continue
        if hexagonal_angle_difference(orientations[first], orientations[second]) < GRAIN_BOUNDARY_ORIENTATION_THRESHOLD_DEGREES:
            root_first, root_second = root(int(first)), root(int(second))
            if root_first != root_second:
                parents[root_second] = root_first
    groups: dict[int, list[int]] = {}
    for index in np.flatnonzero(np.isfinite(orientations)):
        groups.setdefault(root(int(index)), []).append(int(index))
    for label, members in enumerate(group for group in groups.values() if len(group) >= MIN_GRAIN_DOTS):
        labels[members] = label
    return labels


def bcp_grain_overlay_image(
    centers_px: np.ndarray,
    grain_labels: np.ndarray,
    width: int,
    height: int,
    display_size: tuple[int, int],
) -> Image.Image | None:
    """Create a transparent nearest-pillar tessellation colored by grain label."""
    valid = grain_labels >= 0
    if not np.any(valid):
        return None
    stride = max(1, math.ceil(max(width, height) / GRAIN_OVERLAY_MAX_SIDE_PX))
    x_coordinates = np.arange(math.ceil(width / stride)) * stride + stride / 2
    y_coordinates = np.arange(math.ceil(height / stride)) * stride + stride / 2
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    closest_distance = np.full(grid_x.shape, np.inf)
    closest_label = np.full(grid_x.shape, -1, dtype=int)
    for (center_x, center_y), label in zip(centers_px, grain_labels):
        distance = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
        replace = distance < closest_distance
        closest_distance[replace] = distance[replace]
        closest_label[replace] = label
    rgba = np.zeros((*closest_label.shape, 4), dtype=np.uint8)
    mask = closest_label >= 0
    rgba[mask, :3] = GRAIN_OVERLAY_COLORS[closest_label[mask] % len(GRAIN_OVERLAY_COLORS)]
    rgba[mask, 3] = GRAIN_OVERLAY_ALPHA
    return Image.fromarray(rgba, "RGBA").resize(display_size, Image.Resampling.NEAREST)


def central_bcp_centroid(centers_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return the detected centroid closest to the image center."""
    image_center = np.array((width / 2, height / 2), dtype=float)
    return centers_px[np.argmin(np.sum((centers_px - image_center) ** 2, axis=1))]


def color_with_alpha(color: str, alpha: int = 235) -> tuple[int, int, int, int]:
    """Convert a Tk color selected by the user into an RGBA overlay color."""
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return red, green, blue, alpha


def centroid_layout_overlay_image(
    centers_px: np.ndarray,
    anchor_px: np.ndarray,
    diameter_px: float,
    width: int,
    height: int,
    display_size: tuple[int, int],
    outline_color: str = CENTROID_LAYOUT_OUTLINE_COLOR,
    outline_width: int = 1,
    anchor_color: str = CENTROID_LAYOUT_ANCHOR_COLOR,
) -> Image.Image | None:
    """Render equal-size virtual cylinders centered on the detected centroids."""
    if not len(centers_px):
        return None
    display_width, display_height = display_size
    scale_x, scale_y = display_width / width, display_height / height
    radius = diameter_px * min(scale_x, scale_y) / 2
    if not np.isfinite(radius) or radius <= 0:
        return None
    overlay = Image.new("RGBA", display_size)
    draw = ImageDraw.Draw(overlay)
    for x, y in centers_px:
        x, y = x * scale_x, y * scale_y
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color_with_alpha(outline_color), width=outline_width)
    anchor_x, anchor_y = anchor_px[0] * scale_x, anchor_px[1] * scale_y
    draw.ellipse((anchor_x - 4, anchor_y - 4, anchor_x + 4, anchor_y + 4), fill=color_with_alpha(anchor_color), outline=(255, 245, 245, 255), width=1)
    return overlay


def bcp_lattice_geometry(centers_px: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """Return physically local Delaunay segments, Pitch, and grain candidates."""
    edges = retain_local_bcp_edges(centers_px, delaunay_edge_indices(centers_px))
    segments = np.asarray([(*centers_px[first], *centers_px[second]) for first, second in edges], dtype=float).reshape(-1, 4)
    orientations, spacing, _neighbors = local_hexagonal_orientations(centers_px)
    boundary_segments, boundary_indices = [], set()
    for first, second in edges:
        if (
            np.isfinite(orientations[first])
            and np.isfinite(orientations[second])
            and hexagonal_angle_difference(orientations[first], orientations[second])
            >= GRAIN_BOUNDARY_ORIENTATION_THRESHOLD_DEGREES
        ):
            boundary_segments.append((*centers_px[first], *centers_px[second]))
            boundary_indices.update((int(first), int(second)))
    return (
        segments,
        pitch_values_by_direction(centers_px, edges),
        np.asarray(boundary_segments, dtype=float).reshape(-1, 4),
        np.asarray(sorted(boundary_indices), dtype=int),
        spacing,
    )


def mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    """Return a sample mean and its standard error."""
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))


def mean_and_sigma_spread(values: np.ndarray, sigma_multiplier: float) -> tuple[float, float]:
    """Return a sample mean and a user-selected multiple of sample sigma."""
    if not len(values):
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1) * sigma_multiplier) if len(values) > 1 else 0.0


def all_directional_pitch_values(pitch_values_by_direction_px: dict[str, np.ndarray]) -> np.ndarray:
    """Combine retained Pitch edges from all three lattice directions."""
    return np.concatenate([pitch_values_by_direction_px[label] for label in PITCH_DIRECTION_LABELS])


def bcp_calibration_scale(dot_array: np.ndarray, references: list[tuple[float, float, float]]) -> float:
    """Infer the manual-reference dimensional scale without changing point positions."""
    ratios = []
    for center_x, center_y, radius in references:
        distances = np.hypot(dot_array[:, 0] - center_x, dot_array[:, 1] - center_y)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= radius:
            ratios.append(radius * 2 / dot_array[nearest, 2])
    return float(np.median(ratios)) if ratios else 1.0


def calibrate_bcp_dimensions(
    dot_array: np.ndarray,
    references: list[tuple[float, float, float]],
) -> np.ndarray:
    """Use manually marked dot diameters to calibrate the fitted intensity-core size."""
    scale = bcp_calibration_scale(dot_array, references)
    calibrated = dot_array.copy()
    calibrated[:, 2:5] *= scale
    calibrated[:, 6] *= scale**2
    return calibrated


def analyze_bcp_dots(
    image: np.ndarray,
    pixel_size_nm: float | None,
    references: list[tuple[float, float, float]] | None = None,
    expected_diameter_px: float | None = None,
    min_area_fraction: float = 0.25,
    local_horizontal_sections: int = 4,
    local_vertical_sections: int = 4,
    use_local_segmentation: bool = True,
    use_dog_contrast: bool = True,
    internal_hole_fraction: float = 0.12,
) -> BCPAnalysisResult:
    """Detect perpendicular BCP dots, fit their dimensions, and mark grain boundaries."""
    if min(image.shape) < 32:
        raise ValueError("图像过小，无法识别 BCP 点阵。")
    references = references or []
    dots, contour_segments, components = connected_region_bcp_dots(
        image,
        references,
        expected_diameter_px,
        min_area_fraction,
        internal_hole_fraction,
        local_horizontal_sections,
        local_vertical_sections,
        use_local_segmentation,
        use_dog_contrast,
    )
    if len(dots) < 4:
        raise ValueError("识别到的 BCP 点太少；请使用对比度更清晰的俯视图，或关闭预处理后重试。")
    dot_array = np.asarray(dots, dtype=float)
    calibration_scale = bcp_calibration_scale(dot_array, references)
    dot_array = calibrate_bcp_dimensions(dot_array, references)
    centers = dot_array[:, :2]
    directional_cds = np.asarray([component_directional_cds(component) for component in components]) * calibration_scale
    triangulation_segments, directional_pitches, boundary_segments, boundary_indices, spacing = bcp_lattice_geometry(centers)
    return BCPAnalysisResult(
        centers_px=centers,
        equivalent_diameters_px=dot_array[:, 2],
        major_axes_px=dot_array[:, 3],
        minor_axes_px=dot_array[:, 4],
        angles_degrees=dot_array[:, 5],
        areas_px=dot_array[:, 6],
        contour_segments_px=contour_segments,
        components_px=components,
        directional_cds_px=directional_cds,
        cd_means_px=np.mean(directional_cds, axis=1),
        triangulation_segments_px=triangulation_segments,
        pitch_values_by_direction_px=directional_pitches,
        boundary_segments_px=boundary_segments,
        boundary_dot_indices=boundary_indices,
        lattice_spacing_px=spacing,
        pixel_size_nm=pixel_size_nm if pixel_size_nm and pixel_size_nm > 0 else float("nan"),
    )


def finalize_bcp_measurements(result: BCPAnalysisResult) -> BCPAnalysisResult:
    """Calculate CD, Delaunay Pitch, and grain candidates after recognition is complete."""
    scale = bcp_result_dimension_scale(result)
    directional_cds = np.asarray([component_directional_cds(component) for component in result.components_px]) * scale
    triangulation_segments, directional_pitches, boundary_segments, boundary_indices, spacing = bcp_lattice_geometry(result.centers_px)
    return BCPAnalysisResult(
        centers_px=result.centers_px,
        equivalent_diameters_px=result.equivalent_diameters_px,
        major_axes_px=result.major_axes_px,
        minor_axes_px=result.minor_axes_px,
        angles_degrees=result.angles_degrees,
        areas_px=result.areas_px,
        contour_segments_px=result.contour_segments_px,
        components_px=result.components_px,
        directional_cds_px=directional_cds,
        cd_means_px=np.mean(directional_cds, axis=1),
        triangulation_segments_px=triangulation_segments,
        pitch_values_by_direction_px=directional_pitches,
        boundary_segments_px=boundary_segments,
        boundary_dot_indices=boundary_indices,
        lattice_spacing_px=spacing,
        pixel_size_nm=result.pixel_size_nm,
    )


class LERLWRApp(AppBase):
    def __init__(self) -> None:
        super().__init__()
        self.title("SEM measure（LER/LWR + BCP 点阵 + 尺寸标注）")
        self.minsize(1100, 600)
        self.image_path: Path | None = None
        self.original_image: np.ndarray | None = None
        self.raw_image: np.ndarray | None = None
        self.processed_image: np.ndarray | None = None
        self.bcp_preview_image: np.ndarray | None = None
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
        self.detected_edge_mask: np.ndarray | None = None
        self.detected_edge_origin: tuple[int, int] = (0, 0)
        self.detected_edge_overlay: ImageTk.PhotoImage | None = None
        self.annotations: list[MeasurementAnnotation] = []
        self.active_annotation_index: int | None = None
        self.annotation_drag_mode: str | None = None
        self.annotation_drag_anchor: tuple[float, float] | None = None
        self.annotation_start_bounds: tuple[float, float, float, float] | None = None
        self.annotation_was_changed = False
        self.annotation_selection_start: tuple[float, float] | None = None
        self.annotation_selection_bounds: tuple[float, float, float, float] | None = None
        self.annotation_selection_rectangle: int | None = None
        self.undo_history: list[EditorState] = []
        self.pending_undo_state: EditorState | None = None
        self.space_held = False
        self.panning = False
        self.result: AnalysisResult | None = None
        self.analysis_origin: tuple[int, int] | None = None
        self.era_line_samples: list[EraLineSample] = []
        self.era_aggregate_result: EraAggregateResult | None = None
        self.era_active_sample_index: int | None = None
        self.era_pending_roi_target: tuple[int, str] | None = None
        self.era_sample_listbox: tk.Listbox | None = None
        self.bcp_result: BCPAnalysisResult | None = None
        self.bcp_metrics_finalized = False
        self.bcp_reference_circles: list[tuple[float, float, float]] = []
        self.bcp_reference_start: tuple[float, float] | None = None
        self.bcp_reference_preview_radius = 0.0
        self.bcp_reference_mode = False
        self.bcp_reference_active_index: int | None = None
        self.bcp_reference_drag_mode: str | None = None
        self.bcp_reference_drag_anchor: tuple[float, float] | None = None
        self.bcp_reference_start_circle: tuple[float, float, float] | None = None
        self.bcp_completion_mode = False
        self.bcp_completion_drawing = False
        self.bcp_completion_polygon_px: list[tuple[float, float]] = []
        self.bcp_completion_polygons_px: list[list[tuple[float, float]]] = []
        self.bcp_split_mode = False
        self.bcp_split_drawing = False
        self.bcp_split_start_px: tuple[float, float] | None = None
        self.bcp_split_end_px: tuple[float, float] | None = None
        self.bcp_dialog: tk.Toplevel | None = None
        self.bcp_dialog_scroll: VerticalScrollFrame | None = None
        self.fit_style_dialog: tk.Toplevel | None = None
        self.ler_lwr_dialog: tk.Toplevel | None = None
        self.ler_lwr_dialog_scroll: VerticalScrollFrame | None = None
        self.ler_lwr_roi_mode = False
        self.lcdu_cd_samples_nm: list[float] = []
        self.era_lcdu_cd_samples_nm: list[float] = []
        self.lcdu_sample_listbox: tk.Listbox | None = None
        self.metadata_text = "尚未导入 TIFF 文件。"
        self.rotation_degrees = 0.0
        self.info_bar_crop_height_px = 0

        self.pixel_size_var = tk.StringVar(value="")
        self.metadata_var = tk.StringVar(value="尚未读取 TIFF 元数据")
        self.normalize_var = tk.BooleanVar(value=True)
        self.gaussian_denoise_var = tk.BooleanVar(value=True)
        self.blur_method_var = tk.StringVar(value=BLUR_METHOD_GAUSSIAN)
        self.gaussian_kernel_size_var = tk.StringVar(value="3 × 3")
        self.gaussian_sigma_scale_var = tk.DoubleVar(value=1.0)
        self.gaussian_sigma_text_var = tk.StringVar(value="1.00（标准）")
        self.gaussian_kernel_preview_var = tk.StringVar(value=gaussian_kernel_preview(3, 1.0))
        self.edge_detector_var = tk.StringVar(value=EDGE_DETECTOR_SCHARR)
        self.era_polarity_var = tk.StringVar(value=ERA_POLARITY_AUTO)
        self.era_polynomial_degree_var = tk.StringVar(value=str(ERA_DEFAULT_POLYNOMIAL_DEGREE))
        self.edge_kernel_size_var = tk.StringVar(value="3 × 3")
        self.edge_diagonal_weight_var = tk.StringVar(value="3")
        self.edge_axial_weight_var = tk.StringVar(value="10")
        self.canny_high_threshold_var = tk.DoubleVar(value=0.125)
        self.canny_threshold_ratio_var = tk.DoubleVar(value=2.5)
        self.edge_use_vertical_continuity_filter_var = tk.BooleanVar(value=False)
        self.canny_high_threshold_text_var = tk.StringVar(value="0.125")
        self.canny_threshold_ratio_text_var = tk.StringVar(value="2.50")
        self.canny_low_threshold_text_var = tk.StringVar(value="0.050")
        self.edge_kernel_preview_var = tk.StringVar(value="")
        self.bcp_use_dog_var = tk.BooleanVar(value=True)
        self.sigma_multiplier_var = tk.DoubleVar(value=1.0)
        self.outlier_level_var = tk.StringVar(value=DEFAULT_OUTLIER_LEVEL)
        self.lcdu_summary_var = tk.StringVar(value="LCDU 样本：0 条线")
        self.rotation_var = tk.StringVar(value="0.00")
        self.measurement_tool_var = tk.StringVar(value="框选标注")
        self.measurement_color = ANNOTATION_COLORS["length"]
        self.zoom_slider_var = tk.DoubleVar(value=1.0)
        self.zoom_percent_var = tk.StringVar(value="100")
        self.recognition_duration_var = tk.StringVar(value="识别耗时：—")
        self.status_var = tk.StringVar(value="打开 SEM 图像后，可框选线条分析，或直接识别整图 BCP 点阵。")
        self.result_var = tk.StringVar(value="尚未分析")
        self.bcp_reference_var = tk.StringVar(value="当前没有手动样本。")
        self.bcp_expected_diameter_var = tk.StringVar(value="")
        self.bcp_expected_unit_var = tk.StringVar(value="nm")
        self.bcp_min_area_fraction_var = tk.StringVar(value="0.25")
        self.bcp_internal_hole_fraction_var = tk.StringVar(value="0.12")
        self.bcp_horizontal_sections_var = tk.StringVar(value="4")
        self.bcp_vertical_sections_var = tk.StringVar(value="4")
        self.bcp_use_local_segmentation_var = tk.BooleanVar(value=True)
        self.bcp_line_display_var = tk.StringVar(value=TRIANGULATION_DISPLAY_ALL)
        self.bcp_triangulation_overlay_var = tk.BooleanVar(value=True)
        self.bcp_grain_overlay_var = tk.BooleanVar(value=True)
        self.bcp_centroid_layout_overlay_var = tk.BooleanVar(value=False)
        self.fit_line_width_var = tk.StringVar(value=DEFAULT_FIT_LINE_WIDTH)
        self.bcp_reference_display_var = tk.BooleanVar(value=False)
        self.bcp_contour_width_var = tk.StringVar(value=DEFAULT_BCP_OVERLAY_LINE_WIDTH)
        self.bcp_triangulation_width_var = tk.StringVar(value=DEFAULT_BCP_OVERLAY_LINE_WIDTH)
        self.bcp_centroid_diameter_var = tk.StringVar(value=DEFAULT_BCP_CENTROID_DIAMETER)
        self.bcp_centroid_layout_width_var = tk.StringVar(value=DEFAULT_BCP_OVERLAY_LINE_WIDTH)
        self.bcp_contour_color = DEFAULT_BCP_CONTOUR_COLOR
        self.bcp_triangulation_color = DEFAULT_BCP_TRIANGULATION_COLOR
        self.bcp_centroid_color = DEFAULT_BCP_CENTROID_COLOR
        self.bcp_centroid_layout_color = CENTROID_LAYOUT_OUTLINE_COLOR
        self.bcp_centroid_anchor_color = CENTROID_LAYOUT_ANCHOR_COLOR
        self.fit_left_color = DEFAULT_FIT_LEFT_COLOR
        self.fit_right_color = DEFAULT_FIT_RIGHT_COLOR
        self.grain_overlay_image: ImageTk.PhotoImage | None = None
        self.centroid_layout_overlay_image: ImageTk.PhotoImage | None = None

        self._build_ui()
        self.bind("<FocusIn>", self.keep_ler_lwr_dialog_above_main, add="+")
        self.bind_all("<MouseWheel>", self.scroll_active_window, add="+")
        self.bind_all("<Button-4>", self.scroll_active_window, add="+")
        self.bind_all("<Button-5>", self.scroll_active_window, add="+")
        self.bind_all("<Shift-MouseWheel>", self.scroll_active_window, add="+")
        self.bind_all("<Shift-Button-4>", self.scroll_active_window, add="+")
        self.bind_all("<Shift-Button-5>", self.scroll_active_window, add="+")
        self.pixel_size_var.trace_add("write", self.refresh_annotation_labels)

    def _build_ui(self) -> None:
        self.main_scroll = VerticalScrollFrame(self)
        self.main_scroll.pack(fill=tk.BOTH, expand=True)
        body = self.main_scroll.content

        controls = ttk.Frame(body, padding=10)
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
        file_menu.add_command(label="拟合线条样式", command=self.open_fit_line_style_dialog)
        file_menu.add_command(label="导出 CSV", command=self.export_csv)
        file_menu.add_command(label="导出标注/拟合图片", command=self.export_annotated_image)
        ttk.Menubutton(controls, text="菜单 ▾", menu=file_menu).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(controls, text="像素尺寸 (nm/pixel):").grid(row=0, column=1, sticky="e")
        ttk.Entry(controls, textvariable=self.pixel_size_var, width=10).grid(row=0, column=2, padx=(4, 12))
        ttk.Label(controls, textvariable=self.metadata_var, foreground="#426b2d").grid(row=0, column=3, padx=(0, 12), sticky="w")
        self.image_processing_button = ttk.Button(controls, text="图像处理 ▸", command=self.toggle_image_processing_panel)
        self.image_processing_button.grid(row=0, column=4, padx=(0, 12))
        self.analysis_tools_button = ttk.Button(controls, text="分析功能 ▸", command=self.toggle_analysis_tools_panel)
        self.analysis_tools_button.grid(row=0, column=5, padx=(0, 12))

        multiplier = ttk.LabelFrame(controls, text="显示/导出倍数", padding=(8, 2))
        multiplier.grid(row=0, column=6, padx=(8, 0), sticky="ew")
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

        self.image_processing_panel = ttk.LabelFrame(body, text="图像处理", padding=(10, 6))
        self.image_processing_panel_visible = False
        ttk.Checkbutton(
            self.image_processing_panel,
            text="归一化（1%–99%）",
            variable=self.normalize_var,
            command=self.refresh_image_preprocessing,
        ).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(
            self.image_processing_panel,
            text="启用平滑",
            variable=self.gaussian_denoise_var,
            command=self.refresh_image_preprocessing,
        ).pack(side=tk.LEFT, padx=(0, 5))
        blur_selector = ttk.Combobox(
            self.image_processing_panel,
            textvariable=self.blur_method_var,
            state="readonly",
            values=(BLUR_METHOD_GAUSSIAN, BLUR_METHOD_MEAN),
            width=15,
        )
        blur_selector.pack(side=tk.LEFT, padx=(0, 5))
        blur_selector.bind("<<ComboboxSelected>>", self.refresh_image_preprocessing)
        kernel_selector = ttk.Combobox(
            self.image_processing_panel,
            textvariable=self.gaussian_kernel_size_var,
            state="readonly",
            values=tuple(f"{size} × {size}" for size in GAUSSIAN_KERNEL_SIZES),
            width=7,
        )
        kernel_selector.pack(side=tk.LEFT, padx=(0, 18))
        kernel_selector.bind("<<ComboboxSelected>>", self.refresh_image_preprocessing)
        sigma_controls = ttk.Frame(self.image_processing_panel)
        sigma_controls.pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(sigma_controls, text="σ（相对默认）：").pack(side=tk.LEFT)
        ttk.Scale(
            sigma_controls,
            from_=GAUSSIAN_SIGMA_SCALE_MIN,
            to=GAUSSIAN_SIGMA_SCALE_MAX,
            orient=tk.HORIZONTAL,
            variable=self.gaussian_sigma_scale_var,
            command=self.refresh_image_preprocessing,
            length=95,
        ).pack(side=tk.LEFT, padx=(3, 5))
        ttk.Label(sigma_controls, textvariable=self.gaussian_sigma_text_var, width=10).pack(side=tk.LEFT)
        ttk.Checkbutton(
            self.image_processing_panel,
            text="BCP DoG 对比增强（显示/识别）",
            variable=self.bcp_use_dog_var,
            command=self.refresh_bcp_contrast,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(
            self.image_processing_panel,
            text="DoG 过强时可取消勾选，改以原始亮暗强度分割。",
            foreground="#666666",
        ).pack(side=tk.LEFT)
        ttk.Label(
            self.image_processing_panel,
            textvariable=self.gaussian_kernel_preview_var,
            justify=tk.LEFT,
            font=("Menlo", 9),
        ).pack(side=tk.RIGHT, padx=(18, 0))

        self.analysis_tools_panel = ttk.LabelFrame(body, text="分析功能", padding=(10, 6))
        self.analysis_tools_panel_visible = False
        ttk.Button(self.analysis_tools_panel, text="测量 LER / LWR", command=self.open_ler_lwr_dialog).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(self.analysis_tools_panel, text="识别 BCP 点阵", command=self.open_bcp_recognition_dialog).pack(side=tk.LEFT)
        ttk.Label(self.analysis_tools_panel, text="两项分析各自在独立窗口中设置与操作。", foreground="#666666").pack(side=tk.LEFT, padx=(12, 0))

        content = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
        self.main_content = content
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        image_frame = ttk.LabelFrame(content, text="图像工作区", padding=5)
        result_frame = ttk.LabelFrame(content, text="分析结果 / 操作提示", padding=8)
        content.add(image_frame, weight=5)
        content.add(result_frame, weight=1)

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
        ttk.Label(zoom_controls, text="缩放").pack(side=tk.LEFT, padx=(8, 3))
        zoom_entry = ttk.Entry(zoom_controls, textvariable=self.zoom_percent_var, width=5)
        zoom_entry.pack(side=tk.LEFT)
        zoom_entry.bind("<Return>", self.apply_zoom_percent_value)
        zoom_entry.bind("<FocusOut>", self.apply_zoom_percent_value)
        ttk.Label(zoom_controls, text="%").pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(zoom_controls, textvariable=self.recognition_duration_var).pack(side=tk.LEFT, padx=(12, 0))

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

        image_workspace = ttk.Frame(image_frame)
        image_workspace.pack(fill=tk.BOTH, expand=True)
        left_toolbar = ttk.Frame(image_workspace)
        left_toolbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        selection_toolbar = ttk.LabelFrame(left_toolbar, text="标注选择", padding=(6, 8))
        selection_toolbar.pack(fill=tk.X, pady=(0, 6))
        ttk.Radiobutton(
            selection_toolbar,
            text="框选标注",
            value="框选标注",
            variable=self.measurement_tool_var,
            command=self.change_measurement_tool,
        ).pack(anchor="w", pady=3)
        annotation_toolbar = ttk.LabelFrame(left_toolbar, text="通用标注", padding=(6, 8))
        annotation_toolbar.pack(fill=tk.X)
        for label in ("长度", "矩形", "圆形", "正六边形"):
            ttk.Radiobutton(
                annotation_toolbar,
                text=label,
                value=label,
                variable=self.measurement_tool_var,
                command=self.change_measurement_tool,
            ).pack(anchor="w", pady=3)
        self.color_button = tk.Button(annotation_toolbar, text="长度线颜色", command=self.choose_measurement_color, relief=tk.GROOVE)
        self.color_button.pack(fill=tk.X, pady=(9, 0))
        self.update_measurement_color_button()

        canvas_frame = ttk.Frame(image_workspace)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="#202020", highlightthickness=0, width=780, height=620, takefocus=True)
        self.image_vertical_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.image_horizontal_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(
            xscrollcommand=self.image_horizontal_scrollbar.set,
            yscrollcommand=self.image_vertical_scrollbar.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.image_vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        self.image_horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
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

        ttk.Label(
            result_frame,
            textvariable=self.result_var,
            justify=tk.LEFT,
            wraplength=260,
            font=("Menlo", 10),
        ).pack(anchor="nw", fill=tk.X, pady=(0, 12))
        lcdu_controls = ttk.LabelFrame(result_frame, text="多线 LCDU", padding=(8, 6))
        lcdu_controls.pack(fill=tk.X, pady=(0, 12))
        self.build_lcdu_controls(lcdu_controls)
        hint = (
            "操作提示\n"
            "1. 点击导入，或从 Finder 直接拖入 TIFF。\n"
            "2. TIFF 会自动读取 SEM 像素尺寸。\n"
            "3. 展开“分析功能”后，可进入 LER/LWR 或 BCP 的设置窗口。\n"
            "4. 测量结果会同步显示在这里；默认输出 1σ，顶部滑块可改为 kσ。\n\n"
            "通用测量：在左侧工具列选择长度、矩形、圆形或正六边形后拖动创建。\n"
            "点击已有图形可移动或调整；尺寸会按 nm/pixel 自动标注。\n"
            "未填写像素尺寸时，通用测量暂以 px 显示。\n\n"
            "底部缩放：拖动滑块、点 − / +，或直接输入百分比后按回车。\n"
            "按住空格 + 鼠标滚轮：以鼠标位置缩放。\n"
            "按住空格 + 左键拖动：平移图像。\n\n"
            "窗口内容：鼠标滚轮上下滚动；按住 Shift + 滚轮可左右滚动。\n\n"
            "“框选标注”会临时框住并选中已有标注，松开鼠标后选择框自动消失。\n"
            "选中通用标注时 Backspace 删除标注。\n\n"
            "“图像处理”可独立切换归一化、可调高斯卷积和 BCP DoG 对比增强。\n"
            "取消 DoG 后，BCP 将以原始亮暗强度进行分割。\n\n"
            "LER/LWR 假设线条沿竖直方向；ROI 在测量窗口中选择，结果会保留在主窗口。"
            "\n\nBCP 点阵：点击“识别 BCP 点阵”，先标示至少 3 个代表圆柱，再识别整图点位；绿色线为二值连通区域轮廓，蓝线为 Delaunay 三角网，红线为依据局部六重对称取向变化推断的晶界。"
        )
        ttk.Label(result_frame, text=hint, justify=tk.LEFT, wraplength=210).pack(anchor="nw")
        ttk.Label(body, textvariable=self.status_var, anchor="w", padding=(12, 6)).pack(side=tk.BOTTOM, fill=tk.X)

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
                self.original_image, self.info_bar_crop_height_px = remove_bottom_information_bar(np.asarray(image))
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
            self.bcp_expected_unit_var.set("px")
        else:
            self.pixel_size_var.set(f"{metadata_pixel_size:.8g}")
            self.metadata_var.set(f"TIFF 元数据：{metadata_pixel_size:.6g} nm/pixel")
            self.bcp_expected_unit_var.set("nm")
        self.result = None
        self.analysis_origin = None
        self.bcp_result = None
        self.recognition_duration_var.set("识别耗时：—")
        self.clear_bcp_completion_selection(redraw=False)
        self.discard_bcp_reference_circles()
        self.lcdu_cd_samples_nm.clear()
        self.update_lcdu_summary_text()
        self.roi_canvas = None
        self.clear_era_line_samples(redraw=False)
        self.clear_auto_line_candidates(redraw=False)
        self.annotations.clear()
        self.active_annotation_index = None
        self.draw_image(reset_view=True)
        self.result_var.set("请框选一条线及两侧背景，然后点击“分析选区”。")
        crop_note = f"；已自动裁掉底部信息栏 {self.info_bar_crop_height_px} px" if self.info_bar_crop_height_px else ""
        self.status_var.set(
            f"已打开：{self.image_path.name}  ({self.raw_image.shape[1]} × {self.raw_image.shape[0]} px)；"
            f"当前图像处理：{self.preprocessing_description()}{crop_note}"
        )

    def snapshot_editor_state(self) -> EditorState:
        annotations = [MeasurementAnnotation(item.kind, item.bounds_px, item.color) for item in self.annotations]
        return EditorState(
            self.roi_canvas,
            annotations,
            self.active_annotation_index,
            self.rotation_degrees,
            self.normalize_var.get(),
            self.gaussian_denoise_var.get(),
            self.gaussian_kernel_size_var.get(),
            self.gaussian_sigma_scale_var.get(),
            self.blur_method_var.get(),
            self.bcp_use_dog_var.get(),
            self.result,
            self.analysis_origin,
            self.bcp_result,
            self.bcp_metrics_finalized,
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
        self.normalize_var.set(state.normalize_enabled)
        self.gaussian_denoise_var.set(state.gaussian_denoise_enabled)
        self.gaussian_kernel_size_var.set(state.gaussian_kernel_size)
        self.gaussian_sigma_scale_var.set(state.gaussian_sigma_scale)
        self.blur_method_var.set(state.blur_method)
        self.bcp_use_dog_var.set(state.bcp_dog_enabled)
        self.update_gaussian_kernel_preview()
        self.result = state.result
        self.analysis_origin = state.analysis_origin
        self.bcp_result = state.bcp_result
        self.bcp_metrics_finalized = state.bcp_metrics_finalized
        self.rebuild_working_images()
        self.draw_image()
        if self.result is not None:
            self.update_result_text()
        elif self.bcp_result is not None:
            self.update_bcp_result_text()
        else:
            self.result_var.set("已撤销上一步操作。")
        self.status_var.set("已撤销上一步操作。")
        return "break"

    def rebuild_working_images(self) -> None:
        if self.original_image is None:
            return
        self.raw_image = rotate_grayscale_image(self.original_image, self.rotation_degrees)
        self.processed_image = preprocess_image(
            self.raw_image,
            self.normalize_var.get(),
            self.gaussian_denoise_var.get(),
            self.gaussian_kernel_size(),
            self.gaussian_sigma_scale(),
            self.blur_method_var.get(),
        )
        self.rebuild_bcp_preview()

    def bcp_preview_diameter_px(self) -> float:
        """Choose a stable DoG preview scale without requiring BCP dialog input."""
        if self.bcp_reference_circles:
            return float(np.median([radius * 2 for _x, _y, radius in self.bcp_reference_circles]))
        try:
            diameter = float(self.bcp_expected_diameter_var.get())
        except ValueError:
            diameter = 0.0
        if diameter > 0:
            if self.bcp_expected_unit_var.get() == "px":
                return diameter
            pixel_size = self.measurement_pixel_size()
            if pixel_size is not None:
                return diameter / pixel_size
        if self.processed_image is None:
            return 12.0
        return float(np.clip(min(self.processed_image.shape) / 30, 8.0, 48.0))

    def rebuild_bcp_preview(self) -> None:
        """Create the optional DoG canvas preview without changing analysis input."""
        self.bcp_preview_image = None
        if self.processed_image is None or not self.bcp_use_dog_var.get():
            return
        response = bcp_response(self.processed_image, 1.0, self.bcp_preview_diameter_px())
        self.bcp_preview_image = normalize_intensity(response)

    def auto_align_roi(self) -> None:
        if self.edge_detector_var.get() != EDGE_DETECTOR_LAPLACIAN:
            self.status_var.set("自动校正 ROI 只用于拉普拉斯；Scharr/Canny 请用底部手动旋转后重新开始边缘识别。")
            return
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
            self.bcp_result = None
            self.clear_bcp_completion_selection(redraw=False)
            self.discard_bcp_reference_circles()
            self.roi_canvas = None
            self.clear_era_line_samples(redraw=False)
            self.clear_auto_line_candidates(redraw=False)
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
        self.bcp_result = None
        self.clear_bcp_completion_selection(redraw=False)
        self.discard_bcp_reference_circles()
        self.roi_canvas = None
        self.clear_era_line_samples(redraw=False)
        self.clear_auto_line_candidates(redraw=False)
        self.clear_annotations(redraw=False, record_history=False)
        self.draw_image(reset_view=True)
        self.result_var.set("已恢复原始图像方向；请重新框选 ROI，或直接使用整图重新识别 / 分析。")
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
        self.bcp_result = None
        self.clear_bcp_completion_selection(redraw=False)
        self.discard_bcp_reference_circles()
        self.roi_canvas = None
        self.clear_era_line_samples(redraw=False)
        self.clear_auto_line_candidates(redraw=False)
        self.clear_annotations(redraw=False, record_history=False)
        self.draw_image()
        self.restore_view_center(view_center)
        self.result_var.set("图像角度已手动调整；请重新框选 ROI，或直接使用整图重新识别 / 分析。")
        self.status_var.set(f"当前累计旋转角度：{self.rotation_degrees:+.2f}°。")

    def show_metadata(self) -> None:
        window = tk.Toplevel(self)
        window.title("TIFF / SEM 元数据")
        window.geometry("820x650")
        viewer = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Menlo", 11))
        viewer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        viewer.insert("1.0", self.metadata_text)
        viewer.configure(state=tk.DISABLED)

    def toggle_image_processing_panel(self) -> None:
        if self.image_processing_panel_visible:
            self.image_processing_panel.pack_forget()
            self.image_processing_button.configure(text="图像处理 ▸")
        else:
            self.image_processing_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6), before=self.main_content)
            self.image_processing_button.configure(text="图像处理 ▾")
        self.image_processing_panel_visible = not self.image_processing_panel_visible

    def toggle_analysis_tools_panel(self) -> None:
        if self.analysis_tools_panel_visible:
            self.analysis_tools_panel.pack_forget()
            self.analysis_tools_button.configure(text="分析功能 ▸")
        else:
            self.analysis_tools_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6), before=self.main_content)
            self.analysis_tools_button.configure(text="分析功能 ▾")
        self.analysis_tools_panel_visible = not self.analysis_tools_panel_visible

    def gaussian_kernel_size(self) -> int:
        return int(self.gaussian_kernel_size_var.get().split()[0])

    def gaussian_sigma_scale(self) -> float:
        return float(np.clip(self.gaussian_sigma_scale_var.get(), GAUSSIAN_SIGMA_SCALE_MIN, GAUSSIAN_SIGMA_SCALE_MAX))

    def update_gaussian_kernel_preview(self) -> None:
        sigma_scale = self.gaussian_sigma_scale()
        sigma_text = "1.00（标准）" if math.isclose(sigma_scale, 1.0, abs_tol=0.005) else f"{sigma_scale:.2f}"
        self.gaussian_sigma_text_var.set(sigma_text)
        preview = mean_kernel_preview(self.gaussian_kernel_size()) if self.blur_method_var.get() == BLUR_METHOD_MEAN else gaussian_kernel_preview(self.gaussian_kernel_size(), sigma_scale)
        self.gaussian_kernel_preview_var.set(preview)

    def preprocessing_description(self) -> str:
        parts = []
        if self.normalize_var.get():
            parts.append("归一化")
        if self.gaussian_denoise_var.get():
            if self.blur_method_var.get() == BLUR_METHOD_MEAN:
                parts.append(f"{self.gaussian_kernel_size_var.get()} 均值模糊")
            else:
                parts.append(f"{self.gaussian_kernel_size_var.get()} 高斯模糊（σ×{self.gaussian_sigma_scale():.2f}）")
        if self.bcp_use_dog_var.get():
            parts.append("BCP DoG 预览")
        return " + ".join(parts) if parts else "原始灰度图"

    def refresh_image_preprocessing(self, _event: object = None) -> None:
        if self.raw_image is None:
            self.update_gaussian_kernel_preview()
            return
        self.update_gaussian_kernel_preview()
        self.result = None
        self.analysis_origin = None
        self.clear_era_line_samples(redraw=False)
        self.bcp_result = None
        self.clear_bcp_completion_selection(redraw=False)
        self.clear_auto_line_candidates(redraw=False)
        self.rebuild_working_images()
        self.draw_image()
        self.result_var.set("处理方式已切换；请重新开始分析。")
        self.status_var.set(f"当前使用：{self.preprocessing_description()}")

    def refresh_bcp_contrast(self) -> None:
        if self.raw_image is None:
            return
        self.bcp_result = None
        self.clear_bcp_completion_selection(redraw=False)
        self.rebuild_bcp_preview()
        self.draw_image()
        mode = "DoG 对比增强" if self.bcp_use_dog_var.get() else "原始亮暗强度"
        self.result_var.set("BCP 对比方式已切换；画布预览已刷新，请重新点击“开始识别”。")
        self.status_var.set(f"画布已切换为：{mode}；LER/LWR 分析输入不变。")

    def draw_image(self, reset_view: bool = False) -> None:
        if self.raw_image is None:
            return
        image = self.bcp_preview_image if self.bcp_use_dog_var.get() else self.processed_image
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
        self.render_bcp_overlay()
        self.draw_bcp_completion_polygon()
        self.draw_bcp_manual_split()
        self.draw_bcp_reference_circles()
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
            self.draw_era_rois()
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
        self.draw_era_rois()

    def draw_era_rois(self) -> None:
        """Show the stored manual single-edge ROIs without making them editable on canvas."""
        self.canvas.delete("era_roi")
        for index, sample in enumerate(self.era_line_samples, start=1):
            for side, bounds, color in (
                ("L", sample.left_roi_bounds_px, self.fit_left_color),
                ("R", sample.right_roi_bounds_px, self.fit_right_color),
            ):
                if bounds is None:
                    continue
                left, top, right, bottom = bounds
                scale = self.display_scale
                self.canvas.create_rectangle(
                    left * scale,
                    top * scale,
                    right * scale,
                    bottom * scale,
                    outline=color,
                    width=2,
                    dash=(5, 3),
                    tags="era_roi",
                )
                self.canvas.create_text(
                    left * scale + 4,
                    top * scale + 10,
                    text=f"{index}{side}",
                    anchor=tk.W,
                    fill=color,
                    tags="era_roi",
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
        if self.bcp_completion_mode:
            self.canvas.configure(cursor="pencil")
            return
        if self.bcp_reference_mode:
            self.canvas.configure(cursor="crosshair")
            return
        if self.space_held:
            self.canvas.configure(cursor="fleur")
            return
        if self.ler_lwr_roi_mode:
            hit = self.roi_hit_test(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            self.canvas.configure(cursor="fleur" if hit == "move" else "crosshair")
            return
        if self.measurement_tool_var.get() == "框选标注":
            self.canvas.configure(cursor="crosshair")
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
        self.ler_lwr_roi_mode = False
        if tool == "框选标注":
            self.status_var.set("当前工具：框选标注。拖动可临时框选已有标注，松开鼠标后选择框自动消失。")
        else:
            self.status_var.set(f"当前工具：{tool}。拖动可创建；点击已有标注可移动或调整。")

    def change_outlier_level(self, _event: tk.Event | None = None) -> None:
        if self.result is None:
            self.status_var.set(f"异常点剔除：{self.outlier_level_var.get()}。请点击“分析选区”应用。")
            return
        self.result = None
        self.analysis_origin = None
        self.clear_auto_line_candidates(redraw=False)
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

    def open_fit_line_style_dialog(self) -> None:
        if self.fit_style_dialog is not None and self.fit_style_dialog.winfo_exists():
            self.fit_style_dialog.focus_set()
            return
        self.fit_style_dialog = tk.Toplevel(self)
        self.fit_style_dialog.title("拟合线条样式")
        self.fit_style_dialog.geometry("360x180")
        self.fit_style_dialog.transient(self)
        self.fit_style_dialog.protocol("WM_DELETE_WINDOW", self.close_fit_line_style_dialog)
        content = ttk.Frame(self.fit_style_dialog, padding=14)
        content.pack(fill=tk.BOTH, expand=True)
        ttk.Label(content, text="LER/LWR 拟合边缘（画面/导出）", font=("Helvetica", 13, "bold")).pack(anchor="w")
        ttk.Label(content, text="只改变显示与导出，不改变边缘定位或测量结果。", foreground="#666666").pack(anchor="w", pady=(5, 9))
        left = ttk.Frame(content)
        left.pack(anchor="w")
        ttk.Label(left, text="左拟合边缘：").pack(side=tk.LEFT)
        self.fit_left_color_button = tk.Button(left, text="选择颜色", background=self.fit_left_color, activebackground=self.fit_left_color, command=lambda: self.choose_fit_line_color("left"))
        self.fit_left_color_button.pack(side=tk.LEFT)
        right = ttk.Frame(content)
        right.pack(anchor="w", pady=(6, 0))
        ttk.Label(right, text="右拟合边缘：").pack(side=tk.LEFT)
        self.fit_right_color_button = tk.Button(right, text="选择颜色", background=self.fit_right_color, activebackground=self.fit_right_color, command=lambda: self.choose_fit_line_color("right"))
        self.fit_right_color_button.pack(side=tk.LEFT)
        width_controls = ttk.Frame(content)
        width_controls.pack(anchor="w", pady=(9, 0))
        ttk.Label(width_controls, text="两条线粗细：").pack(side=tk.LEFT)
        width_selector = ttk.Combobox(width_controls, textvariable=self.fit_line_width_var, state="readonly", values=("1", "2", "3", "4", "5", "6"), width=3)
        width_selector.pack(side=tk.LEFT, padx=(2, 3))
        width_selector.bind("<<ComboboxSelected>>", self.refresh_fit_line_style)
        ttk.Label(width_controls, text="px").pack(side=tk.LEFT)

    def close_fit_line_style_dialog(self) -> None:
        if self.fit_style_dialog is not None and self.fit_style_dialog.winfo_exists():
            self.fit_style_dialog.destroy()
        self.fit_style_dialog = None

    def fit_line_width(self) -> int:
        try:
            return int(np.clip(int(self.fit_line_width_var.get()), 1, 6))
        except ValueError:
            return int(DEFAULT_FIT_LINE_WIDTH)

    def choose_fit_line_color(self, side: str) -> None:
        color_attribute = "fit_left_color" if side == "left" else "fit_right_color"
        button_attribute = "fit_left_color_button" if side == "left" else "fit_right_color_button"
        label = "左" if side == "left" else "右"
        color = colorchooser.askcolor(color=getattr(self, color_attribute), parent=self.fit_style_dialog or self, title=f"选择{label}拟合边缘颜色")[1]
        if color is None:
            return
        setattr(self, color_attribute, color)
        getattr(self, button_attribute).configure(background=color, activebackground=color)
        self.refresh_fit_line_style()

    def refresh_fit_line_style(self, _event: tk.Event | None = None) -> None:
        self.draw_image()

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
                if len(points) >= 2:
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
        if self.bcp_split_mode and not self.space_held:
            self.start_bcp_manual_split_draw(event)
            return
        if self.bcp_completion_mode and not self.space_held:
            self.start_bcp_completion_draw(event)
            return
        if self.bcp_reference_mode and not self.space_held:
            self.start_bcp_reference(event)
            return
        if self.raw_image is None or self.space_held:
            self.start_roi(event)
            return
        if self.ler_lwr_roi_mode:
            self.active_annotation_index = None
            self.begin_undoable_edit()
            self.start_roi(event)
            return
        if self.measurement_tool_var.get() == "框选标注":
            self.start_annotation_selection(event)
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
            if self.ler_lwr_roi_mode:
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

    def clear_detected_edges(self, redraw: bool = True) -> None:
        """Clear the unpaired edge-recognition overlay and its image origin."""
        self.detected_edge_mask = None
        self.detected_edge_origin = (0, 0)
        self.detected_edge_overlay = None
        if redraw and self.raw_image is not None:
            self.render_analysis_overlay()

    def clear_auto_line_candidates(self, redraw: bool = True) -> None:
        """Compatibility wrapper for invalidation paths predating edge-only mode."""
        self.clear_detected_edges(redraw)

    def move_canvas_action(self, event: tk.Event) -> None:
        if self.annotation_selection_start is not None:
            self.move_annotation_selection(event)
            return
        if self.bcp_split_mode and self.bcp_split_drawing:
            self.move_bcp_manual_split_draw(event)
            return
        if self.bcp_completion_mode and self.bcp_completion_drawing:
            self.move_bcp_completion_draw(event)
            return
        if self.bcp_reference_mode and (self.bcp_reference_start is not None or self.bcp_reference_active_index is not None):
            self.move_bcp_reference(event)
            return
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
        if self.annotation_selection_start is not None:
            self.finish_annotation_selection(event)
            return
        if self.bcp_split_mode and self.bcp_split_drawing:
            self.finish_bcp_manual_split_draw(event)
            return
        if self.bcp_completion_mode and self.bcp_completion_drawing:
            self.finish_bcp_completion_draw(event)
            return
        if self.bcp_reference_mode and (self.bcp_reference_start is not None or self.bcp_reference_active_index is not None):
            self.finish_bcp_reference(event)
            return
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

    def start_annotation_selection(self, event: tk.Event) -> None:
        if self.raw_image is None:
            return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.annotation_selection_start = (x, y)
        self.annotation_selection_bounds = (x, y, x, y)
        self.canvas.delete("annotation_selection")
        self.annotation_selection_rectangle = self.canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline="#ffffff",
            dash=(4, 3),
            width=1,
            tags="annotation_selection",
        )

    def move_annotation_selection(self, event: tk.Event) -> None:
        if self.annotation_selection_start is None:
            return
        start_x, start_y = self.annotation_selection_start
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        self.annotation_selection_bounds = self.normalized_roi((start_x, start_y, end_x, end_y))
        if self.annotation_selection_rectangle is not None:
            self.canvas.coords(self.annotation_selection_rectangle, *self.annotation_selection_bounds)

    def finish_annotation_selection(self, event: tk.Event) -> None:
        self.move_annotation_selection(event)
        selection = self.annotation_selection_bounds
        self.annotation_selection_start = None
        self.annotation_selection_bounds = None
        self.annotation_selection_rectangle = None
        self.canvas.delete("annotation_selection")
        if selection is None:
            return
        left, top, right, bottom = selection
        matched = []
        for index, annotation in enumerate(self.annotations):
            x0, y0, x1, y1 = self.annotation_display_bounds(annotation)
            annotation_left, annotation_right = sorted((x0, x1))
            annotation_top, annotation_bottom = sorted((y0, y1))
            if annotation_right >= left and annotation_left <= right and annotation_bottom >= top and annotation_top <= bottom:
                matched.append(index)
        self.active_annotation_index = matched[-1] if matched else None
        self.draw_annotations()
        if not matched:
            self.status_var.set("框选范围内没有标注。")
        elif len(matched) == 1:
            self.status_var.set("已选中框内标注；可直接拖动调整或删除。")
        else:
            self.status_var.set(f"框选到 {len(matched)} 个标注，已选中最上层标注；可直接拖动调整或删除。")

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
        if self.bcp_split_mode or self.bcp_split_start_px is not None:
            self.clear_bcp_manual_split()
            self.status_var.set("已清除手动分割线。")
            return "break"
        if self.bcp_completion_mode or self.bcp_completion_polygon_px or self.bcp_completion_polygons_px:
            self.clear_bcp_completion_selection()
            self.status_var.set("已清除局部补漏框选。")
            return "break"
        if self.active_annotation_index is not None:
            return self.delete_active_annotation()
        return self.clear_roi() if self.ler_lwr_roi_mode else "break"

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
        self.zoom_percent_var.set("100")

    def step_zoom(self, change: float) -> None:
        self.apply_zoom(self.zoom_factor + change)

    def zoom_from_slider(self, value: str) -> None:
        self.apply_zoom(float(value))

    def apply_zoom_percent_value(self, _event: tk.Event | None = None) -> str:
        text = self.zoom_percent_var.get().strip().removesuffix("%")
        try:
            percent = float(text)
        except ValueError:
            self.zoom_percent_var.set(f"{self.zoom_factor * 100:.0f}")
            self.status_var.set("缩放请输入 25–300 之间的百分比。")
            return "break"
        requested_zoom = percent / 100.0
        if not ZOOM_MIN <= requested_zoom <= ZOOM_MAX:
            self.zoom_percent_var.set(f"{self.zoom_factor * 100:.0f}")
            self.status_var.set(f"缩放范围为 {ZOOM_MIN * 100:.0f}%–{ZOOM_MAX * 100:.0f}%。")
            return "break"
        self.apply_zoom(requested_zoom)
        self.zoom_percent_var.set(f"{self.zoom_factor * 100:.0f}")
        return "break"

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
        self.zoom_percent_var.set(f"{new_zoom * 100:.0f}")
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

    def scroll_active_window(self, event: tk.Event) -> str | None:
        top_level = event.widget.winfo_toplevel()
        if self.bcp_dialog is not None and top_level == self.bcp_dialog and self.bcp_dialog_scroll is not None:
            return self.bcp_dialog_scroll.scroll_wheel(event)
        if top_level == self:
            return self.main_scroll.scroll_wheel(event)
        return None

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
        if self.era_pending_roi_target is not None:
            self.store_pending_era_roi()
            self.roi_was_changed = False
            return
        if self.roi_was_changed:
            self.result = None
            self.analysis_origin = None
            self.clear_auto_line_candidates(redraw=False)
            self.canvas.delete("edge")
            self.result_var.set("ROI 已更新；请重新点击“开始识别 / 分析”。")
        self.status_var.set("ROI 已确定；下一次识别 / 分析将只使用框内图像。")
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
        self.clear_auto_line_candidates(redraw=False)
        self.canvas.delete("roi")
        self.canvas.delete("edge")
        self.result_var.set("ROI 已清除；下一次识别 / 分析将使用整张图像。")
        self.status_var.set("已清除 ROI；将使用整图。")
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
        analysis_image = self.processed_image
        if analysis_image is None:
            raise ValueError("图像未准备完成。")
        return analysis_image[top:bottom, left:right], left, top

    def selected_ler_lwr_region(self) -> tuple[np.ndarray, int, int, bool]:
        """Return the selected LER/LWR image region, or the full image without an ROI."""
        if self.processed_image is None:
            raise ValueError("图像未准备完成。")
        if self.roi_canvas is None:
            return self.processed_image, 0, 0, False
        roi, left, top = self.selected_roi()
        return roi, left, top, True

    def era_polynomial_degree(self) -> int:
        try:
            degree = int(self.era_polynomial_degree_var.get())
        except ValueError as exc:
            raise ValueError("单边拟合次数必须为 2、3 或 4。") from exc
        if degree not in (2, 3, 4):
            raise ValueError("单边拟合次数必须为 2、3 或 4。")
        return degree

    def era_roi_bounds_from_canvas(self) -> tuple[int, int, int, int]:
        if self.raw_image is None or self.roi_canvas is None:
            raise ValueError("请在主图上拖动框选单条边的 ROI。")
        x0, y0, x1, y1 = self.roi_canvas
        left = max(0, math.floor(x0 / self.display_scale))
        top = max(0, math.floor(y0 / self.display_scale))
        right = min(self.raw_image.shape[1], math.ceil(x1 / self.display_scale))
        bottom = min(self.raw_image.shape[0], math.ceil(y1 / self.display_scale))
        if right - left < MIN_ERA_ROI_WIDTH or bottom - top < MIN_ERA_ROI_HEIGHT:
            raise ValueError("单边 ROI 过小；请至少框选约 12 px 宽和 80 px 高。")
        return left, top, right, bottom

    def clear_era_line_samples(self, redraw: bool = True) -> None:
        self.era_line_samples.clear()
        self.era_aggregate_result = None
        self.era_lcdu_cd_samples_nm.clear()
        self.era_active_sample_index = None
        self.era_pending_roi_target = None
        if redraw and self.raw_image is not None:
            self.draw_era_rois()
        self.refresh_era_sample_list()
        self.update_lcdu_summary_text()

    def invalidate_era_results(self) -> None:
        """Hide stale ERA traces whenever its user-defined sample set changes."""
        self.era_aggregate_result = None
        self.era_lcdu_cd_samples_nm.clear()
        self.result = None
        self.analysis_origin = None
        self.canvas.delete("edge")
        self.update_lcdu_summary_text()

    def new_era_line_sample(self) -> None:
        self.era_line_samples.append(EraLineSample())
        self.era_active_sample_index = len(self.era_line_samples) - 1
        self.invalidate_era_results()
        self.refresh_era_sample_list()
        self.status_var.set(f"已新建线样本 {len(self.era_line_samples)}；请分别框选左、右两条单边 ROI。")

    def selected_era_line_sample(self) -> EraLineSample:
        if self.era_active_sample_index is None or not (0 <= self.era_active_sample_index < len(self.era_line_samples)):
            self.new_era_line_sample()
        return self.era_line_samples[self.era_active_sample_index]

    def select_era_line_sample(self, _event: tk.Event | None = None) -> None:
        if self.era_sample_listbox is None:
            return
        selected = self.era_sample_listbox.curselection()
        if selected:
            self.era_active_sample_index = selected[0]

    def refresh_era_sample_list(self) -> None:
        if self.era_sample_listbox is None:
            return
        selected_index = self.era_active_sample_index
        self.era_sample_listbox.delete(0, tk.END)
        for index, sample in enumerate(self.era_line_samples, start=1):
            left = "左边已选" if sample.left_roi_bounds_px is not None else "左边未选"
            right = "右边已选" if sample.right_roi_bounds_px is not None else "右边未选"
            measured = "已计算" if sample.paired_result is not None else "待计算"
            self.era_sample_listbox.insert(tk.END, f"线 {index}: {left}，{right}，{measured}")
        if selected_index is not None and self.era_line_samples:
            self.era_sample_listbox.selection_set(selected_index)
            self.era_sample_listbox.activate(selected_index)

    def delete_selected_era_line_sample(self) -> None:
        if self.era_active_sample_index is None or not self.era_line_samples:
            self.status_var.set("请先选择一个单边线样本。")
            return
        removed_index = self.era_active_sample_index
        self.era_line_samples.pop(removed_index)
        self.era_active_sample_index = min(removed_index, len(self.era_line_samples) - 1) if self.era_line_samples else None
        self.invalidate_era_results()
        self.refresh_era_sample_list()
        self.draw_era_rois()
        self.status_var.set("已删除所选单边线样本。")

    def start_era_roi_selection(self, side: str) -> None:
        if self.raw_image is None:
            return
        self.selected_era_line_sample()
        if side not in ("left", "right"):
            raise ValueError("单边 ROI 只能指定为左边或右边。")
        self.era_pending_roi_target = (self.era_active_sample_index, side)
        self.roi_canvas = None
        self.ler_lwr_roi_mode = True
        self.active_annotation_index = None
        self.draw_roi()
        self.canvas.configure(cursor="crosshair")
        side_name = "左边" if side == "left" else "右边"
        self.status_var.set(f"请在主图拖动框选线样本 {self.era_active_sample_index + 1} 的{side_name}单边 ROI。")

    def store_pending_era_roi(self) -> bool:
        if self.era_pending_roi_target is None:
            return False
        sample_index, side = self.era_pending_roi_target
        try:
            bounds = self.era_roi_bounds_from_canvas()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return False
        sample = self.era_line_samples[sample_index]
        if side == "left":
            sample.left_roi_bounds_px = bounds
            sample.left_edge = None
        else:
            sample.right_roi_bounds_px = bounds
            sample.right_edge = None
        sample.paired_result = None
        self.invalidate_era_results()
        self.era_pending_roi_target = None
        self.roi_canvas = None
        self.roi_rectangle = None
        self.refresh_era_sample_list()
        self.draw_roi()
        side_name = "左边" if side == "left" else "右边"
        self.status_var.set(f"线样本 {sample_index + 1} 的{side_name}单边 ROI 已保存。")
        return True

    def run_era_analysis(self) -> None:
        if self.processed_image is None:
            raise ValueError("图像未准备完成。")
        pixel_size = float(self.pixel_size_var.get())
        degree = self.era_polynomial_degree()
        complete_results = []
        for index, sample in enumerate(self.era_line_samples, start=1):
            if sample.left_roi_bounds_px is not None:
                sample.left_edge = analyze_single_edge_era(
                    self.processed_image,
                    sample.left_roi_bounds_px,
                    pixel_size,
                    degree,
                    self.era_polarity_var.get(),
                    self.outlier_threshold_px(),
                )
            if sample.right_roi_bounds_px is not None:
                sample.right_edge = analyze_single_edge_era(
                    self.processed_image,
                    sample.right_roi_bounds_px,
                    pixel_size,
                    degree,
                    self.era_polarity_var.get(),
                    self.outlier_threshold_px(),
                )
            sample.paired_result = None
            if sample.left_edge is not None and sample.right_edge is not None:
                sample.paired_result = pair_single_edges(sample.left_edge, sample.right_edge, pixel_size)
                complete_results.append(sample.paired_result)
            elif sample.left_roi_bounds_px is not None or sample.right_roi_bounds_px is not None:
                self.status_var.set(f"线样本 {index} 只有一条边；已输出其 LER，但未参与 CD/LWR 汇总。")
        if not complete_results:
            raise ValueError("至少需要一个线样本的左、右单边 ROI 都已框选，才能计算 CD/LWR。")
        self.era_aggregate_result = aggregate_era_line_results(complete_results)
        self.era_lcdu_cd_samples_nm = [result.mean_width_nm for result in complete_results]
        self.result = complete_results[-1]
        self.analysis_origin = (0, 0)
        self.clear_auto_line_candidates(redraw=False)
        self.refresh_era_sample_list()
        self.update_lcdu_summary_text()
        self.update_era_result_text()

    def update_era_result_text(self) -> None:
        if self.era_aggregate_result is None:
            return
        summary = self.era_aggregate_result
        multiplier = self.sigma_multiplier_var.get()
        lines = [
            f"单边测量（ERA，{self.era_polynomial_degree()} 次拟合）",
            f"输出：{multiplier:.1f}σ（主窗口“显示/导出倍数”）",
            f"已完成线样本  {summary.line_count} 根；共同有效点 {summary.point_count}",
            f"平均 CD        {summary.mean_width_nm:.3f} nm",
            f"左 ER（LER_L）  {multiplier * summary.ler_left_sigma_nm:.3f} nm",
            f"右 ER（LER_R）  {multiplier * summary.ler_right_sigma_nm:.3f} nm",
            f"宽度 LWR        {multiplier * summary.lwr_sigma_nm:.3f} nm",
            f"原始 σ（L/R/W） {summary.ler_left_sigma_nm:.3f} / {summary.ler_right_sigma_nm:.3f} / {summary.lwr_sigma_nm:.3f} nm",
        ]
        lcdu_sigma = self.lcdu_sigma_nm()
        if lcdu_sigma is not None:
            lines.append(f"LCDU（{len(self.era_lcdu_cd_samples_nm)} 根） {multiplier * lcdu_sigma:.3f} nm")
        else:
            lines.append("LCDU             还需至少 2 根完整线样本")
        lines.extend(("", "逐线结果："))
        for index, sample in enumerate(self.era_line_samples, start=1):
            if sample.paired_result is not None:
                result = sample.paired_result
                lines.append(
                    f"线 {index}: CD {result.mean_width_nm:.3f} nm，"
                    f"LER_L/R {multiplier * result.ler_left_sigma_nm:.3f}/{multiplier * result.ler_right_sigma_nm:.3f} nm，"
                    f"LWR {multiplier * result.lwr_sigma_nm:.3f} nm"
                )
            elif sample.left_edge is not None or sample.right_edge is not None:
                edge = sample.left_edge or sample.right_edge
                lines.append(f"线 {index}: 单边 LER {multiplier * edge.ler_sigma_nm:.3f} nm（未配对）")
        self.result_var.set("\n".join(lines))

    def active_lcdu_samples(self) -> tuple[list[float], str]:
        if self.era_aggregate_result is not None:
            return self.era_lcdu_cd_samples_nm, "ERA 自动样本"
        return self.lcdu_cd_samples_nm, "手动样本"

    def lcdu_sigma_nm(self) -> float | None:
        samples_nm, _source = self.active_lcdu_samples()
        sample_count = len(samples_nm)
        if sample_count < 2:
            return None
        samples = np.array(samples_nm, dtype=float)
        residual = samples - np.mean(samples)
        return math.sqrt(float(np.dot(residual, residual)) / (sample_count - 1))

    def update_lcdu_summary_text(self) -> None:
        self.refresh_lcdu_sample_list()
        samples_nm, source = self.active_lcdu_samples()
        sample_count = len(samples_nm)
        sigma = self.lcdu_sigma_nm()
        if sigma is None:
            self.lcdu_summary_var.set(f"LCDU（{source}）：{sample_count} 条线；至少需要 2 条线。")
            return
        multiplier = self.sigma_multiplier_var.get()
        self.lcdu_summary_var.set(
            f"LCDU（{source}）：{sample_count} 条线\n"
            f"LCDU：{multiplier * sigma:.3f} nm  ({multiplier:.1f}σ)，原始 σ={sigma:.3f} nm"
        )

    def refresh_lcdu_sample_list(self) -> None:
        if self.lcdu_sample_listbox is None:
            return
        samples_nm, _source = self.active_lcdu_samples()
        selected = self.lcdu_sample_listbox.curselection()
        selected_index = selected[0] if selected else None
        self.lcdu_sample_listbox.delete(0, tk.END)
        for index, mean_cd_nm in enumerate(samples_nm, start=1):
            self.lcdu_sample_listbox.insert(tk.END, f"{index:02d}. CD = {mean_cd_nm:.3f} nm")
        if selected_index is not None and samples_nm:
            next_index = min(selected_index, len(samples_nm) - 1)
            self.lcdu_sample_listbox.selection_set(next_index)
            self.lcdu_sample_listbox.activate(next_index)

    def add_lcdu_sample(self) -> None:
        if self.era_aggregate_result is not None:
            self.status_var.set("当前 ERA 的完整线样本已自动用于 LCDU，无需手动加入。")
            return
        if self.result is None:
            messagebox.showinfo("暂无 CD", "请先分析一条线，再加入 LCDU 样本。")
            return
        self.lcdu_cd_samples_nm.append(self.result.mean_width_nm)
        self.update_lcdu_summary_text()
        self.status_var.set(f"已加入当前线 CD={self.result.mean_width_nm:.3f} nm；LCDU 样本数 {len(self.lcdu_cd_samples_nm)}。")

    def clear_lcdu_samples(self) -> None:
        if self.era_aggregate_result is not None:
            self.status_var.set("当前显示的是 ERA 自动 LCDU；修改 ROI 或删除线样本后会自动更新。")
            return
        self.lcdu_cd_samples_nm.clear()
        self.update_lcdu_summary_text()
        self.status_var.set("已清空 LCDU 样本。")

    def delete_selected_lcdu_sample(self, _event: tk.Event | None = None) -> str:
        if self.era_aggregate_result is not None:
            self.status_var.set("当前显示的是 ERA 自动 LCDU；请修改对应的线样本或 ROI。")
            return "break"
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
            detector, diagonal, axial, kernel_size, high_threshold, threshold_ratio = self.edge_detector_parameters()
            started_at = time.perf_counter()
            if detector == EDGE_DETECTOR_ERA:
                self.run_era_analysis()
                self.recognition_duration_var.set(f"识别耗时：{time.perf_counter() - started_at:.2f} 秒")
                self.bcp_result = None
                self.clear_bcp_completion_selection(redraw=False)
                self.draw_image()
                self.status_var.set("单边 ERA 分析完成；左右边仅按你保存到同一线样本的 ROI 配对。")
                return
            analysis_image, left_offset, top_offset, uses_roi = self.selected_ler_lwr_region()
            region_name = "ROI 内" if uses_roi else "整图"
            if detector == EDGE_DETECTOR_LAPLACIAN:
                pixel_size = float(self.pixel_size_var.get())
                self.clear_auto_line_candidates(redraw=False)
                self.result = analyze_roi(
                    analysis_image,
                    pixel_size,
                    self.outlier_threshold_px(),
                    detector,
                    diagonal,
                    axial,
                    high_threshold,
                    threshold_ratio,
                    kernel_size=kernel_size,
                )
                self.analysis_origin = (left_offset, top_offset)
            else:
                self.result = None
                self.analysis_origin = None
                self.clear_detected_edges(redraw=False)
                self.detected_edge_mask = detect_full_image_edges(
                    analysis_image,
                    detector,
                    diagonal,
                    axial,
                    high_threshold,
                    threshold_ratio,
                    self.edge_use_vertical_continuity_filter_var.get(),
                    kernel_size=kernel_size,
                )
                self.detected_edge_origin = (left_offset, top_offset)
                edge_count = int(np.count_nonzero(self.detected_edge_mask))
                if edge_count == 0:
                    raise ValueError("未识别到边缘；可调整检测器参数或图像处理设置。")
                continuity_note = "已启用局部连续性筛除。" if self.edge_use_vertical_continuity_filter_var.get() else "未启用局部连续性筛除。"
                self.result_var.set(
                    f"{region_name}边缘识别：{edge_count} 个边缘像素\n"
                    f"{continuity_note}\n\n"
                    "当前只显示检测到的边缘，未配对为线条，\n"
                    "因此尚未计算 CD、LER 或 LWR。"
                )
                self.status_var.set(f"{region_name}边缘识别完成：已显示 {edge_count} 个边缘像素；{continuity_note}")
            self.recognition_duration_var.set(f"识别耗时：{time.perf_counter() - started_at:.2f} 秒")
            self.bcp_result = None
            self.clear_bcp_completion_selection(redraw=False)
            self.draw_image()
            if detector == EDGE_DETECTOR_LAPLACIAN:
                self.update_result_text()
                self.status_var.set(f"拉普拉斯{region_name}分析完成。可调整 σ 倍数，或导出轨迹和结果。")
        except ValueError as exc:
            messagebox.showwarning("无法分析", str(exc))

    def open_ler_lwr_dialog(self) -> None:
        if self.raw_image is None:
            messagebox.showinfo("暂无图像", "请先打开一张线条 SEM 图像。")
            return
        if self.ler_lwr_dialog is not None and self.ler_lwr_dialog.winfo_exists():
            self.close_ler_lwr_dialog()
            return
        self.ler_lwr_dialog = tk.Toplevel(self)
        self.ler_lwr_dialog.title("LER / LWR 测量")
        dialog_height = min(700, max(460, self.winfo_screenheight() - 180))
        self.ler_lwr_dialog.geometry(f"480x{dialog_height}")
        self.ler_lwr_dialog.minsize(400, 420)
        self.ler_lwr_dialog.transient(self)
        self.ler_lwr_dialog.attributes("-topmost", True)
        self.ler_lwr_dialog.lift(self)
        self.ler_lwr_dialog.protocol("WM_DELETE_WINDOW", self.close_ler_lwr_dialog)
        self.ler_lwr_dialog_scroll = VerticalScrollFrame(self.ler_lwr_dialog, padding=16)
        self.ler_lwr_dialog_scroll.pack(fill=tk.BOTH, expand=True)
        content = self.ler_lwr_dialog_scroll.content
        ttk.Label(content, text="LER / LWR 测量", font=("Helvetica", 15, "bold")).pack(anchor="w")
        ttk.Label(content, text="Scharr/Canny 只显示未配对边缘；拉普拉斯使用一个线 ROI。单边测量（ERA）则由你分别框选左右单边 ROI，绝不自动找边或配对。", wraplength=410).pack(anchor="w", pady=(7, 10))

        roi_controls = ttk.LabelFrame(content, text="识别区域（可选 ROI）", padding=10)
        roi_controls.pack(fill=tk.X)
        ttk.Label(roi_controls, text="点击框选后回到主图拖动绿色框。保留框时，三种检测器都只处理框内；清除框后，三种检测器都处理整图。", wraplength=390).pack(anchor="w")
        roi_buttons = ttk.Frame(roi_controls)
        roi_buttons.pack(anchor="w", pady=(8, 0))
        ttk.Button(roi_buttons, text="框选 / 调整 ROI", command=self.start_ler_lwr_roi_selection).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(roi_buttons, text="清除 ROI", command=self.clear_roi).pack(side=tk.LEFT)

        detector_controls = ttk.LabelFrame(content, text="边缘检测", padding=10)
        detector_controls.pack(fill=tk.X, pady=(10, 0))
        detector_row = ttk.Frame(detector_controls)
        detector_row.pack(anchor="w")
        ttk.Label(detector_row, text="检测器：").pack(side=tk.LEFT)
        detector_selector = ttk.Combobox(
            detector_row,
            textvariable=self.edge_detector_var,
            state="readonly",
            values=(EDGE_DETECTOR_SCHARR, EDGE_DETECTOR_CANNY, EDGE_DETECTOR_LAPLACIAN, EDGE_DETECTOR_ERA),
            width=16,
        )
        detector_selector.pack(side=tk.LEFT)
        detector_selector.bind("<<ComboboxSelected>>", self.change_edge_detector)
        era_controls = ttk.LabelFrame(content, text="单边测量（ERA）", padding=10)
        era_controls.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(era_controls, text="输入为当前图像处理后的图像；此算法不额外做 5×7 或任何内部空间滤波。每个 ROI 仅覆盖一条边，只有同一线样本的左右两边才计算 CD/LWR。", wraplength=390).pack(anchor="w")
        era_options = ttk.Frame(era_controls)
        era_options.pack(anchor="w", pady=(8, 0))
        ttk.Label(era_options, text="边缘方向：").pack(side=tk.LEFT)
        ttk.Combobox(era_options, textvariable=self.era_polarity_var, state="readonly", values=ERA_POLARITIES, width=15).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(era_options, text="拟合次数：").pack(side=tk.LEFT)
        ttk.Combobox(era_options, textvariable=self.era_polynomial_degree_var, state="readonly", values=("2", "3", "4"), width=3).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(era_controls, text="默认三次。四次是可选扩展：仅从拟合区内、方向正确且最接近导数重心的 0.5 交点中取一个。", foreground="#666666", wraplength=390).pack(anchor="w", pady=(5, 0))
        era_buttons = ttk.Frame(era_controls)
        era_buttons.pack(anchor="w", pady=(8, 0))
        ttk.Button(era_buttons, text="新建线样本", command=self.new_era_line_sample).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(era_buttons, text="框选左边 ROI", command=lambda: self.start_era_roi_selection("left")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(era_buttons, text="框选右边 ROI", command=lambda: self.start_era_roi_selection("right")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(era_buttons, text="删除线样本", command=self.delete_selected_era_line_sample).pack(side=tk.LEFT)
        self.era_sample_listbox = tk.Listbox(era_controls, height=4, exportselection=False)
        self.era_sample_listbox.pack(fill=tk.X, pady=(8, 0))
        self.era_sample_listbox.bind("<<ListboxSelect>>", self.select_era_line_sample)
        self.refresh_era_sample_list()
        kernel_size_row = ttk.Frame(detector_controls)
        kernel_size_row.pack(anchor="w", pady=(7, 0))
        ttk.Label(kernel_size_row, text="一阶核尺寸：").pack(side=tk.LEFT)
        kernel_size_selector = ttk.Combobox(
            kernel_size_row,
            textvariable=self.edge_kernel_size_var,
            state="readonly",
            values=tuple(f"{size} × {size}" for size in FIRST_ORDER_KERNEL_SIZES),
            width=8,
        )
        kernel_size_selector.pack(side=tk.LEFT)
        kernel_size_selector.bind("<<ComboboxSelected>>", self.change_edge_kernel_size)
        weight_row = ttk.Frame(detector_controls)
        weight_row.pack(anchor="w", pady=(7, 0))
        ttk.Label(weight_row, text="斜向权重：").pack(side=tk.LEFT)
        diagonal_entry = ttk.Entry(weight_row, textvariable=self.edge_diagonal_weight_var, width=6)
        diagonal_entry.pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(weight_row, text="轴向权重：").pack(side=tk.LEFT)
        axial_entry = ttk.Entry(weight_row, textvariable=self.edge_axial_weight_var, width=6)
        axial_entry.pack(side=tk.LEFT, padx=(2, 0))
        canny_row = ttk.Frame(detector_controls)
        canny_row.pack(fill=tk.X, pady=(7, 0))
        ttk.Label(canny_row, text="Canny highTh：").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            canny_row,
            from_=0.01,
            to=0.30,
            orient=tk.HORIZONTAL,
            variable=self.canny_high_threshold_var,
            command=lambda _value: self.update_canny_thresholds(),
            length=150,
        ).grid(row=0, column=1, padx=(5, 6))
        ttk.Label(canny_row, textvariable=self.canny_high_threshold_text_var, width=5).grid(row=0, column=2, sticky="w")
        ttk.Label(canny_row, text="Canny Rio：").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Scale(
            canny_row,
            from_=2.0,
            to=3.0,
            orient=tk.HORIZONTAL,
            variable=self.canny_threshold_ratio_var,
            command=lambda _value: self.update_canny_thresholds(),
            length=150,
        ).grid(row=1, column=1, padx=(5, 6), pady=(4, 0))
        ttk.Label(canny_row, textvariable=self.canny_threshold_ratio_text_var, width=5).grid(row=1, column=2, sticky="w", pady=(4, 0))
        ttk.Label(canny_row, text="lowTh（自动）：").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(canny_row, textvariable=self.canny_low_threshold_text_var, width=5).grid(row=2, column=1, sticky="w", padx=(5, 0), pady=(4, 0))
        ttk.Checkbutton(
            detector_controls,
            text="去除孤立短噪声（可选）",
            variable=self.edge_use_vertical_continuity_filter_var,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            detector_controls,
            text="关闭：直接显示 Scharr/Canny 边缘。开启：仅保留在 41 px 高 × 3 px 宽区域内有至少 24 个边缘点的像素。",
            foreground="#666666",
            wraplength=390,
        ).pack(anchor="w", pady=(2, 0))
        for entry in (diagonal_entry, axial_entry):
            entry.bind("<FocusOut>", self.update_edge_kernel_preview)
            entry.bind("<Return>", self.update_edge_kernel_preview)
        ttk.Label(detector_controls, textvariable=self.edge_kernel_preview_var, justify=tk.LEFT, font=("Menlo", 9)).pack(anchor="w", pady=(8, 0))
        ttk.Label(detector_controls, text="3×3 保持原 Scharr/Sobel 核；5×5 以上在求梯度时扩大双项式平滑支撑，定位与 NMS 流程不变。Canny 的 lowTh 自动等于 highTh ÷ Rio，并对 NMS 响应进行 P99 鲁棒归一化后执行双阈值连接。", foreground="#666666", wraplength=390).pack(anchor="w", pady=(5, 0))

        alignment = ttk.LabelFrame(content, text="倾角校正", padding=10)
        alignment.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(alignment, text="自动校正仅针对拉普拉斯 ROI。Scharr/Canny 的边缘识别保留方向性；若图像整体倾斜，可使用主界面底部的手动旋转后重新识别。", wraplength=390).pack(anchor="w")
        alignment_buttons = ttk.Frame(alignment)
        alignment_buttons.pack(anchor="w", pady=(8, 0))
        ttk.Button(alignment_buttons, text="自动校正 ROI 倾角", command=self.auto_align_roi).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(alignment_buttons, text="重置角度", command=self.reset_rotation).pack(side=tk.LEFT)

        analysis = ttk.LabelFrame(content, text="执行分析", padding=10)
        analysis.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(analysis, text="开始识别 / 分析", command=self.run_analysis).pack(anchor="w")
        ttk.Label(analysis, text="结果会显示在主窗口右侧；ERA 有至少 2 根完整线样本时会自动计算 LCDU。", foreground="#666666", wraplength=390).pack(anchor="w", pady=(6, 0))

        self.update_edge_kernel_preview()
        if self.result is not None:
            self.update_result_text()
        else:
            self.result_var.set("可选框选 ROI：有框识别框内，无框识别整图。Scharr/Canny 仅显示边缘；拉普拉斯计算 LER/LWR。")

    def keep_ler_lwr_dialog_above_main(self, _event: tk.Event | None = None) -> None:
        """Keep the non-modal measurement controls visible while the main canvas has focus."""
        dialog = self.ler_lwr_dialog
        if dialog is not None and dialog.winfo_exists():
            dialog.lift(self)

    def close_ler_lwr_dialog(self) -> None:
        self.ler_lwr_roi_mode = False
        if self.ler_lwr_dialog is not None and self.ler_lwr_dialog.winfo_exists():
            self.ler_lwr_dialog.destroy()
        self.ler_lwr_dialog = None
        self.ler_lwr_dialog_scroll = None
        self.era_sample_listbox = None

    def start_ler_lwr_roi_selection(self) -> None:
        self.ler_lwr_roi_mode = True
        self.active_annotation_index = None
        self.status_var.set("LER/LWR ROI 框选已启用：请在主图像上拖动选择识别区域。")
        self.canvas.configure(cursor="crosshair")

    def change_edge_detector(self, _event: tk.Event | None = None) -> None:
        defaults = {
            EDGE_DETECTOR_SCHARR: ("3", "10"),
            EDGE_DETECTOR_CANNY: ("1", "2"),
            EDGE_DETECTOR_LAPLACIAN: ("0", "1"),
            EDGE_DETECTOR_ERA: ("0", "1"),
        }
        diagonal, axial = defaults[self.edge_detector_var.get()]
        self.edge_diagonal_weight_var.set(diagonal)
        self.edge_axial_weight_var.set(axial)
        self.update_edge_kernel_preview()
        self.result = None
        self.analysis_origin = None
        self.clear_auto_line_candidates(redraw=False)
        self.draw_image()

    def edge_kernel_size(self) -> int:
        return int(self.edge_kernel_size_var.get().split()[0])

    def change_edge_kernel_size(self, _event: tk.Event | None = None) -> None:
        self.update_edge_kernel_preview()
        self.result = None
        self.analysis_origin = None
        self.clear_auto_line_candidates(redraw=False)
        self.draw_image()

    def update_canny_thresholds(self) -> None:
        high = self.canny_high_threshold_var.get()
        ratio = self.canny_threshold_ratio_var.get()
        low = high / ratio
        self.canny_high_threshold_text_var.set(f"{high:.3f}")
        self.canny_threshold_ratio_text_var.set(f"{ratio:.2f}")
        self.canny_low_threshold_text_var.set(f"{low:.3f}")
        self.update_edge_kernel_preview()

    def edge_detector_parameters(self) -> tuple[str, float, float, int, float, float]:
        try:
            diagonal = float(self.edge_diagonal_weight_var.get())
            axial = float(self.edge_axial_weight_var.get())
            high = float(self.canny_high_threshold_var.get())
            ratio = float(self.canny_threshold_ratio_var.get())
        except ValueError as exc:
            raise ValueError("边缘检测权重、Canny highTh 和 Rio 必须为数字。") from exc
        if diagonal < 0 or axial <= 0:
            raise ValueError("卷积核权重必须满足：斜向权重 ≥ 0，轴向权重 > 0。")
        kernel_size = self.edge_kernel_size()
        if kernel_size not in FIRST_ORDER_KERNEL_SIZES:
            raise ValueError("一阶卷积核尺寸无效。")
        if not 0 < high <= 1 or ratio <= 1:
            raise ValueError("Canny 参数必须满足：0 < highTh ≤ 1，Rio > 1。")
        return self.edge_detector_var.get(), diagonal, axial, kernel_size, high, ratio

    def update_edge_kernel_preview(self, _event: tk.Event | None = None) -> None:
        try:
            detector, diagonal, axial, kernel_size, high, ratio = self.edge_detector_parameters()
        except ValueError:
            self.edge_kernel_preview_var.set("请输入有效的卷积核权重与 Canny 阈值。")
            return
        if detector == EDGE_DETECTOR_LAPLACIAN:
            center = -4 * axial - 4 * diagonal
            rows = ((diagonal, axial, diagonal), (axial, center, axial), (diagonal, axial, diagonal))
            self.edge_kernel_preview_var.set("拉普拉斯核\n" + "\n".join("  ".join(f"{value:g}" for value in row) for row in rows))
            return
        if detector == EDGE_DETECTOR_ERA:
            self.edge_kernel_preview_var.set(
                "ERA 单边拟合\n"
                "不使用 Scharr/Sobel/Laplacian 卷积核，也不执行额外空间滤波。\n"
                "每行在用户框选的单边 ROI 内：导数重心 → 多项式拟合 → 0.5 交点。"
            )
            return
        kernel_x, kernel_y = first_order_kernels(detector, diagonal, axial, kernel_size)
        label = "Scharr" if detector == EDGE_DETECTOR_SCHARR else "Canny 的 Sobel"
        format_kernel = lambda kernel: "\n".join("  ".join(f"{value:g}" for value in row) for row in kernel)
        threshold_note = f"\nhighTh / Rio / lowTh：{high:.3f} / {ratio:.2f} / {high / ratio:.3f}" if detector == EDGE_DETECTOR_CANNY else ""
        self.edge_kernel_preview_var.set(
            f"{label} {kernel_size}×{kernel_size} Gx\n{format_kernel(kernel_x)}\nGy\n{format_kernel(kernel_y)}"
            f"{threshold_note}\nROI / 整图识别：无角度限制；短噪声筛除可选"
        )

    def build_lcdu_controls(self, parent: tk.Misc) -> None:
        ttk.Label(parent, textvariable=self.lcdu_summary_var, justify=tk.LEFT).pack(anchor="w", fill=tk.X)
        buttons = ttk.Frame(parent)
        buttons.pack(anchor="w", pady=(6, 0))
        ttk.Button(buttons, text="加入当前线 CD", command=self.add_lcdu_sample).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="删除选中样本", command=self.delete_selected_lcdu_sample).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="清空 LCDU 样本", command=self.clear_lcdu_samples).pack(side=tk.LEFT)
        self.lcdu_sample_listbox = tk.Listbox(parent, height=5, exportselection=False)
        self.lcdu_sample_listbox.pack(fill=tk.X, pady=(8, 0))
        self.lcdu_sample_listbox.bind("<Delete>", self.delete_selected_lcdu_sample)
        self.lcdu_sample_listbox.bind("<BackSpace>", self.delete_selected_lcdu_sample)
        self.update_lcdu_summary_text()

    def open_bcp_recognition_dialog(self) -> None:
        if self.raw_image is None:
            messagebox.showinfo("暂无图像", "请先打开一张俯视 BCP SEM 图像。")
            return
        if self.bcp_dialog is not None and self.bcp_dialog.winfo_exists():
            self.close_bcp_recognition_dialog()
            return
        self.bcp_dialog = tk.Toplevel(self)
        self.bcp_dialog.title("BCP 点阵识别")
        dialog_height = min(760, max(500, self.winfo_screenheight() - 160))
        self.bcp_dialog.geometry(f"500x{dialog_height}")
        self.bcp_dialog.minsize(420, 420)
        self.bcp_dialog.protocol("WM_DELETE_WINDOW", self.close_bcp_recognition_dialog)
        self.bcp_dialog_scroll = VerticalScrollFrame(self.bcp_dialog, padding=16)
        self.bcp_dialog_scroll.pack(fill=tk.BOTH, expand=True)
        content = self.bcp_dialog_scroll.content
        if self.bcp_result is not None:
            self.update_bcp_result_text()
        else:
            self.result_var.set("请设置样本或圆柱大致直径，然后点击“开始识别”。")
        ttk.Label(content, text="BCP 点阵识别", font=("Helvetica", 15, "bold")).pack(anchor="w")
        ttk.Label(content, text="自动识别可使用 DoG 或原始亮暗强度，再经局部分割、二值连通区域和轮廓描绘完成；样本和局部分割均为可选辅助。", wraplength=430).pack(anchor="w", pady=(8, 8))

        assistance = ttk.LabelFrame(content, text="样本与局部分割（可选辅助）", padding=10)
        assistance.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            assistance,
            text="启用局部分割（各图块独立计算当前响应阈值）",
            variable=self.bcp_use_local_segmentation_var,
        ).pack(anchor="w")
        tile_controls = ttk.Frame(assistance)
        tile_controls.pack(anchor="w", pady=(5, 0))
        ttk.Label(tile_controls, text="横向分块：").pack(side=tk.LEFT)
        ttk.Entry(tile_controls, textvariable=self.bcp_horizontal_sections_var, width=5).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(tile_controls, text="竖向分块：").pack(side=tk.LEFT)
        ttk.Entry(tile_controls, textvariable=self.bcp_vertical_sections_var, width=5).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(tile_controls, text="份（默认 4 × 4）").pack(side=tk.LEFT)
        area_controls = ttk.Frame(assistance)
        area_controls.pack(anchor="w", pady=(5, 0))
        ttk.Label(area_controls, text="最小保留面积：").pack(side=tk.LEFT)
        ttk.Entry(area_controls, textvariable=self.bcp_min_area_fraction_var, width=6).pack(side=tk.LEFT, padx=(2, 5))
        ttk.Label(area_controls, text="× 样本圈/柱径面积").pack(side=tk.LEFT)
        hole_controls = ttk.Frame(assistance)
        hole_controls.pack(anchor="w", pady=(5, 0))
        ttk.Label(hole_controls, text="柱内空洞忽略上限：").pack(side=tk.LEFT)
        ttk.Entry(hole_controls, textvariable=self.bcp_internal_hole_fraction_var, width=6).pack(side=tk.LEFT, padx=(2, 5))
        ttk.Label(hole_controls, text="× 样本圈/柱径面积（0 = 关闭）").pack(side=tk.LEFT)
        ttk.Label(
            assistance,
            text="样本用于亮暗极性、尺度、面积参数和尺寸校准；局部分割可应对亮度不均，关闭后使用整图阈值。柱内小暗斑会被忽略，不绘制内轮廓。",
            foreground="#666666",
            wraplength=430,
        ).pack(anchor="w", pady=(3, 0))
        ttk.Label(assistance, text="建议标示 3 个以上清晰、孤立的样本；拖蓝圈内部可移动，拖外缘可改大小。", wraplength=430).pack(anchor="w", pady=(8, 2))
        ttk.Label(assistance, textvariable=self.bcp_reference_var, foreground="#2369b0").pack(anchor="w", pady=(0, 5))
        buttons = ttk.Frame(assistance)
        buttons.pack(anchor="w")
        ttk.Button(buttons, text="标示/编辑样本", command=self.start_bcp_reference_marking).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="结束编辑", command=self.stop_bcp_reference_marking).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="清除样本", command=self.clear_bcp_reference_circles).pack(side=tk.LEFT)
        ttk.Checkbutton(
            assistance,
            text="显示标准样本圈（画面/导出）",
            variable=self.bcp_reference_display_var,
            command=self.refresh_bcp_reference_display,
        ).pack(anchor="w", pady=(6, 0))

        recognition = ttk.LabelFrame(content, text="识别", padding=10)
        recognition.pack(fill=tk.X)
        diameter_controls = ttk.Frame(recognition)
        diameter_controls.pack(anchor="w")
        ttk.Label(diameter_controls, text="圆柱大致直径：").pack(side=tk.LEFT)
        ttk.Entry(diameter_controls, textvariable=self.bcp_expected_diameter_var, width=9).pack(side=tk.LEFT, padx=(2, 5))
        ttk.Combobox(diameter_controls, textvariable=self.bcp_expected_unit_var, state="readonly", values=("nm", "px"), width=4).pack(side=tk.LEFT)
        ttk.Label(diameter_controls, text="（无样本时必填）").pack(side=tk.LEFT, padx=(6, 0))
        recognition_buttons = ttk.Frame(recognition)
        recognition_buttons.pack(anchor="w", pady=(10, 0))
        ttk.Button(recognition_buttons, text="开始识别", command=self.perform_bcp_analysis).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(recognition_buttons, text="清除当前结果", command=self.clear_bcp_analysis).pack(side=tk.LEFT)
        ttk.Label(recognition, textvariable=self.result_var, justify=tk.LEFT, wraplength=410).pack(anchor="w", fill=tk.X, pady=(10, 0))

        completion = ttk.LabelFrame(content, text="局部补漏（保留已识别区域）", padding=10)
        completion.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(completion, text="可连续添加多个不规则区域；每个区域各自计算局部阈值，框外绿色轮廓不会重算。", wraplength=430).pack(anchor="w")
        completion_buttons = ttk.Frame(completion)
        completion_buttons.pack(anchor="w", pady=(8, 0))
        ttk.Button(completion_buttons, text="添加补漏区域", command=self.start_bcp_completion_selection).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(completion_buttons, text="识别所选区域", command=self.perform_bcp_completion).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(completion_buttons, text="清除框选", command=self.clear_bcp_completion_selection).pack(side=tk.LEFT)
        ttk.Button(completion, text="完成补漏并计算 CD / Pitch", command=self.finalize_bcp_measurements).pack(anchor="w", pady=(8, 0))

        split = ttk.LabelFrame(content, text="手动分割粘连柱", padding=10)
        split.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(split, text="在一个粘连的绿色轮廓上画一条完整穿过粘连处的分割线；程序只重新计算该轮廓。", wraplength=430).pack(anchor="w")
        split_buttons = ttk.Frame(split)
        split_buttons.pack(anchor="w", pady=(8, 0))
        ttk.Button(split_buttons, text="画分割线", command=self.start_bcp_manual_split).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(split_buttons, text="执行分割", command=self.perform_bcp_manual_split).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(split_buttons, text="清除分割线", command=self.clear_bcp_manual_split).pack(side=tk.LEFT)

        line_display = ttk.LabelFrame(content, text="三角网与晶粒显示（完成计算后）", padding=8)
        line_display.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(line_display, text="三角网可按主方向筛选；同一晶粒会以同色半透明区域显示。", foreground="#666666").pack(anchor="w")
        ttk.Checkbutton(
            line_display,
            text="显示三角网叠加层（蓝色）",
            variable=self.bcp_triangulation_overlay_var,
            command=self.refresh_bcp_line_display,
        ).pack(anchor="w", pady=(5, 0))
        display_choices = ttk.Frame(line_display)
        display_choices.pack(anchor="w", pady=(3, 0))
        for mode in TRIANGULATION_DISPLAY_MODES:
            ttk.Radiobutton(
                display_choices,
                text=mode,
                value=mode,
                variable=self.bcp_line_display_var,
                command=self.refresh_bcp_line_display,
            ).pack(side=tk.LEFT, padx=(0, 9))
        ttk.Checkbutton(
            line_display,
            text="显示晶粒取向叠加层（彩色）",
            variable=self.bcp_grain_overlay_var,
            command=self.refresh_bcp_line_display,
        ).pack(anchor="w", pady=(5, 0))
        ttk.Checkbutton(
            line_display,
            text="显示质心虚拟圆柱布局（显示锚点）",
            variable=self.bcp_centroid_layout_overlay_var,
            command=self.refresh_bcp_line_display,
        ).pack(anchor="w", pady=(5, 0))
        overlay_style = ttk.LabelFrame(content, text="BCP 线条样式（画面/导出）", padding=8)
        overlay_style.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(overlay_style, text="仅改变显示和导出，不改变识别、CD 或 Pitch。", foreground="#666666").pack(anchor="w")
        contour_style = ttk.Frame(overlay_style)
        contour_style.pack(anchor="w", pady=(6, 0))
        ttk.Label(contour_style, text="绿色轮廓：").pack(side=tk.LEFT)
        self.bcp_contour_color_button = tk.Button(
            contour_style,
            text="选择颜色",
            background=self.bcp_contour_color,
            activebackground=self.bcp_contour_color,
            command=self.choose_bcp_contour_color,
        )
        self.bcp_contour_color_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(contour_style, text="粗细：").pack(side=tk.LEFT)
        contour_width = ttk.Combobox(
            contour_style,
            textvariable=self.bcp_contour_width_var,
            state="readonly",
            values=("1", "2", "3", "4", "5", "6"),
            width=3,
        )
        contour_width.pack(side=tk.LEFT)
        contour_width.bind("<<ComboboxSelected>>", self.refresh_bcp_overlay_style)
        triangulation_style = ttk.Frame(overlay_style)
        triangulation_style.pack(anchor="w", pady=(5, 0))
        ttk.Label(triangulation_style, text="蓝色三角网：").pack(side=tk.LEFT)
        self.bcp_triangulation_color_button = tk.Button(
            triangulation_style,
            text="选择颜色",
            background=self.bcp_triangulation_color,
            activebackground=self.bcp_triangulation_color,
            command=self.choose_bcp_triangulation_color,
        )
        self.bcp_triangulation_color_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(triangulation_style, text="粗细：").pack(side=tk.LEFT)
        triangulation_width = ttk.Combobox(
            triangulation_style,
            textvariable=self.bcp_triangulation_width_var,
            state="readonly",
            values=("1", "2", "3", "4", "5", "6"),
            width=3,
        )
        triangulation_width.pack(side=tk.LEFT)
        triangulation_width.bind("<<ComboboxSelected>>", self.refresh_bcp_overlay_style)
        centroid_style = ttk.Frame(overlay_style)
        centroid_style.pack(anchor="w", pady=(5, 0))
        ttk.Label(centroid_style, text="黄色质心点：").pack(side=tk.LEFT)
        self.bcp_centroid_color_button = tk.Button(
            centroid_style,
            text="选择颜色",
            background=self.bcp_centroid_color,
            activebackground=self.bcp_centroid_color,
            command=self.choose_bcp_centroid_color,
        )
        self.bcp_centroid_color_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(centroid_style, text="直径：").pack(side=tk.LEFT)
        centroid_diameter = ttk.Combobox(
            centroid_style,
            textvariable=self.bcp_centroid_diameter_var,
            state="readonly",
            values=("3", "4", "5", "6", "8", "10", "12"),
            width=3,
        )
        centroid_diameter.pack(side=tk.LEFT)
        ttk.Label(centroid_style, text="px").pack(side=tk.LEFT, padx=(3, 0))
        centroid_diameter.bind("<<ComboboxSelected>>", self.refresh_bcp_overlay_style)
        layout_style = ttk.Frame(overlay_style)
        layout_style.pack(anchor="w", pady=(5, 0))
        ttk.Label(layout_style, text="虚拟圆柱轮廓：").pack(side=tk.LEFT)
        self.bcp_centroid_layout_color_button = tk.Button(
            layout_style,
            text="选择颜色",
            background=self.bcp_centroid_layout_color,
            activebackground=self.bcp_centroid_layout_color,
            command=self.choose_bcp_centroid_layout_color,
        )
        self.bcp_centroid_layout_color_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(layout_style, text="粗细：").pack(side=tk.LEFT)
        layout_width = ttk.Combobox(
            layout_style,
            textvariable=self.bcp_centroid_layout_width_var,
            state="readonly",
            values=("1", "2", "3", "4", "5", "6"),
            width=3,
        )
        layout_width.pack(side=tk.LEFT)
        layout_width.bind("<<ComboboxSelected>>", self.refresh_bcp_overlay_style)
        anchor_style = ttk.Frame(overlay_style)
        anchor_style.pack(anchor="w", pady=(5, 0))
        ttk.Label(anchor_style, text="中心锚点：").pack(side=tk.LEFT)
        self.bcp_centroid_anchor_color_button = tk.Button(
            anchor_style,
            text="选择颜色",
            background=self.bcp_centroid_anchor_color,
            activebackground=self.bcp_centroid_anchor_color,
            command=self.choose_bcp_centroid_anchor_color,
        )
        self.bcp_centroid_anchor_color_button.pack(side=tk.LEFT)

    def close_bcp_recognition_dialog(self) -> None:
        if self.bcp_dialog is not None and self.bcp_dialog.winfo_exists():
            self.bcp_dialog.destroy()
        self.bcp_dialog = None
        self.bcp_dialog_scroll = None

    def perform_bcp_analysis(self) -> None:
        if self.raw_image is None:
            messagebox.showinfo("暂无图像", "请先打开一张俯视 BCP SEM 图像。")
            return
        image = self.processed_image
        if image is None:
            return
        try:
            expected_diameter_px = self.expected_bcp_diameter_px()
            if len(self.bcp_reference_circles) < 3 and expected_diameter_px is None:
                raise ValueError("自动识别请先输入圆柱大致直径；或者标示至少 3 个样本圆柱。")
            min_area_fraction = self.bcp_min_area_fraction()
            internal_hole_fraction = self.bcp_internal_hole_fraction()
            use_local_segmentation = self.bcp_use_local_segmentation_var.get()
            local_horizontal_sections, local_vertical_sections = self.bcp_grid_sections() if use_local_segmentation else (0, 0)
            started_at = time.perf_counter()
            self.bcp_result = analyze_bcp_dots(
                image,
                self.measurement_pixel_size(),
                self.bcp_reference_circles,
                expected_diameter_px,
                min_area_fraction,
                local_horizontal_sections,
                local_vertical_sections,
                use_local_segmentation,
                self.bcp_use_dog_var.get(),
                internal_hole_fraction,
            )
            self.recognition_duration_var.set(f"识别耗时：{time.perf_counter() - started_at:.2f} 秒")
            self.bcp_metrics_finalized = False
            self.bcp_reference_display_var.set(False)
        except ValueError as exc:
            messagebox.showwarning("无法识别 BCP 点阵", str(exc))
            return
        self.result = None
        self.analysis_origin = None
        self.clear_auto_line_candidates(redraw=False)
        self.clear_bcp_completion_selection(redraw=False)
        self.rebuild_bcp_preview()
        self.draw_image()
        self.update_bcp_result_text()
        response_mode = "DoG 对比增强" if self.bcp_use_dog_var.get() else "原始亮暗强度"
        segmentation = f"局部分割：横向 {local_horizontal_sections} × 竖向 {local_vertical_sections}" if use_local_segmentation else "整图分割"
        self.status_var.set(
            f"BCP 点阵识别完成（{response_mode}；{segmentation}；"
            f"最小连通面积 {min_area_fraction:.2f} × 参考面积；"
            f"柱内空洞忽略上限 {internal_hole_fraction:.2f} × 参考面积）："
            "标准样本圈已隐藏但校准仍保留；请检查绿色轮廓并完成局部补漏，再计算 CD / Pitch。"
        )

    def expected_bcp_diameter_px(self) -> float | None:
        text = self.bcp_expected_diameter_var.get().strip()
        if not text:
            return None
        diameter = float(text)
        if diameter <= 0:
            raise ValueError("圆柱大致直径必须大于 0。")
        if self.bcp_expected_unit_var.get() == "px":
            return diameter
        pixel_size = self.measurement_pixel_size()
        if pixel_size is None:
            raise ValueError("输入 nm 尺寸前，请先填写像素尺寸；或把单位改为 px。")
        return diameter / pixel_size

    def bcp_min_area_fraction(self) -> float:
        fraction = float(self.bcp_min_area_fraction_var.get())
        if not 0.0 <= fraction <= 0.80:
            raise ValueError("最小保留面积应在 0–0.80 倍参考面积之间。")
        return fraction

    def bcp_internal_hole_fraction(self) -> float:
        fraction = float(self.bcp_internal_hole_fraction_var.get())
        if not 0.0 <= fraction <= 0.30:
            raise ValueError("柱内空洞忽略上限应在 0–0.30 倍参考面积之间。")
        return fraction

    def bcp_grid_sections(self) -> tuple[int, int]:
        horizontal = int(self.bcp_horizontal_sections_var.get())
        vertical = int(self.bcp_vertical_sections_var.get())
        if not 1 <= horizontal <= 30 or not 1 <= vertical <= 30:
            raise ValueError("横向和竖向分块数均应为 1–30 的整数。")
        return horizontal, vertical

    def clear_bcp_analysis(self) -> None:
        if self.bcp_result is None:
            return
        self.bcp_result = None
        self.bcp_metrics_finalized = False
        self.clear_bcp_completion_selection(redraw=False)
        self.draw_image()
        self.result_var.set("已清除 BCP 点阵识别结果。")
        self.status_var.set("已清除 BCP 绿色轮廓、蓝色三角网和红色晶界标记；蓝色样本圈仍保留。")

    def start_bcp_completion_selection(self) -> None:
        if self.bcp_result is None:
            messagebox.showinfo("请先识别", "请先完成整图 BCP 识别，再使用局部补漏。")
            return
        if self.bcp_metrics_finalized:
            self.bcp_metrics_finalized = False
            self.draw_image()
            self.update_bcp_result_text()
        self.stop_bcp_reference_marking()
        self.clear_bcp_manual_split()
        self.bcp_completion_mode = True
        self.bcp_completion_drawing = False
        self.bcp_completion_polygon_px = []
        self.draw_bcp_completion_polygon()
        self.canvas.configure(cursor="pencil")
        selected_count = len(self.bcp_completion_polygons_px)
        self.status_var.set(
            f"局部补漏框选：按住鼠标左键圈出第 {selected_count + 1} 个遗漏区域；"
            "可继续添加，全部选好后点击“识别所选区域”。"
        )

    def start_bcp_completion_draw(self, event: tk.Event) -> None:
        point = self.canvas_to_image_point(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.bcp_completion_polygon_px = [point]
        self.bcp_completion_drawing = True
        self.draw_bcp_completion_polygon()

    def move_bcp_completion_draw(self, event: tk.Event) -> None:
        point = self.canvas_to_image_point(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if not self.bcp_completion_polygon_px or math.dist(point, self.bcp_completion_polygon_px[-1]) >= 1.5:
            self.bcp_completion_polygon_px.append(point)
            self.draw_bcp_completion_polygon()

    def finish_bcp_completion_draw(self, event: tk.Event) -> None:
        self.move_bcp_completion_draw(event)
        self.bcp_completion_drawing = False
        self.bcp_completion_mode = False
        self.canvas.configure(cursor="")
        if len(self.bcp_completion_polygon_px) < 3:
            self.bcp_completion_polygon_px = []
            self.status_var.set("框选区域过小；请重新圈选。")
        else:
            self.bcp_completion_polygons_px.append(self.bcp_completion_polygon_px)
            self.bcp_completion_polygon_px = []
            selected_count = len(self.bcp_completion_polygons_px)
            self.status_var.set(
                f"已添加 {selected_count} 个补漏区域；可继续添加，全部选好后点击“识别所选区域”。"
            )
        self.draw_bcp_completion_polygon()

    def clear_bcp_completion_selection(self, redraw: bool = True) -> None:
        self.bcp_completion_mode = False
        self.bcp_completion_drawing = False
        self.bcp_completion_polygon_px = []
        self.bcp_completion_polygons_px.clear()
        self.clear_bcp_manual_split(redraw=False)
        if self.raw_image is not None:
            self.canvas.configure(cursor="")
        if redraw and self.raw_image is not None:
            self.draw_bcp_completion_polygon()

    def start_bcp_manual_split(self) -> None:
        if self.bcp_result is None:
            messagebox.showinfo("请先识别", "请先完成整图 BCP 识别，再手动分割粘连柱。")
            return
        if self.bcp_metrics_finalized:
            self.bcp_metrics_finalized = False
            self.draw_image()
            self.update_bcp_result_text()
        self.stop_bcp_reference_marking()
        self.clear_bcp_completion_selection()
        self.bcp_split_mode = True
        self.bcp_split_drawing = False
        self.bcp_split_start_px = None
        self.bcp_split_end_px = None
        self.canvas.configure(cursor="crosshair")
        self.status_var.set("手动分割：在一个粘连绿色轮廓上拖出一条穿过粘连处的分割线，再点击“执行分割”。")

    def start_bcp_manual_split_draw(self, event: tk.Event) -> None:
        point = self.canvas_to_image_point(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.bcp_split_start_px = point
        self.bcp_split_end_px = point
        self.bcp_split_drawing = True
        self.draw_bcp_manual_split()

    def move_bcp_manual_split_draw(self, event: tk.Event) -> None:
        self.bcp_split_end_px = self.canvas_to_image_point(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.draw_bcp_manual_split()

    def finish_bcp_manual_split_draw(self, event: tk.Event) -> None:
        self.move_bcp_manual_split_draw(event)
        self.bcp_split_drawing = False
        self.bcp_split_mode = False
        self.canvas.configure(cursor="")
        if self.bcp_split_start_px is None or self.bcp_split_end_px is None or math.dist(self.bcp_split_start_px, self.bcp_split_end_px) < 4.0:
            self.bcp_split_start_px = None
            self.bcp_split_end_px = None
            self.status_var.set("分割线过短；请重新画一条完整穿过粘连处的线。")
        else:
            self.status_var.set("分割线已画好；点击“执行分割”只处理它穿过的一个绿色轮廓。")
        self.draw_bcp_manual_split()

    def clear_bcp_manual_split(self, redraw: bool = True) -> None:
        self.bcp_split_mode = False
        self.bcp_split_drawing = False
        self.bcp_split_start_px = None
        self.bcp_split_end_px = None
        if self.raw_image is not None:
            self.canvas.configure(cursor="")
        if redraw and self.raw_image is not None:
            self.draw_bcp_manual_split()

    def perform_bcp_manual_split(self) -> None:
        if self.bcp_result is None or self.processed_image is None:
            messagebox.showinfo("请先识别", "请先完成整图 BCP 识别，再手动分割粘连柱。")
            return
        if self.bcp_split_start_px is None or self.bcp_split_end_px is None:
            messagebox.showinfo("请先画线", "请先点击“画分割线”，并在一个粘连绿色轮廓上拖出分割线。")
            return
        try:
            cut_mask = line_pixel_mask(self.processed_image.shape, self.bcp_split_start_px, self.bcp_split_end_px)
            updated_result, split_count = split_bcp_component(
                self.bcp_result,
                cut_mask,
                self.bcp_min_area_fraction() * bcp_reference_area(
                    self.bcp_reference_circles,
                    self.expected_bcp_diameter_px() or bcp_layout_diameter_px(self.bcp_result),
                ),
            )
        except ValueError as exc:
            messagebox.showwarning("无法分割粘连柱", str(exc))
            return
        self.bcp_result = updated_result
        self.bcp_metrics_finalized = False
        self.result = None
        self.analysis_origin = None
        self.clear_auto_line_candidates(redraw=False)
        self.clear_bcp_manual_split(redraw=False)
        self.draw_image()
        self.update_bcp_result_text()
        self.status_var.set(f"手动分割完成：已将 1 个粘连轮廓拆为 {split_count} 个区域；请检查绿色轮廓后再计算 CD / Pitch。")

    def perform_bcp_completion(self) -> None:
        if self.bcp_result is None or self.processed_image is None:
            messagebox.showinfo("请先识别", "请先完成整图 BCP 识别，再使用局部补漏。")
            return
        if not self.bcp_completion_polygons_px:
            messagebox.showinfo("请先框选", "请先使用“添加补漏区域”徒手圈出一个或多个不规则区域。")
            return
        try:
            expected_diameter_px = self.expected_bcp_diameter_px()
            if len(self.bcp_reference_circles) < 3 and expected_diameter_px is None:
                raise ValueError("局部补漏需要输入圆柱大致直径，或保留至少 3 个样本圆柱。")
            started_at = time.perf_counter()
            updated_result = self.bcp_result
            added = 0
            for polygon_px in self.bcp_completion_polygons_px:
                selection_mask = polygon_pixel_mask(self.processed_image.shape, polygon_px)
                candidates = local_bcp_completion_candidates(
                    self.processed_image,
                    selection_mask,
                    self.bcp_reference_circles,
                    expected_diameter_px,
                    self.bcp_min_area_fraction(),
                    self.bcp_internal_hole_fraction(),
                    self.bcp_use_dog_var.get(),
                )
                updated_result, added_in_region = append_bcp_completion(updated_result, candidates, selection_mask)
                added += added_in_region
            self.bcp_result = updated_result
            self.recognition_duration_var.set(f"识别耗时：{time.perf_counter() - started_at:.2f} 秒")
            self.bcp_metrics_finalized = False
        except ValueError as exc:
            messagebox.showwarning("无法局部补漏", str(exc))
            return
        self.result = None
        self.analysis_origin = None
        self.clear_auto_line_candidates(redraw=False)
        self.draw_image()
        self.update_bcp_result_text()
        selected_region_count = len(self.bcp_completion_polygons_px)
        self.clear_bcp_completion_selection()
        self.status_var.set(f"局部补漏完成：从 {selected_region_count} 个区域新增 {added} 个轮廓；现在可继续添加补漏区域，确认后再计算 CD / Pitch。")

    def finalize_bcp_measurements(self) -> None:
        if self.bcp_result is None:
            messagebox.showinfo("请先识别", "请先完成整图识别和局部补漏。")
            return
        self.bcp_result = finalize_bcp_measurements(self.bcp_result)
        self.bcp_metrics_finalized = True
        self.draw_image()
        self.update_bcp_result_text()
        self.status_var.set("已按当前全部柱子计算 CD、Pitch、Delaunay 三角网和晶界候选。")

    def displayed_bcp_line_segments(self, segments_px: np.ndarray) -> np.ndarray:
        if self.bcp_result is None or not self.bcp_triangulation_overlay_var.get():
            return np.empty((0, 4), dtype=float)
        primary_axis = triangulation_primary_axis_degrees(self.bcp_result.triangulation_segments_px)
        return filter_triangulation_display_segments(segments_px, self.bcp_line_display_var.get(), primary_axis)

    def bcp_grain_overlay_preview(self) -> Image.Image | None:
        """Build the optional colored grain layer in current canvas coordinates."""
        if self.bcp_result is None or not self.bcp_metrics_finalized or not self.bcp_grain_overlay_var.get():
            return None
        image = self.bcp_preview_image if self.bcp_use_dog_var.get() else self.processed_image
        if image is None:
            return None
        height, width = image.shape
        display_size = (max(1, round(width * self.display_scale)), max(1, round(height * self.display_scale)))
        return bcp_grain_overlay_image(self.bcp_result.centers_px, bcp_grain_labels(self.bcp_result.centers_px), width, height, display_size)

    def bcp_centroid_layout_preview(self) -> Image.Image | None:
        """Build equal-size virtual cylinders centered on the detected dots."""
        if self.bcp_result is None or not self.bcp_metrics_finalized or not self.bcp_centroid_layout_overlay_var.get():
            return None
        image = self.bcp_preview_image if self.bcp_use_dog_var.get() else self.processed_image
        if image is None:
            return None
        height, width = image.shape
        display_size = (max(1, round(width * self.display_scale)), max(1, round(height * self.display_scale)))
        result = self.bcp_result
        anchor = central_bcp_centroid(result.centers_px, width, height)
        return centroid_layout_overlay_image(
            result.centers_px,
            anchor,
            bcp_layout_diameter_px(result),
            width,
            height,
            display_size,
            self.bcp_centroid_layout_color,
            self.bcp_overlay_line_width(self.bcp_centroid_layout_width_var),
            self.bcp_centroid_anchor_color,
        )

    def bcp_centroid_layout_anchor(self) -> np.ndarray | None:
        """Return the original detected centroid used as the layout anchor."""
        if self.bcp_result is None or not self.bcp_metrics_finalized or not self.bcp_centroid_layout_overlay_var.get():
            return None
        image = self.bcp_preview_image if self.bcp_use_dog_var.get() else self.processed_image
        if image is None:
            return None
        height, width = image.shape
        return central_bcp_centroid(self.bcp_result.centers_px, width, height)

    def refresh_bcp_line_display(self) -> None:
        if self.bcp_metrics_finalized:
            self.draw_image()
            self.update_bcp_result_text()

    def bcp_overlay_line_width(self, value: tk.StringVar) -> int:
        try:
            return int(np.clip(int(value.get()), 1, 6))
        except ValueError:
            return int(DEFAULT_BCP_OVERLAY_LINE_WIDTH)

    def bcp_centroid_diameter(self) -> int:
        try:
            return int(np.clip(int(self.bcp_centroid_diameter_var.get()), 3, 12))
        except ValueError:
            return int(DEFAULT_BCP_CENTROID_DIAMETER)

    def refresh_bcp_overlay_style(self, _event: tk.Event | None = None) -> None:
        self.draw_image()

    def choose_bcp_contour_color(self) -> None:
        color = colorchooser.askcolor(color=self.bcp_contour_color, parent=self.bcp_dialog or self, title="选择 BCP 轮廓颜色")[1]
        if color is None:
            return
        self.bcp_contour_color = color
        self.bcp_contour_color_button.configure(background=color, activebackground=color)
        self.refresh_bcp_overlay_style()

    def choose_bcp_triangulation_color(self) -> None:
        color = colorchooser.askcolor(color=self.bcp_triangulation_color, parent=self.bcp_dialog or self, title="选择 BCP 三角网颜色")[1]
        if color is None:
            return
        self.bcp_triangulation_color = color
        self.bcp_triangulation_color_button.configure(background=color, activebackground=color)
        self.refresh_bcp_overlay_style()

    def choose_bcp_centroid_color(self) -> None:
        color = colorchooser.askcolor(color=self.bcp_centroid_color, parent=self.bcp_dialog or self, title="选择 BCP 质心点颜色")[1]
        if color is None:
            return
        self.bcp_centroid_color = color
        self.bcp_centroid_color_button.configure(background=color, activebackground=color)
        self.refresh_bcp_overlay_style()

    def choose_bcp_centroid_layout_color(self) -> None:
        color = colorchooser.askcolor(
            color=self.bcp_centroid_layout_color,
            parent=self.bcp_dialog or self,
            title="选择虚拟圆柱轮廓颜色",
        )[1]
        if color is None:
            return
        self.bcp_centroid_layout_color = color
        self.bcp_centroid_layout_color_button.configure(background=color, activebackground=color)
        self.refresh_bcp_overlay_style()

    def choose_bcp_centroid_anchor_color(self) -> None:
        color = colorchooser.askcolor(
            color=self.bcp_centroid_anchor_color,
            parent=self.bcp_dialog or self,
            title="选择中心锚点颜色",
        )[1]
        if color is None:
            return
        self.bcp_centroid_anchor_color = color
        self.bcp_centroid_anchor_color_button.configure(background=color, activebackground=color)
        self.refresh_bcp_overlay_style()

    def refresh_bcp_reference_display(self) -> None:
        self.draw_bcp_reference_circles()

    def outlier_threshold_px(self) -> float:
        return OUTLIER_LEVELS.get(self.outlier_level_var.get(), OUTLIER_LEVELS[DEFAULT_OUTLIER_LEVEL])

    def render_analysis_overlay(self) -> None:
        self.canvas.delete("edge")
        if self.detected_edge_mask is not None:
            overlay = np.zeros((*self.detected_edge_mask.shape, 4), dtype=np.uint8)
            overlay[self.detected_edge_mask] = (0, 229, 255, 180)
            display_size = (
                max(1, round(self.detected_edge_mask.shape[1] * self.display_scale)),
                max(1, round(self.detected_edge_mask.shape[0] * self.display_scale)),
            )
            rendered = Image.fromarray(overlay, "RGBA").resize(display_size, Image.Resampling.NEAREST)
            self.detected_edge_overlay = ImageTk.PhotoImage(rendered)
            left, top = self.detected_edge_origin
            self.canvas.create_image(
                left * self.display_scale,
                top * self.display_scale,
                image=self.detected_edge_overlay,
                anchor=tk.NW,
                tags="edge",
            )
            return
        if self.result is None or self.analysis_origin is None:
            return
        if self.result.edge_detector == EDGE_DETECTOR_ERA:
            scale = self.display_scale
            for sample in self.era_line_samples:
                for edge, color in ((sample.left_edge, self.fit_left_color), (sample.right_edge, self.fit_right_color)):
                    if edge is None or len(edge.rows_px) < 2:
                        continue
                    points = [coordinate for row, position in zip(edge.rows_px, edge.edge_px) for coordinate in (position * scale, row * scale)]
                    self.canvas.create_line(*points, fill=color, width=self.fit_line_width(), tags="edge")
            return
        overlays = [(self.result, self.analysis_origin)]
        scale = self.display_scale
        for result, (left_offset, top_offset) in overlays:
            points_left = []
            points_right = []
            for row, left, right in zip(result.rows_px, result.left_px, result.right_px):
                points_left.extend([(left + left_offset) * scale, (row + top_offset) * scale])
                points_right.extend([(right + left_offset) * scale, (row + top_offset) * scale])
            if len(points_left) >= 4:
                self.canvas.create_line(*points_left, fill=self.fit_left_color, width=self.fit_line_width(), tags="edge")
            if len(points_right) >= 4:
                self.canvas.create_line(*points_right, fill=self.fit_right_color, width=self.fit_line_width(), tags="edge")

    def render_bcp_overlay(self) -> None:
        self.canvas.delete("bcp")
        self.grain_overlay_image = None
        self.centroid_layout_overlay_image = None
        if self.bcp_result is None:
            return
        scale = self.display_scale
        result = self.bcp_result
        if self.bcp_metrics_finalized:
            overlay = self.bcp_grain_overlay_preview()
            if overlay is not None:
                self.grain_overlay_image = ImageTk.PhotoImage(overlay)
                self.canvas.create_image(0, 0, image=self.grain_overlay_image, anchor=tk.NW, tags="bcp")
            centroid_layout = self.bcp_centroid_layout_preview()
            if centroid_layout is not None:
                self.centroid_layout_overlay_image = ImageTk.PhotoImage(centroid_layout)
                self.canvas.create_image(0, 0, image=self.centroid_layout_overlay_image, anchor=tk.NW, tags="bcp")
            for x0, y0, x1, y1 in self.displayed_bcp_line_segments(result.triangulation_segments_px):
                self.canvas.create_line(
                    x0 * scale,
                    y0 * scale,
                    x1 * scale,
                    y1 * scale,
                    fill=self.bcp_triangulation_color,
                    width=self.bcp_overlay_line_width(self.bcp_triangulation_width_var),
                    tags="bcp",
                )
        for contour in result.contour_segments_px:
            for x0, y0, x1, y1 in contour:
                self.canvas.create_line(
                    x0 * scale,
                    y0 * scale,
                    x1 * scale,
                    y1 * scale,
                    fill=self.bcp_contour_color,
                    width=self.bcp_overlay_line_width(self.bcp_contour_width_var),
                    tags="bcp",
                )
        centroid_radius = self.bcp_centroid_diameter() / 2
        for center_x, center_y in result.centers_px:
            display_x, display_y = center_x * scale, center_y * scale
            self.canvas.create_oval(
                display_x - centroid_radius,
                display_y - centroid_radius,
                display_x + centroid_radius,
                display_y + centroid_radius,
                fill=self.bcp_centroid_color,
                outline="#4a3b00",
                width=1,
                tags="bcp",
            )
        anchor = self.bcp_centroid_layout_anchor()
        if anchor is not None:
            anchor_x, anchor_y = anchor * scale
            self.canvas.create_oval(
                anchor_x - 5,
                anchor_y - 5,
                anchor_x + 5,
                anchor_y + 5,
                fill=self.bcp_centroid_anchor_color,
                outline="#fff5f5",
                width=1,
                tags="bcp",
            )

    def draw_bcp_completion_polygon(self) -> None:
        self.canvas.delete("bcp_completion")
        scale = self.display_scale
        polygons = self.bcp_completion_polygons_px + [self.bcp_completion_polygon_px]
        for polygon_px in polygons:
            if len(polygon_px) < 2:
                continue
            points = [(x * scale, y * scale) for x, y in polygon_px]
            if polygon_px is not self.bcp_completion_polygon_px and len(points) >= 3:
                points.append(points[0])
            self.canvas.create_line(*[coordinate for point in points for coordinate in point], fill="#ffd43b", width=2, dash=(5, 3), tags="bcp_completion")

    def draw_bcp_manual_split(self) -> None:
        self.canvas.delete("bcp_split")
        if self.bcp_split_start_px is None or self.bcp_split_end_px is None:
            return
        start_x, start_y = self.bcp_split_start_px
        end_x, end_y = self.bcp_split_end_px
        self.canvas.create_line(
            start_x * self.display_scale,
            start_y * self.display_scale,
            end_x * self.display_scale,
            end_y * self.display_scale,
            fill="#ff7f50",
            width=2,
            dash=(4, 2),
            tags="bcp_split",
        )

    def draw_bcp_reference_circles(self) -> None:
        self.canvas.delete("bcp_reference")
        if not self.bcp_reference_mode and not self.bcp_reference_display_var.get():
            return
        scale = self.display_scale
        for index, (center_x, center_y, radius) in enumerate(self.bcp_reference_circles, start=1):
            is_active = index - 1 == self.bcp_reference_active_index
            color = "#ffffff" if is_active else "#4ea1ff"
            self.canvas.create_oval(
                (center_x - radius) * scale,
                (center_y - radius) * scale,
                (center_x + radius) * scale,
                (center_y + radius) * scale,
                outline=color,
                width=2,
                tags="bcp_reference",
            )
            self.canvas.create_text(center_x * scale, (center_y - radius) * scale - 5, text=f"样本 {index}", fill="#4ea1ff", tags="bcp_reference")
            if is_active:
                for x, y in ((center_x, center_y), (center_x + radius, center_y)):
                    self.canvas.create_rectangle(x * scale - 4, y * scale - 4, x * scale + 4, y * scale + 4, fill="#ffffff", outline="#2369b0", tags="bcp_reference")
        if self.bcp_reference_start is not None:
            center_x, center_y = self.bcp_reference_start
            radius = self.bcp_reference_preview_radius
            self.canvas.create_oval(
                (center_x - radius) * scale,
                (center_y - radius) * scale,
                (center_x + radius) * scale,
                (center_y + radius) * scale,
                outline="#4ea1ff",
                dash=(4, 2),
                width=2,
                tags="bcp_reference",
            )

    def update_bcp_reference_text(self) -> None:
        count = len(self.bcp_reference_circles)
        self.bcp_reference_var.set("当前没有手动样本；输入柱径后可自动识别。" if not count else f"已标示 {count} 个代表圆柱（建议至少 3 个）。")

    def start_bcp_reference(self, event: tk.Event) -> None:
        if self.raw_image is None:
            return
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        point = self.canvas_to_image_point(canvas_x, canvas_y)
        hit = self.bcp_reference_hit_test(point)
        if hit is not None:
            self.bcp_reference_active_index, self.bcp_reference_drag_mode = hit
            self.bcp_reference_drag_anchor = point
            self.bcp_reference_start_circle = self.bcp_reference_circles[self.bcp_reference_active_index]
            self.draw_bcp_reference_circles()
            return
        self.bcp_reference_active_index = None
        self.bcp_reference_drag_mode = "create"
        self.bcp_reference_start = point
        self.bcp_reference_preview_radius = 0.0
        self.draw_bcp_reference_circles()

    def bcp_reference_hit_test(self, point: tuple[float, float]) -> tuple[int, str] | None:
        x, y = point
        for index in range(len(self.bcp_reference_circles) - 1, -1, -1):
            center_x, center_y, radius = self.bcp_reference_circles[index]
            distance = math.hypot(x - center_x, y - center_y)
            if abs(distance - radius) <= 8 / self.display_scale:
                return index, "resize"
            if distance <= radius:
                return index, "move"
        return None

    def move_bcp_reference(self, event: tk.Event) -> None:
        point = self.canvas_to_image_point(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if self.bcp_reference_drag_mode == "create" and self.bcp_reference_start is not None:
            self.bcp_reference_preview_radius = math.dist(self.bcp_reference_start, point)
            self.draw_bcp_reference_circles()
            return
        if self.bcp_reference_active_index is None or self.bcp_reference_start_circle is None:
            return
        center_x, center_y, radius = self.bcp_reference_start_circle
        if self.bcp_reference_drag_mode == "move" and self.bcp_reference_drag_anchor is not None:
            anchor_x, anchor_y = self.bcp_reference_drag_anchor
            center_x += point[0] - anchor_x
            center_y += point[1] - anchor_y
        elif self.bcp_reference_drag_mode == "resize":
            radius = math.dist((center_x, center_y), point)
        candidate = self.clamp_bcp_reference_circle(center_x, center_y, radius)
        if not self.bcp_reference_overlaps(candidate, skip_index=self.bcp_reference_active_index):
            self.bcp_reference_circles[self.bcp_reference_active_index] = candidate
        self.draw_bcp_reference_circles()

    def finish_bcp_reference(self, event: tk.Event) -> None:
        self.move_bcp_reference(event)
        if self.bcp_reference_drag_mode == "create" and self.bcp_reference_start is not None and self.bcp_reference_preview_radius >= 2.0:
            candidate = self.clamp_bcp_reference_circle(*self.bcp_reference_start, self.bcp_reference_preview_radius)
            if self.bcp_reference_overlaps(candidate):
                self.status_var.set("样本圆柱不能重叠；请重新圈选另一个独立圆柱。")
            else:
                self.bcp_reference_circles.append(candidate)
        self.bcp_reference_start = None
        self.bcp_reference_preview_radius = 0.0
        self.bcp_reference_drag_mode = None
        self.bcp_reference_drag_anchor = None
        self.bcp_reference_start_circle = None
        self.update_bcp_reference_text()
        self.draw_bcp_reference_circles()
        self.status_var.set("可继续添加任意数量的蓝色样本圈；有 3 个以上时尺寸校准更稳定。")

    def clamp_bcp_reference_circle(self, center_x: float, center_y: float, radius: float) -> tuple[float, float, float]:
        if self.raw_image is None:
            return center_x, center_y, radius
        radius = float(np.clip(radius, 2.0, min(self.raw_image.shape) / 2 - 1))
        center_x = float(np.clip(center_x, radius, self.raw_image.shape[1] - radius))
        center_y = float(np.clip(center_y, radius, self.raw_image.shape[0] - radius))
        return center_x, center_y, radius

    def bcp_reference_overlaps(self, circle: tuple[float, float, float], skip_index: int | None = None) -> bool:
        center_x, center_y, radius = circle
        for index, (other_x, other_y, other_radius) in enumerate(self.bcp_reference_circles):
            if index != skip_index and math.hypot(center_x - other_x, center_y - other_y) < radius + other_radius:
                return True
        return False

    def start_bcp_reference_marking(self) -> None:
        if self.raw_image is None:
            return
        self.clear_bcp_completion_selection()
        self.bcp_reference_mode = True
        self.bcp_reference_display_var.set(True)
        self.draw_bcp_reference_circles()
        self.status_var.set("样本编辑模式：拖蓝圈内部移动；拖外缘或白色控制点改大小；空白处拖动可添加更多互不重叠的样本。")
        self.canvas.configure(cursor="crosshair")

    def stop_bcp_reference_marking(self) -> None:
        self.bcp_reference_mode = False
        self.bcp_reference_active_index = None
        self.bcp_reference_drag_mode = None
        self.bcp_reference_start = None
        self.bcp_reference_display_var.set(False)
        self.canvas.configure(cursor="")
        self.draw_bcp_reference_circles()
        self.status_var.set("已结束样本编辑；样本圈已隐藏，校准数据仍可用于识别。")

    def clear_bcp_reference_circles(self) -> None:
        self.bcp_reference_mode = False
        self.bcp_reference_start = None
        self.bcp_reference_preview_radius = 0.0
        self.bcp_reference_active_index = None
        self.bcp_reference_drag_mode = None
        self.bcp_reference_drag_anchor = None
        self.bcp_reference_start_circle = None
        self.bcp_reference_display_var.set(False)
        self.bcp_reference_circles.clear()
        self.canvas.configure(cursor="")
        self.update_bcp_reference_text()
        if self.raw_image is not None:
            self.draw_bcp_reference_circles()
        self.status_var.set("已清除 BCP 样本标示。")

    def discard_bcp_reference_circles(self) -> None:
        self.bcp_reference_mode = False
        self.bcp_reference_start = None
        self.bcp_reference_preview_radius = 0.0
        self.bcp_reference_active_index = None
        self.bcp_reference_drag_mode = None
        self.bcp_reference_drag_anchor = None
        self.bcp_reference_start_circle = None
        self.bcp_reference_display_var.set(False)
        self.bcp_reference_circles.clear()
        self.update_bcp_reference_text()


    def update_result_text(self) -> None:
        if self.result is None:
            return
        if self.result.edge_detector == EDGE_DETECTOR_ERA:
            self.update_era_result_text()
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
            f"剔除阈值  {self.outlier_threshold_px():.0f} px\n"
            f"边缘检测  {result.edge_detector}（{result.edge_kernel_size}×{result.edge_kernel_size}）"
        )

    def update_bcp_result_text(self) -> None:
        if self.bcp_result is None:
            return
        result = self.bcp_result
        if not self.bcp_metrics_finalized:
            self.result_var.set(
                "BCP 轮廓识别\n\n"
                f"识别点数        {len(result.centers_px)}\n\n"
                "绿色为轮廓，黄色小点为几何质心；如有遗漏，可反复使用“局部补漏”。\n"
                "确认无遗漏后，点击“完成补漏并计算 CD / Pitch”。"
            )
            return
        scale = result.pixel_size_nm if np.isfinite(result.pixel_size_nm) else 1.0
        unit = "nm" if np.isfinite(result.pixel_size_nm) else "px"
        mean_diameter = float(np.mean(result.equivalent_diameters_px) * scale)
        diameter_sigma = float(np.std(result.equivalent_diameters_px, ddof=1) * scale) if len(result.centers_px) > 1 else 0.0
        mean_major = float(np.mean(result.major_axes_px) * scale)
        mean_minor = float(np.mean(result.minor_axes_px) * scale)
        spacing = result.lattice_spacing_px * scale
        cd_mean, cd_standard_error = mean_and_standard_error(result.cd_means_px * scale)
        grain_labels = bcp_grain_labels(result.centers_px)
        grain_count = len(np.unique(grain_labels[grain_labels >= 0]))
        pitch_sigma_multiplier = self.sigma_multiplier_var.get()
        pitch_lines = []
        for label in PITCH_DIRECTION_LABELS:
            pitch_mean, pitch_sigma_spread = mean_and_sigma_spread(result.pitch_values_by_direction_px[label] * scale, pitch_sigma_multiplier)
            value = "无有效边" if not np.isfinite(pitch_mean) else f"{pitch_mean:.3f} ± {pitch_sigma_spread:.3f} {unit}（{pitch_sigma_multiplier:g}σ）"
            pitch_lines.append(f"Pitch {label:<7} {value}")
        total_pitch_mean, total_pitch_sigma_spread = mean_and_sigma_spread(
            all_directional_pitch_values(result.pitch_values_by_direction_px) * scale,
            pitch_sigma_multiplier,
        )
        total_pitch_value = (
            "无有效边"
            if not np.isfinite(total_pitch_mean)
            else f"{total_pitch_mean:.3f} ± {total_pitch_sigma_spread:.3f} {unit}（{pitch_sigma_multiplier:g}σ）"
        )
        self.result_var.set(
            "BCP 垂直点阵识别\n\n"
            f"识别点数        {len(result.centers_px)}\n"
            f"CD 平均值       {cd_mean:.3f} {unit}\n"
            f"CD 标准误差     {cd_standard_error:.3f} {unit}\n"
            f"等效直径        {mean_diameter:.3f} {unit}\n"
            f"直径标准差      {diameter_sigma:.3f} {unit}\n"
            f"区域长轴均值    {mean_major:.3f} {unit}\n"
            f"区域短轴均值    {mean_minor:.3f} {unit}\n"
            f"最近邻间距      {spacing:.3f} {unit}\n"
            + "\n".join(pitch_lines)
            + f"\nPitch 总平均    {total_pitch_value}"
            + "\n\n"
            + f"三角网边数      {len(result.triangulation_segments_px)}\n"
            f"晶界候选线段    {len(result.boundary_segments_px)}\n"
            f"晶界相关点      {len(result.boundary_dot_indices)}\n\n"
            "绿色：二值连通区域轮廓\n"
            "黄色：区域几何质心\n"
            f"蓝色：Delaunay 三角网（{'显示：' + self.bcp_line_display_var.get() if self.bcp_triangulation_overlay_var.get() else '隐藏'}）\n"
            f"彩色：晶粒取向分区（{grain_count} 个晶粒，{'显示' if self.bcp_grain_overlay_var.get() else '隐藏'}）\n"
            f"质心虚拟圆柱布局（{'显示' if self.bcp_centroid_layout_overlay_var.get() else '隐藏'}，锚点已标示）"
        )

    def update_displayed_results(self) -> None:
        if self.result is not None:
            self.update_result_text()
        elif self.bcp_result is not None:
            self.update_bcp_result_text()
        self.update_lcdu_summary_text()

    def export_annotated_image(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("暂无图像", "请先打开一张 SEM 图像。")
            return
        image = preprocess_image(
            self.original_image,
            self.normalize_var.get(),
            self.gaussian_denoise_var.get(),
            self.gaussian_kernel_size(),
            self.gaussian_sigma_scale(),
            self.blur_method_var.get(),
        )
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
        self.apply_exported_bcp_grain_overlay(output, width, height)
        self.apply_exported_bcp_centroid_layout_overlay(output, width, height)
        self.apply_exported_edge_detection_overlay(output, width, height)
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
        for sample in self.era_line_samples:
            for bounds, color in ((sample.left_roi_bounds_px, self.fit_left_color), (sample.right_roi_bounds_px, self.fit_right_color)):
                if bounds is None:
                    continue
                left, top, right, bottom = bounds
                roi_points = rotate_points_about_center(
                    [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)],
                    -self.rotation_degrees,
                    width,
                    height,
                )
                draw.line(roi_points, fill=color, width=1)

        self.draw_exported_annotations(draw, width, height)
        if self.result is not None and self.analysis_origin is not None:
            if self.result.edge_detector == EDGE_DETECTOR_ERA:
                self.draw_exported_era_edges(draw, width, height)
            else:
                self.draw_exported_analysis_edges(draw, self.result, self.analysis_origin, width, height)
        self.draw_exported_bcp(draw, width, height)

        try:
            output.save(target, "PNG")
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.status_var.set(f"已导出拟合图片：{target}")

    def draw_exported_analysis_edges(
        self,
        draw: ImageDraw.ImageDraw,
        result: AnalysisResult,
        origin: tuple[int, int],
        width: int,
        height: int,
    ) -> None:
        left_offset, top_offset = origin
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
            draw.line(left_points, fill=self.fit_left_color, width=self.fit_line_width())
        if len(right_points) >= 2:
            draw.line(right_points, fill=self.fit_right_color, width=self.fit_line_width())

    def draw_exported_era_edges(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        """Export every independently selected ERA edge in original image orientation."""
        for sample in self.era_line_samples:
            for edge, color in ((sample.left_edge, self.fit_left_color), (sample.right_edge, self.fit_right_color)):
                if edge is None:
                    continue
                points = rotate_points_about_center(
                    [(float(position), float(row)) for row, position in zip(edge.rows_px, edge.edge_px)],
                    -self.rotation_degrees,
                    width,
                    height,
                )
                if len(points) >= 2:
                    draw.line(points, fill=color, width=self.fit_line_width())

    def apply_exported_edge_detection_overlay(self, output: Image.Image, width: int, height: int) -> None:
        """Blend the unpaired edge mask, including an optional ROI offset, into the export."""
        if self.detected_edge_mask is None:
            return
        overlay_pixels = np.zeros((height, width, 4), dtype=np.uint8)
        left, top = self.detected_edge_origin
        mask_height, mask_width = self.detected_edge_mask.shape
        right = min(width, left + mask_width)
        bottom = min(height, top + mask_height)
        if right <= left or bottom <= top:
            return
        visible_mask = self.detected_edge_mask[: bottom - top, : right - left]
        region = overlay_pixels[top:bottom, left:right]
        region[visible_mask] = (0, 229, 255, 180)
        overlay = Image.fromarray(overlay_pixels, "RGBA")
        if self.rotation_degrees:
            overlay = overlay.rotate(-self.rotation_degrees, resample=Image.Resampling.NEAREST, expand=False)
        output.paste(overlay, (0, 0), overlay)

    def apply_exported_bcp_grain_overlay(self, output: Image.Image, width: int, height: int) -> None:
        """Blend the optional working-image grain layer back into original orientation."""
        if self.bcp_result is None or not self.bcp_metrics_finalized or not self.bcp_grain_overlay_var.get():
            return
        overlay = bcp_grain_overlay_image(
            self.bcp_result.centers_px,
            bcp_grain_labels(self.bcp_result.centers_px),
            width,
            height,
            (width, height),
        )
        if overlay is None:
            return
        if self.rotation_degrees:
            overlay = overlay.rotate(-self.rotation_degrees, resample=Image.Resampling.NEAREST, expand=False)
        output.paste(overlay, (0, 0), overlay)

    def apply_exported_bcp_centroid_layout_overlay(self, output: Image.Image, width: int, height: int) -> None:
        """Blend the optional centroid-centered layout back into original orientation."""
        if self.bcp_result is None or not self.bcp_metrics_finalized or not self.bcp_centroid_layout_overlay_var.get():
            return
        result = self.bcp_result
        anchor = central_bcp_centroid(result.centers_px, width, height)
        overlay = centroid_layout_overlay_image(
            result.centers_px,
            anchor,
            bcp_layout_diameter_px(result),
            width,
            height,
            (width, height),
            self.bcp_centroid_layout_color,
            self.bcp_overlay_line_width(self.bcp_centroid_layout_width_var),
            self.bcp_centroid_anchor_color,
        )
        if overlay is None:
            return
        if self.rotation_degrees:
            overlay = overlay.rotate(-self.rotation_degrees, resample=Image.Resampling.NEAREST, expand=False)
        output.paste(overlay, (0, 0), overlay)

    def draw_exported_bcp(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        if self.bcp_result is None:
            return
        result = self.bcp_result
        if self.bcp_metrics_finalized:
            for x0, y0, x1, y1 in self.displayed_bcp_line_segments(result.triangulation_segments_px):
                points = rotate_points_about_center([(x0, y0), (x1, y1)], -self.rotation_degrees, width, height)
                draw.line(points, fill=self.bcp_triangulation_color, width=self.bcp_overlay_line_width(self.bcp_triangulation_width_var))
        for contour in result.contour_segments_px:
            for x0, y0, x1, y1 in contour:
                points = rotate_points_about_center([(x0, y0), (x1, y1)], -self.rotation_degrees, width, height)
                draw.line(points, fill=self.bcp_contour_color, width=self.bcp_overlay_line_width(self.bcp_contour_width_var))
        centroid_radius = self.bcp_centroid_diameter() / 2
        for center_x, center_y in result.centers_px:
            output_x, output_y = rotate_points_about_center([(center_x, center_y)], -self.rotation_degrees, width, height)[0]
            draw.ellipse(
                (output_x - centroid_radius, output_y - centroid_radius, output_x + centroid_radius, output_y + centroid_radius),
                fill=self.bcp_centroid_color,
                outline=(74, 59, 0),
            )
        if self.bcp_reference_display_var.get():
            for center_x, center_y, radius in self.bcp_reference_circles:
                circle = [
                    (center_x + radius * math.cos(phase), center_y + radius * math.sin(phase))
                    for phase in np.linspace(0, 2 * math.pi, 25)
                ]
                draw.line(rotate_points_about_center(circle, -self.rotation_degrees, width, height), fill=(78, 161, 255), width=2)
        if self.bcp_centroid_layout_overlay_var.get() and self.bcp_metrics_finalized:
            anchor = central_bcp_centroid(result.centers_px, width, height)
            anchor_x, anchor_y = rotate_points_about_center([tuple(anchor)], -self.rotation_degrees, width, height)[0]
            draw.ellipse((anchor_x - 5, anchor_y - 5, anchor_x + 5, anchor_y + 5), fill=self.bcp_centroid_anchor_color, outline=(255, 245, 245), width=1)

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
        if self.result is None and self.bcp_result is None and not self.annotations:
            messagebox.showinfo("暂无结果", "请先完成 LER/LWR、BCP 点阵分析或添加通用测量标注。")
            return
        if self.bcp_result is not None and not self.bcp_metrics_finalized:
            messagebox.showinfo("请先完成补漏", "请确认所有柱子已补全，再点击“完成补漏并计算 CD / Pitch”后导出。")
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
                    aggregate = self.era_aggregate_result if result.edge_detector == EDGE_DETECTOR_ERA else None
                    reported_ler_left = aggregate.ler_left_sigma_nm if aggregate is not None else result.ler_left_sigma_nm
                    reported_ler_right = aggregate.ler_right_sigma_nm if aggregate is not None else result.ler_right_sigma_nm
                    reported_lwr = aggregate.lwr_sigma_nm if aggregate is not None else result.lwr_sigma_nm
                    reported_total_ler = aggregate.total_ler_sigma_nm if aggregate is not None else result.total_ler_sigma_nm
                    reported_correlation = aggregate.edge_correlation if aggregate is not None else result.edge_correlation
                    reported_mean_width = aggregate.mean_width_nm if aggregate is not None else result.mean_width_nm
                    writer.writerow(["edge_detector", result.edge_detector])
                    if result.edge_detector == EDGE_DETECTOR_ERA:
                        writer.writerow(["era_input", "current_preprocessed_image_no_internal_spatial_filter"])
                        writer.writerow(["era_polynomial_degree", self.era_polynomial_degree()])
                        writer.writerow(["era_requested_polarity", self.era_polarity_var.get()])
                        writer.writerow(["era_difference_weight_power", ERA_DIFFERENCE_POWER])
                        if self.era_aggregate_result is not None:
                            aggregate = self.era_aggregate_result
                            writer.writerow(["era_complete_line_count", aggregate.line_count])
                            writer.writerow(["era_total_common_valid_points", aggregate.point_count])
                            writer.writerow(["era_mean_cd_nm", aggregate.mean_width_nm])
                    else:
                        writer.writerow(["edge_detector_kernel_size_px", result.edge_kernel_size])
                        writer.writerow(["edge_detector_diagonal_weight", result.edge_diagonal_weight])
                        writer.writerow(["edge_detector_axial_weight", result.edge_axial_weight])
                        writer.writerow(["canny_high_threshold", result.canny_high_threshold])
                        writer.writerow(["canny_threshold_ratio", result.canny_threshold_ratio])
                        writer.writerow(["canny_low_threshold", result.canny_low_threshold])
                        writer.writerow(["canny_normalization_scale_p99", result.canny_normalization_scale])
                    writer.writerow(["ler_left_sigma_nm", reported_ler_left])
                    writer.writerow(["ler_right_sigma_nm", reported_ler_right])
                    writer.writerow(["lwr_sigma_nm", reported_lwr])
                    writer.writerow(["total_ler_sigma_nm", reported_total_ler])
                    writer.writerow(["edge_correlation_rho", reported_correlation])
                    writer.writerow(["ler_left_display_nm", multiplier * reported_ler_left])
                    writer.writerow(["ler_right_display_nm", multiplier * reported_ler_right])
                    writer.writerow(["lwr_display_nm", multiplier * reported_lwr])
                    writer.writerow(["mean_width_nm", reported_mean_width])
                if self.bcp_result is not None:
                    bcp = self.bcp_result
                    bcp_scale = bcp.pixel_size_nm if np.isfinite(bcp.pixel_size_nm) else 1.0
                    bcp_unit = "nm" if np.isfinite(bcp.pixel_size_nm) else "px"
                    writer.writerow(["bcp_dot_count", len(bcp.centers_px)])
                    writer.writerow(["bcp_size_unit", bcp_unit])
                    writer.writerow(["bcp_mean_equivalent_diameter", float(np.mean(bcp.equivalent_diameters_px) * bcp_scale)])
                    writer.writerow(["bcp_mean_major_axis", float(np.mean(bcp.major_axes_px) * bcp_scale)])
                    writer.writerow(["bcp_mean_minor_axis", float(np.mean(bcp.minor_axes_px) * bcp_scale)])
                    cd_mean, cd_standard_error = mean_and_standard_error(bcp.cd_means_px * bcp_scale)
                    writer.writerow(["bcp_cd_mean", cd_mean])
                    writer.writerow(["bcp_cd_standard_error", cd_standard_error])
                    writer.writerow(["bcp_lattice_spacing", bcp.lattice_spacing_px * bcp_scale])
                    writer.writerow(["bcp_delaunay_edge_count", len(bcp.triangulation_segments_px)])
                    pitch_sigma_multiplier = multiplier
                    writer.writerow(["bcp_pitch_sigma_multiplier", pitch_sigma_multiplier])
                    for label in PITCH_DIRECTION_LABELS:
                        pitch_values = bcp.pitch_values_by_direction_px[label] * bcp_scale
                        pitch_mean, pitch_one_sigma = mean_and_sigma_spread(pitch_values, 1.0)
                        _, pitch_display_spread = mean_and_sigma_spread(pitch_values, pitch_sigma_multiplier)
                        writer.writerow([f"bcp_pitch_{label}_mean", pitch_mean])
                        writer.writerow([f"bcp_pitch_{label}_1sigma", pitch_one_sigma])
                        writer.writerow([f"bcp_pitch_{label}_display_spread", pitch_display_spread])
                    total_pitch_values = all_directional_pitch_values(bcp.pitch_values_by_direction_px) * bcp_scale
                    total_pitch_mean, total_pitch_one_sigma = mean_and_sigma_spread(total_pitch_values, 1.0)
                    _, total_pitch_display_spread = mean_and_sigma_spread(total_pitch_values, pitch_sigma_multiplier)
                    writer.writerow(["bcp_pitch_total_mean", total_pitch_mean])
                    writer.writerow(["bcp_pitch_total_1sigma", total_pitch_one_sigma])
                    writer.writerow(["bcp_pitch_total_display_spread", total_pitch_display_spread])
                    writer.writerow(["bcp_grain_boundary_segment_count", len(bcp.boundary_segments_px)])
                    writer.writerow(["bcp_reference_sample_count", len(self.bcp_reference_circles)])
                lcdu_samples_nm, lcdu_source = self.active_lcdu_samples()
                lcdu_sigma = self.lcdu_sigma_nm()
                writer.writerow(["multi_line_lcdu_source", lcdu_source])
                writer.writerow(["multi_line_lcdu_sample_count", len(lcdu_samples_nm)])
                writer.writerow(["multi_line_lcdu_sigma_nm", "" if lcdu_sigma is None else lcdu_sigma])
                writer.writerow(["multi_line_lcdu_display_nm", "" if lcdu_sigma is None else multiplier * lcdu_sigma])
                writer.writerow([])
                writer.writerow(["multi_line_lcdu_sample_index", "mean_cd_nm"])
                for index, mean_cd_nm in enumerate(lcdu_samples_nm, start=1):
                    writer.writerow([index, mean_cd_nm])
                writer.writerow([])
                writer.writerow(["annotation_index", "type", "x0_px", "y0_px", "x1_px", "y1_px", "display_label"])
                for index, annotation in enumerate(self.annotations, start=1):
                    writer.writerow([index, annotation.kind, *annotation.bounds_px, self.annotation_label(annotation)])
                if self.bcp_result is not None:
                    bcp = self.bcp_result
                    writer.writerow([])
                    writer.writerow([
                        "bcp_dot_index", "center_x_px", "center_y_px", "equivalent_diameter_px",
                        "major_axis_px", "minor_axis_px", "ellipse_angle_degrees", "area_px2",
                        "cd_horizontal_px", "cd_plus60_px", "cd_minus60_px", "cd_mean_px", "is_grain_boundary_dot",
                    ])
                    boundary_indices = set(bcp.boundary_dot_indices.tolist())
                    for index, values in enumerate(zip(
                        bcp.centers_px[:, 0], bcp.centers_px[:, 1], bcp.equivalent_diameters_px,
                        bcp.major_axes_px, bcp.minor_axes_px, bcp.angles_degrees, bcp.areas_px,
                        bcp.directional_cds_px[:, 0], bcp.directional_cds_px[:, 1], bcp.directional_cds_px[:, 2], bcp.cd_means_px,
                    ), start=1):
                        writer.writerow([index, *values, index - 1 in boundary_indices])
                    writer.writerow([])
                    writer.writerow(["bcp_delaunay_edge_index", "x0_px", "y0_px", "x1_px", "y1_px"])
                    for index, segment in enumerate(bcp.triangulation_segments_px, start=1):
                        writer.writerow([index, *segment])
                    writer.writerow([])
                    writer.writerow(["bcp_boundary_segment_index", "x0_px", "y0_px", "x1_px", "y1_px"])
                    for index, segment in enumerate(bcp.boundary_segments_px, start=1):
                        writer.writerow([index, *segment])
                    writer.writerow([])
                    writer.writerow(["bcp_reference_sample_index", "center_x_px", "center_y_px", "radius_px"])
                    for index, circle in enumerate(self.bcp_reference_circles, start=1):
                        writer.writerow([index, *circle])
                if self.result is not None:
                    if self.result.edge_detector == EDGE_DETECTOR_ERA:
                        writer.writerow([])
                        writer.writerow(["era_line_index", "side", "roi_left_px", "roi_top_px", "roi_right_px", "roi_bottom_px", "polarity", "polynomial_degree", "row_px", "edge_px", "edge_residual_nm"])
                        for line_index, sample in enumerate(self.era_line_samples, start=1):
                            for side, edge in (("left", sample.left_edge), ("right", sample.right_edge)):
                                if edge is None:
                                    continue
                                for row, position, residual in zip(edge.rows_px, edge.edge_px, edge.residual_nm):
                                    writer.writerow([line_index, side, *edge.roi_bounds_px, edge.polarity, edge.polynomial_degree, row, position, residual])
                        writer.writerow([])
                        writer.writerow(["era_line_index", "mean_cd_nm", "ler_left_sigma_nm", "ler_right_sigma_nm", "lwr_sigma_nm", "common_valid_point_count"])
                        for line_index, sample in enumerate(self.era_line_samples, start=1):
                            if sample.paired_result is None:
                                continue
                            paired = sample.paired_result
                            writer.writerow([line_index, paired.mean_width_nm, paired.ler_left_sigma_nm, paired.ler_right_sigma_nm, paired.lwr_sigma_nm, len(paired.rows_px)])
                    else:
                        result = self.result
                        writer.writerow([])
                        writer.writerow(["row_px", "left_edge_px", "right_edge_px", "left_residual_nm", "right_residual_nm", "width_nm", "width_residual_nm"])
                        for row in zip(result.rows_px, result.left_px, result.right_px, result.left_residual_nm, result.right_residual_nm, result.width_nm, result.width_residual_nm):
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
