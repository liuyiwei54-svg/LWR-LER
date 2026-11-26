"""生成供主管 GUI 验收的合成测试图 A.png / B.png 与真值说明 JSON/MD。

仅测试数据，绝不作为真实 SEM 数据；无 SEM 信息栏、无像素元数据。用解析 logistic
灰度过渡生成 3 条竖向亮线的左右边界，而非调用任何测量算法，保证 LER/LWR 可测且
真值可写。写入 /tmp/sem-redesign-ui-data/。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path("/tmp/sem-redesign-ui-data")

# 图像尺寸（像素）
WIDTH = 400
HEIGHT = 220

# 三条竖向亮线基准左右边界（x 像素）
EDGES = [(70, 100), (180, 210), (290, 320)]

# logistic 灰度过渡参数
LOW = 40.0       # 暗背景灰度
AMP = 150.0      # 亮线幅度（峰值 = LOW + AMP）
SIGMA = 1.0      # 过渡宽度（约 1 px），解析 logistic 半高宽

# 每条线左右边界的独立 y 方向正弦起伏（可测 LER/LWR）
#   每条线给左右边独立 (amplitude_px, phase_rad, period_rows)，保证左右边缘各自
#   有独立的高频起伏，从而 LER（各自去趋势）与 LWR（线宽去均值）都可测。
ROUGHNESS = {
    "line1": {
        "left":  {"amp": 1.2, "phase": 0.0, "period": 26},
        "right": {"amp": 1.1, "phase": 1.7, "period": 21},
    },
    "line2": {
        "left":  {"amp": 1.4, "phase": 2.3, "period": 31},
        "right": {"amp": 1.0, "phase": 0.9, "period": 17},
    },
    "line3": {
        "left":  {"amp": 0.9, "phase": 3.1, "period": 23},
        "right": {"amp": 1.3, "phase": 4.0, "period": 29},
    },
}

# 缓慢二维加性背景（共同、与 A/B 无关，可写表达式）：
#   background(x, y) = 12 * sin(2π x / 400 * 2) * cos(2π y / 220 * 2) + 10 * sin(2π y / 220 * 3)
BACKGROUND_FUNC = (
    "12·sin(2π·x/400·2)·cos(2π·y/220·2) + 10·sin(2π·y/220·3)"
)

# A / B 各自独立 Gaussian 强度噪声（不同 rng 种子）
NOISE = {
    "A": {"sigma": 3.0, "seed": 1001},
    "B": {"sigma": 5.0, "seed": 2002},
}


def _logistic(x: np.ndarray) -> np.ndarray:
    """数值稳定的向量化 logistic（sigmoid）。"""
    return np.exp(-np.logaddexp(0.0, -x))


def edge_positions(y_idx: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """返回某条线左右边界在每行 y 的实际 x 位置（含独立正弦起伏）。"""
    y = y_idx.astype(float)
    left = spec["left"]
    right = spec["right"]

    def wave(p: dict) -> np.ndarray:
        return p["amp"] * np.sin(2 * math.pi * y / p["period"] + p["phase"])

    return wave(left), wave(right)


def build_grayscale(seed: int, noise_sigma: float) -> np.ndarray:
    """构造一幅灰度图（float，未截断到 uint8）。"""
    rng = np.random.default_rng(seed)
    x = np.arange(WIDTH, dtype=float)[None, :]          # (1, W)
    y = np.arange(HEIGHT, dtype=float)[:, None]         # (H, 1)

    # 缓慢二维加性背景
    background = (
        12.0 * np.sin(2 * math.pi * x / WIDTH * 2) * np.cos(2 * math.pi * y / HEIGHT * 2)
        + 10.0 * np.sin(2 * math.pi * y / HEIGHT * 3)
    )

    # 逐条亮线：logistic 左上升沿 + 右下降沿
    image = np.full((HEIGHT, WIDTH), LOW, dtype=float)
    for (x_left, x_right), key in zip(EDGES, ROUGHNESS):
        left_wave, right_wave = edge_positions(y[:, 0], ROUGHNESS[key])
        left_edge = x_left + left_wave[:, None]
        right_edge = x_right + right_wave[:, None]
        rising = _logistic((x - left_edge) / SIGMA)
        falling = _logistic((right_edge - x) / SIGMA)
        image += AMP * rising * falling

    image += background

    # 独立 Gaussian 强度噪声
    image += rng.normal(0.0, noise_sigma, size=(HEIGHT, WIDTH))

    # 避免饱和：裁剪到 [0, 255]
    image = np.clip(image, 0.0, 255.0)
    return np.round(image).astype(np.uint8)


def _edge_truth_table() -> list[dict]:
    """真值：每条线左右边界在第 0 行的基准 x 与粗糙度项参数。"""
    rows = []
    for i, (x_left, x_right) in enumerate(EDGES, start=1):
        key = f"line{i}"
        rows.append({
            "line": i,
            "left_base_x": x_left,
            "right_base_x": x_right,
            "nominal_width_px": x_right - x_left,
            "left_roughness": ROUGHNESS[key]["left"],
            "right_roughness": ROUGHNESS[key]["right"],
        })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth: dict = {
        "note": "合成测试图，仅用于 PSD/LER-LWR GUI 验收与背景/PSD 数值验证，不代表真实 SEM 实验数据。",
        "image_size_px": [WIDTH, HEIGHT],
        "calibration_hint": "GUI 应手动标定 0.2 nm/px；识别模式选「实体亮线/暗线 → 亮线」。",
        "edge_intensity_model": {
            "low": LOW,
            "amplitude": AMP,
            "peak": LOW + AMP,
            "sigma_px": SIGMA,
            "transition": "解析 logistic 灰度过渡（非测量算法生成）",
        },
        "background": {
            "expression": BACKGROUND_FUNC,
            "note": "共同、缓慢二维加性背景，A/B 完全相同。",
        },
        "noise": {
            "A": {"sigma": NOISE["A"]["sigma"], "seed": NOISE["A"]["seed"]},
            "B": {"sigma": NOISE["B"]["sigma"], "seed": NOISE["B"]["seed"]},
            "note": "各自独立 Gaussian 强度噪声，固定 rng 种子，便于复现。",
        },
        "edges": _edge_truth_table(),
        "file_metadata": "无 SEM 信息栏、无像素 metadata（PNG 不含 TIFF 34682 标签）。",
    }

    for name in ("A", "B"):
        spec = NOISE[name]
        img = build_grayscale(spec["seed"], spec["sigma"])
        path = OUT_DIR / f"{name}.png"
        Image.fromarray(img, mode="L").save(path, "PNG")
        ground_truth[f"{name}_png"] = str(path)
        print(f"wrote {path}  (sigma={spec['sigma']}, seed={spec['seed']})")

    json_path = OUT_DIR / "ground_truth.json"
    json_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    md_path = OUT_DIR / "README.md"
    md_path.write_text(_markdown(ground_truth), encoding="utf-8")
    print(f"wrote {md_path}")


def _markdown(gt: dict) -> str:
    return f"""# 合成 PSD 验收测试图（A.png / B.png）

> 仅测试数据，不放入真实 SEM 日期文件夹；不代表真实实验数据。

## 用途

供主管对「单图 PSD 模型」与「双图差分对照」页做真实 GUI 验收，以及背景/PSD
数值验证：保留 A 窗后载入 B、改动平滑，确认 A 的来源/处理/噪声/文件名上下文
仍固化在打开时刻，不被 live 主窗口改写。

## 图像规格

- 尺寸：{WIDTH}×{HEIGHT} px，灰度。
- 3 条竖向亮线，基准左右边界 {EDGES}。
- 解析 logistic 灰度过渡：low={LOW}、amp={AMP}、sigma≈{SIGMA} px。

## 校准与识别

- GUI 手动标定像素尺寸 **0.2 nm/px**（无 TIFF pixel metadata）。
- 识别模式选「实体亮线/暗线 → 亮线」。

## 真值模型

- 边界强度：`low={LOW}, amp={AMP}, peak={LOW + AMP}, sigma={SIGMA}px`，解析 logistic。
- 每条线左右边界各自独立正弦起伏（见 `{gt['edges']}`），可测 LER（各自去趋势）与 LWR（线宽去均值）。
- 共同二维加性背景：`{BACKGROUND_FUNC}`。
- A/B 各自独立 Gaussian 强度噪声：A σ={NOISE['A']['sigma']}（seed {NOISE['A']['seed']}）、
  B σ={NOISE['B']['sigma']}（seed {NOISE['B']['seed']}），存在共同真值信号。

## 文件

- `A.png` / `B.png`：合成灰度图（无 SEM 信息栏、无像素 metadata）。
- `ground_truth.json`：机器可读真值（边界函数、噪声/背景、source 路径）。
- 本 `README.md`：人工可读说明。
"""


if __name__ == "__main__":
    main()
