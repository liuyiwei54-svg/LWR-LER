# 统一灰度剖面定位 —— 核心集成说明（CORE integration）

本文档记录 `line_localization.py` 的精确 API、坐标约定、行级 stage/reason、定位判据、
背景校正顺序，以及本次返工的实际测试与已知限制。供主线应用集成与后继维护使用。

---

## 1. 模块定位

`line_localization.py` 是单线测量管线的纯 NumPy 核心，绝不接触 UI：

- **过渡定位（单边 ERA）**：极性加权的差分质心引导一个**局部归一化多项式**的 0.5
  高度位置（degree 2/3/4，默认 3）。
- **亮带定位**：两翼分别用各自局部基线做半高目标、再各自用该翼多项式求交点
  （degree 2/3/4，默认 3）。左/右/中心三种位置定义相互独立；`center` 是两次
  半高交点的算术中点，**不是**强度峰值。
- **有界邻域重定位**：相邻有效行仅能引导一次对**同一 ROI 内真实像素**的受限重读，
  位置定义/模型/对比门槛/跳变门限都不变。
- **逐行诊断**与**缺失行统计**、左右公共行配对统计。
- **保守低阶（一/二阶）二维加性背景**建模，在单一归一化全局坐标基上拟合。
  仅使用 NumPy，不引入 SciPy / OpenCV，不加空间平滑。

---

## 2. 坐标约定

- 所有行级返回量（`position`、`candidate_px`、`final_px`、`band_left_px`、
  `band_right_px`、`band_center_px`）都是 **ROI 内的局部 x 像素坐标**（半开列标，
  可含亚像素小数）。
- ``RowLocalization.row_px`` / ``EdgeLocalization`` 的 ``height``、``valid_local_rows``
  和统计中的行数都是 **ROI 内的局部行号**。跨边配对时，通过在
  ``roi_bounds_px[1]``（y 原点的全局行号）上加局部行号还原为**全局工作图像行号**
  再求交集（见 ``pair_localization_stats``）。
- ``roi_bounds_px`` 为半开 ``(left, top, right, bottom)`` 图像像素矩形。
- 背景拟合的特征基是**归一化的全局坐标**：
  `u = (col - (W-1)/2) / max((W-1)/2, 1)`、`v = (row - (H-1)/2) / max((H-1)/2, 1)`，
  与 ROI 无关。因此非零偏移 ROI 也正确，且校正**从不累积**（调用方必须始终传
  **未校正**的原图像）。

---

## 3. 精确 API

### 3.1 过渡定位

```python
transition_row_position(profile, reference_x, polarity_sign, polynomial_degree,
                        fit_bounds=None) -> (position_or_NaN, reason)
```

- `profile`：单行一维灰度。
- `reference_x`：自动局部拟合窗口中心（亚像素）。
- `polarity_sign`：`+1`（上升/亮）或 `-1`（下降/暗）；非有限或 0 视为无效。
- `polynomial_degree`：仅 `2/3/4`，其它返回缺失。
- `fit_bounds`：可选 `(start, end)` 半开像素窗，唯一用来做受限重定位的杠杆；
  长度必须恰为 2 且两端有限，否则返回缺失。右边界窗会被 clamp 到宽度和
  `width-1` 的差分长度，绝不产生长度不匹配。

### 3.2 亮带定位

```python
bright_band_crossings(profile, reference_local, polynomial_degree) -> (left, right, contrast) | None
bright_band_row_position(profile, definition, reference_local, *, polynomial_degree) -> (position_or_NaN, reason)
```

- `reference_local`：预览亮边在 ROI 内的局部 `(left, right)` 半开列。
- `definition`：`"left" | "right" | "center"`。
- `position_function` / `crossing_function` / `search_bounds_function` 仅向后兼容接
  受，**本模块不使用**它们，绝不委托 discovery 模块的线性插值 helper。

### 3.3 边级驱动

```python
localize_transition_edge(region, roi_bounds_px, reference_x, polarity_sign,
                         polynomial_degree, polarity_label, max_jump_px=12.0,
                         row_mask=None, relocalize=True) -> EdgeLocalization
localize_bright_band_edge(region, roi_bounds_px, definition, reference_local, *,
                          max_jump_px=12.0, contrast_threshold=None, relocalize=True,
                          polynomial_degree=3) -> EdgeLocalization
```

- `row_mask`：仅自动发现的对比预选**选择掩码**（每 ROI 行一个布尔值）；被掩码的行
  记为 `low_contrast`，**永不**作为主候选，也**永不**被重定位阶段重试。手动 ERA
  不传掩码，保持原行为。
- 亮带默认每行对比门槛 = ROI 的 P75 行对比 × 0.2。

### 3.4 统计与背景

```python
localization_stats(diag) -> LocalizationStats
pair_localization_stats(left_diag, right_diag) -> PairLocalizationStats
missing_row_indices(diag) -> list[int]
gap_runs(valid) -> list[(start, end)]
valid_row_threshold(height) -> int          # max(30, ceil(0.6 * height))
failure_reason_summary(stats, pair=None) -> str
polynomial_features(rows, cols, degree) -> np.ndarray   # degree 1 or 2
fit_additive_background(image, degree=1, region=None) -> BackgroundModel
```

`BackgroundModel` 提供 `background`、`corrected_image`、`coefficients`、`fit_mean`、
`fit_std`、`label`。

---

## 4. 行级 stage / reason / fields

`RowLocalization` 字段：

| 字段 | 含义 |
|------|------|
| `row_px` | ROI 局部行号 |
| `candidate_px` | 主候选位置（局部 x，缺失为 NaN） |
| `final_px` | 最终位置（主候选或重定位后；缺失为 NaN） |
| `stage` | `primary` / `relocalized` / `missing` |
| `reason` | 失败原因字符串，成功为空 |
| `recovered_from_px` | 重定位时使用的邻域种子（否则 NaN） |
| `band_left_px` / `band_right_px` / `band_center_px` | 亮带两交点与中点（过渡行为 NaN） |

`stage` 值：`STAGE_PRIMARY="primary"`、`STAGE_RELOCALIZED="relocalized"`、
`STAGE_MISSING="missing"`。

`reason` 值：`REASON_NONE=""`、`REASON_LOW_CONTRAST="low_contrast"`、
`REASON_MISSING_FLANK="missing_flank"`、`REASON_AMBIGUOUS="ambiguous_peak"`、
`REASON_JUMP_REJECTED="jump_rejected"`、`REASON_UNRESOLVED="unresolved"`。

**重定位与重试规则**（主候选与重定位使用**同一套门槛**）：

- `row_mask` 掩码行与 `low_contrast` 行**永不**重试恢复。
- `jump_rejected` 行不重试（真实大起伏的候选保留 `candidate_px` 真实值，但
  `final_px` 置 NaN，交由上层复核决定是否保留该跳点——本模块**不**合成平滑位置）。
- 重定位使用**冻结的首轮 seed 集合**（恢复得到的行不再作为 seed，**不链式**）；
  非 center 亮带用**真实 `band_center`**（与定义无关）做种子。
- 恢复后仍在原 ROI、同一目标身份，且**只一次**；无法安全恢复就 NaN。

---

## 5. 过渡定位有效性判据（明示记录）

在差分质心**之前**施加两个硬门槛（首轮与重试相同，无 Scharr/Sobel 核、无空间平滑）：

1. **纯噪声门槛**：窗口端点对比（两端外侧样本中位数的差）低于
   `TRANSSITION_ENDPOINT_NOISE_FACTOR(=3.0) × 稳健窗口噪声` 时，判定为纯噪声窗口，
   返回 `low_contrast` 缺失——绝不用随机多项式给真实轨迹补点。稳健噪声由窗口二阶
   差分的 `median(|d²|)/(√6·0.6745)` 估计（与 `_flank_noise` 同一估计器）。
2. **双阶跃/多过渡门槛**：正差分窗内出现 ≥2 个**经过净阶跃确认的**同向物理阶跃，
   且两阶跃**足够分离**，同时差分质心候选根落在两阶跃之间近似平坦的中间平台上——
   此时返回 `ambiguous_peak` 缺失，不返回两物理阶跃间的差分质心（假边）。
   判据（首轮与重试相同，只读当前行原始灰度，无邻行、无隐藏平滑、不改变最终拟合
   输入）：
   - 噪声门槛 `TRANSSITION_NOISE_FACTOR=3.0`、窗最大值的 `0.08` 相对下限；
   - 候选过渡组分组间隔 `TRANSSITION_GROUP_GAP_PX=4` px（更近的最大值并入同一组）；
   - **净阶跃确认** `_sustained_step_contrast`：候选峰两侧各
     `TRANSSITION_STEP_CONFIRM_WINDOW_PX(=5)` 样本的稳健中位数在极性方向上的净差须
     ≥ `TRANSSITION_STEP_CONFIRM_NET_FACTOR(=3.0)` × 灰度域窗口噪声，否则视为平坦区
     的孤立噪声尖峰而丢弃（其中位数对比≈0）；
   - 只取**两个净阶跃最强**的候选阶跃；二者差分峰索引间距须
     ≥ `TRANSSITION_MIN_STEP_SEPARATION_PX(=7)` px（物理阶跃 8 px 相隔时差分平板
     argmax 的离一偏差已换算进去）；
   - 两个候选阶跃判「双阶跃」的条件是**二选一**（OR）成立：
     - **候选根平坦判据**：差分加权质心（幂 4，与下游 ERA 拟合用同一权重）落在两
       阶跃之间，且该点 ±`TRANSSITION_ROOT_FLAT_WINDOW_PX(=3)` 窗内中位
       `|一阶差分|` / 差分峰强度 ≤ `TRANSSITION_ROOT_FLAT_FRACTION(=0.30)`——即候选
       根恰坐在两物理阶跃之间近似的中间平台上，而不是仍持续有净坡度的宽单过渡上；
     - **阶跃实质占比判据**：较弱一步的净阶跃对比 / 端点对比
       ≥ `TRANSSITION_MIN_STEP_NET_FRACTION(=0.28)`，即两步都大到不可能是斜坡上的
       噪声涟漪；
   - 否则（只有一处强阶跃、或候选根仍在持续斜坡上）按单阶跃正常定位。

   **本次返工修复的根因（如实）**：上一版的
   `TRANSSITION_STEP_STRENGTH_FACTOR(=8.0)` 差分峰强度门槛是**错误的修复**。它要求
   每个物理阶跃的差分峰 ≥ 8×差分域噪声（≈8×√2×灰度噪声），但对「两个幅度各 80、
   x=12 与 x=23」的真实双等阶跃，每步真差分峰约 20，而 8×差分噪声约 22.6，于是两个
   真实阶跃被判为「不满足严格双峰门槛」，`_multiple_transitions` 直接返回 False，
   把两阶跃间的差分质心（假中点 ~17.5）当作单阶跃位置放行。主管已独立运行 32 项
   全过后**实测复现**：σ=2、seed=184 时 92/100 行有效且中位 17.575（实为 x=12/23
   两条真实边之间的假中点）。上一报告声称「保守不造假点」没有证据且与实测相反。
   本返工**删除**该 8σ 门槛，改用上述「净阶跃确认 + 足够分离 + 候选根平坦/阶跃实质
   占比」双判据：对同一 σ=2、seed=184 复现输入，现在 100 行全部 `ambiguous_peak`
   缺失，**零假中点**。

  局部归一化多项式 0.5 位置定义保持不变，不更换模型、不插值补点。

---

## 6. 背景校正前后顺序（由调用方选择）

`fit_additive_background` 只负责从**原始工作图**拟合并返回 `corrected_image`；它
**从不**就地修改、**从不**累积。调用方负责决定在管线的哪一步使用背景校正结果：

- 一次低阶背景拟合，覆盖整个工作图或被选 ROI；背景在整图上求值，再**恢复区域均值**
  （`corrected = image - (background - fit_mean)`）以保持强度水平与局部半高基线含义。
- 稳健迭代（Tukey biweight，纯 NumPy）对亮/暗结构稳健，避免吸收真实周期结构。
- 后续定位仍以**原始未校正图**为输入；若调用方要结合背景校正，必须自行把
  `corrected_image` 作为新的「原始」图传给定位驱动，且不得在同一图上重复叠加校正。

---

## 7. 实际测试（test_line_localization.py，35 项，全部通过）

跨模块覆盖：两个已验收修复、双等阶跃拒绝、单 logistic 半高精度与亮暗极性、
纯噪声/缺失/邻边不误恢复、右边界 `fit_bounds` 不 shape 错、degree/polarity 验证、
三种亮带定义与 degree 2/3/4 两半高中点、一例受限恢复真正读取当前行像素、明显跳点
candidate 真实而缺行 NaN、左右/共同有效统计分母与最长 gap、背景 offset plane/
quadratic 平坦化与条纹几何保留、亮带峰值身份歧义（宽对称参考两根等强峰 → NaN +
`ambiguous_peak`、窄参考 `(16,20)` 只选 18、窄参考 `(30,34)` 只选 32、整边歧义诊断
传播、窄参考整边不被误判歧义），本次返工保留的宽缓过渡噪声回归（宽单阶跃
slope 1/3/6 不再被误判双阶跃、多 seed 与 sigma 2/4 的有效率 ≥80%、暗极性覆盖、uint8
量化覆盖率、误差随宽度单调、纯噪声与极缓过渡不造假点、双同向阶跃 gap 8/11/16 与
整体平移双极性拒绝），以及本次返工新增的**有噪双阶跃回归**（`_noisy_double_step`
helper：斜 logistic 双等阶跃 x=12/23 幅 80，σ=2/4 × 亮/暗 × seed 184/1/42 的 float
与 uint8，直接断言 `final_px` 落点距真实物理边 12/23 而非理由字符串；σ=2 与
seed 184 全零假中点，σ=4 仅 seed 1/42 各 ≤2/≤1 假中点且如实下界）。运行方式：

```bash
cd 'lwr:ler'
/Users/yiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest test_line_localization
/Users/yiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m py_compile line_localization.py test_line_localization.py
```

---

## 8. 已知限制

- 过渡定位多项式度仅接受 2/3/4；背景阶数仅 1/2。
- 双阶跃判据使用固定常数（噪声 3σ、净阶跃 3σ、最小阶跃间距 7 px、候选根平坦
  0.30、较弱步净对比占比 0.28、分组 4 px）；若 SEM 噪声特性显著变化，需在一致地
  改动主候选与重试两处门槛后重跑本测试。
- **剩余局限（如实）**：斜 logistic（slope 1）双等阶跃的中间平台本身就是一段
  ~4 px/px 的缓坡（两个 logistic 尾重叠），与 slope~12 的单宽阶跃在单行灰度上
  **数学上不可区分**；故 σ=4 极端噪声下两阶跃可能被并成单一差分峰而漏判，残留
  ≤ 2/1200 行的假中点（σ=2 与 seed 184 全零）。σ=2 的 slope 12/20 极缓单宽阶跃
  也会有少量行（slope12 约 21/100、slope20 约 28/100）被如实报为 `ambiguous_peak`
  而非给出随机半高——这是「宁可缺失理由明确」而非「强求 100% 有效」。以上均未
  用隐藏滤波或邻行补点掩盖。
- 纯噪声与低对比统一记为 `low_contrast`（不重试）；双阶跃记为 `ambiguous_peak`
  （重试时以同样判据再次拒绝，最终仍缺失）。
- 本模块不做配对之外的 CD/LER/LWR 计算；那些公式（N−2 LER、N−1 LWR、均值宽度）
  由主线应用在配对后的公共全局行上计算，本模块只输出位置与统计。

## 9. 亮带峰值身份歧义（本次返工新增）

当一个宽参考带同时覆盖**两根等强、对称的亮带**（例如参考中心恰在两峰之间、
两峰与参考中心距离近乎相同、且对比显著可比）时，预览参考无法唯一标识目标身份，
行定位返回 NaN 且 reason 为 `ambiguous_peak`，**不再**静默取几何上第一根基峰。

- 判据（首轮与邻域重试**同一策略**，不更换位置定义与多项式，也不改变两翼半高
  位置定义）：
  - 候选基峰按「峰中心到参考中心的距离」选择最近者；
  - 若存在另一 eligible 基峰，其中心到参考中心的距离与最近峰的距离之差
    ≤ `AMBIGUITY_DISTANCE_TOLERANCE_PX = 1.5` px（明确的小像素容差），且其对比
    ≥ `AMBIGUITY_CONTRAST_RATIO = 0.50` × 最近峰对比，则判定为歧义 → `ambiguous_peak`。
- **不**一刀切拒绝所有含多个邻峰的行：当参考带窄到只覆盖一个已知峰（身份明确，
  例如 `(16, 20)` 只指向 x≈18、`(30, 34)` 只指向 x≈32）时，仍正常返回该峰位置。
- 歧义原因**完整传播**：`bright_band_row_position`、`localize_bright_band_edge` 的
  逐行 `RowLocalization.reason`（含整边诊断）都如实报告 `ambiguous_peak`，绝不用
  笼统的 `missing_flank` 掩盖真实原因。

## 10. 弱对比不等于定位必然可靠

对比高过噪声门槛只说明「该行存在可定位的过渡/亮带结构」，**不保证定位必然可靠**：
最终可靠性还取决于多项式两翼拟合结果、峰值身份唯一性（见第 9 节）、行内覆盖与
公共行覆盖率等后续检查。不要把「高于噪声一定可定位」当作绝对结论，也不要在文档
或报告中夸大该保证。
