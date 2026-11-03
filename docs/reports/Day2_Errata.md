# Day 2 勘误：Day 1 报告中的错误、未证实解释与修正依据

> 本文件**不修改** `docs/Day1_Report.md`，只追加勘误。所有数字均来自 Day 2 的
> `src/audit_postprocess.py` 实跑（日志 `results/logs/audit_postprocess.log`，
> 结构化结果 `results/metrics/day2_audit.json`）。

---

## E1【严重】Day 1 的"原始分数"是在**未归一化**的输入上算出来的

### 原表述

> Day 1 四、4.1：「`src/run_padim.py` 改为直接取 `PadimModel` 的原始输出（`infer_raw()`，
> 与 `predict_step` 内部调用完全一致，只是跳过 post_processor）。」
> 指标表：Image AUROC 0.8445 / Pixel AUROC 0.8387。

### 实际发生了什么

anomalib 2.x 把图像预处理**拆成了两半**：

| 环节 | 位置 | 作用 |
|---|---|---|
| `uint8 → [0,1]` | 数据集 `__getitem__` | 仅类型转换 |
| `Resize(256, bilinear, antialias)` | 注入到数据集 `augmentations`（`_update_augmentations`） | 在 DataLoader 内完成 |
| `Normalize(ImageNet)` | **模型 `PreProcessor`（Lightning 回调）** | 只在 Trainer 的 train/val/test/predict 循环里施加 |

Day 1 的 `infer_raw()` 直接 `for batch in dataloader:` 然后 `model.model(batch.image)`，
**绕过了 PreProcessor 回调**，所以模型实际收到的是"已 resize 但未归一化"的 [0,1] 张量。

### 证据（同一张图、同一模型）

| 检查项 | 数值 |
|---|---|
| fit 前 `dataset[0].image` | 840×840，min=0.0627，max=1.0000 |
| fit 后 DataLoader batch | **256×256，min=0.0578，max=1.0000**（归一化后必然出现负值） |
| 显式重建管线 `TRANSFORM(u8)` | 256×256，**min=−1.8069，max=1.7143** |
| batch[0] 与显式管线的最大绝对差 | 1.878（>1，量级完全不对） |
| 同一张图的图像分数 | batch 口径 **77.684** vs 显式管线 **116.719** |
| 同一张图热力图最大绝对差 | 39.155 |

### 影响量化（117 张测试图、同一模型、同一掩码）

| 模型输入 | Image AUROC | Pixel AUROC | Pixel AP |
|---|---|---|---|
| 未归一化 [0,1]（**Day 1 实际**） | 0.8442 | 0.8387 | 0.2762 |
| 已归一化（正确） | **0.9426** | **0.9047** | **0.3279** |
| 差值 | +0.0985 | +0.0660 | +0.0517 |

复现校验：Day 1 报告的 0.8445 / 0.8387 与本表第一行 0.8442 / 0.8387 一致（Image 差异 3e-4，
来自掩码缩放方式，见 E4），**说明勘误定位准确**。

### 修正结论

Day 1 的"第一版指标"**系统性低估**了 PaDiM：
Image AUROC 被低估约 **0.10**，Pixel AUROC 被低估约 **0.07**。
真值应为 Image AUROC ≈ 0.943 / Pixel AUROC ≈ 0.905。

---

## E2 Day 1 对 PostProcessor 差异的解释是**错的**

### 原表述

> 「anomalib 在 predict 之后会施加 PostProcessor 的归一化（anomaly_map 被压到 [0,1]，
> **且逐图重标定**），会破坏跨图可比性。」

"逐图重标定"（per-image min-max）这个说法**没有依据，且与源码不符**。

### 实际情况（源码 + 数值双重确认）

`anomalib/post_processing/post_processor.py` 的 `_normalize`：

```python
threshold = where(isnan(threshold), (norm_max + norm_min) / 2, threshold)
normalized = clip((preds - threshold) / (norm_max - norm_min) + 0.5, 0, 1)
```

即**全局仿射变换 + 截断**，三个统计量全部来自**验证集**，与批次内其他图像无关：

| 统计量 | 值 | 来源 |
|---|---|---|
| `image_min` | 17.118 | 验证集 46 张正常图的 MinMax |
| `image_max` | 65.500 | 同上 |
| `image_threshold` | 65.500 | 验证集 F1AdaptiveThreshold（无正样本 → 取最大分数） |
| `pixel_min` | 7.469 | 验证集所有像素的 MinMax |
| `pixel_max` / `pixel_threshold` | 65.500 | 同上 |
| 归一化后阈值 | 0.5（= 1 − sensitivity） | 固定 |

公式复现误差：图像 **2.6e-8**、像素 **0.0**（即完全吻合）。

### 真正的原因：clamp 截断造成大量并列（ties）

| 现象 | 数值 |
|---|---|
| 图像分数唯一值 | 117 → **47**（70 个值被压成同一值） |
| 归一化后排序与原顺序一致比例 | **41.0%** |
| 像素值被压到 0 的比例 | **95.6%** |
| 像素值被顶到 1 的比例 | 0.1% |

`clip` 单调不减但**不严格单调**，把大量不同分数映射成同一常数。
并列对在 AUROC 中按 0.5 计分，于是：

| 指标 | 原始 | 后处理 | 变化 |
|---|---|---|---|
| Image AUROC | 0.9426 | 0.8849 | **−0.058** |
| Pixel AUROC | 0.9047 | 0.5889 | **−0.316** |
| Pixel AP | 0.3279 | 0.1334 | −0.195 |

### 修正结论

问题不是"逐图重标定"，而是 **`mean+3σ` 式的验证集统计量 + 硬 clamp 导致的饱和**。
`pixel_threshold` 落在 65.5（验证集最大像素分数），而测试图异常像素距离普遍远高于此，
因此 `(x−65.5)/58.0+0.5` 对绝大多数背景像素给出负值 → 全部截断为 0 → 像素级排序崩塌。

**评估必须使用未截断的原始马氏距离**；PostProcessor 的归一化只适合"给人看的热力图"，
不适合算 AUROC。

---

## E3 Day 1 把"两套指标接近"误判为"交叉验证通过"

### 原表述

> 「两者 Image 指标接近、Pixel 指标差异大，差异来源已定位为后处理归一化。」

实际是两个**互相独立的错误**恰好抵消了 Image 指标的差距：

| 数值 | 真实成因 |
|---|---|
| Day 1 官方 evaluator 0.8849 | 输入归一化**正确**，但分数被 clamp 扭曲（真值 0.9426） |
| Day 1 自算 raw 0.8445 | 分数**未截断**（正确），但输入**没归一化**（真值 0.9426） |

两个 0.88 / 0.84 看起来很接近，纯属巧合，掩盖了两个真实缺陷。
Day 2 修正后：正确管线 → 0.9426；官方管线 → 0.8849。

### 修正结论

"两套指标一致"不能作为实现正确的证据。**必须验证输入张量本身**（形状、范围、
是否归一化），而不是只比较最终指标。

---

## E4 Day 1 的 256 掩码是**双线性缩放**的，不是最近邻

anomalib 的 `_update_subset_augmentations` 把模型的 `Resize(256, bilinear, antialias)`
追加到数据集 `augmentations`，该变换**同时作用于图像和掩码**。因此 Day 1 的 256 掩码
是"双线性插值后 >0.5 二值化"，而非任务要求的最近邻。

影响：细裂纹的掩码面积与形状会有差异（`tile` 裂纹本身只有几个像素宽），
使异常像素占比与 Pixel AP 产生小幅偏移（Day 1 与 Day 2 同口径下 Image AUROC 差 3e-4）。

### 修正

Day 2 起：
- **256 评估**使用 `nearest` 缩放的掩码（符合任务要求）
- **840 评估**直接使用原始掩码，**不做任何插值**（最干净）
- 分数图上采样用双线性（连续插值，只在 840 评估时使用，且不改变排序）

---

## E5 Day 1 的阈值规则（mean+3σ）实际落在验证集分布之外

验证集 46 张正常图的原始分数：mean=25.97、std=4.09 → `mean+3σ = 38.25`。
而 PostProcessor 记录的 `image_max = 65.50`（验证集最大分数），说明
**存在正常样本的分数远高于 mean+3σ**，即分数分布右偏严重，`mean+3σ` 并不构成
"3σ 置信"意义上的异常判定边界。

Day 1 因此出现矛盾的组合：阈值 38.25 下 FP=2、FN=38（保守），
而官方阈值 65.5 下几乎全判为异常（激进）。

### 修正

Day 2 仍沿用 `mean+3σ` 作为**唯一的固定阈值**（因为它只依赖验证集、可复现），
但在报告中明确它**不是**统计意义上的 3σ 边界，并同时报告 FPR/Recall 而不只报 F1。
后续应改为"验证集分位数（如 p95/p99）"并做敏感性分析。

---

## E6 Day 1 的"0.01 波动"根因未查明，Day 2 单独验证

Day 1 两次运行 Image AUROC 为 0.8445 / 0.8553。当时归因为"种子未完全固定"。
Day 2 已加入完整确定性设置（见 `Day2_Report.md` 第 3 节）并实测重复性，
结果见报告；本勘误只指出：**Day 1 未归一化的输入会放大数值敏感度**，
不能排除"波动部分来自输入分布异常"这一可能。

---

## E7 其它已在 Day 1 内修正、但需留痕的技术错误

| 问题 | 现象 | 修正 |
|---|---|---|
| `np.trapz` 在 NumPy 2.x 被移除 | AUPRO 计算抛异常，指标缺失 | 改 `np.trapezoid` |
| `TestSplitMode('FROM_DIR')` | 枚举值大小写错误，直接抛 ValueError | 改小写字符串 `from_dir` |
| `predict(dataloaders=…, datamodule=…)` 同时传 | Lightning 抛 MisconfigurationException | 只传 `dataloaders` |
| `val_split_mode=none` 时仍请求 val_dataloader | `AttributeError: no attribute 'val_data'` | `limit_val_batches=0` |
| 中文图标题渲染成方块 | matplotlib 缺 CJK 字体 | 图表文字改英文 + 配置 Microsoft YaHei |
| 按文件名判数据泄漏 | 误报 33 个"泄漏"（train/test 独立编号） | 改内容 MD5 比对，真实泄漏 0 |

---

## 仍然成立的 Day 1 结论（未受影响）

1. **环境**：Python 3.12.7 + torch 2.14.1+cu126 + anomalib 2.7.0，GPU 实测可用，显存峰值 ~1.1 GB。
2. **数据集**：官方渠道获取，SHA256 留痕，230/117/84 计数正确，掩码 84/84 对应，内容级泄漏 0。
3. **划分设计**：184 拟合 / 46 定阈值 / 117 测试，规避了 `SAME_AS_TEST` 的测试集泄漏。
4. **工程结论**：PyPI Windows 版 torch 是 CPU-only、pip 依赖解析会卡死、HF 需镜像 —— 均已在 Day 1 报告中记录且经 Day 2 复用验证。
5. **诊断 3（空间对齐）**：热力图与 GT 无系统性错位（最佳平移中位数 (0,0)），该结论建立在**未归一化**输入上，Day 2 用修正管线复核见报告。
