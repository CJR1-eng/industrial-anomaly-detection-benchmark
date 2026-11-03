# 工业异常检测项目：研究归档总结（Day 1–4）

- **归档日期**：2026-11-03
- **处置**：**停止**"光照鲁棒 + 图像级分数聚合"作为论文主线
- **保留**：全部代码、配置、数据划分、日志、热力图（未删除任何文件）

---

## 一、已验证的研究结论（有真实运行数据支撑）

### 1.1 环境与工程

| 结论 | 证据 |
|---|---|
| PyTorch 2.14.1+cu126 + anomalib 2.7.0 在本机（RTX 3050 4GB）可用，PaDiM 显存峰值 1097.7 MB | `src/env_check.py`、Day 1 §2 |
| 位级可复现：同配置两次独立运行，6 条件×117 图逐图分数 Δ=0 | `results/metrics/day2_reproducibility.json` |
| anomalib 2.x 的 `Normalize` 由 PreProcessor 回调施加，直接遍历 dataloader 会绕过它 | `results/logs/audit_postprocess.log`（batch min=0.058 vs 显式管线 min=−1.807） |
| PostProcessor 是**全局仿射 + clamp[0,1]**（非逐图 Min-Max），会把 95.6% 像素压成 0、排序保持率仅 41% | `results/metrics/day2_audit.json` |

### 1.2 数据

| 结论 | 证据 |
|---|---|
| MVTec AD tile：230/33/84，掩码 84/84 对应，内容级泄漏 0 | `docs/data/dataset_check_tile.md` |
| AD2 sheet_metal：644 张，train 137 / val 19 / test_public 24+90（+90 GT） | `docs/data/ad2_check_sheet_metal.md` |
| AD2 的**光照条件编码在文件名**（regular/overexposed/underexposed/shift_1~3），每场景 6 变体且各有 GT，配对可靠 | 同上 |
| AD2 的 train/validation **全部是 regular 光照**，测试期光照变化即分布外输入 | 同上 |

### 1.3 现象层面

| 结论 | 数值 | 证据 |
|---|---|---|
| 真实光照下 PaDiM 图像级退化明显 | 全局 Image AUROC 0.6931；`shift_2` 正常图误报率 100% | `results/metrics/day3_per_lighting.csv` |
| 退化主要来自**照明方向变化**，不是曝光量 | 曝光条件 Δ≤±2 分、组内 AUROC 0.83–0.92；`shift_*` Δ 达 +10~+27 分 | Day 3 §4.2 |
| Max 聚合在宽幅图上**钉住钢带左右边缘** | 正常图 argmax 落在 10% 边界带内：squash 58.3%、aspect **79.2%**（中位距离 **0.0**） | `day4_extremes_bootstrap.json` |
| 极值随**采样位置数**单调膨胀（受控证据） | aspect：正常 max 36.74→63.21（+72%），缺陷–正常间隔 13.01→3.29 | 同上 |
| 同一批热力图，换聚合可大幅改变图像级指标 | aspect：Max 0.5477 → Global-mean 0.8403（分组 Bootstrap Δ+0.293，CI[+0.136,+0.476]） | `day4_aggregation.csv` |
| "裁边缘 2% + 均值"这一朴素修法优于任何单一聚合规则 | aspect 0.8894（较全图均值再 +0.049） | `day5_audit.json` |
| 四种简单预处理全部无效 | 亮度归一化 0.667 / CLAHE 0.558 / 固定 γ 0.648（基线 0.693） | `day3_ad2_results_all.csv` |

---

## 二、已被否定的研究假设

| 假设 | 否定依据 |
|---|---|
| **H1** "合成光照扰动会显著破坏 PaDiM" | Day 2：6 种合成条件 Image AUROC 仅降 0.0014–0.0177，Pixel AUROC 降 ≤0.0027，**新增误报 0** |
| **H2** "把分辨率提到 840 能显著提升像素指标" | Day 2 §3：256 与 840 两尺度差 ≤0.0002 |
| **H3** "光照失效主要是分数标定失效" | Day 4：aspect 下**组内** Global-mean AUROC 达 0.95–1.00，说明热力图本身可分，主因是 Max 的边缘极值（Day 3 归因不准，已在 Day 4 §一更正） |
| **H4** "Max 的极值偏差可以用 Top-K 修好" | Day 4：squash 下 Top-1%/5%/P99 全部**更差**（−0.019~−0.027）；仅 aspect 有收益 |
| **H5** "存在一个普适更优的聚合规则" | 三配置最优各不相同：aspect→Global-mean、squash→Max、tile→Top-5% |
| **H6** "聚合机制值得作为方法论文主线" | Day 5：`裁边缘2%+均值`两行代码达到 0.8894，超过所有"方法"，无方法空间 |
| **H7** "简单预处理可以修复光照退化" | Day 3 §6：四种方法全部无效或更差 |

---

## 三、尚未证实的推测（不得当作结论引用）

1. "光照方向变化引发镜面反射改变"——物理机制合理，但**我们没有任何反射测量或光照方向元数据**。
2. "在真实产线上光照退化会更大/更小"——本实验只有单类别、单批次、19 个场景。
3. "Top-K 会牺牲微小缺陷"——Day 4 数据显示**不成立**（aspect 下 Top-1% 对小缺陷反而更好）。
4. "换成 PatchCore 会不同"——**完全没测**。
5. "Fabric 或其它类别上结论不同"——Fabric 未下载（10GB，带宽限制）。
6. "裁 ROI 后重训会达到更高"——Day 5 的 0.8894 是在热力图上裁剪的近似，非重训结果。

---

## 四、与已有论文的直接重合点

| 我们的观察 | 已有论文 | 重合程度 |
|---|---|---|
| 光照导致已校准判定翻转 | **Beyond Benchmark Scores**（IEEE Access 14: 145972–145991, 2026）报告 30.9%–46.1% 判定被改变 | 🔴 完全重合 |
| Max vs Top-1% 聚合对照 | 同上（已做 PaDiM 风格方法的对照） | 🔴 完全重合 |
| 聚合方式影响图像级指标 | 同上（image-score aggregation 为显式审计维度） | 🔴 完全重合 |
| 高分辨率 AD 的算力分配 | **HiAD**（arXiv 2508.12931, 2025，有代码）双分支 + 检测器池 + 自适应分配 | 🔴 完全重合 |
| AD2 高分辨率 + 光照 + 分箱二值化 | **RoBiS**（2025）滑窗裁块 + 光照增广 + SAM 精修 | 🔴 完全重合 |
| 光照感知特征增强 | **LL-IAD**（JCDE 12(5), 2025，DOI 10.1093/jcde/qwaf043） | 🔴 重合 |
| 频域滤波提升噪声/鲁棒性 | **WEDGE-Net**（Sensors 26(7): 2154, 2026） | 🔴 重合 |

---

## 五、可复用资产

### 5.1 代码

| 文件 | 作用 | 复用价值 |
|---|---|---|
| `src/eval_core.py` | 确定性设置、管线对齐、原始分数推理、双尺度指标、AUPRO | ⭐⭐⭐ 任何 AD 实验直接用 |
| `src/ad2_eval.py` | searchsorted 加速 AUPRO、大分辨率像素指标 | ⭐⭐⭐ 高分辨率评估必备 |
| `src/run_ad2.py` | AD2 索引/拟合/推理/分光照与配对统计 | ⭐⭐⭐ |
| `src/day4_aggregation.py` | 受控聚合实验（哈希断言）、极值诊断、分组 Bootstrap | ⭐⭐ |
| `src/dataset_check.py` | 掩码对应、内容级泄漏检查 | ⭐⭐ |
| `tools/parallel_download.py` + `patch_parts.py` | 官方源多连接下载（15KB/s → 1.8MB/s），断点续传 | ⭐⭐⭐ 后续下载任何大数据集 |
| `src/env_check.py` | 环境快照（含 GPU 实测算力） | ⭐⭐ |

### 5.2 数据

| 资产 | 位置 | 说明 |
|---|---|---|
| MVTec AD tile | `datasets/raw/mvtec_ad/tile` | 230/33/84，校验通过 |
| AD2 sheet_metal | `datasets/raw/ad2/sheet_metal` | 644 张，6 光照条件，含 90 张 GT |
| 划分清单 | `results/metrics/day2_split_manifest.csv` | 393 行，含 label 与路径 |
| 热力图缓存 | `results/metrics/day4_maps/*.npz`、`day2_maps.npz` | 可直接做聚合/校准实验 |

### 5.3 经验教训（写进未来任何论文的 baseline 章节）

1. **宽幅工业图必须先裁 ROI 再喂网络**，否则 Max 会钉在工件边缘，图像级指标被系统性低估。
2. **评估必须用未截断的原始分数**；框架的 PostProcessor 归一化会毁掉排序（Pixel AUROC 0.905→0.589）。
3. **像素级指标必须报告栅格化口径**：同一批热力图，掩码在 256 与 1024 下 AUPRO 相差 0.057（相对 34%）。
4. **anomalib 的 Normalize 在回调里**，自己写推理循环必须手动补上。
5. **统计要以场景/工件为单位做 Bootstrap**，不能把几十万像素或同一场景的 6 个光照版本当独立样本。

---

## 六、为什么当前选题不继续投入

| 理由 | 证据 |
|---|---|
| 现象已被同数据集论文量化 | IEEE Access 2026 已报告 30.9%–46.1% 判定改变，并做过 Max vs Top-1% 对照 |
| "解法"是工程常识 | 裁边缘 2% + 均值即达 0.8894，无需新方法 |
| 不存在普适规则 | 三个配置的最优聚合方向相反，做不出可推广方法 |
| 统计功效不足 | 19 场景分组 Bootstrap，CI 宽达 0.4；每光照条件仅 4 张正常图 |
| 相关高分辨率子问题也被占 | HiAD（2025，有代码）、RoBiS（2025，AD2 专用） |

**结论：这条路能产出一份扎实的负面结论，但不能产出一篇有方法贡献的论文。**

---

## 七、归档完整性

- 实验文件**全部保留**，未删除任何 `results/`、`figures/`、`datasets/` 内容。
- 本总结只收录**已验证结论**；探索性（看过 test_public 的结果）一律在正文标注，
  正式验证需冻结方法后在独立数据或官方私有评测上做。
- 原始逐日记录：`docs/Day1_Report.md`、`docs/Day2_Report.md`、`docs/Day2_Errata.md`、
  `docs/Day3_Report.md`、`docs/Day3_Novelty_Matrix.md`、`docs/Day3_Decision.md`、
  `docs/Day4_Report.md`、`docs/Day4_Aggregation_Novelty.md`、`docs/Day4_Decision.md`。
