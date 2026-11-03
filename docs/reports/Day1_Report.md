# Day 1 实验报告：环境搭建 + MVTec AD `tile` + PaDiM 基线链路验证

- **日期**：2026-11-02
- **研究方向**：面向光照变化与微小缺陷的轻量级工业异常检测
- **本日目标**：打通"环境 → 数据 → 基线 → 指标 → 可视化"整条链路，不追求性能
- **项目根目录**：仓库根目录（路径略去，见 README）

> **结论先行**：链路已跑通（GPU 训练+推理+指标+10 张可视化），但**第一版指标低于 PaDiM 论文水平**，
> 已完成 3 项归因诊断，定位出 2 个真实问题（后处理归一化破坏像素排序、留出验证集损失 20% 拟合样本）
> 和若干待验证因素（评估分辨率、blur、骨干规模）。**当前数值只能作调试基线，不可写入论文。**

---

## 一、实验环境及软件版本

### 1.1 硬件

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU，4095.5 MB 显存，sm_86 |
| 驱动 / CUDA UMD | 610.62 / 13.3 |
| CPU | 12th Gen Intel Core i5-12500H，16 逻辑核 |
| 内存 | 15.7 GB |
| 磁盘 | D 盘可用 128 GB |
| OS | Windows 11 (AMD64) |

### 1.2 软件栈（关键版本）

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12.7 | 项目内 `.venv` |
| PyTorch | **2.14.1+cu126** | CUDA 12.6 构建，wheel 约 2.5 GB |
| torchvision | **0.29.1+cu126** | 与 torch 2.14 配对 |
| cuDNN | 91002 | 随 torch 分发 |
| Anomalib | **2.7.0** | 官方实现，未改动算法 |
| Lightning | 2.6.6 | anomalib 2.x 依赖 |
| timm | 1.0.30 | 骨干网络加载（经 HuggingFace） |
| NumPy | 2.5.4 | ⚠️ 注意 2.x 移除了 `np.trapz` |
| OpenCV | opencv-python-headless 5.0.0.93 | anomalib 要求 headless 版 |
| Matplotlib | 3.11.2 | |
| scikit-learn / scikit-image / scipy | 1.9.1 / 0.26.0 / 1.18.1 | |
| torchmetrics | 1.9.0 | |
| Pillow / pandas / PyYAML | 13.0.0 / 3.1.0 / 6.0.3 | |

完整依赖快照：`docs/env/requirements_export.txt`
机器可读环境报告：`results/logs/env_report_20261102_181038.json`
人可读摘要：`results/logs/env_report.md`

### 1.3 环境搭建中踩到的坑（重要，会影响后续复现）

1. **PyPI / 清华源上的 `torch` 是纯 CPU 版**。Windows 平台 PyPI 的 `torch-2.x-cp312-win_amd64.whl` 只有约 120 MB，CUDA 版仅存在于 `download.pytorch.org/whl/cu126`。用清华源装 torch 会得到 CPU 版，GPU 直接不可用。
   → **结论：torch/torchvision 必须从 PyTorch 官方索引安装；其余纯 Python 依赖走清华源。**
2. **`--index-url` 指向 PyTorch 索引会导致依赖解析卡死**。该索引不含 `sympy`/`jinja2` 等依赖，pip 在解析阶段挂起（进程 CPU/IO 均为 0，静默卡死约 10 分钟）。
   → **解法：先从官方索引下 wheel，`pip install --no-deps` 安装，再单独从清华源补依赖。** 该 wheel 已留档于 `vendor/wheels/`。
3. **HuggingFace 直连返回 502（代理问题）**。anomalib 2.x 通过 timm 从 HF 拉取 ImageNet 骨干权重，`httpx2.ProxyError: 502`。
   → **解法：`HF_ENDPOINT=https://hf-mirror.com`**，已写入 `configs/padim_tile.yaml` 的 `env` 段，脚本启动时自动设置。权重本身仍来自官方 timm 仓库。
4. **官方数据集服务器单连接限速约 45 KB/s**（mydrive.ch）。335 MB 的 `tile.tar.xz` 单线程需 2 小时以上。
   → 应对：`tools/parallel_download.py`（多连接分段）可提升到 ~600 KB/s；本次最终由人工下载完成。

---

## 二、GPU 是否成功启用

**成功启用**，且做了"真算"验证而非仅 `import torch`：

| 验证项 | 结果 |
|---|---|
| `torch.cuda.is_available()` | True |
| 设备名 / 算力 | NVIDIA GeForce RTX 3050 Laptop GPU / sm_86 |
| fp32 矩阵乘 (2048×2048) | 通过，结果和 54225.07，峰值显存 56.1 MB |
| fp16 矩阵乘 | 通过（max 219.125） |
| PaDiM 训练实际占用峰值 | **1097.7 MB**（4 GB 显存余量充足） |
| 训练设备确认 | Lightning 日志 `GPU available: True (cuda), used: True`、`CUDA_VISIBLE_DEVICES: [0]` |

复现命令：`python src/env_check.py`（退出码 0 表示 CUDA 可用）

---

## 三、数据集：MVTec AD / `tile`

### 3.1 来源（合规性说明）

- **官方入口**：<https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- **分类下载页**（本次使用）：<https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads>
- 官方 `tile` 直链：`https://www.mydrive.ch/shares/163650/55ad2038770deff6fe6771bd784af8cc/download/420938133-1629960456/tile.tar.xz`
- **未使用任何第三方镜像**，数据来自 MVTec 官方分发通道。
- 许可：CC BY-NC-SA 4.0（仅限非商业研究用途，论文中需引用 Bergmann et al., CVPR 2019 / IJCV 2021）。
- 完整性留痕：`datasets/raw/tile.tar.xz`，335.0 MB
  `SHA256 = 392228153b96d0db9f179d3f7c2d63e131e907154883b57309fcbf798643173c`

### 3.2 目录结构

```
datasets/raw/mvtec_ad/tile/
├── train/good/                230 张（全部为无缺陷）
├── test/
│   ├── good/                   33 张（无缺陷）
│   ├── crack/                  17 张
│   ├── glue_strip/             18 张
│   ├── gray_stroke/            16 张
│   ├── oil/                    18 张
│   └── rough/                  15 张
└── ground_truth/               84 张 *_mask.png（与 5 类缺陷图一一对应）
```

### 3.3 校验结果（`python src/dataset_check.py --category tile`）

**结论：通过**（0 错误）

| 校验项 | 结果 |
|---|---|
| 训练图 / 测试图 / 掩码 | 230 / 117 / 84，与官方发布一致 |
| 缺陷图与掩码对应 | 84 ↔ 84，**缺失 0、尺寸不一致 0、全黑 0** |
| 图像分辨率 | 全部 840×840，RGB |
| 掩码分辨率 | 全部 840×840，与图像逐张一致 |
| 缺陷面积占比 | 均值 9.80%，中位 7.93%，范围 0.84%~35.49% |
| train 纯度 | 仅 `good` 子目录，无缺陷子类混入 |
| **数据泄漏（内容哈希比对）** | **0 组重复** |

> ⚠️ 一个易误判的点：`train/good/000.png` 与 `test/good/000.png` **同名但不同图**（各划分独立从 000 编号）。
> 按文件名比对会误报 33 个"泄漏"，必须按内容 MD5 比对。本报告已修正该逻辑。

校验报告：`docs/data/dataset_check_tile.md` / `results/metrics/dataset_check_tile.json`

### 3.4 数据划分与泄漏控制（关键）

MVTec AD **官方没有验证集**。而 anomalib 的 `MVTecAD` 默认 `val_split_mode=SAME_AS_TEST`，
即**验证集就是测试集**。若沿用默认配置用验证集选阈值，等价于在测试集上调参。

本次主动改为 **`val_split_mode=from_train`**：

| 划分 | 来源 | 数量 | 用途 |
|---|---|---|---|
| train | `train/good` 的 80% | **184** | 拟合高斯分布（建立正常特征） |
| val | `train/good` 的 20% | **46** | 仅用于确定判定阈值 |
| test | `test/*` 全部 | **117** | 仅用于最终评估，全程未参与任何拟合/选参 |

所有测试图从未进入特征分布建模，`val` 全部来自正常的训练图。

---

## 四、PaDiM 配置

| 项目 | 取值 | 说明 |
|---|---|---|
| 实现 | `anomalib.models.image.padim.Padim` | 官方实现，**未改动任何算法代码** |
| 特征提取器 | **resnet18**（ImageNet 预训练，经 timm/HF 获取） | 4 GB 显存友好；对应论文的轻量配置 |
| 特征层 | `layer1, layer2, layer3` | 输出 64×64，通道 64+128+256=448 |
| 降维 | `n_features=100` | PaDiM 论文对 resnet18 的给定值 |
| 输入分辨率 | 256×256 | anomalib 默认，未放大 |
| 批大小 | train 8 / eval 8 | 4 GB 显存下的保守值 |
| 精度 | fp32 | PaDiM 需协方差求逆，fp16 数值不稳 |
| 正则化 | 协方差对角 +0.01，求逆前 +1e-5 | anomalib 内部实现（与论文一致） |
| 高斯模糊 | sigma=4（核 33×33） | anomalib 默认 |
| 参数量 | 2.8 M | |
| 阈值 | 图像：val 得分 `mean+3σ` = **38.25**；像素：val 像素得分 99.5 分位 = **21.44** | 全部来自验证集 |
| 随机种子 | 42 | 见"问题 3"：目前仍有轻微不可复现 |

配置文件：`configs/padim_tile.yaml`；执行脚本：`src/run_padim.py`

### 4.1 一处必须记录的实现细节

`engine.predict()` 返回的 `anomaly_map` **已被 anomalib 的 PostProcessor 逐图归一化到 [0,1]**，
导致不同图像的异常分数失去可比性。实测：同一模型下

- 用框架返回的归一化图：Pixel AUROC = **0.5889**
- 用模型原始马氏距离图：Pixel AUROC = **0.8387**

即归一化让像素级指标**掉了一半**。因此 `src/run_padim.py` 改为直接取 `PadimModel` 的原始输出
（`infer_raw()`，与 `predict_step` 内部调用完全一致，只是跳过 post_processor）。
**该差异不是实现 bug，而是评估口径问题，后续所有方法对比必须统一口径。**

---

## 五、运行结果：成功

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 拟合高斯（184 张） | 30.8 s | 含骨干权重首次加载 |
| 验证集推理（46 张） | 12.1 s | 用于定阈值 |
| 测试集推理（117 张） | 11.6 s | |
| anomalib 官方 evaluator 复核 | 18.2 s | 交叉核对 |
| 显存峰值 | **1097.7 MB / 4096 MB** | 4 GB 卡无压力 |

运行日志：`results/logs/run_padim_day1.log`
结构化结果：`results/metrics/padim_tile_day1_summary.json`

---

## 六、第一版实验指标（调试基线，不可作论文结论）

### 6.1 主结果（协议 A：184 拟合 / 46 定阈值 / 117 测试）

| 指标 | 数值 | 口径说明 |
|---|---|---|
| **Image AUROC** | **0.8445** | 逐图得分 = 原始距离图最大值；无阈值 |
| Image AP (AUPR) | 0.9404 | 正类占 84/117，基線较高 |
| **Pixel AUROC** | **0.8387** | 全局池化，256×256 分辨率 |
| Pixel AUROC（逐图平均） | 0.8898 | 消除跨图尺度差异，更反映"图内定位"能力 |
| **Pixel AP** | **0.2762** | 异常像素仅占 7.0%，故 AP 明显低于 AUROC |
| AUPRO（FPR≤0.3） | 0.6791 | 按连通域计 overlap |
| F1@验证集阈值 (38.25) | 0.6970 | P=0.958 / R=0.548，46 TP / 2 FP / 38 FN |
| F1@测试集最优阈值 | 0.8705 | ⚠️**乐观口径，仅作参考，禁止写入论文** |

分数分布（直观可分性）：

| 类别 | min | 中位 | max |
|---|---|---|---|
| 正常（33 张） | 17.02 | 23.80 | 43.89 |
| 缺陷（84 张） | 21.17 | 42.44 | 141.47 |

> 正常与缺陷分布**重叠明显**（正常最大值 43.89 > 缺陷最小值 21.17），这与 Image AUROC 仅 0.84 一致。

### 6.2 交叉核对

| 口径 | Image AUROC | Pixel AUROC |
|---|---|---|
| 本脚本（原始距离，自算） | 0.8445 | 0.8387 |
| anomalib 官方 evaluator（含后处理归一化） | 0.8849 | 0.5889 |

两者 Image 指标接近、Pixel 指标差异大，差异来源已定位为后处理归一化（见 4.1）。

### 6.3 归因诊断（本日主要收获）

**诊断 1：后处理归一化破坏像素级排序**（已确认）
见 4.1，Pixel AUROC 0.589 → 0.839。

**诊断 2：留出 20% 做验证集，代价是 Image AUROC 掉约 7 个点**（已确认）
消融脚本 `tools/ablation_full_train.py`：

| 协议 | 拟合样本 | Image AUROC | Pixel AUROC | Pixel AUROC(逐图) | Pixel AP |
|---|---|---|---|---|---|
| A：留出 val（主实验） | 184 | 0.8553 | 0.8708 | 0.9079 | 0.3186 |
| B：用满 train/good（文献口径） | 230 | **0.9257** | 0.8207 | 0.8710 | 0.2767 |

结论：样本量对**图像级**判别影响显著（+0.07），对像素级反而略有下降（可能因高斯估计更"紧"）。
这意味着"为了能合法定阈值而留出验证集"是有真实精度代价的，需要在论文中明说。

**诊断 3：热力图与 GT 空间基本对齐，错位不是主因**（已确认）
`alignment_check`：84 张缺陷图，最佳平移量的中位数为 **(0, 0)**，零平移相关系数 0.473（最佳 0.556）。
说明不存在系统性错位，低 Pixel AUROC 主要来自"定位糊"（33×33 高斯核 + 64×64 特征网格），而非"定位偏"。

**尚未验证的剩余差距来源**（列在此处，避免只报好消息）：
- 评估分辨率：GT 原图 840×840，但指标算在 256×256。`tile` 的裂纹极细，840→256 降采样后仅剩亚像素宽度，被 sigma=4 模糊进一步抹平。论文口径通常在原分辨率评估，本报告尚未做 840 分辨率复算。
- 骨干规模：resnet18（2.8 M）明显弱于论文主表使用的 wide_resnet50_2。
- 单类别、单种子：仅 tile 一类，未做多种子重复。

### 6.4 可复现性提醒

两次完全同配置运行，Image AUROC 出现 0.8445 与 0.8553 的差异（约 0.01）。
说明**种子尚未完全固定**（数据集划分、DataLoader 顺序、cuBLAS 非确定性）。
正式实验前必须解决，否则指标波动会掩盖方法改进。

---

## 七、可视化结果

目录：`figures/day1_padim_tile/`（共 10 张，含总览）

| 文件 | 类型 | 说明 |
|---|---|---|
| `fig_00_overview_grid.png` | 总览 | 9 例拼接，N=正常 / A=异常 |
| `fig_01~03_normal_false_positive__good__*.png` | **正常样本误报** | 得分最高的 3 张正常图，可见纹理单元被误判 |
| `fig_04~06_defect_localized__*.png` | **异常正确检出** | glue_strip×2、crack×1，IoU>0.1 且得分高 |
| `fig_07~09_defect_missed__*.png` | **异常漏检** | gray_stroke×2、rough×1，得分低于阈值 |

每张为 4 联图：原图 / 热力图叠加 / 原始距离图 / GT 掩码（正常样本第 4 联为预测掩码）。
图表文字统一英文（便于直接用于论文）。

全部热力图张量（117 张，float16）：`results/metrics/padim_tile_day1_test_maps.npz`
逐图得分明细：`results/metrics/padim_tile_day1_per_image.csv`

---

## 八、任务完成状态

### ✅ 已完成

1. 环境配置：Python 3.12.7 + PyTorch 2.14.1+cu126 + torchvision 0.29.1+cu126 + anomalib 2.7.0，依赖版本全部留痕。
2. GPU 启用并实测（矩阵乘 + 显存峰值），非仅 import 通过。
3. 项目结构分层：`datasets/raw`（只读原始数据）/ `src` / `configs` / `results` / `figures` / `docs` / `tools`。
4. MVTec AD `tile` 通过官方渠道获取，SHA256 留痕，完整性+泄漏校验全部通过（0 错误）。
5. PaDiM 基线跑通：正常训练图拟合、测试推理、图像级得分、像素级热力图、耗时与显存记录。
6. 指标计算双路交叉核对（自算 + 框架），并定位口径差异来源。
7. 10 张可视化（正常误报 / 正确检出 / 漏检三类齐全）。
8. 3 项归因诊断 + 1 项样本量消融。

### ⚠️ 未完成

1. **尚未在 840×840 原分辨率复算像素级指标**（当前 256×256），与论文口径不完全可比。
2. 仅跑 `tile` 一类，未跑其它类别以确认链路通用性。
3. 未做多种子重复，可复现性问题未解决。
4. 未搭建"光照变化"实验框架——这是本课题的核心变量，今日未涉及。
5. 未做 Pixel AP / PRO 的其它实现口径交叉验证（仅一份实现）。

### ❌ 失败/受阻（含已解决与未解决）

| 问题 | 状态 |
|---|---|
| 清华源装 torch 得到 CPU 版 | 已识别，改用官方索引，**已解决** |
| pip 依赖解析静默卡死 ~10 min | 已定位，改 `--no-deps` 本地 wheel 安装，**已解决** |
| HuggingFace 骨干权重 502 | 已改用 `HF_ENDPOINT=hf-mirror.com`，**已解决** |
| `TestSplitMode('FROM_DIR')` 大小写错误 | 已修正为小写枚举值，**已解决** |
| `predict(dataloaders=…, datamodule=…)` 冲突 | 已改为只传 dataloaders，**已解决** |
| `np.trapz` 在 NumPy 2.x 被移除，AUPRO 计算失败 | 已改 `np.trapezoid`，**已解决** |
| `val_split_mode=none` 时 Lightning 仍请求 val_dataloader | 已加 `limit_val_batches=0`，**已解决** |
| 官方数据集下载过慢 | 人工下载完成，当前**不阻塞**；`tools/parallel_download.py` 留作后续自动化 |
| 指标低于论文水平 | **未完全解决**，已给出 3 条归因与 3 条待验证假设 |

---

## 九、当前问题、限制与下一步建议

### 9.1 必须优先解决的三个问题

1. **可复现性**：固定 `torch.use_deterministic_algorithms`、DataLoader `worker_init_fn`、划分种子落盘；同一配置跑 5 次报均值±标准差。否则后续"光照增强提升了多少"无法判断真伪。
2. **评估分辨率**：把热力图双线性上采样回 840×840，在**原分辨率**重算 Pixel AUROC/AP/PRO，与论文口径对齐；这很可能直接把 Pixel AUROC 拉高到 0.9 以上（细裂纹在 256 下被抹平）。
3. **同源对比基线**：在相同协议（184 拟合 / 46 阈值 / 256 或 840 评估）下，用 `wide_resnet50_2` 跑一次，确认"骨干规模"能解释多少差距。

### 9.2 与课题主线（光照鲁棒性）的衔接建议

今日刻意没碰光照，但链路已经为它准备好了：

- **必须做的对照实验设计**：固定模型与权重，仅对**测试集**施加光照扰动（亮度/色温/局部阴影/低照度噪声），保持训练集不变 → 分离"光照对特征分布"与"光照对成像质量"两种影响。
- **必须补的数据**：tile 现有 840×840 原图可做物理合理的 gamma/色温扰动；建议同时准备一套**真实拍摄的受控光照数据**，否则"光照"只是离线增广，审稿人会质疑外部效度。
- **当前最该警惕的**：PaDiM 的像素级定位本身就很弱（AUPRO 0.68、漏检 38/84），如果直接在此之上做光照改进，改进量可能被基线噪声淹没。建议**先修好基线的分辨率与可复现性问题，再叠加光照变量**。

### 9.3 明确不做（与今日范围一致）

未设计新网络或注意力模块、未改动 PaDiM 算法、未做超参搜索、未下载 MVTec AD 2、未开始论文写作。

---

## 十、关键文件索引

| 用途 | 路径 |
|---|---|
| 项目说明 | `README.md` |
| 实验配置 | `configs/padim_tile.yaml` |
| 环境自检 | `src/env_check.py` |
| 数据集校验 | `src/dataset_check.py` |
| **PaDiM 主实验** | `src/run_padim.py` |
| 样本量消融 | `tools/ablation_full_train.py` |
| 热力图调试 | `tools/debug_padim_map.py` |
| 数据集解压/留痕 | `tools/prepare_dataset.py` |
| 多连接下载 | `tools/parallel_download.py` |
| 主结果 | `results/metrics/padim_tile_day1_summary.json` |
| 逐图得分 | `results/metrics/padim_tile_day1_per_image.csv` |
| 热力图张量 | `results/metrics/padim_tile_day1_test_maps.npz` |
| 数据集校验 | `docs/data/dataset_check_tile.md` |
| 可视化 | `figures/day1_padim_tile/` |
| 依赖快照 | `docs/env/requirements_export.txt` |

---

## 附：参考

- Bergmann et al. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019; IJCV 129(4), 2021.
- Defard et al. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. ICPR 2021. arXiv:2011.08785
- Akcay et al. *Anomalib: A Deep Learning Library for Anomaly Detection*. 2022.
