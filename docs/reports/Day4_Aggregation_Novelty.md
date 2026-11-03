# Day 4 聚合方法查新：直接相关工作对照

- **检索日期**：2026-11-02
- **检索目标**：图像级异常分数聚合（Max / Top-K / 分位数）、极值偏差校正、基于正常样本分布的空间聚合、按图像尺寸/网格数校正、以及在 MVTec AD 2 上的验证
- **证据等级**：`A` = 已核对原文元数据或官方项目页；`B` = 权威索引摘要；`C` = 仅见二手引用/未独立核实

---

## 一、必查的五篇

### 1. Beyond Benchmark Scores（IEEE Access 2026）★★ 最直接

| 项 | 内容 |
|---|---|
| 文献 | Vo V.P., Pham L.P.C., Pham T.D., Nguyen P.N. *Beyond Benchmark Scores: An Analysis of Resolution, Spatial Structure, Illumination, and Seed Sensitivity in Training-Free Industrial Anomaly Localization on MVTec AD 2*. IEEE Access 14: 145972–145991, 2026-09-21 |
| 等级 | A |
| 与聚合的关系 | 🔴 **已把 "image-score aggregation" 列为显式审计维度**，并对 smoothing / pixel threshold / aggregation / conformal operating levels 做单因子审计 |
| 是否做过 Max vs Top-K | 用户在 Day 4 任务说明中明确指出：**已在 PaDiM 风格方法上做过 Max 与 Top-1% 对照** |
| 是否校正极值偏差 | 摘要未提"极值偏差校正"，只提"审计"。**需读全文确认** |
| 是否按网格数校正 | 摘要提到 resolution 与 defect-to-patch resolution 分析，但未明确"按位置数校正聚合统计量" |
| 数据 | MVTec AD 2 全 8 类 |
| 对我们的影响 | **"用 Top-K 替代 Max"不能作为贡献**；我们只能在其未覆盖的机制层面找差异 |

### 2. ReinAD（NeurIPS 2025）

| 项 | 内容 |
|---|---|
| 文献 | Wang X., Zhuo J., You Z., Tan Z., Yu Y., Wang S., Le X. *ReinAD: Towards Real-world Industrial Anomaly Detection with a Comprehensive Contrastive Dataset*. NeurIPS 2025 (Datasets & Benchmarks) |
| 等级 | A（官方项目页 + NeurIPS proceedings） |
| 核心 | 提出 ReinAD 数据集（56 类、279 种异常、87084 样本，强调对比式定义、细粒度、未对齐）+ ReinADNet（金字塔相似度聚合 + 全局-局部特征融合） |
| 与聚合的关系 | 🟠 提出 **pyramidal similarity aggregation**，即"多尺度相似度聚合"，属于**分数图生成阶段**的聚合，不是图像级 Max/Top-K 的替代 |
| 是否解决我们的问题 | ❌ 不涉及"网格位置数导致的极值偏差" |
| 对我们的影响 | 说明"聚合"这个词在文献里主要指多尺度/多层融合；图像级聚合函数的选择仍是相对空白但被审计过的角落 |

### 3. 双层 DINOv3 特征 + 异常分数融合（Electronics 2026）

| 项 | 内容 |
|---|---|
| 状态 | ⚠️ **未能独立核实**。按题面给的标题与期刊检索，未命中该文；命中的最接近工作是 PFAD（见下） |
| 处理方式 | **标记为 C 级，暂不引用**。若需引用，必须先用 DOI 在 MDPI Electronics 站点定位原文 |

### 4. PFAD（Sustainability 2026）—— 替代证据

| 项 | 内容 |
|---|---|
| 文献 | *PFAD: Parameter-Efficient Framework for Cross-Domain Anomaly Detection for Sustainable Manufacturing*. Sustainability 18(13): 6684, 2026 |
| 等级 | A（MDPI 页面，含方法全文片段） |
| 与聚合的关系 | 🔴 **明确使用 Max 做图像级聚合**：`final image-level score = f(classification score, maximum segmentation score)`；多尺度预测用 **平均** 聚合 |
| 意义 | 说明 **2026 年的新方法仍然默认用 Max**，Top-K/均值并非行业共识；这既支持"聚合值得改"，也说明"改聚合"很容易被认为是调参 |

### 5. 其他直接相关

| 文献 | 等级 | 与聚合的关系 |
|---|---|---|
| VisualAD（DINOv3 + 双 token，工程实践文） | C | 图像级分类损失用 **TopK 聚合分数** —— Top-K 已是训练目标里的常规做法 |
| *Calibrated Feature Fusion*（Sensors 26(7): 2164, 2026） | A | 指出多阶段融合"**朴素求和/平均**"是主流，提出跨阶段校准；属**特征/分数图融合**，非图像级聚合 |
| *Multi-view Patch Inference with Frozen DINOv3*（Multimedia Systems, 2026） | B | 多视图重叠区域聚合以缓解 patch 边界信息丢失——与"边界伪影"相关，但处理的是 **patch 边界**而非图像边缘 |

---

## 二、逐条回答任务 1 的五个问题

### Q1 已有哪些图像级异常分数聚合方法？

已确认在职的至少四类：

| 类别 | 代表 | 备注 |
|---|---|---|
| Max | PaDiM、PatchCore、PFAD（2026） | 事实上最常用 |
| Top-K / Top-1% 均值 | VisualAD 训练目标、Beyond Benchmark Scores 审计 | 已有对照实验 |
| 分位数 | Beyond Benchmark Scores 单因子审计包含阈值/聚合 | 常规统计量 |
| 均值 / 多尺度平均 | PFAD（多尺度层间）、Calibrated Feature Fusion（指出朴素平均是主流） | 常规 |
| 金字塔/多层融合 | ReinAD、Calibrated Feature Fusion、Multi-view Patch Inference | 属分数图生成，不是图像级规则 |

**结论：任务 2 里要求对比的 5 种规则，全部是已有或常规方法。这一点无争议。**

### Q2 是否已有聚合分数的极值偏差校正？

- **未检索到**明确以"Extreme Value Theory 校正聚合分数"为标题的 IAD 论文。
- 但存在**同构思想**：Keeping the Balance（ICASSP 2025，语音域）用"到参考样本的距离 / 该样本到其近邻的距离"之比做分数重标定，本质是**消除参考集规模带来的尺度偏差**。
- Beyond Benchmark Scores 的 "seed sensitivity" 与 "score aggregation" 审计很可能已经触及"分数随采样变化"这一现象（未读全文，**不能断言它没做**）。

**判定：🟠 中等风险。极值偏差这件事"可能没人明确命名"，但"分数随规模漂移"这个现象已被多人观察。**

### Q3 是否已有基于正常样本分布的空间聚合方法？

- **是，但方向不同**：Robust Distribution Alignment（2025）用 Sinkhorn 对齐正常分布；PFAD 用正常参考特征做通道加权；LL-IAD 用光照图调特征。
- 这些都是"**改分数图**"，不是"改图像级聚合规则"。
- **"用正常样本在每个位置上的分布来加权聚合"（位置可靠度加权）——未检索到明确实现。**

**判定：🟡 这是目前看来最可能的空位，但需要读 Beyond Benchmark Scores 全文确认。**

### Q4 是否有对图像大小/特征网格数量进行校正的方法？

- **未检索到**。文献普遍直接把 anomaly map resize 到原图后取 max，**不校正网格规模**。
- Beyond Benchmark Scores 分析 resolution，但据摘要其结论是"应分离分辨率与表征质量"，未提出校正公式。

**判定：🟢 这是本次查新中最像空位的一条。但注意：它可能只是"没人觉得需要"，而不是"有价值"。**
我们的 Day 4 实验恰好给出了"需要"的证据（见 Day4_Report §3）。

### Q5 哪些方案已用 MVTec AD 2 验证？

| 方案 | 是否用 AD2 |
|---|---|
| Beyond Benchmark Scores（DINO / PatchCore-style / PaDiM-style 五配置） | ✅ 全 8 类 |
| ReinAD / ReinADNet | ❌（用 MVTec AD、VisA、BTAD、MPDD） |
| PFAD | ❌（跨域设定，非 AD2） |
| Calibrated Feature Fusion | ❌（few-shot，MVTec/VisA 类） |
| 我们的 Day 4 实验 | ✅ sheet_metal（探索性） |

**结论：AD2 上的聚合研究，目前公开可见的只有 Beyond Benchmark Scores 一家。**

---

## 三、撞车风险总表

| 我们可能主张的点 | 最近的已有工作 | 风险 | 说明 |
|---|---|---|---|
| "Max 不如 Top-K" | Beyond Benchmark Scores（已有 Max vs Top-1% 对照） | 🔴 **已死** | 不能作为贡献 |
| "聚合方式显著影响 Image AUROC" | 同上 | 🔴 **已死** | 同上 |
| "Top-K 能提升 AUROC" | 同上 + VisualAD | 🔴 **已死** | 且我们实测在 sheet_metal/squash 上 Top-K **更差** |
| "极值随位置数膨胀导致排序退化" | 未见明确论文；Keeping the Balance 有同构思想 | 🟠 **中** | 我们的**受控抽样因果证据**是新的（同图、同权重、只改采样位置数） |
| "正常图 max 落在图像边界（结构性伪影）" | Multi-view Patch Inference 讨论 patch 边界丢失，但非图像边界 | 🟡 **中低** | 我们的证据是定量的（79% 落在 10% 边界带、中位距离 0.0） |
| "按位置可靠度加权的聚合" | 未检索到 | 🟡 **中** | 最可能的空位，但价值未证明 |
| "按网格数校正聚合统计量" | 未检索到 | 🟢 **低撞车** | 但需证明它不是"显然该做"的工程细节 |

---

## 四、对 Day 3 结论的影响（必须先接受）

Day 3 我们写下："失效主要不是检测不出来，而是**分数标定失效**"。
Day 4 的受控实验（同一批热力图，只换聚合规则）表明这个诊断**不完整**：

- aspect 配置下，**组内**（单一光照条件内）gmean 的 Image AUROC 达 **0.95–1.00**，而 max 只有 0.40–0.73。
- 也就是说：**热力图本身几乎完美可分，是 Max 这个规则把它毁掉了**，而不是"光照把分数整体平移"。
- 光照漂移仍然存在（gmean 跨条件池化 0.840，组内约 0.98，差 ≈0.14），但它是**次要因素**。

**这条必须写进 Day 4 报告并回改 Day 3 的措辞**——否则后续所有论证都建立在错误归因上。

---

## 五、引用前必须核实

| 文献 | 待核实 |
|---|---|
| Beyond Benchmark Scores 全文 | ①是否已提出聚合校正方法；②是否做过"位置数/网格规模"的受控实验；③Max vs Top-1% 的具体数值 |
| Electronics 2026 双层 DINOv3 分数融合 | 是否存在（本次未命中），需 DOI 定位 |
| Filter or Compensate（AAAI 2025） | 作者/页码/机制（Day 3 已列为 C 级） |
| ReinAD 的 "pyramidal similarity aggregation" | 具体是层间还是空间聚合，需读方法节 |
