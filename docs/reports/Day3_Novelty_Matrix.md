# Day 3 论文查新矩阵

- **检索日期**：2026-11-02
- **检索范围**：PaDiM / PatchCore / MVTec AD 2 / WEDGE-Net / Beyond Benchmark Scores + 2023–2026 年
  illumination-invariant IAD、score calibration under shift、illumination-robust localization、tiny defect preservation
- **证据等级**：`A` = 已核对原文元数据（DOI/卷期页/摘要）；`B` = 仅据检索摘要或权威索引页；`C` = 仅见于他人参考文献，未能独立核实
- **原则**：已经被研究过的现象不再当作创新点。发现高撞车风险时直接标注，不强行制造差异。

---

## 一、必读的五篇（题面指定）

### 1. PaDiM
| 项 | 内容 |
|---|---|
| 文献 | Defard T., Setkov A., Loesch A., Audigier R. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. ICPR 2021, pp. 475–489. arXiv:2011.08785 |
| 等级 | A（本项目已复现，代码级核对） |
| 研究问题 | 无监督异常检测 + 定位，只用正常样本建模 |
| 输入/监督 | 仅正常图像；ImageNet 预训练 CNN 特征 |
| 核心机制 | 多层特征拼接 → 逐空间位置拟合多元高斯 → 马氏距离作异常分数 |
| 数据/协议 | MVTec AD 15 类；Image AUROC / Pixel AUROC / PRO |
| 真实光照 | ❌ 完全不涉及 |
| 正常样本校准 | ❌ 有高斯建模，但无"分布偏移后校准" |
| 微小缺陷保护 | ❌ 无专门机制（σ=4 模糊本身不利于微小缺陷） |
| 与我们重合 | 我们的基线就是它 |
| 未解决 | 对光照/域偏移无任何显式处理；分数尺度随域漂移 |

### 2. PatchCore
| 项 | 内容 |
|---|---|
| 文献 | Roth K., Pemula L., Zepeda J., Schölkopf B., Brox T., Gehler P. *Towards Total Recall in Industrial Anomaly Detection*. CVPR 2022, pp. 14318–14328 |
| 等级 | A（领域标准方法，元数据可靠） |
| 研究问题 | 用尽量少的正常样本获得高召回 |
| 输入/监督 | 仅正常图像；WideResNet50 中层特征 |
| 核心机制 | 贪婪 coreset 子采样 + 最近邻记忆库；局部邻域聚合 |
| 数据/协议 | MVTec AD；Image/Pixel AUROC、PRO |
| 真实光照 | ❌ 不涉及 |
| 正常样本校准 | ❌ 无 |
| 微小缺陷保护 | ⚠️ 间接（高分辨率特征图）；无显式机制 |
| 与我们重合 | 潜在的第二基线（题面要求若失效明显再加入） |
| 未解决 | 记忆库对光照引起的特征漂移同样敏感，且更大 |

### 3. MVTec AD 2 ★关键
| 项 | 内容 |
|---|---|
| 文献 | Heckler-Kram L., Neudeck J.-H., Scheler U., König R., Steger C. *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*. IJCV 134(4), 2026. DOI 10.1007/s11263-026-02743-0（2026-03-09 在线） |
| 等级 | A（官方页面 + 期刊页双源核对） |
| 研究问题 | 现有基准（MVTec AD / VisA）在 AU-PRO 上已饱和，SOTA 差距 <1 个百分点，缺乏区分力 |
| 数据规模 | 8 个类别、>8000 张高分辨率图像；含透明/重叠物体、暗场与背光照明、正常数据高方差、极微小缺陷 |
| 划分 | `train`（无缺陷）/ `validation`（无缺陷）/ `test_public`（各光照条件 + 像素级 GT）/ `test_private`（常规光照，GT 不公开）/ `test_private_mixed`（同场景、不同光照，GT 不公开） |
| 评价方式 | 私有部分只能通过 **benchmark.mvtec.com** 评测服务器；本地只能用 test_public |
| 关键结论 | SOTA 方法平均 AU-PRO **仍低于 60%**；数据集**专门提供光照变化测试场景**以考察真实分布偏移下的鲁棒性 |
| 真实光照 | ✅ **是**，这是该数据集的核心设计之一 |
| 正常样本校准 | ⚠️ 数据集不提供方法，但提供了可做校准研究的协议 |
| 微小缺陷保护 | ✅ 明确包含"extremely small defects"场景 |
| 与我们重合 | **数据集层面完全重合**：它就是为了"光照变化 + 微小缺陷"造的 |
| 未解决 | 论文本身是基准，不提出方法；光照鲁棒方法仍空缺 |

> **判断**：我们想研究的现象，**基准作者已经明确指出并量化了**（光照场景 + SOTA <60%）。
> 这意味着"发现光照会降低性能"**不构成创新**，只能作为问题动机。

### 4. WEDGE-Net ★高撞车
| 项 | 内容 |
|---|---|
| 文献 | Park J.-M., Kim G.-Y. *WEDGE-Net: Wavelet-Driven Memory-Efficient Anomaly Detection for Industrial Edge Computing*. Sensors 2026, 26(7): 2154. DOI 10.3390/s26072154（2026-03-31） |
| 等级 | A（DOAJ + MDPI 双源核对） |
| 研究问题 | 边缘设备上的内存效率 + 环境噪声鲁棒性 |
| 输入/监督 | 仅正常图像；双流结构 |
| 核心机制 | ①频率流：**DWT 小波变换**物理滤除环境噪声，隔离结构缺陷；②上下文流：语义模块引导特征提取保证物体一致性；两流融合 |
| 数据/协议 | MVTec AD；平均 Image AUROC 97.82%；1% 记忆压缩；686.5 FPS（RTX 4090）；对 **Tile** 类做噪声鲁棒性压力测试 |
| 真实光照 | ⚠️ 部分——"environmental noise"与高频纹理鲁棒性，不等于光照 |
| 正常样本校准 | ❌ 无 |
| 微小缺陷保护 | ⚠️ 通过频率分离间接实现 |
| 与我们重合 | **"频域滤波 + 噪声/光照鲁棒 + 轻量边缘部署"与我们设想的切入点高度重合** |
| 未解决 | 未用真实光照条件评估；未处理曝光/局部光照梯度；未做分数校准 |

> **判断**：任何"频域滤波提升光照鲁棒"的想法**基本被它占了**。
> 且它已经在 **Tile（我们的类别）** 上做过鲁棒性论证——撞车面很具体。

### 5. Beyond Benchmark Scores ★★最危险
| 项 | 内容 |
|---|---|
| 文献 | Vo V.P., Pham L.P.C., Pham T.D., Nguyen P.N. *Beyond Benchmark Scores: An Analysis of Resolution, Spatial Structure, Illumination, and Seed Sensitivity in Training-Free Industrial Anomaly Localization on MVTec AD 2*. IEEE Access, vol. 14, pp. 145972–145991, 2026-09-21（Xplore 11701006） |
| 等级 | A（IEEE Xplore 摘要页 + 作者页双源） |
| 研究问题 | 基准分数不足以说明运行可靠性；需分离表征质量、聚合、校准、区域重叠、光照鲁棒性、实现保真度、可复现性 |
| 输入/监督 | Training-free（DINO 系列 3 配置 + PatchCore-style WideResNet + **PaDiM-inspired ResNet18 positional-Gaussian**） |
| 核心机制 | 固定协议可靠性分析管线；对 smoothing / pixel threshold / **image-score aggregation** / conformal operating levels 做单因子审计 |
| 数据/协议 | **MVTec AD 2 全部 8 类**；image ranking、**normal-only calibration**、pixel localization、AU-PRO、false-positive area、**illumination consistency**、defect-to-patch resolution、seed sensitivity |
| 关键数字 | ①Can 类微小缺陷定位全部失效（最佳 Pixel AP 1.50×10⁻⁴，mIoU 1.00×10⁻³）；②**私有光照偏移改变 30.9%–46.1% 的已校准判定**；③自报置信区间常重叠 |
| 真实光照 | ✅ **是**，用的正是 MVTec AD 2 的私有光照场景 |
| 正常样本校准 | ✅ **是**，"normal-only calibration" 是它的显式评测维度之一 |
| 微小缺陷保护 | ✅ 有（defect-to-patch resolution 分析） |
| 与我们重合 | **几乎全部重合**：它是 PaDiM 风格基线 + 真实光照 + normal-only 校准 + 分辨率/微小缺陷 + 种子敏感性 |
| 未解决 | 它是**分析型**论文，不提出新方法；且只做 training-free 配置，未提出校准算法 |

> **判断（最重要的一条）**：
> 我们要问的"光照是否破坏分数校准"——它已经用 MVTec AD 2 私有光照集量化了
> （**30.9%–46.1% 的判定被改变**），并且把 "normal-only calibration" 单独列为评测维度。
> **我们再"发现"这个现象，是重复劳动。**
> 唯一还开着口的是：它**只做诊断，不做方法**。

---

## 二、2023–2026 直接相关工作（补充检索）

### 6. 低光照 IAD + 配对数据（LL-IAD）
| 项 | 内容 |
|---|---|
| 文献 | Cuong H.D., Tan P.X. 等. *An efficient and memory-friendly unsupervised industrial anomaly detection model*. Journal of Computational Design and Engineering, 12(5), 2025. DOI 10.1093/jcde/qwaf043（2025-05-01） |
| 等级 | A（期刊 DOI + 机构新闻稿 + 两处独立报道） |
| 问题 | 低光照/光照变化下 embedding 类方法失效 |
| 输入 | **成对**的"良好光照 + 低光照"图像（关键：需要配对监督） |
| 机制 | Low-pass Feature Enhancement（低频增强抑噪）+ **Illumination-aware Feature Enhancement（估计光照图动态调特征）** + 自适应融合（通道+空间注意力）；用良好光照特征重建低光照特征做异常比对 |
| 数据 | 自建 **LL-IAD**：6600 对、10 类；另测 Insulator / Clutch |
| 真实光照 | ✅ 真实低光照（非合成） |
| 正常校准 | ⚠️ 特征层光照对齐 ≈ 校准的前身 |
| 微小缺陷 | ❌ 无专门机制 |
| 撞车 | **"光照感知特征增强 + 光照图估计"已被实现**，且带数据集 |
| 未解决 | **强依赖配对数据**——真实产线往往拿不到"同一场景另一光照"的配对。这是可攻击点 |

### 7. DiRNet（域不变重建）
| 项 | 内容 |
|---|---|
| 文献 | Xu H. 等. *DiRNet: A Domain-Invariant Reconstruction Framework for Unsupervised Anomaly Detection Under Distribution Shift*. PRCV 2025, pp. 426–439. DOI 10.1007/978-981-95-4987-0_30 |
| 等级 | A（ACM DL 页面） |
| 机制 | 结构级特征校准（通道响应调制以滤除**风格敏感信息**）+ 自监督结构一致性正则 + 域感知解码器（全局结构先验） |
| 明确动机 | 训练/测试分布因**光照、成像设备、环境**改变而不一致 |
| 真实光照 | ⚠️ 用带分布偏移的基准（含光照），非 MVTec AD 2 |
| 撞车 | "抑制风格/光照相关通道响应"= 我们设想的"分数校准"的上游版本 |
| 未解决 | 需训练（重建框架），非 training-free；4GB 卡成本高 |

### 8. Filter or Compensate（AAAI 2025）
| 项 | 内容 |
|---|---|
| 文献 | Chen Z., Luo X., Wang W., Zhao Z., Su F., Men A. *Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection*. AAAI 2025, vol. 39, pp. 2420–2428 |
| 等级 | C（见于 DiRNet 参考文献，未独立核对原文，**引用前必须核实**） |
| 机制（据题名+引用语境） | 显式对比"过滤（filter）"与"补偿（compensate）"两条实现不变表示的路线 |
| 撞车 | **它把"过滤 vs 补偿"这个二分法已经做成了论文的核心论点** |
| 影响 | 我们若提"用滤波还是用校准"，会直接落进它的框架 |

### 9. 分布偏移下的异常检测（ICCV 2023 起的一条线）
| 项 | 内容 |
|---|---|
| 文献 | Cao T., Zhu J., Pang G. *Anomaly Detection under Distribution Shift*. ICCV 2023, pp. 6511–6523 |
| 等级 | C（见于参考文献列表；该文为领域内被广泛引用的工作） |
| 机制 | 广义无监督异常检测 + 偏移下的泛化 |
| 撞车 | 问题设定层面已覆盖"分布偏移下的 AD" |

### 10. Robust Distribution Alignment（记忆库 + Sinkhorn）
| 项 | 内容 |
|---|---|
| 文献 | Liao J., Xu X., Su Y., Tu R.-C., Liu Y., Tao D., Yang X. *Robust distribution alignment for industrial anomaly detection under distribution shift*. arXiv:2503.14910（2025-03-19） |
| 等级 | A（arXiv + Google Scholar 条目） |
| 动机 | **光照变化 / 传感器漂移**导致的未见域偏移 |
| 机制 | 在记忆库类方法上，用有限目标域训练数据优化 **robust Sinkhorn distance**，提升对未见目标域的泛化 |
| 数据 | 2D + 3D AD 基准 + **模拟**分布偏移 |
| 撞车 | "用少量目标域正常样本对齐分数分布"——正是"正常样本校准"的标准做法 |
| 未解决 | 需目标域样本；模拟偏移而非真实光照 |

### 11. 异常分数重标定用于域泛化（语音域，方法可迁移）
| 项 | 内容 |
|---|---|
| 文献 | Wilkinghoff K., Yang H., Ebbers J., Germain F.G., Wichern G., Le Roux J. *Keeping the Balance: Anomaly Score Calculation for Domain Generalization*. ICASSP 2025（MERL TR2025-030） |
| 等级 | A（MERL 技术报告全文） |
| 机制 | 用"测试样本到正常参考样本的余弦距离 / 该参考样本到其近邻的距离"之比来**重标定分数**，缓解源域与目标域的分数尺度错配 |
| 领域 | 异常**声音**检测（DCASE2020/2023），非图像 |
| 撞车 | **"用近邻距离比做无监督分数重标定"这个机制本身已被提出** |
| 机会 | 尚未在**图像 IAD + 真实光照**上验证——这是少数还开着的口子 |

### 12. 加权保形异常检测（低数据 + 分辨率崩塌）
| 项 | 内容 |
|---|---|
| 文献 | *Between Resolution Collapse and Variance Inflation: Weighted Conformal Anomaly Detection in Low-Data Regimes*. arXiv:2603.23205（2026-03） |
| 等级 | B（仅见 arXiv 摘要片段，作者信息未获取，**引用前需核实**） |
| 机制 | 加权保形（weighted conformal）框架，用似然比 dQ/dP 重加权校准集，处理协变量偏移；讨论分辨率崩塌与方差膨胀的权衡 |
| 撞车 | "用密度比给校准集重加权以应对偏移" = 校准类方法的上界形态 |
| 未解决 | 面向低数据表格/通用设定，未做工业图像光照 |

---

## 三、撞车风险总表（按威胁程度排序）

| 我们可能的切入点 | 最近的已有工作 | 撞车风险 | 说明 |
|---|---|---|---|
| "发现光照会降低 IAD 性能" | MVTec AD 2（IJCV 2026）；Beyond Benchmark Scores（IEEE Access 2026） | 🔴 **已死** | 前者把光照场景做进基准，后者量化了 30.9%–46.1% 判定改变 |
| "真实光照下的分数校准" | Beyond Benchmark Scores（normal-only calibration 评测维度） | 🔴 **已死（作为发现）** | 但"提出校准算法"仍空 |
| "光照感知特征增强 / 光照图估计" | LL-IAD / JCDE 2025 | 🔴 **已死** | 已实现且有配套数据集 |
| "频域/小波滤波提升光照鲁棒" | WEDGE-Net（Sensors 2026） | 🔴 **已死** | 且已在 Tile 类验证 |
| "过滤 vs 补偿"的路线论证 | Filter or Compensate（AAAI 2025） | 🟠 **高** | 二分法被占（引用前需核实原文） |
| "用少量目标域样本对齐分数分布" | Robust Distribution Alignment（2025） | 🟠 **高** | Sinkhorn 对齐已是标准解 |
| "无监督分数重标定（近邻距离比）" | Keeping the Balance（ICASSP 2025，语音域） | 🟡 **中** | 机制已提出，**未在图像 IAD + 真实光照验证** |
| "微小缺陷在高分辨率下的定位失败" | Beyond Benchmark Scores（defect-to-patch resolution） | 🟠 **高** | 它已量化 Can 类 Pixel AP 1.5×10⁻⁴ |
| "分辨率 vs 光照 哪个更致命" | Beyond Benchmark Scores（resolution 是显式维度） | 🟠 **高** | 已做单因子审计 |

---

## 四、仍然开着的口子（诚实版）

1. **校准"算法"本身**。Beyond Benchmark Scores 只诊断不提出方法；Robust Distribution Alignment 需要目标域训练样本；Keeping the Balance 是语音域且需近邻比。**"无需目标域样本、无需重训、只用源域正常统计量"的图像域分数校准**仍是空位——但空位很窄。
2. **配对假设的破除**。LL-IAD 依赖"良好光照/低光照"配对；真实产线拿不到配对。**"无配对条件下的光照鲁棒"**是它的明确弱点。
3. **微小缺陷与光照的交互**。现有工作分别研究"微小缺陷"和"光照"，**二者交叉（光照是否专门吞掉微小缺陷）**的定量证据仍少。
4. **在 4GB 显存约束下的可行方案**。现有方法（DiRNet、LL-IAD）都要训练；轻量 training-free 校准在消费级卡上的可行性有空间。

---

## 五、对我们原设想的三条否定性结论（必须先接受）

1. **"光照是未被充分研究的脏数据问题"——错误。** MVTec AD 2 就是为此造的基准。
2. **"频域滤波是没人做过的光照鲁棒手段"——错误。** WEDGE-Net 已做，且在我们的类别上验证过。
3. **"分数校准是新想法"——错误。** 已被列为评测维度（Beyond Benchmark Scores）与机制（Keeping the Balance / Sinkhorn 对齐）。

**因此：Day 2 观察到的"合成光照影响很小"不是坏消息，而是与文献一致**——
在温和光度扰动下，现代特征提取器本来就稳；真正的失效来自真实光照的
**空间非平稳性 + 曝光饱和 + 反射变化**，而这些只有在真实数据上才会显现。

---

## 六、引用前必须核实的三条（等级 C/B）

| 文献 | 待核实项 | 核实方式 |
|---|---|---|
| Filter or Compensate（AAAI 2025） | 作者、页码、机制细节 | 查 AAAI 2025 proceedings 原文 |
| Anomaly Detection under Distribution Shift（ICCV 2023） | 页码、作者顺序 | 查 CVF Open Access |
| Weighted Conformal AD（arXiv 2603.23205） | 作者、机构、是否正式发表 | 查 arXiv 列表页 |

> 本矩阵中所有标 A 的条目均已核对到 DOI 或卷期页；标 C 的仅见于他人参考文献，
> **写入论文前必须回溯原文**，否则有引用错误风险。
