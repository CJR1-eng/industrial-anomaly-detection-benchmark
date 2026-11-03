# 数据集校验报告：tile

- 生成时间: 2026-11-02T18:03:51
- 目录: `datasets\raw\mvtec_ad\tile`
- 结论: **通过**

## 样本数量

| 划分 | 子类 | 数量 |
| --- | --- | --- |
| train | good | 230 |
| test | crack | 17 |
| test | glue_strip | 18 |
| test | good | 33 |
| test | gray_stroke | 16 |
| test | oil | 18 |
| test | rough | 15 |
| ground_truth | crack | 17 |
| ground_truth | glue_strip | 18 |
| ground_truth | gray_stroke | 16 |
| ground_truth | oil | 18 |
| ground_truth | rough | 15 |
| **合计** | train=230 / test=117 / gt=84 | |

## 分辨率与通道

- 图像尺寸分布: {'840x840': 84}
- 掩码尺寸分布: {'840x840': 84}
- 通道模式: {'RGB': 84}

## 缺陷面积占比（像素级）

- mean=0.0980 / median=0.0793 / min=0.0084 / max=0.3549

## 掩码对应性

- 缺失掩码: 0
- 尺寸不一致: 0
- 全黑掩码: 0
- train/test 同名文件（正常编号现象）: 33
- train/test 内容重复（真实泄漏）: 0

## 与官方计数对照

- train_good: 一致 (期望 230)
- test_good: 一致 (期望 33)
- test_defect: 一致 (期望 84)
