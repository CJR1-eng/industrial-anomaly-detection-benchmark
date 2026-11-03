# MVTec AD 2 数据核查：sheet_metal

- 目录: `datasets\raw\ad2\sheet_metal`
- 图像总数: 644
- 顶层目录: ['test_private', 'test_private_mixed', 'test_public', 'train', 'validation']

## 各划分样本数

| 划分 | 子集 | 数量 |
| --- | --- | --- |
| test_private | <direct> | 142 |
| test_private_mixed | <direct> | 142 |
| test_public | bad | 90 |
| test_public | good | 24 |
| test_public | ground_truth | 90 |
| train | good | 137 |
| validation | good | 19 |

## 分辨率与通道（抽样 ≤120 张）

- 尺寸分布: {'4224x1056': 120}
- 通道模式: {'L': 120}

## test_public（唯一可本地评估的测试集）

- 子集: ['bad', 'good', 'ground_truth']
- 含 ground_truth 目录: True
- GT 计数: {'bad': 90}

## 光照条件线索探查

- 文件名关键词命中: {'mixed': 142, 'expos': 68}
- 文件名示例: {"bad": ["000_overexposed.png", "000_regular.png", "000_shift_1.png", "000_shift_2.png", "000_shift_3.png", "000_underexposed.png"], "good": ["000_overexposed.png", "000_regular.png", "000_shift_1.png", "000_shift_2.png", "000_shift_3.png", "000_underexposed.png"], "ground_truth": ["000_overexposed_mask.png", "000_regular_mask.png", "000_shift_1_mask.png", "000_shift_2_mask.png", "000_shift_3_mask.png", "000_underexposed_mask.png"]}

