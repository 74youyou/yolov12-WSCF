# YOLOv12-WSCF 两阶段训练计划书

> 文档编号：YOLOv12-WSCF-TRAIN-002
> 编制日期：2026-09-17
> 训练方式：**两阶段训练（阶段 A 冻结主干初始化 → 阶段 B 全量微调）**
> 适用模型：YOLOv12-WSCF（Wavelet Skip Cross-scale Fusion）
> 配套脚本：`train/WSCF/train_stageA.py`、`train/WSCF/train_stageB.py`、`val_yolov12n.py`

---

## 0. 基线指标澄清（重要）

本仓库 `README.md` 的 Main Results 有两套 YOLO12n 数据：

| 版本 | 权重 | mAP50-95 | params | GFLOPs |
|------|------|----------|--------|--------|
| **Turbo（默认）** | `turbo/yolov12n.pt` | **40.4** | 2.5M | 6.0 |
| v1.0 | `v1.0/yolov12n.pt` | **40.6** | 2.6M | 6.5 |

本仓库的 `yolov12n.pt` 参数量为 **2,553,904**，架构对应 **Turbo**，官方参考值应为 **40.4**，**不是 v1.0 的 40.6**。

用 `val_yolov12n.py` 在本机 COCO val2017（5000 张，imgsz=640）实测：

```
mAP50-95 = 0.4021   （官方 Turbo=0.4040 / v1.0=0.4060）
mAP50    = 0.5549
```

> **结论：拿"改后模型 40.1"去比"官网 40.6"，会凭空多出约 0.4 个点的心理落差。**
> 真实基线是本机实测的 **≈40.2**（Turbo 官方 40.4），差距只有 **~0.1**，属于训练方差范围。

---

## 1. 方案概述

WSCF 在 YOLOv12 主干中插入了小波支线与三条融合路径，所有融合模块初始为**恒等/零初始化**（训练起点等价于原始 YOLOv12）。为了让这些新增模块与主干相互适配，采用两阶段训练：

| 阶段 | 目的 | 冻结策略 | 训练对象 |
|------|------|---------|---------|
| **阶段 A** | **初始化**小波/融合模块，使其脱离恒等起点 | 冻结原主干 **+ Neck/Head**（默认） | 仅新增模块（约 11.8 万参数） |
| **阶段 B** | **整体适配**，主干与新增模块联合微调到最优 | 解冻全部 | 全网络 |

- 阶段 A 的作用是"**把新增模块从 0/恒等推到一个有意义的工作点**"，因此**可以短**；
- 阶段 B 的作用是"**让主干和新增模块互相适配**"，需要**较长**的训练。

---

## 2. 冻结策略（阶段 A）

WSCF YAML 层号如下：

```
0  Identity         | 1  WaveletBranch | 2-5 Index(W1/W2/W3/W4)
6  Conv(P1)         | 7  P1WaveletFusion
8  Conv(P2)         | 9  C3k2           | 10 Conv(P3)
11 CrossScaleFusion(P3<-W2)
12 C3k2             | 13 Conv(P4)
14 CrossScaleFusion(P4<-W3)
15 A2C2f            | 16 Conv(P5)
17 CrossScaleFusion(P5<-W4)
18 A2C2f            | 19~30 Neck(PAN)   | 31 Detect
```

阶段 A 由脚本顶部的 `FREEZE_HEAD` 开关控制：

| `FREEZE_HEAD` | 冻结范围 | 可训练 | 可训练参数量 |
|---------------|---------|--------|-------------|
| **True（默认，推荐）** | 除新增模块外的全部层（原主干 + Neck/Head） | 仅 1/7/11/14/17 | **117,768（4.41%）** |
| False | 仅原主干 `[6,8,9,10,12,13,15,16,18]` | 新增模块 + Neck/Head | 1,331,800（49.8%） |

**为什么默认 `FREEZE_HEAD=True`？**

1. 阶段 A 的定位就是"**只初始化新增模块**"。冻结 Head 后，**唯一能降低 loss 的途径**就是让小波/融合模块学到有用的残差，梯度信号最聚焦；
2. 若 Head 也可训练，Head 可以"自己补偿"，新增模块反而可能学得少；而且 Head 会在"冻结主干特征"上过拟合，阶段 B 解冻主干后还得重新适配，等于白练；
3. 参照旧 CSF 项目：阶段 A 就是冻结除 CSF 外的全部参数（含 Head），仅训练新增模块。

> 若你希望 Head 提前适配新特征，把 `FREEZE_HEAD` 改为 `False` 即可。
> 注意：无论哪种，都不要用 `freeze=18`（按前 N 层冻结）——那会把第 1/7/11/14/17 层的**新增模块也冻结**。

---

## 3. 超参数配置

| 参数 | 阶段 A | 阶段 B | 说明 |
|------|--------|--------|------|
| `model` | `yolov12-wscf.yaml` | `train/WSCF/A_stage_result/stageA_wavelet_init/weights/best.pt` | |
| `data` | `coco-csf.yaml` | `coco-csf.yaml` | 勿用 `coco.yaml`（会触发联网下载） |
| `epochs` | **40**（20~40） | **400**（300~400） | 见第 4 节 |
| `imgsz` | 640 | 640 | |
| `batch` | 16 | 16 | 显存不足降至 8/4 |
| `workers` | 8 | 4 | Windows 下 0~8 |
| `device` | 0 | 0 | 多卡 `0,1` |
| `lr0` | **0.009** | **0.009** | 与官方默认 0.01 同量级 |
| `lrf` | 0.01 | 0.01 | 最终 lr = 9e-5 |
| `freeze` | `FREEZE_HEAD` 计算得到 | `None` | |
| `optimizer` | auto | SGD | 阶段 B 与论文一致 |
| `momentum` | - | 0.937 | |
| `weight_decay` | - | 0.0005 | |
| `warmup_epochs` | 3.0 | 3.0 | |
| `cos_lr` | - | False | 论文用线性衰减 |
| `close_mosaic` | 10 | 10 | 最后 10 epoch 关 mosaic |
| `patience` | 100 | 200 | 早停 |
| `amp` | True | True | 混合精度 |
| `seed` | 0 | 0 | 可复现 |
| `project` | `train/WSCF/A_stage_result` | `train/WSCF/B_stage_result` | |
| `name` | `stageA_wavelet_init` | `stageB_full_finetuning` | |

---

## 4. 训练轮数（epochs）分析

### 4.1 结论

| 阶段 | 设定 | 判断 |
|------|------|------|
| 阶段 A | **40**（20~40） | ✅ 初始化足够，不宜过长 |
| 阶段 B | **400**（300~400） | ✅ 可接受，略偏长 |

**总轮数：40 + 400 = 440。**

### 4.2 阶段 A 为什么不宜过长

1. 阶段 A 只训练约 **11.8 万参数**（新增模块），且主干/Head 冻结，优化目标非常简单；
2. 它的定位是"初始化"而非"收敛"，融合模块从恒等/零出发，只需要学到"有用的增量方向"；
3. 冻结/解冻的特征分布不一致：阶段 A 里学到的残差是针对"冻结主干"的，阶段 B 解冻后主干变化，必须重新适配。**阶段 A 越长，这部分"白练"越多**；
4. 旧 CSF 项目阶段 A 仅 40 epoch（且只训 CSF）即完成初始化，WSCF 阶段 A 同为 40 合理。

### 4.3 阶段 B 为什么 400 可以接受

1. 阶段 B 是"主干 + 新增模块 + Neck/Head"的**整体适配**，需要足够长时间协同收敛；
2. 旧 CSF 项目阶段 B 用了 **600 epoch**，400 相对保守；
3. 注意收益递减：全量微调通常 **250~350 epoch** 进入平台期。建议保留 `patience=200` 自动早停，或先跑 300 观察 `results.png` 再决定。

### 4.4 关于 lr0=0.009

- 官方从头训练默认 `lr0=0.01`，`0.009` 属同一量级，作为全量微调（阶段 B）合适；
- 阶段 A 用 `0.009` 的高学习率，主要作用于新增模块（快速脱离恒等）；若阶段 A 出现 loss 抖动/发散，可降到 `0.005`。

---

## 5. 执行步骤

### 5.1 阶段 A

```powershell
conda activate yolov12
python train/WSCF/train_stageA.py
```

脚本行为：
1. 构建 `yolov12-wscf.yaml`，加载 `yolov12n.pt`（预期 `Transferred 739/845`）；
2. 打印可训练参数占比（`FREEZE_HEAD=True` 时约 4.41%）；
3. 冻结主干（+Head），训练 40 epoch；
4. 输出到 `train/WSCF/A_stage_result/stageA_wavelet_init/`。

### 5.2 阶段 B

```powershell
python train/WSCF/train_stageB.py
```

脚本行为：
1. 从阶段 A 的 `best.pt` 继续；
2. `freeze=None` 解冻全部，`lr0=0.009`、`lrf=0.01`、SGD、线性衰减、warmup 3，训练 400 epoch；
3. 输出到 `train/WSCF/B_stage_result/stageB_full_finetuning/`。

### 5.3 断点续训

两个脚本都内置断点检测：若输出目录存在 `weights/last.pt`，会自动 `resume=True` 恢复。

---

## 6. 输出目录与产物

```
train/WSCF/
├── train_stageA.py
├── train_stageB.py
├── A_stage_result/stageA_wavelet_init/
│   ├── weights/best.pt, last.pt
│   ├── results.csv, results.png
│   └── args.yaml
└── B_stage_result/stageB_full_finetuning/
    ├── weights/best.pt, last.pt
    ├── results.csv, results.png
    ├── confusion_matrix.png
    └── val_batch*_pred.jpg
```

---

## 7. 监控与评估

### 7.1 阶段 A 观察重点

- 三项目 loss 是否稳定下降（新增模块在学增量）；
- 因为 Head/主干被冻结，阶段 A 的 mAP 提升会**很小**，这是正常的（它只是初始化）。

### 7.2 阶段 B 观察重点

- `mAP50-95` 是否持续上升并趋平；
- 验证 loss 是否在后期回升（过拟合信号 → 依赖 `patience` 早停或取 best.pt）。

### 7.3 最终评估

```python
from ultralytics import YOLO

model = YOLO("train/WSCF/B_stage_result/stageB_full_finetuning/weights/best.pt")
metrics = model.val(data="coco-csf.yaml", imgsz=640, batch=16, device=0)
print("mAP50-95:", metrics.box.map)
print("mAP50   :", metrics.box.map50)
print("mAP75   :", metrics.box.map75)
print("precision:", metrics.box.mp)
print("recall   :", metrics.box.mr)
print("每类 AP  :", metrics.box.maps)  # 80 个类别的 AP 数组
```

> 注意：Ultralytics 原生指标**不提供** `AP_small/medium/large`（没有 `metrics.box.small_map` 属性）。如需分尺度 AP，需 `save_json=True` 导出预测后用 pycocotools 做 COCO 官方评测。

### 7.4 A/B 对比（关键）

**必须用同一脚本、同一数据、同一 imgsz 分别测基线与 WSCF**：

```powershell
python val_yolov12n.py                                            # 基线（预训练权重）
# 再对 WSCF best.pt 跑同样的 val
```

| 指标 | YOLOv12n（本机实测） | YOLOv12-WSCF | Δ |
|------|---------------------|--------------|---|
| mAP50-95 | 0.4021 | | |
| mAP50 | 0.5549 | | |
| 参数量 | 2,553,904 | 2,671,672 | +117,768 |
| GFLOPs | 6.15（thop）/ 7.0（真实） | 6.7（thop）/ 7.5（真实） | +0.53（真实） |

> 若 WSCF 训练轮数与基线不同（基线是官方 600 epoch，WSCF 只跑 440），**不能直接对比**；正确做法是用相同的两阶段配置与轮数训练 `yolov12.yaml` 基线。

---

## 8. 风险与注意事项

| 风险 | 表现 | 预案 |
|------|------|------|
| 对比基准错误 | 拿 WSCF 40.1 比官网 40.6，误判"掉了 0.5" | 用 `val_yolov12n.py` 实测基线（≈40.2）；同脚本对比 |
| 训练轮数不同 | WSCF 440 epoch vs 官方 600 epoch | A/B 必须同配置同轮数 |
| 阶段 A 误用 `freeze=18` | 新增模块未学习 | 用脚本内的 `FREEZE_HEAD` 逻辑 |
| 阶段 A 过长 | mAP 早早变平、算力浪费 | 40 epoch，观察曲线决定是否提前停 |
| 阶段 A loss 抖动 | lr0=0.009 偏高 | 阶段 A 的 `lr0` 降至 0.005 |
| 显存不足 OOM | CUDA out of memory | 降 `batch`/`imgsz`；或优化 P3 层 CSF 注意力 |
| 阶段 B 后期过拟合 | 验证 loss 回升 | `patience=200` 早停；取 `best.pt` |
| 训练中断 | 断电/终止 | 直接重跑脚本，自动 `resume=True` |
| Windows DataLoader 崩溃 | worker exited unexpectedly | `workers=0` |
| 路径失效 | 找不到数据/权重 | `coco-csf.yaml` 的 `path` 为绝对路径 |

---

## 9. 文件清单

```
train/WSCF/train_stageA.py                     # 阶段 A 脚本（40 epoch，可配 FREEZE_HEAD）
train/WSCF/train_stageB.py                     # 阶段 B 脚本（400 epoch）
val_yolov12n.py                                # 基线复现脚本（COCO val2017）
ultralytics/cfg/models/v12/yolov12-wscf.yaml   # 模型定义
ultralytics/nn/modules/block.py                # WaveletBranch / P1WaveletFusion / CrossScaleFusion
ultralytics/nn/tasks.py                        # parse_model 支持 + 权重重映射
yolov12n.pt                                    # 官方预训练权重
coco-csf.yaml                                  # 数据集配置
docs/YOLOv12-WSCF两阶段训练计划书.md             # 本文档
```
