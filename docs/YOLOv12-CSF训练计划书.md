# YOLOv12-CSF 训练计划书

> 计划书编号：YOLOv12-CSF-TRAIN-001
> 编制日期：2026-08-15
> 适用模型：YOLOv12-CSF（Cross-Scale Cross-Attention Residual Fusion，跨尺度交叉注意力残差融合检测模型）
> 本文档是训练任务的**完整执行计划**，包含目标、数据、策略、超参、监控、评估、验收与风险预案。

---

## 1. 训练目标

在 **COCO 2017 数据集**（本地副本）上训练 YOLOv12-CSF 检测模型，实现以下目标：

| 编号 | 目标 | 验收量化标准 |
|------|------|-------------|
| G1 | 完成模型收敛训练，训练损失稳定下降、无明显发散 | 训练末期 loss 曲线进入平台期（最后 10 个 epoch 波动 < 5%） |
| G2 | 按 COCO 官方指标算法评估模型精度 | 输出 mAP50-95、mAP50、mAP75、AP_s/AP_m/AP_l、AR 等全套指标 |
| G3 | 与原始 YOLOv12（相同数据、相同超参）做 A/B 对比，验证 CSF 模块有效性 | CSF 版 mAP50-95 ≥ 基线版，且小目标 AP_s 不劣化 |
| G4 | 产出可复现的训练产物 | best.pt / last.pt / results.csv / results.png / 混淆矩阵 / 样例预测图 |
| G5 | 记录训练全过程损失变化曲线，形成训练报告 | 本文档第 7 节模板 + `runs/` 下完整日志 |

---

## 2. 数据集说明

### 2.1 数据概况

数据集为 COCO 2017（本地副本，位于 `D:\yolov12\yolov12\datasets\`，已在 `.gitignore` 中排除，不入版本库）：

| 项目 | 数值 |
|------|------|
| 类别数 | 80 类（person、bicycle、car … toothbrush） |
| 训练集图片 | 118,287 张（`datasets/coco/images/train2017/*.jpg`） |
| 验证集图片 | 5,000 张（`datasets/coco/images/val2017/*.jpg`） |
| 训练集标签 | 117,265 个 txt（YOLO 格式，由 COCO json 转换而来，部分无目标图片无标签文件，属正常现象） |
| 验证集标签 | 4,952 个 txt |
| 图片格式 | 全部为 `.jpg`，文件名形如 `000000000241.jpg`（COCO 12 位零填充编号） |
| 标签格式 | YOLO txt：`class_id x_center y_center width height`（归一化坐标） |

### 2.2 目录结构

```
datasets/
├── coco/
│   ├── images/
│   │   ├── train2017/      # 118287 张 jpg
│   │   └── val2017/        # 5000 张 jpg
│   ├── labels/
│   │   ├── train2017/      # 117265 个 txt
│   │   └── val2017/        # 4952 个 txt
│   └── annotations/        # 原始 COCO json 标注（instances_*.json）
├── train2017/  val2017/    # 图片另一副本（coco/ 目录下 images 的 junction 来源）
└── coco_converted/         # 转换过程中间产物
```

### 2.3 数据集配置文件

根目录 `coco-csf.yaml` 已就绪：

```yaml
path: D:/yolov12/yolov12/datasets/coco   # 数据集根目录（绝对路径）
train: images/train2017                  # 训练子集
val: images/val2017                      # 验证子集
names: 0: person, 1: bicycle, ... 79: toothbrush   # COCO 80 类
```

> 注意：`path` 为绝对路径，若迁移机器需同步修改。

---

## 3. 环境准备

### 3.1 软硬件环境

| 项目 | 要求 | 当前状态 |
|------|------|---------|
| 操作系统 | Windows 10/11 | ✅ |
| Python | 3.11 | ✅（conda 环境 `yolov12`） |
| PyTorch | 2.6.0 + cu124（本地已备离线 whl） | ✅ |
| GPU | NVIDIA 显卡，显存 ≥ 8GB，CUDA ≥ 12.4 | 待确认（`device=0`） |
| 依赖 | `requirements_window.txt`（含 flash-attn 离线 whl） | 需 `pip install -r requirements_window.txt` |

### 3.2 环境验证（训练前必跑，约 5 分钟）

```powershell
conda activate yolov12

# 1) 确认 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2) 冒烟验证：模型能构建、能前向、能加载预训练权重
python -c "
from ultralytics import YOLO
m = YOLO('ultralytics/cfg/models/v12/yolov12-csf.yaml')
print(m.model.info())
m.load('yolov12n.pt')   # 预期输出 Transferred 739/759 items
"

# 3) coco8 冒烟训练（1 epoch，确认整套训练管线 OK）
python -c "
from ultralytics import YOLO
m = YOLO('ultralytics/cfg/models/v12/yolov12-csf.yaml')
m.load('yolov12n.pt')
m.train(data='coco8.yaml', epochs=1, imgsz=640, batch=4, workers=2, device=0, plots=False)
"
```

> 若 `Transferred` 数不是 739/759，说明权重重映射逻辑未生效，先排查再训练（见第 9 节）。

---

## 4. 训练策略

### 4.1 总体方案：预训练权重 + 两阶段微调（推荐）

新模型与官方 YOLOv12 相比**只新增了 CSF 模块（第 21 层）**，其余结构完全一致，且 CSF 输出投影零初始化 ⇒ 模块初始为恒等映射 ⇒ 可直接复用官方预训练权重，无需从头训练。

**阶段 A：冻结主干，训练新增模块与 PAN（20 epoch）**

- 目的：让 CSF 注意力模块与 PAN 路径先适配，避免大学习率破坏已收敛的主干特征。
- `freeze=10`：冻结第 0~9 层（backbone 0~8 层 + 首个上采样层），仅训练 PAN + CSF + Detect。
- 小学习率 `lr0=0.0005`（官方默认 0.01 的 1/20），保证稳定。

**阶段 B：解冻全部参数，全局微调（60 epoch）**

- 从阶段 A 的 `best.pt` 继续（保留优化器状态，训练连续）。
- `lr0=0.001`（官方默认的 1/10），`lrf=0.01` 余弦退火到极小值。
- 显存占用增加可忽略（注意力在 20×20=400 token 上计算）。

### 4.2 备选方案

| 方案 | 适用场景 | 说明 |
|------|---------|------|
| 方案二：从头训练（300 epoch） | 追求极致精度、时间充裕 | 无预训练热启动，收敛慢，一般不必要 |
| 方案三：单阶段直接微调（80 epoch） | 时间紧张、快速出结果 | 直接全量微调 + `lr0=0.001`，跳过阶段 A |
| 方案四：多尺度训练 | 目标尺寸分布广 | 在阶段 B 基础上加 `multi_scale=True`，验证集精度通常 +0.3~0.5 mAP |

### 4.3 超参数配置表

| 参数 | 阶段 A | 阶段 B | 说明 |
|------|--------|--------|------|
| `model` | `ultralytics/cfg/models/v12/yolov12-csf.yaml` | `runs/csf_stageA/train/weights/best.pt` | 默认 n 尺度（如需 s/m 见 4.4） |
| `data` | `coco-csf.yaml` | `coco-csf.yaml` | 绝对路径，勿用 `coco.yaml`（会触发联网下载） |
| `epochs` | 20 | 60 | 共 80 epoch |
| `imgsz` | 640 | 640 | 与预训练一致 |
| `batch` | 16 | 16 | 按显存调整（8GB 显存降为 8） |
| `workers` | 4 | 4 | Windows 下 0~8 均可 |
| `device` | 0 | 0 | 单卡；多卡用 `0,1`（自动 DDP） |
| `lr0` | 0.0005 | 0.001 | 见 4.1 |
| `lrf` | 0.01 | 0.01 | 余弦退火终值系数 |
| `freeze` | 10 | None | 阶段 B 解冻全部 |
| `optimizer` | auto（SGD） | auto | 与官方一致 |
| `amp` | True | True | 混合精度 |
| `cache` | False | False | 118K 图无法全量缓存进内存；可尝试 `cache=ram` 若内存 ≥ 64GB |
| `plots` | True | True | 输出 loss/mAP 曲线 |
| `project` / `name` | `runs/csf_stageA` / `train` | `runs/csf_stageB` / `train` | 输出目录 |
| `resume` | - | 用 `best.pt` 路径作为 model 传入 | 等价续训 |

> 其余超参（mosaic=1.0、mixup=0.0、box/cls/dfl 损失权重等）保持官方默认，与基线完全一致，保证 A/B 公平性。

### 4.4 模型尺度选择

`yolov12-csf.yaml` 的 `scales` 定义了 n/s/m/l/x 五档，默认 `n`：

| 尺度 | 参数量 | GFLOPs | 适用场景 |
|------|--------|--------|---------|
| n（默认） | 2.55M（含 CSF 0.115M） | 6.2 | 本计划默认选择：训练快、显存小、便于快速验证 |
| s | 9.13M | 19.7 | 精度/速度均衡 |
| m | 19.67M | 60.4 | 精度优先 |
| l / x | 26.5M / 59.4M | 83.3 / 185.9 | 追求 SOTA，需多卡/长训练 |

> 建议流程：先用 **n 尺度完整跑通**两阶段 + 评估 + A/B 对比（1~2 天量级），确认 CSF 收益后再决定是否升尺度。

---

## 5. 训练执行步骤

### 5.1 训练脚本（已提供 `train_csf_coco.py`，按下面配置执行）

**阶段 A**（新建脚本 `train_stageA.py`）：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolov12-csf.yaml")
model.load("yolov12n.pt")
model.train(
    data="coco-csf.yaml",
    epochs=20,          # 阶段 A：冻结主干，训练新增模块与 PAN
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    freeze=10,          # 冻结第 0~9 层（backbone + 首个上采样）
    lr0=0.0005,         # 官方 lr0(0.01) 的 1/20
    lrf=0.01,
    project="runs/csf_stageA",
    name="train",
    plots=True,
)
```

**阶段 B**（新建脚本 `train_stageB.py`）：

```python
from ultralytics import YOLO

model = YOLO("runs/csf_stageA/train/weights/best.pt")
model.train(
    data="coco-csf.yaml",
    epochs=60,          # 阶段 B：全局微调
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    freeze=None,        # 解冻全部
    lr0=0.001,          # 官方 lr0 的 1/10
    lrf=0.01,
    project="runs/csf_stageB",
    name="train",
    plots=True,
)
```

### 5.2 执行命令

```powershell
conda activate yolov12
python train_stageA.py          # 阶段 A
python train_stageB.py          # 阶段 B（依赖 A 完成）
```

### 5.3 训练中断恢复

```python
model = YOLO("runs/csf_stageB/train/weights/last.pt")
model.train(resume=True)
```

### 5.4 资源与时间估算（单卡 4090/3090 量级，仅供参考）

| 阶段 | 每 epoch 耗时 | 总耗时 |
|------|--------------|--------|
| 阶段 A（20 epoch，冻结主干） | ~8~12 分钟 | 约 3~4 小时 |
| 阶段 B（60 epoch，全量） | ~12~20 分钟 | 约 12~20 小时 |
| 合计 | | **约 15~24 小时** |

> 实际取决于 GPU 型号、batch、`workers`、是否开 `amp`。建议训练期间保持终端前台运行或使用 `nohup`/任务计划，避免中断。

---

## 6. 训练监控：损失值变化

### 6.1 监控内容

训练过程由 Ultralytics 自动记录到 `runs/csf_stageB/train/results.csv`（每 epoch 一行），关键列：

| 列名 | 含义 | 预期趋势 |
|------|------|---------|
| `train/box_loss` | 边界框回归损失（CIoU） | 单调下降后趋平 |
| `train/cls_loss` | 分类损失（BCE） | 单调下降后趋平 |
| `train/dfl_loss` | 分布焦点损失（DFL） | 单调下降后趋平 |
| `metrics/precision` | 验证集精确率 P | 上升趋平 |
| `metrics/recall` | 验证集召回率 R | 上升趋平 |
| `metrics/mAP50` | mAP@IoU=0.5 | 上升趋平 |
| `metrics/mAP50-95` | mAP@0.5:0.95（主指标） | 上升趋平 |
| `val/box_loss` 等 | 验证集各项损失 | 下降后趋平（发散信号：后期回升） |

同时生成 `results.png`（曲线图）、`confusion_matrix.png`、`train_batch*.jpg`（增强后训练样本）、`val_batch*.jpg`（预测样本）等可视化产物。

### 6.2 损失收敛判据

1. **训练损失**：三项目 `box/cls/dfl` 及总和 `train/loss` 单调下降，最终 10 个 epoch 相对波动 < 5%；
2. **验证损失**：与训练损失同步下降并趋平；若验证损失上升而训练损失下降 ⇒ 过拟合，应提前停止（`patience=20` 或手动中断取 best.pt）；
3. **mAP 判据**：`mAP50-95` 最后 10 个 epoch 增幅 < 0.1 视为收敛；
4. **数值参考范围**（n 尺度，COCO，全量训练 80 epoch 量级）：最终 `train/cls_loss` 约 0.4~0.6、`box_loss` 约 0.7~0.9、`dfl_loss` 约 0.9~1.1（与官方 600 epoch 日志 `logs/yolov12n.csv` 对照看趋势即可，绝对值因训练轮数不同不强行对齐）。

### 6.3 监控操作方法

```python
# 训练结束后绘图（trainer 已自动生成，以下为手动复现/查看）
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/csf_stageB/train/results.csv")
plt.figure(figsize=(10, 6))
for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
    plt.plot(df[col], label=col)
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid()
plt.savefig("loss_curves.png", dpi=150)
```

---

## 7. COCO 指标评估（按 COCO 官方算法）

### 7.1 评估脚本

```python
from ultralytics import YOLO

model = YOLO("runs/csf_stageB/train/weights/best.pt")
metrics = model.val(data="coco-csf.yaml", imgsz=640, batch=16, device=0)

print("mAP50-95 :", metrics.box.map)      # 主指标
print("mAP50    :", metrics.box.map50)    # AP@0.5
print("mAP75    :", metrics.box.map75)    # AP@0.75
print("AP_small :", metrics.box.small_map)   # 面积 < 32²
print("AP_medium:", metrics.box.medium_map)  # 32² ~ 96²
print("AP_large :", metrics.box.large_map)   # 面积 > 96²
print("mAR1     :", metrics.box.mar1)     # 每图最多 1 个检测的召回
print("mAR10    :", metrics.box.mar10)
print("mAR100   :", metrics.box.mar100)   # 每图最多 100 个检测
print("每类 AP   :", metrics.box.maps)    # 80 个类别的 AP 数组
```

### 7.2 COCO 指标算法要点（评估口径说明）

| 指标 | 算法要点 |
|------|---------|
| AP@IoU | 在 **10 个 IoU 阈值**（0.5 : 0.05 : 0.95）上分别计算 AP，再取平均 ⇒ `mAP50-95` |
| 单阈值 AP | 每个类别先按置信度降序排列预测框，与真值匹配（IOU ≥ 阈值），算 PR 曲线下面积（101 点插值法，COCO 官方 pycocotools 算法） |
| 面积分档 | 按 GT 框面积分：AP_s（<32²）、AP_m（32²~96²）、AP_l（>96²） |
| 召回上限 | AR 按每图最多 1 / 10 / 100 个检测计算（maxDets=1/10/100） |
| 匹配规则 | 每类单独计算；每个 GT 只匹配一次（贪心匹配最高 IOU） |

> Ultralytics 的 `val.py` 内部即实现上述 COCO 算法（默认不依赖 pycocotools，自研 metrics.py；若安装 pycocotools 亦可选用其验证后端）。训练中每个 epoch 结束时自动执行一次验证，`results.csv` 中的 `metrics/mAP50-95` 即为该指标。

### 7.3 输出产物（`runs/csf_stageB/train/`）

```
├── weights/best.pt, last.pt        # 最优/最新权重
├── results.csv                     # 全量逐 epoch 指标（损失 + mAP 全记录）
├── results.png                     # 损失与 mAP 曲线
├── confusion_matrix.png            # 混淆矩阵（含背景）
├── train_batch0.jpg ...            # 增强训练样本
├── val_batch0_pred.jpg ...         # 验证集预测样例
├── args.yaml                       # 本次训练完整超参存档
└── opt.yaml                        # 配置快照
```

---

## 8. A/B 对比实验（验证 CSF 收益）

### 8.1 基线训练（条件完全一致）

```python
from ultralytics import YOLO

# 基线：官方 YOLOv12（yolov12.yaml），同样的两阶段策略与超参
model = YOLO("ultralytics/cfg/models/v12/yolov12.yaml")
model.load("yolov12n.pt")
# 阶段 A / 阶段 B 配置与 CSF 版逐项相同（freeze=10 → lr0=0.0005 → lr0=0.001）
```

### 8.2 对比表格（模板）

| 指标 | YOLOv12（基线） | YOLOv12-CSF | 差值 Δ |
|------|----------------|-------------|--------|
| mAP50-95 | | | |
| mAP50 | | | |
| mAP75 | | | |
| AP_small | | | |
| AP_medium | | | |
| AP_large | | | |
| 参数量 | 2.44M | 2.55M | +115K（+4.7%） |
| GFLOPs | 6.1 | 6.2 | +0.1 |

### 8.3 重点关注与结论判定

1. **主指标**：mAP50-95 提升即 CSF 有效；
2. **小目标**：`AP_small` 是本模块设计的重点收益方向（P3 高分辨率细节 → P5），若 AP_small 明显提升而总 mAP 持平，说明模块价值在小目标场景；
3. **代价**：参数量仅 +4.7%、FLOPs 几乎不变，若收益 ≥ +0.3 mAP50-95 即判定**值得保留**；
4. **统计显著性**：单次训练有随机性，若 Δ 在 ±0.2 以内，可固定 `seed=0`（`YOLO.train(seed=0)`）重跑一次确认。

---

## 9. 风险与预案

| 风险 | 表现 | 预案 |
|------|------|------|
| 权重加载数不对 | `Transferred 618/759` 而非 `739/759` | `tasks.py` 的 `_remap_shifted_layers` 未生效（代码缓存旧版）。重启 Python / 重装 `pip install -e .` 后再试 |
| 显存不足 OOM | CUDA out of memory | `batch` 16→8→4；或 `imgsz` 640→512；`workers` 调小 |
| Windows 多进程报错 | DataLoader worker 崩溃 | `workers=0` 或 `2`；确认路径无中文空格 |
| 训练发散 | loss 突增 / NaN | 调小 `lr0`（0.0005→0.0002）；检查 `amp=False` 是否恢复 |
| 过拟合 | 验证 loss 回升 | 提前停止（`patience=20`）；或降低 epoch 数 |
| 训练中断 | 断电/手动终止 | `resume=True` 从 `last.pt` 恢复，优化器状态连续 |
| 路径失效 | 找不到数据集 | `coco-csf.yaml` 中 `path` 为绝对路径，迁移机器需同步修改 |
| 验证集类别缺失 | 个别类 AP 为 0 | 属正常（如 hair drier 等稀有类）；关注 80 类平均 |
| 评估结果异常低 | mAP 与预期差距大 | 检查是否误用 `coco.yaml`（联网下载口径）；检查 imgsz 与训练一致 |

---

## 10. 里程碑与时间表

| 阶段 | 内容 | 预计耗时 | 完成标志 |
|------|------|---------|---------|
| M1 环境准备 | 安装依赖、GPU 检查、coco8 冒烟 | 0.5 天 | 冒烟训练无报错 |
| M2 阶段 A 训练 | 冻结主干 20 epoch | 3~4 小时 | 得到 `runs/csf_stageA/train/weights/best.pt` |
| M3 阶段 B 训练 | 全量微调 60 epoch | 12~20 小时 | 损失收敛、得到 best.pt |
| M4 指标评估 | 跑第 7 节评估脚本 | 10~20 分钟 | 输出全套 COCO 指标 |
| M5 A/B 对比 | 训练基线 + 对比表 | 1~2 天 | 填写第 8.2 节表格，给出结论 |
| M6 训练报告 | 整理损失曲线 + 指标 + 结论 | 0.5 天 | 产出最终训练报告 |

**总计：约 3~5 天（含并行/备用 GPU 可压缩）。**

---

## 11. 交付物清单

| 交付物 | 路径 |
|--------|------|
| 最优权重 | `runs/csf_stageB/train/weights/best.pt` |
| 训练日志（逐 epoch 损失 + 指标） | `runs/csf_stageB/train/results.csv` |
| 损失/mAP 曲线图 | `runs/csf_stageB/train/results.png` |
| 混淆矩阵 | `runs/csf_stageB/train/confusion_matrix.png` |
| 验证预测样例图 | `runs/csf_stageB/train/val_batch*_pred.jpg` |
| 训练超参存档 | `runs/csf_stageB/train/args.yaml` |
| 基线权重 | `runs/orig_stageB/train/weights/best.pt` |
| 对比结论 | 第 8.2 节表格（填于训练报告） |

---

## 12. 附：训练结果报告模板（训练完成后填写）

### 12.1 损失曲线摘要

| epoch | train/box | train/cls | train/dfl | val/box | val/cls | val/dfl |
|-------|-----------|-----------|-----------|---------|---------|---------|
| 1 | | | | | | |
| 20 | | | | | | |
| 40 | | | | | | |
| 60 | | | | | | |
| 80（终） | | | | | | |

### 12.2 COCO 指标结果

| 指标 | 数值 | 指标 | 数值 |
|------|------|------|------|
| mAP50-95 | | mAP50 | |
| mAP75 | | AR@100 | |
| AP_small | | AP_medium | |
| AP_large | | 每类 AP 分布 | 80 个值见附表 |

### 12.3 结论

- [ ] CSF 模块在 COCO 上带来 mAP50-95 提升
- [ ] 小目标 AP_small 得到改善
- [ ] 参数量/算力代价可接受
- [ ] 值得继续调优（升尺度 / 多尺度 / 调 embed_dim）
