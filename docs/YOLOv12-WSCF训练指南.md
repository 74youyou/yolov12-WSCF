# YOLOv12-WSCF 训练指南

> 文档编号：YOLOv12-WSCF-TRAIN-001
> 编制日期：2026-09-17
> 适用模型：YOLOv12-WSCF（Wavelet Skip Cross-scale Fusion，小波跳跃跨尺度融合检测模型）
> 本文档介绍如何构建、验证、训练与评估 YOLOv12-WSCF，包含超参配置、执行步骤、监控与排错。

---

## 1. 改进概述

在保持 YOLOv12 主干、检测头、损失函数与推理流程主体不变的前提下，新增一条**小波支线**与**四条融合路径**：

- **小波支线**：从原始 RGB 图像执行 4 级 Haar 小波分解，输出 W1（H/2）、W2（H/4）、W3（H/8）、W4（H/16）；每级经 GRN-v3（输入原始图像）生成各子带缩放权重；
- **路径 1**：P1 与 W1 通道拼接 → 1×1 卷积 → P1'（恒等初始化，训练开始 P1' = P1）；
- **路径 2**：P3 做 Query、W2 做 Key/Value，经 `CrossScaleFusion`（**3×3 窗口注意力，stride=2**）融合；
- **路径 3**：P4 做 Query、W3 做 Key/Value，经 `CrossScaleFusion`（窗口注意力）融合；
- **路径 4**：P5 做 Query、W4 做 Key/Value，经 `CrossScaleFusion`（窗口注意力）融合。

所有融合模块输出投影零初始化 / 恒等初始化 ⇒ 训练开始时模型退化为原始 YOLOv12，可直接复用官方预训练权重。

### 1.1 涉及的代码改动

| 文件 | 改动 |
|------|------|
| `ultralytics/nn/modules/block.py` | 新增 `SubbandGRNv3`、`WaveletLevelBlock`、`WaveletBranch`、`P1WaveletFusion`；重写 `CrossScaleFusion` 为**窗口注意力** |
| `ultralytics/nn/modules/__init__.py` | 导出上述模块 |
| `ultralytics/nn/tasks.py` | `parse_model` 支持 `WaveletBranch`（多输出）与 `P1WaveletFusion`（多输入）；修正 `CrossScaleFusion` 输出通道 `c2` |
| `ultralytics/utils/torch_utils.py` | `get_flops` 改用模型声明的输入通道（否则首参不再是 stem 卷积，FLOPs 会报 0） |
| `ultralytics/cfg/models/v12/yolov12-wscf.yaml` | 新模型定义（新增） |
| `train_wscf.py` | 训练入口（新增） |

> 多输出说明：设计书建议的 `[layer, out_idx]` 索引语法需改核心前向，本实现改用框架自带的 `Index` 层，从**单次**小波计算中选择 W1/W2/W3/W4，功能等价且不改 `_predict_once`。

### 1.2 网络结构（YAML 层索引）

```yaml
backbone:
  - [-1, 1, nn.Identity, []]                        # 0: 原始输入图像 (B,3,H,W)
  - [-1, 1, WaveletBranch, [True, 0.25]]            # 1: 4 级 Haar + GRN-v3（输入原始图像），返回 (W1,W2,W3,W4)，各 12ch
  - [1, 1, Index, [12, 0]]                          # 2: W1 (H/2)
  - [1, 1, Index, [12, 1]]                          # 3: W2 (H/4)
  - [1, 1, Index, [12, 2]]                          # 4: W3 (H/8)
  - [1, 1, Index, [12, 3]]                          # 5: W4 (H/16)
  - [0, 1, Conv, [64, 3, 2]]                        # 6: P1/2
  - [[6, 2], 1, P1WaveletFusion, []]                # 7: P1 + W1 -> C1
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]                # 8: P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]               # 9
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]                # 10: P3/8
  - [[3, 10], 1, CrossScaleFusion, [256, 16, 3, 2, 0.0]] # 11: P3 <- W2（窗口注意力，embed=64/heads=4 @ n）
  - [-1, 2, C3k2, [512, False, 0.25]]               # 12
  - [-1, 1, Conv, [512, 3, 2]]                      # 13: P4/16
  - [[4, 13], 1, CrossScaleFusion, [512, 32, 3, 2, 0.0]] # 14: P4 <- W3（embed=128/heads=8 @ n）
  - [-1, 4, A2C2f, [512, True, 4]]                  # 15
  - [-1, 1, Conv, [1024, 3, 2]]                     # 16: P5/32
  - [[5, 16], 1, CrossScaleFusion, [512, 32, 3, 2, 0.0]] # 17: P5 <- W4（embed=128/heads=8 @ n）
  - [-1, 4, A2C2f, [1024, True, 1]]                 # 18
head:
  # 19~30: 与原 YOLOv12 一致的 PAN（Upsample / Concat / A2C2f / C3k2）
  - [[24, 27, 30], 1, Detect, [nc]]                 # 31: Detect(P3, P4, P5)
```

> `CrossScaleFusion` 的 args 为 `[embed_dim, num_heads, window_size, stride, dropout]`（窗口注意力，3×3、stride=2）。`embed_dim`/`num_heads` 按宽度缩放（n 尺度 ×0.25）。

### 1.3 模块参数（n 尺度，宽度缩放后）

| 层 | 模块 | 关键参数 | 参数量 |
|----|------|---------|--------|
| 1 | WaveletBranch | in_ch=3；GRN-v3 输入原始图像；W1/W2/W3/W4 各 4×3=12ch | 288 |
| 7 | P1WaveletFusion | c_p1=16, c_w1=12 | 464 |
| 11 | CrossScaleFusion | shallow=W2(12), deep=P3(64), embed=64, heads=4, window 3×3 | 10,376 |
| 14 | CrossScaleFusion | shallow=W3(12), deep=P4(128), embed=128, heads=8, window 3×3 | 36,872 |
| 17 | CrossScaleFusion | shallow=W4(12), deep=P5(256), embed=128, heads=8, window 3×3 | 69,768 |
| — | **合计新增** | | **117,768** |

> 小波支线**不做 1×1 升维**：GRN-v3 以**原始图像**为输入，生成各层级子带（LL/LH/HL/HH）的缩放权重，缩放后的 4 个子带直接拼接（4×in_ch=12 通道）。`CrossScaleFusion` 采用**窗口注意力**：深层每个位置只与浅层以 `(2i,2j)` 为中心的 3×3 邻域（9 个候选）做交叉注意力，K/V 在浅层全分辨率生成，`embed_dim`/`num_heads` 随宽度缩放。

### 1.4 各尺度整网参数

| 尺度 | 层数 | 参数量 | GFLOPs（thop / 真实*） |
|------|------|--------|--------|
| n（默认） | 588 | 2,671,672 | 6.7 / 7.5 |
| 基线 YOLOv12-n | 497 | 2,553,904 | 6.15 / 7.0 |

\* thop 漏算 `q@kᵀ`/`attn@v` 矩阵乘，真实 FLOPs 用 torch profiler 测得。

> n 尺度相比基线：参数 +117,768（+4.6%），真实 FLOPs +0.53（+7.6%）。窗口注意力（3×3、stride=2）替换了原来的全注意力：P3 用 embed=64、P4/P5 用 embed=128，真实 FLOPs 仍仅比基线高约 0.5G。

---

## 2. 环境准备

```powershell
conda activate yolov12            # Python 3.11 + PyTorch 2.6.0+cu124
pip install -r requirements_window.txt   # 如尚未安装依赖
```

GPU 显存建议 ≥ 8GB（P3 层 CSF 注意力矩阵在 batch=16、640 输入时约 5GB 量级，见第 7 节显存说明）。

---

## 3. 快速验证（训练前必跑）

### 3.1 构建模型并检查输出形状

```python
import torch
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolov12-wscf.yaml")   # 默认 n 尺度
print(model.model.info())
# 预期：YOLOv12-wscf summary: 588 layers, 2,671,672 parameters, 6.7 GFLOPs

x = torch.randn(1, 3, 640, 640)
y = model.model(x)
for t in y:                       # Detect 输出
    print(t.shape)                # [1, 84, 8400] 附近（推理态）或三尺度特征
```

### 3.2 加载官方预训练权重并验证恒等映射

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolov12-wscf.yaml")
model.load("yolov12n.pt")
# 预期：Transferred 739/845 items from pretrained weights
#   739 = 官方全部参数；845-739 = 106 个新增模块参数（保持零/恒等初始化）
```

> 恒等性验证：加载预训练权重后，融合模块为恒等映射，模型前向结果应与原始 YOLOv12 完全一致（实测 max abs diff = 0.0）。

### 3.3 单元测试（可选）

```python
import torch
from ultralytics.nn.modules import WaveletBranch, P1WaveletFusion, CrossScaleFusion

# 小波支线：4 级 Haar + GRN（输入原始图像），缩放后 4 子带直接拼接（每级 4×3=12 通道）；128 -> W1 64 / W2 32 / W3 16 / W4 8
wb = WaveletBranch(3)
w1, w2, w3, w4 = wb(torch.randn(1, 3, 128, 128))
assert (w1.shape, w2.shape, w3.shape, w4.shape) == ((1, 12, 64, 64), (1, 12, 32, 32), (1, 12, 16, 16), (1, 12, 8, 8))

# P1WaveletFusion：初始化即恒等映射（W1 为 12 通道）
p1 = torch.randn(1, 64, 32, 32)
out = P1WaveletFusion(64, 12)([p1, torch.randn(1, 12, 32, 32)])
assert torch.allclose(out, p1, atol=1e-6)

# CrossScaleFusion（窗口注意力）：输出零初始化 -> 恒等映射
deep = torch.randn(1, 64, 16, 16)
out = CrossScaleFusion(12, 64, embed_dim=64, num_heads=4)(
    [torch.randn(1, 12, 32, 32), deep]
)
assert out.shape == deep.shape and torch.allclose(out, deep, atol=1e-6)
```

### 3.4 冒烟训练（coco8）

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolov12-wscf.yaml")
model.load("yolov12n.pt")
model.train(data="coco8.yaml", epochs=1, imgsz=640, batch=4, workers=2, device=0, plots=False)
```

> 也可直接运行仓库根目录的 `train_wscf.py`（默认指向 `coco-csf.yaml` 与 50 epoch，请按需修改）。

---

## 4. 正式训练

### 4.1 方案一（推荐）：单阶段全量微调

由于所有融合模块在初始化时退化为恒等映射，模型起点等价于原始 YOLOv12，直接全量微调即可，训练稳定、流程简单。

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolov12-wscf.yaml")
model.load("yolov12n.pt")
model.train(
    data="coco-csf.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    lr0=0.001,          # 官方默认(0.01)的 1/10
    lrf=0.01,
    project="runs/wscf",
    name="train",
    plots=True,
)
```

### 4.2 方案二：两阶段微调（更保守）

**阶段 A：冻结原始主干，训练新增模块与 PAN**

WSCF 的 backbone 为第 0~16 层、head 从第 17 层开始。为只训练**新增模块**（WaveletBranch/Index/P1WaveletFusion/CSF）与 PAN，可用列表冻结原主干中来自预训练的卷积块：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolov12-wscf.yaml")
model.load("yolov12n.pt")
model.train(
    data="coco-csf.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    freeze=[5, 7, 8, 9, 11, 12, 14, 15, 16],  # 冻结原主干 Conv/C3k2/A2C2f
    lr0=0.0005,
    lrf=0.01,
    project="runs/wscf_stageA",
    name="train",
    plots=True,
)
```

> `freeze` 支持 int（冻结前 N 层）或层索引列表。注意不要用 `freeze=18`：那会把第 1、7、11、14、17 层等**新增模块也一并冻结**，导致小波融合无法学习。

**阶段 B：解冻全部参数，全局微调**

```python
from ultralytics import YOLO

model = YOLO("runs/wscf_stageA/train/weights/best.pt")   # 从阶段 A 最佳权重继续
model.train(
    data="coco-csf.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    freeze=None,
    lr0=0.001,
    lrf=0.01,
    project="runs/wscf_stageB",
    name="train",
    plots=True,
)
```

### 4.3 超参数配置表

| 参数 | 方案一（单阶段） | 方案二阶段 A | 方案二阶段 B | 说明 |
|------|----------------|-------------|-------------|------|
| `model` | `yolov12-wscf.yaml` | `yolov12-wscf.yaml` | `runs/wscf_stageA/.../best.pt` | n 尺度 |
| `data` | `coco-csf.yaml` | `coco-csf.yaml` | `coco-csf.yaml` | 勿用 `coco.yaml`（会触发联网下载） |
| `epochs` | 50 | 20 | 60 | |
| `imgsz` | 640 | 640 | 640 | 与预训练一致 |
| `batch` | 16（显存不足降至 8/4） | 16 | 16 | |
| `workers` | 4 | 4 | 4 | Windows 下可用 0~8 |
| `device` | 0 | 0 | 0 | 多卡写 `0,1`（自动 DDP） |
| `lr0` | 0.001 | 0.0005 | 0.001 | 保持稳定 |
| `lrf` | 0.01 | 0.01 | 0.01 | 余弦退火终值系数 |
| `freeze` | None | `[5,7,8,9,11,12,14,15,16]` | None | 见 4.2 说明 |
| `optimizer` | auto | auto | auto | 与官方一致 |
| `amp` | True | True | True | 混合精度 |
| `plots` | True | True | True | 输出曲线 |

> 其余超参（mosaic=1.0、mixup=0.0、box/cls/dfl 权重等）保持官方默认，保证与基线 A/B 公平。

### 4.4 尺度选择

`yolov12-wscf.yaml` 的 `scales` 定义 n/s/m/l/x 五档，默认 n：

| 尺度 | 参数量 | GFLOPs | 适用场景 |
|------|--------|--------|---------|
| n（默认） | 2.59M | 6.4 | 快速验证、显存小 |
| s | 9.2M+ | 待测 | 精度/速度均衡 |
| m / l / x | 20M+ | 待测 | 精度优先，需大显存/长训练 |

> 建议先用 **n 尺度**跑通全流程并做 A/B 对比，确认收益后再升尺度。

### 4.5 训练中断恢复

```python
from ultralytics import YOLO

model = YOLO("runs/wscf/train/weights/last.pt")
model.train(resume=True)
```

---

## 5. 验证与评估

### 5.1 标准验证

```python
from ultralytics import YOLO

model = YOLO("runs/wscf/train/weights/best.pt")
metrics = model.val(data="coco-csf.yaml", imgsz=640, batch=16, device=0)

print("mAP50-95 :", metrics.box.map)          # 主指标
print("mAP50    :", metrics.box.map50)
print("mAP75    :", metrics.box.map75)
print("precision:", metrics.box.mp)
print("recall   :", metrics.box.mr)
print("每类 AP   :", metrics.box.maps)
# 注意：Ultralytics 原生指标不提供 AP_small/medium/large（无 metrics.box.small_map 属性）。
# 如需分尺度 AP，需 save_json=True 导出预测后用 pycocotools 做 COCO 官方评测。
```

### 5.2 输出产物（`runs/wscf/train/`）

```
├── weights/best.pt, last.pt   # 最优 / 最新权重
├── results.csv                # 逐 epoch 损失与指标
├── results.png                # 损失与 mAP 曲线
├── confusion_matrix.png       # 混淆矩阵
├── train_batch*.jpg           # 增强训练样本
├── val_batch*_pred.jpg        # 验证集预测样例
└── args.yaml                  # 本次训练超参存档
```

---

## 6. A/B 对比实验

```python
from ultralytics import YOLO

# 基线：原始 YOLOv12，数据/超参/训练轮次与 WSCF 完全一致
base = YOLO("ultralytics/cfg/models/v12/yolov12.yaml")
base.load("yolov12n.pt")
# ... 用同样的配置训练基线 ...

m_base = YOLO("runs/base/train/weights/best.pt")
m_wscf = YOLO("runs/wscf/train/weights/best.pt")
mb = m_base.val(data="coco-csf.yaml", device=0)
mw = m_wscf.val(data="coco-csf.yaml", device=0)
print("基线 mAP50-95:", mb.box.map)
print("WSCF mAP50-95:", mw.box.map)
```

| 指标 | YOLOv12（基线） | YOLOv12-WSCF | Δ |
|------|----------------|--------------|---|
| mAP50-95 | | | |
| mAP50 | | | |
| AP_small | | | |
| AP_medium | | | |
| AP_large | | | |
| 参数量 | 2,553,904 | 2,671,672 | +117,768 |
| GFLOPs | 6.15（thop）/ 7.0（真实） | 6.7（thop）/ 7.5（真实） | +0.53（真实） |

**结论判定**：主指标 mAP50-95 提升即有效；重点关注 `AP_small`（小波细节注入的主要收益方向）。若 Δ 在 ±0.2 以内，可用 `seed=0` 重跑确认。

---

## 7. 调优与显存说明

### 7.1 计算量/显存说明

CSF 已采用**窗口注意力（3×3、stride=2）**：深层每个位置只与浅层以 `(2i,2j)` 为中心的 9 个邻域候选做注意力，P3 的注意力矩阵从 6400×6400 降为 6400×9。整网真实 FLOPs 约 **7.5G**（thop 6.7G），比基线 7.0G 仅 **+0.53G**，显存也大幅下降。

若仍需压缩，按顺序尝试：

1. 降低 `batch`（16 → 8 → 4）或 `imgsz`（640 → 512）；
2. 减小 `num_heads` 或 `embed_dim`（注意保持 `head_dim >= 16`）。

### 7.2 其他调优方向

| 调整 | 方式 | 适用场景 |
|------|------|---------|
| 增强注意力容量 | 调大 `CrossScaleFusion` 的 `embed_dim`（如 P5 用 512） | 精度优先 |
| 多头多样性 | 调大 `num_heads`（须被 `embed_dim` 整除，且 `head_dim>=16`） | 精度优先 |
| 抑制过拟合 | `dropout` 0.0 → 0.1 | 小数据集 |
| 关闭 GRN | `WaveletBranch` 第二个参数改 `False` | 消融/简化 |
| 多尺度训练 | 阶段 B 加 `multi_scale=True` | 目标尺寸分布广 |
| 升尺度 | n → s/m/l/x | 精度优先 |

---

## 8. 常见问题排查

| 问题 | 原因与处理 |
|------|-----------|
| `Transferred 739/845` 而非 845/845 | 正常：739 为官方全部参数，106 个新增参数保持零/恒等初始化 |
| 加载权重后精度异常 / 前向不一致 | 检查融合模块初始化是否被覆盖；确认使用的是本仓库的 `yolov12-wscf.yaml` |
| GFLOPs 显示 0.0 | `get_flops` 未使用模型声明输入通道（本仓库已修复）；确认代码已更新 |
| `freeze` 后新增模块未学习 | 勿用 `freeze=18`（会把 1/7/11/14/17 层新增模块冻结）；应使用列表冻结原主干层（见 4.2） |
| 显存不足 OOM | 降 `batch`/`imgsz`；或按 7.1 优化 P3 层注意力 |
| 训练发散 | 调小 `lr0`（0.001 → 0.0005 → 0.0002）；必要时 `amp=False` |
| Windows 多进程报错 | `workers=0` 或 `2`；确认路径无中文/空格 |
| 训练中断 | `resume=True` 从 `last.pt` 恢复 |

> 环境备注：本机（RTX 3060 Laptop 6GB）在 GPU 训练时出现 `CUDA error: unknown error`，但**原始 YOLOv12 基线同样报错**，属该机 GPU/驱动环境问题，与模型无关；CPU 上完整训练管线（前向/损失/反向/验证/保存）已验证通过。

---

## 9. 文件清单

```
ultralytics/cfg/models/v12/yolov12-wscf.yaml   # WSCF 模型定义（新增）
ultralytics/nn/modules/block.py                # WaveletBranch / P1WaveletFusion / CrossScaleFusion
ultralytics/nn/modules/__init__.py             # 模块导出
ultralytics/nn/tasks.py                        # parse_model 多输出/多输入支持 + 权重加载重映射
ultralytics/utils/torch_utils.py               # get_flops 输入通道修正
train_wscf.py                                  # 训练入口（新增）
yolov12n.pt                                    # 官方预训练权重（加载用）
coco-csf.yaml                                  # 数据集配置（COCO 本地副本）
docs/YOLOv12-WSCF 模型介绍书.md                 # 设计规划书
docs/YOLOv12-WSCF训练指南.md                    # 本文档
```
