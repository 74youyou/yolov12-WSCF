# YOLOv12-CSF 训练与验证指南

本指南介绍如何训练和验证 **YOLOv12-CSF**（Cross-Scale Cross-Attention Residual Fusion，跨尺度交叉注意力残差融合）检测模型。

## 1. 改进概述

在保持 YOLOv12 主干、检测头、损失函数与推理流程完全不变的前提下，仅在 PAN 路径末端新增一条 **P3 → P5 大跨度跳跃连接**：

- 浅层特征 P3（高分辨率，含细节）经两次 stride=2 卷积下采样到 P5 分辨率，作为注意力计算的 Key/Value；
- 深层特征 P5 作为 Query；
- 多头交叉注意力输出经 **零初始化输出投影** 后，作为残差信号直接加回 P5；
- 输出投影初始化为零 ⇒ 训练开始时模块为恒等映射，不破坏预训练权重；
- 原有相邻尺度融合（P3→P4、P4→P5）完全保留。

### 1.1 涉及的代码改动

| 文件 | 改动 |
|------|------|
| `ultralytics/nn/modules/block.py` | 新增 `CrossScaleFusion` 模块 |
| `ultralytics/nn/modules/__init__.py` | 导出 `CrossScaleFusion` |
| `ultralytics/nn/tasks.py` | `parse_model` 支持多输入模块；`DetectionModel.load` 增加层序号偏移的权重重映射（插入的新层使 Detect 从 `model.21` 变为 `model.22`，加载官方预训练权重时自动对齐） |
| `ultralytics/cfg/models/v12/yolov12-csf.yaml` | 新模型定义：在第 20 层（P5）后插入第 21 层 `CrossScaleFusion`，Detect 改为读取 `[14, 17, 21]` |

### 1.2 模块参数

YAML 中第 21 层配置：

```yaml
- [[14, 20], 1, CrossScaleFusion, [256, 8, 2, 0.0]] # 21: CSF, P3 -> P5
```

参数含义（`c1_shallow`、`c1_deep` 由解析器根据 `from` 自动推算）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `embed_dim` | 256 | 注意力统一嵌入维度（按模型宽度缩放） |
| `num_heads` | 8 | 注意力头数（按模型宽度缩放） |
| `num_blocks` | 2 | 下采样卷积个数（2 个 stride=2 卷积 = 4 倍下采样，P3→P5） |
| `dropout` | 0.0 | 注意力丢弃概率，可调 0.0~0.1 |

各缩放尺度的实际参数（宽度缩放后）：

| 尺度 | P3 通道 | P5 通道 | embed_dim | num_heads | CSF 参数量 |
|------|---------|---------|-----------|-----------|------------|
| n | 64 | 256 | 64 | 2 | 115,392 |
| s | 128 | 512 | 128 | 4 | 460,160 |
| m | 256 | 512 | 256 | 8 | 1,575,168 |
| l | 256 | 512 | 256 | 8 | 1,575,168 |
| x | 384 | 768 | 384 | 12 | 3,542,400 |

> 注意：`embed_dim` 必须能被 `num_heads` 整除（解析器自动缩放后总是满足；若手动修改配置破坏该约束，模块会抛出断言错误）。

## 2. 环境准备

本仓库建议使用 `yolov12` conda 环境（Python 3.11 + PyTorch 2.6.0+cu124）：

```powershell
# 激活环境
conda activate yolov12
# 安装依赖（若未安装）
pip install -r requirements_window.txt
pip install "torch>=1.9"  # 如尚未安装 PyTorch
```

## 3. 快速验证（建议先跑通）

### 3.1 构建模型并检查输出形状

```python
import torch
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-csf.yaml')  # 默认 n 尺度
print(model.model.info())  # YOLOv12-csf summary: 509 layers, 2,669,296 parameters, 6.3 GFLOPs (n)

x = torch.randn(1, 3, 640, 640)
y = model.model(x)  # 三个检测头输出
for t in y:
    print(t.shape)  # [1, 144, 80, 80], [1, 144, 40, 40], [1, 144, 20, 20]
```

### 3.2 加载官方预训练权重并验证恒等映射

```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-csf.yaml')
model.load('yolov12n.pt')  # 输出 "Transferred 739/759 items" 即正常（其余 20 个为新增 CSF 参数）
```

> 加载日志含义：739 = 官方权重中除 CSF 外的全部参数（包括自动重映射到 `model.22` 的 Detect 头）；20 = 新模块参数，保持默认初始化（输出投影为零 ⇒ 恒等映射）。
>
> 验证恒等性：加载预训练权重后，CSF 输出与原始 P5 完全相同，因此模型前向结果与原始 YOLOv12 一致（diff = 0）。

### 3.3 单元测试（可选）

```python
import torch
from ultralytics.nn.modules import CrossScaleFusion
m = CrossScaleFusion(64, 256, embed_dim=64, num_heads=2, num_blocks=2)
out = m([torch.randn(1, 64, 80, 80), torch.randn(1, 256, 20, 20)])
assert out.shape == (1, 256, 20, 20)  # 输出与深层输入形状一致
# 非整除尺寸自动走插值兜底：
out = m([torch.randn(1, 64, 84, 88), torch.randn(1, 256, 21, 21)])
assert out.shape == (1, 256, 21, 21)
```

### 3.4 冒烟训练（coco8）

```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-csf.yaml')
model.load('yolov12n.pt')
model.train(data='coco8.yaml', epochs=1, imgsz=640, batch=4, workers=2, device=0, plots=False)
```

## 4. 正式训练

### 4.1 方案一（推荐）：预训练权重 → 两阶段微调

**阶段 A：冻结主干，训练新增模块与 PAN**

```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-csf.yaml')
model.load('yolov12n.pt')                      # 加载官方预训练权重
model.train(
    data='coco.yaml',
    epochs=20,                                 # 阶段 A 少量轮次即可
    imgsz=640,
    batch=16,
    device=0,
    freeze=10,                                 # 冻结主干（backbone 为第 0~8 层），仅训练 PAN+CSF
    lr0=0.0005,                                # 较小学习率，保持稳定
    project='runs/csf_stageA',
    name='train',
)
```

**阶段 B：解冻全部参数，全局微调**

```python
model = YOLO('runs/csf_stageA/train/weights/best.pt')  # 从阶段 A 最佳权重继续
model.train(
    data='coco.yaml',
    epochs=80,
    imgsz=640,
    batch=16,
    device=0,
    freeze=None,                               # 解冻全部
    lr0=0.001,                                 # 原始 lr0(0.01) 的 0.1 倍
    project='runs/csf_stageB',
    name='train',
)
```

### 4.2 方案二：从头训练

```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-csf.yaml')
model.train(data='coco.yaml', epochs=300, imgsz=640, batch=16, device=0, project='runs/csf_scratch', name='train')
```

### 4.3 训练注意事项

- `freeze` 按层序号冻结：backbone 为 0~8 层，`freeze=10` 表示冻结第 0~9 层（主干+首个上采样）；CSF 在第 21 层、Detect 在第 22 层。
- 显存占用极小：注意力在 20×20（序列长度 400）上进行，与原始模型相比增加可忽略。
- 其余训练配置（数据增强、优化器、损失函数）与原始 YOLOv12 完全一致，无需修改。
- 推荐使用 `.pt` 继续训练而非每次重新 load，保证优化器状态连续。
- 训练中断后可用 `resume=True` 恢复。

## 5. 验证

### 5.1 标准验证

```python
from ultralytics import YOLO
model = YOLO('runs/csf_stageB/train/weights/best.pt')
metrics = model.val(data='coco.yaml', imgsz=640, device=0, batch=16)
print(metrics.box.map)        # mAP50-95
print(metrics.box.map50)      # mAP50
```

### 5.2 与原始 YOLOv12 对比（A/B 测试）

```python
from ultralytics import YOLO

# 原始模型（训练方式、超参数与 CSF 完全一致，保证公平对比）
orig = YOLO('ultralytics/cfg/models/v12/yolov12.yaml')
orig.load('yolov12n.pt')
# ... 用同样的数据/超参训练原始模型 ...

m_csf  = YOLO('runs/csf_stageB/train/weights/best.pt')
m_orig = YOLO('runs/orig/train/weights/best.pt')
mc = m_csf.val(data='coco.yaml', device=0)
mo = m_orig.val(data='coco.yaml', device=0)
print('CSF  mAP50-95:', mc.box.map)
print('Orig mAP50-95:', mo.box.map)
```

重点观察指标：

- **小目标**：`metrics.box.small_map`（AP_small）
- **中等目标**：`metrics.box.medium_map`（AP_medium）
- **各类别 mAP**：`metrics.box.maps`，关注边缘/遮挡密集类

### 5.3 调优建议

如果效果不明显，按顺序尝试：

1. 增大 `embed_dim`（如 512）或调整 `num_heads`；
2. 增大 `dropout`（0.0 → 0.1）；
3. 在注意力输出后增加归一化/前馈网络（修改 `CrossScaleFusion`，注意保持输出投影零初始化）；
4. 增加 P3→P4 或 P4→P5 的跨尺度连接（在 yaml 中仿照第 21 行再插入一行，例如 `[[14, 17], 1, CrossScaleFusion, [256, 8, 1, 0.0]]`，`num_blocks=1` 表示 2 倍下采样）；
5. 阶段 B 学习率可在原始 `lr0` 的 0.1~0.5 倍之间调整。

## 6. 推理与部署

```python
from ultralytics import YOLO
model = YOLO('runs/csf_stageB/train/weights/best.pt')

# 单图预测
results = model.predict('bus.jpg', imgsz=640, conf=0.25, device=0)

# 批量预测并保存标注图
results = model.predict('datasets/coco8/images/val', imgsz=640, save=True)

# 视频/摄像头（推理流程与原始 YOLOv12 完全一致）
model.predict('input.mp4', save=True)
```

导出（导出流程不变）：

```python
model.export(format='onnx')      # ONNX
model.export(format='torchscript')
```

## 7. 常见问题排查

| 问题 | 原因与处理 |
|------|-----------|
| 加载权重提示 `Transferred 618/759`（而非 739/759） | 使用了旧版本代码/缓存。确认 `DetectionModel.load` 中的 `_remap_shifted_layers` 生效（新增层会使 Detect 序号从 21 变为 22，需自动重映射）；重新启动 Python 进程再试 |
| `embed_dim must be divisible by num_heads` 断言错误 | 手动修改了 embed/heads 配置且不满足整除；恢复 yaml 默认配置 |
| 加载 csf 权重到原始 yaml | 正常：多余的 CSF 参数被自动丢弃（`Transferred 739/739`） |
| 推理时报维度不匹配 | 确认使用本仓库的 `yolov12-csf.yaml` 构建模型，且未手动改动 P3/P5 通道数 |
| 显存不足 | CSF 计算量极小，先检查 `batch`/`imgsz` 设置；若开启 `multi_scale`，偶发非整除尺寸会走插值兜底，不影响显存 |

## 8. 文件清单

```
ultralytics/cfg/models/v12/yolov12-csf.yaml   # CSF 模型定义（新增）
ultralytics/nn/modules/block.py               # CrossScaleFusion 模块（新增类）
ultralytics/nn/modules/__init__.py            # 模块导出
ultralytics/nn/tasks.py                       # parse_model 支持 + 权重加载重映射
yolov12n.pt                                   # 官方预训练权重（加载用）
```
