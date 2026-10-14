# YOLOv12-WSCF 完整设计规划书

> 文档编号：YOLOv12-WSCF-DESIGN-005  
> 编制日期：2026-09-17  
> 目标：在 YOLOv12 中新增小波支线与三条融合路径，实现多尺度小波信息注入，提升中小目标与恶劣天气下的检测性能。  
> 适用对象：可直接依据本规划书实现代码的工程人员/Agent。

---

## 1. 设计目标

在 YOLOv12 的 backbone 中新增：

- **小波支线**：从原始 RGB 图像出发，执行 5 级 Haar 小波分解，输出 W1（320×320）、W2（160×160）、W3（80×80）。
- **路径 1**：P1 + W1 通道拼接 → 1×1 卷积 → C1（从最浅层注入低频）。
- **路径 2**：P3 做 Q，W2 做 KV，使用 CrossScaleFusion 融合（P3 80×80，W2 160×160，num_blocks=1）。
- **路径 3**：P4 做 Q，W3 做 KV，使用 CrossScaleFusion 融合（P4 40×40，W3 80×80，num_blocks=1）。
- **P5 不做小波融合**，保持原样。

所有融合模块输出投影零初始化，训练开始时退化为原始 YOLOv12，不破坏预训练权重。

---

## 2. 总体架构

```
输入图像 (640×640×3)
   │
   ├──────────────────────────────────────┐
   │                                      │
   ▼                                      ▼
YOLOv12 Backbone                    小波支线（5 级分解）
   │                                      │
   ├── P1 (320×320, C1) ── P1+W1 拼接 ──► P1' (C1)
   │      │                    ▲              │
   │      │                    │              │
   │      │                    └── W1 (320×320, C_w1)
   │      │                                   │
   │      ▼ 下采样                             │
   │   P2 (160×160)                        W2 (160×160, C_w2)
   │      │                                   │
   │      ▼ 下采样                             │
   │   P3 (80×80) ──► CSF(Q=P3, KV=W2) ──► P3'
   │      │                                   │
   │      ▼ 下采样                             ▼
   │   P4 (40×40) ──► CSF(Q=P4, KV=W3) ──► P4'
   │      │                                   │
   │      ▼                                   │
   │   P5 (20×20)                          W3 (80×80, C_w3)
   │
   ▼
Neck (PAN) + Detect(P3', P4', P5)
```

---

## 3. 小波支线实现（WaveletBranch）

### 3.1 功能
- 输入：原始 RGB 图像 `(B, 3, H, W)`；
- 执行 5 级 Haar 小波分解；
- 输出：W1（320×320）、W2（160×160）、W3（80×80）；
- 通道数：`c_w1=32, c_w2=128, c_w3=256`（可按需调整）。

### 3.2 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletBranch(nn.Module):
    """小波支线：5 级 Haar 分解，输出 W1, W2, W3。"""
    def __init__(self, in_ch=3, c_w1=32, c_w2=128, c_w3=256):
        super().__init__()
        self.conv_w1 = nn.Conv2d(4 * in_ch, c_w1, 1)
        self.conv_w2 = nn.Conv2d(4 * in_ch, c_w2, 1)
        self.conv_w3 = nn.Conv2d(4 * in_ch, c_w3, 1)

    def haar2d(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='replicate')
        ll = torch.tensor([[1,1],[1,1]], dtype=x.dtype, device=x.device) / 2.0
        lh = torch.tensor([[1,-1],[1,-1]], dtype=x.dtype, device=x.device) / 2.0
        hl = torch.tensor([[1,1],[-1,-1]], dtype=x.dtype, device=x.device) / 2.0
        hh = torch.tensor([[1,-1],[-1,1]], dtype=x.dtype, device=x.device) / 2.0
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # (4,1,2,2)
        filters = filters.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        out = F.conv2d(x, filters, stride=2, groups=C)
        out = out.view(B, C, 4, H//2, W//2).permute(0, 2, 1, 3, 4)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def forward(self, x):
        ll1, lh1, hl1, hh1 = self.haar2d(x)      # 320×320
        ll2, lh2, hl2, hh2 = self.haar2d(ll1)    # 160×160
        ll3, lh3, hl3, hh3 = self.haar2d(ll2)    # 80×80
        w1 = self.conv_w1(torch.cat([ll1, lh1, hl1, hh1], dim=1))  # (B,c_w1,320,320)
        w2 = self.conv_w2(torch.cat([ll2, lh2, hl2, hh2], dim=1))  # (B,c_w2,160,160)
        w3 = self.conv_w3(torch.cat([ll3, lh3, hl3, hh3], dim=1))  # (B,c_w3,80,80)
        return w1, w2, w3
```

---

## 4. P1 + W1 融合模块（P1WaveletFusion）

### 4.1 设计
- 输入：P1（C1=64，320×320）、W1（C_w1=32，320×320）
- 拼接后通道：C1 + C_w1 = 96
- 1×1 卷积输出：C1 = 64
- 部分恒等初始化：前 C1 个输入通道权重恒等，后 C_w1 个输入通道权重为零
- 训练开始时 P1' = P1

### 4.2 代码

```python
class P1WaveletFusion(nn.Module):
    """P1 + W1 拼接后，1×1 卷积输出 C1。零初始化，训练开始为恒等映射。"""
    def __init__(self, c_p1, c_w1):
        super().__init__()
        self.fuse = nn.Conv2d(c_p1 + c_w1, c_p1, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        with torch.no_grad():
            for i in range(c_p1):
                self.fuse.weight[i, i, 0, 0] = 1.0

    def forward(self, p1, w1):
        x = torch.cat([p1, w1], dim=1)
        return self.fuse(x)  # 输出通道 = c_p1
```

---

## 5. CSF 模块（复用已有 CrossScaleFusion）

### 5.1 已有代码

```python
class CrossScaleFusion(nn.Module):
    def __init__(self, c1_shallow, c1_deep, embed_dim=256, num_heads=8, num_blocks=2, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.c2 = c1_deep
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        k_layers = []
        in_ch = c1_shallow
        for i in range(num_blocks):
            k_layers.append(DWConv(in_ch, in_ch, 3, 2, act=False))
            k_layers.append(Conv(in_ch, embed_dim, 1, 1, act=False))
            in_ch = embed_dim
        if num_blocks == 0:
            k_layers = [Conv(c1_shallow, embed_dim, 1, 1, act=False)]
        self.k_path = nn.Sequential(*k_layers)

        v_layers = []
        in_ch = c1_shallow
        for i in range(num_blocks):
            v_layers.append(DWConv(in_ch, in_ch, 3, 2, act=False))
            v_layers.append(Conv(in_ch, embed_dim, 1, 1, act=False))
            in_ch = embed_dim
        if num_blocks == 0:
            v_layers = [Conv(c1_shallow, embed_dim, 1, 1, act=False)]
        self.v_path = nn.Sequential(*v_layers)

        self.q_proj = nn.Linear(c1_deep, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, c1_deep)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        shallow, deep = x
        B, C_s, H_s, W_s = shallow.shape
        B, C_d, H_d, W_d = deep.shape

        k = self.k_path(shallow)
        v = self.v_path(shallow)

        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        q = deep.flatten(2).transpose(1, 2)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        N = q.shape[1]
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(B, C_d, H_d, W_d)

        return deep + out
```

### 5.2 用于 P3 ← W2
- `c1_shallow` = W2 通道数（128）
- `c1_deep` = P3 通道数（256）
- `num_blocks` = 1

### 5.3 用于 P4 ← W3
- `c1_shallow` = W3 通道数（256）
- `c1_deep` = P4 通道数（512）
- `num_blocks` = 1

---

## 6. YAML 配置

```yaml
backbone:
  - [-1, 1, Identity, []]                    # 0: 原始图像
  - [[0], 1, WaveletBranch, [32, 128, 256]]  # 1: 小波支线，输出 W1, W2, W3（多输出）

  - [[0], 1, Conv, [64, 3, 2]]               # 2: P1/2 (C1=64)
  - [[2, 1], 1, P1WaveletFusion, [64]]       # 3: P1 + W1 → C1
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]         # 4: P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]        # 5
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]         # 6: P3/8
  - [[6, 1], 1, CrossScaleFusion, [256, 8, 1, 0.0]]  # 7: P3 ← W2
  - [-1, 2, C3k2, [512, False, 0.25]]        # 8
  - [-1, 1, Conv, [512, 3, 2]]               # 9: P4/16
  - [[9, 1], 1, CrossScaleFusion, [256, 8, 1, 0.0]]  # 10: P4 ← W3
  - [-1, 4, A2C2f, [512, True, 4]]           # 11
  - [-1, 1, Conv, [1024, 3, 2]]              # 12: P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]          # 13

head:
  # ... 原有 PAN 路径 ...
  # Detect 使用 P3'（层 7）、P4'（层 10）、P5（层 13）
```

**说明**：
- 层 1 是 `WaveletBranch`，返回 `(W1, W2, W3)`，在后续层通过 `[1, 0]`、`[1, 1]`、`[1, 2]` 分别引用。
- 层 3 的 `from` 为 `[2, 1]`，表示取层 2（P1）和层 1 的第 1 个输出（W1）。需要 `parse_model` 支持这种索引。
- 层 7 的 `from` 为 `[6, 1]`，但这里 `1` 是层索引，实际应写为 `[[6], [1, 1]]` 或类似，表示层 6 和层 1 的第 1 个输出（W2）。为了清晰，我们定义一种语法：`from` 列表中的每个元素可以是整数（层索引）或 `[layer_idx, output_idx]`。在 `parse_model` 中解析。

---

## 7. parse_model 修改

在 `ultralytics/nn/tasks.py` 的 `parse_model` 中：

```python
# 解析 from 列表，支持 [layer_idx, output_idx]
from_list = []
for f in from:
    if isinstance(f, (list, tuple)):
        layer_idx, out_idx = f
        from_list.append((layer_idx, out_idx))
    else:
        from_list.append((f, 0))  # 默认取第 0 个输出

# 获取输入通道数
ch = []
for layer_idx, out_idx in from_list:
    if isinstance(m, WaveletBranch) and layer_idx == 1:  # 假设小波支线层索引为1
        # 根据 out_idx 选择通道数
        if out_idx == 0:
            ch.append(c_w1)
        elif out_idx == 1:
            ch.append(c_w2)
        elif out_idx == 2:
            ch.append(c_w3)
    else:
        ch.append(ch_list[layer_idx])
```

具体实现时，需要维护一个 `ch_list` 记录每层输出的通道数。对于多输出模块，需要记录每个输出的通道数。这需要更细致的修改。为了简化，可以在 `WaveletBranch` 中返回一个列表，并在 `parse_model` 中特殊处理。

另一种更简单的方案：将小波支线拆成三个独立模块，分别输出 W1、W2、W3，但这样会重复计算小波分解。考虑到计算量不大（Haar 小波仅加减法），也可以接受。但为了优雅，推荐多输出方案。

在规划书中，我们给出多输出方案的实现思路，并说明如果实现困难，可以退化为三个独立模块。

---

## 8. 初始化与训练策略

| 模块                     | 初始化             | 训练开始时行为 |
| ------------------------ | ------------------ | -------------- |
| P1WaveletFusion          | 部分恒等           | P1' = P1       |
| CrossScaleFusion (P3←W2) | 输出投影零初始化   | P3' = P3       |
| CrossScaleFusion (P4←W3) | 输出投影零初始化   | P4' = P4       |
| WaveletBranch            | 1×1 卷积随机初始化 | 输出小波特征   |

**训练开始时**：所有融合模块退化为恒等映射，模型等价于原始 YOLOv12。  
**训练开始后**：梯度逐渐流向小波分支，学习增量精化信号。

---

## 9. 计算量估算

| 模块            | 空间尺度 | token 数 | 注意力矩阵 | 估算 FLOPs           |
| --------------- | -------- | -------- | ---------- | -------------------- |
| P1WaveletFusion | 320×320  | -        | -          | 极低（1×1 卷积）     |
| CSF (P3←W2)     | 80×80    | 6400     | 6400×6400  | 需窗口注意力，否则大 |
| CSF (P4←W3)     | 40×40    | 1600     | 1600×1600  | 可接受               |

**建议**：
- P3 层 CSF 用**窗口注意力**（8×8 窗口）或 KV 下采样，控制计算量；
- P4 层 CSF 标准注意力即可。

---

## 10. 验证与测试

| 检查项       | 预期结果                    |
| ------------ | --------------------------- |
| 模型构建     | 无报错                      |
| 前向输出形状 | P3'、P4'、P5 形状与原始一致 |
| 加载预训练   | 成功加载，Transferred 正常  |
| 恒等映射     | 加载后前向输出与基线 diff=0 |
| 单元测试     | 三个模块形状断言通过        |
| 冒烟训练     | 1 epoch 正常完成            |

---

## 11. 文件清单

| 文件                                           | 改动                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `ultralytics/nn/modules/block.py`              | 新增 `WaveletBranch`、`P1WaveletFusion`；复用 `CrossScaleFusion` |
| `ultralytics/nn/modules/__init__.py`           | 导出新模块                                                   |
| `ultralytics/nn/tasks.py`                      | `parse_model` 支持 `WaveletBranch` 多输出、`P1WaveletFusion`、`CrossScaleFusion` 多输入 |
| `ultralytics/cfg/models/v12/yolov12-wscf.yaml` | 新增网络定义                                                 |
| `train_wscf.py`                                | 训练入口                                                     |

---

## 12. 总结

完整设计包含三条融合路径：

1. **P1 + W1 拼接 → 1×1 卷积 → C1**：从最浅层注入 W1 低频；
2. **P3 ← W2 CSF**：P3 从 W2 中采集细节（num_blocks=1）；
3. **P4 ← W3 CSF**：P4 从 W3 中采集细节（num_blocks=1）。

三条路径分别覆盖 **P1（最浅层）、P3（小目标层）、P4（中目标层）**，从低到高形成多尺度小波信息注入。P5 不做小波融合，保持原样。

所有融合模块零初始化，训练开始退化为原始 YOLOv12，不破坏预训练权重。其他 Agent 可依据本规划书直接实现。