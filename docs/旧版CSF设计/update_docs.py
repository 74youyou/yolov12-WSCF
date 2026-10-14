#!/usr/bin/env python3
import sys

with open('D:\\yolov12\\yolov12\\docs\\YOLOv12-CSF模型介绍书.md', 'r', encoding='utf-8') as f:
    content = f.read()

old = """### 3.2 前向计算流程

输入：`x = [shallow, deep]`，其中 `shallow` 为 P3（如 `(B, 256, 80, 80)`），`deep` 为 P5（如 `(B, 1024, 20, 20)`）。

```
P3 (B,C3,80,80) ──▼ 下采样分支(2×stride-2 Conv) ──► (B,embed,20,20)
                                                          │ flatten
                                                          ▼
                                                    (B, 400, embed) ── K/V
P5 (B,C5,20,20) ──► q_proj ──► (B, 400, embed) ──► Q
                                                          │
                       多头交叉注意力: attn = softmax(QKᵀ/√d) · V
                                                          │
                                                          ▼
                                                    (B, 400, embed)
                                                          │ out_proj（零初始化）
                                                          ▼
                                                    (B, C5, 20, 20)
                                                          │
                                                          │
输出 = P5 + out   （残差相加，恒等映射性质）
```

分步说明：

1. **下采样**：P3 经 `num_blocks=2` 个 stride=2 卷积（通道逐渐过渡到 `embed_dim`），得到与 P5 同分辨率的特征（80×80 → 40×40 → 20×20，即 4 倍下采样）；
   - 若因非整除尺寸导致分辨率不一致，自动走 `F.interpolate`（bilinear）对齐（兜底逻辑）；
2. **Q/K/V 生成**：深层 P5 展平后经 `q_proj` 生成 Query；下采样后的浅层特征展平后经 `k_proj`/`v_proj` 生成 Key/Value；
3. **多头交叉注意力**：`attn = softmax(QKᵀ / √head_dim)`，`head_dim = embed_dim / num_heads`；支持可选的注意力 dropout（默认 0.0）；每个 Query 位置（P5 的每个空间位置）从整个浅层特征图（400 个位置）聚合上下文；
4. **输出投影**：注意力输出经 `out_proj` 投影回 P5 的通道数（`c1_deep`）；
5. **残差融合**：`return deep + out`."""
)

new = """### 3.2 前向计算流程

输入：`x = [shallow, deep]`，其中 `shallow` 为 P3（如 `(B, 256, 80, 80)`），`deep` 为 P5（如 `(B, 1024, 20, 20)`）。

```
P3 (B,C3,80,80) ──► 独立 K path (深度可分离卷积) ──► (B,embed,20,20)
                       │                                    │ flatten
                       ▼                                    ▼
                                              (B, 400, embed) ── K
P3 (B,C3,80,80) ──► 独立 V path (深度可分离卷积) ──► (B,embed,20,20)
                       │                                    │ flatten
                       ▼                                    ▼
                                              (B, 400, embed) ── V
P5 (B,C5,20,20) ──► q_proj ──► (B, 400, embed) ──► Q
                                                          │
                       多头交叉注意力: attn = softmax(QKᵀ/√d) · V
                                                          │
                                                          ▼
                                                    (B, 400, embed)
                                                          │ out_proj（零初始化）
                                                          ▼
                                                    (B, C5, 20, 20)
                                                          │
输出 = P5 + out   （残差相加，恒等映射性质）
```

分步说明：

1. **K/V 生成路径（两条独立路径）**：
   - **Key path**：原始浅层特征（P3）经 `num_blocks` 个深度可分离卷积（每个卷积包含 depthwise + pointwise，stride=2）逐步下采样至深层分辨率（80×80 → 40×40 → 20×20，即 4× 下采样），输出通道数为 `embed_dim`；
   - **Value path**：完全独立的相同架构，也从原始浅层特征出发，生成独立的 Value 流；
   - 两条路径权重不共享，分别学习不同的 K 和 V 投影；
2. **Q/K/V 生成**：
   - 深层 P5 展平后经 `q_proj` 生成 Query；（B, 400, embed）
   - Key path 输出展平后经 `k_proj` 生成 Key；（B, 400, embed）
   - Value path 输出展平后经 `v_proj` 生成 Value；（B, 400, embed）
3. **多头交叉注意力**：`attn = softmax(QKᵀ / √head_dim)`，`head_dim = embed_dim / num_heads`；支持可选的注意力 dropout（默认 0.0）；每个 Query 位置（P5 的每个空间位置）从整个浅层特征图（400 个位置）聚合上下文；
4. **输出投影**：注意力输出经 `out_proj` 投影回 P5 的通道数（`c1_deep`）；
5. **残差融合**：`return deep + out`."""

if old in content:
    content = content.replace(old, new)
    with open('D:\\yolov12\\yolov12\\docs\\YOLOv12-CSF模型介绍书.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replacement successful!")
else:
    print("Old string not found!")
    # Debug: find the section
    idx = content.find('### 3.2 前向计算流程')
    if idx >= 0:
        print("Found section at index:", idx)
        print("Context:", repr(content[idx:idx+300]))