# -*- coding: utf-8 -*-
"""
YOLOv12-WSCF v2 两阶段训练 —— 阶段 A：冻结主干（可选冻结 Head），初始化小波/融合模块。

阶段 A 的目的：
    让新增模块（含 SASG 的 WaveletBranch、P1WaveletFusion、CrossScaleFusion）从
    "恒等 / 零初始化"出发完成初始化，学到有用的增量信号。阶段 B 再解冻全部参数做整体适配。

冻结策略（FREEZE_HEAD 开关）：
    - FREEZE_HEAD = True（默认，推荐）：
        冻结原 YOLOv12 主干 **和** Neck/Head，只训练新增模块（约 10 万参数）。
        这样"唯一能降低 loss 的途径"就是让小波/融合模块学到有用的残差，
        最符合"阶段 A 只做初始化"的定位，也不会让 Head 过拟合到冻结特征。
    - FREEZE_HEAD = False：
        只冻结原主干，Neck/Head 也参与训练。适合希望 Head 提前适配的场景。

    WSCF YAML 层号：
        0 Identity | 1 WaveletBranch | 2-5 Index(W1/W2/W3/W4) | 6 Conv(P1) | 7 P1WaveletFusion
        8 Conv(P2) | 9 C3k2 | 10 Conv(P3) | 11 CSF(P3<-W2) | 12 C3k2 | 13 Conv(P4)
        14 CSF(P4<-W3) | 15 A2C2f | 16 Conv(P5) | 17 CSF(P5<-W4) | 18 A2C2f | 19~30 PAN | 31 Detect

运行方式（在仓库根目录）:
    conda activate yolov12
    python train/WSCF_v2/train.py a
"""
import os
from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 路径配置
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录

MODEL_YAML = ROOT / "ultralytics" / "cfg" / "models" / "v12" / "yolov12-wscf-v2.yaml"
PRETRAINED = ROOT / "yolov12n.pt"
DATA_YAML = Path(os.environ.get("WSCF_V2_DATA_YAML", str(ROOT / "coco-csf.yaml")))

OUTPUT_PROJECT = ROOT / "train" / "WSCF_v2" / "A_stage_result"
NAME = "stageA_wavelet_init"

# 新增模块所在层（小波支线 + P1 融合 + 三个 CSF）；阶段 A 只训练这些层
NEW_MODULE_LAYERS = [1, 7, 11, 14, 17]
# 原 YOLOv12 主干层（来自预训练的 Conv/C3k2/A2C2f）
BACKBONE_LAYERS = [6, 8, 9, 10, 12, 13, 15, 16, 18]
# 阶段 A 是否冻结 Neck/Head（True=只初始化新增模块；False=同时训练 Neck/Head）
FREEZE_HEAD = True

# 计算实际冻结层
if FREEZE_HEAD:
    # 冻结除新增模块以外的全部层（原主干 + Neck/Head）
    FREEZE_LAYERS = [i for i in range(32) if i not in NEW_MODULE_LAYERS]
else:
    FREEZE_LAYERS = BACKBONE_LAYERS

# ============================================================================
# 训练超参数（阶段 A）
# ============================================================================
EPOCHS = 40            # 仅做融合模块初始化，无需过长；建议 20~40
IMGSZ = 640
BATCH = 16
WORKERS = 8
DEVICE = 0
LR0 = 0.009            # 与阶段 B 一致的高学习率，让新增模块快速脱离恒等起点
LRF = 0.01
WARMUP_EPOCHS = 3.0
CLOSE_MOSAIC = 0
PATIENCE = 100
AMP = True
SEED = 0
PLOTS = True


def summarize_trainable(model):
    """打印阶段 A 的可训练/冻结参数占比，便于确认冻结策略生效。"""
    frozen = tuple(f"model.{i}." for i in FREEZE_LAYERS)
    trainable = total = 0
    for name, p in model.model.named_parameters():
        total += p.numel()
        if not any(name.startswith(f) for f in frozen):
            trainable += p.numel()
    print(f"阶段A 可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    print(f"FREEZE_HEAD={FREEZE_HEAD}，已冻结层: {FREEZE_LAYERS}")


def main():
    last_pt = OUTPUT_PROJECT / NAME / "weights" / "last.pt"

    if last_pt.exists():
        print(f"检测到断点 {last_pt}，恢复阶段 A 训练...")
        model = YOLO(str(last_pt))
        model.train(resume=True, epochs=EPOCHS, device=DEVICE, plots=PLOTS)
        return

    print("开始阶段 A：冻结主干%s，初始化小波/融合模块 ..." % ("和 Neck/Head" if FREEZE_HEAD else ""))
    model = YOLO(str(MODEL_YAML))
    model.load(str(PRETRAINED))  # 先运行 scripts/wscf_v2_audit.py 核对逐层映射
    summarize_trainable(model)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        freeze=FREEZE_LAYERS,
        lr0=LR0,
        lrf=LRF,
        warmup_epochs=WARMUP_EPOCHS,
        close_mosaic=CLOSE_MOSAIC,
        patience=PATIENCE,
        amp=AMP,
        seed=SEED,
        project=str(OUTPUT_PROJECT),
        name=NAME,
        plots=PLOTS,
        optimizer="SGD"
    )


if __name__ == "__main__":
    main()
