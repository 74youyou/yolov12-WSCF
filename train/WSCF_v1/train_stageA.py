# -*- coding: utf-8 -*-
"""
YOLOv12-WSCF 两阶段训练 —— 阶段 A：冻结 YOLOv12 主干，训练小波/融合分支（+ Neck/Head）。

阶段 A 的目的：
    让新增模块（WaveletBranch、P1WaveletFusion、CrossScaleFusion）从
    "恒等 / 零初始化"出发完成初始化，学到有用的增量信号；主干权重保持冻结，
    避免较高学习率破坏预训练特征。阶段 B 再解冻全部参数做整体适配（train_stageB.py）。

冻结策略：
    只冻结原 YOLOv12 主干层（Conv/C3k2/A2C2f），新增模块与 Neck/Head 保持可训练。
    WSCF YAML 层号：
        0 Identity | 1 WaveletBranch | 2-4 Index(W1/W2/W3) | 5 Conv(P1) | 6 P1WaveletFusion
        7 Conv(P2) | 8 C3k2 | 9 Conv(P3) | 10 CSF(P3<-W2) | 11 C3k2 | 12 Conv(P4)
        13 CSF(P4<-W3) | 14 A2C2f | 15 Conv(P5) | 16 A2C2f | 17~28 PAN | 29 Detect

运行方式（在仓库根目录）:
    conda activate yolov12
    python train/WSCF/train_stageA.py
"""
from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 路径配置
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]/"yolov12"  # 仓库根目录

MODEL_YAML = ROOT / "ultralytics" / "cfg" / "models" / "v12" / "yolov12-wscf.yaml"
PRETRAINED = ROOT / "yolov12n.pt"
DATA_YAML = ROOT / "coco-csf.yaml"

OUTPUT_PROJECT = ROOT / "train" / "WSCF" / "A_stage_result"
NAME = "stageA_wavelet_init"

# 需要冻结的原 YOLOv12 主干层（来自预训练的 Conv/C3k2/A2C2f）
BACKBONE_LAYERS = [5, 7, 8, 9, 11, 12, 14, 15, 16]

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
    frozen = tuple(f"model.{i}." for i in BACKBONE_LAYERS)
    trainable = total = 0
    for name, p in model.model.named_parameters():
        total += p.numel()
        if not any(name.startswith(f) for f in frozen):
            trainable += p.numel()
    print(f"阶段A 可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    print(f"已冻结主干层: {BACKBONE_LAYERS}")


def main():
    last_pt = OUTPUT_PROJECT / NAME / "weights" / "last.pt"

    if last_pt.exists():
        print(f"检测到断点 {last_pt}，恢复阶段 A 训练...")
        model = YOLO(str(last_pt))
        model.train(resume=True, epochs=EPOCHS, device=DEVICE, plots=PLOTS)
        return

    print("开始阶段 A：冻结原 YOLOv12 主干，训练小波/融合分支 + Neck/Head ...")
    model = YOLO(str(MODEL_YAML))
    model.load(str(PRETRAINED))  # 预期 Transferred 739/811
    summarize_trainable(model)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        freeze=BACKBONE_LAYERS,  # 冻结原 YOLOv12 主干，其余（新增模块 + Neck/Head）可训练
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
        optimizer="SGD",
    )


if __name__ == "__main__":
    main()
