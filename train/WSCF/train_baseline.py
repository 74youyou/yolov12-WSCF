# -*- coding: utf-8 -*-
"""
原始 YOLOv12 基线训练脚本 —— 用于与 WSCF 做公平 A/B 对比。

为什么需要它：
    官方 yolov12n.pt 的 40.4 是"从头训练 600 epoch"的结果；而 WSCF 是"官方权重 + 两阶段微调"。
    直接把 WSCF 的精度和官方权重比是不公平的。要判断 WSCF 是否真的有效，必须让**基线也用相同的
    训练预算**（同样的预训练起点、总轮数、学习率计划），再用同一个验证脚本对比。

本脚本用与 WSCF 相同的总轮数（EPOCHS = 440 = 阶段A 40 + 阶段B 400）和 LR 计划训练原始 yolov12.yaml。

对比流程：
    1) python train/WSCF/train_baseline.py            # 训练基线（本脚本）
    2) python train/WSCF_v1/train.py a              # 训练 WSCF 阶段 A
    3) python train/WSCF_v1/train.py b              # 训练 WSCF 阶段 B
    4) python val_yolov12n.py train/WSCF/baseline_result/baseline_440/weights/best.pt
       python val_yolov12n.py train/WSCF/B_stage_result/stageB_full_finetuning/weights/best.pt
    5) 比较两者 mAP50-95；Δ = WSCF - 基线

运行方式（在仓库根目录）:
    conda activate yolov12
    python train/WSCF/train_baseline.py
"""
from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 路径配置
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]

MODEL_YAML = ROOT / "ultralytics" / "cfg" / "models" / "v12" / "yolov12.yaml"  # 原始基线
PRETRAINED = ROOT / "yolov12n.pt"
DATA_YAML = ROOT / "coco-csf.yaml"

OUTPUT_PROJECT = ROOT / "train" / "WSCF" / "baseline_result"
NAME = "baseline_440"

# ============================================================================
# 训练超参数（与 WSCF 两阶段的总预算/学习率对齐）
# ============================================================================
EPOCHS = 440            # = WSCF 阶段A(40) + 阶段B(400)
IMGSZ = 640
BATCH = 16
WORKERS = 8
DEVICE = 0

LR0 = 0.009
LRF = 0.01

OPTIMIZER = "SGD"
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
COS_LR = False
CLOSE_MOSAIC = 10
PATIENCE = 200
AMP = True
SEED = 0
PLOTS = True


def main():
    last_pt = OUTPUT_PROJECT / NAME / "weights" / "last.pt"

    if last_pt.exists():
        print(f"检测到基线断点 {last_pt}，恢复训练...")
        model = YOLO(str(last_pt))
        model.train(resume=True, epochs=EPOCHS, device=DEVICE, plots=PLOTS)
        return

    print("训练原始 YOLOv12 基线（与 WSCF 同预算）...")
    model = YOLO(str(MODEL_YAML))
    model.load(str(PRETRAINED))  # 预期 Transferred 739/739

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        lr0=LR0,
        lrf=LRF,
        optimizer=OPTIMIZER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        cos_lr=COS_LR,
        close_mosaic=CLOSE_MOSAIC,
        patience=PATIENCE,
        amp=AMP,
        seed=SEED,
        project=str(OUTPUT_PROJECT),
        name=NAME,
        plots=PLOTS,
    )


if __name__ == "__main__":
    main()
