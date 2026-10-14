# -*- coding: utf-8 -*-
"""
YOLOv12-WSCF 两阶段训练 —— 阶段 B：解冻全部参数，整体适配的全局微调。

阶段 B 的目的：
    在阶段 A 初始化好的小波/融合模块基础上，解冻全部参数（主干 + 新增模块 + Neck/Head），
    以 lr0=0.009、lrf=0.01 做整体微调，让主干与新增模块相互适配到最优。

运行方式（在仓库根目录）:
    conda activate yolov12
    python train/WSCF/train_stageB.py
"""
from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 路径配置
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]/"yolov12"  # 仓库根目录

# 阶段 A 输出（需与 train_stageA.py 保持一致）
STAGE_A_PROJECT = ROOT / "train" / "WSCF" / "A_stage_result"
STAGE_A_NAME = "stageA_wavelet_init"
STAGE_A_BEST = STAGE_A_PROJECT / STAGE_A_NAME / "weights" / "best.pt"

DATA_YAML = ROOT / "coco-csf.yaml"

OUTPUT_PROJECT = ROOT / "train" / "WSCF" / "B_stage_result"
NAME = "stageB_full_finetuning"

# ============================================================================
# 训练超参数（阶段 B：从阶段 A best.pt 开始，全量微调）
# ============================================================================
EPOCHS = 400           # 整体适配阶段，轮数可稍长；300~400 均可
IMGSZ = 640
BATCH = 16
WORKERS = 4
DEVICE = 0

LR0 = 0.009            # 与阶段 A 一致
LRF = 0.01             # 最终学习率 = LR0 * LRF = 9e-5

# 优化器（与论文/官方一致：SGD + 线性衰减 + 3 epoch warmup）
OPTIMIZER = "SGD"
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
COS_LR = False
CLOSE_MOSAIC = 0
PATIENCE = 200
AMP = True
SEED = 0
PLOTS = True


def main():
    last_pt = OUTPUT_PROJECT / NAME / "weights" / "last.pt"

    if last_pt.exists():
        print(f"检测到阶段 B 断点 {last_pt}，恢复训练...")
        model = YOLO(str(last_pt))
        model.train(resume=True, epochs=EPOCHS, device=DEVICE, plots=PLOTS,save_period=10)
        return

    if not STAGE_A_BEST.exists():
        raise FileNotFoundError(f"未找到阶段 A 权重：{STAGE_A_BEST}，请先完成阶段 A 训练。")

    print(f"从阶段 A 权重开始阶段 B 全量微调：{STAGE_A_BEST}")
    model = YOLO(str(STAGE_A_BEST))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        freeze=None,          # 解冻全部参数
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
