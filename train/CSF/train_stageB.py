# -*- coding: utf-8 -*-
"""
YOLOv12-CSF 训练脚本 —— 阶段 B（解冻全部参数，全局微调）
支持断点续训：若检测到阶段 B 上次训练的 last.pt，则自动恢复；否则从阶段 A best.pt 开始。
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================================
# 路径配置
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent

# 阶段 A 输出目录（需与 train_stageA.py 中的设置一致）
STAGE_A_PROJECT = ROOT / "train" / "A_stage_result"
STAGE_A_NAME = "stageA_onlyCSF"
STAGE_A_BEST = STAGE_A_PROJECT / STAGE_A_NAME / "weights" / "best.pt"

DATA_YAML = ROOT / "coco-csf.yaml"

# 阶段 B 输出目录
OUTPUT_PROJECT = ROOT / "train" / "B_stage_result"
NAME = "stageB_full_finetuning"

# ============================================================================
# 训练超参数（阶段 B：从阶段 A best.pt 开始，全局微调）
# ============================================================================
EPOCHS = 600        # 阶段 B 训练轮数，可根据需要调整
IMGSZ = 640
BATCH = 16
WORKERS = 4
DEVICE = 0

# 微调学习率：从阶段 A 的 best.pt 继续，解冻全部参数。
# 论文从头训练用 lr0=0.01，但这里是从阶段 A 微调，建议用更小的值。
LR0 = 0.009        # 推荐 0.001；若欠拟合可试 0.009
LRF = 0.01         # 最终学习率 = LR0 * LRF = 1e-5

# 论文中的优化器设置（SGD + 线性衰减 + 3 epoch warmup）
OPTIMIZER = "SGD"
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
COS_LR = False     # 论文使用线性衰减，不是余弦
CLOSE_MOSAIC = 10
PATIENCE = 200
AMP = True
SEED = 0

PLOTS = True


def main():
    # 阶段 B 断点文件路径
    last_pt = OUTPUT_PROJECT / NAME / "weights" / "last.pt"

    if last_pt.exists():
        print(f"检测到阶段 B 上次训练 checkpoint：{last_pt}")
        print("将恢复阶段 B 训练...")
        model = YOLO(str(last_pt))
        # 恢复训练：只传 resume=True 和必要参数，沿用 checkpoint 中的配置
        # 注意：resume=True 不会应用下面新设置的 LR0 等超参数
        model.train(
            resume=True,
            epochs=EPOCHS,
            device=DEVICE,
            plots=PLOTS,
        )
    else:
        print("未检测到阶段 B checkpoint，将从阶段 A best.pt 开始训练...")
        if not STAGE_A_BEST.exists():
            raise FileNotFoundError(f"未找到阶段 A 权重：{STAGE_A_BEST}，请先完成阶段 A 训练。")

        # 加载阶段 A 最优权重（包含模型结构、权重等）
        model = YOLO(str(STAGE_A_BEST))

        # 全模型微调：freeze=None 解冻所有层
        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            workers=WORKERS,
            device=DEVICE,
            freeze=None,
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