# -*- coding: utf-8 -*-
"""
YOLOv12-CSF 训练脚本 —— 阶段 B（解冻全部参数，全局微调）
支持断点恢复：如果输出目录中已存在 last.pt，则自动从断点继续训练。
"""
from pathlib import Path
from ultralytics import YOLO

# ============================================================================
# 路径配置
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent

# 阶段 A 输出目录（保留，未使用）
STAGE_A_PROJECT = ROOT / "train" / "A_stage_result"
STAGE_A_NAME = "stageA_onlyCSF"
STAGE_A_BEST = STAGE_A_PROJECT / STAGE_A_NAME / "weights" / "best.pt"

DATA_YAML = ROOT / "coco-csf.yaml"

# 阶段 B 输出根目录
OUTPUT_PROJECT = ROOT / "train" / "B_stage_result"

# 预训练权重：阶段 B 完整微调后的 last.pt（作为新微调的起点）
PRETRAINED_WEIGHTS = OUTPUT_PROJECT / "stageB_full_finetuning" / "weights" / "last.pt"

# 本次训练的输出名称
OUTPUT_NAME = "stageB_finetune_continue"
OUTPUT_DIR = OUTPUT_PROJECT / OUTPUT_NAME
LAST_PT = OUTPUT_DIR / "weights" / "last.pt"

# ============================================================================
# 训练超参数（全新微调计划）
# ============================================================================
EPOCHS = 110
IMGSZ = 640
BATCH = 16
WORKERS = 4
DEVICE = 0
LR0 = 0.0002          # 初始学习率 1e-4
LRF = 0.01            # 最终学习率 = 2e-4 * 0.01 = 2e-6
PLOTS = True
PATIENCE = 100
SAVE_PERIOD = 10
WARMUP_EPOCHS = 0
CLOSE_MOSAIC=40

# 是否自动恢复：若 OUTPUT_DIR 下存在 last.pt，则自动从中恢复
AUTO_RESUME = True


def main():
    # ---------- 断点恢复逻辑 ----------
    if AUTO_RESUME and LAST_PT.exists():
        print(f"检测到断点权重：{LAST_PT}")
        print("正在恢复训练（将从断点处继续，忽略下方超参数）...")
        model = YOLO(str(LAST_PT))
        model.train(resume=True,patience=PATIENCE)   # Ultralytics 会自动恢复 epoch、优化器、学习率调度等
        return

    # ---------- 无断点，开始新训练 ----------
    if not PRETRAINED_WEIGHTS.exists():
        raise FileNotFoundError(
            f"未找到预训练权重：{PRETRAINED_WEIGHTS}\n"
            f"请确认阶段 B 完整微调已完成，或修改 PRETRAINED_WEIGHTS 路径。"
        )

    print(f"加载预训练权重：{PRETRAINED_WEIGHTS}")
    print("开始新一轮微调训练（全新学习率计划，无预热）...")

    model = YOLO(str(PRETRAINED_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        lr0=LR0,
        lrf=LRF,
        optimizer="SGD",
        project=str(OUTPUT_PROJECT),
        name=OUTPUT_NAME,
        plots=PLOTS,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        warmup_epochs=WARMUP_EPOCHS,
        close_mosaic=CLOSE_MOSAIC
    )


if __name__ == "__main__":
    main()