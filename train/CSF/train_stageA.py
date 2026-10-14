# -*- coding: utf-8 -*-
"""
YOLOv12-CSF 训练脚本 —— 阶段 A（只训练 CrossScaleFusion 模块）
支持断点续训：若检测到上次训练保存的 last.pt，则自动恢复训练。
"""

from pathlib import Path
from ultralytics import YOLO
import torch.nn as nn

# ============================================================================
# 路径配置
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent

MODEL_YAML = ROOT / "ultralytics" / "cfg" / "models" / "v12" / "yolov12-csf.yaml"
PRETRAINED = ROOT / "yolov12n.pt"
DATA_YAML = ROOT / "coco-csf.yaml"

# 训练输出目录（用于检查 checkpoint 和恢复训练）
OUTPUT_PROJECT = ROOT / "train" / "A_stage_result"
NAME = "stageA_onlyCSF"

# ============================================================================
# 训练超参数
# ============================================================================
EPOCHS  = 40
IMGSZ   = 640
BATCH   = 16
WORKERS = 8
DEVICE  = 0
LR0     = 0.005
LRF     = 0.01
PLOTS   = True


def freeze_all_except_csf(model):
    """
    冻结模型中除 CrossScaleFusion 模块之外的所有参数。
    """
    for name, param in model.model.named_parameters():
        # 如果参数所属模块包含 'CrossScaleFusion'，则保持可训练，否则冻结
        if 'CrossScaleFusion' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    # 打印可训练参数数量，便于验证
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"可训练参数: {trainable:,} / 总参数: {total:,} ({100*trainable/total:.2f}%)")


def main():
    # 上次训练可能保存的 checkpoint 路径
    last_pt = OUTPUT_PROJECT / NAME / "weights" / "last.pt"

    if last_pt.exists():
        print(f"检测到上次训练 checkpoint：{last_pt}")
        print("将从断点恢复训练...")
        model = YOLO(str(last_pt))
        # 恢复训练时也重新应用冻结策略（确保只训练 CSF）
        freeze_all_except_csf(model)
        model.train(
            resume=True,
            epochs=EPOCHS,
            device=DEVICE,
            plots=PLOTS,
        )
    else:
        print("未检测到 checkpoint，开始全新训练...")
        # 从 YAML 构建模型
        model = YOLO(str(MODEL_YAML))
        # 加载官方预训练权重（跳过新增 CSF 模块）
        model.load(str(PRETRAINED))
        # 冻结所有非 CSF 模块
        freeze_all_except_csf(model)
        # 启动训练
        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            workers=WORKERS,
            device=DEVICE,
            lr0=LR0,
            lrf=LRF,
            project=str(OUTPUT_PROJECT),
            name=NAME,
            plots=PLOTS,
            optimizer="SGD"
        )


if __name__ == "__main__":
    main()