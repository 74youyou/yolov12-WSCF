
from ultralytics import YOLO
import os
# 第一阶段的最佳权重
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录
# 阶段 A 输出（需与 train_stageA.py 保持一致）
STAGE_B_PROJECT = ROOT / "train" / "WSCF_v2" / "B_stage_result"
STAGE_B_NAME = "stageB_full_finetuning"
STAGE_B_BEST = STAGE_B_PROJECT / STAGE_B_NAME / "weights" / "best.pt"
model = YOLO(STAGE_B_BEST)
DATA_YAML = Path(os.environ.get("WSCF_V2_DATA_YAML", str(ROOT / "coco-csf.yaml")))
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,             # 替换成第一阶段的 batch

    # 必须核对第一阶段实际使用的优化器
    optimizer="SGD",      # 仅在第一阶段确实使用 SGD 时保留

    lr0=0.0005,
    lrf=0.2,
    warmup_epochs=0.0,

    # 第二阶段保持较稳定的数据分布
    mosaic=0.0,
    mixup=0.0,

    patience=30,
    resume=False,
    project="runs/detect",
    name="wscf_v2_stage2",
)
