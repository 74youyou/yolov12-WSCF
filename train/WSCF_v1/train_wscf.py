# -*- coding: utf-8 -*-
"""
YOLOv12-WSCF 训练入口脚本。

WSCF 在 YOLOv12 骨干上新增小波支线与三条融合路径：
    1) P1 + W1 拼接 -> 1x1 卷积（P1WaveletFusion，恒等初始化）
    2) P3 (Q) <- W2 (KV)（CrossScaleFusion, num_blocks=1, 零初始化）
    3) P4 (Q) <- W3 (KV)（CrossScaleFusion, num_blocks=1, 零初始化）
所有融合模块初始化时退化为恒等映射，可直接加载官方 yolov12n.pt 预训练权重。

运行方式（仓库根目录）:
    python train_wscf.py

行为:
    - runs/wscf/train/weights/last.pt 存在 -> 断点续训
    - 不存在 -> 从头训练
"""
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA = str(BASE_DIR / "coco-csf.yaml")
EPOCHS = 60
IMGSZ = 640
BATCH = 16
WORKERS = 4
DEVICE = 0
PRETRAINED = str(BASE_DIR / "yolov12n.pt")
PROJECT = str(BASE_DIR / "runs/wscf")
NAME = "train"

MODEL_YAML = str(BASE_DIR / "ultralytics/cfg/models/v12/yolov12-wscf.yaml")
LAST_WEIGHT = Path(PROJECT) / NAME / "weights" / "last.pt"


def main():
    # 断点续训：last.pt 存在则直接恢复
    if LAST_WEIGHT.exists():
        print(f"[WSCF] 检测到断点权重，继续训练: {LAST_WEIGHT}")
        model = YOLO(str(LAST_WEIGHT))
        model.train(resume=True, exist_ok=True)
        return

    # 从头训练
    print("[WSCF] 未检测到断点权重，从头开始训练")
    model = YOLO(MODEL_YAML)
    model.load(PRETRAINED)  # 融合模块零初始化，加载后等价于原始 YOLOv12
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        lr0=0.001,
        lrf=0.01,
        project=PROJECT,
        name=NAME,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    main()