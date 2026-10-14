# -*- coding: utf-8 -*-
"""
验证官方预训练权重 yolov12n.pt 在 COCO val2017 上的精度，用于复现官网指标。

背景（本仓库 README 的 Main Results）：
    Turbo (默认)  YOLO12n: mAP50-95 = 40.4   (2.5M params / 6.0 GFLOPs)
    v1.0          YOLO12n: mAP50-95 = 40.6   (2.6M params / 6.5 GFLOPs)

本仓库的 yolov12n.pt 参数量为 2,553,904（约 2.5M），架构对应 **Turbo** 版本，
因此其官方参考值应为 **40.4**，而不是 v1.0 的 40.6。

预训练权重是在 COCO **train2017** 上训练、在 COCO **val2017** 上评测得到上述 mAP 的。
本脚本即在本地的 COCO val2017（5000 张）上复现该指标。

运行方式（在仓库根目录）:
    conda activate yolov12
    python val_yolov12n.py

若显存不足：把 DEVICE 改为 "cpu"（会慢很多），或调小 BATCH。
"""
from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 配置
# ============================================================================
ROOT = Path(__file__).resolve().parent
MODEL="/home/a10/xsy/yolov12/train/WSCF/B_stage_result/stageB_full_finetuning/weights/best.pt"
# MODEL = ROOT / "yolov12n.pt"
DATA = ROOT / "coco-csf.yaml"  # 指向本地 COCO val2017

IMGSZ = 640
BATCH = 16
DEVICE = 0
WORKERS = 0  # Windows 下多进程 DataLoader 易崩溃，置 0 更稳

# 官方参考值（mAP50-95）
OFFICIAL_TURBO = 0.404
OFFICIAL_V1 = 0.406


def main():
    model = YOLO(str(MODEL))
    metrics = model.val(
        data=str(DATA),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        plots=False,
        save_json=False,
        verbose=True,
    )

    box = metrics.box
    print("=" * 60)
    print("yolov12n.pt @ COCO val2017（5000 张，imgsz=640）")
    print("=" * 60)
    print(f"  mAP50-95 : {box.map:.4f}   (官方 Turbo={OFFICIAL_TURBO:.4f} / v1.0={OFFICIAL_V1:.4f})")
    print(f"  mAP50    : {box.map50:.4f}")
    print(f"  mAP75    : {box.map75:.4f}")
    print(f"  precision: {box.mp:.4f}")
    print(f"  recall   : {box.mr:.4f}")
    print(f"  每类 AP  : {len(box.maps)} 类，均值 {box.maps.mean():.4f}")
    print("=" * 60)
    print("说明：Ultralytics 原生指标不提供 AP_small/medium/large；")
    print("      如需分尺度 AP，可加 save_json=True 后用 pycocotools 做 COCO 官方评测。")


if __name__ == "__main__":
    main()
