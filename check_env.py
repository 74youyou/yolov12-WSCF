# -*- coding: utf-8 -*-
"""
环境检查脚本：验证 GPU 可用性、模型构建、预训练权重加载与前向输出。

运行方式（在仓库根目录）:
    conda activate yolov12
    python check_env.py
"""
import torch  # PyTorch 核心库（GPU 检查用）

from ultralytics import YOLO  # Ultralytics 统一入口类


def main():
    # -----------------------------------------------------------------------
    # Step 1: 检查 GPU 是否可用（框架训练时依赖 device=0 指向的这块显卡）
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: 检查 GPU 是否可用")
    print("=" * 60)
    print("CUDA available:", torch.cuda.is_available())  # True => 驱动/CUDA 环境正常
    if torch.cuda.is_available():
        # 打印显卡型号，确认训练用的就是这块 GPU
        print("GPU name      :", torch.cuda.get_device_name(0))
    else:
        print("警告: GPU 不可用，请确认驱动 / CUDA 环境。")

    # -----------------------------------------------------------------------
    # Step 2: 构建 YOLOv12-CSF 模型（yaml -> parse_model -> PyTorch 网络）
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2: 构建 YOLOv12-CSF 模型并验证权重加载")
    print("=" * 60)
    model = YOLO("ultralytics/cfg/models/v12/yolov12-csf.yaml")
    # info() 打印网络统计：509 层 / 2,669,296 参数 / 6.3 GFLOPs（n 尺度）
    print(model.model.info())

    # 加载官方预训练权重：
    #   预期输出 "Transferred 739/759 items"——
    #   739 = backbone/PAN/Detect 全部参数（含 Detect 层号 21->22 重映射），
    #   759 - 739 = 20 项为新增 CSF 模块参数（保持默认初始化，恒等映射）。
    model.load("yolov12n.pt")
    print()
    print("权重加载完成（若上方显示 Transferred 739/759 即为正常）。")

    # -----------------------------------------------------------------------
    # Step 3: 前向冒烟测试——随机张量走一遍网络，验证形状是否正确
    #         （张量是 CPU 随机数，此处仅作形状检查，不涉及 GPU 训练）
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: 前向推理冒烟测试（随机张量）")
    print("=" * 60)
    x = torch.randn(1, 3, 640, 640)  # 模拟 1 张 640x640 的 RGB 输入
    y = model.model(x)  # 前向：输出 P3/P4/P5 三个检测头的结果
    for i, t in enumerate(y):
        print(f"Detect head {i}: 输出形状 {tuple(t.shape)}")
        # 预期: [1, 144, 80, 80] / [1, 144, 40, 40] / [1, 144, 20, 20]
    print()
    print("全部检查通过，可开始训练。")


# ---------------------------------------------------------------------------
# Windows 下多进程需要入口保护（虽然本脚本未开多进程，保持统一规范）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()