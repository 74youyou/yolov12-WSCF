# YOLOv12-WSCF v2 两阶段训练指南

v2 **继续使用原始 WSCF 的 A/B 两阶段方案**。不要把仓库根目录旧的 train_wscf.py（50 epoch 单阶段入口）当作正式对照方案。v2 的启动器 train/WSCF_v2/train.py 要求明确指定阶段，不带参数不会开始训练。

## 1. 服务器准备

在仓库根目录运行。使用项目依赖环境，并确认有以下文件：

- ultralytics/cfg/models/v12/yolov12-wscf-v2.yaml
- yolov12n.pt（官方预训练权重）
- COCO 训练/验证图像与 YOLO 格式标签
- 指向服务器数据集的 YAML

仓库的 coco-csf.yaml 含有本机 D 盘绝对路径。服务器上应使用一份路径正确的 COCO YAML，并设置环境变量：

    export WSCF_V2_DATA_YAML=/server/path/coco.yaml

这个变量只指定数据 YAML 文件；该 YAML 内的 path、train、val 也必须指向服务器上的真实数据。A/B 两阶段都会读取同一个变量。若不设置，则默认读取仓库根目录的 coco-csf.yaml。

先执行：

    python -m unittest tests.test_wscf_v2 -v
    python -m scripts.wscf_v2_audit --checkpoint yolov12n.pt
    python -m scripts.wscf_v2_validate --device cuda --steps 4 --pretrained

审计应报告 739 个预训练参数及缓冲区按显式层映射完全匹配、无缺失或值不匹配，严格恒等初始化的模型输出与原版一致。四步测试只证明完整检测损失和梯度链路可运行，不代替正式训练。服务器首次正式训练时，检查 AMP、loss 是否有限，以及 SASG 两层卷积和 CSF Q/K/V 在若干更新后的梯度。

## 2. 阶段 A：新增模块初始化

    python train/WSCF_v2/train.py a

脚本：train/WSCF_v2/train_stageA.py。它从 v2 YAML 建模，加载 yolov12n.pt，冻结除层 1、7、11、14、17 以外的层。可训练部分是 W1 GRN、W2-W4 SASG、P1WaveletFusion 和三个 CrossScaleFusion；主干、Neck、Detect Head 冻结。

参数沿用原 WSCF 阶段 A：40 epoch、640 输入、batch 16、lr0 0.009、lrf 0.01、warmup 3、seed 0、AMP 开启。输出目录：

    train/WSCF_v2/A_stage_result/stageA_wavelet_init/

中断后重新执行同一条命令，若 last.pt 存在，脚本会在阶段 A 内续训。

若使用旧版代码时在第 1 轮验证报 `Input type (float) and bias type (c10::Half) should be the same`，更新服务器上的 `ultralytics/nn/modules/block.py` 后重新运行上述阶段 A 命令。修复仅在 SASG 卷积入口对齐输入精度，不改变参数结构；若已有 `last.pt`，脚本会尝试续训。

## 3. 阶段 B：全模型微调

阶段 A 完成并生成 best.pt 后运行：

    python train/WSCF_v2/train.py b

脚本：train/WSCF_v2/train_stageB.py。从阶段 A 的 best.pt 启动**新的训练阶段**，解冻全部模型参数，并重新开始阶段 B 的学习率计划。这里不是对阶段 A 使用 resume=True。

参数沿用原 WSCF 阶段 B：400 epoch、640 输入、batch 16、SGD、lr0 0.009、lrf 0.01、momentum 0.937、weight decay 0.0005、warmup 3、seed 0、AMP 开启。输出目录：

    train/WSCF_v2/B_stage_result/stageB_full_finetuning/

阶段 B 中断后重跑同一命令，脚本会从阶段 B 的 last.pt 恢复优化器和训练状态。完整训练尚未由本地任务启动。

## 4. 评估与对照

用阶段 B 的 best.pt 和 last.pt 分别验证，并与原始 WSCF 的对应检查点比较。数据划分、预训练权重、输入尺寸、batch、随机种子、训练预算和验证参数应一致。mAP50-95、mAP50、Precision、Recall 与效率都要记录；多随机种子实验在首轮结构验证后进行。四步冒烟测试的 loss 下降不代表最终 mAP 会提高。

## 5. AMP 说明

本机 RTX 3060 Laptop GPU 上，原始 WSCF 与 v2 在 cuDNN 开启时的完整模型 FP16 反向都出现过显存分配失败；单独 SASG 的 AMP 测试通过。服务器上先按原方案保持 AMP=True 做短周期测试，不要直接依据本机现象改正式训练设置。如果服务器也失败，先比较原 WSCF 与 v2 的同条件结果，再统一决定后端或 AMP 设置，以维持实验可比性。
