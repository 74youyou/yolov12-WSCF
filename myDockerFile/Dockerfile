# 1. 基础镜像依旧用 Ubuntu 20.04（与 CUDA 11.8 兼容性最佳）
FROM ubuntu:20.04

# 2. 环境变量（保留了原先的优秀设置，新增 conda 路径）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    # conda 路径，方便后续直接使用 conda 命令
    PATH="/opt/conda/bin:$PATH"

# 3. 换源（保留，加速 apt）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 4. 安装基础系统依赖
#    不再需要 deadsnakes PPA、python3.11-dev、cuda-toolkit 等，
#    因为 conda 会自行管理 Python 和 CUDA 库
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc git zip unzip wget curl htop \
        libgl1 libglib2.0-0 libsm6 libusb-1.0-0 \
        gnupg g++ \
        # 为了 conda 安装顺利，有时需要 bzip2 等
        bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*


    # 5. 安装 Miniconda（轻量、无交互）
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    # 接受默认频道的服务条款
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    #tos 是 Terms of Service（服务条款） 的缩写。
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    # 可选：禁止自动激活 base 环境，避免干扰
    /opt/conda/bin/conda config --set auto_activate_base false && \
    /opt/conda/bin/conda init bash

# 6. 用 conda 创建开发环境 yolov12
#    这里先装上所有能用 conda 完美管理的“重器”：
#    python 3.11、supervision、pytorch 全家桶，以及 CUDA 11.8 运行时
RUN conda create -n yolov12 python=3.11 supervision \
        pytorch torchvision torchaudio pytorch-cuda=11.8 \
        -c pytorch -c nvidia -c conda-forge -y && \
    conda clean -a -y

# 7. 设置 conda 环境为后续 RUN 命令的默认 shell 环境
#    这样 RUN 后面的命令都会在 yolov12 环境中执行
SHELL ["conda", "run", "-n", "yolov12", "/bin/bash", "-c"]

# 8. 工作目录与文件拷贝（和之前逻辑一致）
WORKDIR /yolov12Project_workspace
COPY requirements.txt /var/tmp/
COPY flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl  /yolov12Project_workspace/
# 9. 在 yolov12 环境内用 pip 安装余下依赖
#    pip 会自动识别当前 conda 环境的 Python，不会跑偏
RUN pip install -r /var/tmp/requirements.txt && \
    pip install numpy && \
    # supervision 已在 conda 步骤安装，这里不用再装
    rm -f /var/tmp/requirements.txt /yolov12Project_workspace/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 10. 确保容器启动时自动激活 yolov12 环境
RUN echo "conda activate yolov12" >> ~/.bashrc

# 11. 元数据标签
LABEL maintainer="your-email@example.com" \
      description="YOLOv12 dev env with conda, CUDA 11.8, Python 3.11, PyTorch 2.2" \
      version="2.0" \
      environment="development"

# 12. 默认启动命令
CMD ["/bin/bash"]
