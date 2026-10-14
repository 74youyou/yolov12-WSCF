# YOLOv12 天气数据集准备与服务器迁移

本文按本仓库当前实际实现记录训练前准备：原始 VOC 数据布局、划分、XML 转 YOLO 标签、统一类别、生成文件、环境及服务器迁移。统计依据为 2026-10-14 核对的 `datasets/weather_yolo/summary.json`。准备流程供原始 YOLOv12 和 WSCF 共用，不会启动训练。

本地仓库为 `D:\yolov12\yolov12`。后文 Linux 示例假设服务器仓库为 `/srv/yolov12`；将它替换为自己的目录。命令中的项目内路径均相对于仓库根目录。

## 1. 准备了什么

已有 `scripts/build_weather_datasets.py` 将 ExDark、RTTS、VOC-Rain&Snow 的 VOC XML 标注转换为 YOLO 检测标签，生成六种实验视图：`exdark`、`rtts`、`rain`、`snow`、`rain_snow`、`merged`。各视图共用一份 `merged` 图片和标签，独立保存划分清单及配置文件。

已有 `scripts/verify_weather_datasets.py` 校验生成的图片、标签、数量及清单。本次增加的 `scripts/relocate_weather_datasets.py` 用于迁移后更新路径。原始 YOLOv12 的六组训练入口新增于 `train/YOLOv12_baseline/`；已有天气 WSCF 入口仍位于 `train/train_*_wscf.py`。

## 2. 原始数据的必要布局

构建器按下列路径读取已解压的 VOC 格式数据，目录名及嵌套层级需要匹配。脚本不下载、解压数据，也不将其他版本的原始 ExDark 格式自动转换为 VOC。

```text
datasets/
├── ExDark/ExDark/VOC2007/
│   ├── JPEGImages/<id>.jpg
│   ├── Annotations/<id>.xml
│   └── ImageSets/Main/{train,test}.txt
├── RTTS/RTTS/VOC2007/
│   ├── JPEGImages/<id>.jpg
│   ├── Annotations/<id>.xml
│   └── ImageSets/Main/{train,test}.txt
└── VOC-Rain&Snow/123/VOC2007/
    ├── rain/<id>.jpg
    ├── snow/<id>.jpg
    ├── Annotations/<id>.xml
    └── ImageSets/Main/{train,val,trainval,test}.txt
```

清单保存图片 ID（不带扩展名），按空白分隔读取，重复 ID 会报错。雨图和雪图使用相同 ID 和同一份 XML。构建器检查清单中的图片、XML、尺寸及类别；缺文件、未知类别、重复 ID 或同一来源划分重叠会停止构建。

如服务器原始数据放置于其他位置，可调整构建器顶部 `SOURCE` 三个路径；默认输出根为 `OUTPUT = ROOT / "datasets/weather_yolo"`。当前训练入口也固定读取这个输出根，保持默认输出布局可直接使用全部脚本。

## 3. 环境准备

仅构建数据需要 Python 3.10 或以上及 Pillow；路径迁移和现有完整数据校验也会导入构建器，所以同样需要 Pillow。新基线的 `--dry-run` 另外需要 PyYAML。使用现有 Python 环境时：

```bash
python -m pip install Pillow PyYAML
```

正式训练需要安装本地仓库及 GPU 对应的 PyTorch/torchvision。下面以仓库已有 Linux wheel 所对应的 Python 3.11、Torch 2.6.0/cu124 环境为示例；安装前确认服务器驱动和 GPU 能运行该 CUDA 构建：

```bash
cd /srv/yolov12
conda create -n yolov12 python=3.11 -y
conda activate yolov12
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

离线环境可使用仓库已有 `torch-2.6.0+cu124-cp311-cp311-linux_x86_64.whl`，但仍需提前准备对应 torchvision 和 `pyproject.toml` 所列依赖的 Linux wheel；仅拷贝当前 Torch wheel 不足以完成离线安装。使用本地源码可保留本仓库的自定义模块及兼容处理。

当前 `requirements_linux.txt` 写的是 Torch 2.2.2/torchvision 0.17.2，并引用仓库根目录目前没有的 FlashAttention 2.7.3 wheel，和根目录现有 Torch 2.6.0/FlashAttention 2.7.4 的 Linux wheel 属于不同组合。不要将这两套直接混装；上面的安装示例不读取该旧清单。`requirements_window.txt` 及 Windows wheel 也不能用于 Linux。

本仓库 `ultralytics/nn/modules/block.py` 允许 FlashAttention 不可用时使用 PyTorch 注意力计算，原始 `AAttn` 的非 Flash 分支使用矩阵乘法及 softmax。因此基本训练无需先安装 FlashAttention；若要复现 FlashAttention 性能，需另外核对 GPU、Torch/CUDA、Python、操作系统及 ABI 匹配的版本。新基线 `--dry-run` 不验证这些运行环境。

用下面的命令检查实际环境和源码位置：

```bash
python -c "import torch, ultralytics; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('source:', ultralytics.__file__)"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

第二条要求 CUDA 已可用。`source` 应指向服务器当前仓库中的 `ultralytics/__init__.py`。

## 4. 转换、划分与标签的具体实现

### 4.1 固定数据划分

`source_splits()` 对 ExDark 和 RTTS 保留原始 `test.txt`。原始训练集没有独立验证划分时，将训练 ID 按 `SHA256("weather-val-0:" + id)` 的摘要排序，取前 `max(1, round(训练数 × val_fraction))` 个 ID 作为验证集，默认 `val_fraction=0.1`。剩余 ID 进入训练集，清单输出仍保留原始 ID 顺序。它不依赖系统随机状态，相同输入和参数会得到相同划分。

雨和雪沿用原始 train/val/test，校验 `trainval = train ∪ val`，并要求 rain 与 snow 的划分完全相同。这样同一底图的雨版与雪版始终位于同一 split。合并时按来源拼接相应 split，不重新随机打散，不将测试样本加入训练。

### 4.2 统一 21 类

`NAMES` 固定为 VOC 20 类加 `cup`，所有 YAML 都使用相同顺序，标签 ID 必须保持一致：

| ID | 类别 | ID | 类别 | ID | 类别 |
| --- | --- | --- | --- | --- | --- |
| 0 | aeroplane | 7 | cat | 14 | person |
| 1 | bicycle | 8 | chair | 15 | pottedplant |
| 2 | bird | 9 | cow | 16 | sheep |
| 3 | boat | 10 | diningtable | 17 | sofa |
| 4 | bottle | 11 | dog | 18 | train |
| 5 | bus | 12 | horse | 19 | tvmonitor |
| 6 | car | 13 | motorbike | 20 | cup |

`parse_annotation()` 去除类别名前后空白并转为小写，使用 `ALIASES` 将 ExDark 的 `People` 映射为 `person`、`Table` 映射为 `diningtable`。未知类别直接报错。`Table` 到 `diningtable` 是当前实验的近似映射；如果修改类别定义，需要重新生成全部标签和 YAML。

ExDark 实际标注 12 类、RTTS 实际标注 5 类、雨雪标注 VOC 20 类，但输出统一为 21 类。没有标注的类别并不会自动补标，混合训练可能把图中的其他类别当作背景；单数据集结果也应结合其实际标注类别解释。

### 4.3 VOC XML 转 YOLO

转换读取 XML 的 `size` 和图片实际宽高。EXIF 方向为 6 或 8 时交换实际宽高，与现有加载逻辑的尺寸口径对应。XML 和实际尺寸不一致时按 `scale_x = 实际宽 / XML宽`、`scale_y = 实际高 / XML高` 缩放框，并在摘要中计数；这一步属于尺寸调整，不是通用的任意 EXIF 框旋转变换。

VOC 框按一基、包含端点处理：先将 `xmin/ymin` 减 1，`xmax/ymax` 保持，再缩放并裁剪到图片边界。只有 `xmin < xmax` 且 `ymin < ymax` 的框保留。`difficult=1` 的目标排除；完全落在图外或无面积的框跳过。

每行 YOLO 标签为：

```text
class_id x_center y_center width height
```

四个坐标均按实际图片尺寸归一化并保留六位小数：

```text
x_center = (xmin + xmax) / (2 * image_width)
y_center = (ymin + ymax) / (2 * image_height)
width    = (xmax - xmin) / image_width
height   = (ymax - ymin) / image_height
```

没有剩余有效目标的图片仍保留，并生成空标签文件。图片按原文件复制，没有生成新的雨雪效果或做离线增强。

### 4.4 共用图片与防止重名

`build()` 先检查并解析所有来源，再写出生成文件。合并图片与标签使用来源前缀：`exdark__<id>.jpg`、`rtts__<id>.jpg`、`rain__<id>.jpg`、`snow__<id>.jpg`，对应标签同名 `.txt`。前缀避免不同来源 ID 冲突。

默认 `materialize()` 使用 `shutil.copy2`，所有视图只共用这一次复制。`--mode hardlink` 使用硬链接，需要源与输出在同一文件系统。当前摘要记录的是 `copy`，直接重复构建需保持相同选项。

### 4.5 执行命令

```bash
python scripts/build_weather_datasets.py --mode copy --val-fraction 0.1
python scripts/verify_weather_datasets.py
```

构建和校验均根据脚本位置定位仓库，可以从其他目录用绝对脚本路径运行。同样输入及选项可重复运行，文本仅在变化时写入，已有图片只核对大小后跳过复制；这不是图片内容哈希验证。更换划分比例、复制方式或原始数据时，应先备份并移走旧 `weather_yolo` 再完整重建，构建器不自动清除旧文件。

`verify()` 校验合并图片数量、文件非空、标签存在、每行五字段、类别 ID 范围、归一化框及其边界；校验五组视图清单的数量、唯一性、图片位置、对应标签，并检查雨雪底图配对。它不逐张解码图片，也不跨独立来源做图像内容去重。

## 5. 生成了哪些文件

### 5.1 数据产物

```text
datasets/weather_yolo/
├── summary.json
├── merged/
│   ├── data.yaml
│   ├── images/{train,val,test}/<source>__<id>.jpg
│   └── labels/{train,val,test}/<source>__<id>.txt
├── exdark/{data.yaml,train.txt,val.txt,test.txt}
├── rtts/{data.yaml,train.txt,val.txt,test.txt}
├── rain/{data.yaml,train.txt,val.txt,test.txt}
├── snow/{data.yaml,train.txt,val.txt,test.txt}
└── rain_snow/{data.yaml,train.txt,val.txt,test.txt}
```

构建器生成 6 个 YAML、15 个图片清单、1 个摘要，以及 35,099 张合并图片和 35,099 个对应标签文件。图片全部位于 `merged/images`；五个视图没有各自复制的图片或标签目录，所以迁移单独视图时也必须携带它引用的合并图片及标签。最简单的是复制整个 `weather_yolo`。

当前数量如下，雨雪合并包含两种图片版本，而非只计一张底图：

| 数据集 | train | val | test | 总图片数 |
| --- | ---: | ---: | ---: | ---: |
| ExDark | 5,963 | 663 | 737 | 7,363 |
| RTTS | 3,500 | 389 | 433 | 4,322 |
| rain | 9,482 | 1,054 | 1,171 | 11,707 |
| snow | 9,482 | 1,054 | 1,171 | 11,707 |
| rain_snow | 18,964 | 2,108 | 2,342 | 23,414 |
| merged | 28,427 | 3,160 | 3,512 | 35,099 |

`summary.json` 的 `classes` 是该视图所有划分合计的目标框数，`splits` 是图片数。当前元数据记录：RTTS 跳过 1 个无效框，修正 4 个 XML/图片尺寸不一致记录；其他来源这两项均为 0。`invalid_boxes_skipped` 不包括按规则排除的 difficult 目标。

### 5.2 支撑脚本与本次新增文件

| 路径 | 来源与作用 |
| --- | --- |
| `scripts/build_weather_datasets.py` | 已有：划分、类别统一、转换、图片复制、配置与摘要生成 |
| `scripts/verify_weather_datasets.py` | 已有：生成数据的完整标签和清单校验 |
| `train/weather_common.py` | 已有：天气 WSCF v2 的两阶段共享训练逻辑 |
| `train/train_{exdark,rtts,rain,snow,rain_snow}_wscf.py` | 已有：5 个天气 WSCF 入口 |
| `train/train_merged_weather_wscf.py` | 已有：全部合并 WSCF 入口 |
| `train/WEATHER_WSCF.md` | 已有：天气 WSCF 操作说明 |
| `scripts/relocate_weather_datasets.py` | 本次新增：迁移后重写 6 YAML 和 15 清单中的绝对路径 |
| `train/YOLOv12_baseline/common.py` | 本次新增：原始 YOLOv12n 基线共享训练逻辑 |
| `train/YOLOv12_baseline/resume_trainer.py` | 本次新增：基线续训时更新数据和输出路径 |
| `train/YOLOv12_baseline/train.py` | 本次新增：统一选择 6 个数据集 |
| `train/YOLOv12_baseline/train_{exdark,rtts,rain,snow,rain_snow,merged}.py` | 本次新增：6 个专用原始模型入口 |
| `train/YOLOv12_baseline/README.md` | 本次新增：训练参数、输出、续训和对比说明 |
| `docs/YOLOv12天气数据集准备与服务器迁移.md` | 本次新增：本文 |
| `train/README.md` | 本次更新：新增目录及本文索引 |

本次没有重新生成整套数据或启动六组正式训练。未来训练会产生 checkpoint、`args.yaml`、`results.csv`、指标图和预览图等；Ultralytics 还可能生成 `.cache` 与运行时日志，它们不是转换器生成的源标签，迁移时可省略缓存并让服务器重建。

## 6. 迁移到服务器：两条完整路径

### 6.1 方案 A：携带原始数据，在服务器重新构建

复制本地仓库源码（包括 `ultralytics/`、`pyproject.toml`、`scripts/`、`train/`、`docs/`）、根目录 `yolov12n.pt`，以及第 2 节的三个完整 VOC 原始目录。COCO 数据和历史训练结果不属于这六组天气训练的必要输入。保证 `SOURCE` 布局匹配后在服务器执行：

```bash
cd /srv/yolov12
conda activate yolov12
python scripts/build_weather_datasets.py --mode copy --val-fraction 0.1
python scripts/verify_weather_datasets.py
python train/YOLOv12_baseline/train.py exdark --dry-run
```

构建器会写入服务器当前绝对路径，无需手动替换 Windows 路径。若也复制了已有 `weather_yolo`，输入及选项相同的重新构建会更新配置和清单路径并复用现有图片；改变输入或选项时请按第 4 节重新构建。

### 6.2 方案 B：携带转换产物，服务器只更新路径

复制同样的源码和 `yolov12n.pt`，以及整个 `datasets/weather_yolo/`。这条路线不需要原始 ExDark/RTTS/VOC-Rain&Snow 目录，适合减少重复图片传输。

当前 `data.yaml` 的 `path` 和五组视图的图片清单都保存类似 `D:\yolov12\yolov12\datasets\weather_yolo\...` 的 Windows 绝对路径。只修改 YAML 不足以修好图片清单，也不能原样放到 Linux 使用。在服务器执行：

```bash
cd /srv/yolov12
conda activate yolov12
python scripts/relocate_weather_datasets.py --dry-run
python scripts/relocate_weather_datasets.py
python scripts/verify_weather_datasets.py
python train/YOLOv12_baseline/train.py exdark --dry-run
```

迁移工具先验证合并图片数量、非空图片和标签存在性、每个清单的数量、唯一性及来源前缀，再用原图片文件名和原 split 构造新绝对路径。工具保留清单顺序，兼容旧 Windows/Unix 路径，仅在需要时改写 6 个 YAML 和 15 个 TXT；图片、标签、类别顺序、划分、`summary.json` 均保持不变。`--dry-run` 不写文件。迁移到新的服务器目录后再执行一次即可。

若目标目录为其他路径，例如 `/data/projects/yolov12`，无需修改迁移工具，脚本从自身位置推导 `ROOT`。迁移后应使用完整校验命令检查标签内容；迁移工具本身不承担这一检查。

### 6.3 训练断点迁移

新基线输出为 `train/YOLOv12_baseline/runs/<dataset>/<name>/`。需要恢复中断训练时，复制整个对应运行目录，以保留权重、训练设置和指标历史，更新数据路径后执行：

```bash
python train/YOLOv12_baseline/train.py exdark --name baseline_440 --resume --device 0 --batch 16 --workers 4
```

新基线的恢复训练器更新断点中原机器的数据和输出路径，保留优化器和训练进度。已完成并清理优化器的 checkpoint 不能作为中断续训使用。现有 WSCF 启动器没有使用这个新增的恢复训练器，迁移旧 WSCF 断点时还需要检查其已保存的训练与输出路径。

## 7. 原始模型训练命令

准备完成后，原始 YOLOv12n 六个专用入口如下，统一封装在新目录内：

```bash
python train/YOLOv12_baseline/train_exdark.py --device 0 --batch 16
python train/YOLOv12_baseline/train_rtts.py --device 0 --batch 16
python train/YOLOv12_baseline/train_rain.py --device 0 --batch 16
python train/YOLOv12_baseline/train_snow.py --device 0 --batch 16
python train/YOLOv12_baseline/train_rain_snow.py --device 0 --batch 16
python train/YOLOv12_baseline/train_merged.py --device 0 --batch 16
```

统一入口也可使用 `python train/YOLOv12_baseline/train.py <dataset>`。默认 n 尺度、COCO 预训练初始化、全模型单阶段微调 440 轮、640 分辨率、SGD、种子 0；训练器按数据的 21 类重建预测头。默认提前停止 patience 为 200，实际可能少于 440 轮。与 WSCF 的 40+400 轮对比时需记录阶段冻结、学习率重启及提前停止等差异，具体参数和输出说明见 [基线 README](../train/YOLOv12_baseline/README.md)。

如只检查迁移是否完整，给任意入口加 `--dry-run`；它检查全部划分的图片/标签位置和存在性、21 类顺序及权重文件，不实际加载权重或启动训练。正式比较使用各自 val 选择最佳权重，test 留给最终评估；六个入口默认都从同一 `yolov12n.pt` 独立开始。

## 8. 本次本地验证记录

2026-10-14 已完成以下检查：

- 现有 35,099 组图片/标签通过 `scripts/verify_weather_datasets.py` 完整校验。
- 6 个专用基线入口及统一入口通过 `--dry-run`。
- 实际加载原始 n 尺度模型及 COCO 权重，确认 21 类模型重建后没有 WSCF/小波模块；以 2 张 64×64 合成图片完成 CPU 前向、损失计算及反向传播。原始 80 类模型加载 739/739 个权重项，重建 21 类后加载 688/739 个兼容项。
- 迁移工具在临时目录中成功更新模拟 Windows 路径；验证预览无写入、图片/标签/摘要不变、原清单顺序保留及重复执行幂等。
- 恢复训练器的路径更新逻辑通过隔离检查，保留保存的总轮数；未执行完整断点训练恢复实验。

这些检查未启动六组正式训练，也未在 Linux 服务器上完成实际安装或长周期 GPU 训练；服务器迁移后仍按第 6 节运行环境、数据及入口检查。
