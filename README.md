# RetinexNet 与 LIME 低光照图像增强实验

这是计算机视觉课程作业的实验代码，用于复现 RetinexNet，并与 LIME 比较低光照图像增强效果。RetinexNet 基于 [Chen Wei 等人的 TensorFlow 实现](https://github.com/weichen582/RetinexNet)，对应论文为 [Deep Retinex Decomposition for Low-Light Enhancement（BMVC 2018）](https://arxiv.org/abs/1808.04560)。论文和数据集入口见[原项目主页](https://daooshee.github.io/BMVC2018website/)。

实验使用 LOL 数据集中的 485 对图像训练，另外 15 对图像用于测试，共 1,000 张 PNG。运行前请确认项目根目录下的 `data/`、`LIME/` 和 `runs/` 材料齐全。其中，`data/` 存放数据，`LIME/` 存放 MATLAB 实现和结果，`runs/` 存放 RetinexNet 的训练记录、最佳权重和测试结果。数据组成与来源见 [DATASET.md](https://github.com/Mu-Shene/RetinexNet-LIME/blob/main/DATASET.md)。

材料齐全后，可以直接加载已有权重测试，也可以从头训练。

## 环境准备

RetinexNet 使用 TensorFlow 1.x 接口。本次实验在 Windows 64 位 CPU 环境下运行，主要依赖如下：

| 依赖 | 版本 |
| --- | --- |
| Python | 3.7.16 |
| TensorFlow | 1.15.5 |
| NumPy | 1.18.5 |
| Pillow | 9.5.0 |
| protobuf | 3.20.3 |
| Matplotlib | 3.5.3 |

在 Anaconda Prompt 中进入项目根目录，创建环境并安装依赖：

```bat
conda create -n retinex-tf1 python=3.7.16 pip
conda activate retinex-tf1
python -m pip install -r requirements.txt
```

[requirements.txt](https://github.com/Mu-Shene/RetinexNet-LIME/blob/main/requirements.txt) 按本地实验环境固定了依赖版本，尚未在全新环境中验证安装。Python 版本由 Conda 指定；TensorFlow 1.x 安装包也需要与 Python 版本和操作系统匹配。

默认使用 CPU，即 `--use_gpu=0`。配置好兼容 TensorFlow 1.x 的 CUDA 环境后，可改为 `--use_gpu=1`。LIME 使用 MATLAB，本机测试版本为 R2024a。

## 使用已有权重测试

本次实验使用的权重和记录按以下结构存放：

```text
runs/
├── latest.json
└── 2026-11-03/
    └── 20261103_110748_560831/
        ├── config.json
        ├── checkpoints/
        │   └── best/
        │       ├── Decom/
        │       └── Relight/
        ├── train.log
        ├── loss.csv
        ├── epochs.csv
        └── test_results/
```

默认测试读取 `data/eval/low/`，并用 `data/eval/high/` 中的同名图像计算误差。在项目根目录运行：

```bat
conda activate retinex-tf1
python test_latest.py
```

也可以使用：

```bat
python main.py --phase=test
```

程序会加载 `latest.json` 指向的实验权重。每次测试都会在对应实验目录下新建 `tests/YYYY-MM-DD/run-id/`，保存增强图、日志、配置和指标。原实验结果保存在 `test_results/`。

测试自己的图像时，指定输入和参考图目录：

```bat
python test_latest.py --test_dir "D:\images\low" --reference_dir "D:\images\high"
```

参考图需要与输入图同名、同尺寸。没有参考图时仍可生成增强图，对应指标会标记为不可用。

如果要始终使用本文对应的实验，可直接指定运行目录：

```bat
python test_latest.py --run_dir "runs/2026-11-03/20261103_110748_560831"
```

新的训练完成后，`latest.json` 会指向新实验。Decom 和 Relight 两个阶段的权重及配置需要一起保留，单独一个阶段的权重无法完成增强测试。

## 从头训练

训练数据放在 `data/our485/low/` 和 `data/our485/high/`，测试数据放在 `data/eval/low/` 和 `data/eval/high/`。在项目根目录运行：

```bat
python main.py
```

Decom 和 Relight 默认各训练 100 轮。每次运行都会从头训练并建立新的实验目录。训练只使用 `our485` 中的 485 对图像，`eval` 中的 15 对图像用于训练结束后的测试，不参与参数更新或最佳权重选择。

每个阶段按平均训练损失最低的轮次选择最佳权重。Relight 使用选出的最佳 Decom 权重训练，两个阶段结束后自动测试。

默认训练参数如下，也可以在命令行修改：

```bat
python main.py --epoch=100 --batch_size=16 --patch_size=48 --start_lr=0.001 --use_gpu=0
```

`--epoch` 表示每个阶段的轮数。数据目录通过 `--data_root` 指定，实验输出目录通过 `--output_root` 指定。使用自定义输出目录后，独立测试也需要传入同一个 `--output_root`。其他参数见：

```bat
python main.py --help
```

训练结果按北京时间保存在 `runs/YYYY-MM-DD/run-id/`，包括配置、训练日志、批次损失 `loss.csv`、逐轮损失 `epochs.csv`、最佳权重、阶段快照、训练样图和测试结果。用于发布的实验材料只保留最佳权重，不包含 `checkpoints/last/` 和训练样图。

## 评价指标与实验结果

MAE 和 MSE 根据保存后的增强 PNG 计算。程序将增强图和同名参考图转为 RGB，像素值除以 255，先计算每张图的误差，再对 15 张图等权平均。RetinexNet 的增强图以 `_S` 结尾，指标写入 `metrics.csv` 和 `metrics_summary.json`。

JSON 中还记录了按 RGB 像素元素数加权的结果。缺少参考图或尺寸不匹配的样本会标记为不可评估，并从均值中排除。

本次实验中，最佳 Decom 权重来自第 98 轮、2,940 步，最佳 Relight 权重来自第 91 轮、2,730 步。结果如下，MAE 和 MSE 均为越低越好：

| 方法 | 有效测试图 | MAE | MSE |
| --- | ---: | ---: | ---: |
| RetinexNet | 15/15 | 0.12286475 | 0.02389174 |
| LIME | 15/15 | 0.12054261 | 0.02428075 |

RetinexNet 的平均 MSE 略低，平均 MAE 则略高，两项指标的排序不同。

上述数值最初由实验保存的 PNG 重新计算。随后在本机加载对应权重重新测试 RetinexNet，并按下文参数重新运行 LIME，两种方法生成的图像均与原保存结果逐像素一致。本次核验使用了本地补充的数据、权重和 LIME 代码。

训练代码没有固定 Python、NumPy 和 TensorFlow 的随机种子，因此从头训练时，权重和指标可能与本次实验不同。

## 绘制损失曲线

在 `retinex-tf1` 环境中运行：

```bat
python plot_loss.py "runs/2026-11-03/20261103_110748_560831/train.log"
```

默认绘制每轮的平均批次损失，并在日志旁保存 `loss_curve.png`。如果要查看每个批次的变化，运行：

```bat
python plot_loss.py "runs/2026-11-03/20261103_110748_560831/train.log" --x-axis=batch
```

训练结束后，也可以直接运行 `python plot_loss.py`，读取最近一次完整实验的日志。绘图只读取记录，不需要加载模型。

## LIME 对照实验

LIME 的 MATLAB 入口是 `LIME/demo_LIME.m`，核心函数是 `LIME/lime_enhance.m`。该实现无需训练，也没有使用 BM3D。

在 MATLAB 中进入项目根目录后运行：

```matlab
run(fullfile('LIME', 'demo_LIME.m'));
```

默认情况下，脚本读取 `LIME/results_lol/` 中已有的 15 张增强图并计算指标。要重新生成结果，需要将 `demo_LIME.m` 中的输出目录改为新的目录，并关闭结果复用：

```matlab
outputDir = fullfile(demoDir, 'results_new');
evaluateExistingResults = false;
```

当前参数为 `alpha=0.15`、`gamma=0.8`、`sigma=2`、`epsilon=0.001`。这组参数已在本机 MATLAB R2024a 中重新运行，生成图像与已有结果逐像素一致。原结果没有单独保存生成参数和时间记录，后续实验建议一并保留。

LIME 为课程实验中的独立实现，尚未与作者源码逐项核对。具体实现和使用说明见 [LIME/README.md](https://github.com/Mu-Shene/RetinexNet-LIME/blob/main/LIME/README.md)。

## 数据与引用

数据下载：[百度网盘](https://pan.baidu.com/s/1DaIjwzIOtel1990njctn2w?pwd=ia2w)，提取码 `ia2w`。数据组成和原始来源见 [DATASET.md](https://github.com/Mu-Shene/RetinexNet-LIME/blob/main/DATASET.md)。

RetinexNet 上游代码的 MIT 许可证及 Chen Wei 版权声明保留在 [LICENSE](https://github.com/Mu-Shene/RetinexNet-LIME/blob/main/LICENSE) 中。原项目展示图用于介绍方法，与本次训练结果分开标注。

RetinexNet 论文引用：

```bibtex
@inproceedings{Chen2018Retinex,
  title={Deep Retinex Decomposition for Low-Light Enhancement},
  author={Chen Wei and Wenjing Wang and Wenhan Yang and Jiaying Liu},
  booktitle={British Machine Vision Conference},
  year={2018},
  organization={British Machine Vision Association}
}
```
