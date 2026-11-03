# 数据集说明

本仓库**已包含当前实验使用的训练集和测试集**，共 500 对、1,000 张 PNG：485 对训练图和 15 对最终测试图。数据来自原 RetinexNet 项目所介绍的 LOL 配对低照度数据；原始出处见 [原项目与数据页面](https://daooshee.github.io/BMVC2018website/) 和 [上游仓库](https://github.com/weichen582/RetinexNet)。本发布副本保留这些来源引用；根目录的 MIT 许可证对应上游 RetinexNet 源码。

## 已附目录

以下目录已放在项目根目录，也就是 `main.py` 所在目录。下载完整发布目录后，无需另行下载数据或建立这些目录：

```text
data/
├── our485/
│   ├── low/       485 张训练低照度 PNG
│   └── high/      485 张同名正常曝光参考 PNG
└── eval/
    ├── low/       15 张最终测试低照度 PNG
    └── high/      15 张同名正常曝光参考 PNG
```

发布副本将原数据中可能命名为 `eval15/` 的测试划分整理为 `data/eval/`，与当前代码的默认路径一致。程序直接读取 `data/our485/low/` 等目录，不需要再嵌套 `LOLdataset/`。

当前实验使用的 500 对图像均为 600×400 RGB 图像。每对 low/high 必须完整同名、同尺寸；不要将不同图像按序号强行配对。`our485` 仅用于训练，`eval` 用于两个阶段训练完成后的最终测试及误差计算，不参与当前最佳模型选择。训练采用每个阶段逐轮平均训练损失最低的规则。

## 默认运行方式

先按 [README.md](README.md) 配置 Python 3.7 和 `requirements.txt` 中的依赖，再在项目根目录运行 `python test_latest.py`，即可用已附最佳权重测试已附的 15 张低照度图并计算指标；运行 `python main.py` 则会使用已附训练集从头训练，并在训练结束后自动测试。数据和权重已经具备，运行仍需要正确安装 Python/TensorFlow 环境。

| 操作 | 使用的数据（均已包含） |
|---|---|
| 用已附权重进行默认 RetinexNet 测试 | `data/eval/low/` 输入及 `data/eval/high/` 指标参考图 |
| 从头运行默认完整训练和自动测试 | `data/our485/{low,high}/` 以及 `data/eval/{low,high}/` |
| 运行当前 LIME 演示或评估已有结果 | `data/eval/{low,high}/`，默认严格检查 15 对 RGB PNG |
| 阅读已有指标或绘制已有训练日志 | 不需要数据集 |

本仓库同时保留 `test_results/` 和 `LIME/results_lol/` 中已保存的增强结果。它们与 `data/eval/low/` 的低照度输入、`data/eval/high/` 的正常曝光参考图分别存放。

数据文件完整性可按根目录 [文件清单_SHA256.csv](文件清单_SHA256.csv) 核对；发布副本的检查结果见 [验证记录](验证记录.md)。

## 使用其他数据

RetinexNet 可使用 `--data_root` 指定另一处同结构的数据根目录，或用 `--test_dir`、`--reference_dir` 指定测试/参考目录。相对路径按项目目录解释。例如：

```bat
python test_latest.py --test_dir "D:\datasets\LOL\eval15\low" --reference_dir "D:\datasets\LOL\eval15\high"
```

`LIME/demo_LIME.m` 的路径单独在脚本开头配置，不读取 Python 命令行参数。更换 LIME 测试集时还需调整 `expectedImageCount`，并使用新的输出目录，具体见 [LIME/README.md](LIME/README.md)。

本发布目录只保留当前使用的一套 1,000 张 PNG，没有复制 `.DS_Store`、`LOLdataset.zip`、`LOLdataset/`、`data1/` 或旧版 `data/test/` 的 3 张 BMP。保留与排除情况见 [整理说明](整理说明.md)。
