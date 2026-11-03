# LIME Baseline

本目录是课程实验使用的传统低照度增强 Baseline，与 RetinexNet 的 TensorFlow 网络独立，不需要训练和网络权重。

| 文件或目录 | 用途 |
|---|---|
| `demo_LIME.m` | 批量读取测试图及参考图，增强或复用已有结果，在终端打印逐图及平均 MAE/MSE |
| `lime_enhance.m` | 照明估计、加权平滑与稀疏线性求解、gamma 校正及图像增强 |
| `results_lol/` | 已保存的 15 张测试增强 PNG，文件名与 `data/eval/low/` 对应 |

## 运行现有结果的评估

发布目录已包含 `data/eval/low/` 和 `data/eval/high/` 的 15 对测试图，以及本目录的 15 张增强结果，无需另行准备数据。组成与原始出处见 [数据说明](../DATASET.md)。脚本默认检查 15 对同名、同尺寸的单帧 RGB PNG；请保留完整数据目录。

在 MATLAB 中打开并运行 `demo_LIME.m`，或在项目根目录执行：

```matlab
run(fullfile('LIME', 'demo_LIME.m'));
```

脚本通过自身位置定位项目，默认路径为：

```text
输入：../data/eval/low/
参考：../data/eval/high/
输出：results_lol/
```

`evaluateExistingResults=true` 且现有 15 张输出完整时，默认**只读取这些 PNG 进行评估**，不重新执行增强，也不应用本次脚本设置的增强参数。指标使用保存图和参考图的归一化 RGB 差异，最终对各图的 MAE/MSE 等权平均；当前只打印到 MATLAB 终端，不自动保存独立指标 CSV 或 JSON。

## 重新生成增强结果

在 `demo_LIME.m` 开头将 `outputDir` 改为一个新的、未存放结果的目录，例如：

```matlab
outputDir = fullfile(demoDir, 'results_new');
evaluateExistingResults = false;
```

随后运行脚本。不要仅关闭复用选项后仍指向已有 `results_lol/`：脚本会报错阻止覆盖。已有部分输出时也会报错，避免混用不同运行的结果。更换测试集还需调整 `inputDir`、`referenceDir`、`expectedImageCount`。

当前脚本中的生成参数为：

| 参数 | 当前值 |
|---|---:|
| `alpha` | 0.15 |
| `gamma` | 0.8 |
| `sigma` | 2 |
| `epsilon` | 0.001 |

**现有 `results_lol/` 缺少独立的生成参数和时间记录，因此不能证明上述当前参数就是这 15 张现有图的生成参数。** 重新生成时请同时保存参数、MATLAB 版本和终端输出，使用新的结果目录。

## 实现范围与来源记录

`lime_enhance.m` 从 RGB 各通道最大值获得初始照明，使用梯度权重、稀疏线性方程和 gamma 校正增强图像。Gaussian 平滑由内部 `conv2` 实现，当前代码**没有调用 BM3D**，也不依赖 RetinexNet 权重。不要将这些结果标注为含 BM3D 的处理流程。

运行需要支持脚本内局部函数、`isfolder` 和 `isfile` 的 MATLAB。发布整理时未执行 MATLAB，尚未确认该实现的具体运行版本；Python 的 `requirements.txt` 不负责 MATLAB 环境。

本目录沿用课程材料中的 LIME 命名和实现。现有文件没有附独立上游代码来源或引用记录；发布整理未对照原论文或作者源码验证一致性，因此不宣称这是作者官方实现或完整复现其所有步骤。根目录的上游 RetinexNet MIT 许可证对应其原有源码，不能单独说明这些新增文件的来源。
