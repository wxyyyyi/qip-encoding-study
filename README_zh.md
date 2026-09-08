# QIP Encoding Study 可复现发布包

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22658398.svg)](https://doi.org/10.5281/zenodo.22658398)

这是论文《Quantum Data Encoding Trade-offs in Parallel Hybrid Image Classification》的公开可复现仓库。它包含规范化训练入口、处理后结果、主要图以及固定数据划分协议。

审核后的发布包已公开托管在 [wxyyyyi/qip-encoding-study](https://github.com/wxyyyyi/qip-encoding-study)。固定的 `v1.0.0` release 已永久归档到 Zenodo：[https://doi.org/10.5281/zenodo.22658398](https://doi.org/10.5281/zenodo.22658398)。仓库原创内容采用 MIT License。未经统一加载验证的旧 checkpoint 不进入首个公开版本。

## 可复现范围

本包支持两类工作：

1. **审计论文证据。** `results/` 中的处理后表格和图对应稿件使用的结果。运行 `python code/validation/validate_release.py`，可核对固定预算主结果、Python 语法、原始数据排除、常见敏感信息和 SHA256 清单。
2. **重新运行规范模型。** `continuous_encoding_experiment.py` 覆盖 Amplitude 和 Angle，`iqp_all_experiment.py` 覆盖 IQP-All 与 IQP-NN，`basis_ste_experiment.py` 覆盖 Basis-STE；它们采用 `configs/protocol.md` 所列数据划分、优化器、epoch、seed、四分支结构和验证集选 checkpoint 策略。

`results/` 中的处理后表格是面向论文的发布记录。依赖未收录 run-level 归档的历史报告脚本保存在本地私有归档中，不再作为“可独立运行代码”放入公开候选包；边界详见 `results/RESULTS_MANIFEST.md` 和 `CODE_AUDIT.md`。

## 目录说明

- `code/experiments/continuous_encoding_experiment.py`：Amplitude/Angle 规范训练入口；
- `code/experiments/iqp_all_experiment.py`：IQP-All/IQP-NN 规范训练入口；
- `code/experiments/basis_ste_experiment.py`：Basis-STE 训练与诊断；
- `code/experiments/classical_bottleneck_baseline.py`：验证集选模的经典瓶颈对照；
- `code/analysis/quantum_resource_analysis.py`：可独立运行的量子资源统计；
- `code/validation/`：发布包自检、校验和生成及 Amplitude 输入梯度检查；
- `CODE_AUDIT.md`：公开代码审计、修正和验收记录；
- `LICENSE`：仓库原创内容适用的 MIT 许可条款；
- `results/`：稿件所用的处理后 CSV、JSON、Markdown 和结果图；
- `figures/`：主要论文图的副本；
- `checkpoints/README.md`：说明为何公开候选包未包含旧权重；
- `data/README.md`：数据来源和确定性切分规则，不含原始数据。

## 快速自检

```bash
python code/validation/validate_release.py
python -m unittest discover -s code/validation -p "test_*.py" -v
python code/validation/generate_sha256_manifest.py
python code/validation/validate_release.py
```

第一条命令只使用 Python 标准库，不安装科研依赖也能运行。为保证 Windows 与 Linux 结果一致，SHA256 对文本文件先将 CRLF 换行规范为 LF，二进制文件仍逐字节计算。

## Smoke test

先按 `data/README.md` 下载 MNIST 和 Fashion-MNIST，然后运行：

```bash
python code/experiments/continuous_encoding_experiment.py --encodings amplitude angle --datasets mnist --seeds 42 --smoke-test --data-root path/to/data
python code/experiments/iqp_all_experiment.py --datasets mnist --seeds 42 --smoke-test --data-root path/to/data
python code/experiments/basis_ste_experiment.py --datasets mnist --seeds 42 --smoke-test --data-root path/to/data
```

输出保存在被 Git 忽略的 `runs/` 下；smoke 和 full 使用不同目录，IQP 两种拓扑也使用不同目录。Smoke test 只证明程序能执行，不代表已经数值复现完整预算结果。

## 完整固定预算运行

以下命令默认运行 MNIST 10 epochs、Fashion-MNIST 25 epochs 和 seeds `42 123 456`：

```bash
python code/experiments/continuous_encoding_experiment.py --data-root path/to/data
python code/experiments/iqp_all_experiment.py --data-root path/to/data
python code/experiments/basis_ste_experiment.py --data-root path/to/data
python code/experiments/classical_bottleneck_baseline.py --data-root path/to/data
```

自定义评估协议使用官方训练集 `0:10000`、官方 test 的 `0:2000` 作为 validation、`2000:10000` 作为独立 final holdout；它不是标准的完整 test benchmark。

这些脚本是依据固定协议和原始实现整理出的规范化重跑入口。由于硬件、依赖构建、批处理方式和历史随机数消耗可能不同，本仓库不承诺在所有环境中逐浮点数恢复论文数值。

## 发布与引用

引用或复现本研究时，请使用固定的 `v1.0.0` Zenodo 记录：[https://doi.org/10.5281/zenodo.22658398](https://doi.org/10.5281/zenodo.22658398)。该 DOI 对应论文使用的精确 GitHub release；后续变更应发布新版本，不覆盖已经归档的记录。

仓库原创内容采用 MIT License。MNIST、Fashion-MNIST 及所有第三方依赖仍遵循各自的许可和引用要求；本仓库不再分发其原始数据或源代码。
