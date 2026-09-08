# QIP Encoding Study 可复现发布包

这是论文《Quantum Data Encoding Trade-offs in Parallel Hybrid Image Classification》的本地发布候选包。它包含规范化训练入口、处理后结果、主要图以及固定数据划分协议。

该包已托管在私有 GitHub 仓库 [wxyyyyi/qip-encoding-study](https://github.com/wxyyyyi/qip-encoding-study)，用于完成最终发布审核；目前还不是公开或已归档的正式记录。归档 DOI 和软件许可仍须由作者在投稿前补全。未经统一加载验证的旧 checkpoint 不进入公开候选包。

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

第一条命令只使用 Python 标准库，不安装科研依赖也能运行。在作者作出决定前，缺少软件许可的 warning 属于预期状态；归档 DOI 仍是后续正式发布任务。

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

## 投稿前发布

作者需要选择软件许可、将审核后的仓库设为公开、建立带版本的 release，并把该 release 归档到 Zenodo 或其他可信仓库。取得 DOI 后，应把同一永久地址同步写入稿件的 Data availability、Code availability 和本 README。`CITATION.cff` 已记录经确认的源代码仓库 URL，但在真实 DOI 生成前不会填写 DOI。

本仓库目前尚未授予软件许可；未来选择的代码许可也不会改变 MNIST、Fashion-MNIST 及第三方依赖各自的许可。
