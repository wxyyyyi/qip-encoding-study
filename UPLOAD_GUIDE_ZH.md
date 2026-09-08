# 上传到 GitHub 和 Zenodo 的具体步骤

本文件针对第一次建立代码仓库的情况。当前发布包仍在本地，以下步骤需要你在自己的 GitHub 账户中完成。

## 方案 A：GitHub Desktop（推荐）

1. 安装并打开 GitHub Desktop，登录你的 GitHub 账号。
2. 选择 `File` -> `Add local repository` -> `Choose...`，选择本发布包目录 `QIP_Encoding_Study_release_20260822`。
3. 如果 GitHub Desktop 提示“不是 Git 仓库”，选择 `create a repository`，名称建议使用 `qip-encoding-study`，本地路径保持当前发布包目录。
4. 在提交前运行 `python code/validation/validate_release.py`；必须为 0 failures。运行 `python code/validation/generate_sha256_manifest.py` 后把 `SHA256SUMS.txt` 一并纳入提交。
5. 在左下角填写提交说明，例如 `Prepare reproducibility package for QIP submission`，点击 `Commit to main`。
6. 点击 `Publish repository`。仓库可以先设为 `Private`，确认文件无隐私信息和许可问题后再改为 `Public`。
7. 在网页上检查 README、`data/README.md`、`results/RESULTS_MANIFEST.md`、`CITATION.cff` 和 `SHA256SUMS.txt`，并确认没有上传 MNIST 原始数据、个人路径、虚拟环境或临时文件。
8. 创建一个版本标签，例如 `v1.0.0`，作为论文投稿时使用的固定版本。仓库设置为 Public 后，把仓库 URL 复制下来。

## 方案 B：Zenodo 生成 DOI

1. 用 GitHub 账号登录 [Zenodo](https://zenodo.org/)。
2. 在 Zenodo 的 GitHub 集成设置中开启目标仓库。
3. 在 GitHub 仓库中创建并推送一个 release，例如 `v1.0.0`。
4. Zenodo 会自动归档该 release。检查标题、作者、摘要、许可协议和文件清单。
5. 发布 Zenodo record，复制 DOI landing page URL 和 DOI。
6. 只有在 DOI 已经可以从无登录浏览器打开后，才把 DOI 写入论文。

## 投稿前要同步的四处内容

将真实 URL 或 DOI 写入：

- 论文的 `Data availability`；
- 论文的 `Code availability`；
- 仓库 README 和 `CITATION.cff` 中的仓库地址/DOI（如果作者决定在 CFF 中填写）。

三处必须使用同一个永久地址。若仓库内容发生改变，应创建新的版本，而不是覆盖已经引用的 release。

## 建议的最终论文声明

```text
Data availability
MNIST and Fashion-MNIST are publicly available through torchvision and are described in the repository data README. The repository at [PERMANENT DOI OR URL] contains the deterministic split specification, processed result tables and analysis outputs supporting this study.

Code availability
Source code, experimental configurations, processed result tables, training logs and figures are available at [PERMANENT DOI OR URL].
```

当前候选包不公开 checkpoint，论文声明不能声称包含 selected checkpoints。若以后决定公开，须先完成逐文件加载验证并建立 model/dataset/seed 清单，再创建新版本。
