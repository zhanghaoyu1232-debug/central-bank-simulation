# 论文修订包（2026-09-08）

本包包含修订后的 LaTeX、图表、重新出图脚本及与正式实验结果一致的可执行源代码快照。

## 编译论文

在本目录运行：

```powershell
latexmk -xelatex main.tex
```

也可以在 VS Code 中用 LaTeX Workshop 编译 `main.tex`。保留了原来的主文件名，方便已有工程继续使用。需要 XeLaTeX、xeCJK 及日文字体；Windows 优先使用游明朝/游ゴシック，其他系统可用 Noto CJK。

## 这一轮不需要重跑模拟

实际代码的核心指纹为 `11dda58061ded609`，与 14 份正式和贷款上限结果文件一致。经济模型源码没有改动；改的是论文公式、统计口径说明、附录摘录和三张图。

- 图 7.1：按指标调整纵轴，保留点估计、原始配对 run 0 和完整区间。
- θ 和 w1：DEN/CEN 分面，共用坐标范围和参数色标，保留全部参数曲线。
- 上限实验：明确另一套 20 种子，保留真实的独立批次结果。
- 情景比较：按每对实验的共同原生窗口说明统计方法。
- 初始表：保留真实 CSV，说明 solvency 公式和初始 LCR 缓存时点。
- 附录：补齐无支付违约时的负权益检查、合同残余处理和债权人核销。

## 重新出这三张图

不需要 PyTorch，不需要载入或训练 GNN：

```powershell
python scripts/rebuild_review_figures.py
```

脚本使用本包的 `reproduction/plot_inputs.json`。如需从完整原始结果重新构造输入：

```powershell
python scripts/rebuild_review_figures.py --artifacts-dir "你的figures目录"
```

输出保存在 `figures/`，同时提供 PNG 和矢量 PDF。

## 复核代码与原始结果

```powershell
python scripts/verify_reproduction.py
python scripts/verify_reproduction.py --artifacts-dir "你的figures目录"
```

这里是针对本次疑点的轻量验证，不运行 Monte Carlo，也不代表完整模型的全量回归测试。它会核验源文件摘要、两个机制的原始递归方法、实际共享核销程序和 30 行初始化数据。

`reproduction/simulation_source/` 是实际执行代码快照。已包含对应的 v6 训练检查点；核心源文件逐字节保持不变。完整原始输出位于 GitHub Release 中的 `figures.zip`，本包通过 `reproduction/manifest.json` 记录每个结果文件的 SHA-256 和精确种子。

## GitHub 版本

正式源码、训练检查点和论文工程已推送到：

`https://github.com/zhanghaoyu1232-debug/central-bank-simulation`

2026-09-08 通过 `git ls-remote` 核对的 `master` 分支提交为
`fe3445374adac3e90c2d29cbf90100d02b6ee135`。第 6.1 节和附录 A 已写入该
真实提交号。核心源码指纹仍为 `11dda58061ded609`。

原始输出归档 `figures.zip` 发布于标签 `thesis-2026-09-08` 对应的 Release：

`https://github.com/zhanghaoyu1232-debug/central-bank-simulation/releases/tag/thesis-2026-09-08`

该 annotated tag 已通过 `git ls-remote` 核对，指向上述源码提交。

该提交中的 `paper/` 是写入提交号和 Release 链接之前的论文副本。当前包已补入两者；
如需让 GitHub 上的论文也保持最新，将当前 `paper/` 覆盖仓库中的同名目录，
再提交一次即可。论文引用的模拟源码版本仍是上述提交。
