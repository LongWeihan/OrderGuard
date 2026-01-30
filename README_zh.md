# OrderGuard：让 LLM 判题 / 重排 / 工具选择对“候选顺序”更鲁棒（LaTIn / PCons）

候选项的顺序本身并不是有意义的信号——但很多 LLM 流水线会在不经意间把它当成信号。

当你用 LLM 从 N 个候选中**选 1 个**（LLM-as-a-judge、RAG reranking、Agent 的 tool/action selection），只要把同一组候选**换个排列顺序**，最终赢家就可能改变，导致系统不稳定、评测噪声大、线上难复现。

**OrderGuard** 是一个无需训练的推理期包装器：通过对候选顺序做（近似）**置换边缘化**（marginalize over permutations），使用强约束的 **logprob forced-choice 打分** + **自适应 early-stop**，显著降低“顺序敏感性”，并在多选 benchmark 上带来稳定的准确率增益。

**TL;DR**

- 单次（single-shot）的“从列表里选一个”对重排极其不稳定（在我们的 Qwen3 测试里，只做 10 次随机重排就有 58-75% 的样本会翻转赢家）。
- OrderGuard 在 Qwen3 上把宏平均准确率提升 **+2.8 到 +4.6 个百分点**，单个数据集最高可到 **+7.6pp**。
- LaTIn 通常比随机置换更省算力（方差更小、需要的置换次数更少）。

**论文式表述：对置换群的 group-averaging 推理 + 低方差采样设计 + 自适应停止准则**

- **Permutation-group averaging（推理期不变性）：**把候选顺序当作干扰变量，通过对多个置换的打分聚合来边缘化顺序。
- **Low-variance design（LaTIn）：**用位置均衡的循环调度（Latin-square 风格），使每个候选在各个位置出现次数相同，从而比随机置换方差更低。
- **Adaptive stopping：**用 JS divergence 判断聚合分布是否稳定（阈值触发 early-stop），把更多 test-time compute 分配给“难样本”。

## 最小 API（同样适用于 tool/action selection）

```python
from orderguard.methods import latin_consensus
from orderguard.modeling import load_lm

lm = load_lm("Qwen/Qwen3-1.7B", torch_dtype=None)

question = "Pick the best next tool for: extract the answer from a table."
choices = [
    "WebSearch: use the browser to find information online.",
    "Calculator: do arithmetic precisely.",
    "TableParser: read structured tables and extract fields.",
    "WriteCode: write a short script to compute the result.",
]

res = latin_consensus(lm, question, choices, max_perms=7, min_perms=3, js_eps=0.005, seed=0)
print("winner:", choices[res.pred_index], "perms_used:", res.meta["perms_used"])
```

## 为什么重要（问题有多严重）

在 Qwen3 上，*single-shot* 多选题对顺序非常敏感：只做 **10 次随机重排**，预测赢家会发生变化的比例为：

- `Qwen/Qwen3-0.6B`：**75.3%**（7 个数据集平均；最高 **89.0%** 出现在 TruthfulQA(MC1)）
- `Qwen/Qwen3-1.7B`：**58.2%**（最高 **82.5%** 出现在 HellaSwag）

这里的“变化”定义是：对同一个样本，只要 10 次重排中**任意一次**的赢家与原始顺序的赢家不同，就算发生变化。

本项目的目标很朴素：让“从列表里选一个”尽可能像它应当表现的那样——**对重排稳定**——并且不需要重新训练大模型。

## 核心结果（Qwen3，可完全复现）

7 个多选 benchmark（ARC-C、OpenBookQA、CSQA、TruthfulQA(MC1)、MMLU(all)、HellaSwag、WinoGrande-XS）的宏平均准确率：

| 模型 | Single | PCons（随机） | LaTIn（Latin） |
|---|---:|---:|---:|
| Qwen/Qwen3-0.6B | 43.6% | 47.5%（**+3.9 pp**） | 48.2%（**+4.6 pp**） |
| Qwen/Qwen3-1.7B | 57.4% | 60.2%（**+2.8 pp**） | 60.7%（**+3.3 pp**） |

单个数据集上最大的提升（相对 single 的准确率增益，绝对百分点 pp）：

| 模型 | 数据集 | 提升 | 方法 |
|---|---|---:|---|
| Qwen/Qwen3-0.6B | OpenBookQA | **+7.6 pp** | LaTIn |
| Qwen/Qwen3-0.6B | TruthfulQA(MC1) | **+7.0 pp** | PCons |
| Qwen/Qwen3-0.6B | CSQA | **+6.6 pp** | PCons |
| Qwen/Qwen3-1.7B | HellaSwag | **+7.4 pp** | LaTIn |
| Qwen/Qwen3-1.7B | OpenBookQA | **+5.2 pp** | LaTIn |
| Qwen/Qwen3-1.7B | MMLU(all) | **+3.6 pp** | LaTIn |

复现实验产物（所有数字 + per-example logs）：`reports/paper_qwen3/`。

算力说明（同一次 run）：`max_perms=7`、`min_perms=3`、`js_eps=0.005`。LaTIn 平均每题使用 **~3.7-3.8** 次置换，PCons 为 **~4.3-4.7** 次（single-shot 为 1x）。

从产物重新生成相同图表：

```powershell
.\.venv\Scripts\python -m orderguard.plots.make_figures --run_dir reports/paper_qwen3
```

![Macro accuracy](assets/figures/macro_accuracy.png)

按数据集展示增益（paired bootstrap 95% CI；相对 single 的增益）：

![Accuracy gains by task](assets/figures/accuracy_gain_by_task.png)

## 核心思想：置换群 group-averaging + 低方差设计 + 自适应停止

我们想要一个“重排不变”的决策规则。候选顺序的置换构成一个对称群（permutation group），让决策对该群不变的标准做法是 **group averaging**。把呈现顺序当作干扰变量 `pi`。

对每个置换 `pi`，我们让模型回答一个强约束的 forced-choice 问题（"Answer with A/B/C/..."），只记录字母答案的 log-prob：

`ell_pi(i) = log p(letter = pos(i under pi) | question, options permuted by pi)`

然后把字母位置映射回原始候选 id，并把不同置换下的分数聚合：

`score(i) = sum_{pi in S} ell_pi(i)`  ->  `p(i) = softmax(score(i))`

这可以看成是对置换边缘化目标 `E_pi[log p(i | pi)]` 的 Monte-Carlo 近似：如果模型存在**位置偏置**（比如更偏好靠前选项或某个字母），对置换做平均会把这种偏置抵消掉。

### 为什么 LaTIn 往往比随机更好

用一个简单分解解释“方差更小”：

`ell_pi(i) = u(i) + b(position of i in pi) + eps(i,pi)`

- `u(i)`：候选 `i` 的“真实偏好/语义相关性”
- `b(pos)`：位置偏置
- `eps`：噪声项

随机置换可以在**期望意义**上抵消 `b`；而 **LaTIn** 用 Latin 风格的循环调度，让每个候选在每个位置出现次数相同，因此在一个循环内能更“精确”地抵消位置偏置（方差更低 -> 更少置换达到同等稳定性）。

### 自适应 early-stop（为什么省算力）

每做一次置换，我们更新聚合后的分布 `p_t`。当 `JS(p_t, p_{t-1}) < js_eps`（并且已达到 `min_perms`）就停止。
简单样本很快收敛，难样本才消耗更多 test-time compute。

## 与相关方法 / 常见基线的对比

你很难从常见基线里同时获得以下优点：

- **无需训练**、可即插即用：不需要 SFT/RL，不需要新模型。
- **短输出打分**：只对 `A/B/C/...` 做 logprob（快、确定性更强，避免长输出格式漂移）。
- **对重排鲁棒是“构造出来的”**：直接对顺序做边缘化，而不是靠 prompt 祈祷模型忽略顺序。
- **自适应算力**：只在需要时多做置换。
- **低方差调度（LaTIn）**：位置均衡的置换序列通常更省置换次数。

对比表（都在解决同一个问题：顺序偏置 / 列表不稳定）：

| 方法 | 无需训练 | 需要 logprobs | 不需要 logprobs 也能用 | 抗顺序机制 | 计算量 | 常见问题 |
|---|---|---|---|---|---:|---|
| Single-shot | 是 | 可选 | 是 | 无 | 1x | 对顺序/位置偏置非常脆弱 |
| "请忽略顺序" prompting | 是 | 可选 | 是 | 靠提示词 | 1x | 不可靠、难验证 |
| Shuffle once | 是 | 可选 | 是 | 随机打散一次 | 1x | 方差大，仍会翻转 |
| K 次重排投票（生成） | 是 | 否 | 是 | 通过投票集成 | Kx | 慢；解析/格式漂移；非确定性 |
| Pairwise tournament | 是 | 可选 | 是 | 两两比较 | O(N^2) | 候选多时开销大 |
| 训练期去偏（SFT/RL） | 否 | N/A | N/A | 改模型本身 | 昂贵 | 不可插拔、难复现 |
| **OrderGuard PCons** | 是 | 是 | 否 | 置换边缘化 + JS early-stop | ~Kx（自适应） | 需要 logprob |
| **OrderGuard LaTIn** | 是 | 是 | 否 | 位置均衡置换 + JS early-stop | ~Kx（通常更小） | 需要 logprob |

## 相关工作与定位

OrderGuard 聚焦一个在实际 LLM 系统中反复出现但经常被低估的问题：列表决策中的**顺序偏置**。我们的贡献是把“置换群平均推理 + 低方差置换调度 + 自适应停止”落成一个可复现、可度量、可即插即用的方法体系，并给出成体系的评测与产物。

为便于严谨定位，这里按研究维度列出代表性工作：

- **投票 / 集成：**例如 *Self-Consistency Improves Chain of Thought Reasoning in Language Models*（Wang et al., arXiv:2203.11171）推广了多次采样并用投票/一致性聚合的思路。
- **logprob 强约束打分：**经典 LM 评测常用 likelihood 来给离散答案打分而不是靠自由生成（例如 *Language Models are Few-Shot Learners*，Brown et al., arXiv:2005.14165；以及更近的 MCQA 打分分析 *Choices Speak Louder than Questions*，arXiv:2502.18798）。
- **位置偏置 / 顺序效应：**测量 LLM 在具体应用里的 position bias（例如 arXiv:2401.01989、arXiv:2508.02020）。
- **稳定性指标（prompt/order sensitivity）：**显式度量 LLM 对表面变化的敏感性/鲁棒性（例如 *ProSA*，arXiv:2410.12405；arXiv:2509.01790）。
- **低方差采样 / 平衡设计：**Latin square 设计（Fisher, *The Design of Experiments*, 1935）与 Latin hypercube sampling（McKay et al., 1979）是经典的方差降低工具；LaTIn 是其在“置换群平均”场景下的离散类比。

## 快速开始

```powershell
cd c:\26spring\py\sf260129\orderguard
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .
```

运行（Qwen3 小模型）：

```powershell
.\.venv\Scripts\python -m orderguard.bench.run --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B
.\.venv\Scripts\python -m orderguard.bench.sensitivity --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --examples 200 --perms 10
.\.venv\Scripts\python -m orderguard.plots.make_figures --run_dir reports/latest
```

输出：

- `reports/latest/summary.csv`：task x method x model 的指标
- `reports/latest/per_example__*.jsonl`：逐样本日志（用于 paired bootstrap）
- `reports/latest/figures/*.png|svg`：论文风格图表

## 目录结构

- `src/orderguard/`：方法 + 任务适配 + 模型封装
- `src/orderguard/bench/`：benchmark（accuracy / sensitivity）
- `src/orderguard/plots/`：最小且清晰的论文图表
- `reports/paper_qwen3/`：冻结的可复现实验产物（Qwen3-0.6B/1.7B）

## 引用

见 `CITATION.cff`。
