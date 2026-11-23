# 让 Qwen3-1.7B 写出不含字母 e 的回答

[English](README.md) · [简体中文](README_ZH_CN.md)

这个 fork 在原项目的基础上加入了 Apple Silicon 本地实验课和[中文源码导读](docs/mps_source_guide_zh_CN.md)。原实验的目标是让 Qwen3 在避开 `e` 和 `E` 的同时，给出通顺、切题的回答。 [上游仓库](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj)

原项目用 PyTorch 手写了 Qwen3 forward pass、KV cache（Key–Value Cache）、LoRA（Low-Rank Adaptation）、SFT（Supervised Fine-Tuning）和 GRPO（Group Relative Policy Optimization），并在一块 16 GB CUDA GPU 上完成了实验。

## 在 Mac 上跟着实验课运行

课程以 MPS（Metal Performance Shaders）为后端。网页展示步骤、源码片段、真实日志和指标。你在终端执行每章的命令，网页会自动读取结果。历史 CUDA 曲线可直接浏览。

在仓库根目录启动网页服务。

```sh
python3 -m mps_lab web
```

打开 [Qwen3 MPS 实验课](http://127.0.0.1:8765)，在另一个终端依次执行环境检查、Forward parity、小规模 LoRA SFT 和 GRPO 单步。

```sh
uv venv --python 3.13 .venv
uv pip install --no-cache --python .venv/bin/python -r requirements-mps.txt
.venv/bin/python -m mps_lab setup
.venv/bin/python -m mps_lab parity
.venv/bin/python -m mps_lab sft
.venv/bin/python -m mps_lab grpo-one --adapter /path/to/latest.pt
```

在 GRPO 单步页面选择已完成的 SFT run，即可复制带实际 `latest.pt` 路径的命令。下载权重和 tokenizer 后，前三个实验可离线运行。

`setup` 会验证 MPS 上的 forward、backward、AdamW 和 sampling，并下载约 4.08 GB 的固定 revision 权重。首次下载前需留出至少 5 GiB。安装和完整实验合计建议预留约 12 GiB。输出默认保存到 `.mps-lab/runs/`。如需使用其他卷，在两个终端里都将 `MPS_LAB_DIR` 设为同一个绝对路径。

### 300-step GRPO

最后一章有 Base 和 SFT 180 两条路线。先在环境变量或仓库根目录的 `.env` 中配置 `OPENAI_API_KEY`，再运行 `judge-check` 发送一条探针请求。每条完整路线使用 `gpt-6-luna` judge，包含 step 0 和后续 eval，约需 12,200 次请求，另计重试。

```sh
.venv/bin/python -m mps_lab judge-check
.venv/bin/python -m mps_lab grpo300 base
.venv/bin/python -m mps_lab sft180
.venv/bin/python -m mps_lab grpo300 sft180 --adapter /path/to/best.pt
```

SFT 180 完成后，在网页选择对应 run 以获取 `best.pt` 路径。每次运行会生成一个新的 run ID。按 Ctrl+C 可以中断实验，并在网页查看已保存的文件。再次执行 GRPO 命令会从头开始。

### 开发验证

```sh
.venv/bin/python -m unittest discover -s tests -v
```

测试会用 CPU 小模型验证 SFT 梯度累积，并在支持 MPS 的设备上用小模型和模拟 judge 验证两个 GRPO 起点。查看每一步的输入、输出和关键函数。 [中文源码导读](docs/mps_source_guide_zh_CN.md)

## 原 CUDA 实验结果

每个条件评估 200 个回答（50 条 held-out prompts × 4 个样本）。满足字符约束，且 LLM（Large Language Model）judge 在通顺度和切题度上都给出至少 4/5 分的回答，计入 **usable rate**。

| 设置 | Usable rate |
|---|---:|
| 原始 Qwen3-1.7B，无字符约束，只评通顺度和切题度 | 80.5% |
| 只在 prompt 中要求避免字母 e | 0% |
| 同样的 prompt，并用 token mask 屏蔽含 e 的 token（Base） | 26% |
| Base + 用 180 条无 e 回答做 SFT | 35% |
| Base + 300 GRPO steps | 43% |
| SFT adapter + 300 GRPO steps | **60.5%** |

Token mask 负责字符约束。训练帮助模型在允许的 token 范围内写出更通顺、切题的回答。上表来自上游已发布的 CUDA 实验。 [完整实验记录](docs/experiment_history.md)

## 源码入口

| 想看什么 | 从哪里开始 |
|---|---|
| 网页读日志、指标和源码片段 | [`mps_lab/web.py`](mps_lab/web.py) |
| 实验参数和运行顺序 | [`mps_lab/experiments.py`](mps_lab/experiments.py) |
| 设备选择、autocast 和 MPS 内存指标 | [`core/devices.py`](core/devices.py) |
| Qwen3 forward、RoPE（Rotary Position Embedding）和 KV cache | [`core/model.py`](core/model.py) |
| LoRA、token mask 和生成 | [`core/layers.py`](core/layers.py)、[`core/sampling.py`](core/sampling.py)、[`core/generation.py`](core/generation.py) |
| Forward parity 和 SFT | [`scripts/check_parity.py`](scripts/check_parity.py)、[`scripts/sft_rewrites.py`](scripts/sft_rewrites.py)、[`training/sft/sft.py`](training/sft/sft.py) |
| GRPO、judge 和 eval | [`training/rl/grpo.py`](training/rl/grpo.py)、[`scripts/rl_judge.py`](scripts/rl_judge.py)、[`core/judge.py`](core/judge.py)、[`scripts/judge_eval.py`](scripts/judge_eval.py) |
| Judge 质量和 RL 候选检查 | [`scripts/judge_check.py`](scripts/judge_check.py)、[`scripts/rl_check.py`](scripts/rl_check.py) |

## 已发布的实验产物

历史 run 目录保存了 config、metrics、judge 评分和回答。阅读回答可打开 `responses.txt` 或 `samples_to_read.txt`，汇总结果见 `short_eval_report.json`。模型 checkpoint 由运行命令生成。

| 目录 | 内容 |
|---|---|
| [`runs/short_eval_20261016/`](runs/short_eval_20261016) | 1.7B 和 4B 模型的无约束、仅 prompt 指令及 token mask 对照 |
| [`runs/judge_check_20261016_v3/`](runs/judge_check_20261016_v3) | Judge 对已知好坏样本和早期已审核样本的评分 |
| [`runs/gadsby_short_eval_20261017/`](runs/gadsby_short_eval_20261017) | 用小说 *Gadsby* 训练的 adapter，usable rate 为 24.5% |
| [`runs/sft_sizes_20261016/`](runs/sft_sizes_20261016) | 15 至 180 条训练样本的 SFT 比较 |
| [`runs/rl_check_20261016/`](runs/rl_check_20261016) | 每个 prompt 生成多个候选，检查 RL（Reinforcement Learning）的学习信号 |
| [`runs/rl_20261016/`](runs/rl_20261016) | Base 和 SFT 180 两条 GRPO 路线，每 25 steps 评估一次 |

## 运行原 CUDA 实验

安装原始 CUDA 依赖，并在仓库根目录配置权重和 OpenAI key。

```sh
pip install -r requirements.txt
hf download Qwen/Qwen3-1.7B --local-dir weights/qwen3-1.7b
cp .env.example .env
```

在 `.env` 填好 `OPENAI_API_KEY` 后依次运行。

```sh
python -m data.prompts.make_prompt_data
bash runs/sft_sizes_20261016/run_all.sh
bash runs/rl_check_20261016.sh
bash runs/rl_20261016/run_both.sh
python -m scripts.judge_eval compare \
  --run-dirs runs/short_eval_20261016/base_1.7b_masked runs/rl_20261016/sft180_masked/eval/step_0300
```

原脚本会向上面的 run 目录写文件。如需保留已发布记录，先备份这些目录。可以用 `PY=/path/to/python` 指定 Python 解释器。

## 数据、许可与历史

- 代码使用 [MIT License](LICENSE)。
- Prompts 来自 [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)（CC BY-NC 4.0）和 [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)（CC BY-SA 3.0），各自沿用原许可。
- *Gadsby* 文本由脚本从 [Hellisotherpeople/Lipogram-e](https://huggingface.co/datasets/Hellisotherpeople/Lipogram-e) 下载。
- [`data/rewrites/`](data/rewrites/) 中的 200 条无 e 回答由 GPT 模型起草，经过审核和字符检查。每条记录保留来源和审核信息。
- 早期单 prompt RL 尝试可在上游 Git 历史中查看。 [上游提交 `21dcbfa`](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/tree/21dcbfa9d403e42529013e811eb1e06b18ba0e06)
