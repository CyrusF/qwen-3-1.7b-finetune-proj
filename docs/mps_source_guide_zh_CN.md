# Qwen3 MPS 源码导读

[简体中文 README](../README_ZH_CN.md) · [English README](../README.md)

按环境检查、Forward parity、SFT、GRPO 单步、300-step GRPO 的顺序阅读。每章从终端入口开始，再打开对应函数查看输入、输出和通过条件。网页里的 **源码导读** 会展示当前 checkout 的代码片段。 [课程入口](http://127.0.0.1:8765)

## 0. 从终端命令追到 MPS

`python -m mps_lab <stage>` 创建一个 run ID，再用同一个 Python 环境启动实验。每次运行的退出状态和日志分别保存在 `run.json` 与 `terminal.log`。网页从目录读取这些文件，它的启动命令是 `python3 -m mps_lab web`。 [终端入口](../mps_lab/__main__.py) · [网页服务](../mps_lab/web.py)

`setup()` 先用真实 MPS（Metal Performance Shaders）张量跑矩阵乘法和反向传播，再验证 BF16（Brain Floating Point 16-bit）SDPA（Scaled Dot-Product Attention）、AdamW 和 sampling。后续实验显式传 `device="mps"`。训练在 MPS 上使用 PyTorch loss，内存由 `torch.mps` 报告，其中 `driver_gib` 表示 Metal driver 的当前分配量。 [环境探针](../mps_lab/experiments.py) · [设备选择](../core/devices.py)

## 1. Forward parity

从 `scripts.check_parity.main()` 进入。相同的 tokenizer 为两个实现生成 `input_ids`、`attention_mask` 和 `position_ids`。脚本先加载 Hugging Face reference，再加载自定义 Qwen3，两者都走 FP32（32-bit Floating Point）和 eager attention。 [Parity runner](../scripts/check_parity.py)

`Qwen3Attention.forward()` 会计算 Q、K、V，向 Q 和 K 应用 RoPE（Rotary Position Embedding），再把新增的 K 和 V 拼到 KV cache（Key–Value Cache）。在 `ours_token_by_token()` 中，attention mask 的长度随着 cache 增长。比较时排除 left padding 位置。 [Attention 实现](../core/model.py) · [逐 token 检查](../scripts/check_parity.py)

查看 run 的 `output/report.json`。单 prompt 和 ragged batch 各有两项检查，一项比对 Hugging Face logits，另一项比对 cached decode 和单次 forward。每项的最大绝对误差都应小于 `1e-3`。

## 2. 小规模 LoRA SFT

`sft()` 从已有 split 中按 token 数选 4 条训练样本和 2 条验证样本，交给 `scripts.sft_rewrites.main()`。`load_sft_dataset()` 将 chat prompt 的 labels 设为 `-100`，把回答和 EOS（End of Sequence）保留为目标。Loss 会把 labels 移一位，以便用当前 token 预测下一个 token。 [SFT 实验入口](../mps_lab/experiments.py) · [Labels 数据流](../training/sft/sft_utils.py) · [Loss 和训练循环](../training/sft/sft.py)

`LoRALinear` 用 A、B 两个低秩矩阵给基座投影添加增量。基座权重冻结，adapter 保持 FP32。在 MPS 的 SFT 180 中，4 样本 batch 会拆成单样本 microbatch。每个 microbatch 的 loss 按目标 token 数加权，累积后执行一次 optimizer step。 [LoRA 层](../core/layers.py) · [SFT runner](../scripts/sft_rewrites.py)

在网页展开真实样本，蓝色 token 参与 loss。再看 `metrics.jsonl` 中的 loss 和 gradient norm，以及 `latest.pt` 对应的 LoRA B 参数变化。小规模实验的 validation loss 衡量模型预测目标回答的能力。

## 3. 一次 GRPO update

`grpo_one()` 加载上一章的 `latest.pt`。对同一个 prompt 采样 4 个 candidates，使用固定的教学 reward。采样和训练时对词表使用同一张 no-e token mask。 [单步实验](../mps_lab/experiments.py) · [生成与 EOS](../core/generation.py) · [Token mask](../core/sampling.py)

沿着 `GRPOEngine` 读这四个动作。

1. `generate_rollouts()` 留下 candidate tokens、用于评分的 reward mask 和用于更新的 loss mask。
2. `compute_advantages()` 用每个 reward 减去组均值，再除以组内标准差。全组同分时，每个 advantage 都为零。
3. `token_log_probs()` 重新计算候选在训练 forward 下的 log probability。更新时以新旧 log probability 的差计算 policy ratio，再应用 clipping 和 KL（Kullback–Leibler）惩罚。
4. `fit()` 按 microbatch 计算 gradient 并累积，最后执行一次 optimizer step。

[GRPO 引擎](../training/rl/grpo.py)

在网页对照 candidate、reward 和 advantage，再把练习滑块设为同分。通过条件是 gradient norm 有限且为正、adapter 参数确实变化，并写出 `step_0001.pt`。

## 4. Judge-based GRPO 与 300-step 复现

`scripts.rl_judge.main()` 用同一训练池和 eval set 启动两条路线。Base 使用新建的 LoRA 参数，SFT 180 加载完整 SFT run 的 `best.pt`。每步采样 4 条 prompts × 8 个 candidates。Base 的 KL reference 通过关闭 adapter 得到，SFT 180 临时换入起始 adapter 值来计算 reference log probability。 [GRPO runner](../scripts/rl_judge.py) · [Reference policy 实现](../training/rl/grpo.py)

`JudgeReward` 解码回答，调用 `gpt-6-luna` 计算 coherence 和 responds，再把分数合成 reward。Judge 请求 ID 与样本一起记录。训练前先测本机 step 0，之后每 25 steps 保存 checkpoint 并跑 50 条 held-out prompts × 4 个 seeds。网页将本机的 usable rate 曲线与仓库已发布的 CUDA 曲线叠加。 [Judge 与 usable 规则](../core/judge.py) · [Eval 汇总](../scripts/judge_eval.py)

原实验还用了两个检查。

- `judge_check` 用已知好坏样本检验 judge 的 rubric。 [脚本](../scripts/judge_check.py) · [历史结果](../runs/judge_check_20261016_v3/)
- `rl_check` 根据已经评分的候选组，统计含 usable 候选的 prompts 占比和组内 reward 多样性。两项指标说明采样有没有提供可区分的学习信号。 [脚本](../scripts/rl_check.py) · [历史结果](../runs/rl_check_20261016/)

完整复现以 `output/training/checkpoints/step_0300.pt` 和 `output/training/eval/step_0300/short_eval_report.json` 为完成标志。 [原实验第 13 节](experiment_history.md#13-rl-with-the-judge-as-the-reward-oct-16)
