# Teaching Qwen3-1.7B to write without the letter "e"

[English](README.md) · [简体中文](README_ZH_CN.md)

This fork adds a [local Apple Silicon course](#local-mps-course-apple-silicon) and a
[Chinese source walkthrough](docs/mps_source_guide_zh_CN.md) for the original
[*Post-training Qwen3 1.7B* project](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj).
The task is to make Qwen3 answer questions clearly without using `e` or `E`.

The original experiment uses a hand-written PyTorch forward pass, LoRA (Low-Rank Adaptation),
generation with a KV (Key–Value) cache, SFT (Supervised Fine-Tuning), and GRPO
(Group Relative Policy Optimization). It ran on one 16 GB CUDA GPU.

## Published CUDA results

Each number is the **usable rate** on 200 answers (50 held-out prompts × 4 samples). An answer is usable
if it contains no `e` and an LLM judge scores it at least 4 out of 5 on both coherence and answering the
prompt.

| Setup | Usable |
|---|---:|
| Plain Qwen3-1.7B, no e rule (judged on coherence and answering only) | 80.5% |
| Told not to use e, nothing else | 0% |
| Told not to use e, with tokens containing e masked out (**baseline**) | 26% |
| Baseline + SFT on 180 hand-built e-free answers | 35% |
| Baseline + 300 GRPO steps | 43% |
| SFT adapter + 300 GRPO steps | **60.5%** |

The token mask enforces the character constraint. Training helps the model answer coherently
with the remaining tokens.

## Where to find things

| Topic | Files |
|---|---|
| Local course and terminal presets | [`mps_lab/`](mps_lab/) |
| MPS device checks, autocast, and memory reporting | [`core/devices.py`](core/devices.py) |
| Qwen3 forward pass (RoPE, grouped-query attention, RMSNorm, KV cache) | [`core/model.py`](core/model.py) |
| LoRA, and switching it off to get the reference model | [`core/layers.py`](core/layers.py) (`LoRALinear`, `lora_disabled`) |
| The mask that blocks every token containing e | [`core/sampling.py`](core/sampling.py) (`build_forbidden_token_mask`) |
| Generation | [`core/generation.py`](core/generation.py) |
| The e check | [`core/checks.py`](core/checks.py) |
| The LLM judge and the usable rule | [`core/judge.py`](core/judge.py) (`JUDGE_SYSTEM`, `usable`) |
| Generating eval answers | [`scripts/evaluate.py`](scripts/evaluate.py) |
| Judging, reports, and run comparisons (with bootstrap ranges) | [`scripts/judge_eval.py`](scripts/judge_eval.py) |
| Checking the judge on known good and bad answers | [`scripts/judge_check.py`](scripts/judge_check.py) |
| Prompt dataset (No Robots + Dolly, filtered and split) | [`data/prompts/make_prompt_data.py`](data/prompts/make_prompt_data.py) |
| Eval prompts, and the ones dropped because they need an e | [`data/prompts/short_eval_50.jsonl`](data/prompts/short_eval_50.jsonl), [`short_eval_dropped.jsonl`](data/prompts/short_eval_dropped.jsonl) |
| RL prompt pool (1,235 prompts) and the keep/drop decisions behind it | [`data/prompts/rl_pool.jsonl`](data/prompts/rl_pool.jsonl), [`merge_rl_pool.py`](data/prompts/merge_rl_pool.py) |
| SFT on the novel *Gadsby* | [`data/sft/make_gadsby_data.py`](data/sft/make_gadsby_data.py), [`scripts/sft_gadsby.py`](scripts/sft_gadsby.py) |
| The 200 e-free answers, and the 15–180 example training splits | [`data/rewrites/verified_200_20261015.jsonl`](data/rewrites/verified_200_20261015.jsonl), [`data/rewrites/splits/`](data/rewrites/splits) |
| Training engine base class (checkpoints, metrics) | [`training/engine.py`](training/engine.py) |
| SFT engine | [`training/sft/sft.py`](training/sft/sft.py) (`SFTEngine`), runner [`scripts/sft_rewrites.py`](scripts/sft_rewrites.py) |
| GRPO engine | [`training/rl/grpo.py`](training/rl/grpo.py) (`GRPOEngine`) |
| GRPO with the judge as the reward | [`scripts/rl_judge.py`](scripts/rl_judge.py), pre-check [`scripts/rl_check.py`](scripts/rl_check.py) |
| REINFORCE engine | [`training/rl/reinforce.py`](training/rl/reinforce.py) |
| First attempt: rewarding single tokens without e | [`train_loop/simple_train.py`](train_loop/simple_train.py) |
| Checking the forward pass against Hugging Face | [`scripts/check_parity.py`](scripts/check_parity.py) |
| Full lab notebook, every experiment in order | [`docs/experiment_history.md`](docs/experiment_history.md) |

## Runs included

Each run folder keeps its config, metrics, judge scores, and the answers themselves. To read answers, open
`responses.txt` or `samples_to_read.txt`; the numbers are in `short_eval_report.json`. Model checkpoints
are not included.

| Folder | What it is |
|---|---|
| [`runs/short_eval_20261016/`](runs/short_eval_20261016) | 1.7B and 4B before any training: no rule, instruction only, and with the mask |
| [`runs/judge_check_20261016_v3/`](runs/judge_check_20261016_v3) | The judge scored on known good, earlier reviewed, and known bad answers |
| [`runs/gadsby_short_eval_20261017/`](runs/gadsby_short_eval_20261017) | The *Gadsby*-trained adapter on the eval set (24.5% usable) |
| [`runs/sft_sizes_20261016/`](runs/sft_sizes_20261016) | SFT on 15, 30, 45, 90, 135, and 180 examples, each evaluated |
| [`runs/rl_check_20261016/`](runs/rl_check_20261016) | 16 samples per prompt, to confirm RL has usable answers to learn from |
| [`runs/rl_20261016/`](runs/rl_20261016) | GRPO from the baseline and from the SFT adapter, evaluated every 25 steps |

## Running it

### Local MPS course (Apple Silicon)

The Chinese course walks through MPS (Metal Performance Shaders) setup, FP32 forward parity,
small LoRA SFT, one GRPO update, and the two 300-step runs. Expand **源码导读** to open excerpts
from the actual source. Run each experiment from the terminal. The read-only web UI displays
logs, metrics, artifacts, and the published CUDA curves.

```sh
python3 -m mps_lab web
```

Open [Qwen3 MPS 实验课](http://127.0.0.1:8765) and copy the commands from each lesson. In a second
terminal at the repository root, the short path is:

```sh
uv venv --python 3.13 .venv
uv pip install --no-cache --python .venv/bin/python -r requirements-mps.txt
.venv/bin/python -m mps_lab setup
.venv/bin/python -m mps_lab parity
.venv/bin/python -m mps_lab sft
.venv/bin/python -m mps_lab grpo-one --adapter /path/to/latest.pt
```

Select the completed SFT run in the GRPO lesson to copy the command with its actual checkpoint
path. The short experiments run offline once setup has downloaded the weights and tokenizer.

The setup probe checks native MPS forward, backward, AdamW, and sampling. It downloads about
4.08 GB of pinned Qwen3 weights. Allow 5 GiB before the initial download and budget around 12 GiB
before installing and running the full course. To store weights and experiment outputs on another
volume, set `MPS_LAB_DIR` to an absolute directory in both terminals before starting the server and
running the experiments. Outputs otherwise go under `.mps-lab/runs/`.

The final chapter has independent Base and SFT 180 routes. Put `OPENAI_API_KEY` in the environment or
the repo's `.env` before running `judge-check`. Each route runs a local step 0 evaluation and about
12,200 judge requests total, plus retries. Plan to run the two routes one at a time.

```sh
.venv/bin/python -m mps_lab judge-check
.venv/bin/python -m mps_lab grpo300 base
.venv/bin/python -m mps_lab sft180
# Select the completed SFT 180 run to copy the command with its actual best.pt path.
.venv/bin/python -m mps_lab grpo300 sft180 --adapter /path/to/best.pt
```

Use Ctrl+C in the experiment terminal to interrupt a run. You can inspect its saved files from
the run selector. Rerunning a GRPO command starts a fresh run.

Run `.venv/bin/python -m unittest discover -s tests -v` to check SFT gradient accumulation on a
small CPU model and rehearse both GRPO routes on MPS with a small model and a simulated judge.

### Original CUDA experiments

You need a CUDA GPU for training and evaluation (16 GB was enough), and an OpenAI API key for the judge.

```sh
pip install -r requirements.txt
hf download Qwen/Qwen3-1.7B --local-dir weights/qwen3-1.7b
cp .env.example .env    # then add OPENAI_API_KEY
```

Run everything from the repository root, in this order:

```sh
python -m data.prompts.make_prompt_data   # build the train/val/test prompt splits
bash runs/sft_sizes_20261016/run_all.sh    # SFT on 15-180 examples, then evaluate each
bash runs/rl_check_20261016.sh             # check both RL starting points
bash runs/rl_20261016/run_both.sh          # GRPO from the baseline and from the SFT adapter
python -m scripts.judge_eval compare \
  --run-dirs runs/short_eval_20261016/base_1.7b_masked runs/rl_20261016/sft180_masked/eval/step_0300
```

The scripts write into the run folders above, so copy those somewhere first if you want to keep the
published records. Set `PY=/path/to/python` to use a specific interpreter.

## Data and licenses

The code is MIT licensed (see [`LICENSE`](LICENSE)).

- Prompts come from [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (CC BY-NC 4.0)
  and [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
  (CC BY-SA 3.0), and keep those licenses.
- The *Gadsby* text is downloaded from
  [Hellisotherpeople/Lipogram-e](https://huggingface.co/datasets/Hellisotherpeople/Lipogram-e) by the
  script and is not stored here.
- The 200 e-free answers in `data/rewrites/` were drafted by GPT models working as Codex subagents, then
  reviewed and checked for e. Each row records its prompt's source and how it was checked.

## History

The September version of this repo, including the first single-prompt RL attempt, is in the git history.
The write-up links to commit [`21dcbfa`](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/tree/21dcbfa9d403e42529013e811eb1e06b18ba0e06).
