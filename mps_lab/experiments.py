import fcntl
import json
import math
import os
from pathlib import Path
import subprocess

from mps_lab.records import (
    GIB, MODEL, REVISION, ROOT, SHARD_SIZES, data_dir, free_gib, json_lines, read_json, weights_dir,
    weights_ready, write_json,
)

SPLITS = ROOT / "data/rewrites/splits"
PROMPTS = ROOT / "data/prompts"
REWARDS = [0.0, 0.25, 0.75, 1.0]


def mps_device():
    from core.devices import resolve_device

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK to verify native MPS execution")
    return resolve_device("mps", require_accelerator=True)


def setup(output):
    import torch
    import transformers
    from core.devices import autocast, mps_memory

    device = mps_device()
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    fp32 = torch.ones((2, 2), device=device, requires_grad=True)
    (fp32 @ fp32).mean().backward()
    adapter = torch.nn.Parameter(torch.full((1, 2, 4, 8), 0.1, device=device))
    optimizer = torch.optim.AdamW([adapter], lr=0.01)
    with autocast(device, torch.bfloat16):
        q = torch.randn((1, 2, 4, 8), device=device, dtype=torch.bfloat16)
        attn = sdpa(q + adapter.to(q.dtype), q, q, is_causal=True)
        loss = attn.float().square().mean()
    loss.backward()
    optimizer.step()
    sampled = torch.multinomial(torch.softmax(attn.detach().float().reshape(-1, 8), dim=-1), 1)
    torch.mps.synchronize()
    if not (fp32.grad.isfinite().all() and adapter.grad.isfinite().all() and sampled.device.type == "mps"):
        raise RuntimeError("MPS probe did not produce finite gradients and MPS samples")
    destination = weights_dir()
    report = dict(device=str(device), torch=torch.__version__, transformers=transformers.__version__,
                  memory=mps_memory(), weights_dir=str(destination), revision=REVISION,
                  free_gib=round(free_gib(data_dir()), 2),
                  probes=["FP32 matmul backward", "BF16 SDPA backward", "FP32 AdamW", "multinomial"])
    print(f"MPS probes passed. Available disk: {report['free_gib']:.2f} GiB")
    write_json(output / "report.json", dict(report, weights_ready=weights_ready(destination)))
    if report["free_gib"] < 12:
        print("Full course planning target: 12 GiB before download. The short lessons need less.")
    if not weights_ready(destination):
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / ".download.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another setup is downloading weights to {destination}") from error
            download_weights(destination)
    if not weights_ready(destination):
        raise RuntimeError(f"Model files are incomplete in {destination}")
    report["weights_ready"] = True
    write_json(output / "report.json", report)


def download_weights(destination):
    if not weights_ready(destination):
        downloaded = 0
        for name, size in SHARD_SIZES.items():
            final = destination / name
            partial = final.with_suffix(final.suffix + ".part")
            if final.is_file():
                downloaded += size
            elif partial.is_file():
                downloaded += partial.stat().st_size
        required = max(1, 5 - downloaded / GIB)
        if free_gib(data_dir()) < required:
            raise RuntimeError(f"Model download needs {required:.2f} GiB available. Set MPS_LAB_DIR to another volume.")
        print(f"Downloading {MODEL} at {REVISION} to {destination}")
        names = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json"]
        for name in names:
            download_file(destination, name)
        index = read_json(destination / "model.safetensors.index.json")
        for name in sorted(set(index["weight_map"].values()), reverse=True):
            download_file(destination, name)


def download_file(destination, name):
    final = destination / name
    if final.is_file():
        if name in SHARD_SIZES and final.stat().st_size != SHARD_SIZES[name]:
            raise RuntimeError(f"Wrong file size for {final}. Move it aside and rerun setup.")
        return
    partial = final.with_suffix(final.suffix + ".part")
    print(f"Downloading {name}")
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "20",
                    "--retry", "2", "--continue-at", "-", "--output", str(partial),
                    f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"], check=True)
    if name in SHARD_SIZES and partial.stat().st_size != SHARD_SIZES[name]:
        raise RuntimeError(f"Incomplete shard: {partial}")
    partial.replace(final)


def sft(output, full):
    import torch
    from core.utils import get_encoder
    from scripts import sft_rewrites
    from scripts.evaluate import NO_E_INSTRUCTION
    from training.sft.sft_utils import load_sft_dataset

    train, val = SPLITS / ("train_180.jsonl" if full else "train_15.jsonl"), SPLITS / "val.jsonl"
    if not full:
        tokenizer = get_encoder(str(weights_dir()))
        samples = []
        for split, source, count in (("train", train, 4), ("val", val, 2)):
            pairs = json_lines(source)
            encoded = load_sft_dataset(source, tokenizer, system_prompt=NO_E_INSTRUCTION)
            shortest = sorted(zip(pairs, encoded), key=lambda pair: (len(pair[1]["labels"]), pair[0]["id"]))[:count]
            path = output / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row, _ in shortest))
            for row, example in shortest:
                samples.append(dict(split=split, id=row["id"], prompt=row["prompt"], answer=row["response"],
                                    tokens=[dict(text=text, target=label != -100) for text, label in zip(
                                        tokenizer.convert_ids_to_tokens(example["input_ids"]), example["labels"])]))
            if split == "train":
                train = path
            else:
                val = path
        write_json(output / "tokens.json", samples)

    training_dir = output / "training"
    result = sft_rewrites.main(train_path=train, val_path=val, run_dir=training_dir,
                               device="mps", weights_dir=str(weights_dir()), batch_size=4 if full else 1,
                               max_length=3072 if full else 256, epochs=10 if full else 1,
                               learning_rate=1e-4, patience=10, eval_limit=0)
    if not full:
        checkpoint = torch.load(training_dir / "checkpoints/latest.pt", map_location="cpu")
        squares = [p.float().square().sum().item() for name, p in checkpoint["model_state_dict"].items()
                   if name.endswith(".B")]
        adapter_delta = math.sqrt(sum(squares))
        train_metrics = [row for row in json_lines(training_dir / "metrics.jsonl") if row["event"] == "train"]
        if not train_metrics or not all(
            math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in train_metrics
        ) or not math.isfinite(adapter_delta) or adapter_delta <= 0:
            raise RuntimeError("SFT did not produce finite metrics and a nonzero LoRA B update")
        write_json(output / "report.json", dict(adapter_delta_l2=adapter_delta, **result))


def grpo_one(output, adapter):
    import torch
    from core.devices import autocast, mps_memory
    from core.sampling import build_forbidden_token_mask
    from core.utils import get_encoder
    from scripts.rl_judge import load_model, prompt_batches
    from training.rl.grpo import GRPOEngine

    device = mps_device()
    torch.manual_seed(0)
    tokenizer = get_encoder(str(weights_dir()))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(adapter, device=device, weights_dir=str(weights_dir()))
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    reference = {name: p.detach().clone() for name, p in params.items()}
    mask = build_forbidden_token_mask(tokenizer, model.config.vocab_size, ascii_only=True, device=device)
    optimizer = torch.optim.AdamW(params.values(), lr=2e-5, weight_decay=0)
    engine = GRPOEngine(model, tokenizer, optimizer, "torch",
                        ref_adapter_state=reference, forbidden_token_mask=mask)
    engine.epoch, engine.global_step = 0, 0
    prompt = json_lines(PROMPTS / "rl_pool.jsonl")[0]
    write_json(output / "config.json", dict(
        device=str(device), precision="bf16", adapter_precision="float32", start_adapter=adapter,
        base_weights=str(weights_dir()), seed=0, reward=REWARDS, prompt_id=prompt["id"],
        group_size=4, max_new_tokens=16, micro_batch_size=1, temperature=0.7,
        learning_rate=2e-5, weight_decay=0, kl_beta=0.02, mask_e=True,
    ))
    batch = next(prompt_batches([prompt], tokenizer, 1, seed=0))
    batch.pop("ids")

    def report_step(info):
        rollouts = info["rollouts"]
        scores = info["scores"]
        advantages = engine.compute_advantages(rollouts, None, scores).flatten().tolist()
        grad_norm = math.sqrt(sum(p.grad.detach().float().square().sum().item()
                                  for p in params.values() if p.grad is not None))
        delta = math.sqrt(sum((p.detach() - reference[name]).square().sum().item()
                              for name, p in params.items()))
        answers = []
        for group, advantage in enumerate(advantages):
            tokens = rollouts.response_ids[group, 0][rollouts.reward_mask[group, 0]].tolist()
            answers.append(dict(answer=tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                                reward=REWARDS[group], advantage=advantage, tokens=tokens))
        if not all(math.isfinite(value) and value > 0 for value in (grad_norm, delta)):
            raise RuntimeError("GRPO did not produce a finite, nonzero gradient and adapter update")
        write_json(output / "report.json", dict(
            prompt=prompt["prompt"], answers=answers, grad_norm=grad_norm, adapter_delta_l2=delta,
            kl=info["kl"], logprob_gap=info["logprob_gap"], step=1, memory=mps_memory(),
        ))
        engine.global_step = 1
        engine.save_checkpoint(output / "checkpoints/step_0001.pt")
        print(f"GRPO step 1: gradient norm={grad_norm:.6g}, adapter delta={delta:.6g}")

    def teaching_reward(tokens, reward_mask, *, prompts):
        print(f"Generated {tokens.shape[0]} candidates. Applying fixed teaching rewards.")
        return torch.tensor(REWARDS).reshape(-1, 1)

    with autocast(device, torch.bfloat16):
        engine.fit([batch], max_new_tokens=16, temperature=0.7, group_size=4,
                   reward_fn=teaching_reward, kl_beta=0.02, micro_batch_size=1,
                   on_step=report_step, recompute_old_log_probs=True, parallel_groups=True)


def judge_check(output):
    from core.judge import judge_one, make_judge_client

    result = judge_one(make_judge_client(), "What is 2 + 2?", "Four.")
    write_json(output / "report.json", result)
    if result["error"]:
        raise RuntimeError(result["error"])
    print(f"Judge ready: {result['model']}; request IDs: {result['request_ids']}")


def run(task, location, route, adapter):
    output = location / "output"
    output.mkdir()
    if task == "judge-check":
        return judge_check(output)
    mps_device()
    if task == "setup":
        return setup(output)
    if not weights_ready(weights_dir()):
        raise RuntimeError("Run setup first to download the Qwen3 weights")
    if task == "parity":
        from scripts.check_parity import main

        return main("mps", str(weights_dir()), output / "report.json")
    if task in ("sft", "sft180"):
        return sft(output, full=task == "sft180")
    if task == "grpo-one":
        return grpo_one(output, adapter)

    from scripts.rl_judge import main

    if free_gib(data_dir()) < 3:
        raise RuntimeError("A 300-step run needs at least 3 GiB free for checkpoints and logs")
    if route == "sft180":
        path = Path(adapter)
        config = read_json(path.parent.parent / "config.json")
        if (path.name != "best.pt" or not path.is_file() or not config
                or config["data"]["splits"]["train"]["examples"] != 180):
            raise ValueError("Choose best.pt from a completed SFT 180 run")
    training_dir = output / "training"
    main(training_dir, prompts=PROMPTS / "rl_pool.jsonl", start_adapter=adapter,
         eval_prompts=PROMPTS / "short_eval_50.jsonl", device="mps", weights_dir=str(weights_dir()),
         baseline_eval=True, stop_on_decline=False)
    if not (training_dir / "checkpoints/step_0300.pt").is_file() or not (
        training_dir / "eval/step_0300/short_eval_report.json"
    ).is_file():
        raise RuntimeError("The 300-step checkpoint or final eval report is missing")
