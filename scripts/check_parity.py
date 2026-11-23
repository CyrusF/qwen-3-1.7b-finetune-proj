"""Check that the hand-written model matches HuggingFace, and that the KV cache
produces the same numbers as running without one.

    python -m scripts.check_parity
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from core.devices import resolve_device
from core.model import Qwen3ForCausalLM
from core.utils import load_qwen3_weights, get_encoder


# fp32 weights, so differences should be tiny. anything above this is a real bug.
TOLERANCE = 1e-3

PROMPTS_SINGLE = ["the capital of france is"]
PROMPTS_RAGGED = ["the capital of france is", "hi"]


def make_batch(tokenizer, prompts, device):
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    ids, mask = batch.input_ids, batch.attention_mask
    positions = (mask.cumsum(dim=-1) - 1).clamp(min=0)
    return ids, mask, positions


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def report(name, a, b, mask=None):
    """Print the largest difference between two logit tensors, per batch row."""
    diff = (a.float() - b.float()).abs()
    if mask is not None:
        diff = diff.masked_fill(~mask.unsqueeze(-1).bool(), 0)

    worst = diff.max().item()
    status = "PASS" if worst < TOLERANCE else "FAIL"
    print(f"  [{status}] {name}: max diff {worst:.6f}")

    if diff.size(0) > 1:
        for row in range(diff.size(0)):
            print(f"           row {row}: {diff[row].max().item():.6f}")
    return dict(name=name, max_diff=worst if torch.isfinite(diff).all() else None,
                passed=worst < TOLERANCE)



def hf_logits(tokenizer, device, weights_dir, prompt_sets):
    from transformers import Qwen3ForCausalLM as HFQwen3

    hf = HFQwen3.from_pretrained(
        weights_dir or "./weights/qwen3-1.7b",
        dtype=torch.float32,
        attn_implementation="eager",  # match the hand-written attention
        local_files_only=True,
    ).to(device).eval()

    results = []
    with torch.no_grad():
        for prompts in prompt_sets:
            ids, mask, positions = make_batch(tokenizer, prompts, device)
            results.append(hf(input_ids=ids, attention_mask=mask, position_ids=positions,
                              use_cache=False).logits.cpu())
    del hf
    free(device)
    return results


def ours_no_cache(model, ids, mask, positions, num_layers):
    with torch.no_grad():
        logits, _ = model(ids, positions, mask, [None] * num_layers)
    return logits


def ours_token_by_token(model, ids, mask, positions, num_layers):
    """Feed one token at a time through the cached path, exactly like decoding."""
    caches = [None] * num_layers
    steps = []

    with torch.no_grad():
        for i in range(ids.size(1)):
            # the mask has to cover every key in the cache, not just the new token
            logits, caches = model(
                ids[:, i : i + 1],
                positions[:, i : i + 1],
                mask[:, : i + 1],
                caches,
            )
            steps.append(logits)

    return torch.cat(steps, dim=1)


def main(device=None, weights_dir=None, report_path=None):
    device = resolve_device(device)
    print(f"device: {device}\n")

    tokenizer = get_encoder(weights_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # huggingface first, then free it - two fp32 copies of a 1.7b model do not
    # fit on a 16gb card at the same time
    print("loading huggingface reference...")
    ref_single, ref_ragged = hf_logits(tokenizer, device, weights_dir, [PROMPTS_SINGLE, PROMPTS_RAGGED])

    print("loading our model...")
    cfg, sd = load_qwen3_weights(weights_dir)
    num_layers = cfg.num_hidden_layers
    model = Qwen3ForCausalLM(cfg)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    del sd
    free(device)

    passed = []

    print("\n1. our forward pass vs huggingface")

    ids, mask, positions = make_batch(tokenizer, PROMPTS_SINGLE, device)
    ours = ours_no_cache(model, ids, mask, positions, num_layers)
    passed.append(report("HF / single prompt", ours.cpu(), ref_single))

    ids_r, mask_r, pos_r = make_batch(tokenizer, PROMPTS_RAGGED, device)
    ours_r = ours_no_cache(model, ids_r, mask_r, pos_r, num_layers)
    passed.append(report("HF / ragged batch", ours_r.cpu(), ref_ragged, mask=mask_r.cpu()))

    print("\n2. cached decode vs single forward pass")

    stepped = ours_token_by_token(model, ids, mask, positions, num_layers)
    passed.append(report("KV cache / single prompt", stepped.cpu(), ours.cpu()))

    stepped_r = ours_token_by_token(model, ids_r, mask_r, pos_r, num_layers)
    passed.append(report("KV cache / ragged batch", stepped_r.cpu(), ours_r.cpu(), mask=mask_r.cpu()))

    count = sum(check["passed"] for check in passed)
    print(f"\n{count}/{len(passed)} checks passed")
    if report_path:
        Path(report_path).write_text(json.dumps(dict(device=str(device), dtype="float32",
                                                   tolerance=TOLERANCE, checks=passed), indent=2))
    return 0 if count == len(passed) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device")
    parser.add_argument("--weights-dir")
    parser.add_argument("--report-path")
    raise SystemExit(main(**vars(parser.parse_args())))
