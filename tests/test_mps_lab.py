import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from core.llm_api import Completion
from core.model import Qwen3ForCausalLM
from mps_lab.records import json_lines, read_json
from scripts import rl_judge
from training.sft.sft import SFTEngine
from training.sft.sft_utils import collate_sft


def small_model():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, head_dim=8,
                         num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
                         max_position_embeddings=64, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config, use_sdpa=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.endswith((".A", ".B")))
        if parameter.ndim == 2 and not parameter.requires_grad:
            torch.nn.init.normal_(parameter, std=0.1)
    return model


class SmallTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, texts, **kwargs):
        ids = torch.tensor([[2, 3, len(text) % 12 + 6] for text in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def get_vocab(self):
        return {}

    def decode(self, ids, skip_special_tokens=False):
        return "".join("e" if token == 5 else chr(65 + token % 26)
                       for token in ids if not skip_special_tokens or token not in self.all_special_ids)

    def batch_decode(self, sequences):
        return [self.decode(ids) for ids in sequences]


class FakeJudge:
    def complete(self, **kwargs):
        text = kwargs["messages"][0]["content"]
        score = 3 + sum(map(ord, text)) % 3
        return Completion('{"coherence": %d, "responds": %d, "note": "test"}' % (score, score),
                          "fake-judge", "stop", False, False, request_id="test-request")


class LabTests(unittest.TestCase):
    def test_sft_microbatch_preserves_token_weighted_update(self):
        examples = [
            {"input_ids": [2, 3, 12, 8, 1], "labels": [-100, -100, 12, 8, 1]},
            {"input_ids": [2, 10, 1], "labels": [-100, 10, 1]},
        ]
        batch = collate_sft(examples, pad_token_id=0)
        model = small_model()
        split = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as directory:
            for index, (current, size) in enumerate(((model, None), (split, 1))):
                optimizer = torch.optim.SGD((p for p in current.parameters() if p.requires_grad), lr=0.1)
                engine = SFTEngine(current, None, optimizer, False, "torch", precision=torch.float32,
                                   device="cpu", micro_batch_size=size, path=Path(directory) / f"{index}.pt")
                engine.fit([batch], [batch], epochs=1, eval_every=1, save_every=1, log_every=1)
                self.assertEqual(engine.global_step, 1)
                self.assertTrue(Path(engine.path).is_file())
        for left, right in zip(model.parameters(), split.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)

    @unittest.skipUnless(torch.backends.mps.is_available(), "Needs an MPS GPU")
    def test_base_and_adapter_rehearsals(self):
        tokenizer = SmallTokenizer()
        base = small_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            val = root / "eval.jsonl"
            train.write_text(json.dumps(dict(id="train", prompt="Train prompt")) + "\n")
            val.write_text(json.dumps(dict(id="eval", prompt="Eval prompt", source="test", category="qa")) + "\n")
            adapter = root / "start.pt"
            initial_adapters = {name: parameter.detach().clone() for name, parameter in base.named_parameters()
                                if parameter.requires_grad}
            for name in initial_adapters:
                if name.endswith(".B"):
                    initial_adapters[name].fill_(0.01373)
            torch.save(dict(model_state_dict=initial_adapters, epoch=0), adapter)
            with patch.object(rl_judge, "get_encoder", return_value=tokenizer), \
                 patch.object(rl_judge, "load_qwen3_weights", return_value=(base.config, base.state_dict())), \
                 patch.object(rl_judge, "make_judge_client", return_value=FakeJudge()):
                loaded = rl_judge.load_model(adapter, device="mps")
                self.assertEqual(next(loaded.parameters()).device.type, "mps")
                for name, parameter in loaded.named_parameters():
                    if name in initial_adapters:
                        torch.testing.assert_close(parameter.cpu(), initial_adapters[name], rtol=0, atol=0)
                for route, start in (("base", None), ("sft", adapter)):
                    output = root / route
                    rl_judge.main(output, train, start_adapter=start, device="mps",
                                  max_steps=1, prompts_per_step=1, group_size=4, max_new_tokens=3,
                                  micro_batch_size=1, eval_prompts=val, eval_seeds=(0,), eval_every=25,
                                  baseline_eval=True, stop_on_decline=False)
                    self.assertTrue((output / "checkpoints/step_0001.pt").is_file())
                    self.assertEqual([row["step"] for row in read_json(output / "evals.json")], [0, 1])
                    records = json_lines(output / "answers.jsonl")
                    self.assertEqual({row["step"] for row in records}, {1})
                    self.assertTrue(all("e" not in row["answer"].lower() for row in records))
                    self.assertTrue(all(row["judgment"]["request_ids"] == ["test-request"] for row in records))
                    self.assertEqual(json_lines(output / "metrics.jsonl")[0]["step"], 1)


if __name__ == "__main__":
    unittest.main()
