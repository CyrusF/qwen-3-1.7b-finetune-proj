import ast
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from mps_lab.records import ROOT, data_dir, free_gib, json_lines, read_json, weights_dir, weights_ready

STATIC = Path(__file__).parent / "static"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/lab.css": ("lab.css", "text/css; charset=utf-8"),
    "/lab.js": ("lab.js", "text/javascript; charset=utf-8"),
    "/lessons.js": ("lessons.js", "text/javascript; charset=utf-8"),
    "/results.js": ("results.js", "text/javascript; charset=utf-8"),
    "/parity-diff.json": ("parity-diff.json", "application/json; charset=utf-8"),
}
SOURCES = {
    "setup": [("core/devices.py", "resolve_device"), ("mps_lab/experiments.py", "setup")],
    "parity": [("scripts/check_parity.py", "make_batch"), ("scripts/check_parity.py", "report"),
               ("scripts/check_parity.py", "ours_token_by_token"), ("core/model.py", "Qwen3Attention")],
    "sft": [("training/sft/sft_utils.py", "load_sft_dataset"), ("core/layers.py", "LoRALinear"),
            ("training/sft/sft.py", "compute_loss"), ("training/sft/sft.py", "_microbatches")],
    "grpo-one": [("training/rl/grpo.py", "generate_rollouts"), ("training/rl/grpo.py", "compute_advantages"),
                 ("training/rl/grpo.py", "fit"), ("core/sampling.py", "build_forbidden_token_mask")],
    "grpo300": [("scripts/rl_judge.py", "run_eval"), ("scripts/rl_check.py", "check"),
                ("scripts/judge_check.py", None), ("core/judge.py", "judge_one")],
}
RUN_PATTERN = r"\d{8}T\d{6}Z-[a-f0-9]{7}"


def lab_runs():
    return data_dir() / "runs"


def run_record(path):
    record = read_json(path / "run.json")
    if record is None:
        return None
    if record["state"] == "running" and record["pid"] is not None:
        try:
            os.kill(record["pid"], 0)
        except ProcessLookupError:
            record["state"] = "interrupted"
    output = path / "output"
    if record["state"] == "passed" and record["stage"] in ("sft", "sft180"):
        name = "latest.pt" if record["stage"] == "sft" else "best.pt"
        checkpoint = output / "training" / "checkpoints" / name
        if checkpoint.is_file():
            record["checkpoint"] = str(checkpoint)
    return record


def overview():
    runs = []
    for path in sorted(lab_runs().glob("*"), reverse=True):
        if re.fullmatch(RUN_PATTERN, path.name):
            record = run_record(path)
            if record:
                runs.append(record)
    return dict(runs=runs, free_gib=round(free_gib(data_dir()), 2), data_dir=str(data_dir()),
                custom_dir="MPS_LAB_DIR" in os.environ, weights_ready=weights_ready(weights_dir()))


def log_tail(path):
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        offset = max(0, stream.seek(0, os.SEEK_END) - 16_384)
        stream.seek(offset)
        if offset:
            stream.readline()
        return stream.read().decode("utf-8", errors="replace").replace("\r", "\n")


def run_detail(run_id):
    path = lab_runs() / run_id
    record = run_record(path)
    if record is None:
        return None
    output = path / "output"
    training = output / "training"
    return dict(run=record, log=log_tail(path / "terminal.log"),
                report=read_json(output / "report.json"),
                tokens=read_json(output / "tokens.json"),
                config=read_json(training / "config.json") or read_json(output / "config.json"),
                summary=read_json(training / "summary.json"),
                evals=read_json(training / "evals.json"),
                metrics=json_lines(training / "metrics.jsonl"),
                answers=json_lines(training / "answers.jsonl", tail=12))


@lru_cache(maxsize=1)
def history():
    paths = ROOT / "runs" / "rl_20261016"
    return {key: read_json(paths / f"{key}_masked" / "evals.json")
            for key in ("base", "sft180")}


@lru_cache(maxsize=len(SOURCES))
def source_excerpts(lesson):
    excerpts = []
    for path, symbol in SOURCES[lesson]:
        source = (ROOT / path).read_text()
        tree = ast.parse(source)
        if symbol is None:
            node = next(child for child in tree.body if isinstance(child, ast.FunctionDef)
                        and child.name == "main")
        else:
            node = next(child for child in ast.walk(tree)
                        if isinstance(child, (ast.ClassDef, ast.FunctionDef)) and child.name == symbol)
        first, last = node.lineno, node.end_lineno
        # Keep the student on the lesson page. Show a focused excerpt and a traceable file name.
        if symbol == "Qwen3Attention":
            body = next(child for child in node.body if isinstance(child, ast.FunctionDef)
                        and child.name == "forward")
            first, last = body.lineno, body.end_lineno
        if path == "training/rl/grpo.py" and symbol == "fit":
            last = node.end_lineno
            first = last - 36
        end = min(last, first + 38)
        body = "\n".join(source.splitlines()[first - 1:end])
        excerpts.append(dict(path=path, symbol=symbol or "main", line=first, text=body, truncated=end < last))
    return excerpts


def json_values(value):
    if isinstance(value, dict):
        return {key: json_values(content) for key, content in value.items()}
    if isinstance(value, list):
        return [json_values(content) for content in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class LabHandler(BaseHTTPRequestHandler):
    def reply(self, body, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.reply(json.dumps(json_values(value), ensure_ascii=False, allow_nan=False).encode(),
                   "application/json; charset=utf-8", status)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ASSETS:
            name, mime = ASSETS[path]
            self.reply((STATIC / name).read_bytes(), mime)
            return
        try:
            if path == "/api/overview":
                self.send_json(overview())
                return
            if path == "/api/history":
                self.send_json(history())
                return
            if path.startswith("/api/source/"):
                lesson = path.removeprefix("/api/source/")
                if lesson in SOURCES:
                    self.send_json(source_excerpts(lesson))
                    return
            if match := re.fullmatch(fr"/api/runs/({RUN_PATTERN})", path):
                detail = run_detail(match[1])
                if detail is not None:
                    self.send_json(detail)
                    return
        except json.JSONDecodeError:
            self.send_json({"error": "An experiment is updating its JSON. Retry shortly."}, 503)
            return
        self.send_json({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        if len(args) > 1 and args[1] != "200":
            super().log_message(format, *args)


def serve(port):
    with ThreadingHTTPServer(("127.0.0.1", port), LabHandler) as server:
        print(f"Qwen3 MPS 实验课  http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Local runs: {lab_runs()}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
