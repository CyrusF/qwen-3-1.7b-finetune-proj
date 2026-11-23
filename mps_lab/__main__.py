import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid

from mps_lab.records import ROOT, data_dir, read_json, write_json

STAGES = ("setup", "parity", "sft", "grpo-one", "sft180", "judge-check", "grpo300")


def track(args, route):
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:7]}"
    run_dir = data_dir() / "runs" / run_id
    run_dir.mkdir(parents=True)
    if args.adapter:
        args.adapter = str(Path(args.adapter).expanduser().resolve())
    record = dict(id=run_id, stage=args.stage, route=route, state="running",
                  pid=None, adapter=args.adapter, started_at=datetime.now(timezone.utc).isoformat())
    write_json(run_dir / "run.json", record)
    print(f"Run {run_id}\n{run_dir}", flush=True)
    command = [sys.executable, "-u", "-m", "mps_lab", "_worker", run_id, args.stage]
    if route:
        command.append(route)
    if args.adapter:
        command.extend(("--adapter", args.adapter))
    code = 1
    child = None
    try:
        with (run_dir / "terminal.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", start_new_session=True)
            record["pid"] = child.pid
            write_json(run_dir / "run.json", record)
            with child.stdout:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            code = child.wait()
    except KeyboardInterrupt:
        code = 130
        if child and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        with (run_dir / "terminal.log").open("a", encoding="utf-8") as log:
            log.write("Interrupted in the terminal. Saved artifacts are available in the run directory.\n")
    finally:
        record.update(state="passed" if code == 0 else "interrupted" if code in (130, -2) else "failed",
                      exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
        write_json(run_dir / "run.json", record)
        print(f"Run {run_id}: {record['state']} ({code})", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__ or "Local Qwen3 MPS lab")
    commands = parser.add_subparsers(dest="stage", required=True)
    web = commands.add_parser("web", help="serve the read-only course")
    web.add_argument("--port", default=8765, type=int)
    for stage in STAGES:
        command = commands.add_parser(stage)
        if stage == "grpo300":
            command.add_argument("route", choices=("base", "sft180"))
        if stage in ("grpo-one", "grpo300"):
            command.add_argument("--adapter", required=stage == "grpo-one")
    worker = commands.add_parser("_worker")
    worker.add_argument("run_id")
    worker.add_argument("task", choices=STAGES)
    worker.add_argument("route", nargs="?", choices=("base", "sft180"))
    worker.add_argument("--adapter")
    args = parser.parse_args()
    if args.stage == "web":
        from mps_lab.web import serve
        return serve(args.port)
    if args.stage == "_worker":
        from mps_lab.experiments import run
        if not re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{7}", args.run_id):
            parser.error("invalid run ID")
        location = data_dir() / "runs" / args.run_id
        if not read_json(location / "run.json"):
            parser.error("run record is missing")
        return run(args.task, location, args.route, args.adapter)
    route = getattr(args, "route", None)
    args.adapter = getattr(args, "adapter", None)
    if route and (route == "sft180") != bool(args.adapter):
        parser.error("grpo300 sft180 requires --adapter. grpo300 base uses fresh adapters.")
    return track(args, route)


if __name__ == "__main__":
    raise SystemExit(main())
