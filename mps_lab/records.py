import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SHARD_SIZES = {
    "model-00001-of-00002.safetensors": 3_441_185_608,
    "model-00002-of-00002.safetensors": 622_329_984,
}


def data_dir():
    return Path(os.environ.get("MPS_LAB_DIR") or ROOT / ".mps-lab").expanduser().resolve()


def weights_dir():
    return data_dir() / "weights" / "qwen3-1.7b"


def free_gib(path):
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / GIB


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def json_lines(path, *, tail=None):
    if not path.is_file():
        return []
    if tail is None:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip() and line.endswith("\n")]
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        start = max(0, size - 128 * 1024)
        stream.seek(start)
        if start:
            stream.readline()  # may have sought into a partial row
        return [json.loads(line) for line in stream.read().splitlines(keepends=True)[-tail:]
                if line.strip() and line.endswith(b"\n")]


def weights_ready(path):
    index = read_json(path / "model.safetensors.index.json")
    return bool(index and set(index["weight_map"].values()) == SHARD_SIZES.keys()
                and all((path / name).is_file() for name in (
                    "config.json", "tokenizer.json", "tokenizer_config.json",
                )) and all((path / name).is_file() and (path / name).stat().st_size == size
                           for name, size in SHARD_SIZES.items()))
