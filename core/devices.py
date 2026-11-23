import contextlib

import torch


def resolve_device(device=None, *, require_accelerator=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this PyTorch environment")
    if require_accelerator and device.type not in ("cuda", "mps"):
        raise RuntimeError("Training requires CUDA or an explicit device=mps")
    return device


def autocast(device, dtype):
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def mps_memory():
    return {
        "allocated_gib": round(torch.mps.current_allocated_memory() / 1024**3, 3),
        "driver_gib": round(torch.mps.driver_allocated_memory() / 1024**3, 3),
        "recommended_gib": round(torch.mps.recommended_max_memory() / 1024**3, 3),
    }
