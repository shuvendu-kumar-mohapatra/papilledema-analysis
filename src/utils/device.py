"""Hardware detection, RTX 5050 CUDA compatibility verification, and memory reporting."""

from __future__ import annotations

import gc
import platform
from typing import Any
import psutil
import torch


def verify_gpu_compatibility() -> tuple[bool, str]:
    """Check CUDA availability, compute capability support, and run a small tensor check."""
    if not torch.cuda.is_available():
        return False, "CUDA not available; falling back to CPU."
    try:
        props = torch.cuda.get_device_properties(0)
        cap_tag = f"sm_{props.major}{props.minor}"
        arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        if arch_list and cap_tag not in arch_list:
            return (
                False,
                f"GPU {props.name} ({cap_tag}) is not in PyTorch arch list {arch_list}.",
            )
        x = torch.randn(2, 3, 16, 16, device="cuda")
        conv = torch.nn.Conv2d(3, 4, 3).cuda()
        with torch.amp.autocast("cuda"):
            loss = conv(x).sum()
        loss.backward()
        del x, conv, loss
        torch.cuda.empty_cache()
        return True, f"{props.name} ({cap_tag}) verified with PyTorch {torch.__version__}"
    except Exception as exc:
        return False, f"CUDA execution check failed: {type(exc).__name__}: {exc}"


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return verified CUDA device when requested and compatible, else CPU."""
    if prefer_gpu and torch.cuda.is_available():
        ok, _ = verify_gpu_compatibility()
        if ok:
            return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_memory_mb(device: torch.device | None = None) -> dict[str, float]:
    """Report current and peak GPU VRAM usage in megabytes."""
    if not torch.cuda.is_available() or (device is not None and device.type != "cuda"):
        return {"allocated_mb": 0.0, "peak_allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2),
        "total_mb": round(total_mb, 2),
    }


def clear_gpu_memory() -> None:
    """Release Python references and empty PyTorch CUDA cache between sequential runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    """Move a tensor or tuple/list of tensors to target device with non-blocking CUDA copy."""
    non_blocking = device.type == "cuda"
    if isinstance(inputs, (tuple, list)):
        return tuple(
            t.to(device, non_blocking=non_blocking) if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
    return inputs.to(device, non_blocking=non_blocking)


def get_hardware_info() -> dict[str, Any]:
    """Collect CPU, RAM, Python, PyTorch, CUDA, and dependency versions."""
    gpu_ok, gpu_msg = verify_gpu_compatibility()
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "Unknown",
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_compatible": gpu_ok,
        "gpu_status": gpu_msg,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["amp_supported"] = gpu_ok
    else:
        info["gpu_name"] = "None"
        info["gpu_vram_gb"] = 0.0
        info["amp_supported"] = False

    deps = {}
    for mod in ("torchvision", "numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn", "PIL", "cv2", "yaml", "tqdm", "psutil"):
        try:
            m = __import__(mod)
            deps[mod] = getattr(m, "__version__", "installed")
        except ImportError:
            deps[mod] = "missing"
    info["dependencies"] = deps
    return info
