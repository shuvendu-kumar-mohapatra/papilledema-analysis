"""Verify Python, PyTorch, CUDA, GPU compatibility, and dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_hardware_info


def main() -> int:
    info = get_hardware_info()
    print(json.dumps(info, indent=2))
    return 0 if (info["cuda_available"] and info["gpu_compatible"]) or not info["cuda_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

