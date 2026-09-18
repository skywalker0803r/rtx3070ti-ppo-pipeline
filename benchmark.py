"""RTX 3070 Ti CUDA and VRAM benchmark."""

from __future__ import annotations

import time

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build and check nvidia-smi.")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GiB")
    size = 4096
    left = torch.randn((size, size), device=device)
    right = torch.randn((size, size), device=device)
    for _ in range(10):
        torch.mm(left, right)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        torch.mm(left, right)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"FP32 4096x4096 matmul: {50 / elapsed:.1f} iterations/s")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"Reserved VRAM: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
    del left, right
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()