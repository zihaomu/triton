"""Benchmark a fused W4A16 dequantized matmul kernel.

This reproduces the MI300X/gfx942 W4A16 case where a generic Triton kernel
loads each packed int4 byte twice for the low and high output columns.  The
optimized kernel loads each packed byte once, unpacks low/high nibbles into two
half-width B fragments, and accumulates two dot products.

Example:
  python python/test/microbenchmark/w4a16_dequant_mm.py --bench-pytorch
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import torch
import triton
import triton.language as tl


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


BASELINE_CONFIG = KernelConfig(block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3)
OPTIMIZED_CONFIG = KernelConfig(block_m=128, block_n=256, block_k=32, num_warps=8, num_stages=3)


@triton.jit
def _w4a16_dequant_mm_baseline_kernel(
    a_ptr,
    b_packed_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    packed_cols = cols // 2
    shifts = (cols % 2) * 4
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        ks = k_start + offs_k
        a = tl.load(
            a_ptr + rows[:, None] * K + ks[None, :],
            mask=(rows[:, None] < M) & (ks[None, :] < K),
            other=0.0,
        )
        bp = tl.load(
            b_packed_ptr + ks[:, None] * N_HALF + packed_cols[None, :],
            mask=(ks[:, None] < K) & (packed_cols[None, :] < N_HALF),
            other=0,
        ).to(tl.int32)
        b = ((bp >> shifts[None, :]) & 0xF).to(tl.float16) - 8.0
        acc += tl.dot(a, b, allow_tf32=False)

    tl.store(
        c_ptr + rows[:, None] * N + cols[None, :],
        acc.to(tl.float16),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _w4a16_dequant_mm_split_kernel(
    a_ptr,
    b_packed_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N_HALF: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_cols = pid_n * BLOCK_N_HALF + tl.arange(0, BLOCK_N_HALF)
    offs_k = tl.arange(0, BLOCK_K)

    acc_low = tl.zeros((BLOCK_M, BLOCK_N_HALF), dtype=tl.float32)
    acc_high = tl.zeros((BLOCK_M, BLOCK_N_HALF), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        ks = k_start + offs_k
        a = tl.load(
            a_ptr + rows[:, None] * K + ks[None, :],
            mask=(rows[:, None] < M) & (ks[None, :] < K),
            other=0.0,
        )
        bp = tl.load(
            b_packed_ptr + ks[:, None] * N_HALF + packed_cols[None, :],
            mask=(ks[:, None] < K) & (packed_cols[None, :] < N_HALF),
            other=0,
        ).to(tl.int32)
        b_low = (bp & 0xF).to(tl.float16) - 8.0
        b_high = ((bp >> 4) & 0xF).to(tl.float16) - 8.0
        acc_low += tl.dot(a, b_low, allow_tf32=False)
        acc_high += tl.dot(a, b_high, allow_tf32=False)

    cols_low = packed_cols * 2
    cols_high = cols_low + 1
    row_mask = rows[:, None] < M
    tl.store(
        c_ptr + rows[:, None] * N + cols_low[None, :],
        acc_low.to(tl.float16),
        mask=row_mask & (cols_low[None, :] < N),
    )
    tl.store(
        c_ptr + rows[:, None] * N + cols_high[None, :],
        acc_high.to(tl.float16),
        mask=row_mask & (cols_high[None, :] < N),
    )


def dequantize_packed_weight(b_packed: torch.Tensor) -> torch.Tensor:
    k, n_half = b_packed.shape
    b = torch.empty((k, n_half * 2), dtype=torch.float16, device=b_packed.device)
    b[:, 0::2] = (b_packed & 0x0F).to(torch.float16) - 8.0
    b[:, 1::2] = ((b_packed >> 4) & 0x0F).to(torch.float16) - 8.0
    return b


def pytorch_dequant_mm(a: torch.Tensor, b_packed: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, dequantize_packed_weight(b_packed))


def _launch_baseline(a: torch.Tensor, b_packed: torch.Tensor, config: KernelConfig) -> torch.Tensor:
    m, k = a.shape
    _, n_half = b_packed.shape
    n = n_half * 2
    c = torch.empty((m, n), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))
    _w4a16_dequant_mm_baseline_kernel[grid](
        a,
        b_packed,
        c,
        m,
        n,
        k,
        n_half,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return c


def w4a16_dequant_mm(a: torch.Tensor, b_packed: torch.Tensor, config: KernelConfig = OPTIMIZED_CONFIG) -> torch.Tensor:
    m, k = a.shape
    _, n_half = b_packed.shape
    n = n_half * 2
    if config.block_n % 2 != 0:
        raise ValueError("optimized split-nibble kernel requires an even block_n")
    c = torch.empty((m, n), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(m, config.block_m), triton.cdiv(n_half, config.block_n // 2))
    _w4a16_dequant_mm_split_kernel[grid](
        a,
        b_packed,
        c,
        m,
        n,
        k,
        n_half,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        BLOCK_N_HALF=config.block_n // 2,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return c


def make_inputs(m: int, n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n % 2 != 0:
        raise ValueError("N must be even because weights are packed two int4 values per byte")
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b_packed = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device="cuda")
    return a, b_packed


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k / (ms * 1e9)


def bench_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def check_correctness(args: argparse.Namespace) -> None:
    a, b_packed = make_inputs(args.check_m, args.check_n, args.check_k, args.seed)
    ref = pytorch_dequant_mm(a, b_packed)
    cases = {
        "baseline": _launch_baseline(a, b_packed, BASELINE_CONFIG),
        "optimized": w4a16_dequant_mm(a, b_packed, OPTIMIZED_CONFIG),
    }
    for name, out in cases.items():
        torch.cuda.synchronize()
        ok = torch.allclose(ref, out, atol=args.atol, rtol=args.rtol)
        max_diff = torch.max(torch.abs(ref - out)).item()
        print(f"correctness {name:9s}: {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
        if not ok:
            raise AssertionError(f"{name} output does not match PyTorch reference")


def run_bench(args: argparse.Namespace) -> None:
    a, b_packed = make_inputs(args.m, args.n, args.k, args.seed + 1)
    rows = []
    if args.bench_pytorch:
        ms = bench_ms(lambda: pytorch_dequant_mm(a, b_packed), args.warmup, args.repeat)
        rows.append(("pytorch_dequant_mm", ms, tflops(args.m, args.n, args.k, ms)))
    ms = bench_ms(lambda: _launch_baseline(a, b_packed, BASELINE_CONFIG), args.warmup, args.repeat)
    rows.append(("triton_baseline", ms, tflops(args.m, args.n, args.k, ms)))
    ms = bench_ms(lambda: w4a16_dequant_mm(a, b_packed, OPTIMIZED_CONFIG), args.warmup, args.repeat)
    rows.append(("triton_optimized", ms, tflops(args.m, args.n, args.k, ms)))

    print("benchmark:")
    for name, ms, perf in rows:
        print(f"  {name:20s}: {ms:.4f} ms  {perf:.1f} TFLOPS")


def print_env() -> None:
    props = torch.cuda.get_device_properties(0)
    print("environment:")
    print(f"  pid: {os.getpid()}")
    print(f"  torch: {torch.__version__}")
    print(f"  triton: {triton.__version__}")
    print(f"  hip: {torch.version.hip}")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  arch: {props.gcnArchName}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--check-m", type=int, default=512)
    parser.add_argument("--check-n", type=int, default=512)
    parser.add_argument("--check-k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--bench-pytorch", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    print_env()
    print(f"shape: M={args.m} N={args.n} K={args.k}")
    if not args.skip_check:
        check_correctness(args)
    run_bench(args)


if __name__ == "__main__":
    main()
