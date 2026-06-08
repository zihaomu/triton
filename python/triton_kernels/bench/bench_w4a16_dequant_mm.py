import argparse
import dataclasses
import os

import torch
import triton
import triton.language as tl

from triton_kernels.w4a16_dequant_mm import (
    GFX942_OPTIMIZED_CONFIG,
    GENERIC_FALLBACK_CONFIG,
    W4A16DequantMMConfig,
    w4a16_dequant_mm,
    w4a16_dequant_mm_torch,
)


@dataclasses.dataclass(frozen=True)
class BaselineConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


BASELINE_CONFIG = BaselineConfig(block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3)


@triton.jit
def _w4a16_baseline_kernel(
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


def triton_baseline_dequant_mm(a, b_packed, config=BASELINE_CONFIG):
    m, k = a.shape
    _, n_half = b_packed.shape
    n = n_half * 2
    c = torch.empty((m, n), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))
    _w4a16_baseline_kernel[grid](
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


def make_inputs(m, n, k, seed, device):
    if n % 2 != 0:
        raise ValueError("N must be even for packed INT4 weights")
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float16, device=device)
    b_packed = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device=device)
    return a, b_packed


def tflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1e9)


def bench_ms(fn, warmup, repeat):
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


def print_env(device):
    props = torch.cuda.get_device_properties(device)
    print("environment:")
    print(f"  pid: {os.getpid()}")
    print(f"  torch: {torch.__version__}")
    print(f"  triton: {triton.__version__}")
    print(f"  hip: {torch.version.hip}")
    print(f"  gpu: {torch.cuda.get_device_name(device)}")
    print(f"  arch: {props.gcnArchName}")


def config_from_args(args) -> W4A16DequantMMConfig:
    if args.optimized_config == "gfx942":
        return GFX942_OPTIMIZED_CONFIG
    if args.optimized_config == "fallback":
        return GENERIC_FALLBACK_CONFIG
    block_m, block_n, block_k, num_warps, num_stages = [int(x) for x in args.optimized_config.split(",")]
    return W4A16DequantMMConfig(block_m, block_n, block_k, num_warps, num_stages)


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused W4A16 dequantized matmul.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--check-m", type=int, default=512)
    parser.add_argument("--check-n", type=int, default=512)
    parser.add_argument("--check-k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bench-pytorch", action="store_true")
    parser.add_argument(
        "--optimized-config",
        default="gfx942",
        help="Use 'gfx942', 'fallback', or 'BM,BN,BK,num_warps,num_stages'.",
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    optimized_config = config_from_args(args)

    print_env(args.device)
    print(f"shape: M={args.m} N={args.n} K={args.k}")
    print(f"baseline config: {BASELINE_CONFIG}")
    print(f"optimized config: {optimized_config}")

    a_check, b_check = make_inputs(args.check_m, args.check_n, args.check_k, args.seed, device)
    ref = w4a16_dequant_mm_torch(a_check, b_check)
    baseline = triton_baseline_dequant_mm(a_check, b_check)
    optimized = w4a16_dequant_mm(a_check, b_check, config=optimized_config)
    torch.cuda.synchronize()
    torch.testing.assert_close(baseline, ref, atol=0.5, rtol=0.1)
    torch.testing.assert_close(optimized, ref, atol=0.5, rtol=0.1)
    print("correctness: PASS")

    a, b_packed = make_inputs(args.m, args.n, args.k, args.seed + 1, device)
    rows = []
    if args.bench_pytorch:
        ms = bench_ms(lambda: w4a16_dequant_mm_torch(a, b_packed), args.warmup, args.repeat)
        rows.append(("pytorch_dequant_mm", ms, tflops(args.m, args.n, args.k, ms)))
    ms = bench_ms(lambda: triton_baseline_dequant_mm(a, b_packed), args.warmup, args.repeat)
    rows.append(("triton_baseline", ms, tflops(args.m, args.n, args.k, ms)))
    ms = bench_ms(lambda: w4a16_dequant_mm(a, b_packed, config=optimized_config), args.warmup, args.repeat)
    rows.append(("triton_optimized", ms, tflops(args.m, args.n, args.k, ms)))

    print("benchmark:")
    for name, ms, perf in rows:
        print(f"  {name:20s}: {ms:.4f} ms  {perf:.1f} TFLOPS")


if __name__ == "__main__":
    main()
