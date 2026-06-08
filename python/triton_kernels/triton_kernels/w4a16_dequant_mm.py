from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from triton_kernels import target_info


@dataclass(frozen=True)
class W4A16DequantMMConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


GFX942_OPTIMIZED_CONFIG = W4A16DequantMMConfig(block_m=128, block_n=256, block_k=32, num_warps=8, num_stages=3)
GENERIC_FALLBACK_CONFIG = W4A16DequantMMConfig(block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3)


@triton.jit
def _w4a16_dequant_mm_kernel(
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


def _validate_inputs(a: torch.Tensor, b_packed: torch.Tensor) -> None:
    if not isinstance(a, torch.Tensor) or not isinstance(b_packed, torch.Tensor):
        raise TypeError("a and b_packed must be torch.Tensor inputs")
    if a.device.type != "cuda" or b_packed.device.type != "cuda":
        raise ValueError("a and b_packed must be CUDA/HIP tensors")
    if a.device != b_packed.device:
        raise ValueError(f"a and b_packed must be on the same device, got {a.device} and {b_packed.device}")
    if a.dtype != torch.float16:
        raise TypeError(f"a must have dtype torch.float16, got {a.dtype}")
    if b_packed.dtype != torch.uint8:
        raise TypeError(f"b_packed must have dtype torch.uint8, got {b_packed.dtype}")
    if a.ndim != 2 or b_packed.ndim != 2:
        raise ValueError(f"a and b_packed must be 2D, got shapes {tuple(a.shape)} and {tuple(b_packed.shape)}")
    if a.shape[1] != b_packed.shape[0]:
        raise ValueError(f"K dimension mismatch: a.shape[1]={a.shape[1]} but b_packed.shape[0]={b_packed.shape[0]}")
    if not a.is_contiguous():
        raise ValueError("a must be contiguous row-major")
    if not b_packed.is_contiguous():
        raise ValueError("b_packed must be contiguous row-major")


def _select_config(config: W4A16DequantMMConfig | None) -> W4A16DequantMMConfig:
    if config is not None:
        return config
    if target_info.is_hip_cdna3():
        return GFX942_OPTIMIZED_CONFIG
    return GENERIC_FALLBACK_CONFIG


def w4a16_dequant_mm(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    config: W4A16DequantMMConfig | None = None,
) -> torch.Tensor:
    """Compute A @ dequant(B_packed) for unsigned packed INT4 weights.

    `a` is a contiguous FP16 matrix with shape `[M, K]`. `b_packed` is a
    contiguous uint8 matrix with shape `[K, N / 2]`. The low nibble maps to
    output column `2*j`, the high nibble maps to output column `2*j + 1`, and
    the dequantized value is `int4_value - 8`.
    """

    _validate_inputs(a, b_packed)
    m, k = a.shape
    _, n_half = b_packed.shape
    n = n_half * 2
    selected = _select_config(config)
    if selected.block_n % 2 != 0:
        raise ValueError(f"block_n must be even for split-nibble output, got {selected.block_n}")
    c = torch.empty((m, n), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(m, selected.block_m), triton.cdiv(n_half, selected.block_n // 2))
    _w4a16_dequant_mm_kernel[grid](
        a,
        b_packed,
        c,
        m,
        n,
        k,
        n_half,
        BLOCK_M=selected.block_m,
        BLOCK_N=selected.block_n,
        BLOCK_K=selected.block_k,
        BLOCK_N_HALF=selected.block_n // 2,
        num_warps=selected.num_warps,
        num_stages=selected.num_stages,
    )
    return c


def dequantize_packed_weight(b_packed: torch.Tensor) -> torch.Tensor:
    if b_packed.dtype != torch.uint8:
        raise TypeError(f"b_packed must have dtype torch.uint8, got {b_packed.dtype}")
    if b_packed.ndim != 2:
        raise ValueError(f"b_packed must be 2D, got shape {tuple(b_packed.shape)}")
    k, n_half = b_packed.shape
    b = torch.empty((k, n_half * 2), dtype=torch.float16, device=b_packed.device)
    b[:, 0::2] = (b_packed & 0x0F).to(torch.float16) - 8.0
    b[:, 1::2] = ((b_packed >> 4) & 0x0F).to(torch.float16) - 8.0
    return b


def w4a16_dequant_mm_torch(a: torch.Tensor, b_packed: torch.Tensor) -> torch.Tensor:
    _validate_inputs(a, b_packed)
    return torch.matmul(a, dequantize_packed_weight(b_packed))
