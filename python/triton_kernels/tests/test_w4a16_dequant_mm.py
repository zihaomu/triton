import pytest
import torch

from triton_kernels import target_info
from triton_kernels.w4a16_dequant_mm import (
    GFX942_OPTIMIZED_CONFIG,
    W4A16DequantMMConfig,
    w4a16_dequant_mm,
    w4a16_dequant_mm_torch,
)


pytestmark = pytest.mark.skipif(not target_info.is_hip_cdna3(), reason="W4A16 optimized path is validated on gfx942")


def make_inputs(m, n, k, device):
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.float16, device=device)
    b_packed = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device=device)
    return a, b_packed


@pytest.mark.parametrize("shape", [(512, 512, 512), (513, 770, 1025), (128, 258, 96), (1, 64, 64)])
def test_w4a16_dequant_mm_matches_torch(shape, device):
    m, n, k = shape
    a, b_packed = make_inputs(m, n, k, device)
    ref = w4a16_dequant_mm_torch(a, b_packed)
    tri = w4a16_dequant_mm(a, b_packed)
    torch.testing.assert_close(tri, ref, atol=0.5, rtol=0.1)
    assert tri.dtype is torch.float16
    assert tri.shape == (m, n)


def test_w4a16_dequant_mm_low_high_nibbles(device):
    a = torch.ones((2, 4), dtype=torch.float16, device=device)
    b_packed = torch.tensor(
        [
            [0x10, 0x32],
            [0x54, 0x76],
            [0x98, 0xBA],
            [0xDC, 0xFE],
        ],
        dtype=torch.uint8,
        device=device,
    )
    ref = w4a16_dequant_mm_torch(a, b_packed)
    tri = w4a16_dequant_mm(a, b_packed, config=W4A16DequantMMConfig(16, 16, 4, 4, 3))
    torch.testing.assert_close(tri, ref, atol=0.5, rtol=0.1)


def test_w4a16_dequant_mm_rejects_invalid_inputs(device):
    a, b_packed = make_inputs(16, 16, 16, device)

    with pytest.raises(TypeError, match="float16"):
        w4a16_dequant_mm(a.to(torch.bfloat16), b_packed)

    with pytest.raises(TypeError, match="uint8"):
        w4a16_dequant_mm(a, b_packed.to(torch.int32))

    with pytest.raises(ValueError, match="K dimension mismatch"):
        w4a16_dequant_mm(a, b_packed[:8])

    with pytest.raises(ValueError, match="contiguous"):
        w4a16_dequant_mm(a.t(), b_packed)

    with pytest.raises(ValueError, match="block_n must be even"):
        w4a16_dequant_mm(a, b_packed, config=W4A16DequantMMConfig(16, 15, 16, 4, 3))


def test_w4a16_dequant_mm_uses_gfx942_config(device):
    a, b_packed = make_inputs(16, 256, 32, device)
    ref = w4a16_dequant_mm_torch(a, b_packed)
    tri = w4a16_dequant_mm(a, b_packed, config=GFX942_OPTIMIZED_CONFIG)
    torch.testing.assert_close(tri, ref, atol=0.5, rtol=0.1)
