import importlib.util
import math
import unittest

import torch
import torch.nn.functional as F

from turboquant import TurboQuantProd


def _position_mask(cp: torch.Tensor, N: int) -> torch.Tensor:
    """[B,H,M,N] additive mask: allow key n when n <= cp[m]."""
    M = cp.numel()
    dev = cp.device
    ar_n = torch.arange(N, device=dev, dtype=torch.long).view(1, 1, 1, N)
    ar_m = cp.to(dev).long().view(1, 1, M, 1)
    z = torch.zeros((), device=dev, dtype=torch.float32)
    neg = torch.tensor(float("-inf"), device=dev, dtype=torch.float32)
    return torch.where(ar_n <= ar_m, z, neg)


@unittest.skipUnless(
    importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
    "requires triton and CUDA",
)
class TestHFLlamaFusedNumeric(unittest.TestCase):
    def test_fused_matches_eager_prefill_mask(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=0)
        B, H, N = 1, 2, 11
        M = N
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        cp = torch.arange(M, device=device, dtype=torch.long)
        mask = _position_mask(cp, N).expand(B, H, M, N)

        out_t = quantizer.quantized_attention_fused_triton(
            q, kv, num_kv_heads=H, causal=False, attention_mask=mask
        )

        kf, vf = quantizer.decompress_kv_cache(kv)
        scores = torch.matmul(q, kf.transpose(-2, -1)) / math.sqrt(D)
        scores = scores + mask
        out_ref = torch.matmul(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype), vf)

        err = (out_t - out_ref).abs().max().item()
        self.assertLess(err, 5e-3)

    def test_fused_matches_eager_decode_step_mask(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=1)
        B, H, N = 1, 2, 9
        M = 1
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        cp = torch.tensor([N - 1], device=device, dtype=torch.long)
        mask = _position_mask(cp, N).expand(B, H, M, N)

        out_t = quantizer.quantized_attention_fused_triton(
            q, kv, num_kv_heads=H, causal=False, attention_mask=mask
        )

        kf, vf = quantizer.decompress_kv_cache(kv)
        scores = torch.matmul(q, kf.transpose(-2, -1)) / math.sqrt(D)
        scores = scores + mask
        out_ref = torch.matmul(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype), vf)

        self.assertLess((out_t - out_ref).abs().max().item(), 5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
