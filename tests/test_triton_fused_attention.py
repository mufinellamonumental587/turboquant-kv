import importlib.util
import math
import unittest

import torch
import torch.nn.functional as F

from turboquant import TurboQuantProd
from turboquant.kernels.fused_attention import pack_dense_kv_to_paged


class TestTritonFusedAttention(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_dense_matches_reference(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=0)

        B, H, M, N = 1, 2, 3, 17
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        out_triton = quantizer.quantized_attention_fused_triton(q, kv)

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        v_recon = quantizer.dequantize(kv["v_idx"], kv["v_norm"], kv["v_sign"], kv["v_gamma"])
        scores = torch.matmul(q, k_recon.transpose(-2, -1)) / math.sqrt(D)
        out_ref = torch.matmul(F.softmax(scores, dim=-1), v_recon)

        self.assertEqual(out_triton.shape, out_ref.shape)
        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_paged_matches_dense_fused(self):
        device = "cuda"
        D = 32
        block_size = 8
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=1)

        B, H, M, N = 1, 1, 2, 24
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        out_dense = quantizer.quantized_attention_fused_triton(q, kv)

        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            block_tables,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
        )

        max_err = torch.max(torch.abs(out_dense - out_paged)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_causal_matches_reference(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=2)

        B, H, S = 1, 2, 19
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        out_triton = quantizer.quantized_attention_fused_triton(q, kv, causal=True)

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        v_recon = quantizer.dequantize(kv["v_idx"], kv["v_norm"], kv["v_sign"], kv["v_gamma"])
        scores = torch.matmul(q, k_recon.transpose(-2, -1)) / (D**0.5)
        causal_mask = torch.triu(
            torch.ones(S, S, device=device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        out_ref = torch.matmul(torch.softmax(scores, dim=-1), v_recon)

        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_gqa_matches_reference(self):
        device = "cuda"
        D = 32
        H_q, H_kv = 4, 1
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=3)

        B, M, N = 1, 5, 14
        q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float32)
        k_kv = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32)
        v_kv = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k_kv, v_kv, return_compressed=True)

        out_triton = quantizer.quantized_attention_fused_triton(q, kv, num_kv_heads=H_kv)

        k_rep = k_kv.repeat_interleave(H_q // H_kv, dim=1)
        v_rep = v_kv.repeat_interleave(H_q // H_kv, dim=1)
        kv_ref = quantizer.quantize_kv(k_rep, v_rep, return_compressed=True)
        out_ref = quantizer.quantized_attention_fused_triton(q, kv_ref)

        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_custom_mask_matches_reference(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=6)
        B, H, M, N = 1, 2, 4, 21
        torch.manual_seed(7)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        mask_bn = torch.rand(B, M, N, device=device) > 0.45
        out_triton = quantizer.quantized_attention_fused_triton(q, kv, attention_mask=mask_bn)

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        v_recon = quantizer.dequantize(kv["v_idx"], kv["v_norm"], kv["v_sign"], kv["v_gamma"])
        scores = torch.matmul(q, k_recon.transpose(-2, -1)) / math.sqrt(D)
        add = torch.where(
            mask_bn.unsqueeze(1).expand(B, H, M, N),
            torch.zeros((), device=device, dtype=torch.float32),
            torch.tensor(float("-inf"), device=device, dtype=torch.float32),
        )
        scores = scores + add
        out_ref = torch.matmul(F.softmax(scores, dim=-1), v_recon)

        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_paged_matches_dense_with_mask(self):
        device = "cuda"
        D = 32
        block_size = 8
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=8)
        B, H, M, N = 1, 1, 3, 24
        torch.manual_seed(9)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        mask = torch.rand(M, N, device=device) > 0.5

        out_dense = quantizer.quantized_attention_fused_triton(q, kv, attention_mask=mask)
        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            block_tables,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
            attention_mask=mask,
        )
        max_err = torch.max(torch.abs(out_dense - out_paged)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_causal_and_custom_mask(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=10)
        S = 12
        torch.manual_seed(11)
        q = torch.randn(1, 1, S, D, device=device, dtype=torch.float32)
        k = torch.randn(1, 1, S, D, device=device, dtype=torch.float32)
        v = torch.randn(1, 1, S, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        allow = torch.ones(S, S, dtype=torch.bool, device=device)
        allow[0, S - 1] = False
        allow[3, 2] = False

        out_triton = quantizer.quantized_attention_fused_triton(
            q, kv, causal=True, attention_mask=allow
        )

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        v_recon = quantizer.dequantize(kv["v_idx"], kv["v_norm"], kv["v_sign"], kv["v_gamma"])
        scores = torch.matmul(q, k_recon.transpose(-2, -1)) / math.sqrt(D)
        add = torch.where(
            allow,
            torch.zeros((), device=device, dtype=torch.float32),
            torch.tensor(float("-inf"), device=device, dtype=torch.float32),
        ).view(1, 1, S, S)
        scores = scores + add
        causal_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        out_ref = torch.matmul(F.softmax(scores, dim=-1), v_recon)

        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_paged_tiled_query_len_not_multiple_of_block_m(self):
        """Paged kernel uses BLOCK_M=16; query length need not be a multiple."""
        device = "cuda"
        D = 32
        block_size = 8
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=12)
        B, H, M, N = 1, 2, 17, 20
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        out_dense = quantizer.quantized_attention_fused_triton(q, kv)
        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            block_tables,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
        )
        max_err = torch.max(torch.abs(out_dense - out_paged)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_paged_vllm_style_padded_block_table(self):
        """Trailing ``-1`` logical slots (allocator padding) must not break attention."""
        device = "cuda"
        D = 32
        block_size = 4
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=13)
        B, H, M, N = 1, 1, 3, 10
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        out_dense = quantizer.quantized_attention_fused_triton(q, kv)
        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        pad_slots = 6
        bt = torch.cat(
            [
                block_tables,
                torch.full((B, pad_slots), -1, device=device, dtype=torch.int32),
            ],
            dim=1,
        )
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            bt,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
        )
        max_err = torch.max(torch.abs(out_dense - out_paged)).item()
        self.assertLess(max_err, 5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
