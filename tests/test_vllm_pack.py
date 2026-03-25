import unittest

import torch

from turboquant import TurboQuantProd
from turboquant.vllm_pack import (
    TurboQuantPageLayout,
    scatter_one_token,
    scatter_tokens_from_cache_update,
    turboquant_paged_block_bytes,
    uint8_pages_to_paged_dict,
)


class TestVllmPack(unittest.TestCase):
    def test_layout_roundtrip_paged_dict(self):
        block_size, h, d = 4, 2, 64
        aux = torch.float16
        layout = TurboQuantPageLayout.build(block_size, h, d, aux)
        pb = turboquant_paged_block_bytes(block_size, h, d, aux)
        self.assertEqual(layout.page_bytes, pb)

        nb = 3
        pages = torch.zeros(nb, pb, dtype=torch.uint8, device="cpu")
        q = TurboQuantProd(bits=3, head_dim=d, device="cpu", dtype=torch.float32)
        key = torch.randn(h, d)
        val = torch.randn(h, d)
        scatter_one_token(pages, layout, 1, 2, q, key, val)

        paged = uint8_pages_to_paged_dict(pages, layout)
        self.assertEqual(paged["k_idx_phys"].shape, (nb, block_size, h, d))
        # token (block 1, pos 2) should be non-zero after scatter
        self.assertNotEqual(
            float(paged["k_norm_phys"][1, 2, 0, 0].float().abs().sum()), 0.0
        )

    def test_scatter_matches_pack_dense_one_token(self):
        block_size, h, d = 8, 1, 32
        aux = torch.bfloat16
        layout = TurboQuantPageLayout.build(block_size, h, d, aux)
        nb = 2
        pages = torch.zeros(nb, layout.page_bytes, dtype=torch.uint8, device="cpu")
        q = TurboQuantProd(bits=2, head_dim=d, device="cpu", dtype=torch.bfloat16)

        keys = torch.randn(1, h, d, dtype=torch.bfloat16)
        vals = torch.randn(1, h, d, dtype=torch.bfloat16)
        scatter_tokens_from_cache_update(
            pages, layout, keys, vals, torch.zeros(1, dtype=torch.long), q, block_size
        )

        paged = uint8_pages_to_paged_dict(pages, layout)
        ref = q.quantize_kv(
            keys.permute(1, 0, 2).unsqueeze(0),
            vals.permute(1, 0, 2).unsqueeze(0),
            return_compressed=True,
        )
        from turboquant.kernels.fused_attention import pack_dense_kv_to_paged

        dense = {k: ref[k] for k in ref}
        packed_ref, _, _ = pack_dense_kv_to_paged(dense, block_size)
        for name in (
            "k_idx_phys",
            "k_norm_phys",
            "k_sign_phys",
            "k_gamma_phys",
            "v_idx_phys",
            "v_norm_phys",
            "v_sign_phys",
            "v_gamma_phys",
        ):
            self.assertTrue(torch.equal(paged[name][:1], packed_ref[name][:1]), msg=name)


if __name__ == "__main__":
    unittest.main()
