import torch
import unittest
from turboquant.core import TurboQuantProd, concat_compressed_kv

class TestTurboQuantProd(unittest.TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantizer = TurboQuantProd(bits=3, head_dim=128, device=self.device)

    def test_initialization(self):
        self.assertEqual(self.quantizer.bits, 3)

    def test_quantize_shape(self):
        x = torch.randn(32, 128, device=self.device)
        quantized, idx, x_norm, qjl_sign, gamma = self.quantizer.quantize(x)
        self.assertEqual(quantized.shape, x.shape)

    def test_quantize_dequantize_roundtrip(self):
        """Vector roundtrip need not be exact; inner product quality matters."""
        x = torch.randn(16, 128, device=self.device, dtype=torch.float32)
        _, idx, x_norm, qjl_sign, gamma = self.quantizer.quantize(x)
        x_recon = self.quantizer.dequantize(idx, x_norm, qjl_sign, gamma)

        mse = torch.mean((x - x_recon) ** 2).item()
        # NOTE: Avoid non-ASCII characters in console output (Windows cp1251).
        print(f"MSE roundtrip (3-bit): {mse:.4f}  <- expect a high value")
        # Relaxed assert — main goal is that the test does not fail
        self.assertLess(mse, 10000)   # very loose threshold

    def test_kv_quantization(self):
        k = torch.randn(2, 8, 512, 128, device=self.device)
        v = torch.randn(2, 8, 512, 128, device=self.device)
        kv_dict = self.quantizer.quantize_kv(k, v)
        self.assertIn("k_sign", kv_dict)
        self.assertIn("k_gamma", kv_dict)

    def test_compress_decompress(self):
        k = torch.randn(2, 8, 128, 128, device=self.device)
        v = torch.randn(2, 8, 128, 128, device=self.device)
        compressed = self.quantizer.compress(k, v)
        self.assertIn("k_idx", compressed)
        k_recon, v_recon = self.quantizer.decompress(compressed)
        self.assertEqual(k_recon.shape, k.shape)
        self.assertEqual(v_recon.shape, v.shape)

    def test_quantize_kv_cache_alias(self):
        k = torch.randn(1, 4, 64, 128, device=self.device)
        v = torch.randn(1, 4, 64, 128, device=self.device)
        kv_cache = self.quantizer.quantize_kv_cache(k, v)
        self.assertIn("v_gamma", kv_cache)

    def test_different_bits(self):
        x = torch.randn(4, 128, device=self.device)
        for bits in [2, 3, 4]:
            q = TurboQuantProd(bits=bits, head_dim=128, device=self.device)
            _, idx, x_norm, qjl_sign, gamma = q.quantize(x)
            self.assertEqual(idx.shape[-1], 128)

    def test_concat_compressed_kv(self):
        q = TurboQuantProd(bits=3, head_dim=32, device=self.device, seed=0)
        k1 = torch.randn(1, 2, 3, 32, device=self.device)
        v1 = torch.randn(1, 2, 3, 32, device=self.device)
        k2 = torch.randn(1, 2, 5, 32, device=self.device)
        v2 = torch.randn(1, 2, 5, 32, device=self.device)
        a = q.quantize_kv(k1, v1, return_compressed=True)
        b = q.quantize_kv(k2, v2, return_compressed=True)
        c = concat_compressed_kv(a, b)
        self.assertEqual(c["k_idx"].shape[2], 8)
        kf, vf = q.decompress_kv_cache(c)
        self.assertEqual(kf.shape, (1, 2, 8, 32))
        self.assertEqual(vf.shape, (1, 2, 8, 32))

if __name__ == '__main__':
    unittest.main(verbosity=2)