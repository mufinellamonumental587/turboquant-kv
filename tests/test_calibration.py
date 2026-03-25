import torch
import unittest

from turboquant.calibration import (
    CalibrationMode,
    calibrate_turboquant_from_tensor,
    kmeans_1d,
)
from turboquant.core import TurboQuantProd, _centroid_levels_for


class TestCalibration(unittest.TestCase):
    def test_kmeans_1d_three_clusters(self):
        g = torch.Generator()
        g.manual_seed(1)
        c_true = torch.tensor([-2.0, 0.0, 3.0])
        x = torch.cat(
            [
                c_true[0] + 0.1 * torch.randn(500, generator=g),
                c_true[1] + 0.1 * torch.randn(500, generator=g),
                c_true[2] + 0.1 * torch.randn(500, generator=g),
            ]
        )
        g2 = torch.Generator()
        g2.manual_seed(2)
        c = kmeans_1d(x, 3, generator=g2)
        self.assertEqual(c.shape, (3,))
        self.assertTrue(torch.allclose(c, c_true, atol=0.15))

    def test_ternary_codebook_shapes(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = TurboQuantProd(bits=3, head_dim=32, device=device, codebook="ternary")
        self.assertEqual(q._centroids.numel(), 3)
        x = torch.randn(4, 32, device=device)
        _, idx, _, _, _ = q.quantize(x)
        self.assertTrue((idx >= 0).all() and (idx < 3).all())

    def test_calibrate_paper_improves_mse_vs_random_centroids_on_same_pi(self):
        device = "cpu"
        head_dim = 16
        seed = 7
        bits = 3
        torch.manual_seed(0)
        base = TurboQuantProd(bits=bits, head_dim=head_dim, device=device, seed=seed)
        samples = torch.randn(200, head_dim)
        x = samples.float()
        xn = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        y = xn @ base.Pi.T
        c_bad = torch.linspace(y.min(), y.max(), 4)
        q_bad = TurboQuantProd(
            bits=bits,
            head_dim=head_dim,
            device=device,
            seed=seed,
            Pi=base.Pi,
            S=base.S,
            centroids=c_bad.to(base.dtype),
        )
        q_fit = calibrate_turboquant_from_tensor(
            samples,
            head_dim=head_dim,
            mode=CalibrationMode.PAPER_POW2,
            bits=bits,
            seed=seed,
            device=device,
            max_samples=50_000,
        )
        self.assertTrue(torch.allclose(q_fit.Pi, base.Pi))
        self.assertTrue(torch.allclose(q_fit.S, base.S))

        def mse_quant(qz):
            recon, _, _, _, _ = qz.quantize(samples)
            return torch.mean((samples - recon) ** 2).item()

        self.assertLess(mse_quant(q_fit), mse_quant(q_bad))

    def test_calibrate_ternary(self):
        device = "cpu"
        head_dim = 8
        q = calibrate_turboquant_from_tensor(
            torch.randn(300, head_dim),
            head_dim=head_dim,
            mode="ternary_158",
            bits=3,
            seed=11,
            device=device,
        )
        self.assertEqual(q.codebook, "ternary")
        self.assertEqual(_centroid_levels_for(3, "ternary"), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
