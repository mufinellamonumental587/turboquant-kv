import tempfile
import unittest
from pathlib import Path

import torch

from turboquant.core import TurboQuantProd


class TestCentroidsCache(unittest.TestCase):
    def tearDown(self) -> None:
        TurboQuantProd.clear_centroids_cache()

    def test_preload_populates_cache_for_paper(self) -> None:
        TurboQuantProd.clear_centroids_cache()
        n = TurboQuantProd.preload_centroids(head_dims=(48,), bits=(4,), codebook="paper")
        self.assertEqual(n, 1)
        keys = TurboQuantProd.centroids_cache_keys()
        self.assertIn(("paper", 48, 3), keys)

    def test_preload_ternary_does_not_call_build(self) -> None:
        TurboQuantProd.clear_centroids_cache()
        n = TurboQuantProd.preload_centroids(head_dims=(48,), bits=(4,), codebook="ternary")
        self.assertEqual(n, 0)
        self.assertEqual(TurboQuantProd.centroids_cache_len(), 0)

    def test_legacy_two_tuple_key_migrates_on_read(self) -> None:
        TurboQuantProd.clear_centroids_cache()
        TurboQuantProd.preload_centroids(head_dims=(40,), bits=(4,), codebook="paper")
        with TurboQuantProd._cache_lock:
            t = TurboQuantProd._CENTROIDS_CACHE[("paper", 40, 3)].clone()
            del TurboQuantProd._CENTROIDS_CACHE[("paper", 40, 3)]
            TurboQuantProd._CENTROIDS_CACHE[(40, 3)] = t
        got = TurboQuantProd._paper_centroids_cache_get(40, 3)
        self.assertIsNotNone(got)
        self.assertTrue(torch.equal(got, t))
        with TurboQuantProd._cache_lock:
            self.assertNotIn((40, 3), TurboQuantProd._CENTROIDS_CACHE)
            self.assertIn(("paper", 40, 3), TurboQuantProd._CENTROIDS_CACHE)

    def test_save_load_roundtrip(self) -> None:
        TurboQuantProd.clear_centroids_cache()
        TurboQuantProd.preload_centroids(head_dims=(56,), bits=(4,), codebook="paper")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cq.pt"
            w = TurboQuantProd.save_centroids_cache(p)
            self.assertGreaterEqual(w, 1)
            TurboQuantProd.clear_centroids_cache()
            self.assertEqual(TurboQuantProd.centroids_cache_len(), 0)
            m = TurboQuantProd.load_centroids_cache(p, merge=True)
            self.assertGreaterEqual(m, 1)
            self.assertIn(("paper", 56, 3), TurboQuantProd.centroids_cache_keys())

    def test_quantizer_matches_after_preload(self) -> None:
        TurboQuantProd.clear_centroids_cache()
        TurboQuantProd.preload_centroids(head_dims=(72,), bits=(4,), codebook="paper")
        q0 = TurboQuantProd(bits=4, head_dim=72, device="cpu", seed=999)
        TurboQuantProd.clear_centroids_cache()
        q1 = TurboQuantProd(bits=4, head_dim=72, device="cpu", seed=999)
        self.assertTrue(torch.allclose(q0._centroids, q1._centroids))

    def test_load_merge_false_replaces_entire_cache(self) -> None:
        TurboQuantProd.clear_centroids_cache()
        TurboQuantProd.preload_centroids(head_dims=(48, 64), bits=(4,), codebook="paper")
        self.assertEqual(TurboQuantProd.centroids_cache_len(), 2)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "snap.pt"
            TurboQuantProd.save_centroids_cache(p)
            TurboQuantProd.clear_centroids_cache()
            TurboQuantProd.preload_centroids(head_dims=(80,), bits=(4,), codebook="paper")
            self.assertIn(("paper", 80, 3), TurboQuantProd.centroids_cache_keys())
            TurboQuantProd.load_centroids_cache(p, merge=False)
            keys = set(TurboQuantProd.centroids_cache_keys())
            self.assertEqual(keys, {("paper", 48, 3), ("paper", 64, 3)})
