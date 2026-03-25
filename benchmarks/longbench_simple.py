"""
LongBench (simple proxy benchmark).

Real LongBench needs a dataset and a model, but the roadmap asks for "even simple".
Here we measure:
  - MSE between original and reconstructed attention scores
  - fraction of cases where the top-1 score is preserved as seq_len grows
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import torch

from turboquant import TurboQuantProd


def proxy_metrics(
    quantizer: TurboQuantProd,
    *,
    seq_len: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    needle_pos: int,
    strength: float = 5.0,
) -> Tuple[float, float]:
    # Single batch/head for simplicity.
    q = torch.randn(1, 1, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, 1, seq_len, head_dim, device=device, dtype=dtype)
    v = k

    q_dir = q / (torch.linalg.norm(q.float(), dim=-1, keepdim=True).to(dtype=q.dtype) + 1e-8)
    k[:, :, needle_pos, :] = k[:, :, needle_pos, :] + strength * q_dir.squeeze(2)

    orig_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [1,1,1,L]
    orig_top = int(torch.argmax(orig_scores.flatten()).item())

    compressed = quantizer.compress(k, v)
    k_recon, _ = quantizer.decompress(compressed)
    recon_scores = torch.matmul(q, k_recon.transpose(-2, -1)) / math.sqrt(head_dim)
    recon_top = int(torch.argmax(recon_scores.flatten()).item())

    acc = 1.0 if recon_top == orig_top else 0.0
    mse_scores = torch.mean((orig_scores - recon_scores) ** 2).item()
    return acc, mse_scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seq-lens", type=str, default="256,512,1024,2048")
    p.add_argument("--needle-pos", type=int, default=-1)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    seq_lens: List[int] = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    quantizer = TurboQuantProd(bits=args.bits, head_dim=args.head_dim, device=args.device)

    for L in seq_lens:
        needle_pos = args.needle_pos if args.needle_pos >= 0 else L // 2
        accs = []
        mses = []
        for _ in range(args.trials):
            acc, mse_scores = proxy_metrics(
                quantizer,
                seq_len=L,
                head_dim=args.head_dim,
                device=args.device,
                dtype=dtype,
                needle_pos=needle_pos,
            )
            accs.append(acc)
            mses.append(mse_scores)
        print(
            f"seq_len={L:5d}  top-1 acc={sum(accs)/len(accs):.3f}  scores MSE={sum(mses)/len(mses):.6f}"
        )


if __name__ == "__main__":
    main()

