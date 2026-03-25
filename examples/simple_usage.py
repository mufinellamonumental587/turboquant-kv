"""
TurboQuant — simplified quality smoke test
"""

import argparse
import math

import torch

from turboquant.core import TurboQuantProd


def _bar_line(label: str, value: float, fill01: float, width: int = 26) -> str:
    fill01 = max(0.0, min(1.0, fill01))
    n = int(round(fill01 * width))
    bar = "#" * n + "-" * (width - n)
    return f"{label:28} [{bar}] {value:.6f}"


def print_quality_diagram(mse_ip: float, cos_sim: float) -> None:
    """ASCII bars: higher cosine is better; lower MSE is better (bar = quality vs cap)."""
    print("\n--- attention score matrix (K K^T / sqrt(d)) ---")
    print(_bar_line("Cosine similarity (higher better)", cos_sim, cos_sim))
    mse_cap = 2.0
    mse_good = 1.0 - min(mse_ip / mse_cap, 1.0)
    print(
        _bar_line(
            f"MSE inner product (lower better, cap={mse_cap:g})",
            mse_ip,
            mse_good,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description="TurboQuant KV compress/decompress quality smoke test")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible K/V (default: 42)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute device (default: auto = cuda if available else cpu)",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}\n")

    quantizer = TurboQuantProd(bits=3, head_dim=128, device=device)

    # Smaller seq_len for speed and lower memory
    batch, heads, seq_len, dim = 1, 8, 256, 128
    k = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32)

    print("Compressing KV...")
    compressed_kv = quantizer.compress(k, v)

    print("Decompressing KV...")
    k_recon, v_recon = quantizer.decompress(compressed_kv)

    # Inner product quality
    orig_ip = torch.matmul(k, k.transpose(-2, -1)) / math.sqrt(dim)
    recon_ip = torch.matmul(k_recon, k_recon.transpose(-2, -1)) / math.sqrt(dim)

    mse_ip = torch.mean((orig_ip - recon_ip) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_ip.flatten(), recon_ip.flatten(), dim=0
    ).item()

    print(f"MSE inner product : {mse_ip:.6f}")
    print(f"Cosine similarity : {cos_sim:.6f}")
    print_quality_diagram(mse_ip, cos_sim)

    if cos_sim > 0.85:
        print("\n[OK] Good quality (cosine > 0.85)")
    elif cos_sim > 0.7:
        print("\n[--] Acceptable quality (cosine > 0.7)")
    else:
        print("\n[!!] Low quality — improve codebook and QJL")


if __name__ == "__main__":
    main()
