"""
TurboQuant — simplified quality smoke test
"""

import torch
import math
from turboquant.core import TurboQuantProd

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    if cos_sim > 0.85:
        print("✅ Good quality")
    elif cos_sim > 0.7:
        print("👍 Acceptable quality")
    else:
        print("⚠️  Quality is still low — improve codebook and QJL")

if __name__ == "__main__":
    main()