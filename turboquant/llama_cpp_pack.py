"""
Binary sidecar and layout notes for **llama.cpp** (C++/CUDA) integration.

llama.cpp does not ship TurboQuant; this module defines a **portable sidecar**
(``*.tqmeta``) with ``Π``, ``S``, centroids, and ``qjl_factor`` so native code can
decode the same packed KV pages as :mod:`turboquant.vllm_pack`.

**Paged KV bytes** per physical block are identical to vLLM:
:class:`~turboquant.vllm_pack.TurboQuantPageLayout`, :func:`~turboquant.vllm_pack.scatter_one_token`,
:func:`~turboquant.vllm_pack.uint8_pages_to_paged_dict`.

See ``integrations/llama_cpp/README.md`` for upstream hook points and a reference C++ decoder.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import BinaryIO, Union

import torch

from .core import TurboQuantProd, _centroid_levels_for

# File format v1: little-endian, float32 payload for Π, S, centroids (portable).
_MAGIC = b"TURBOQT1"
_VERSION = 1
_HEADER_STRUCT = struct.Struct("<8sIIIId")  # magic, ver, bits, head_dim, k, qjl (double)


def _infer_codebook_from_header(bits: int, k: int) -> str:
    ek = _centroid_levels_for(int(bits), "paper")
    if int(k) == ek:
        return "paper"
    if int(k) == 3:
        return "ternary"
    raise ValueError(
        f"turboquant metadata: bits={bits} and k={k} do not match paper (expect k={ek}) or ternary (k=3)"
    )


def serialize_quantizer_metadata(quantizer: TurboQuantProd) -> bytes:
    """
    Pack ``TurboQuantProd`` state into bytes (CPU float32 Π/S/centroids + double qjl_factor).

    Use with the same ``bits``/``head_dim`` when allocating KV pages; load in C++ or
    :func:`deserialize_quantizer_metadata`.
    """
    d = int(quantizer.head_dim)
    bits = int(quantizer.bits)
    k = int(quantizer._centroids.numel())
    pi_f = quantizer.Pi.detach().float().cpu().contiguous().numpy().tobytes()
    s_f = quantizer.S.detach().float().cpu().contiguous().numpy().tobytes()
    c_f = quantizer._centroids.detach().float().cpu().contiguous().numpy().tobytes()
    if len(pi_f) != d * d * 4 or len(s_f) != d * d * 4:
        raise RuntimeError("internal shape error for Pi/S")
    if len(c_f) != k * 4:
        raise RuntimeError("internal shape error for centroids")
    header = _HEADER_STRUCT.pack(_MAGIC, _VERSION, bits, d, k, float(quantizer._qjl_factor))
    return header + c_f + pi_f + s_f


def deserialize_quantizer_metadata(
    data: bytes,
    *,
    device: Union[str, None] = None,
    dtype: torch.dtype = torch.float32,
) -> TurboQuantProd:
    """Rebuild :class:`~turboquant.TurboQuantProd` from :func:`serialize_quantizer_metadata` output."""
    hlen = _HEADER_STRUCT.size
    if len(data) < hlen:
        raise ValueError("truncated turboquant metadata")
    magic, ver, bits, head_dim, k, qjl = _HEADER_STRUCT.unpack_from(data, 0)
    if magic != _MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver != _VERSION:
        raise ValueError(f"unsupported version {ver}")
    need = hlen + k * 4 + head_dim * head_dim * 8
    if len(data) < need:
        raise ValueError(f"truncated payload: need {need} bytes, got {len(data)}")
    off = hlen
    centroids = torch.frombuffer(bytearray(data[off : off + k * 4]), dtype=torch.float32).clone()
    off += k * 4
    d2 = head_dim * head_dim * 4
    pi = torch.frombuffer(bytearray(data[off : off + d2]), dtype=torch.float32).reshape(head_dim, head_dim).clone()
    off += d2
    s = torch.frombuffer(bytearray(data[off : off + d2]), dtype=torch.float32).reshape(head_dim, head_dim).clone()
    expected_qjl = math.sqrt(math.pi / 2.0) / float(head_dim)
    if not math.isclose(qjl, expected_qjl, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"corrupt qjl_factor in header: {qjl} vs expected {expected_qjl}")
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    codebook = _infer_codebook_from_header(bits, k)
    return TurboQuantProd(
        bits=int(bits),
        head_dim=int(head_dim),
        device=dev,
        dtype=dtype,
        codebook=codebook,
        Pi=pi,
        S=s,
        centroids=centroids,
    )


def write_quantizer_metadata(path: Union[str, Path], quantizer: TurboQuantProd) -> None:
    Path(path).write_bytes(serialize_quantizer_metadata(quantizer))


def read_quantizer_metadata(
    path: Union[str, Path],
    *,
    device: Union[str, None] = None,
    dtype: torch.dtype = torch.float32,
) -> TurboQuantProd:
    return deserialize_quantizer_metadata(Path(path).read_bytes(), device=device, dtype=dtype)


def append_metadata_to_file(fp: BinaryIO, quantizer: TurboQuantProd) -> None:
    """Write one metadata blob (for multi-quantizer archives; caller defines outer framing)."""
    fp.write(serialize_quantizer_metadata(quantizer))
