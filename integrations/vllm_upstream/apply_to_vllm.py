#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply TurboQuant integration to a vLLM source tree via a pinned git patch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PATCH_NAME = "vllm_turboquant_e38817f.patch"
# vLLM revision used to generate the patch (short SHA): e38817f — see integrations/vllm_upstream/README.md.
PATCH_UPSTREAM_SHORT = "e38817f"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply TurboQuant KV-cache changes to a vLLM checkout using "
            f"`git apply`. The patch was built from upstream ~{PATCH_UPSTREAM_SHORT}; "
            "use a matching or recent main, or `git apply --3way` if hunks drift."
        )
    )
    parser.add_argument(
        "vllm_root",
        type=Path,
        help="Root of a clone of https://github.com/vllm-project/vllm",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only run `git apply --check` (no changes on disk).",
    )
    parser.add_argument(
        "--3way",
        action="store_true",
        help="Run `git apply --3way` to attempt a three-way merge on conflicts.",
    )
    args = parser.parse_args()
    root = args.vllm_root.resolve()
    if not (root / "vllm").is_dir():
        print(f"error: not a vLLM root (expected vllm/ package): {root}", file=sys.stderr)
        return 1

    patch_path = Path(__file__).resolve().parent / "patches" / PATCH_NAME
    if not patch_path.is_file():
        print(f"error: missing patch file: {patch_path}", file=sys.stderr)
        return 1

    cmd = ["git", "apply"]
    if args.check:
        cmd.append("--check")
    if args.3way:
        cmd.append("--3way")
    cmd.append(str(patch_path))

    proc = subprocess.run(cmd, cwd=root)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
