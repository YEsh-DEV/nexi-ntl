#!/usr/bin/env python3
"""Validates that a lockfile is strictly CPU-only with no CUDA/NVIDIA packages."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN_PREFIXES = ("nvidia-", "triton")
CUDA_TORCH_PATTERN = re.compile(r"torch.*(\+cu|cuda)", re.IGNORECASE)


def validate_lock_content(content: str) -> list[str]:
    """Scan lockfile text and return a list of violation messages."""
    violations: list[str] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Extract package name (handles 'package==version', 'package @ ...', etc.)
        pkg_part = line.split(";")[0].split("\\")[0].strip()
        pkg_name = re.split(r"[=<>~!@ ]", pkg_part)[0].strip().lower()

        for prefix in FORBIDDEN_PREFIXES:
            if pkg_name.startswith(prefix):
                violations.append(
                    f"Line {line_number}: Forbidden package '{pkg_name}' (starts with '{prefix}')"
                )

        if "torch" in pkg_name and CUDA_TORCH_PATTERN.search(pkg_part):
            violations.append(
                f"Line {line_number}: Torch package has CUDA build tag: '{pkg_part}'"
            )

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lock_file",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "requirements-cpu.lock",
        help="Path to requirements-cpu.lock (defaults to repo root)",
    )
    args = parser.parse_args()

    if not args.lock_file.exists():
        print(f"ERROR: Lock file not found at {args.lock_file}", file=sys.stderr)
        sys.exit(1)

    content = args.lock_file.read_text(encoding="utf-8")
    violations = validate_lock_content(content)

    if violations:
        print(f"FAILED: Found {len(violations)} CUDA/NVIDIA violation(s) in {args.lock_file}:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        sys.exit(1)

    print(f"PASS: {args.lock_file.name} is verified CPU-only (no nvidia/triton/cuda packages).")
    sys.exit(0)


if __name__ == "__main__":
    main()
