#!/usr/bin/env python3
"""Standard library Word Error Rate (WER) calculator using Levenshtein distance on word tokens."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase word tokens, stripping punctuation."""
    clean = re.sub(r"[^\w\s]", "", text.lower())
    return [w for w in clean.split() if w]


def word_levenshtein(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int, int, int]:
    """Compute (substitutions, deletions, insertions, total_errors) between token lists."""
    r_len = len(ref_words)
    h_len = len(hyp_words)

    # DP table: (r_len + 1) x (h_len + 1)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                dp[i][j] = min(substitution, deletion, insertion)

    # Backtrace to count individual error categories
    i, j = r_len, h_len
    subs, dels, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break

    total_errors = dp[r_len][h_len]
    return subs, dels, ins, total_errors


def compute_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) as errors / len(ref_words)."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    _, _, _, total_errors = word_levenshtein(ref_tokens, hyp_tokens)
    return total_errors / len(ref_tokens)


def evaluate_csv(csv_path: Path) -> dict[str, object]:
    """Read CSV with 'reference' and 'hypothesis' columns and compute aggregate WER."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row.get("reference", "")
            hyp = row.get("hypothesis", "")
            wer = compute_wer(ref, hyp)
            rows.append({"reference": ref, "hypothesis": hyp, "wer": wer})

    mean_wer = sum(r["wer"] for r in rows) / len(rows) if rows else 0.0
    return {"total_samples": len(rows), "mean_wer": round(mean_wer, 4), "samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Reference ground truth text")
    parser.add_argument("--hyp", help="Hypothesis transcribed text")
    parser.add_argument("--csv", type=Path, help="CSV file with reference,hypothesis columns")
    args = parser.parse_args()

    if args.csv:
        result = evaluate_csv(args.csv)
        print(f"Evaluated {result['total_samples']} samples | Mean WER: {result['mean_wer'] * 100:.2f}%")
    elif args.ref and args.hyp:
        wer = compute_wer(args.ref, args.hyp)
        print(f"Reference:  {args.ref}")
        print(f"Hypothesis: {args.hyp}")
        print(f"WER: {wer * 100:.2f}%")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
