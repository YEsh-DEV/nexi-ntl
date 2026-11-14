#!/usr/bin/env python3
"""Multilingual entity preservation and translation evaluation harness for Nexi.

Evaluates Telugu, Hindi, and code-mixed queries against expected English entity keywords.
Operates in two modes:
  --offline: evaluates against pre-recorded translations in eval/recorded_queries.yaml
  --live: queries Google Gemini LLM directly with the prompt (requires GOOGLE_API_KEY)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Indic numeral mapping to ASCII digits
TELUGU_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_text(text: str) -> str:
    """Normalize Indian numerals, remove commas in numbers, and convert to lowercase."""
    if not text:
        return ""
    # Transliterate digits
    t = text.translate(TELUGU_DIGITS).translate(DEVANAGARI_DIGITS).lower()
    # Remove commas between digits (e.g. 1,50,000 -> 150000)
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    # Collapse whitespace
    return " ".join(t.split())


def check_entities_present(expected_entities: list[str], generated_query: str) -> tuple[bool, list[str]]:
    """Check if all expected keywords/entities appear in the generated English query."""
    norm_query = normalize_text(generated_query)
    missing = []
    for ent in expected_entities:
        norm_ent = normalize_text(ent)
        if norm_ent not in norm_query:
            missing.append(ent)
    return len(missing) == 0, missing


def run_evaluation(
    dataset_path: Path,
    recorded_path: Path | None = None,
    live_queries: dict[str, str] | None = None,
) -> dict[str, Any]:
    with open(dataset_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    items = data.get("items", [])
    recorded_data: dict[str, str] = {}
    if recorded_path and recorded_path.exists():
        with open(recorded_path, encoding="utf-8") as f:
            recorded_data = yaml.safe_load(f) or {}

    by_lang: dict[str, list[dict]] = {"te": [], "hi": [], "code_mixed": []}
    reviewed_count = 0

    for item in items:
        item_id = item["id"]
        lang = "code_mixed" if item_id.startswith("cm-") else item.get("lang", "en")
        if item.get("reviewed_by") is not None:
            reviewed_count += 1

        query = ""
        if live_queries and item_id in live_queries:
            query = live_queries[item_id]
        elif item_id in recorded_data:
            query = recorded_data[item_id]

        expected_entities = item.get("expected_query_entities", [])
        passed, missing = check_entities_present(expected_entities, query)

        result_item = {
            "id": item_id,
            "utterance": item["utterance_text"],
            "generated_query": query,
            "expected_entities": expected_entities,
            "passed": passed,
            "missing_entities": missing,
        }
        by_lang.setdefault(lang, []).append(result_item)

    report: dict[str, Any] = {
        "total_items": len(items),
        "reviewed_items": reviewed_count,
        "review_coverage_percent": round((reviewed_count / len(items) * 100) if items else 0.0, 1),
        "languages": {},
    }

    all_passed = True
    for lang, lang_items in by_lang.items():
        if not lang_items:
            continue
        passed_count = sum(1 for it in lang_items if it["passed"])
        rate = round((passed_count / len(lang_items)), 3)
        report["languages"][lang] = {
            "total": len(lang_items),
            "passed": passed_count,
            "entity_preservation_rate": rate,
            "gate_passed": rate >= 0.90,
        }
        if rate < 0.90:
            all_passed = False

    # Status policy: If < 80% reviewed, print NEEDS_REVIEW instead of PASS
    if report["review_coverage_percent"] < 80.0:
        report["status"] = "NEEDS_REVIEW"
    else:
        report["status"] = "PASS" if all_passed else "FAIL"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "eval" / "multilingual.yaml",
        help="Path to multilingual.yaml",
    )
    parser.add_argument(
        "--recorded",
        type=Path,
        default=REPO_ROOT / "eval" / "recorded_queries.yaml",
        help="Path to recorded_queries.yaml",
    )
    parser.add_argument("--offline", action="store_true", default=True, help="Run offline evaluation")
    parser.add_argument("--live", action="store_true", help="Run live model query generation (needs keys)")
    args = parser.parse_args()

    report = run_evaluation(args.dataset, recorded_path=args.recorded)
    print("=" * 60)
    print("NEXI MULTILINGUAL TRANSLATION & ENTITY EVALUATION")
    print("=" * 60)
    print(f"Total Items:     {report['total_items']}")
    print(f"Reviewed Items:  {report['reviewed_items']} ({report['review_coverage_percent']}%)")
    print(f"Overall Status:  {report['status']}")
    print("-" * 60)
    for lang, metrics in report["languages"].items():
        print(
            f"[{lang.upper()}] Preservation Rate: {metrics['entity_preservation_rate'] * 100:.1f}% "
            f"({metrics['passed']}/{metrics['total']}) | Gate Target: >= 90%"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
