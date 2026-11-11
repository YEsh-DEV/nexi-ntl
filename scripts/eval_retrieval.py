"""Offline retrieval + grounding + anti-hallucination benchmark for Nexi.

Loads eval/questions.yaml, runs every query through Retriever.search(), and prints a
benchmark report. Fully offline: no LLM, no TTS/STT, no network, no test framework.

Usage:
    python scripts/eval_retrieval.py

Writes eval/benchmark_report.json and prints a summary to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import yaml

# Make the repo root importable (agent/, rag/) when run as `python scripts/eval_retrieval.py`.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.config import Settings  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

DEFAULT_QUESTIONS = REPO / "eval" / "questions.yaml"
DEFAULT_REPORT = REPO / "eval" / "benchmark_report.json"

# Dummy credentials suffice: Retriever never touches API keys.
OFFLINE_KEY = "eval-offline"


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    questions = data.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions, data


def percentile(sorted_values: list[float], percentile_rank: float) -> float:
    """Linear-interpolation percentile (no numpy dependency)."""
    if not sorted_values:
        return 0.0
    ordered = sorted(sorted_values)
    rank = (percentile_rank / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def fmt_ms(value: float) -> str:
    return f"{value:,.1f} ms"


def _build_report(
    details: list[dict],
    questions: list[dict],
    metadata: dict,
    retriever_knowledge_version: str | None,
) -> dict:
    by_category: dict[str, list[dict]] = {}
    for detail in details:
        by_category.setdefault(detail["category"], []).append(detail)

    answerable = by_category.get("answerable", [])
    off_topic = by_category.get("out_of_domain", [])
    unanswerable = by_category.get("unanswerable_university", [])
    adversarial = by_category.get("adversarial_injection", [])

    # Evidence Hit Rate @ 4: answerable question whose expected source was retrieved.
    answerable_hits = sum(
        1
        for d in answerable
        if d["returned_results"] and d["expected_source"] in d["retrieved_sources"]
    )
    hit_rate = (answerable_hits / len(answerable) * 100) if answerable else 0.0

    # False Acceptance Rate: off-topic question that returned any evidence.
    false_accepts = sum(1 for d in off_topic if d["returned_results"])
    far = (false_accepts / len(off_topic) * 100) if off_topic else 0.0

    # Abstention rates: question that returned NO evidence.
    unanswerable_abstains = sum(1 for d in unanswerable if not d["returned_results"])
    unanswerable_abstain_rate = (
        unanswerable_abstains / len(unanswerable) * 100 if unanswerable else 0.0
    )
    adversarial_abstains = sum(1 for d in adversarial if not d["returned_results"])
    adversarial_abstain_rate = (
        adversarial_abstains / len(adversarial) * 100 if adversarial else 0.0
    )

    all_latencies = [d["latency_ms"] for d in details]
    ordered_latencies = sorted(all_latencies)

    report = {
        "meta": {
            "schema_version": metadata.get("version"),
            "knowledge_version_yaml": metadata.get("knowledge_version"),
            "knowledge_version_index": retriever_knowledge_version,
            "retrieval_top_n": metadata.get("retrieval_top_n"),
            "total_questions": len(questions),
            "counts_by_category": {
                name: len(items) for name, items in by_category.items()
            },
        },
        "metrics": {
            "evidence_hit_rate_at_4_percent": round(hit_rate, 2),
            "answerable_hits": answerable_hits,
            "answerable_total": len(answerable),
            "false_acceptance_rate_off_topic_percent": round(far, 2),
            "off_topic_false_accepts": false_accepts,
            "off_topic_total": len(off_topic),
            "unanswerable_abstain_rate_percent": round(unanswerable_abstain_rate, 2),
            "unanswerable_abstains": unanswerable_abstains,
            "unanswerable_total": len(unanswerable),
            "adversarial_abstain_rate_percent": round(adversarial_abstain_rate, 2),
            "adversarial_abstains": adversarial_abstains,
            "adversarial_total": len(adversarial),
            "p50_latency_ms": round(percentile(ordered_latencies, 50), 2),
            "p95_latency_ms": round(percentile(ordered_latencies, 95), 2),
            "min_latency_ms": round(min(all_latencies), 2) if all_latencies else 0.0,
            "max_latency_ms": round(max(all_latencies), 2) if all_latencies else 0.0,
            "mean_latency_ms": round(sum(all_latencies) / len(all_latencies), 2)
            if all_latencies
            else 0.0,
        },
        "details": details,
    }
    return report


def print_summary(report: dict) -> None:
    meta = report["meta"]
    metrics = report["metrics"]
    print("=" * 70)
    print("NEXI RETRIEVAL + GROUNDING + ANTI-HALLUCINATION BENCHMARK")
    print("=" * 70)
    print(f"Knowledge version (yaml): {meta['knowledge_version_yaml']}")
    print(f"Knowledge version (index): {meta['knowledge_version_index']}")
    print(f"Total questions: {meta['total_questions']}")
    print(f"Counts by category: {meta['counts_by_category']}")
    print("-" * 70)
    print("RETRIEVAL / GROUNDING METRICS")
    print("-" * 70)
    print(
        f"Evidence Hit Rate @4 (answerable): "
        f"{metrics['evidence_hit_rate_at_4_percent']:.1f}%  "
        f"({metrics['answerable_hits']}/{metrics['answerable_total']})"
    )
    print(
        f"False Acceptance Rate (off-topic):  "
        f"{metrics['false_acceptance_rate_off_topic_percent']:.1f}%  "
        f"({metrics['off_topic_false_accepts']}/{metrics['off_topic_total']})  target 0%"
    )
    print(
        f"Abstain Rate (unanswerable uni):    "
        f"{metrics['unanswerable_abstain_rate_percent']:.1f}%  "
        f"({metrics['unanswerable_abstains']}/{metrics['unanswerable_total']})  informational"
    )
    print(
        f"Abstain Rate (adversarial):         "
        f"{metrics['adversarial_abstain_rate_percent']:.1f}%  "
        f"({metrics['adversarial_abstains']}/{metrics['adversarial_total']})  target 100%"
    )
    print("-" * 70)
    print("LATENCY (wall-clock, Retriever.search, all queries)")
    print("-" * 70)
    print(f"p50: {fmt_ms(metrics['p50_latency_ms'])}")
    print(f"p95: {fmt_ms(metrics['p95_latency_ms'])}")
    print(f"min: {fmt_ms(metrics['min_latency_ms'])}")
    print(f"max: {fmt_ms(metrics['max_latency_ms'])}")
    print(f"mean: {fmt_ms(metrics['mean_latency_ms'])}")
    print("=" * 70)


async def run_all(
    settings: Settings, questions: list[dict]
) -> tuple[list[dict], str]:
    """Run every query through one Retriever instance; return (details, knowledge version)."""
    retriever = Retriever(settings)
    details: list[dict] = []
    try:
        for item in questions:
            start = time.perf_counter()
            result = await retriever.search(item["question"])
            latency_ms = (time.perf_counter() - start) * 1000.0
            sources = [evidence.source for evidence in result.evidence]
            details.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "domain": item.get("domain"),
                    "question": item["question"],
                    "expected_source": item.get("expected_source"),
                    "expected_page": item.get("expected_page"),
                    "retrieved_sources": sources,
                    "returned_results": len(sources) > 0,
                    "latency_ms": round(latency_ms, 2),
                }
            )
    finally:
        retriever.close()
    return details, retriever.knowledge_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to questions.yaml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to write benchmark_report.json",
    )
    args = parser.parse_args()

    questions, metadata = load_questions(args.questions)

    # Dummy credentials suffice: Retriever never touches API keys.
    settings = Settings(
        deepgram_api_key=OFFLINE_KEY,
        google_api_key=OFFLINE_KEY,
        sarvam_api_key=OFFLINE_KEY,
    )

    details, index_knowledge_version = asyncio.run(run_all(settings, questions))

    report = _build_report(
        details,
        questions,
        metadata,
        retriever_knowledge_version=index_knowledge_version,
    )

    print_summary(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
