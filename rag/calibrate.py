"""Calibrate rerank_min_score threshold exclusively using the calibration split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.config import Settings, load_settings
from rag.manifest import resolve_artifact_path

REPO_ROOT = Path(__file__).resolve().parent.parent


def calibrate_threshold(
    eval_items: list[dict[str, Any]],
    score_func: Any,
) -> tuple[float, dict[str, Any]]:
    """Determine rerank_min_score separating answerable from unanswerable calibration items.

    Strictly refuses to inspect any item whose split is not 'calibration'.
    """
    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []

    for item in eval_items:
        if item.get("split") != "calibration":
            raise ValueError(f"Split leakage: item {item.get('id')} has split='{item.get('split')}' (expected 'calibration')")

        q = item.get("question", "").strip()
        top_score = score_func(q, item)
        if item.get("category") == "answerable" and not item.get("abstain", False):
            answerable_scores.append(top_score)
        else:
            unanswerable_scores.append(top_score)

    if not answerable_scores:
        chosen_threshold = 0.20
    elif not unanswerable_scores:
        chosen_threshold = min(answerable_scores) * 0.8
    else:
        answerable_scores.sort()
        unanswerable_scores.sort()
        # Find score separating bottom of answerable and top of unanswerable
        min_ans = answerable_scores[0]
        max_unans = unanswerable_scores[-1]
        if min_ans > max_unans:
            chosen_threshold = round((min_ans + max_unans) / 2.0, 3)
        else:
            # Overlap: pick 10th percentile of answerable scores
            idx = max(0, int(len(answerable_scores) * 0.10))
            chosen_threshold = round(answerable_scores[idx], 3)

    stats = {
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "items_used": len(eval_items),
        "answerable_count": len(answerable_scores),
        "unanswerable_count": len(unanswerable_scores),
        "answerable_min": min(answerable_scores) if answerable_scores else None,
        "answerable_max": max(answerable_scores) if answerable_scores else None,
        "unanswerable_min": min(unanswerable_scores) if unanswerable_scores else None,
        "unanswerable_max": max(unanswerable_scores) if unanswerable_scores else None,
        "chosen_threshold": chosen_threshold,
    }
    return chosen_threshold, stats


def run_calibration(
    artifact_dir: Path,
    settings: Settings,
    questions_path: Path | None = None,
    mock_score_func: Any = None,
) -> dict[str, Any]:
    questions_path = questions_path or (REPO_ROOT / "eval" / "questions.yaml")
    if not questions_path.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    raw_data = yaml.safe_load(questions_path.read_text(encoding="utf-8"))
    all_questions: list[dict[str, Any]] = raw_data.get("questions", [])

    # Filter strictly to calibration split
    calib_questions = [q for q in all_questions if q.get("split") == "calibration"]
    if not calib_questions:
        raise ValueError("No calibration split items found in questions file")

    if mock_score_func is not None:
        score_func = mock_score_func
    else:
        from rag.retriever import Retriever

        retriever = Retriever(settings)
        try:
            def _real_score_func(query: str, item: dict[str, Any]) -> float:
                import asyncio
                res = asyncio.run(retriever.search(query))
                if res.evidence:
                    return res.evidence[0].score
                return -1.0
            score_func = _real_score_func
        finally:
            retriever.close()

    threshold, stats = calibrate_threshold(calib_questions, score_func)

    manifest_file = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["rerank_min_score"] = threshold
    manifest["calibration"] = stats
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate rerank_min_score on calibration split")
    parser.add_argument("version", help="Artifact version (hex or staging/version)")
    parser.add_argument("--questions", default=None, help="Path to questions.yaml")

    args = parser.parse_args()
    settings = load_settings()
    root = settings.chroma_dir.resolve()

    version = args.version
    if (root / "staging" / version).is_dir():
        artifact_dir = root / "staging" / version
    else:
        artifact_dir = resolve_artifact_path(root, version)

    if not artifact_dir.is_dir():
        print(f"Error: artifact directory not found for version '{version}'", file=sys.stderr)
        sys.exit(1)

    q_path = Path(args.questions) if args.questions else None
    stats = run_calibration(artifact_dir, settings, questions_path=q_path)
    print(f"Calibrated rerank_min_score: {stats['chosen_threshold']} (used {stats['items_used']} calibration items)")


if __name__ == "__main__":
    main()
