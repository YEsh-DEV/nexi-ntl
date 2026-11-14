"""Evidence-level benchmark and quality gate runner for knowledge artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from agent.config import Settings, load_settings
from rag.manifest import resolve_artifact_path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def normalize_text(text: str) -> str:
    """Normalize text for evidence-hit matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def check_evidence_hit(
    evidence_list: list[Any],
    expected_evidence_rules: list[dict[str, Any]],
) -> bool:
    """Check if any returned evidence matches source (and page) and contains must_contain string."""
    if not expected_evidence_rules:
        return False

    for exp in expected_evidence_rules:
        exp_src = (exp.get("source") or "").lower()
        exp_page = exp.get("page")
        must_contain = normalize_text(exp.get("must_contain") or "")

        for rank, ev in enumerate(evidence_list, start=1):
            ev_src = getattr(ev, "source", "").lower()
            ev_text = normalize_text(getattr(ev, "text", ""))
            ev_loc = str(getattr(ev, "page_or_path", "")).lower()

            # Source match
            if exp_src not in ev_src and ev_src not in exp_src:
                continue

            # Page match if specified
            if exp_page is not None:
                exp_page_str = str(exp_page).lower()
                if exp_page_str not in ev_loc:
                    continue

            # Must contain match
            if must_contain and must_contain in ev_text:
                return True
            elif not must_contain:
                return True

    return False


def compute_mrr(evidence_list: list[Any], expected_evidence_rules: list[dict[str, Any]]) -> float:
    if not expected_evidence_rules:
        return 0.0

    for exp in expected_evidence_rules:
        exp_src = (exp.get("source") or "").lower()
        must_contain = normalize_text(exp.get("must_contain") or "")

        for rank, ev in enumerate(evidence_list, start=1):
            ev_src = getattr(ev, "source", "").lower()
            ev_text = normalize_text(getattr(ev, "text", ""))
            if (exp_src in ev_src or ev_src in exp_src) and (not must_contain or must_contain in ev_text):
                return 1.0 / rank

    return 0.0


def compute_percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0 to 100) using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return s[f] + (s[c] - s[f]) * d


def get_hardware_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or "cpu",
        "machine": platform.machine(),
    }


def run_evaluation(
    artifact_dir: Path,
    settings: Settings,
    questions_path: Path | None = None,
    gates_path: Path | None = None,
    mock_retriever: Any = None,
    latency_waiver_reason: str | None = None,
    latency_waiver_approver: str | None = None,
) -> tuple[dict[str, Any], bool]:
    questions_path = questions_path or (REPO_ROOT / "eval" / "questions.yaml")
    gates_path = gates_path or (REPO_ROOT / "eval" / "gates.yaml")

    q_data = yaml.safe_load(questions_path.read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = q_data.get("questions", [])

    gates_data = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    gates: dict[str, float] = gates_data.get("gates", {})

    if mock_retriever is not None:
        retriever = mock_retriever
    else:
        from rag.retriever import Retriever
        retriever = Retriever(settings)

    # Review coverage
    reviewed_items = [q for q in questions if q.get("reviewed_by") is not None]
    review_coverage = len(reviewed_items) / (len(questions) or 1)

    # 1. Warm-up
    warmup_qs = questions[:min(10, len(questions))]
    for wq in warmup_qs:
        q_text = wq.get("question", "")
        if hasattr(retriever, "search"):
            try:
                call_res = retriever.search(q_text)
                if asyncio.iscoroutine(call_res):
                    asyncio.run(call_res)
            except Exception as warmup_exc:
                logger.debug("Warmup query exception ignored: %s", warmup_exc)

    # 2. Main eval run with 3 repetitions for latency
    latencies_ms: list[float] = []
    results: list[dict[str, Any]] = []

    answerable_hits_heldout: list[bool] = []
    answerable_hits_all: list[bool] = []
    mrr_scores: list[float] = []
    legacy_hits: list[bool] = []

    off_topic_abstentions: list[bool] = []
    borderline_abstentions: list[bool] = []
    false_abstentions: list[bool] = []

    for q in questions:
        qid = q.get("id", "")
        cat = q.get("category", "")
        q_text = q.get("question", "")
        is_heldout = q.get("split") == "heldout"
        expected_ev = q.get("expected_evidence", [])

        # Execute search 3 times for latency measurement
        ev_returned = []
        for rep in range(3):
            t0 = perf_counter()
            if hasattr(retriever, "search"):
                call_res = retriever.search(q_text)
                if asyncio.iscoroutine(call_res):
                    res = asyncio.run(call_res)
                else:
                    res = call_res
                ev_returned = list(res.evidence)
            t1 = perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        hit = check_evidence_hit(ev_returned, expected_ev)
        mrr = compute_mrr(ev_returned, expected_ev)

        # Legacy source-only hit
        legacy_hit = False
        if q.get("expected_source"):
            exp_src = str(q["expected_source"]).lower()
            for ev in ev_returned:
                if exp_src in getattr(ev, "source", "").lower():
                    legacy_hit = True
                    break

        if cat == "answerable":
            answerable_hits_all.append(hit)
            legacy_hits.append(legacy_hit)
            mrr_scores.append(mrr)
            if is_heldout:
                answerable_hits_heldout.append(hit)
            if not ev_returned:
                false_abstentions.append(True)
            else:
                false_abstentions.append(False)
        elif cat == "out_of_domain":
            # Off topic must return NO_RESULTS / empty evidence
            off_topic_abstentions.append(len(ev_returned) == 0)
        elif cat in ("unanswerable_university", "adversarial_injection"):
            borderline_abstentions.append(len(ev_returned) == 0)

        results.append({
            "id": qid,
            "category": cat,
            "split": q.get("split"),
            "hit": hit,
            "returned_count": len(ev_returned),
        })

    heldout_hit_rate = (sum(answerable_hits_heldout) / len(answerable_hits_heldout)) if answerable_hits_heldout else 1.0
    all_hit_rate = (sum(answerable_hits_all) / len(answerable_hits_all)) if answerable_hits_all else 1.0
    avg_mrr = (sum(mrr_scores) / len(mrr_scores)) if mrr_scores else 0.0
    legacy_hit_rate = (sum(legacy_hits) / len(legacy_hits)) if legacy_hits else 1.0

    off_topic_acc = (sum(off_topic_abstentions) / len(off_topic_abstentions)) if off_topic_abstentions else 1.0
    borderline_acc = (sum(borderline_abstentions) / len(borderline_abstentions)) if borderline_abstentions else 1.0
    false_abstention_rate = (sum(false_abstentions) / len(false_abstentions)) if false_abstentions else 0.0

    p50_ms = compute_percentile(latencies_ms, 50)
    p95_ms = compute_percentile(latencies_ms, 95)
    p99_ms = compute_percentile(latencies_ms, 99)

    # Gate verification
    gate_checks = {
        "heldout_evidence_hit_at_4": {
            "value": heldout_hit_rate,
            "threshold": gates.get("heldout_evidence_hit_at_4", 0.85),
            "pass": heldout_hit_rate >= gates.get("heldout_evidence_hit_at_4", 0.85),
        },
        "off_topic_abstention": {
            "value": off_topic_acc,
            "threshold": gates.get("off_topic_abstention", 1.0),
            "pass": off_topic_acc >= gates.get("off_topic_abstention", 1.0),
        },
        "borderline_abstention": {
            "value": borderline_acc,
            "threshold": gates.get("borderline_abstention", 0.90),
            "pass": borderline_acc >= gates.get("borderline_abstention", 0.90),
        },
        "false_abstention_max": {
            "value": false_abstention_rate,
            "threshold": gates.get("false_abstention_max", 0.10),
            "pass": false_abstention_rate <= gates.get("false_abstention_max", 0.10),
        },
        "retrieval_p95_ms": {
            "value": p95_ms,
            "threshold": gates.get("retrieval_p95_ms", 800.0),
            "pass": p95_ms <= gates.get("retrieval_p95_ms", 800.0),
        },
    }

    # Latency waiver handling
    waiver_info = None
    if not gate_checks["retrieval_p95_ms"]["pass"] and latency_waiver_reason and latency_waiver_approver:
        gate_checks["retrieval_p95_ms"]["pass"] = True
        waiver_info = {
            "reason": latency_waiver_reason,
            "approved_by": latency_waiver_approver,
            "waived_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    all_gates_pass = all(gc["pass"] for gc in gate_checks.values())

    # Determine status
    if not all_gates_pass:
        quality_status = "failed"
    elif review_coverage < gates.get("min_review_coverage", 0.80):
        quality_status = "needs_review"
    else:
        quality_status = "passed"

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": get_hardware_info(),
        "total_queries": len(questions),
        "total_latency_samples": len(latencies_ms),
        "metrics": {
            "heldout_evidence_hit_at_4": heldout_hit_rate,
            "all_evidence_hit_at_4": all_hit_rate,
            "legacy_source_hit": legacy_hit_rate,
            "mrr": avg_mrr,
            "off_topic_abstention": off_topic_acc,
            "borderline_abstention": borderline_acc,
            "false_abstention_rate": false_abstention_rate,
            "latency_p50_ms": p50_ms,
            "latency_p95_ms": p95_ms,
            "latency_p99_ms": p99_ms,
            "review_coverage": review_coverage,
        },
        "gates": gate_checks,
        "latency_waiver": waiver_info,
        "quality_status": quality_status,
    }

    report_path = artifact_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Update manifest
    manifest_file = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["quality_status"] = quality_status
    manifest["eval_report"] = "eval_report.json"
    manifest["gates_used"] = gates
    manifest["latency_waiver"] = waiver_info
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if mock_retriever is None and hasattr(retriever, "close"):
        retriever.close()

    return report, (quality_status in ("passed", "needs_review"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate knowledge artifact against quality gates")
    parser.add_argument("version", help="Artifact version (hex or staging/version)")
    parser.add_argument("--latency-waiver", dest="latency_waiver_reason", default=None, help="Reason for latency waiver")
    parser.add_argument("--approved-by", dest="latency_waiver_approver", default=None, help="Name of latency waiver approver")
    parser.add_argument("--questions", default=None, help="Custom questions yaml path")
    parser.add_argument("--gates", default=None, help="Custom gates yaml path")

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
    g_path = Path(args.gates) if args.gates else None

    report, passed = run_evaluation(
        artifact_dir=artifact_dir,
        settings=settings,
        questions_path=q_path,
        gates_path=g_path,
        latency_waiver_reason=args.latency_waiver_reason,
        latency_waiver_approver=args.latency_waiver_approver,
    )

    print(json.dumps(report["metrics"], indent=2))
    print(f"\nFinal Quality Status: {report['quality_status']}")
    if not passed or report["quality_status"] == "failed":
        print("FAILED: Artifact failed quality gates.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
