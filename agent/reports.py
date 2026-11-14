"""Operational reports for latency percentiles, call summaries, and actionable knowledge gaps."""

from __future__ import annotations

import argparse
import csv
import io
import math
import platform
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent.config import load_settings
from agent.storage import TelemetryStore


def nearest_rank_percentile(values: list[float], p: float) -> float:
    """Calculate nearest-rank percentile (0 <= p <= 100).

    Rank index is ceil(P / 100 * N) - 1. Documented for reproducible reporting.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    idx = max(0, min(n - 1, math.ceil((p / 100.0) * n) - 1))
    return s[idx]


def parse_since(since_str: str) -> datetime:
    """Parse string like '7d', '24h', '30m' into datetime."""
    now = datetime.now(timezone.utc)
    since_str = since_str.strip().lower()
    if since_str.endswith("d"):
        days = int(since_str[:-1])
        return now - timedelta(days=days)
    elif since_str.endswith("h"):
        hours = int(since_str[:-1])
        return now - timedelta(hours=hours)
    elif since_str.endswith("m"):
        mins = int(since_str[:-1])
        return now - timedelta(minutes=mins)
    return now - timedelta(days=7)


def generate_latency_report(
    storage: TelemetryStore,
    since: str = "7d",
    by: str = "channel,language",
) -> list[dict[str, Any]]:
    cutoff = parse_since(since).isoformat()
    group_cols = [c.strip() for c in by.split(",") if c.strip() in ("channel", "language")]
    group_by_clause = ", ".join(group_cols) if group_cols else "channel, language"

    query = f"""
    SELECT
        {group_by_clause},
        eou_delay_ms,
        llm_ttft_ms,
        tts_ttfb_ms,
        retrieval_ms,
        total_ms
    FROM turn_metrics
    WHERE ts_utc >= ?
    ORDER BY {group_by_clause};
    """

    groups: dict[tuple[str, ...], dict[str, list[float]]] = {}
    with storage._connection() as conn:
        cursor = conn.execute(query, (cutoff,))
        for row in cursor.fetchall():
            key = tuple(str(row[i]) for i in range(len(group_cols)))
            if key not in groups:
                groups[key] = {
                    "eou_delay_ms": [],
                    "llm_ttft_ms": [],
                    "tts_ttfb_ms": [],
                    "retrieval_ms": [],
                    "total_ms": [],
                }
            offset = len(group_cols)
            if row[offset] is not None:
                groups[key]["eou_delay_ms"].append(float(row[offset]))
            if row[offset + 1] is not None:
                groups[key]["llm_ttft_ms"].append(float(row[offset + 1]))
            if row[offset + 2] is not None:
                groups[key]["tts_ttfb_ms"].append(float(row[offset + 2]))
            if row[offset + 3] is not None:
                groups[key]["retrieval_ms"].append(float(row[offset + 3]))
            if row[offset + 4] is not None:
                groups[key]["total_ms"].append(float(row[offset + 4]))

    report_rows = []
    for key in sorted(groups.keys()):
        metrics = groups[key]
        n = len(metrics["total_ms"])
        row_dict: dict[str, Any] = {
            "group": " / ".join(key) if key else "all",
            "n": n,
            "sample_size_warning": n < 30,
        }
        for metric_name, vals in metrics.items():
            row_dict[f"{metric_name}_p50"] = nearest_rank_percentile(vals, 50)
            row_dict[f"{metric_name}_p95"] = nearest_rank_percentile(vals, 95)
            row_dict[f"{metric_name}_p99"] = nearest_rank_percentile(vals, 99)
        report_rows.append(row_dict)

    return report_rows


def generate_call_summary(storage: TelemetryStore, since: str = "7d") -> dict[str, Any]:
    cutoff = parse_since(since).isoformat()
    with storage._connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM calls WHERE started_at_utc >= ?;", (cutoff,))
        rows = [dict(r) for r in cursor.fetchall()]

    total = len(rows)
    end_reasons: dict[str, int] = {}
    total_duration = 0.0
    transfer_attempts = 0
    transfer_successes = 0
    no_results_total = 0
    switches_total = 0
    abandoned_count = 0

    for r in rows:
        reason = r.get("end_reason") or "unspecified"
        end_reasons[reason] = end_reasons.get(reason, 0) + 1
        total_duration += float(r.get("duration_s") or 0.0)
        transfer_attempts += int(r.get("transfer_attempted") or 0)
        transfer_successes += int(r.get("transfer_succeeded") or 0)
        no_results_total += int(r.get("no_results_count") or 0)
        switches_total += int(r.get("language_switches") or 0)
        if r.get("status") == "abandoned":
            abandoned_count += 1

    return {
        "period_since": since,
        "total_calls": total,
        "abandoned_calls": abandoned_count,
        "average_duration_s": round(total_duration / total, 2) if total else 0.0,
        "end_reasons": end_reasons,
        "transfer_attempts": transfer_attempts,
        "transfer_success_rate": round(transfer_successes / (transfer_attempts or 1), 3),
        "no_results_total": no_results_total,
        "language_switches_total": switches_total,
    }


def format_latency_text(rows: list[dict[str, Any]]) -> str:
    out = []
    for r in rows:
        out.append(f"=== Group: {r['group']} (N={r['n']}) ===")
        if r["sample_size_warning"]:
            out.append("  [WARNING: N < 30 samples, percentiles are indicative only]")
        for stage in ("eou_delay", "llm_ttft", "tts_ttfb", "retrieval", "total"):
            p50 = r.get(f"{stage}_ms_p50", 0.0)
            p95 = r.get(f"{stage}_ms_p95", 0.0)
            p99 = r.get(f"{stage}_ms_p99", 0.0)
            out.append(f"  {stage:<12}: p50={p50:6.1f}ms  p95={p95:6.1f}ms  p99={p99:6.1f}ms")
        out.append("")
    return "\n".join(out)


def format_latency_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Group | N | Total p50 | Total p95 | LLM TTFT p95 | TTS TTFB p95 | Retrieval p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        warn = " *" if r["sample_size_warning"] else ""
        lines.append(
            f"| {r['group']} | {r['n']}{warn} | {r.get('total_ms_p50', 0):.1f}ms | "
            f"{r.get('total_ms_p95', 0):.1f}ms | {r.get('llm_ttft_ms_p95', 0):.1f}ms | "
            f"{r.get('tts_ttfb_ms_p95', 0):.1f}ms | {r.get('retrieval_ms_p95', 0):.1f}ms |"
        )
    return "\n".join(lines)


def format_gaps_csv(gaps: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=["id", "count", "status", "reason", "language", "channel", "query_redacted", "first_seen_utc", "last_seen_utc"],
    )
    writer.writeheader()
    for g in gaps:
        writer.writerow({
            "id": g.get("id"),
            "count": g.get("count"),
            "status": g.get("status"),
            "reason": g.get("reason"),
            "language": g.get("language"),
            "channel": g.get("channel"),
            "query_redacted": g.get("query_redacted"),
            "first_seen_utc": g.get("first_seen_utc"),
            "last_seen_utc": g.get("last_seen_utc"),
        })
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Operational telemetry reporting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lat_p = subparsers.add_parser("latency", help="Latency percentiles by group")
    lat_p.add_argument("--since", default="7d", help="Window e.g. 7d, 24h")
    lat_p.add_argument("--by", default="channel,language", help="Grouping fields")
    lat_p.add_argument("--format", default="text", choices=["text", "md", "csv"], help="Output format")
    lat_p.add_argument("--save", default=None, help="Save baseline markdown file")

    sum_p = subparsers.add_parser("summary", help="Call outcome summary")
    sum_p.add_argument("--since", default="7d", help="Window e.g. 7d, 24h")

    gap_p = subparsers.add_parser("gaps", help="Actionable knowledge gaps")
    gap_p.add_argument("--status", default=None, choices=["open", "resolved", "ignored"], help="Filter by status")
    gap_p.add_argument("--limit", type=int, default=50, help="Max entries")
    gap_p.add_argument("--format", default="text", choices=["text", "csv"], help="Output format")

    set_gap_p = subparsers.add_parser("gap-set", help="Update knowledge gap status")
    set_gap_p.add_argument("id", type=int, help="Gap ID")
    set_gap_p.add_argument("--status", required=True, choices=["open", "resolved", "ignored"])
    set_gap_p.add_argument("--note", default=None, help="Resolution note")

    args = parser.parse_args()
    settings = load_settings()
    storage = TelemetryStore(settings)

    if args.command == "latency":
        rows = generate_latency_report(storage, since=args.since, by=args.by)
        if args.format == "text":
            content = format_latency_text(rows)
            print(content)
        elif args.format == "md":
            content = format_latency_markdown(rows)
            print(content)
        elif args.format == "csv":
            content = ""  # csv format
            print("CSV not implemented for latency")

        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            md_doc = f"# Voice Latency Baseline - {datetime.now(timezone.utc).date().isoformat()}\n\n"
            md_doc += f"Platform: {platform.platform()}\nPython: {sys.version.split()[0]}\n\n"
            md_doc += format_latency_markdown(rows)
            save_path.write_text(md_doc, encoding="utf-8")
            print(f"Saved baseline report to {save_path}")

    elif args.command == "summary":
        summary = generate_call_summary(storage, since=args.since)
        import json
        print(json.dumps(summary, indent=2))

    elif args.command == "gaps":
        gaps = storage.query_knowledge_gaps(limit=args.limit, status=args.status)
        if args.format == "text":
            print(f"=== Knowledge Gaps ({len(gaps)} items) ===")
            for g in gaps:
                print(f"[{g['status'].upper()}] Count: {g['count']:<4} Lang: {g['language']} Reason: {g['reason']}")
                print(f"  Query: {g['query_redacted']}")
                if g.get("resolved_note"):
                    print(f"  Note: {g['resolved_note']}")
                print("")
        elif args.format == "csv":
            print(format_gaps_csv(gaps))

    elif args.command == "gap-set":
        ok = storage.update_gap_status(args.id, args.status, note=args.note)
        if ok:
            print(f"Updated gap {args.id} to '{args.status}'")
        else:
            print(f"Gap {args.id} not found", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
