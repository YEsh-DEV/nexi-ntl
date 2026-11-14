"""Generate capacity and resource usage report from sampled metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from agent.config import load_settings
from agent.reports import nearest_rank_percentile


def read_samples_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Read capacity samples CSV file."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Samples CSV not found at {csv_path}")

    samples: list[dict[str, Any]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                samples.append({
                    "timestamp_utc": row["timestamp_utc"],
                    "pid": int(row.get("pid", 0)),
                    "cpu_pct": float(row["cpu_pct"]),
                    "rss_mb": float(row["rss_mb"]),
                    "active_rooms": int(row["active_rooms"]),
                })
            except (ValueError, KeyError):
                continue
    return samples


def get_latency_samples_for_window(
    db_path: Path,
    start_utc: str,
    end_utc: str,
) -> list[float]:
    """Retrieve response latency (eou_to_first_audio_ms) from turn_metrics within time window."""
    if not db_path.is_file():
        return []

    latencies: list[float] = []
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                """
                SELECT eou_to_first_audio_ms
                FROM turn_metrics
                WHERE ts_utc >= ? AND ts_utc <= ? AND eou_to_first_audio_ms IS NOT NULL;
                """,
                (start_utc, end_utc),
            )
            latencies = [float(row[0]) for row in cur.fetchall() if row[0] is not None]
        finally:
            conn.close()
    except Exception:
        return []
    return latencies


def compute_capacity_summary(
    samples: list[dict[str, Any]],
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Analyze samples by concurrency level."""
    if not samples:
        return {"concurrency_levels": {}, "idle_rss_mb": 0.0, "total_samples": 0}

    # Determine idle RSS from concurrency level 0
    idle_samples = [s["rss_mb"] for s in samples if s["active_rooms"] == 0]
    idle_rss_mb = round(sum(idle_samples) / len(idle_samples), 2) if idle_samples else 0.0

    # Group by active_rooms
    by_concurrency: dict[int, list[dict[str, Any]]] = {}
    for s in samples:
        c = s["active_rooms"]
        by_concurrency.setdefault(c, []).append(s)

    results: dict[int, dict[str, Any]] = {}
    for c in sorted(by_concurrency.keys()):
        rows = by_concurrency[c]
        n = len(rows)
        cpu_values = [r["cpu_pct"] for r in rows]
        rss_values = [r["rss_mb"] for r in rows]

        peak_rss = max(rss_values) if rss_values else 0.0
        cpu_p95 = nearest_rank_percentile(cpu_values, 95) if cpu_values else 0.0

        est_mb_per_call: float | None = None
        if c > 0 and idle_rss_mb > 0:
            est_mb_per_call = round(max(0.0, (peak_rss - idle_rss_mb) / c), 2)

        # Time range for this concurrency level
        timestamps = [r["timestamp_utc"] for r in rows]
        min_ts = min(timestamps)
        max_ts = max(timestamps)

        latency_p95: float | None = None
        if db_path is not None:
            latencies = get_latency_samples_for_window(db_path, min_ts, max_ts)
            if latencies:
                latency_p95 = nearest_rank_percentile(latencies, 95)

        results[c] = {
            "concurrency": c,
            "sample_count": n,
            "warning": "Sample size N < 30; statistical power limited" if n < 30 else None,
            "peak_rss_mb": round(peak_rss, 2),
            "estimated_mb_per_call": est_mb_per_call,
            "cpu_p95": cpu_p95,
            "latency_p95_ms": latency_p95,
        }

    return {
        "idle_rss_mb": idle_rss_mb,
        "total_samples": len(samples),
        "concurrency_levels": results,
    }


def format_text_report(summary: dict[str, Any]) -> str:
    lines = [
        "============================================================",
        "                 Nexi Capacity & Resource Report            ",
        "============================================================",
        f"Total Samples: {summary['total_samples']}",
        f"Baseline Idle Memory: {summary['idle_rss_mb']:.2f} MB",
        "",
        f"{'Concurrency':<12} {'N':<6} {'Peak RSS':<12} {'Est MB/Call':<14} {'CPU p95 (%)':<14} {'Latency p95 (ms)':<18}",
        "-" * 80,
    ]

    for c, stats in summary["concurrency_levels"].items():
        est_str = f"{stats['estimated_mb_per_call']:.2f}" if stats["estimated_mb_per_call"] is not None else "N/A"
        lat_str = f"{stats['latency_p95_ms']:.1f}" if stats["latency_p95_ms"] is not None else "N/A"
        line = (
            f"{c:<12} {stats['sample_count']:<6} {stats['peak_rss_mb']:<12.2f} "
            f"{est_str:<14} {stats['cpu_p95']:<14.1f} {lat_str:<18}"
        )
        if stats["warning"]:
            line += f"  * {stats['warning']}"
        lines.append(line)

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate capacity analysis report")
    parser.add_argument("--samples", default="data/telemetry/capacity_samples.csv", help="Samples CSV path")
    parser.add_argument("--db", default=None, help="Database path for turn_metrics latency")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()
    samples_path = Path(args.samples).resolve()

    if not samples_path.is_file():
        print(f"Error: Samples file not found: {samples_path}", file=sys.stderr)
        sys.exit(1)

    settings = load_settings()
    db_path = Path(args.db).resolve() if args.db else settings.database_path

    samples = read_samples_csv(samples_path)
    summary = compute_capacity_summary(samples, db_path=db_path)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print(format_text_report(summary))


if __name__ == "__main__":
    main()
