"""Sample CPU, RSS memory, and active room count for capacity profiling."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil

from agent.config import Settings, load_settings

LOGGER = logging.getLogger("nexi.capacity_sample")


def find_worker_pid() -> int | None:
    """Find PID of running LiveKit agent worker process."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] == current_pid:
                continue
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "agent.main" in cmd or "agent/main.py" in cmd:
                return int(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def get_process_tree_metrics(parent_pid: int) -> tuple[float, float]:
    """Compute combined CPU% and RSS memory (MB) for process and all its children."""
    try:
        parent = psutil.Process(parent_pid)
        procs = [parent] + parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0, 0.0

    total_rss_bytes = 0
    total_cpu = 0.0

    for p in procs:
        try:
            total_rss_bytes += p.memory_info().rss
            total_cpu += p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    total_rss_mb = round(total_rss_bytes / (1024 * 1024), 2)
    return round(total_cpu, 1), total_rss_mb


async def fetch_active_room_count(settings: Settings, timeout_s: float = 2.0) -> int:
    """Query LiveKit API for active rooms count with fallback to 0."""
    if not settings.livekit_url or not settings.livekit_api_key or not settings.livekit_api_secret:
        return 0
    import asyncio

    from livekit import api

    try:
        async with asyncio.timeout(timeout_s):
            lk_api = api.LiveKitAPI(
                settings.livekit_url,
                settings.livekit_api_key,
                settings.livekit_api_secret,
            )
            try:
                res = await lk_api.room.list_rooms(api.ListRoomsRequest())
                return len([r for r in res.rooms if r.num_participants > 0])
            finally:
                await lk_api.aclose()
    except Exception as exc:
        LOGGER.warning("Could not query LiveKit rooms: %s", exc)
        return 0


def sample_step(
    pid: int,
    get_metrics_fn: Callable[[int], tuple[float, float]] = get_process_tree_metrics,
    get_rooms_fn: Callable[[], int] = lambda: 0,
) -> dict[str, Any]:
    """Capture a single capacity measurement sample."""
    ts_utc = datetime.now(timezone.utc).isoformat()
    cpu_pct, rss_mb = get_metrics_fn(pid)
    active_rooms = get_rooms_fn()
    return {
        "timestamp_utc": ts_utc,
        "pid": pid,
        "cpu_pct": cpu_pct,
        "rss_mb": rss_mb,
        "active_rooms": active_rooms,
    }


def run_sampler(
    pid: int,
    output_path: Path,
    interval_s: float = 2.0,
    duration_s: float | None = None,
    clock_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    get_metrics_fn: Callable[[int], tuple[float, float]] = get_process_tree_metrics,
    get_rooms_fn: Callable[[], int] = lambda: 0,
    max_samples: int | None = None,
) -> int:
    """Run sampling loop and append rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    fieldnames = ["timestamp_utc", "pid", "cpu_pct", "rss_mb", "active_rooms"]
    samples_count = 0
    start_time = clock_fn()

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        while True:
            if max_samples is not None and samples_count >= max_samples:
                break
            if duration_s is not None and (clock_fn() - start_time) >= duration_s:
                break

            sample = sample_step(pid, get_metrics_fn=get_metrics_fn, get_rooms_fn=get_rooms_fn)
            writer.writerow(sample)
            f.flush()
            samples_count += 1

            sleep_fn(interval_s)

    return samples_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Capacity metrics sampler for Nexi worker")
    parser.add_argument("--pid", type=int, default=None, help="Worker PID (auto-detected if omitted)")
    parser.add_argument("--output", type=str, default="data/telemetry/capacity_samples.csv", help="CSV output path")
    parser.add_argument("--interval", type=float, default=2.0, help="Sample interval seconds (default: 2.0)")
    parser.add_argument("--duration", type=float, default=None, help="Total duration seconds (default: run until stopped)")

    args = parser.parse_args()
    pid = args.pid or find_worker_pid()
    if not pid:
        print("Error: Could not find running Nexi worker process. Specify --pid manually.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).resolve()
    print(f"Sampling worker PID {pid} every {args.interval}s into {output_path}...")

    settings = load_settings()

    def _sync_room_count() -> int:
        import asyncio
        try:
            return asyncio.run(fetch_active_room_count(settings))
        except Exception:
            return 0

    try:
        count = run_sampler(
            pid=pid,
            output_path=output_path,
            interval_s=args.interval,
            duration_s=args.duration,
            get_rooms_fn=_sync_room_count,
        )
        print(f"Sampling complete. Recorded {count} samples.")
    except KeyboardInterrupt:
        print("\nSampling stopped by user.")


if __name__ == "__main__":
    main()
