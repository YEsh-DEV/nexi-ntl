"""Operational maintenance routines: stale call reconciliation, retention purge, and vacuum."""

from __future__ import annotations

import argparse
import fcntl
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from agent.config import Settings, load_settings
from agent.storage import TelemetryStore

LOGGER = logging.getLogger("nexi.maintenance")


def reconcile_stale_calls(
    storage: TelemetryStore,
    settings: Settings,
    now_func: Callable[[], datetime] | None = None,
) -> int:
    """Mark started calls exceeding duration threshold as abandoned.

    Guarded by file lock.
    """
    now = now_func() if now_func else datetime.now(timezone.utc)
    max_duration = max(settings.max_call_seconds_web, settings.max_call_seconds_phone)
    threshold_seconds = max_duration * 2 + 120
    cutoff_time = (now - timedelta(seconds=threshold_seconds)).isoformat()

    lock_file = storage.db_dir / "reconcile.lock"
    with open(lock_file, "w", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process is actively reconciling; exit cleanly
            return 0

        try:
            with storage._connection() as conn:
                cursor = conn.execute(
                    """
                    UPDATE calls SET
                        status = 'abandoned',
                        end_reason = 'unknown_process_exit',
                        ended_at_utc = ?
                    WHERE status = 'started' AND started_at_utc < ?;
                    """,
                    (now.isoformat(), cutoff_time),
                )
                return cursor.rowcount
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except OSError:
                pass


def purge_retention(
    storage: TelemetryStore,
    settings: Settings,
    dry_run: bool = False,
    vacuum: bool = False,
    now_func: Callable[[], datetime] | None = None,
) -> dict[str, int]:
    """Purge old calls, metrics, and resolved knowledge gaps in 1000-row batches."""
    now = now_func() if now_func else datetime.now(timezone.utc)
    calls_cutoff = (now - timedelta(days=settings.call_log_retention_days)).isoformat()
    metrics_cutoff = (now - timedelta(days=settings.turn_metrics_retention_days)).isoformat()
    gaps_cutoff = (now - timedelta(days=settings.knowledge_gap_retention_days)).isoformat()

    counts = {"calls": 0, "turn_metrics": 0, "knowledge_gaps": 0}

    with storage._connection() as conn:
        # 1. Turn Metrics
        cur = conn.execute("SELECT COUNT(*) FROM turn_metrics WHERE ts_utc < ?;", (metrics_cutoff,))
        counts["turn_metrics"] = cur.fetchone()[0]
        if not dry_run and counts["turn_metrics"] > 0:
            while True:
                del_cur = conn.execute(
                    """
                    DELETE FROM turn_metrics WHERE id IN (
                        SELECT id FROM turn_metrics WHERE ts_utc < ? LIMIT 1000
                    );
                    """,
                    (metrics_cutoff,),
                )
                if del_cur.rowcount == 0:
                    break

        # 2. Calls
        cur = conn.execute("SELECT COUNT(*) FROM calls WHERE started_at_utc < ?;", (calls_cutoff,))
        counts["calls"] = cur.fetchone()[0]
        if not dry_run and counts["calls"] > 0:
            while True:
                del_cur = conn.execute(
                    """
                    DELETE FROM calls WHERE call_id IN (
                        SELECT call_id FROM calls WHERE started_at_utc < ? LIMIT 1000
                    );
                    """,
                    (calls_cutoff,),
                )
                if del_cur.rowcount == 0:
                    break

        # 3. Knowledge Gaps (ONLY resolved and ignored; OPEN GAPS ARE ALWAYS KEPT)
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM knowledge_gaps
            WHERE status IN ('resolved', 'ignored') AND last_seen_utc < ?;
            """,
            (gaps_cutoff,),
        )
        counts["knowledge_gaps"] = cur.fetchone()[0]
        if not dry_run and counts["knowledge_gaps"] > 0:
            while True:
                del_cur = conn.execute(
                    """
                    DELETE FROM knowledge_gaps WHERE id IN (
                        SELECT id FROM knowledge_gaps
                        WHERE status IN ('resolved', 'ignored') AND last_seen_utc < ?
                        LIMIT 1000
                    );
                    """,
                    (gaps_cutoff,),
                )
                if del_cur.rowcount == 0:
                    break

    if vacuum and not dry_run:
        conn_raw = sqlite3.connect(str(storage.db_path))
        try:
            conn_raw.execute("VACUUM;")
        finally:
            conn_raw.close()

    return counts


def run_loop(
    storage: TelemetryStore,
    settings: Settings,
    reconcile_interval_s: float = 300.0,
    purge_interval_s: float = 86400.0,
    clock_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_iterations: int | None = None,
) -> None:
    """Continuous maintenance daemon running reconcile and purge at configured intervals."""
    LOGGER.info(
        "Starting maintenance daemon (reconcile every %ss, purge every %ss)",
        reconcile_interval_s,
        purge_interval_s,
    )
    last_reconcile = -reconcile_interval_s
    last_purge = -purge_interval_s
    iterations = 0

    while True:
        if max_iterations is not None and iterations >= max_iterations:
            break
        iterations += 1
        now = clock_fn()

        if now - last_reconcile >= reconcile_interval_s:
            try:
                rec_count = reconcile_stale_calls(storage, settings)
                LOGGER.info("Maintenance loop reconciled %d stale call(s)", rec_count)
            except Exception as exc:
                LOGGER.error("Maintenance loop reconcile error: %s", exc)
            last_reconcile = now

        if now - last_purge >= purge_interval_s:
            try:
                p_res = purge_retention(storage, settings)
                LOGGER.info(
                    "Maintenance loop purged calls=%d metrics=%d gaps=%d",
                    p_res["calls"],
                    p_res["turn_metrics"],
                    p_res["knowledge_gaps"],
                )
            except Exception as exc:
                LOGGER.error("Maintenance loop purge error: %s", exc)
            last_purge = now

        now_after = clock_fn()
        time_to_reconcile = max(0.0, reconcile_interval_s - (now_after - last_reconcile))
        time_to_purge = max(0.0, purge_interval_s - (now_after - last_purge))
        sleep_dur = max(0.1, min(time_to_reconcile, time_to_purge))
        sleep_fn(sleep_dur)


def main() -> None:
    parser = argparse.ArgumentParser(description="Operational telemetry maintenance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reconcile", help="Reconcile orphaned started calls to abandoned")

    purge_p = subparsers.add_parser("purge", help="Purge telemetry exceeding retention limits")
    purge_p.add_argument("--dry-run", action="store_true", help="Print counts without deleting")
    purge_p.add_argument("--vacuum", action="store_true", help="Run SQLite VACUUM after deletion")

    subparsers.add_parser("run", help="Run continuous maintenance loop daemon")

    args = parser.parse_args()
    settings = load_settings()
    storage = TelemetryStore(settings)

    if args.command == "reconcile":
        count = reconcile_stale_calls(storage, settings)
        print(f"Reconciled {count} abandoned call(s).")
    elif args.command == "purge":
        res = purge_retention(storage, settings, dry_run=args.dry_run, vacuum=args.vacuum)
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"{action}: {res['calls']} calls, {res['turn_metrics']} metrics, {res['knowledge_gaps']} resolved/ignored gaps.")
    elif args.command == "run":
        try:
            run_loop(storage, settings)
        except KeyboardInterrupt:
            print("Maintenance daemon stopped by operator.")


if __name__ == "__main__":
    main()
