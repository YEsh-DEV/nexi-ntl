# Runbook: Telemetry Database Recovery and Spill Replay

## Symptom
SQLite telemetry database (`data/telemetry/nexi.sqlite3`) is corrupt, disk I/O errors occur, or critical events spilled to disk (`data/telemetry/telemetry_spill.jsonl`) due to process crash or queue saturation.

## Check
Verify database file integrity and inspect spill file size:
```bash
sqlite3 data/telemetry/nexi.sqlite3 "PRAGMA integrity_check;"
ls -la data/telemetry/
```

## Command

### 1. Handling Database Corruption
If `sqlite3` reports `file is not a database` or corruption errors:
The telemetry store automatically detects corruption on startup, renames the corrupt file to `nexi.sqlite3.corrupt-<UTC>`, and initializes a fresh database schema.
To trigger manual quarantine and recreation:
```bash
mv data/telemetry/nexi.sqlite3 data/telemetry/nexi.sqlite3.corrupt-$(date -u +%Y%m%dT%H%M%SZ)
python -m agent.maintenance reconcile
```

### 2. Replaying Critical Telemetry Spill
When the async queue was saturated or database was temporarily unreachable, critical events (`call_started`, `call_ended`, `knowledge_gap`) were appended to `telemetry_spill.jsonl`.
Replay all valid events into SQLite:
```bash
python -c "
from agent.config import load_settings
from agent.storage import TelemetryStore
from agent.telemetry import TelemetryPipeline
settings = load_settings()
store = TelemetryStore(settings)
pipeline = TelemetryPipeline(settings, store)
replayed = pipeline.replay_spill()
print(f'Replayed {replayed} events from spill file.')
"
```

### 3. Reconciling Orphaned Calls
If an agent worker crashed or was killed mid-session (`kill -9`), started calls remain unclosed in the database. Run the reconciliation CLI:
```bash
python -m agent.maintenance reconcile
```
This acquires the file lock, identifies calls whose duration exceeds timeout bounds, and marks them `abandoned` with UTC end timestamps.

## Expected result
- `PRAGMA integrity_check;` returns `ok`.
- `reconcile` reports the count of orphaned calls marked `abandoned`.
- `telemetry_spill.jsonl` is replayed and truncated cleanly without lost events.

## Escalation
- If disk space is exhausted (`No space left on device`), inspect disk utilization: `df -h /data`.
- Run retention purge: `python -m agent.maintenance purge --vacuum`.
