-- 0001_baseline.sql: Reproduce existing schema exactly
CREATE TABLE IF NOT EXISTS call_logs (
    call_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    channel TEXT NOT NULL,
    language TEXT NOT NULL,
    status TEXT NOT NULL,
    turns INTEGER NOT NULL,
    tool_calls INTEGER NOT NULL,
    no_results INTEGER NOT NULL,
    errors INTEGER NOT NULL,
    timeouts INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    knowledge_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    usage_summary TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    language TEXT NOT NULL,
    stage TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_gaps_legacy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL,
    language TEXT NOT NULL,
    knowledge_version TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
