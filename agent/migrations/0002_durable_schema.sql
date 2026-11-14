-- 0002_durable_schema.sql: Phase 6 durable operational schema

CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY,
    job_id TEXT,
    room_hash TEXT,
    channel TEXT,
    language_initial TEXT,
    language_final TEXT,
    status TEXT,                -- started | ended | abandoned
    started_at_utc TEXT,
    ended_at_utc TEXT,
    duration_s REAL,
    end_reason TEXT,
    caller_hash TEXT,
    caller_mask TEXT,
    turns INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    no_results_count INTEGER DEFAULT 0,
    language_switches INTEGER DEFAULT 0,
    language_switch_failures INTEGER DEFAULT 0,
    transfer_attempted INTEGER DEFAULT 0,
    transfer_succeeded INTEGER DEFAULT 0,
    transfer_department TEXT,
    room_delete_failures INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    usage_json TEXT
);

CREATE TABLE IF NOT EXISTS turn_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT REFERENCES calls(call_id),
    ts_utc TEXT,
    channel TEXT,
    language TEXT,
    eou_delay_ms REAL,
    llm_ttft_ms REAL,
    tts_ttfb_ms REAL,
    total_ms REAL,
    retrieval_ms REAL,
    tool_calls INTEGER
);

CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT UNIQUE,
    query_redacted TEXT,
    first_seen_utc TEXT,
    last_seen_utc TEXT,
    count INTEGER DEFAULT 1,
    language TEXT,
    channel TEXT,
    reason TEXT,                -- no_results | below_threshold | stale_only | timeout
    top_score REAL,
    nearest_topic TEXT,
    status TEXT DEFAULT 'open', -- open | resolved | ignored
    resolved_note TEXT,
    resolved_at_utc TEXT
);

CREATE TABLE IF NOT EXISTS daily_counters (
    day_utc TEXT,
    kind TEXT,
    count INTEGER,
    PRIMARY KEY(day_utc, kind)
);

CREATE INDEX IF NOT EXISTS idx_calls_started ON calls(started_at_utc);
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_ts ON turn_metrics(ts_utc);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_chan_lang ON turn_metrics(channel, language);
CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_status_seen ON knowledge_gaps(status, last_seen_utc);
