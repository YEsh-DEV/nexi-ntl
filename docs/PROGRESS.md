# Engineering Progress Log

Tracks each Work Package (WP) execution, changes, test counts, commands run, and acceptance status.

## WP0 - Baseline, configuration and reproducibility hygiene - 2026-11-14
Status: done
Changed:
- docs/ACCEPTANCE.md: seeded criteria matrix A0.1-A7.5, updated A0.1-A0.4
- docs/DECISIONS.md: created decision log with naming and test framework choices
- docs/PROGRESS.md: progress tracker initialized
- agent/config.py: added ENV_VARS registry, validation for AGENT_NAME fail-fast and bounds
- .env.example: grouped sections, complete ENV_VARS documented with safe defaults
- README.md: status matrix, CPU-only installation guide, dynamic settings table markers
- requirements-cpu.lock: compiled strictly CPU-only transitive lock without nvidia/cuda
- requirements-cpu-linux.lock: linux-compatible CPU-only lock
- scripts/check_lock.py: verification script ensuring zero GPU/CUDA dependencies
- scripts/gen_settings_table.py: automated generator for README settings table
- scripts/smoke_providers.py: provider smoke test script with --skip option
- tests/test_config_consistency.py: 6 unit tests verifying config, README, lock, and smoke mocks
Tests: before 38, after 44. Command used: `.venv/bin/python -m unittest discover -s tests`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
SDK facts verified:
- `torch.cuda.is_available() == False` with torch==2.8.0+cpu
- Zero `nvidia-*` or `triton` packages in requirements-cpu.lock
## WP1 - Call lifecycle, duration limit, explicit end and office transfer - 2026-11-14
Status: done
Changed:
- agent/lifecycle.py: CallLifecycle controller, EndReason enum, bounded steps, duration limiter, inactivity monitor, office transfer
- agent/language.py: unified phrase table and LanguageState tracking
- agent/assistant.py: end_call function tool, transfer_to_office tool, CallState lifecycle integration
- agent/prompts.py: prompt guidance for end_call and transfer_to_office
- agent/main.py: centralized all teardown into CallLifecycle
- tests/test_lifecycle.py: 8 unit tests covering idempotence, duration limits, timeouts, transfer, and source scan
Tests: before 44, after 52. Command used: `.venv/bin/python -m unittest discover -s tests`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
SDK facts verified:
- LiveKitAPI.sip.transfer_sip_participant (livekit-api==1.0.5)
- LiveKitAPI.room.delete_room (livekit-api==1.0.5)
Acceptance IDs touched: A1.1 AUTOMATED, A1.2 PENDING-HUMAN, A1.3 PENDING-HUMAN, A1.4 PENDING-HUMAN, A1.5 PENDING-HUMAN
Open issues and decisions needed from the human: Twilio Elastic SIP Trunk must support SIP REFER for live PSTN transfer (noted in ACCEPTANCE.md)
Commit: WP1: call lifecycle, duration limit, explicit end and office transfer

## WP2 - Multilingual Phase 4 completion - 2026-11-14
Status: done
Changed:
- agent/language.py: unified phrase tables for EN/TE/HI, NEEDS_NATIVE_REVIEW catalog, PhoneLanguageMenu DTMF state machine with repeat-and-default, switch_call_language agent handover controller with error recovery
- agent/providers/sarvam_adapter.py: NexiSarvamTTS adapter wrapping livekit-plugins-sarvam 1.2.6, stripping unsupported v3 payload fields and sanitizing error messages
- agent/tts_adapter.py: re-exported NexiSarvamTTS and adapter helpers
- agent/assistant.py: added switch_language function tool and passed language-specific STT/TTS models
- agent/main.py: integrated PhoneLanguageMenu for incoming SIP calls and configured language-specific Deepgram and Sarvam models
- scripts/wer.py: Word Error Rate (WER) computation module with normalized Levenshtein distance
- eval/multilingual.yaml: 50 multilingual evaluation queries across English, Telugu, and Hindi with entity verification criteria
- eval/recorded_queries.yaml: catalog of recorded sample queries for STT evaluation
- scripts/eval_multilingual.py: multilingual evaluation runner supporting both offline dry-runs and live provider calls
- docs/runbooks/stt-eval.md: runbook and human listening protocol for STT and TTS quality validation
- tests/test_language.py: 20 unit tests covering phrase completeness, phone menu DTMF and timeouts, language handover, switch limits, rollback, adapter hygiene, and WER metrics
Tests: before 52, after 72. Command used: `.venv/bin/python -m unittest discover -s tests`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
SDK facts verified:
- `rtc.Room.on("sip_dtmf_received", ...)` with `rtc.SipDTMF(digit=..., code=...)` in livekit==1.0.12
- `AgentSession.update_agent(new_agent)` with `chat_ctx` in livekit-agents==1.2.6
- `AgentSession._stt` and `AgentSession._tts` properties with agent fallback in livekit-agents==1.2.6
Acceptance IDs touched: A2.1 AUTOMATED (PASS), A2.2-A2.7 PENDING-HUMAN
Open issues and decisions needed from the human: Native speaker review for Telugu/Hindi prompt phrases in `agent/language.py:NEEDS_NATIVE_REVIEW` and test dataset in `eval/multilingual.yaml`
Commit: WP2: multilingual Phase 4 completion

## WP3 - Knowledge quality gates - 2026-11-14
Status: done
Changed:
- rag/chunking.py: implemented parent context propagation for nested JSON records (`context:` prefix), table row boundary protection and header row repetition for PDF tables, and date/topic key derivation order (record -> sidecar -> filename convention)
- rag/coverage.py: coverage audit reporting and hash-bound human approval CLI (`report`, `approve`)
- eval/coverage_expected.yaml: defined required sources and minimum chunk thresholds for knowledge validation
- eval/gates.yaml: defined quality thresholds (heldout evidence hit >= 0.85, off-topic abstention 1.0, borderline abstention >= 0.90, false abstention <= 0.10, p95 <= 800ms)
- eval/questions.yaml: expanded evaluation benchmark to 65 questions (35 answerable with `expected_evidence: [{source, page, must_contain}]`, 15 unanswerable university, 10 off-topic, 5 adversarial), stratified into 44 calibration and 21 held-out items (32.3% held out)
- rag/calibrate.py: threshold calibration runner strictly isolated from held-out items
- rag/eval.py: evidence-level evaluation runner, MRR, latency benchmarks (p50/p95/p99) with 10 warmup queries and 3 repetitions, and gate checker supporting latency waiver
- rag/ingest.py: staging build pipeline, atomic activation with hash-bound approval verification, version rollback, and last-3 retention pruning
- rag/manifest.py: runtime loading verification (refuses unverified artifacts unless development override active)
- rag/retriever.py: multi-stage freshness and temporal filtering (dropping expired, latest effective date per topic key, deterministic tie-breaking, out-of-date prefix note) and query embedding LRU cache
- agent/prompts.py: added prompt rule instructing agent to announce out-of-date note when present in evidence
- tests/test_knowledge_gates.py: 17 unit tests verifying chunking, freshness filtering, split leakage prevention, percentile math, evidence hits, activation states, atomic swap, rollback, retention pruning, and independent gate failures
Tests: before 72, after 89. Command used: `.venv/bin/python -m unittest discover -s tests`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
Acceptance IDs touched: A3.1 AUTOMATED (PASS), A3.2-A3.5 PENDING-HUMAN
Open issues and decisions needed from the human: Human coverage approval of real corpus and native/domain review of questions dataset in `eval/questions.yaml`
Commit: WP3: knowledge quality gates

## WP4 - Durable telemetry, knowledge gaps, migrations and reports - 2026-11-14
Status: done
Changed:
- pyproject.toml: enabled ruff rules `S110` and `S112` and eliminated all silent exceptions across codebase
- agent/migrations/0001_baseline.sql: baseline reproduction migration for existing schema
- agent/migrations/0002_durable_schema.sql: durable schema adding `calls`, `turn_metrics`, `knowledge_gaps`, `daily_counters`, and indexes
- agent/storage.py: robust migration runner, corrupt database recovery (`.corrupt-<UTC>`), HMAC-SHA256 caller identity hashing, telephone number masking, question PII regex redaction, durable write methods, and backward-compatible sync/async wrappers
- agent/telemetry.py: bounded telemetry pipeline (`TelemetryPipeline`) with 1000-element queue, single async writer loop, rate-limited drop logging (60s), disk spill (`telemetry_spill.jsonl`), idempotent replay, non-blocking shutdown drain, and degraded mode tracking
- agent/maintenance.py: stale call reconciliation CLI (`reconcile`) guarded by non-blocking flock marking stale started calls `abandoned`, and 1000-row batched retention purge CLI (`purge`) protecting open knowledge gaps with optional `--vacuum`
- agent/reports.py: nearest-rank latency percentiles (p50/p95/p99) with sample size warning when N < 30, text/markdown/CSV formatters, `--save` baseline option, call summary report, and knowledge gap inspection/status CLI
- agent/assistant.py: updated `CallState` with telemetry/usage attributes, integrated knowledge gap event emissions for retrieval abstentions/timeouts
- agent/lifecycle.py: integrated `call_ended` telemetry recording and `telemetry.drain` execution prior to job shutdown
- agent/main.py: initialized telemetry pipeline in prewarm, reconciled stale calls at worker start, recorded `call_started` event, and joined `EOUMetrics`, `LLMMetrics`, and `TTSMetrics` by `speech_id`
- rag/retriever.py: updated `SearchResult` and `_search_sync` to record rejection reasons (`no_results`, `below_threshold`, `stale_only`), top score, and nearest topic
- tests/test_telemetry_durable.py: 15 unit tests covering migrations, corrupt DB recovery, PII hashing, PII table scan, queue bounds, critical spill/replay, drain timeout, writer error degraded toggle, reconcile, retention purge, report percentiles, CSV output, and assistant/lifecycle telemetry integration
Tests: before 89, after 104. Command used: `.venv/bin/python -m unittest discover -s tests`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
SDK facts verified:
- LiveKit metric classes (`EOUMetrics`, `LLMMetrics`, `TTSMetrics`) carry `speech_id` attribute for turn correlation
- `metrics.UsageCollector.get_summary()` returns a dataclass serializable via `dataclasses.asdict`
Acceptance IDs touched: A4.1 AUTOMATED (PASS), A4.2 PENDING-HUMAN, A4.3 PENDING-HUMAN
Open issues and decisions needed from the human: Voice latency baseline from live calls (`docs/baselines/voice-latency.md`) and live killed process verification
Commit: WP4: durable telemetry, knowledge gaps, migrations and reports

## WP5 - Token service admission control, rate limits and readiness - 2026-11-14
Status: done
Changed:
- agent/config.py: added `token_ttl_s` bounds validation (60s to 3600s), fail-fast validation for production wildcard CORS
- agent/ratelimit.py: implemented `RateLimiter` sliding-window limiter (10 req/10 min per client IP, 60 req/min global), LRU eviction with 10k max clients, and `X-Forwarded-For` parsing strictly gated on `TRUSTED_PROXIES`
- agent/admission.py: implemented `AdmissionController` active call capacity query via LiveKit rooms API (timeout 2s, cache 5s, counting `web-` and `call-` rooms with >= 1 participant) with fail-closed semantics and development fail-open override, and daily token budget atomic quota on `daily_counters` with Retry-After to midnight
- agent/readiness.py: implemented `ReadinessChecker` providing `/healthz` (liveness) and `/readyz` (deep readiness across LiveKit API, worker HTTP probe/heartbeat, knowledge artifact status, database write check, and provider configuration) cached for 5s with zero secrets in responses
- agent/token_service.py: hardened token service with strict `Content-Type: application/json` enforcement, 16KB body cap, request body object validation, language check (`en`, `te`, `hi`), fixed channel (`web`), minimal video grants, bounded room configuration (empty_timeout 300, departure_timeout 30, max_participants 2), strict CORS with `Vary: Origin`, `X-Request-Id` headers, and keyed client hash logging
- README.md: regenerated dynamic settings table with all WP5 environment variables
- tests/test_token_admission.py: 17 unit tests covering sliding-window edges, per-IP isolation, global limits, LRU eviction, X-Forwarded-For trust rules, under/at capacity, room list fail-closed and fail-open, 5s capacity caching, daily budget exhaustion/rollover, readiness pass/fail independent scenarios, 400/413/415 validation, and CORS preflight
- tests/test_token_service.py: updated room prefix assertion (`web-`) and verified token generation
Tests: before 104, after 121. Command used: `.venv/bin/python -m unittest discover -s tests`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
SDK facts verified:
- `livekit.protocol.room.RoomConfiguration` supports `empty_timeout`, `departure_timeout`, `max_participants`
- `livekit-agents` worker runs an internal HTTP server with health routes on port 8081
Acceptance IDs touched: A5.1 AUTOMATED (PASS), A5.2-A5.4 PENDING-HUMAN
Commit: WP5: token service admission, rate limits and readiness

## WP6 - Session-flow tests for failure-prone paths - 2026-11-14
Status: done
Changed:
- tests/fakes.py: created complete test doubles harness including `FakeClock` (monotonic time and fast-forward sleep), `FakeSession` (recording spoken text, agent updates, and closes), `FakeJobContext` (mocked room, job metadata, proc userdata), `FakeRoomApi` (mocking room deletion, SIP REFER transfer, room listing with latency/failure injection), `FakeRetriever` (deterministic evidence, delay, below-threshold, stale rejection), `FakeTelemetrySink` (collecting critical and turn events with failure injection), and `ScriptedLLM` (emitting planned tool calls)
- tests/test_session_scenarios.py: 14 deterministic session-flow scenario tests covering:
  1. Complete happy-path call (greeting, retrieval, response, user "thank you goodbye", farewell spoken, session closed with USER_ENDED)
  2. Immediate interruption during greeting (teardown skipped, prompt reset, user utterance handled)
  3. Retrieval below threshold (rejection prompt, NO_RESULTS gap logged)
  4. Retrieval timeout (fallback apology, no crash, RETRIEVAL_TIMEOUT gap logged)
  5. Mid-call language switch en->te (new greeting, agent updated, context preserved)
  6. Mid-call language switch with provider failure (graceful rollback, apology in original language)
  7. Hard duration limit web 300s (warning at limit - 30s, farewell at 300s, teardown with MAX_DURATION)
  8. Hard duration limit phone 600s with room deletion attempt
  9. Inactivity timeout 8s warning + 8s termination (farewell, session close, room deletion, IDLE_TIMEOUT)
  10. Transfer success ends with TRANSFERRED; transfer failure/timeout stays in call and increments failure count
  11. Participant disconnect ends call without speech
  12. Simultaneous idle timeout and shutdown trigger exactly one teardown
  13. Telemetry records call_started at start and call_ended with exact end reason at teardown
  14. Telemetry database storage failure does not disrupt the call and does not raise
- agent/lifecycle.py: added injectable `room_api` parameter to `CallLifecycle` to allow deterministic room deletion and SIP transfer testing without network calls
- tests/test_knowledge_gates.py: added `TestBM25AndFusion` (BM25 top_k, empty queries, RRF fusion ordering), `TestCalibrationAndClis` (threshold edge cases, calibration CLI, coverage CLI, eval CLI), and `TestRetrieverSyncAndAsync` (query embedding cache, async retrieval, below threshold rejection)
- tests/test_main_wiring.py: added unit tests for `read_channel`, `mask_phone_number`, `pick_noise_cancellation`, `prewarm` error recovery, and console-mode `entrypoint`
- tests/test_token_admission.py: configured `chroma_dir` and unverified override in test fixtures ensuring `/api/health` and `/readyz` pass
- .gitignore: added `.coverage` and `.coverage.*`
Tests: before 121, after 149. Command used: `.venv/bin/coverage run -m unittest discover -s tests -p "test_*.py"`
Coverage: 80% line coverage across `agent/` and `rag/` (3311 stmts, 662 missed). Command used: `.venv/bin/coverage report --include="agent/*,rag/*"`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
Acceptance IDs touched: A6.1 AUTOMATED (PASS), A6.2 AUTOMATED (PASS), A6.3 PENDING-HUMAN
Commit: WP6: session-flow tests for failure-prone paths

## WP7 - CI, Docker, Compose, runbooks and capacity tooling - 2026-11-14
Status: done
Changed:
- .github/workflows/ci.yml: created CI pipeline with 6 jobs (`lint`, `lock`, `test`, `eval-smoke`, `eval-real`, `docker`) and pinned GitHub actions (`actions/checkout@v4.2.2`, `actions/setup-python@v5.4.0`, `docker/setup-buildx-action@v3.10.0`, `docker/build-push-action@v6.15.0`, `actions/cache@v4.2.2`) enforcing ruff (with S110/S112), lock verification, 80% line coverage gate, and container smoke checks
- Dockerfile: created multi-stage build (`builder` with Linux CPU lock, `runtime-base` with non-root user 10001, `agent` with 660s drain timeout, `token` publishing port 8080) with baked offline Hugging Face models and LiveKit agent dependency assets
- .dockerignore: added ignore rules for git, env secrets, data directory, virtualenvs, and test caches
- docker-compose.yml: configured `agent`, `token`, `maintenance`, and `ingest` services with shared named volume `nexi-data` on `/data`, graceful shutdown `stop_grace_period: 700s`, optional env files, log rotation, and documented resource limit placeholders
- agent/maintenance.py: implemented `run_loop` daemon and `run` CLI subcommand for automated periodic reconciliation (every 300s) and retention purging (every 86400s) with injectable clock/sleep
- scripts/capacity_sample.py: implemented process tree CPU and RSS memory sampler with active room count polling into CSV
- scripts/capacity_report.py: implemented capacity report generator computing baseline idle RSS, peak RSS, estimated MB per call, CPU p95, turn latency p95, and N < 30 sample size warnings
- docs/runbooks/: created 10 structured production runbooks conforming to Symptom, Check, Command, Expected result, Escalation:
  1. `start-stop.md`
  2. `rotate-keys.md`
  3. `knowledge-update.md`
  4. `provider-outage.md`
  5. `database-recovery.md`
  6. `latency-triage.md`
  7. `stuck-calls.md`
  8. `sip-troubleshooting.md`
  9. `capacity-test.md`
  10. `stt-eval.md`
- docs/DECISIONS.md: documented worker drain timeout (660s), Compose stop grace period (700s), offline model baking command (`download-files`), and psutil capacity sampling
- tests/test_runbooks_and_capacity.py: 6 unit tests validating runbook commands existence, maintenance daemon scheduling, capacity report math on synthetic fixtures, lockfile validation, and `docker compose config`
Tests: before 149, after 155. Command used: `.venv/bin/coverage run -m unittest discover -s tests -p "test_*.py"`
Coverage: 80% line coverage across `agent/` and `rag/` (3349 stmts, 671 missed). Command used: `.venv/bin/coverage report --include="agent/*,rag/*"`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
Acceptance IDs touched: A7.2 AUTOMATED (PASS), A7.1, A7.3, A7.4, A7.5 PENDING-HUMAN
Commit: WP7: CI, Docker, Compose, runbooks and capacity tooling

## 2026-11-14T03:41:00Z - Work Package WP8: Documentation Truth Pass & Live Acceptance Pack
Files touched:
- agent/main.py: refactored entrypoint into clean wiring-only flow (144 lines, strictly under 150 lines target)
- agent/assistant.py: extracted `build_providers`, `connect_participant`, `record_call_started`, and comprehensive agent function tools
- agent/lifecycle.py: added `attach_session_handlers()` to consolidate all LiveKit session event listeners and turn metrics flushing inside CallLifecycle
- README.md: documentation truth pass including test count badges (162 passing), 80% coverage, truthful status table, honest known limitations, and quickstart commands
- SYSTEM_ARCHITECTURE.md: updated architectural documentation reflecting completed phases 0-6 with explicit justified deviations
- docs/ACCEPTANCE.md: complete 33-criteria audit matrix partitioning automated tests (14 PASS) from real-world human verification (19 PENDING-HUMAN)
- docs/LIVE_ACCEPTANCE.md: step-by-step human operator manual with exact terminal commands, verification scripts, and log checks for all 19 PENDING-HUMAN criteria
- docs/HANDOFF.md: executive one-page signoff and project handoff briefing
- tests/test_main_wiring.py: added 7 comprehensive unit tests for `build_providers`, `connect_participant`, `record_call_started`, `_build_transfer_tool`, and `NexiAgent` tools
Tests: before 155, after 162. Command used: `.venv/bin/coverage run -m unittest discover -s tests -p "test_*.py"`
Coverage: 80.16% line coverage across `agent/` and `rag/` (3367 stmts, 668 missed). Command used: `.venv/bin/coverage report --include="agent/*,rag/*"`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
Acceptance IDs touched: A8.1 through A8.5, and all 33 criteria categorized into AUTOMATED (14 PASS) and PENDING-HUMAN (19 procedures documented).
Commit: WP8: documentation truth pass, live acceptance pack, and final gate

## 2026-11-14T19:30:00Z - Work Package WP9: Gap Closure, DTMF Runtime Integration & Real Corpus Evaluation
Status: done
Changed:
- docs/GAP_MATRIX.md: comprehensive traceability matrix mapping every numbered requirement (0.1 through 9.6) to exact files and lines
- agent/main.py: integrated PhoneLanguageMenu directly into live agent runtime post session.start (148 lines, strictly maintaining < 150 lines target)
- docker-compose.yml: added `--drain-timeout 660` to agent command; maintained 700s grace period (700s > 660s > 600s phone max call duration)
- .github/workflows/ci.yml: added `--branch` coverage check, split CI evaluation into offline `eval-smoke` and model-backed `eval-real` with HuggingFace caching
- rag/calibrate.py: fixed retriever lifecycle handling and unverified knowledge flag, calibrated active artifact reranker score to 3.564
- rag/eval.py: fixed retriever closing and unverified knowledge flag, evaluated active knowledge artifact across 65 queries and 195 latency samples
- tests/test_session_scenarios.py: added Web vs Phone duration limit & warning test cases and transfer timeout failure handling
- tests/test_main_wiring.py: added `test_entrypoint_phone_language_menu` testing DTMF menu execution in agent runtime
- docs/ACCEPTANCE.md: reconciled acceptance table to 36 criteria rows (11 Automated PASS, 25 PENDING-HUMAN) with honest links to calibration and eval reports
- docs/HANDOFF.md & README.md: updated test count (163 passing) and truthful criteria counts
Tests: before 162, after 163. Command used: `.venv/bin/coverage run -m unittest discover -s tests -p "test_*.py"`
Coverage: 80% line coverage across `agent/` and `rag/` (3372 stmts, 667 missed). Command used: `.venv/bin/coverage report --include="agent/*,rag/*" --fail-under=80`
Ruff: pass. Command used: `.venv/bin/ruff check agent rag tests scripts`
SDK facts verified:
- WorkerOptions(drain_timeout=660) in livekit-agents==1.4.3
- LiveKit rtc.Room sip_dtmf_received event listening in agent runtime
Acceptance IDs touched: A3.4 (updated with real corpus eval), A1.1, A3.1, A6.1, A7.2
Commit: WP9: close specification gaps, runtime DTMF menu, real corpus evaluation, and gap matrix



