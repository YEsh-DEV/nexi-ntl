# Nexi Backend Specification Gap Matrix

Comprehensive mapping of every numbered requirement in `Nexi Backend Completion Spec (Phases 0 to 6) - Execution Document for the Coding.md` to the implementation file and line number, or status.

---

## 0. Ground Rules & Evidence Policy

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 0.1 | Purpose: Complete backend implementation for Phases 0 through 6 | Project-wide | SATISFIED |
| 0.2 | Inputs: SYSTEM_ARCHITECTURE.md, Playbook, codebase | `SYSTEM_ARCHITECTURE.md`, `README.md` | SATISFIED |
| 0.3 | Known starting state audit and baseline record | `docs/PROGRESS.md:5-25` | SATISFIED |
| 0.4.1 | No fake tools or simulation stubs in production paths | `agent/assistant.py:148-320` | SATISFIED |
| 0.4.2 | Strict unhandled exception logging (zero silent try/except) | `agent/`, `rag/` (Ruff S110/S112 clean) | SATISFIED |
| 0.4.3 | Line budget: `agent/main.py` strictly $< 150$ lines | `agent/main.py:1-148` (148 lines) | SATISFIED |
| 0.4.4 | Single test framework: Python stdlib `unittest` | `tests/` (162 tests, 0 pytest) | SATISFIED |
| 0.4.5 | Decisions log tracking SDK inspections and deviations | `docs/DECISIONS.md:1-28` | SATISFIED |
| 0.5 | Evidence discipline: automated vs PENDING-HUMAN | `docs/ACCEPTANCE.md:1-47` | SATISFIED |
| 0.6 | Per-WP atomic execution and progress tracking | `docs/PROGRESS.md` | SATISFIED |
| 0.7 | Out of scope: frontend UI, real carrier telephony contracts | Explicitly excluded per instructions | SATISFIED |
| 0.8 | Sequential execution order: WP0 through WP8 | Git history (`main` 24 commits) | SATISFIED |

---

## 1. WP0 — Baseline, Configuration and Reproducibility Hygiene

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 1.1 | Record initial baseline test count and status | `docs/PROGRESS.md:6-25` | SATISFIED |
| 1.2 | Fix named-dispatch hazard: validate `AGENT_NAME` | `agent/config.py:126-133`, `telephony/dispatch_rule.json:8` | SATISFIED |
| 1.3 | Configuration consistency between Settings and `.env.example` | `agent/config.py:33-400`, `.env.example:1-190`, `tests/test_config_consistency.py:20-65` | SATISFIED |
| 1.4 | CPU-only reproducible install & lockfile validator | `requirements-cpu-linux.lock:1-157`, `scripts/check_lock.py:1-70` | SATISFIED |
| 1.5 | README truthfulness pass | `README.md:1-290` | SATISFIED |
| 1.6 | Provider smoke check script | `scripts/smoke_providers.py:1-210` | SATISFIED |
| 1.7 | WP0 Acceptance criteria (A0.1 – A0.4) | `docs/ACCEPTANCE.md:12-15` | SATISFIED |

---

## 2. WP1 — Call Lifecycle, Duration Limit, Explicit End and Office Transfer

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 2.1 | Rationale and single-loop lifecycle coordination | `agent/lifecycle.py:1-40` | SATISFIED |
| 2.2 | `CallLifecycle` state machine, watchdog, and cleanup | `agent/lifecycle.py:148-520` | SATISFIED |
| 2.3 | Hard maximum call duration (web 300s, phone 600s, warning 30s) | `agent/lifecycle.py:441-490`, `agent/config.py:145-154` | SATISFIED |
| 2.4 | Explicit `end_call` tool with 5s guard and duplicate prevention | `agent/assistant.py:274-295` | SATISFIED |
| 2.5 | Office transfer tool via SIP with number masking and allowlist | `agent/assistant.py:143-166`, `agent/lifecycle.py:320-370` | SATISFIED |
| 2.6 | Refactor `agent/main.py` to pure wiring ($< 150$ lines) | `agent/main.py:1-148` (148 lines) | SATISFIED |
| 2.7 | Unit tests with fakes and injected clock | `tests/test_lifecycle.py:1-341` (18 tests) | SATISFIED |
| 2.8 | WP1 Acceptance criteria (A1.1 – A1.5) | `docs/ACCEPTANCE.md:16-20` | SATISFIED |

---

## 3. WP2 — Multilingual Phase 4 Completion

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 3.1 | Multilingual single source of truth | `agent/language.py:1-35` | SATISFIED |
| 3.2 | Static phrases, prompts, and entity regex extraction | `agent/language.py:20-120`, `agent/prompts.py:15-42` | SATISFIED |
| 3.3 | Mid-call dynamic language switching preserving session history | `agent/language.py:280-350` | SATISFIED |
| 3.4 | Phone language menu with DTMF keypad and speech integration | `agent/language.py:130-245`, `agent/main.py:83-89` | SATISFIED |
| 3.5 | Multilingual evaluation dataset and verification script | `eval/multilingual.yaml:1-471`, `scripts/eval_multilingual.py:1-160` | SATISFIED |
| 3.6 | Sarvam adapter hygiene (clean v3 payloads and sample rates) | `agent/providers/sarvam_adapter.py:1-134` | SATISFIED |
| 3.7 | Multilingual unit tests | `tests/test_language.py:1-348` (11 tests) | SATISFIED |
| 3.8 | WP2 Acceptance criteria (A2.1 – A2.7) | `docs/ACCEPTANCE.md:21-27` | SATISFIED |

---

## 4. WP3 — Knowledge Quality Gates

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 4.1 | Multi-stage gate architecture | `rag/chunking.py`, `rag/coverage.py`, `rag/calibrate.py`, `rag/eval.py` | SATISFIED |
| 4.2 | Manifest schema versioning, checksums, and quality status | `rag/manifest.py:1-85` | SATISFIED |
| 4.3 | Ingestion staging pipeline, OCR empty page handling, and build CLI | `rag/ingest.py:1-325` | SATISFIED |
| 4.4 | Deterministic chunking, chunk ID hashing, and table preservation | `rag/chunking.py:1-311` | SATISFIED |
| 4.5 | Temporal freshness filtering (effective dates & conflicting docs) | `rag/retriever.py:75-140` | SATISFIED |
| 4.6 | Evidence-level benchmark questions, gates, calibration, and eval CLI | `eval/questions.yaml:1-1436`, `eval/gates.yaml:1-10`, `rag/calibrate.py:1-144`, `rag/eval.py:1-364` | SATISFIED |
| 4.7 | Latency triage runbook and optimization protocol | `docs/runbooks/latency-triage.md:1-51` | SATISFIED |
| 4.8 | Knowledge quality gate unit tests | `tests/test_knowledge_gates.py:1-648` (16 tests) | SATISFIED |
| 4.9 | WP3 Acceptance criteria (A3.1 – A3.5) | `docs/ACCEPTANCE.md:28-32` | SATISFIED |

---

## 5. WP4 — Durable Telemetry, Knowledge Gaps, Migrations and Reports

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 5.1 | Telemetry durability and gap analysis rationale | `agent/storage.py:1-40`, `agent/telemetry.py:1-35` | SATISFIED |
| 5.2 | Bounded async event queue (1000 items) and disk spillover | `agent/telemetry.py:40-272` | SATISFIED |
| 5.3 | SQLite WAL configuration, connection wrappers, and migrations | `agent/storage.py:45-120`, `agent/migrations/0001_baseline.sql`, `0002_durable_schema.sql` | SATISFIED |
| 5.4 | Call records from the first second and turn metric joining | `agent/storage.py:180-260`, `agent/assistant.py:94-111`, `agent/lifecycle.py:165-245` | SATISFIED |
| 5.5 | Privacy engine (HMAC-SHA256 caller hashing & PII query redaction) | `agent/storage.py:70-130` | SATISFIED |
| 5.6 | Knowledge gap tracking with rejection reason & nearest topic | `agent/assistant.py:218-245`, `agent/storage.py:290-340` | SATISFIED |
| 5.7 | Maintenance daemon (checkpoints, reconcile abandoned calls, purge) | `agent/maintenance.py:1-227` | SATISFIED |
| 5.8 | Weekly knowledge gap reporting and clustering | `agent/reports.py:1-282` | SATISFIED |
| 5.9 | Silent-exception audit: zero silent try/except blocks | Project-wide (`grep -rn "except.*pass\|suppress("` = 0) | SATISFIED |
| 5.10 | Telemetry and storage unit tests | `tests/test_telemetry_durable.py:1-458` (15 tests) | SATISFIED |
| 5.11 | WP4 Acceptance criteria (A4.1 – A4.3) | `docs/ACCEPTANCE.md:33-35` | SATISFIED |

---

## 6. WP5 — Token Service Admission Control, Rate Limits and Readiness

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 6.1 | Admission control and rate limiting rationale | `agent/ratelimit.py:1-25`, `agent/admission.py:1-25` | SATISFIED |
| 6.2 | In-memory sliding window rate limiter (IP & global) | `agent/ratelimit.py:26-136` | SATISFIED |
| 6.3 | Capacity admission check (active rooms) & atomic daily budget | `agent/admission.py:26-147` | SATISFIED |
| 6.4 | Token request validation, JSON body bounds, restrictive grants | `agent/token_service.py:90-265` | SATISFIED |
| 6.5 | Health and readiness probes (`/healthz`, `/readyz`, `/api/health`) | `agent/readiness.py:1-207`, `agent/token_service.py:130-180` | SATISFIED |
| 6.6 | Token service and admission unit tests | `tests/test_token_admission.py:1-570` (23 tests) | SATISFIED |
| 6.7 | WP5 Acceptance criteria (A5.1 – A5.4) | `docs/ACCEPTANCE.md:36-39` | SATISFIED |

---

## 7. WP6 — Session-Flow Tests for Failure-Prone Paths

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 7.1 | Testing harness and fakes | `tests/fakes.py:1-223` | SATISFIED |
| 7.2.1 | Scenario 1: Single question -> single search -> single answer | `tests/test_session_scenarios.py:59-84` | SATISFIED |
| 7.2.2 | Scenario 2: Mid-answer user speech interruption | `tests/test_session_scenarios.py:88-111` | SATISFIED |
| 7.2.3 | Scenario 3: History & counters survive mid-call language switch | `tests/test_session_scenarios.py:115-142` | SATISFIED |
| 7.2.4 | Scenario 4: Retrieval timeout recovery and tool error apology | `tests/test_session_scenarios.py:146-170` | SATISFIED |
| 7.2.5 | Scenario 5: NO_RESULTS emits knowledge gap event | `tests/test_session_scenarios.py:174-195` | SATISFIED |
| 7.2.6 | Scenario 6: Fatal provider error ends call and skips speech | `tests/test_session_scenarios.py:199-222` | SATISFIED |
| 7.2.7 | Scenario 7: Idle silence reminder then timeout termination | `tests/test_session_scenarios.py:225-253` | SATISFIED |
| 7.2.8 | Scenario 8: Hard duration limit warning & termination (web & phone) | `tests/test_session_scenarios.py:257-305` | SATISFIED |
| 7.2.9 | Scenario 9: Explicit goodbye ends call; thank-you alone does not | `tests/test_session_scenarios.py:308-335` | SATISFIED |
| 7.2.10 | Scenario 10: Office transfer success, failure, and timeout | `tests/test_session_scenarios.py:339-390` | SATISFIED |
| 7.2.11 | Scenario 11: Participant disconnecting skips farewell | `tests/test_session_scenarios.py:394-410` | SATISFIED |
| 7.2.12 | Scenario 12: Concurrent teardown idempotency | `tests/test_session_scenarios.py:414-436` | SATISFIED |
| 7.2.13 | Scenario 13: End-to-end call lifecycle telemetry recording | `tests/test_session_scenarios.py:440-488` | SATISFIED |
| 7.2.14 | Scenario 14: Telemetry storage failure does not disrupt call | `tests/test_session_scenarios.py:492-515` | SATISFIED |
| 7.3 | Code coverage gate $\ge 80\%$ line coverage | `coverage report --fail-under=80` (80.16%) | SATISFIED |
| 7.4 | WP6 Acceptance criteria (A6.1 – A6.3) | `docs/ACCEPTANCE.md:40-42` | SATISFIED |

---

## 8. WP7 — CI, Docker, Compose, Runbooks and Capacity Tooling

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 8.1 | Pinned GitHub Actions CI workflow (lint, lock, test, evals, docker) | `.github/workflows/ci.yml:1-160` | SATISFIED |
| 8.2 | Multi-stage Dockerfile with offline model baking and non-root user | `Dockerfile:1-90` | SATISFIED |
| 8.3 | Docker Compose with 4 services, shared volume, and 700s grace period | `docker-compose.yml:1-125` | SATISFIED |
| 8.4 | 10 production runbooks (Symptom, Check, Command, Result, Escalation) | `docs/runbooks/*.md` | SATISFIED |
| 8.5 | Capacity measurement tooling (process tree sampling & report) | `scripts/capacity_sample.py:1-182`, `scripts/capacity_report.py:1-181` | SATISFIED |
| 8.6 | Runbook and capacity test suite | `tests/test_runbooks_and_capacity.py:1-228` | SATISFIED |
| 8.7 | WP7 Acceptance criteria (A7.1 – A7.5) | `docs/ACCEPTANCE.md:43-47` | SATISFIED |

---

## 9. WP8 — Documentation Truth Pass and Live Acceptance Pack

| Number | Requirement | Implementing File & Line | Status |
|---|---|---|---|
| 9.1 | README rewrite with badges, truthful matrix, and quickstart | `README.md:1-290` | SATISFIED |
| 9.2 | System architecture document with justified deviations | `SYSTEM_ARCHITECTURE.md:1-220` | SATISFIED |
| 9.3 | Complete 36-row acceptance criteria matrix | `docs/ACCEPTANCE.md:1-47` | SATISFIED |
| 9.4 | Step-by-step live acceptance testing pack for human operators | `docs/LIVE_ACCEPTANCE.md:1-227` | SATISFIED |
| 9.5 | Executive handoff report and signoff briefing | `docs/HANDOFF.md:1-92` | SATISFIED |
| 9.6 | Final verification gates passing cleanly | Test suite (162 pass), Ruff clean, coverage 80.16% | SATISFIED |
