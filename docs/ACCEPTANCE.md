# Nexi Acceptance Criteria

Tracking acceptance states for all requirements across Phases 0 through 6 per the Backend Completion Specification.

States:
- **AUTOMATED**: Proven by automated offline tests/commands in the repo.
- **PENDING-HUMAN**: Requires live provider keys, real phone calls, carrier paperwork, or physical hardware. Runbook provided.
- **UNVERIFIED**: Implementation pending.

| ID | Criterion | Type | State | Evidence Link or Command |
|---|---|---|---|---|
| A0.1 | Fresh virtualenv installs from the CPU lock and tests pass, `torch.cuda.is_available()` is False | AUTOMATED | PASS | `scripts/check_lock.py` & `python -c "import torch; print(torch.cuda.is_available())"` |
| A0.2 | Empty `AGENT_NAME` fails fast without the dev override | AUTOMATED | PASS | `tests/test_config_consistency.py::TestConfigConsistency.test_agent_name_fail_fast` |
| A0.3 | Config, `.env.example` and README stay consistent | AUTOMATED | PASS | `tests/test_config_consistency.py::TestConfigConsistency.test_all_env_vars_present_in_dotenv_example` |
| A0.4 | Provider smoke script passes with real keys | PENDING-HUMAN | PENDING-HUMAN | `python scripts/smoke_providers.py` (offline mocked in `test_smoke_providers_mock_run`) |
| A1.1 | Lifecycle, duration, end-call and transfer unit tests | AUTOMATED | PASS | `tests/test_lifecycle.py` (8/8 passed) |
| A1.2 | A real phone call hits the duration limit and ends cleanly | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/sip-troubleshooting.md` |
| A1.3 | A real goodbye ends the call | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/sip-troubleshooting.md` |
| A1.4 | A real transfer to one verified number succeeds | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/sip-troubleshooting.md` |
| A1.5 | A real failed transfer returns to the conversation | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/sip-troubleshooting.md` |
| A2.1 | Switching, menu, entity-check and adapter tests | AUTOMATED | PASS | `tests/test_language.py` |
| A2.2 | Native review of Telugu and Hindi phrases | PENDING-HUMAN | PENDING-HUMAN | `agent/language.py:NEEDS_NATIVE_REVIEW` |
| A2.3 | Native review of at least 80 percent of multilingual items | PENDING-HUMAN | PENDING-HUMAN | `eval/multilingual.yaml` |
| A2.4 | Live multilingual evaluation run, results recorded | PENDING-HUMAN | PENDING-HUMAN | `python scripts/eval_multilingual.py --live` |
| A2.5 | A real phone call using the language menu | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/sip-troubleshooting.md` |
| A2.6 | A real mid-call language switch | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/sip-troubleshooting.md` |
| A2.7 | STT and TTS listening protocol completed | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/stt-eval.md` |
| A3.1 | Gate, activation, rollback, freshness and chunking tests | AUTOMATED | PASS | `tests/test_knowledge_gates.py` |
| A3.2 | Human coverage approval of the real corpus | PENDING-HUMAN | PENDING-HUMAN | `python -m rag.coverage approve` |
| A3.3 | Human review of at least 80 percent of evaluation items | PENDING-HUMAN | PENDING-HUMAN | `eval/questions.yaml` |
| A3.4 | Full evaluation on the real corpus recorded (pass or honest failure) | AUTOMATED | PENDING-HUMAN | `python -m rag.eval run` |
| A3.5 | Latency waiver decision, only if the latency gate fails | PENDING-HUMAN | PENDING-HUMAN | `manifest.json:latency_waiver` |
| A4.1 | Telemetry, migration, privacy, retention and report tests | AUTOMATED | PASS | `tests/test_telemetry_durable.py` |
| A4.2 | Saved voice latency baseline from real calls | PENDING-HUMAN | PENDING-HUMAN | `docs/baselines/voice-latency.md` |
| A4.3 | Killed process leaves an `abandoned` row after reconcile | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/database-recovery.md` |
| A5.1 | Limiter, admission, budget, validation and readiness tests | AUTOMATED | PASS | `tests/test_token_admission.py` |
| A5.2 | Real endpoint returns 429 past the limit | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/stuck-calls.md` |
| A5.3 | Real sessions fill capacity and the endpoint returns 503 | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/capacity-test.md` |
| A5.4 | Stopping the worker turns `/readyz` to 503 | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/provider-outage.md` |
| A6.1 | All 14 session-flow scenarios pass | AUTOMATED | PASS | `tests/test_session_scenarios.py` |
| A6.2 | Line coverage at least 80 percent for `agent/` and `rag/` | AUTOMATED | PASS | `coverage report` |
| A6.3 | Real-audio interruption, provider error, idle and duration checks | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/stt-eval.md` |
| A7.1 | CI green on the remote repository | PENDING-HUMAN | PENDING-HUMAN | `.github/workflows/ci.yml` |
| A7.2 | Both Docker targets build and compose config validates | AUTOMATED | PASS | `docker compose config` |
| A7.3 | `docker compose up` shows healthy services | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/start-stop.md` |
| A7.4 | Capacity measured at 1, 5 and 10 sessions | PENDING-HUMAN | PENDING-HUMAN | `scripts/capacity_report.py` |
| A7.5 | Human dry run of the runbooks | PENDING-HUMAN | PENDING-HUMAN | `docs/runbooks/` |
