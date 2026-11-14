# Nexi Live Acceptance Pack & Operator Verification Guide

This document provides exact, step-by-step procedures for every **PENDING-HUMAN** verification item defined in [docs/ACCEPTANCE.md](file:///mnt/nexus/F-Drive/nexi-ntl/docs/ACCEPTANCE.md).

---

## 🔑 Operational Prerequisites

Before executing these acceptance tests, the human operator must ensure the following are configured:

1. **Provider Accounts & Keys:**
   - LiveKit Cloud project with websocket URL (`wss://...`) and API key/secret.
   - Deepgram API key with access to `nova-3` model.
   - Google Gemini API key with access to `gemini-3.5-flash`.
   - Sarvam AI API key with access to `bulbul:v3` and `saarika:v2`.
   - A 32-byte hexadecimal `PII_HASH_KEY` generated via `python -c "import secrets; print(secrets.token_hex(32))"`.

2. **Twilio Telephony & SIP Configuration:**
   - Twilio Elastic SIP Trunk with Origination URI pointing to `sip:<project-subdomain>.sip.livekit.cloud`.
   - Inbound trunk dispatch rule matching `AGENT_NAME=nexi` in `telephony/dispatch_rule.json`.
   - Twilio Trunk Settings -> Call Control -> **SIP REFER enabled** for office transfers.
   - At least two verified phone numbers in Twilio console (one for dialing in, one for office transfer destination).
   - `TRANSFER_TARGETS='{"admissions": "+91XXXXXXXXXX"}'` set in `.env.local`.

3. **Local Environment:**
   - Active virtual environment with dependencies installed from `requirements-cpu-linux.lock`.
   - Working copy configured with `.env.local`.

---

## 📋 Human Acceptance Matrix & Verification Procedures

### A0.4: Provider Smoke Script Connectivity
- **Objective:** Verify real provider credentials and network connectivity from deployment host.
- **Prerequisites:** All 4 provider keys populated in `.env.local`.
- **Command:**
  ```bash
  python scripts/smoke_providers.py
  ```
- **Expected Observable Result:**
  Prints:
  ```
  Running Nexi Provider Smoke Tests:
    [OK] Gemini LLM
    [OK] Sarvam TTS
    [OK] Deepgram STT
    [OK] LiveKit Server API
  --------------------------------------------------
  Smoke Test Summary: ALL PASSED
  ```
- **Evidence to Record:** Terminal output log.

---

### A1.2: Real Phone Call Duration Limit Enforcement
- **Objective:** Verify a live telephone caller receives the 30s advance warning and is disconnected cleanly at 600s.
- **Prerequisites:** Worker running with `AGENT_NAME=nexi`, `MAX_CALL_SECONDS_PHONE=600`, `CALL_WARNING_LEAD_S=30`.
- **Steps:**
  1. Call the university telephone number from a verified mobile phone.
  2. Maintain the conversation until 9 minutes 30 seconds (570s).
  3. Listen for the advance spoken warning.
  4. Continue waiting until 10 minutes (600s).
- **Expected Observable Result:**
  - At 570s, Nexi speaks: *"Please note, this call will automatically end in about thirty seconds. Thank you for calling SRM University-AP."*
  - At 600s, Nexi speaks farewell: *"Thank you for contacting SRM University-AP. Have a wonderful day!"*
  - Carrier line disconnects.
  - Worker logs show `call_ended` with `reason: max_duration`.
- **Evidence to Record:** Worker log snippet containing `call_ending` and `call_ended` with `reason: "max_duration"`.

---

### A1.3: Explicit User Goodbye Termination
- **Objective:** Verify saying goodbye triggers immediate farewell and clean teardown without lingering room.
- **Steps:**
  1. Dial into Nexi via phone or web console (`http://localhost:8080/`).
  2. Ask: *"What are the hostel timings?"* Nexi answers.
  3. Say: *"Thank you, that is all. Goodbye!"*
- **Expected Observable Result:**
  - Nexi speaks farewell: *"Thank you for contacting SRM University-AP. Have a wonderful day!"*
  - WebRTC session closes immediately.
  - Worker logs show `call_ended` with `reason: user_ended`.
- **Evidence to Record:** Audio recording or console log showing `session_close: ok` and `reason: user_ended`.

---

### A1.4: Real Telephony SIP REFER Transfer (Success)
- **Objective:** Verify office transfer to admissions desk via carrier SIP REFER.
- **Prerequisites:** `TRANSFER_TARGETS='{"admissions": "<VERIFIED_PHONE_NUMBER>"}'` set in `.env.local`.
- **Steps:**
  1. Call the telephone number.
  2. Say: *"I need to speak directly with an admissions officer."*
- **Expected Observable Result:**
  - Nexi speaks transfer notice: *"I will transfer you to the admissions office right now. Please hold on."*
  - Mobile carrier screen switches or forwards to the verified admissions telephone number.
  - Admissions phone rings and connects.
  - Nexi teardown logs show `call_ended` with `reason: transferred`.
- **Evidence to Record:** Twilio Trunk Call Log showing SIP 202 Accepted / NOTIFY event and target phone ring screenshot.

---

### A1.5: Real Telephony SIP REFER Transfer (Failure / Busy Rollback)
- **Objective:** Verify failed or unassigned transfer returns caller to voice assistant.
- **Steps:**
  1. Configure `TRANSFER_TARGETS='{"admissions": "+910000000000"}'` (an invalid/unreachable destination).
  2. Call Nexi and request transfer to admissions.
- **Expected Observable Result:**
  - Nexi attempts transfer; after 10s timeout, Nexi says: *"I apologize, but the office line is currently busy or unavailable. How else may I assist you today?"*
  - Call remains connected; caller can continue asking questions.
- **Evidence to Record:** Worker log showing `transfer_failed` and conversation continuation.

---

### A2.2 & A2.3: Native Human Review of Telugu & Hindi Translations
- **Objective:** Verify grammatical, cultural, and phonetic correctness of Indian language strings.
- **Steps:**
  1. Provide [agent/language.py](file:///mnt/nexus/F-Drive/nexi-ntl/agent/language.py) phrase catalog (`PHRASES`) and [eval/multilingual.yaml](file:///mnt/nexus/F-Drive/nexi-ntl/eval/multilingual.yaml) to native Telugu and Hindi speakers.
  2. Review all phrases marked `NEEDS_NATIVE_REVIEW`.
  3. Check university terms: "ఫీజులు" (fees), "హాస్టల్" (hostel), "హాజరు" (attendance), "ప్రవేశాలు" (admissions).
- **Expected Observable Result:**
  - Sign-off on phrasing accuracy with any phonetic pronunciation adjustments recorded.
- **Evidence to Record:** Signed review notes or PR with comments from native reviewers.

---

### A2.4: Live Multilingual Provider Evaluation
- **Objective:** Measure entity preservation and semantic accuracy on live models.
- **Command:**
  ```bash
  python scripts/eval_multilingual.py --live
  ```
- **Expected Observable Result:**
  - All 50 queries evaluated; entity check accuracy >= 90%.
- **Evidence to Record:** Generated `eval/multilingual_report.json`.

---

### A2.5: Inbound Phone Call DTMF Language Menu
- **Objective:** Verify keypad selection on incoming telephone calls.
- **Steps:**
  1. Dial into Nexi phone line without metadata.
  2. Nexi speaks: *"Welcome to SRM University-AP. For English, press 1. తెలుగు కొరకు, 2 నొక్కండి. हिंदी के लिए, 3 दबाएं."*
  3. Press `2` on phone keypad.
- **Expected Observable Result:**
  - Nexi switches to Telugu and speaks: *"నమస్కారం! నేను నెక్సి, SRM యూనివర్సిటీ-AP అసిస్టెంట్‌ని..."*
- **Evidence to Record:** Audio recording of IVR selection.

---

### A2.6: Mid-Call Language Switch Handover
- **Objective:** Verify language switch mid-conversation preserves context.
- **Steps:**
  1. Start call in English and ask: *"What is the hostel fee?"*
  2. Nexi answers in English.
  3. Say: *"Can you please switch to Telugu?"* or *"తెలుగులో మాట్లాడండి"*.
- **Expected Observable Result:**
  - Nexi speaks acknowledgment in Telugu: *"నేను తెలుగుకు మారుతున్నాను. నేను మీకు ఎలా సహాయపడగలను?"*
  - Subsequent questions and answers remain in Telugu.
- **Evidence to Record:** Log output showing `language_switched: en -> te`.

---

### A2.7: STT & TTS Human Listening Protocol
- **Objective:** Listen to synthesized speech over mobile handset and verify clarity.
- **Steps:**
  Follow [docs/runbooks/stt-eval.md](file:///mnt/nexus/F-Drive/nexi-ntl/docs/runbooks/stt-eval.md).
- **Evidence to Record:** Completed listening score sheet.

---

### A3.2, A3.3, A3.4 & A3.5: Real Knowledge Corpus Approval & Gates
- **Objective:** Approve coverage, review evaluation questions, and run gates on real university PDFs.
- **Commands:**
  ```bash
  # 1. Coverage approval
  python -m rag.coverage report 337ce61b42e1426896ba9cce2beaba4a
  python -m rag.coverage approve 337ce61b42e1426896ba9cce2beaba4a --approved-by "Registrar Office"

  # 2. Review eval questions
  cat eval/questions.yaml

  # 3. Full evaluation run
  python -m rag.eval 337ce61b42e1426896ba9cce2beaba4a
  ```
- **Evidence to Record:** `coverage_report.json` and `manifest.json` with `quality_status`.

---

### A4.2: Saved Voice Latency Baseline from Real Calls
- **Objective:** Establish production voice turnaround baseline.
- **Steps:**
  Conduct 30+ live interactions, then generate baseline report:
  ```bash
  python -m agent.reports latency --period 24h --save docs/baselines/voice-latency-baseline.md
  ```
- **Evidence to Record:** Committed `docs/baselines/voice-latency-baseline.md`.

---

### A4.3: Killed Process Orphaned Call Recovery
- **Objective:** Verify SIGKILL recovery leaves `abandoned` status.
- **Steps:**
  1. Start call.
  2. Kill worker with `kill -9 <PID>`.
  3. Run: `python -m agent.maintenance reconcile`
  4. Query: `sqlite3 data/telemetry/nexi.sqlite3 "SELECT call_id, end_reason FROM calls WHERE end_reason = 'abandoned';"`
- **Evidence to Record:** Query output showing abandoned row.

---

### A5.2, A5.3, A5.4: Live Rate Limiting, Capacity & Readiness
- **A5.2 (Rate limit):** Send 11 rapid POST requests to `/api/token`. 11th request returns HTTP 429.
- **A5.3 (Capacity):** Connect 6 simultaneous sessions when `MAX_CONCURRENT_CALLS=5`. 6th returns HTTP 503.
- **A5.4 (Readiness):** Stop agent worker process. Query `curl -i http://localhost:8080/readyz`. Returns HTTP 503.

---

### A6.3: Live Audio Interruption & Idle Warning
- **Interruption:** Talk over Nexi while Nexi is answering. Nexi stops speaking within 300ms.
- **Idle Silence:** Stay silent for 8 seconds. Nexi prompts: *"Are you still there?..."* Stay silent for 8 more seconds. Nexi speaks farewell and ends call.

---

### A7.1, A7.3, A7.4, A7.5: CI, Deployment & Capacity
- **A7.1:** Push branch to GitHub remote; verify all 6 CI workflow jobs green.
- **A7.3:** Run `docker compose up -d`; verify all 3 services report healthy status.
- **A7.4:** Run [docs/runbooks/capacity-test.md](file:///mnt/nexus/F-Drive/nexi-ntl/docs/runbooks/capacity-test.md) with 1, 5, 10 sessions. Save report.
- **A7.5:** Human operator executes dry-run of all 10 runbooks in [docs/runbooks/](file:///mnt/nexus/F-Drive/nexi-ntl/docs/runbooks/).
