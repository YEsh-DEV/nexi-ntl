# Runbook: Upstream Provider Outage Triage and Handling

## Symptom
One or more upstream cloud providers (Deepgram STT, Google Gemini LLM, Sarvam TTS, or LiveKit Cloud) is degraded or experiencing an outage. Callers hear apologies or connection fails.

## Check
1. Query deep readiness probe on Token Service:
```bash
curl -s http://127.0.0.1:8080/readyz
```
2. Run provider smoke check:
```bash
python scripts/smoke_providers.py
```
3. Check recent application error logs:
```bash
docker compose logs --tail=50 agent
```

## Command

### 1. Identifying the Impacted Provider
- **Deepgram STT Outage:**
  - *Logs:* `deepgram connection error`, `websocket closed 1006`.
  - *Call experience:* Agent does not react to user speech; user hears silence until 8s inactivity prompt plays: *"Are you still there? I can help with admissions, hostels, or fees."*
- **Google Gemini LLM Outage:**
  - *Logs:* `google.api_core.exceptions`, `503 Service Unavailable`, `429 Resource Exhausted`.
  - *Call experience:* User speaks, agent pauses and speaks fallback apology: *"I am having trouble looking that up right now. Could you please repeat your question?"*
- **Sarvam TTS Outage:**
  - *Logs:* `Sarvam TTS failed`, `500 Internal Server Error`, `HTTP 502`.
  - *Call experience:* Agent generates text but audio synthesis fails; session logs error and aborts turn without speech.
- **LiveKit Server Outage:**
  - *Logs:* `Cannot connect to LiveKit server`, `Failed querying LiveKit room list`.
  - *Readiness:* `/readyz` returns HTTP 503 `livekit: fail`.

### 2. Operational Mitigation
- If Google Gemini LLM returns 429 quota exhaustion, switch `LLM_MODEL` in `.env.local` to alternate model (e.g. `gemini-2.5-flash` or `gemini-2.0-flash`):
  ```bash
  # In .env.local:
  # LLM_MODEL=gemini-2.5-flash
  docker compose restart agent
  ```
- If token requests should be temporarily paused to prevent caller frustration:
  ```bash
  docker compose stop token
  ```

## Expected result
- When healthy, `/readyz` returns HTTP 200 `status: "ok"`.
- During an active provider outage, `/readyz` reports `status: "degraded"` or `"fail"` with clear indication of the failing component.
- Calls never raise unhandled exceptions; fallbacks play within bounds.

## Escalation
- Check official provider status pages:
  - Deepgram: https://status.deepgram.com/
  - Google Cloud: https://status.cloud.google.com/
  - LiveKit: https://status.livekit.io/
- Notify admissions department supervisors if voice line is affected.
