# Runbook: Secret and API Key Rotation

## Symptom
Provider API keys (LiveKit, Google Gemini, Deepgram, Sarvam) or internal `PII_HASH_KEY` must be rotated due to scheduled security policy, credential leakage, or team changes.

## Check
Verify current configuration validity without exposing secret values:
```bash
python scripts/smoke_providers.py
curl -s http://127.0.0.1:8080/readyz
```

## Command

### 1. Rotating Provider Keys (LiveKit, Deepgram, Gemini, Sarvam)
Update the values in `.env.local`:
```bash
# Edit .env.local with new keys
# LIVEKIT_API_KEY=...
# LIVEKIT_API_SECRET=...
# DEEPGRAM_API_KEY=...
# GOOGLE_API_KEY=...
# SARVAM_API_KEY=...
```
Restart the services sequentially:
```bash
docker compose up -d --no-deps token
docker compose up -d --no-deps agent
```

### 2. Rotating PII_HASH_KEY
Generate a fresh 32-byte hexadecimal secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Update `PII_HASH_KEY=<new_key>` in `.env.local` and restart services:
```bash
docker compose up -d
```

> [!IMPORTANT]
> **Operational Impact of Rotating `PII_HASH_KEY`:**
> Changing `PII_HASH_KEY` alters HMAC-SHA256 digests generated for telephone numbers and SIP identities. Existing records in `calls.caller_hash` will no longer correlate with new calls from the same phone number. Historical calls remain preserved and intact; operators must accept that cross-rotation frequency queries will treat existing callers as new callers.

## Expected result
- `scripts/smoke_providers.py` passes all connectivity checks with new credentials.
- `/readyz` endpoint returns HTTP 200 with `providers.status: "ok"`.
- Log emissions show no authentication errors (401/403) from providers.

## Escalation
- If LiveKit returns 401 Unauthorized, verify project URL and ensure both `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` belong to the same project.
- If Deepgram or Sarvam returns 401, test standalone curl request with the new API key.
