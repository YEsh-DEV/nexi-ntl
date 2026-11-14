# Runbook: Telephony and SIP Trunk Troubleshooting

## Symptom
Inbound phone calls fail to connect, drop immediately after dialing, lack audio, play Twilio trial announcements, or agent does not greet the caller.

## Check
1. Inspect Twilio SIP trunk call logs for SIP response codes (e.g. 404, 480, 503).
2. Inspect worker logs for incoming SIP participant events:
```bash
docker compose logs --tail=100 agent | grep "sip"
```
3. Verify agent dispatch name matches SIP dispatch rule:
```bash
python -c "from agent.config import load_settings; print('AGENT_NAME:', load_settings().agent_name)"
cat telephony/dispatch_rule.json
```

## Command

### SIP Symptom Resolution Matrix

| Symptom | Root Cause | Command / Check | Remediation |
|---|---|---|---|
| **Twilio call drops immediately with busy tone** | LiveKit SIP Inbound Trunk origination URI or auth credentials misconfigured | Inspect `telephony/inbound_trunk.json` | Ensure `SIP URI` in Twilio points to `sip.livekit.cloud` with matching SIP username/password. |
| **Call connects but agent never speaks** | Agent dispatch name mismatch (e.g. trunk dispatches to `nexi`, worker runs as default `agent`) | Compare `AGENT_NAME` in `.env.local` with `dispatch_rule.json` | Set `AGENT_NAME=nexi` in `.env.local` and restart worker: `docker compose restart agent`. |
| **Call connects with "You have a trial account" audio** | Twilio trial account restriction | Review Twilio account status | Upgrade Twilio project to paid production tier or ensure caller phone number is verified in Twilio console. |
| **Audio sounds muffled, distorted, or metallic** | Telephony sample rate mismatch | Check `PHONE_SAMPLE_RATE` in `.env.local` | Ensure `PHONE_SAMPLE_RATE=8000` is set for 8kHz G.711 narrowband SIP audio. |
| **Echo or heavy background noise heard by caller** | Noise cancellation disabled or incorrect DSP model | Check `NOISE_CANCELLATION` in `.env.local` | Verify `NOISE_CANCELLATION=true`. Ensure worker selects `BVCTelephony` profile for `PARTICIPANT_KIND_SIP`. |
| **Phone caller cannot select language via keypad** | DTMF tone format or missing DTMF events | Inspect `agent.language` DTMF handler | Verify carrier sends RFC 2833 DTMF tones. Test speech fallback: saying "two" or "telugu". |
| **Office transfer fails (call drops upon transfer)** | Carrier does not support SIP REFER | Inspect `agent.lifecycle` transfer logs | Check if Twilio trunk has SIP REFER enabled in trunk configuration. Verify target number is in E.164 format (`+91...`). |

## Expected result
- Caller hears initial welcome prompt and language menu (or direct greeting in default language).
- LiveKit dashboard shows `PARTICIPANT_KIND_SIP` participant connected to room with prefix `call-`.
- Clean bidirectional audio with G.711 / 8kHz sampling.

## Escalation
- If Twilio returns 503 Service Unavailable, verify LiveKit Cloud SIP service status.
- Check LiveKit Cloud Project Settings -> Telephony -> Inbound Trunks.
