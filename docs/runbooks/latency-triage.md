# Runbook: Voice Latency Triage and Performance Optimization

## Symptom
Callers perceive conversational sluggishness or noticeable lag (> 1500ms) between their question and the start of agent speech.

## Check
Generate the latency report from turn metrics recorded over the last 24 hours:
```bash
python -m agent.reports latency --period 24h
```
Sample output provides nearest-rank p50/p95/p99 percentiles across the 4 key stages:
- `EOU to LLM First Token (ms)` (LLM generation delay)
- `LLM Token to Audio First Frame (ms)` (TTS synthesis delay)
- `End-of-Utterance to First Audio (ms)` (Total voice turnaround latency)
- `RAG Retrieval Duration (ms)` (Knowledge lookup latency)

## Command

### 1. High RAG Retrieval Duration (RAG p95 > 250ms)
- *Cause:* Dense embedding search or cross-encoder reranking slow on CPU.
- *Check:* Verify thread count in `.env.local`: `INFERENCE_THREADS=2`.
- *Action:* If running inside Docker/container, verify CPU allocation in `docker-compose.yml`:
  ```yaml
  resources:
    limits:
      cpus: "2.0"
  ```
  Ensure `CHROMA_DIR` is stored on high-speed NVMe/SSD storage and not a slow network mount.

### 2. High LLM First Token Latency (LLM p95 > 1000ms)
- *Cause:* Gemini API latency or token context budget exceeded.
- *Check:* Verify `CONTEXT_MAX_CHARS=2400` in `.env.local`. Ensure huge documents are not injected into prompt context.
- *Action:* Test direct network round-trip to Google API:
  ```bash
  python scripts/smoke_providers.py
  ```

### 3. High TTS Audio Synthesis Latency (TTS p95 > 600ms)
- *Cause:* Sarvam API latency or network packet jitter.
- *Check:* Review Telugu/Hindi vs English synthesis latency.
- *Action:* For phone calls, ensure `PHONE_SAMPLE_RATE=8000` is configured in `.env.local` to prevent unnecessary upsampling.

## Expected result
- Normal operational baselines:
  - Total voice turnaround (`eou_to_first_audio_ms`): p50 <= 1200ms, p95 <= 1800ms.
  - RAG retrieval latency: p95 <= 200ms.
  - Reranker latency: p95 <= 150ms.

## Escalation
- If sample size N < 30, collect more live call data before concluding systematic degradation.
- Save current report as baseline: `python -m agent.reports latency --save docs/baselines/voice-latency-$(date +%F).md`.
