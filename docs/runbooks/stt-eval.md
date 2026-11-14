# Runbook: STT and Audio Quality Evaluation

## Symptom
Evaluate real-world transcription accuracy (Word Error Rate - WER) and audio clarity for Deepgram Nova-3 (English) and Sarvam Saarika (Telugu and Hindi) across varying acoustic environments (laptop mic and mobile phone).

## Check
Verify evaluation datasets and scripts:
```bash
python scripts/wer.py --help
cat eval/multilingual.yaml
```

## Command

1. **Prepare the Evaluation CSV:**
   Create a CSV file (e.g. `eval/stt_samples.csv`) with the columns `reference` and `hypothesis`:
   ```csv
   reference,hypothesis
   "what is the tuition fee for btech computer science","what is the tuition fee for btech computer science"
   "hostel two bed ac room fees","hostel to bed ac room fees"
   ```

2. **Compute Word Error Rate (WER):**
   Run the standard library WER tool:
   ```bash
   python scripts/wer.py --csv eval/stt_samples.csv
   ```

## Expected result
- English (Deepgram Nova-3): WER <= 8.0% on laptop mic, <= 12.0% on phone SIP.
- Telugu & Hindi (Sarvam Saarika): WER <= 15.0% on clear audio.

## Escalation
- If terms like "CSE", "ECE", or faculty names are mis-transcribed, update `agent/prompts.py` pronunciation hints (e.g. "C S E").
- If background noise causes high deletions, verify `NOISE_CANCELLATION=true` and confirm `BVCTelephony` profile on SIP.
