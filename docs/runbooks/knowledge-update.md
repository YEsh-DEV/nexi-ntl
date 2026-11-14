# Runbook: Knowledge Base Ingestion, Quality Gates, Activation, and Rollback

## Symptom
University policies, fee structures, or academic calendar updates need to be ingested into the knowledge base, validated against quality gates, activated for production, or rolled back if regressions occur.

## Check
Check currently active knowledge artifact version and verification status:
```bash
python -c "from agent.config import load_settings; from rag.manifest import read_active; print(read_active(load_settings()))"
```

## Command

### 1. Ingest Raw Documents into Staging Artifact
Add new PDFs or JSON records into `data/sources/` or `data/` and run the offline ingest build:
```bash
python -m rag.ingest build
```
This prints the staging version hash `<STAGING_VERSION>` (e.g. `337ce61b42e1426896ba9cce2beaba4a`).

### 2. Generate and Inspect Coverage Report
```bash
python -m rag.coverage report <STAGING_VERSION>
```
Review the output to ensure `partial: false` and all documents have chunks > 0.

### 3. Human Sign-Off / Approval
If coverage is satisfactory, sign off on the report:
```bash
python -m rag.coverage approve <STAGING_VERSION> --approved-by "Registrar Office"
```

### 4. Calibrate Rerank Threshold (Calibration Split Only)
```bash
python -m rag.calibrate <STAGING_VERSION>
```

### 5. Run Held-Out Quality Gates Evaluation
```bash
python -m rag.eval <STAGING_VERSION>
```
If evaluation passes, the quality status updates to `passed`. If a known CPU slowdown occurs, a waiver can be approved:
```bash
python -m rag.eval <STAGING_VERSION> --latency-waiver "CPU emulation overhead" --approved-by "VP Engineering"
```

### 6. Atomic Activation
Point the `ACTIVE` pointer to the validated staging artifact:
```bash
python -m rag.ingest activate <STAGING_VERSION>
```
Restart the agent and token services to load the new artifact:
```bash
docker compose restart agent token
```

### 7. Rollback to Previous Artifact (If Regressions Observed)
If callers report issues, immediately revert to the previous verified version:
```bash
python -m rag.ingest rollback
docker compose restart agent token
```

## Expected result
- `rag.eval` prints `Final Quality Status: passed`.
- `ACTIVE` pointer contains `<STAGING_VERSION>`.
- Token service `/readyz` returns HTTP 200 with `knowledge.status: "ok"`.

## Escalation
- If `rag.eval` fails with `heldout_evidence_hit_at_4 < 0.85`, inspect failing queries in `questions.yaml` and verify document OCR text. Do NOT force activate an artifact with failed status.
- In case of persistent failure, execute `python -m rag.ingest rollback`.
