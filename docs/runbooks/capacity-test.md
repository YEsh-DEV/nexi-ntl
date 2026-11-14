# Runbook: Concurrency Capacity Testing and Sizing Procedure

## Symptom
Operator or engineer needs to establish production resource sizing (CPU, RAM) and define safe `MAX_CONCURRENT_CALLS` limits for the deployment.

## Check
Verify the worker is running and capacity sampling scripts are executable:
```bash
python scripts/capacity_sample.py --help
python scripts/capacity_report.py --help
```

## Command

### 1. Launch Continuous Metrics Sampler
In a dedicated terminal, start the capacity metrics sampler. It polls worker process tree CPU and RSS memory alongside active rooms every 2 seconds:
```bash
python scripts/capacity_sample.py --interval 2.0 --output data/telemetry/capacity_samples.csv
```

### 2. Execute Staged Load Profiles
Run 3 distinct 3-minute testing stages with real sessions (web browser tabs or phone calls):
- **Stage 1 (Idle & Baseline):** Run for 2 minutes with 0 active calls to record baseline idle memory.
- **Stage 2 (Single Session):** Connect 1 session, ask 3 questions from a fixed script (e.g. tuition fees, hostel rules, attendance criteria) over 3 minutes.
- **Stage 3 (5 Concurrent Sessions):** Open 5 simultaneous web tabs, each asking continuous questions for 3 minutes.
- **Stage 4 (10 Concurrent Sessions):** Open 10 simultaneous sessions for 3 minutes.

### 3. Generate Capacity Analysis Report
Once sampling is complete, generate the capacity report:
```bash
python scripts/capacity_report.py --samples data/telemetry/capacity_samples.csv --format text
```

### 4. Save Baseline and Update Sizing Configuration
Save the report output to a permanent baseline artifact:
```bash
mkdir -p docs/baselines
python scripts/capacity_report.py --samples data/telemetry/capacity_samples.csv --format text > docs/baselines/capacity-report-$(date +%F).md
```
Update `.env.local` and `docker-compose.yml` based on the measured data:
- Set `MAX_CONCURRENT_CALLS` to the concurrency level where CPU p95 < 85% and response latency p95 < 2000ms.
- Set container `memory` limit to `peak_rss_mb * 1.3`.
- Set container `cpus` limit to `cpu_p95 / 100 * 1.5`.

> [!WARNING]
> **Strict Measurement Rule:**
> You must NEVER write capacity numbers or claim concurrency capabilities that have not been physically measured with live sessions. Until this human procedure has been run, documentation and README must state that capacity is unmeasured.

## Expected result
- `capacity_samples.csv` contains consecutive timestamped rows recording CPU, RSS, and room count.
- `scripts/capacity_report.py` outputs a structured table reporting peak memory, estimated MB per call, CPU p95, and response latency p95 across all concurrency levels.

## Escalation
- If CPU spikes to 100% at 5 concurrent calls, decrease `INFERENCE_THREADS` to prevent CPU thrashing or scale horizontally.
