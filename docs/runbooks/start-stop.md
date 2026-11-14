# Runbook: Worker and Token Service Lifecycle (Start, Stop, Rolling Restart)

## Symptom
Operator needs to start services, gracefully drain and restart workers during code/model updates without terminating active calls abruptly, or perform emergency shutdown.

## Check
Verify container/process status and inspect active connections:
```bash
docker compose ps
curl -s http://127.0.0.1:8080/readyz
```

## Command

### 1. Normal Production Startup
Start all services via Docker Compose:
```bash
docker compose up -d
```
Or start locally using virtualenv:
```bash
python -m agent.main start --drain-timeout 660 &
python -m agent.token_service &
python -m agent.maintenance run &
```

### 2. Rolling Restart (Zero Dropped Calls)
LiveKit agents support a drain period (`--drain-timeout 660`). When SIGTERM is sent, the worker stops accepting new rooms and allows active calls up to 10 minutes to conclude naturally:
```bash
docker compose up -d --no-deps --build agent
```
Docker will signal SIGTERM to the old agent and wait up to `stop_grace_period: 700s` before killing.

### 3. Graceful Shutdown
```bash
docker compose stop agent
docker compose down
```

## Expected result
- `/readyz` on port 8080 returns HTTP 200 with `{"status": "ok"}`.
- Running agent worker logs show `prewarm_ready` and successful registration with LiveKit Cloud.
- When stopping, worker logs show `draining jobs` and waits for active calls to end before process exit.

## Escalation
- If container fails to start, inspect logs: `docker compose logs --tail=100 agent`.
- If worker health probe fails on port 8081, check if `WORKER_HEALTH_URL` is reachable and `python -m agent.readiness` status.
- In case of hung worker, issue hard stop: `docker compose kill agent`.
