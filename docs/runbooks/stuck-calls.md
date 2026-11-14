# Runbook: Stuck Calls, Zombie Rooms, and Capacity Recovery

## Symptom
Active call capacity is reached (503 Service Unavailable on `/api/token`), but few or no real people are talking. LiveKit dashboard shows lingering rooms without participants, or database has orphaned `started` rows.

## Check
1. Query active rooms via admission capacity controller or LiveKit CLI:
```bash
python -c "
import asyncio
from agent.config import load_settings
from agent.admission import fetch_active_room_names
settings = load_settings()
print(asyncio.run(fetch_active_room_names(settings)))
"
```
2. Check database summary for abandoned or started calls:
```bash
python -m agent.reports summary --period 24h
```

## Command

### 1. Reconciling Orphaned Calls in Database
Mark any calls that started more than 2x max duration ago as `abandoned`:
```bash
python -m agent.maintenance reconcile
```

### 2. Identifying and Deleting Zombie Rooms
Zombie rooms occur if a network disconnect prevented clean room teardown:
```bash
python -c "
import asyncio
from agent.config import load_settings
from livekit import api
settings = load_settings()

async def clean_zombies():
    client = api.LiveKitAPI(settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret)
    try:
        res = await client.room.list_rooms(api.ListRoomsRequest())
        for r in res.rooms:
            # Delete rooms with 0 participants that have existed > 5 minutes
            if r.num_participants == 0:
                print(f'Deleting empty zombie room: {r.name}')
                await client.room.delete_room(api.DeleteRoomRequest(room=r.name))
    finally:
        await client.aclose()

asyncio.run(clean_zombies())
"
```

### 3. Immediate Room Deletion by Name
To delete a specific room safely without touching others:
```bash
python -c "
import asyncio, sys
from agent.config import load_settings
from livekit import api
settings = load_settings()
room_name = sys.argv[1]

async def delete_one(name):
    client = api.LiveKitAPI(settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret)
    try:
        await client.room.delete_room(api.DeleteRoomRequest(room=name))
        print(f'Room {name} successfully deleted.')
    finally:
        await client.aclose()

asyncio.run(delete_one(room_name))
" <ROOM_NAME>
```

## Expected result
- Zombie rooms are purged from LiveKit Cloud.
- Admission controller cache expires after 5s and `/readyz` or `/api/token` accepts new calls.
- `python -m agent.reports summary` shows calls marked `abandoned` rather than indefinitely `started`.

## Escalation
- If LiveKit room service returns 5xx errors, verify LiveKit Cloud status at https://status.livekit.io/.
- Restart token service to flush admission controller room cache: `docker compose restart token`.
