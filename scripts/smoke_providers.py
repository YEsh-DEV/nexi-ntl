#!/usr/bin/env python3
"""Smoke testing script for live external provider connections without leaking secrets.

Tests:
1. Gemini LLM (one-line text completion)
2. Sarvam TTS (8000 Hz & 24000 Hz audio synthesis)
3. Deepgram STT (round-trip audio transcription)
4. LiveKit Server API (room listing)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.config import Settings, load_settings  # noqa: E402


async def smoke_gemini(settings: Settings) -> bool:
    """Test Gemini LLM one-line completion."""
    try:
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.llm_model,
            contents="Say 'OK' in one word.",
        )
        text = response.text.strip() if response and response.text else ""
        if text:
            print("  [PASS] Gemini LLM: received response")
            return True
        print("  [FAIL] Gemini LLM: empty response text", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"  [FAIL] Gemini LLM: {type(exc).__name__}", file=sys.stderr)
        return False


async def smoke_sarvam_tts(settings: Settings) -> tuple[bool, bytes]:
    """Test Sarvam TTS synthesis at 24000 Hz and 8000 Hz."""
    import httpx

    sample_text = "Welcome to SRM University AP."
    audio_bytes_24k = b""
    try:
        url = "https://api.sarvam.ai/text-to-speech"
        headers = {"api-subscription-key": settings.sarvam_api_key}

        # 24kHz check
        payload_24k = {
            "inputs": [sample_text],
            "target_language_code": "en-IN",
            "speaker": settings.tts_speaker,
            "model": settings.tts_model,
            "speech_sample_rate": 24000,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp_24k = await client.post(url, json=payload_24k, headers=headers)
            if resp_24k.status_code != 200:
                print(f"  [FAIL] Sarvam TTS (24kHz): HTTP {resp_24k.status_code}", file=sys.stderr)
                return False, b""

            # 8kHz check (telephony rate)
            payload_8k = {
                "inputs": [sample_text],
                "target_language_code": "en-IN",
                "speaker": settings.tts_speaker,
                "model": settings.tts_model,
                "speech_sample_rate": 8000,
            }
            resp_8k = await client.post(url, json=payload_8k, headers=headers)
            if resp_8k.status_code != 200:
                print(f"  [FAIL] Sarvam TTS (8kHz): HTTP {resp_8k.status_code}", file=sys.stderr)
                return False, b""

            data = resp_24k.json()
            audios = data.get("audios", [])
            if audios and isinstance(audios[0], str):
                import base64

                audio_bytes_24k = base64.b64decode(audios[0])
                print("  [PASS] Sarvam TTS: successfully synthesized 24kHz and 8kHz audio")
                return True, audio_bytes_24k

            print("  [FAIL] Sarvam TTS: missing audios field in response", file=sys.stderr)
            return False, b""
    except Exception as exc:
        print(f"  [FAIL] Sarvam TTS: {type(exc).__name__}", file=sys.stderr)
        return False, b""


async def smoke_deepgram(settings: Settings, audio_data: bytes) -> bool:
    """Test Deepgram STT transcription."""
    if not audio_data:
        # Generate dummy 1-second 16kHz PCM WAV if no TTS audio available
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00" * 32000)
        audio_data = buf.getvalue()

    try:
        import httpx

        url = "https://api.deepgram.com/v1/listen?model=nova-3&language=en"
        headers = {
            "Authorization": f"Token {settings.deepgram_api_key}",
            "Content-Type": "audio/wav",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=audio_data, headers=headers)
            if resp.status_code == 200:
                print("  [PASS] Deepgram STT: transcription endpoint reachable and accepted audio")
                return True
            print(f"  [FAIL] Deepgram STT: HTTP {resp.status_code}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"  [FAIL] Deepgram STT: {type(exc).__name__}", file=sys.stderr)
        return False


async def smoke_livekit(settings: Settings) -> bool:
    """Test LiveKit Server API connection."""
    try:
        from livekit import api

        lk_api = api.LiveKitAPI(
            settings.livekit_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        try:
            rooms = await asyncio.wait_for(
                lk_api.room.list_rooms(api.ListRoomsRequest()), timeout=5.0
            )
            print(f"  [PASS] LiveKit Server API: connected successfully (active rooms: {len(rooms.rooms)})")
            return True
        finally:
            await lk_api.aclose()
    except Exception as exc:
        print(f"  [FAIL] LiveKit Server API: {type(exc).__name__}", file=sys.stderr)
        return False


async def run_smoke(skip: list[str]) -> int:
    settings = load_settings()
    results: list[bool] = []

    print("Running Nexi Provider Smoke Tests:")

    # 1. Gemini
    if "gemini" in skip or "llm" in skip:
        print("  [SKIP] Gemini LLM")
    else:
        results.append(await smoke_gemini(settings))

    # 2. Sarvam TTS
    tts_audio = b""
    if "sarvam" in skip or "tts" in skip:
        print("  [SKIP] Sarvam TTS")
    else:
        tts_ok, tts_audio = await smoke_sarvam_tts(settings)
        results.append(tts_ok)

    # 3. Deepgram STT
    if "deepgram" in skip or "stt" in skip:
        print("  [SKIP] Deepgram STT")
    else:
        results.append(await smoke_deepgram(settings, tts_audio))

    # 4. LiveKit
    if "livekit" in skip:
        print("  [SKIP] LiveKit Server API")
    else:
        results.append(await smoke_livekit(settings))

    all_passed = all(results) if results else True
    print("-" * 50)
    print("Smoke Test Summary:", "ALL PASSED" if all_passed else "SOME FAILED")
    return 0 if all_passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Provider to skip (gemini, sarvam, deepgram, livekit)",
    )
    args = parser.parse_args()
    skip_list = [s.strip().lower() for s in args.skip]
    sys.exit(asyncio.run(run_smoke(skip_list)))


if __name__ == "__main__":
    main()
