"""Model-aware Sarvam AI TTS adapter for Bulbul v3 compatibility.

The upstream livekit-plugins-sarvam (1.2.6) hardcodes `pitch`, `loudness`, and
`enable_preprocessing` into all requests. Bulbul v3 rejects or errors when
those parameters are sent. This adapter strips them cleanly when using bulbul:v3
while preserving full compatibility with bulbul:v2.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import replace

import aiohttp
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.plugins import sarvam

logger = logging.getLogger("agent.tts_adapter")


class ModelAwareChunkedStream(sarvam.tts.ChunkedStream):
    def __init__(self, *, tts: sarvam.TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: sarvam.TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the Sarvam.ai TTS request and emit audio via the output emitter."""
        payload: dict[str, object] = {
            "target_language_code": self._opts.target_language_code,
            "text": self._input_text,
            "speaker": self._opts.speaker,
            "pace": self._opts.pace,
            "speech_sample_rate": self._opts.speech_sample_rate,
            "model": self._opts.model,
        }
        # bulbul:v3 explicitly does NOT support pitch, loudness, or preprocessing
        if not str(self._opts.model).startswith("bulbul:v3"):
            payload["pitch"] = self._opts.pitch
            payload["loudness"] = self._opts.loudness
            payload["enable_preprocessing"] = self._opts.enable_preprocessing

        headers = {
            "api-subscription-key": self._opts.api_key,
            "Content-Type": "application/json",
        }
        try:
            async with self._tts._ensure_session().post(
                url=self._opts.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    logger.error(f"Sarvam TTS API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Sarvam TTS API Error: {error_text}", status_code=res.status
                    )

                response_json = await res.json()
                request_id = response_json.get("request_id", "")
                audios = response_json.get("audios", [])
                if not audios or not isinstance(audios, list):
                    raise APIConnectionError("Sarvam TTS API response invalid: no audio data")

                output_emitter.initialize(
                    request_id=request_id or "unknown",
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/wav",
                )
                for b64 in audios:
                    wav_bytes = base64.b64decode(b64)
                    output_emitter.push(wav_bytes)
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Sarvam TTS API request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Sarvam TTS API connection error: {e}") from e


class NexiSarvamTTS(sarvam.TTS):
    """Model-aware Sarvam TTS synthesizing with v3 or v2 compliant payloads."""

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions | None = None
    ) -> tts.ChunkedStream:
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        return ModelAwareChunkedStream(tts=self, input_text=text, conn_options=conn_options)
