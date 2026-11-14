"""Backward compatibility wrapper for agent.providers.sarvam_adapter."""

from agent.providers.sarvam_adapter import (
    ModelAwareChunkedStream,
    NexiSarvamTTS,
    sanitize_error_message,
)

__all__ = ["ModelAwareChunkedStream", "NexiSarvamTTS", "sanitize_error_message"]
