import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Room,
  RoomEvent,
  Track,
  ConnectionState,
} from 'livekit-client';
import { fetchToken } from '../lib/api';

export function useLiveKit({ language = 'en', onError, onStatusChange } = {}) {
  const [status, setStatus] = useState('disconnected');
  const [isMuted, setIsMuted] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const roomRef = useRef(null);
  const attachedAudioElementsRef = useRef([]);
  const animFrameRef = useRef(null);

  const updateStatus = useCallback(
    (newStatus) => {
      setStatus(newStatus);
      onStatusChange?.(newStatus);
    },
    [onStatusChange]
  );

  const cleanupAudio = useCallback(() => {
    attachedAudioElementsRef.current.forEach((el) => {
      el.pause();
      el.remove();
    });
    attachedAudioElementsRef.current = [];
  }, []);

  const disconnect = useCallback(async () => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    if (roomRef.current) {
      try {
        await roomRef.current.disconnect();
      } catch {
        // Ignore disconnect errors
      }
      roomRef.current = null;
    }

    cleanupAudio();
    setIsMuted(false);
    setAudioLevel(0);
    setIsAgentSpeaking(false);
    updateStatus('disconnected');
  }, [cleanupAudio, updateStatus]);

  const connect = useCallback(async () => {
    if (status === 'connecting' || status === 'connected') {
      return;
    }

    setErrorMessage(null);
    updateStatus('connecting');

    try {
      // 1. Fetch ephemeral token from backend
      const { token, url } = await fetchToken(language);

      if (!url) {
        throw new Error('LiveKit WebSocket URL is missing or not configured');
      }

      // 2. Initialize Room
      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      roomRef.current = room;

      // 3. Register Event Listeners
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          const element = track.attach();
          attachedAudioElementsRef.current.push(element);
        }
      });

      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        const detached = track.detach();
        detached.forEach((el) => el.remove());
        attachedAudioElementsRef.current = attachedAudioElementsRef.current.filter(
          (el) => !detached.includes(el)
        );
      });

      room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const remoteSpeaker = speakers.find((s) => !s.isLocal);
        setIsAgentSpeaking(Boolean(remoteSpeaker && remoteSpeaker.isSpeaking));
      });

      room.on(RoomEvent.ConnectionStateChanged, (connState) => {
        if (connState === ConnectionState.Connected) {
          updateStatus('connected');
        } else if (connState === ConnectionState.Disconnected) {
          updateStatus('disconnected');
        }
      });

      room.on(RoomEvent.Disconnected, () => {
        disconnect();
      });

      // 4. Connect to Room
      await room.connect(url, token);

      // 5. Enable Local Microphone
      try {
        await room.localParticipant.setMicrophoneEnabled(true);
        setIsMuted(false);
      } catch (micErr) {
        console.warn('Microphone permission or capture issue:', micErr);
      }

      updateStatus('connected');

      // 6. Audio level reactivity loop
      const checkAudioLevels = () => {
        if (roomRef.current && roomRef.current.state === ConnectionState.Connected) {
          const localLevel = roomRef.current.localParticipant?.audioLevel || 0;
          const remoteLevel =
            roomRef.current.activeSpeakers.find((s) => !s.isLocal)?.audioLevel || 0;
          const maxLevel = Math.max(localLevel, remoteLevel);
          setAudioLevel((prev) => prev * 0.7 + maxLevel * 0.3);
        } else {
          setAudioLevel(0);
        }
        animFrameRef.current = requestAnimationFrame(checkAudioLevels);
      };
      animFrameRef.current = requestAnimationFrame(checkAudioLevels);
    } catch (err) {
      const errorObj = err instanceof Error ? err : new Error(String(err));
      setErrorMessage(errorObj.message);
      updateStatus('error');
      onError?.(errorObj);
      cleanupAudio();
    }
  }, [language, status, updateStatus, onError, cleanupAudio, disconnect]);

  const toggleMute = useCallback(async () => {
    if (!roomRef.current || roomRef.current.state !== ConnectionState.Connected) {
      return;
    }
    const targetState = !isMuted;
    try {
      await roomRef.current.localParticipant.setMicrophoneEnabled(!targetState);
      setIsMuted(targetState);
    } catch (err) {
      console.error('Failed to toggle microphone state:', err);
    }
  }, [isMuted]);

  // Teardown on unmount
  useEffect(() => {
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
      if (roomRef.current) {
        roomRef.current.disconnect();
      }
      cleanupAudio();
    };
  }, [cleanupAudio]);

  return {
    status,
    isMuted,
    audioLevel,
    isAgentSpeaking,
    errorMessage,
    connect,
    disconnect,
    toggleMute,
  };
}
