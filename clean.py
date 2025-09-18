from dotenv import load_dotenv
import os
import logging
import asyncio
import inspect
from typing import Optional
import time

logger = logging.getLogger(__name__)

from livekit import agents, rtc
from livekit.agents import (
    AgentSession, Agent, RoomInputOptions, 
    JobContext, JobRequest,
    BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip  # ✅ Added for background audio
)
from livekit.plugins import (
    deepgram,
    noise_cancellation,
    silero,
    google,
    sarvam
)

# Import optimized tools
from tools import intelligent_search
from rag_engine import initialize_unified_rag
from prompt_template import OPTIMIZED_NEXI_PROMPT

# Optimized logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose logs
for logger_name in ["urllib3", "httpx", "google", "deepgram"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

load_dotenv(".env.local")

class OptimizedAssistant(Agent): 
    def __init__(self) -> None:
        super().__init__(
            instructions=OPTIMIZED_NEXI_PROMPT,
            # CRITICAL: Only one unified tool to prevent LLM confusion
            tools=[intelligent_search]
        )
        
        self._processing_query = False
        self._query_count = 0
        self._error_count = 0
        self._timeout_count = 0  # ✅ Track timeouts separately
        self._start_time = time.time()
        self._active_sessions = {}
        
        # ✅ Initialize background audio player reference
        self.background_audio = None
        
        logger.info("Optimized Assistant initialized with single unified tool")

    def set_background_audio(self, background_audio: BackgroundAudioPlayer):
        """Set the background audio player reference"""
        self.background_audio = background_audio
        logger.info("Background audio player reference set")

    async def process_query(self, participant_identity: str, query: str, session: AgentSession):
        """Optimized query processing with comprehensive error handling"""
        
        # Prevent concurrent processing per user
        if self._processing_query:
            logger.warning(f"Concurrent query blocked for {participant_identity}")
            return "Please wait, I'm still processing your previous question."
        
        self._processing_query = True
        query_start = time.time()
        thinking_sound_started = False  # ✅ Track thinking sound state
        
        try:
            self._query_count += 1
            logger.info(f"Query #{self._query_count} from {participant_identity}: {query[:50]}...")
            
            # Quick query validation
            if not query or len(query.strip()) < 2:
                return "Please ask a clear question about the university."
            
            # ✅ Start thinking sound when processing begins (with better error handling)
            if self.background_audio:
                try:
                    await asyncio.wait_for(self.background_audio.start_thinking(), timeout=2.0)
                    thinking_sound_started = True
                    logger.debug("🎵 Started thinking sound")
                except asyncio.TimeoutError:
                    logger.warning("Thinking sound start timed out")
                except Exception as e:
                    logger.warning(f"Failed to start thinking sound: {e}")
            
            # Session management - simplified
            if participant_identity not in self._active_sessions:
                self._active_sessions[participant_identity] = {
                    'start_time': time.time(),
                    'query_count': 0
                }
            
            self._active_sessions[participant_identity]['query_count'] += 1
            
            # ✅ Enhanced response generation with multiple fallback strategies
            response = None
            
            # Strategy 1: Full prompt with reasonable timeout
            try:
                focused_instruction = f"{OPTIMIZED_NEXI_PROMPT}\n\nQuestion: {query}"
                response_task = session.generate_reply(instructions=focused_instruction)
                response = await asyncio.wait_for(response_task, timeout=10.0)  # ✅ More reasonable timeout
                
                if response and isinstance(response, str) and response.strip():
                    logger.info(f"✅ Strategy 1 successful in {time.time() - query_start:.2f}s")
                else:
                    response = None
                    
            except asyncio.TimeoutError:
                self._timeout_count += 1
                logger.warning(f"Strategy 1 timeout after 10s (timeout #{self._timeout_count})")
                response = None
            except Exception as e:
                logger.warning(f"Strategy 1 failed: {e}")
                response = None
            
            # Strategy 2: Minimal prompt with very short timeout
            if not response:
                try:
                    logger.info("Trying strategy 2: minimal prompt")
                    minimal_instruction = f"Answer briefly: {query}"
                    response_task = session.generate_reply(instructions=minimal_instruction)
                    response = await asyncio.wait_for(response_task, timeout=6.0)  # ✅ Reasonable fallback timeout
                    
                    if response and isinstance(response, str) and response.strip():
                        logger.info(f"✅ Strategy 2 successful in {time.time() - query_start:.2f}s")
                    else:
                        response = None
                        
                except asyncio.TimeoutError:
                    self._timeout_count += 1
                    logger.warning(f"Strategy 2 timeout after 4s (timeout #{self._timeout_count})")
                    response = None
                except Exception as e:
                    logger.warning(f"Strategy 2 failed: {e}")
                    response = None
            
            # Strategy 3: Fallback response if all else fails
            if not response:
                logger.error("All strategies failed, using fallback response")
                if "fee" in query.lower() or "cost" in query.lower():
                    response = "I'm having trouble accessing fee information right now. Please contact the accounts office directly or try asking again in a moment."
                elif "hostel" in query.lower():
                    response = "I'm having trouble accessing hostel information right now. Please contact the hostel office directly or try asking again in a moment."
                elif "library" in query.lower():
                    response = "I'm having trouble accessing library information right now. Please visit the library directly or try asking again in a moment."
                else:
                    response = "I'm experiencing some technical difficulties right now. Please try asking your question again in a simpler way, or contact the university office directly."
            
            # ✅ Stop thinking sound when processing is done (with better error handling)
            if thinking_sound_started and self.background_audio:
                try:
                    await asyncio.wait_for(self.background_audio.stop_thinking(), timeout=2.0)
                    logger.debug("🔇 Stopped thinking sound")
                except asyncio.TimeoutError:
                    logger.warning("Thinking sound stop timed out")
                except Exception as e:
                    logger.warning(f"Failed to stop thinking sound: {e}")
            
            # Log performance
            elapsed = time.time() - query_start
            logger.info(f"Query processed in {elapsed:.2f}s")
            
            # Simple goodbye detection
            if any(word in query.lower() for word in ['bye', 'goodbye', 'thanks', 'thank you']):
                if participant_identity in self._active_sessions:
                    del self._active_sessions[participant_identity]
                logger.info(f"Session ended for {participant_identity}")
            
            return response
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Query processing error: {e}", exc_info=True)
            
            return "I encountered an issue. Please try rephrasing your question."
        
        finally:
            # ✅ Always ensure thinking sound is stopped and processing flag is reset
            if thinking_sound_started and self.background_audio:
                try:
                    await asyncio.wait_for(self.background_audio.stop_thinking(), timeout=1.0)
                    logger.debug("🔇 Thinking sound stopped in finally block")
                except:
                    pass  # Silent fail in cleanup
            
            self._processing_query = False

    def get_health_stats(self):
        """Get agent health statistics"""
        uptime = time.time() - self._start_time
        return {
            'uptime_hours': round(uptime / 3600, 2),
            'total_queries': self._query_count,
            'error_count': self._error_count,
            'timeout_count': self._timeout_count,  # ✅ Added timeout tracking
            'active_sessions': len(self._active_sessions),
            'error_rate': self._error_count / max(self._query_count, 1),
            'timeout_rate': self._timeout_count / max(self._query_count, 1),  # ✅ Added timeout rate
            'background_audio_enabled': self.background_audio is not None
        }

async def entrypoint(ctx: JobContext):
    """Optimized entrypoint with improved stability"""
    
    session = None
    assistant = None
    background_audio = None
    
    try:
        logger.info("Starting optimized Nexi agent...")
        
        # Initialize unified RAG system
        logger.info("Initializing unified RAG engine...")
        try:
            await asyncio.wait_for(initialize_unified_rag(), timeout=45.0)
            logger.info("RAG engine ready")
        except asyncio.TimeoutError:
            logger.error("RAG initialization timeout")
            raise RuntimeError("RAG setup failed")

        # Initialize components with production settings
        
        # STT - Optimized settings
        stt = deepgram.STT(
            model="nova-2", 
            language="en-US", 
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            smart_format=True,
            interim_results=False,
            filler_words=False,
            punctuate=True
        )
        logger.info("STT optimized")

        # ✅ LLM - Enhanced settings (keeping only supported parameters)
        llm = google.LLM(
            model="gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            # ✅ Basic settings that are widely supported:
            temperature=0.2,                          # Faster, more consistent  
            top_p=0.8,                               # Reduce randomness
            top_k=20,                                # Limit token choices
        )
        logger.info("LLM optimized with basic settings")

        # TTS - Optimized
        tts = sarvam.TTS(
            target_language_code="en-IN",
            speaker="anushka",
            api_key=os.getenv("sarvam_api_key"),
        )
        logger.info("TTS ready")

        # VAD - Tuned for responsiveness
        vad = silero.VAD.load()
        logger.info("VAD ready")

        # Session with optimized turn detection
        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )

        # Create optimized assistant
        assistant = OptimizedAssistant()

        # ✅ Initialize Background Audio Player (simplified configuration)
        try:
            background_audio = BackgroundAudioPlayer(
                thinking_sound=[
                    # ✅ Simple configuration that should work
                    AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
                ],
            )
            logger.info("🎵 Background audio player initialized (simplified config)")
            
            # ✅ Set background audio reference in assistant
            assistant.set_background_audio(background_audio)
            
        except Exception as e:
            logger.warning(f"Failed to initialize background audio: {e}")
            logger.info("Continuing without background audio...")
            background_audio = None

        # Connection with retry and better error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting... (attempt {attempt + 1})")
                await asyncio.wait_for(ctx.connect(), timeout=20.0)
                logger.info("Connected successfully")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Connection failed: {e}, retrying...")
                await asyncio.sleep(2 ** attempt)

        # Start session with optimized room settings
        desired_room_opts = {
            "noise_cancellation": noise_cancellation.BVC(),
            "auto_subscribe": True,
        }
        try:
            sig = inspect.signature(RoomInputOptions)
            supported_room_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            supported_room_params = {"noise_cancellation"}

        filtered_room_opts = {k: v for k, v in desired_room_opts.items() if k in supported_room_params}
        room_input_opts = RoomInputOptions(**filtered_room_opts) if filtered_room_opts else None

        await session.start(
            room=ctx.room,
            agent=assistant,
            **({"room_input_options": room_input_opts} if room_input_opts is not None else {}),
        )
        logger.info("Session started")

        # ✅ Start background audio player with timeout protection
        if background_audio:
            try:
                await asyncio.wait_for(
                    background_audio.start(room=ctx.room, agent_session=session), 
                    timeout=10.0
                )
                logger.info("🎵 Background audio player started successfully")
            except asyncio.TimeoutError:
                logger.warning("Background audio start timed out - continuing without it")
                background_audio = None
                assistant.set_background_audio(None)
            except Exception as e:
                logger.warning(f"Failed to start background audio player: {e}")
                background_audio = None
                assistant.set_background_audio(None)

        # Optimized transcription handler
        @session.on("transcription")  
        def on_transcription(event):
            if not event.text or len(event.text.strip()) < 2:
                return

            user_text = event.text.strip()
            participant_id = str(event.participant.identity if event.participant else "user")

            logger.info(f"Input: {user_text[:50]}...")

            # Process with error isolation
            async def handle_query():
                try:
                    await assistant.process_query(participant_id, user_text, session)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

            # Non-blocking task creation
            asyncio.create_task(handle_query())

        # ✅ Send optimized greeting with timeout protection
        try:
            logger.info("Sending greeting...")
            greet_task = session.generate_reply(
                instructions="Say exactly: 'Hi! I'm Nexi, your SRM AP assistant. How can I help you?'"
            )
            await asyncio.wait_for(greet_task, timeout=5.0)
            logger.info("✅ Greeting sent successfully")
        except asyncio.TimeoutError:
            logger.warning("Greeting timed out - agent will continue without initial greeting")
        except Exception as e:
            logger.warning(f"Greeting failed: {e} - agent will continue without initial greeting")

        logger.info("Agent running optimally")
        if background_audio:
            logger.info("🎵 Background audio (thinking sounds) enabled for latency coverage")
        
        # Health monitoring loop
        last_health_check = time.time()
        
        while True:
            # Health check every 30 seconds
            if time.time() - last_health_check > 30:
                stats = assistant.get_health_stats()
                logger.info(f"Health: {stats}")
                last_health_check = time.time()
            
            # Check connection
            if ctx.room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
                logger.warning("Connection lost")
                break
                
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("Shutting down...")
        
        # ✅ Cleanup background audio if it exists
        if background_audio:
            try:
                logger.info("Cleaning up background audio...")
                # Add any specific cleanup if needed by the BackgroundAudioPlayer
            except Exception as e:
                logger.warning(f"Background audio cleanup error: {e}")
        
        if session:
            end_fn = getattr(session, "end", None) or getattr(session, "stop", None) or getattr(session, "close", None)
            if end_fn:
                try:
                    if asyncio.iscoroutinefunction(end_fn):
                        await end_fn()
                    else:
                        end_fn()
                except Exception as e:
                    logger.warning(f"Session end error: {e}")

if __name__ == "__main__":
    import inspect

    desired_kwargs = {
        "entrypoint_fnc": entrypoint,
        "reconnect_attempts": 3,
        "reconnect_interval": 5.0,
        "room_join_timeout": 20.0,
    }

    try:
        sig = inspect.signature(agents.WorkerOptions)
        supported_params = set(p for p in sig.parameters.keys() if p != "self")
    except (ValueError, TypeError):
        supported_params = {"entrypoint_fnc", "room_join_timeout"}

    filtered_kwargs = {k: v for k, v in desired_kwargs.items() if k in supported_params}

    worker_options = agents.WorkerOptions(**filtered_kwargs)
    agents.cli.run_app(worker_options)