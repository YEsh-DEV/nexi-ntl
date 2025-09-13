from dotenv import load_dotenv
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
    openai
)
from tools import get_rag_answer 
from livekit_session_manager import LiveKitSessionManager

# ✅ Import from your RAG engine
from rag_engine import initialize_rag_engine, get_rag_answer_async
from prompt_template import NEXI_PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")
prompt = NEXI_PROMPT_TEMPLATE

class Assistant(Agent): 
    def __init__(self) -> None:
        super().__init__(
            instructions=prompt,
            tools=[get_rag_answer]
        )
        # ✅ Initialize session manager with 30-second timeout
        self.session_manager = LiveKitSessionManager(timeout_seconds=30)
        logger.info("Assistant initialized with session manager")

    async def process_query(self, participant_identity: str, query: str, session: AgentSession):
        """Process user query using agent tools and session management """
        
        try:
            logger.info(f"Processing query from {participant_identity}: {query}")
            
            # ✅ Get or create user session
            user_session = self.session_manager.get_or_create_session(participant_identity)
            # ✅ Get conversation history from current session
            history_context = user_session.get_current_context(message_count=3)
            # ✅ Generate response
            response = await session.generate_reply(instructions=f"{prompt}\n\nStudent Question:{query}")
            # ✅ Handle message with session manager (automatically handles goodbye and saves interaction)
            session_ended = self.session_manager.handle_message(participant_identity, query, str(response))
            
            if session_ended:
                logger.info(f"✅ Session ended for {participant_identity} - new session started")
                # Send a brief acknowledgment for goodbye
                await session.generate_reply(
                    instructions="Say a warm goodbye and mention you're here if they need help again later."
                )
            
            logger.info(f"✅ Response generated for {participant_identity}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query for {participant_identity}: {e}", exc_info=True)
            
            # Try to still handle the message for session management
            try:
                self.session_manager.handle_message(participant_identity, query, "Error occurred")
            except:
                pass
                
            return "I'm having trouble right now. Could you please try again?"

    def __del__(self):
        """Cleanup when assistant is destroyed"""
        if hasattr(self, 'session_manager'):
            self.session_manager.stop_cleanup_task()

async def entrypoint(ctx: agents.JobContext):
    try:
        logger.info("🚀 Starting Nexi Agent...")
        
        # ✅ Initialize RAG engine
        logger.info("📚 Initializing RAG engine...")
        await initialize_rag_engine()
        logger.info("✅ RAG engine initialized successfully!")

        # ✅ Initialize STT
        stt = deepgram.STT(model="nova-2", language="en-US")
        logger.info("🎤 STT initialized with Deepgram")

        # ✅ Initialize LLM
        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url="http://127.0.0.1:11434/v1",
        )
        logger.info("🧠 LLM initialized with Ollama")

        # ✅ Initialize TTS
        cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise ValueError("❌ CARTESIA_API_KEY not found in environment variables")

        tts = cartesia.TTS(
            model="sonic-english",
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",
            api_key=cartesia_api_key,
        )
        logger.info("🔊 TTS initialized with Cartesia")

        # ✅ Initialize VAD
        vad = silero.VAD.load()
        logger.info("👂 VAD initialized with Silero")

        # ✅ Create session
        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )

        # ✅ Create assistant
        assistant = Assistant()

        max_retries = 5
        for attempt in range(max_retries):
            try:
                 await ctx.connect()
                 logger.info("🌐 Connected to LiveKit room")
                 break
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
                await asyncio.sleep(min(2 ** attempt, 30))
        else:
            raise RuntimeError("Failed to connect to LiveKit after retries")

        # ✅ Start the session
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        logger.info("🎯 Session started successfully")

        # ✅ Handle transcriptions
        @session.on("transcription")
        def on_transcription(event):
            if not event.text or not event.text.strip():
                return

            user_text = event.text.strip()
            participant_identity = str(event.participant.identity if event.participant else "anonymous")

            logger.info(f"👤 Received from {participant_identity}: {user_text}")

            # ✅ Process all messages through the assistant (no manual goodbye handling)
            async def handle_query():
                try:
                    await assistant.process_query(participant_identity, user_text, session)
                except Exception as e:
                    logger.error(f"❌ Error in handle_query: {e}")

            # Create and run async task
            task = asyncio.create_task(handle_query())
            task.add_done_callback(
                lambda t: logger.error(f"❌ Task error: {t.exception()}") if t.exception() else None
            )

        # ✅ Send initial greeting
        logger.info("👋 Sending initial greeting...")
        await session.generate_reply(
            instructions="Say: Hi! I'm Nexi, your SRM AP university assistant. I can help with questions about hostel, mess, library, fees, and campus rules. What would you like to know? Keep it simple and friendly, no symbols."
        )

        logger.info("✅ Agent started successfully!")
        logger.info("🎤 Listening for user input...")
        logger.info("⏰ Session timeout: 30 seconds")
        logger.info("👋 Automatic goodbye detection enabled")

    except Exception as e:
        logger.error(f"💥 Agent startup error: {e}", exc_info=True)
        raise
 

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))