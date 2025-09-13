# livekit_session_manager.py
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LiveKitSessionManager:
    def __init__(self, timeout_seconds=30):
        self.sessions = {}
        self.session_dir = Path("session_data")
        self.session_dir.mkdir(exist_ok=True)
        self.timeout = timeout_seconds  # 30 seconds default
        self.cleanup_task = None
        self.is_running = False
        
        # Start automatic cleanup
        self.start_cleanup_task()
        logger.info(f"Session manager initialized with {timeout_seconds}s timeout")
    
    def start_cleanup_task(self):
        """Start the automatic session cleanup task"""
        if not self.cleanup_task or self.cleanup_task.done():
            self.is_running = True
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")
    
    async def _cleanup_loop(self):
        """Background task to check and cleanup inactive sessions"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                self.check_inactive_sessions()
            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def stop_cleanup_task(self):
        """Stop the automatic cleanup task"""
        self.is_running = False
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            logger.info("Session cleanup task stopped")
    
    def get_or_create_session(self, user_id):
        """Get existing session or create new one"""
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(user_id, self.session_dir)
            logger.info(f"✅ New session created for user: {user_id}")
        else:
            self.sessions[user_id].update_activity()
            logger.info(f"🔄 Session activity updated for user: {user_id}")
        
        return self.sessions[user_id]
    
    def end_session(self, user_id, reason="manual"):
        """End a specific user session"""
        if user_id in self.sessions:
            # Save the session data before ending
            self.sessions[user_id].save()
            
            # Start new session (clear current conversation but keep history)
            self.sessions[user_id].start_new_session()
            
            logger.info(f"🔚 Session ended for {user_id} - Reason: {reason}")
            logger.info(f"🆕 New session started for {user_id}")
            
            return True
        return False
    
    def check_inactive_sessions(self):
        """Check and cleanup inactive sessions"""
        current_time = time.time()
        inactive_users = []
        
        for user_id, session in self.sessions.items():
            if current_time - session.last_activity > self.timeout:
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            logger.warning(f"⏰ Session timeout detected for {user_id}")
            self.end_session(user_id, reason="timeout")
    
    def is_goodbye_message(self, message):
        """Check if message is a goodbye/exit message"""
        goodbye_phrases = [
            "goodbye", "bye", "see you", "exit", "quit", 
            "thank you bye", "good bye", "farewell", 
            "see you later", "talk to you later", "ttyl"
        ]
        
        message_lower = message.lower().strip()
        return any(phrase in message_lower for phrase in goodbye_phrases)
    
    def handle_message(self, user_id, message, response):
        """Handle a message and check for session ending conditions"""
        session = self.get_or_create_session(user_id)
        
        # Add the interaction
        session.add_interaction(message, response)
        
        # Check if it's a goodbye message
        if self.is_goodbye_message(message):
            logger.info(f"👋 Goodbye message detected from {user_id}: {message}")
            self.end_session(user_id, reason="goodbye")
            return True  # Indicates session was ended
        
        return False  # Session continues
    
    def get_session_count(self):
        """Get current active session count"""
        return len(self.sessions)


class UserSession:
    def __init__(self, user_id, session_dir):
        self.user_id = user_id
        self.session_dir = session_dir
        self.file_path = session_dir / f"{user_id}_history.json"
        self.last_activity = time.time()
        self.session_start_time = time.time()
        
        # Current conversation (gets cleared when new session starts)
        self.current_conversation = []
        
        # Persistent history structure
        self.history = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "all_sessions": []  # List of completed sessions
        }
        
        # Load existing history
        self.load()
        logger.info(f"📂 User session initialized for {user_id}")
    
    def add_interaction(self, question, answer):
        """Add interaction to current conversation"""
        self.update_activity()
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": str(answer)
        }
        
        self.current_conversation.append(interaction)
        
        # Auto-save every interaction
        self.save()
        logger.info(f"💬 Interaction added for {self.user_id} (current session: {len(self.current_conversation)} messages)")
    
    def start_new_session(self):
        """Start a new session - save current and reset"""
        if self.current_conversation:
            # Save current conversation to history
            session_data = {
                "session_id": len(self.history["all_sessions"]) + 1,
                "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "conversation": self.current_conversation.copy(),
                "message_count": len(self.current_conversation)
            }
            
            self.history["all_sessions"].append(session_data)
            logger.info(f"💾 Session saved: {len(self.current_conversation)} messages archived")
        
        # Reset for new session
        self.current_conversation = []
        self.session_start_time = time.time()
        self.update_activity()
        
        # Save the updated history
        self.save()
        logger.info(f"🆕 New session started for {self.user_id}")
    
    def get_current_context(self, message_count=3):
        """Get recent messages from current conversation for context"""
        if not self.current_conversation:
            return ""
        
        recent = self.current_conversation[-message_count:]
        context_lines = []
        
        for msg in recent:
            context_lines.append(f"User: {msg['question']}")
            context_lines.append(f"Assistant: {msg['answer']}")
        
        return "\n".join(context_lines)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def save(self):
        """Save session data to file"""
        try:
            # Prepare data to save
            save_data = self.history.copy()
            save_data["current_session"] = {
                "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
                "conversation": self.current_conversation,
                "last_activity": datetime.fromtimestamp(self.last_activity).isoformat()
            }
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"❌ Failed to save session for {self.user_id}: {e}")
    
    def load(self):
        """Load session data from file"""
        try:
            if self.file_path.exists():
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    # Load persistent history
                    self.history = {
                        "user_id": data.get("user_id", self.user_id),
                        "created_at": data.get("created_at", datetime.now().isoformat()),
                        "all_sessions": data.get("all_sessions", [])
                    }
                    
                    # Load current session if it exists
                    current_session = data.get("current_session", {})
                    if current_session and current_session.get("conversation"):
                        self.current_conversation = current_session["conversation"]
                        logger.info(f"📥 Loaded session for {self.user_id}: {len(self.current_conversation)} current messages")
                    
                    logger.info(f"📚 Total history: {len(self.history['all_sessions'])} completed sessions")
                
        except Exception as e:
            logger.error(f"❌ Failed to load session for {self.user_id}: {e}")