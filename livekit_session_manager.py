"""
Optimized Session Manager - Simplified and more reliable
Removes over-engineering while maintaining essential functionality
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """Simplified user session with essential data only"""
    participant_id: str
    start_time: float = field(default_factory=time.time)
    query_count: int = 0
    last_activity: float = field(default_factory=time.time)
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=5))  # Keep last 5 interactions
    
    def add_interaction(self, query: str, response: str):
        """Add interaction to history"""
        self.conversation_history.append({
            'timestamp': time.time(),
            'query': query[:100],  # Truncate long queries
            'response': response[:200],  # Truncate long responses
        })
        self.query_count += 1
        self.last_activity = time.time()
    
    def get_context_summary(self) -> str:
        """Get a brief context summary for the LLM"""
        if not self.conversation_history:
            return ""
        
        # Return only the last 2 interactions for context
        recent = list(self.conversation_history)[-2:]
        context_parts = []
        
        for interaction in recent:
            context_parts.append(f"Previous Q: {interaction['query']}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def is_expired(self, timeout_seconds: int = 180) -> bool:  # 3 minutes default
        """Check if session has expired"""
        return (time.time() - self.last_activity) > timeout_seconds
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information for monitoring"""
        return {
            'participant_id': self.participant_id,
            'duration_minutes': round((time.time() - self.start_time) / 60, 2),
            'query_count': self.query_count,
            'last_activity_seconds_ago': round(time.time() - self.last_activity, 1),
            'has_history': len(self.conversation_history) > 0
        }

class OptimizedSessionManager:
    """Simplified session manager focused on reliability"""
    
    def __init__(self, session_timeout: int = 180):  # 3 minutes default
        self.sessions: Dict[str, UserSession] = {}
        self.session_timeout = session_timeout
        self.cleanup_task: Optional[asyncio.Task] = None
        self.total_sessions_created = 0
        self.start_time = time.time()
        
        # Start cleanup task only if an event loop is already running.
        # Creating a task at import time (no running loop) raises RuntimeError.
        try:
            asyncio.get_running_loop()
            self._start_cleanup_task()
        except RuntimeError:
            logger.info("No running event loop at init; cleanup task will be started when an event loop is available.")
        
        logger.info(f"Session manager initialized with {session_timeout}s timeout")
    
    def get_or_create_session(self, participant_id: str) -> UserSession:
        """Get existing session or create new one"""
        
        # Clean up expired session first
        if participant_id in self.sessions:
            session = self.sessions[participant_id]
            if session.is_expired(self.session_timeout):
                logger.info(f"Session expired for {participant_id}")
                del self.sessions[participant_id]
            else:
                return session
        
        # Create new session
        session = UserSession(participant_id=participant_id)
        self.sessions[participant_id] = session
        self.total_sessions_created += 1
        
        logger.info(f"New session created for {participant_id} (total: {self.total_sessions_created})")
        return session
    
    def handle_interaction(self, participant_id: str, query: str, response: str) -> bool:
        """Handle user interaction - returns True if session should end"""
        
        session = self.get_or_create_session(participant_id)
        session.add_interaction(query, response)
        
        # Simple goodbye detection
        query_lower = query.lower().strip()
        goodbye_words = ['bye', 'goodbye', 'thanks', 'thank you', 'see you', 'good night']
        
        # Check if query ends with goodbye (more reliable than contains)
        is_goodbye = any(query_lower.endswith(word) for word in goodbye_words)
        
        if is_goodbye:
            logger.info(f"Goodbye detected for {participant_id}")
            # Don't delete session immediately - let it expire naturally
            return True
        
        return False
    
    def get_session_context(self, participant_id: str) -> str:
        """Get conversation context for a participant"""
        if participant_id not in self.sessions:
            return ""
        
        session = self.sessions[participant_id]
        return session.get_context_summary()
    
    def _start_cleanup_task(self):
        """Start background cleanup task if an event loop is available."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — caller should start the task later from within an async context
            logger.debug("Cannot start cleanup task: no running event loop")
            return

        if self.cleanup_task is None or self.cleanup_task.done():
            # Use loop.create_task to attach to the currently running loop
            self.cleanup_task = loop.create_task(self._cleanup_loop())

    def ensure_cleanup_started(self) -> bool:
        """Attempt to start the cleanup task. Returns True if started, False otherwise.

        Call this from within an async context (i.e. when an event loop is running).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("ensure_cleanup_started called but no running event loop was found.")
            return False

        self._start_cleanup_task()
        return self.cleanup_task is not None
    
    async def _cleanup_loop(self):
        """Background task to clean up expired sessions"""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._cleanup_expired_sessions()
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        if not self.sessions:
            return
        
        expired_ids = []
        for participant_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout):
                expired_ids.append(participant_id)
        
        for participant_id in expired_ids:
            session_info = self.sessions[participant_id].get_session_info()
            del self.sessions[participant_id]
            logger.info(f"Cleaned up expired session: {session_info}")
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired sessions")
    
    def stop_cleanup_task(self):
        """Stop the cleanup task"""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        active_sessions = len(self.sessions)
        uptime_hours = (time.time() - self.start_time) / 3600
        
        return {
            'active_sessions': active_sessions,
            'total_sessions_created': self.total_sessions_created,
            'uptime_hours': round(uptime_hours, 2),
            'session_timeout_minutes': self.session_timeout / 60,
            'sessions_per_hour': round(self.total_sessions_created / max(uptime_hours, 0.1), 2)
        }
    
    def get_all_session_info(self) -> Dict[str, Any]:
        """Get information about all active sessions"""
        return {
            participant_id: session.get_session_info() 
            for participant_id, session in self.sessions.items()
        }
    
    def force_end_session(self, participant_id: str) -> bool:
        """Force end a specific session"""
        if participant_id in self.sessions:
            session_info = self.sessions[participant_id].get_session_info()
            del self.sessions[participant_id]
            logger.info(f"Force ended session: {session_info}")
            return True
        return False
    
    def __del__(self):
        """Cleanup when manager is destroyed"""
        self.stop_cleanup_task()

# Global instance for easy import
session_manager = OptimizedSessionManager()

# Utility functions for backwards compatibility
def get_or_create_session(participant_id: str) -> UserSession:
    return session_manager.get_or_create_session(participant_id)

def handle_message(participant_id: str, query: str, response: str) -> bool:
    return session_manager.handle_interaction(participant_id, query, response)

if __name__ == "__main__":
    # Test the session manager
    async def test_session_manager():
        print("Testing Optimized Session Manager...")
        
        manager = OptimizedSessionManager(session_timeout=10)  # 10 second timeout for testing
        
        # Test session creation
        session1 = manager.get_or_create_session("user1")
        print(f"✓ Created session: {session1.participant_id}")
        
        # Test interaction handling
        is_goodbye = manager.handle_interaction("user1", "Hello", "Hi there!")
        print(f"✓ Handled interaction, goodbye: {is_goodbye}")
        
        # Test goodbye detection
        is_goodbye = manager.handle_interaction("user1", "Thanks, goodbye!", "You're welcome!")
        print(f"✓ Goodbye detected: {is_goodbye}")
        
        # Test context
        context = manager.get_session_context("user1")
        print(f"✓ Context: {context}")
        
        # Test stats
        stats = manager.get_manager_stats()
        print(f"✓ Stats: {stats}")
        
        # Test cleanup
        await asyncio.sleep(12)  # Wait for expiration
        await manager._cleanup_expired_sessions()
        
        final_stats = manager.get_manager_stats()
        print(f"✓ Final stats: {final_stats}")
        
        manager.stop_cleanup_task()
        print("✓ All tests completed")
    
    asyncio.run(test_session_manager())