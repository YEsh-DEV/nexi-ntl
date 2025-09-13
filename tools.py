from livekit.agents import function_tool, RunContext, ToolError
from rag_engine import get_rag_answer_async
import asyncio
from typing import Any, Dict

@function_tool
async def get_rag_answer(context: RunContext, query: str) -> Dict[str, Any]:
    """
    Function tool to Retrive the top-k documents for a query from the rag_engine  
    
    Args:
        query (str): The user's question about the univeristy information.
        
    Returns:Dict[str, Any]: A dictionary containing the context about the univeristy information
    """

    async def updating(delay: float = 0.5):
        """Send a temporary 'loading' message."""
        await asyncio.sleep(delay)
        try:
            await context.session.generate_reply("Retrieving information from the knowledge base...")
        except Exception:
            pass

    status_task = asyncio.create_task(updating())

    try:
        result = await get_rag_answer_async(query)
        context_text = result.get("context", "")
        if not context_text:
            raise ToolError("No relevant information found.")
        return {"query": query, "context": context_text}

    except Exception as e:
        raise ToolError(f"Error retrieving RAG answer: {e}")

    finally:
        if not status_task.done():
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass
        # Send final message safely
        try:
            await context.session.generate_reply("Information retrieval complete.")
        except Exception:
            pass
