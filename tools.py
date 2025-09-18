from livekit.agents import function_tool, RunContext, ToolError
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from rag_engine import search_unified_knowledge

logger = logging.getLogger(__name__)

# Query classification patterns
QUERY_PATTERNS = {
    'calendar': ['when', 'date', 'time', 'schedule', 'semester', 'exam', 'holiday', 'calendar', 'academic year'],
    'people': ['who', 'faculty', 'professor', 'staff', 'dr', 'registrar', 'dean', 'head', 'director'],
    'procedures': ['how', 'apply', 'join', 'register', 'admission', 'enrollment', 'rules', 'policy'],
    'locations': ['where', 'room', 'building', 'hostel', 'mess', 'library', 'lab', 'department'],
    'fees': ['fee', 'cost', 'payment', 'scholarship', 'financial', 'tuition', 'charges'],
    'general': ['what', 'about', 'tell', 'explain', 'describe', 'info', 'information']
}

def classify_query(query: str) -> str:
    """Intelligently classify query to optimize search strategy"""
    query_lower = query.lower()
    
    # Count matches for each category
    scores = {}
    for category, keywords in QUERY_PATTERNS.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            scores[category] = score
    
    if not scores:
        return 'general'
    
    # Return category with highest score
    return max(scores.items(), key=lambda x: x[1])[0]

def optimize_search_params(query_type: str, query_length: int) -> Dict[str, Any]:
    """Optimize search parameters based on query characteristics"""
    
    base_params = {
        'k': 10,           # Default retrieval count
        'top_n': 3,        # Default return count
        'rerank': True     # Use reranking by default
    }
    
    # Adjust based on query type
    if query_type == 'people':
        base_params.update({'k': 15, 'top_n': 5})  # More thorough for people searches
    elif query_type == 'calendar':
        base_params.update({'k': 12, 'top_n': 4})  # Good coverage for dates
    elif query_type == 'procedures':
        base_params.update({'k': 20, 'top_n': 6})  # Comprehensive for how-to
    elif query_type in ['fees', 'locations']:
        base_params.update({'k': 8, 'top_n': 3})   # Focused search
    
    # Adjust based on query complexity
    if query_length > 100:  # Complex query
        base_params['k'] = min(base_params['k'] + 5, 25)
        base_params['top_n'] = min(base_params['top_n'] + 2, 8)
    elif query_length < 20:  # Simple query
        base_params['k'] = max(base_params['k'] - 3, 5)
        base_params['top_n'] = max(base_params['top_n'] - 1, 2)
    
    return base_params

@function_tool
async def intelligent_search(context: RunContext, query: str) -> Dict[str, Any]:
    """
    Unified intelligent search tool that handles all university information queries.
    
    This single tool replaces all previous search tools and uses smart routing
    to find the most relevant information efficiently.
    
    Args:
        context: LiveKit RunContext for session management
        query: The user's question about university information
        
    Returns:
        Dict containing the most relevant information and sources
    """
    
    if not query or not query.strip():
        raise ToolError("Please provide a valid question.")
    
    query = query.strip()
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Step 1: Classify query for optimization
        query_type = classify_query(query)
        search_params = optimize_search_params(query_type, len(query))
        
        logger.info(f"Query classified as '{query_type}': {query[:50]}...")
        logger.debug(f"Search params: {search_params}")
        
        # Step 2: Execute optimized search with timeout
        search_task = search_unified_knowledge(
            query=query,
            **search_params
        )
        
        # Timeout based on query complexity
        timeout = 8.0 if len(query) > 50 else 5.0
        result = await asyncio.wait_for(search_task, timeout=timeout)
        
        # Step 3: Validate and format results
        context_text = result.get("context", "")
        sources = result.get("sources", [])
        
        if not context_text or not context_text.strip():
            # Try a broader search as fallback
            logger.warning(f"No results for '{query}', trying broader search")
            
            fallback_params = {'k': 20, 'top_n': 5, 'rerank': True}
            fallback_result = await asyncio.wait_for(
                search_unified_knowledge(query, **fallback_params), 
                timeout=6.0
            )
            
            context_text = fallback_result.get("context", "")
            sources = fallback_result.get("sources", [])
        
        if not context_text or not context_text.strip():
            raise ToolError("I couldn't find specific information about that. Try asking in a different way or contact university administration.")
        
        # Step 4: Enhanced response formatting
        response = {
            "query": query,
            "context": context_text,
            "sources": sources,
            "query_type": query_type,
            "search_params": search_params
        }
        
        # Step 5: Log performance metrics
        elapsed_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f}s - Type: {query_type}, Results: {len(sources)} sources")
        
        return response

    except asyncio.TimeoutError:
        logger.error(f"Search timeout for query: {query}")
        raise ToolError("Search is taking too long. Please try a simpler question.")
    
    except Exception as e:
        logger.error(f"Search error for '{query}': {e}", exc_info=True)
        
        # Provide helpful error messages based on query type
        if query_type == 'people':
            error_msg = "I couldn't find information about that person. Try using their full name or department."
        elif query_type == 'calendar':
            error_msg = "I couldn't find that date information. Try asking about specific events or semesters."
        elif query_type == 'procedures':
            error_msg = "I couldn't find those procedure details. Try asking about specific processes or requirements."
        else:
            error_msg = "I couldn't find that information. Please try rephrasing your question."
        
        raise ToolError(error_msg)

# Additional utility functions for query processing

def extract_key_entities(query: str) -> List[str]:
    """Extract key entities from query for better search"""
    entities = []
    
    # Extract names (capitalized words)
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    entities.extend(names)
    
    # Extract dates
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', query)
    entities.extend(dates)
    
    # Extract department names
    dept_patterns = [
        r'\b(?:Computer Science|CSE|ECE|Mechanical|Civil|Electrical|IT|Information Technology)\b',
        r'\b(?:Engineering|Department|School|College|Faculty)\b'
    ]
    
    for pattern in dept_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities.extend(matches)
    
    return list(set(entities))  # Remove duplicates

def enhance_query(query: str) -> str:
    """Enhance query with additional context for better search"""
    query_lower = query.lower()
    
    # Add context for ambiguous queries
    if 'registrar' in query_lower and 'who' in query_lower:
        query += " university registrar office contact"
    elif 'exam' in query_lower and 'when' in query_lower:
        query += " examination schedule dates"
    elif 'hostel' in query_lower:
        query += " accommodation residence"
    elif 'mess' in query_lower:
        query += " dining food cafeteria"
    
    return query

# Performance monitoring
class SearchMetrics:
    def __init__(self):
        self.total_searches = 0
        self.successful_searches = 0
        self.avg_response_time = 0.0
        self.query_type_distribution = {}
    
    def record_search(self, query_type: str, response_time: float, success: bool):
        self.total_searches += 1
        if success:
            self.successful_searches += 1
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_searches - 1) + response_time) / 
            self.total_searches
        )
        
        # Track query type distribution
        self.query_type_distribution[query_type] = (
            self.query_type_distribution.get(query_type, 0) + 1
        )
    
    def get_stats(self):
        return {
            'total_searches': self.total_searches,
            'success_rate': self.successful_searches / max(self.total_searches, 1),
            'avg_response_time': round(self.avg_response_time, 3),
            'query_distribution': self.query_type_distribution
        }

# Global metrics instance
search_metrics = SearchMetrics()