"""
Unified RAG Engine - Complete implementation combining PDF and JSON search
Optimized for production use with comprehensive error handling
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time

from langchain_community.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'pdf_dir': Path("E:/University Informations"),
    'json_dir': Path("E:/uni_json_files"),
    'persist_dir': Path("E:/unified_chroma"),
    'embed_model': "sentence-transformers/all-MiniLM-L6-v2",
    'reranker_model': "cross-encoder/ms-marco-MiniLM-L-6-v2",
    'chunk_size': 800,
    'chunk_overlap': 100,
    'max_context_length': 4000
}

# Global instances
unified_store: Optional[Chroma] = None
reranker: Optional[CrossEncoder] = None
initialization_lock = asyncio.Lock()

class UnifiedRAGEngine:
    """Production-optimized unified RAG engine"""
    
    def __init__(self):
        self.embeddings = None
        self.store = None
        self.reranker = None
        self.is_initialized = False
        self.initialization_time = None
        self.document_count = 0
    
    async def initialize(self):
        """Initialize the unified RAG system"""
        global unified_store, reranker
        
        async with initialization_lock:
            if self.is_initialized:
                logger.info("RAG engine already initialized")
                return
            
            start_time = time.time()
            
            try:
                logger.info("Initializing unified RAG engine...")
                
                # Initialize embeddings
                logger.info("Loading embedding model...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=CONFIG['embed_model'],
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Create persist directory
                CONFIG['persist_dir'].mkdir(parents=True, exist_ok=True)
                
                # Check if we need to rebuild the database
                should_rebuild = self._should_rebuild_database()
                
                if should_rebuild:
                    logger.info("Building new unified database...")
                    await self._build_unified_database()
                else:
                    logger.info("Loading existing unified database...")
                    unified_store = Chroma(
                        persist_directory=str(CONFIG['persist_dir']),
                        embedding_function=self.embeddings
                    )
                
                self.store = unified_store
                
                # Initialize reranker
                logger.info("Loading reranker model...")
                self.reranker = await asyncio.to_thread(
                    CrossEncoder, CONFIG['reranker_model']
                )
                reranker = self.reranker
                
                # Get document count
                try:
                    collection = self.store._collection
                    self.document_count = collection.count()
                except Exception as e:
                    logger.warning(f"Could not get document count: {e}")
                    self.document_count = 0
                
                self.is_initialized = True
                self.initialization_time = time.time() - start_time
                
                logger.info(f"Unified RAG engine ready in {self.initialization_time:.2f}s")
                logger.info(f"Database contains {self.document_count} document chunks")
                
            except Exception as e:
                logger.error(f"Failed to initialize RAG engine: {e}", exc_info=True)
                raise RuntimeError(f"RAG initialization failed: {e}")
    
    def _should_rebuild_database(self) -> bool:
        """Check if database needs rebuilding"""
        try:
            # Check if persist directory exists and is not empty
            if not CONFIG['persist_dir'].exists() or not any(CONFIG['persist_dir'].iterdir()):
                logger.info("No existing database found")
                return True
            
            # Check if source directories exist
            pdf_exists = CONFIG['pdf_dir'].exists() and any(CONFIG['pdf_dir'].glob('*.pdf'))
            json_exists = CONFIG['json_dir'].exists() and any(CONFIG['json_dir'].glob('*.json'))
            
            if not pdf_exists and not json_exists:
                logger.warning("No source files found in configured directories")
                return False
            
            # Simple timestamp check
            try:
                db_files = list(CONFIG['persist_dir'].rglob('*'))
                if not db_files:
                    return True
                
                db_time = max(f.stat().st_mtime for f in db_files if f.is_file())
                
                source_time = 0
                for source_dir in [CONFIG['pdf_dir'], CONFIG['json_dir']]:
                    if source_dir.exists():
                        source_files = list(source_dir.rglob('*'))
                        if source_files:
                            dir_time = max(f.stat().st_mtime for f in source_files if f.is_file())
                            source_time = max(source_time, dir_time)
                
                if source_time > db_time:
                    logger.info("Source files are newer than database")
                    return True
                    
            except Exception as e:
                logger.warning(f"Could not check file timestamps: {e}")
            
            logger.info("Using existing database")
            return False
            
        except Exception as e:
            logger.warning(f"Database check failed: {e}")
            return True
    
    async def _build_unified_database(self):
        """Build unified database from PDF and JSON sources"""
        global unified_store
        
        all_documents = []
        
        # Process PDF files
        if CONFIG['pdf_dir'].exists():
            logger.info("Processing PDF files...")
            pdf_docs = await self._load_pdf_documents()
            all_documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF document chunks")
        
        # Process JSON files
        if CONFIG['json_dir'].exists():
            logger.info("Processing JSON files...")
            json_docs = await self._load_json_documents()
            all_documents.extend(json_docs)
            logger.info(f"Loaded {len(json_docs)} JSON document chunks")
        
        if not all_documents:
            raise RuntimeError("No documents found in source directories")
        
        # Create unified vector store
        logger.info(f"Creating vector store with {len(all_documents)} documents...")
        unified_store = await asyncio.to_thread(
            Chroma.from_documents,
            documents=all_documents,
            embedding=self.embeddings,
            persist_directory=str(CONFIG['persist_dir'])
        )
        
        logger.info("Unified database created successfully")
    
    async def _load_pdf_documents(self) -> List[Document]:
        """Load and process PDF documents"""
        documents = []
        
        def _load_and_split_pdfs():
            try:
                # Load PDFs
                loader = DirectoryLoader(
                    str(CONFIG['pdf_dir']), 
                    glob="**/*.pdf", 
                    loader_cls=PyPDFLoader,
                    show_progress=False
                )
                raw_docs = loader.load()
                
                if not raw_docs:
                    logger.warning("No PDF documents found")
                    return []
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CONFIG['chunk_size'],
                    chunk_overlap=CONFIG['chunk_overlap'],
                    add_start_index=True
                )
                
                split_docs = text_splitter.split_documents(raw_docs)
                
                # Add metadata
                for doc in split_docs:
                    doc.metadata['source_type'] = 'pdf'
                    doc.metadata['processed_at'] = datetime.now().isoformat()
                    
                    # Clean up source path
                    source = doc.metadata.get('source', '')
                    if source:
                        doc.metadata['source'] = Path(source).name
                
                return split_docs
                
            except Exception as e:
                logger.error(f"Error processing PDF documents: {e}")
                return []
        
        documents = await asyncio.to_thread(_load_and_split_pdfs)
        return documents
    
    async def _load_json_documents(self) -> List[Document]:
        """Load and process JSON documents"""
        def _process_json_files():
            json_docs = []
            
            for json_file in CONFIG['json_dir'].glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Convert JSON to readable text
                    text_chunks = self._json_to_text_chunks(data, json_file.stem)
                    
                    for i, text in enumerate(text_chunks):
                        if text and text.strip():
                            json_docs.append(Document(
                                page_content=text,
                                metadata={
                                    'source': json_file.name,
                                    'source_type': 'json',
                                    'chunk_index': i,
                                    'total_chunks': len(text_chunks),
                                    'processed_at': datetime.now().isoformat()
                                }
                            ))
                
                except Exception as e:
                    logger.warning(f"Error processing {json_file}: {e}")
            
            return json_docs
        
        documents = await asyncio.to_thread(_process_json_files)
        return documents
    
    def _json_to_text_chunks(self, data: Any, filename: str) -> List[str]:
        """Convert JSON data to searchable text chunks"""
        texts = []
        
        def process_value(value: Any, context_path: str = "") -> None:
            """Recursively process JSON values"""
            
            if isinstance(value, dict):
                # Handle specific object types
                if self._is_event_or_holiday(value):
                    text = self._format_event_holiday(value)
                    if text:
                        texts.append(f"[{filename}] {text}")
                
                elif self._is_person_profile(value):
                    text = self._format_person_profile(value)
                    if text:
                        texts.append(f"[{filename}] {text}")
                
                elif self._is_faq_item(value):
                    text = self._format_faq_item(value)
                    if text:
                        texts.append(f"[{filename}] {text}")
                
                else:
                    # Process dictionary recursively
                    for key, sub_value in value.items():
                        new_context = f"{context_path}.{key}" if context_path else key
                        process_value(sub_value, new_context)
            
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    new_context = f"{context_path}[{i}]" if context_path else f"item_{i}"
                    process_value(item, new_context)
            
            elif value is not None and str(value).strip():
                # Handle primitive values
                if context_path:
                    readable_context = context_path.replace("_", " ").replace(".", " ").title()
                    texts.append(f"[{filename}] {readable_context}: {value}")
        
        process_value(data)
        return [text for text in texts if text and text.strip()]
    
    def _is_event_or_holiday(self, obj: Dict) -> bool:
        """Check if object represents an event or holiday"""
        return (
            isinstance(obj, dict) and (
                ("type" in obj and "title" in obj and "start_date" in obj) or
                ("name" in obj and "date" in obj) or
                ("event" in obj and ("date" in obj or "from_date" in obj))
            )
        )
    
    def _is_person_profile(self, obj: Dict) -> bool:
        """Check if object represents a person profile"""
        return (
            isinstance(obj, dict) and 
            "name" in obj and 
            any(key in obj for key in ["title", "designation", "department", "bio"])
        )
    
    def _is_faq_item(self, obj: Dict) -> bool:
        """Check if object represents an FAQ item"""
        return isinstance(obj, dict) and "q" in obj and "a" in obj
    
    def _format_event_holiday(self, obj: Dict) -> str:
        """Format event/holiday object to readable text"""
        title = obj.get("title") or obj.get("name") or obj.get("event", "Event")
        
        # Get date information
        date_info = (
            obj.get("start_date") or 
            obj.get("date") or 
            obj.get("from_date", "")
        )
        
        text = title
        if date_info:
            text += f" on {date_info}"
        
        # Add additional info
        if obj.get("type"):
            text += f" ({obj['type']})"
        
        return text
    
    def _format_person_profile(self, obj: Dict) -> str:
        """Format person profile to readable text"""
        name = obj.get("name", "")
        title = obj.get("title") or obj.get("designation", "")
        department = obj.get("department", "")
        
        text = name
        if title:
            text += f" is {title}"
        if department:
            text += f" in {department} department"
        
        return text
    
    def _format_faq_item(self, obj: Dict) -> str:
        """Format FAQ item to readable text"""
        question = obj.get("q", "")
        answer = obj.get("a", "")
        
        return f"Q: {question} A: {answer}"
    
    async def search(self, query: str, k: int = 10, top_n: int = 3, rerank: bool = True) -> Dict[str, Any]:
        """Search the unified knowledge base"""
        if not self.is_initialized:
            await self.initialize()
        
        if not self.store:
            raise RuntimeError("Vector store not available")
        
        search_start = time.time()
        
        try:
            # Retrieve documents
            def _retrieve_docs():
                retriever = self.store.as_retriever(search_kwargs={"k": k})
                return retriever.get_relevant_documents(query)
            
            docs = await asyncio.to_thread(_retrieve_docs)
            
            if not docs:
                return {
                    "context": "",
                    "sources": [],
                    "total_results": 0,
                    "search_time": time.time() - search_start
                }
            
            # Rerank if enabled
            if rerank and self.reranker and len(docs) > 1:
                try:
                    pairs = [(query, doc.page_content) for doc in docs]
                    scores = await asyncio.to_thread(self.reranker.predict, pairs)
                    
                    # Sort by relevance score (higher is better)
                    scored_docs = list(zip(docs, scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    
                    top_docs = [doc for doc, _ in scored_docs[:top_n]]
                    
                except Exception as e:
                    logger.warning(f"Reranking failed, using original order: {e}")
                    top_docs = docs[:top_n]
            else:
                top_docs = docs[:top_n]
            
            # Build response
            context_parts = []
            sources = []
            
            for doc in top_docs:
                source = doc.metadata.get("source", "unknown")
                source_type = doc.metadata.get("source_type", "unknown")
                
                context_parts.append(f"Source: {source}\n{doc.page_content}")
                sources.append(f"{source} ({source_type})")
            
            combined_context = "\n\n".join(context_parts)
            
            # Truncate if too long
            if len(combined_context) > CONFIG['max_context_length']:
                combined_context = combined_context[:CONFIG['max_context_length']]
                combined_context += "...\n[Content truncated for brevity]"
            
            search_time = time.time() - search_start
            
            return {
                "context": combined_context,
                "sources": list(dict.fromkeys(sources)),  # Remove duplicates
                "total_results": len(top_docs),
                "search_time": round(search_time, 3),
                "search_metadata": {
                    "query": query,
                    "k": k,
                    "top_n": top_n,
                    "rerank_used": rerank and self.reranker is not None,
                    "context_length": len(combined_context),
                    "total_docs_found": len(docs)
                }
            }
        
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {e}")

# Global instance
_rag_engine = UnifiedRAGEngine()

# Public interface functions
async def initialize_unified_rag():
    """Initialize the unified RAG engine"""
    await _rag_engine.initialize()

async def search_unified_knowledge(
    query: str, 
    k: int = 10, 
    top_n: int = 3, 
    rerank: bool = True
) -> Dict[str, Any]:
    """Search the unified knowledge base"""
    return await _rag_engine.search(query, k, top_n, rerank)

# Backwards compatibility functions
async def get_rag_answer_async(query: str, k: int = 5, top_n: int = 2) -> Dict[str, Any]:
    """Legacy function for backwards compatibility"""
    result = await search_unified_knowledge(query, k, top_n, rerank=True)
    
    # Convert to legacy format
    return {
        "context": result["context"],
        "sources": [source.split(" (")[0] for source in result["sources"]]  # Remove type info
    }

def get_rag_health() -> Dict[str, Any]:
    """Get RAG engine health status"""
    return {
        "initialized": _rag_engine.is_initialized,
        "has_embeddings": _rag_engine.embeddings is not None,
        "has_store": _rag_engine.store is not None,
        "has_reranker": _rag_engine.reranker is not None,
        "document_count": _rag_engine.document_count,
        "initialization_time": _rag_engine.initialization_time,
        "config": {
            "embed_model": CONFIG['embed_model'],
            "chunk_size": CONFIG['chunk_size'],
            "max_context_length": CONFIG['max_context_length'],
            "pdf_dir": str(CONFIG['pdf_dir']),
            "json_dir": str(CONFIG['json_dir']),
            "persist_dir": str(CONFIG['persist_dir'])
        }
    }

# Test function
async def test_unified_rag():
    """Test the unified RAG engine"""
    print("Testing Unified RAG Engine...")
    
    try:
        print("1. Initializing RAG engine...")
        await initialize_unified_rag()
        print("   ✓ Initialization successful")
        
        # Check health
        health = get_rag_health()
        print(f"   ✓ Document count: {health['document_count']}")
        print(f"   ✓ Initialization time: {health['initialization_time']:.2f}s")
        
        # Test queries
        test_queries = [
            "Who is the Vice Chancellor?",
            "When does the semester start?",
            "What are the hostel rules?",
            "How to join Next Tech Lab?",
            "What are the examination dates?",
            "Tell me about some cse faculty members"
        ]
        
        print("\n2. Testing search queries...")
        for i, query in enumerate(test_queries, 1):
            try:
                result = await search_unified_knowledge(query, k=8, top_n=3)
                
                print(f"   Query {i}: {query}")
                print(f"   ✓ Found {result['total_results']} results in {result['search_time']}s")
                print(f"   ✓ Sources: {', '.join(result['sources'][:2])}{'...' if len(result['sources']) > 2 else ''}")
                print(f"   ✓ Context length: {len(result['context'])} characters")
                snippet = (result.get('context') or "").strip()
                if snippet:
                    max_chars = 1000
                    if len(snippet) > max_chars:
                        snippet_display = snippet[:max_chars] + "\n...[truncated]"
                    else:
                        snippet_display = snippet
                else:
                    snippet_display = "[No context returned]"
                
                print("   ✓ Result context snippet:\n" + "\n".join("      " + line for line in snippet_display.splitlines()))
                # Check if we got meaningful results
                if not result['context'].strip():
                    print(f"   ⚠ Warning: No context found for query")
                
            except Exception as e:
                print(f"   ✗ Query {i} failed: {e}")
        
        print("\n3. Testing backwards compatibility...")
        legacy_result = await get_rag_answer_async("test query")
        print(f"   ✓ Legacy function works: {len(legacy_result['context'])} chars")
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_unified_rag()) 