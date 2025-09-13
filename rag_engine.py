# rag_engine.py
from pathlib import Path
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from sentence_transformers import CrossEncoder

# Globals
vector_store = None
reranker = None

def initialize_rag_engine_blocking():
    """
    Initializes the RAG pipeline only once.
    Loads persisted ChromaDB if it exists, otherwise builds it.
    """
    global vector_store, reranker

    file_path = Path("E:/University Informations")
    persist_dir = Path(r"E:\chroma_langchain")
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ If DB exists, just load it
    if persist_dir.exists() and any(persist_dir.iterdir()):
        print("✅ Loading existing ChromaDB...")
        vector_store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings
        )
    else:
        print("⚡ Creating new ChromaDB (first time only)...")
        loader = DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, add_start_index=True
        )
        texts = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )

    # ✅ Simple reranker (cross-encoder, small model for speed)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("✅ RAG engine initialized!")

async def initialize_rag_engine():
    """Async wrapper to initialize RAG engine in a non-blocking way."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, initialize_rag_engine_blocking)

async def get_rag_answer_async(query: str, k: int = 5, top_n: int = 2):
    """
    Retrieve documents for a query, rerank them, and return context + sources.
    """
    global vector_store, reranker
    if not vector_store:
        raise RuntimeError("RAG engine not initialized. Call initialize_rag_engine() first.")
    def _retrieve():
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)

    docs = await asyncio.to_thread(_retrieve)

    if not docs:
        return {"context": "", "sources": []}

    # Step 2: Rerank docs (blocking)
    if reranker:
        pairs = [(query, d.page_content) for d in docs]

        # reranker.predict is synchronous and can be expensive; run in thread
        scores = await asyncio.to_thread(reranker.predict, pairs)
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = reranked[:top_n]
        final_context = "\n\n".join(d.page_content for d, _ in top_docs)
        sources = [d.metadata.get("source", "unknown") for d, _ in top_docs]
    else:
        final_context = "\n\n".join(d.page_content for d in docs[:top_n])
        sources = [d.metadata.get("source", "unknown") for d in docs[:top_n]]

    return {"context": final_context, "sources": sources}
# --- Testing Block ---
if __name__ == "__main__":
    import asyncio
    import pprint

    async def test_rag():
        """Main function to test the RAG engine."""
        print("--- Initializing RAG Engine for Testing ---")
        await initialize_rag_engine()
        print("\n--- Testing RAG Query ---")

        # You can change this query to test different things
        test_query = "What are the Attendance Policies?"
        print(f"Query: {test_query}\n")

        result = await get_rag_answer_async(test_query)

        print("--- RAG Result ---")
        pprint.pprint(result)

    # This allows you to run `python rag_engine.py` to test it
    asyncio.run(test_rag())