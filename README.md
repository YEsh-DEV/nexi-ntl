# Nexi - AI University Assistant

Nexi is an intelligent, voice-driven AI assistant designed to help university students get quick and accurate answers to their questions about campus life. It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide information from a dedicated knowledge base of university documents.

 <!-- Optional: Add a screenshot or logo here -->

## Core Features

*   **Voice-First Interaction**: Students can talk to Nexi naturally, thanks to real-time Speech-to-Text (STT) and Text-to-Speech (TTS).
*   **RAG-Powered Knowledge**: Provides answers based on a specific set of university documents (PDFs), ensuring accuracy and relevance.
*   **Persistent User Sessions**: Remembers the context of a conversation and can recall a user's previous interactions, even if they reconnect.
*   **Session Management**: Automatically handles session timeouts and archives conversations for logging and analysis.
*   **Unique IDs**: Assigns unique identifiers to each session and interaction for robust tracking and debugging.
*   **Cloud & Local LLM Support**: Flexible architecture that can run with cloud-based LLMs (like Groq) for speed or local models (like Ollama) for privacy.

---

## How It Works

Nexi's architecture is built around a real-time media pipeline and a sophisticated backend.

1.  **Audio Input**: The user's speech is captured and sent to a real-time transcription service (Deepgram).
2.  **Query Processing**: The transcribed text is sent to the `Assistant` agent.
3.  **Session Handling**: The `LiveKitSessionManager` retrieves or creates a session for the user, providing conversation history as context.
4.  **RAG Pipeline**: The user's query is sent to the `UniversityRAGEngine`, which searches a ChromaDB vector database for relevant information from the university documents.
5.  **LLM Response Generation**: The query, conversation history, and RAG context are combined into a prompt for a Large Language Model (LLM) to generate a helpful, natural-sounding answer.
6.  **Audio Output**: The LLM's text response is converted back into speech using a TTS service (Cartesia) and streamed to the user.

---

## Technology Stack

*   **Real-time Communication**: LiveKit
*   **AI Agent Framework**: `livekit-agents`
*   **Language Models (LLM)**: Groq, Ollama
*   **Speech-to-Text (STT)**: Deepgram
*   **Text-to-Speech (TTS)**: Cartesia
*   **RAG Framework**: LlamaIndex
*   **Vector Database**: ChromaDB
*   **Embedding Models**: Hugging Face Sentence Transformers

---

## Setup and Installation

### 1. Prerequisites
*   Python 3.9+
*   Git

### 2. Clone the Repository
```bash
git clone https://github.com/YEsh-DEV/Nexi.git
cd Nexi
```

### 3. Set up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
This project includes a `requirements.txt` file to install all necessary packages.
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys
Create a file named `.env.local` in the root directory and add your API keys.
```env
# .env.local
GROQ_API_KEY="gsk_..."
CARTESIA_API_KEY="..."
DEEPGRAM_API_KEY="..."
```

### 6. Add Knowledge Base Documents
Place all your university-related PDF documents inside the `E:\University Informations` folder. You can change this path in `rag_engine.py` by modifying the `DATA_DIR` variable.

---

## Usage

### 1. Ingest Your Documents (First-time setup)
Before running the agent for the first time, you need to process your PDF documents and store them in the vector database. Run the RAG engine script directly:

```bash
python rag_engine.py
```

This will create a `chromadb` folder containing your indexed knowledge base. You only need to do this once, or whenever you add or update your documents.

### 2. Run the Agent
Start the main application to connect the agent to your LiveKit instance.

```bash
python clean.py
```

The agent will start, initialize all services, and wait for a user to join the LiveKit room.

---

## Project Structure

```
Nexi/
├── University Informations/  # Your source PDF documents
├── chromadb/               # (Generated) Vector database storage
├── session_data/           # (Generated) User conversation history logs
├── .gitignore              # Specifies files for Git to ignore
├── clean.py                # Main application entrypoint for the LiveKit agent
├── rag_engine.py           # Handles document ingestion and retrieval (RAG)
├── livekit_session_manager.py # Manages user sessions and history
├── requirements.txt        # Project dependencies
└── README.md               # This file
```
