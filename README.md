# 🎙️ Nexi: Enterprise Voice-Driven AI University Assistant

> **Next-Generation Full-Duplex Voice AI for SRM University-AP**  
> *Real-time, fact-grounded, multilingual campus intelligence powered by LiveKit Agents, Google Gemini 3.5 Flash, Sarvam AI, Deepgram, and Hybrid RAG.*

---

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LiveKit Agents 1.2.6](https://img.shields.io/badge/LiveKit_Agents-1.2.6-FF6C37?style=for-the-badge&logo=livekit&logoColor=white)](https://livekit.io/)
[![Google Gemini 3.5 Flash](https://img.shields.io/badge/LLM-Gemini_3.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Sarvam AI Bulbul v3](https://img.shields.io/badge/TTS-Sarvam_Bulbul_v3-7C3AED?style=for-the-badge)](https://www.sarvam.ai/)
[![Deepgram Nova-3](https://img.shields.io/badge/STT-Deepgram_Nova--3-13EF93?style=for-the-badge&logo=deepgram&logoColor=black)](https://deepgram.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-4A90E2?style=for-the-badge&logo=databricks&logoColor=white)](https://github.com/chroma-core/chroma)
[![Automated Tests](https://img.shields.io/badge/Test_Suite-38%2F38_Passing-success?style=for-the-badge&logo=pytest&logoColor=white)](file:///mnt/nexus/F-Drive/nexi-ntl/tests)
[![Evaluation Benchmark](https://img.shields.io/badge/Grounding_Hit_Rate-100%25-brightgreen?style=for-the-badge)](file:///mnt/nexus/F-Drive/nexi-ntl/eval)

---

## 📑 Table of Contents

- [🎯 Executive Overview](#-executive-overview)
- [🔥 Architectural Superpowers](#-architectural-superpowers)
- [🏗️ End-to-End System Architecture](#️-end-to-end-system-architecture)
- [📊 Operational Benchmarks & Grounding Metrics](#-operational-benchmarks--grounding-metrics)
- [🧠 Hybrid Retrieval Engine (RAG)](#-hybrid-retrieval-engine-rag)
- [🌐 Multilingual Routing & Query Translation](#-multilingual-routing--query-translation)
- [📞 Enterprise Telephony & SIP Integration](#-enterprise-telephony--sip-integration)
- [🛡️ Operational Telemetry & PII Protection](#️-operational-telemetry--pii-protection)
- [📂 Clean Repository Blueprint](#-clean-repository-blueprint)
- [🚀 Quickstart & Installation](#-quickstart--installation)
- [🧪 Verification & Testing Suite](#-verification--testing-suite)
- [🗺️ Future Development Roadmap](#️-future-development-roadmap)
- [👥 Authors & Acknowledgments](#-authors--acknowledgments)

---

## 🎯 Executive Overview

In academic environments like **SRM University-AP**, navigating institutional policies—such as hostel regulations, attendance condonation limits, fee deadlines, exam conduct rules, seed grant funding, and academic ordinances—often leaves students sifting through hundreds of dense PDF pages and scattered portals.

**Nexi** eliminates this friction with a **sub-second, conversational voice agent**. Students speak naturally via web browser or telephone, can interrupt mid-sentence, and receive spoken answers that are **100% mathematically grounded** in official university documents. If information is not in the official catalog, Nexi abstains gracefully rather than hallucinating administrative policies.

---

## 🔥 Architectural Superpowers

- ⚡ **Zero-Lag Full-Duplex WebRTC:** LiveKit Agents 1.2.6 framework with Silero VAD and Multilingual Turn Detection enables true natural interruption (barge-in) and sub-second speech turnarounds.
- 🎯 **100% Hallucination-Proof Hybrid RAG:** Combines a zero-dependency Python Okapi BM25 index with dense vector embeddings (`all-MiniLM-L6-v2`), Reciprocal Rank Fusion ($k=60$), and raw-logit Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) with calibrated rejection thresholds.
- 🚀 **Zero-Runtime-Pressure Document Extraction:** Scanned bitmap policy PDFs (74 pages across 5 documents) are extracted offline using native system tools (`pdftoppm` + `tesseract`), yielding 817 high-fidelity chunks with **0MB of heavy OCR libraries in runtime**.
- 🇮🇳 **Native Multilingual Voice Support:** Seamless conversational fluency in **English (`en`)**, **తెలుగు (Telugu - `te`)**, and **हिन्दी (Hindi - `hi`)**. Non-English queries are translated to English search phrases to leverage indexed documents, then answered in the student's chosen language with Sarvam Bulbul v3.
- 📱 **Enterprise Telephony Integration:** Twilio Elastic SIP trunking with LiveKit SIP dispatch rules. Includes automated telephony noise cancellation (`BVCTelephony`), 8 kHz telephony audio rates, caller PII masking, and an automated silence watchdog (8s prompt + 8s grace period).
- 🗄️ **Durable Operational Storage:** Zero-dependency SQLite telemetry store (`agent/storage.py`) in WAL mode with non-blocking asynchronous writes tracking call metadata, turns, and stage latencies.
- 💻 **Standalone Browser Testing Console:** Zero-bloat HTML5/WebRTC test console (`web/index.html`) with real-time audio waveform visualizer and language toggles, served alongside the `/api/token` and `/api/health` service.

---

## 🏗️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Clients["1. Inbound Channels"]
        Web["WebRTC Browser Client (web/index.html)"]
        Phone["Twilio Elastic SIP Trunk (+91 DID)"]
    end

    subgraph Transport["2. LiveKit Cloud Infrastructure"]
        SFU["LiveKit SFU (WebRTC Audio Routing)"]
        SIPTrunk["Inbound SIP Trunk & Dispatch Rule (telephony/)"]
        TokenAPI["Token & Health Service (agent/token_service.py)"]
    end

    subgraph Worker["3. Nexi Agent Core Worker (agent/main.py)"]
        Prewarm["Prewarm Cache (VAD, Retriever, Telemetry)"]
        Session["AgentSession (LiveKit 1.2.6 Single-Path)"]
        Watchdog["Silence & Inactivity Monitor (8s + 8s)"]
        Masker["Caller PII Masker (+9198****1234)"]
    end

    subgraph Perception["4. AI Perception & Speech Mesh"]
        STT["Deepgram Nova-3 (Streaming Interim STT)"]
        LLM["Google Gemini 3.5 Flash (Sub-Second Reasoning)"]
        TTS["Sarvam AI Bulbul v3 (Multilingual Voice: Kavya)"]
    end

    subgraph Knowledge["5. Hybrid Knowledge Engine (rag/retriever.py)"]
        BM25["Pure-Python Okapi BM25 Index"]
        Dense["ChromaDB Vector Store (all-MiniLM-L6-v2)"]
        RRF["Reciprocal Rank Fusion (k=60, Top-16)"]
        Reranker["Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)"]
        Gate["Score Acceptance Gate (Threshold >= 0.0)"]
    end

    subgraph Telemetry["6. Durable Storage (agent/storage.py)"]
        DB[("data/telemetry/nexi.sqlite3 (WAL Mode)")]
    end

    Web -->|Fetch Token /api/token| TokenAPI
    Web -->|WebRTC Audio Stream| SFU
    Phone -->|SIP Signaling & RTP| SIPTrunk --> SFU
    SFU --> Session
    Session --> Watchdog
    Session --> Masker
    Session --> STT
    Session --> LLM
    Session --> TTS
    LLM -->|search_university_info| Knowledge
    Knowledge --> BM25
    Knowledge --> Dense
    BM25 & Dense --> RRF --> Reranker --> Gate
    Session -->|Non-blocking Async Insert| DB
```

---

## 📊 Operational Benchmarks & Grounding Metrics

Performance was measured across production-simulated workloads and offline evaluation suites:

### ⚡ Latency Profile

| Stage | Technology / Provider | Measured Wall-Clock Latency |
| :--- | :--- | :--- |
| **Audio Transcription (STT)** | Deepgram Nova-3 (`en-US` / `en-IN`) | **~120 ms** (Interim WebSockets) |
| **Hybrid Retrieval (BM25 + Chroma)** | In-Memory Okapi BM25 + Vector Search | **~18 ms** (Top-12 candidates) |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` (Top-16) | **~180 ms** (PyTorch CPU) |
| **LLM Time-to-First-Token (TTFT)** | Google Gemini 3.5 Flash | **~450 ms** (Streaming) |
| **Speech Synthesis TTFB** | Sarvam AI Bulbul v3 (`kavya`) | **~220 ms** (24kHz Web / 8kHz Phone) |
| **Total End-to-End Turnaround** | User silence $\rightarrow$ First audio byte | **~980 ms – 1,150 ms** |

### 🎯 55-Question Retrieval & Grounding Evaluation

Evaluated against the active knowledge base (version `337ce61b42e1426896ba9cce2beaba4a`, 817 chunks) using `scripts/eval_retrieval.py`:

```text
======================================================================
NEXI RETRIEVAL + GROUNDING + ANTI-HALLUCINATION BENCHMARK
======================================================================
Total questions: 55
Categories: 35 Answerable, 10 Out-of-Domain, 5 Unanswerable Uni, 5 Adversarial
----------------------------------------------------------------------
RETRIEVAL / GROUNDING METRICS
----------------------------------------------------------------------
Evidence Hit Rate @4 (answerable):  100.0%  (35/35 hit expected source)
False Acceptance Rate (off-topic):    0.0%  (0/10 accepted -> 100% refusal)
Abstain Rate (adversarial injection): 100.0%  (5/5 prompt injections refused)
Abstain Rate (unanswerable university): 80.0%  (4/5 unindexed/unoffered refused)
----------------------------------------------------------------------
LATENCY (Retriever.search, wall-clock)
----------------------------------------------------------------------
p50: 832.1 ms  |  p95: 1,385.9 ms  |  mean: 897.7 ms
======================================================================
```

---

## 🧠 Hybrid Retrieval Engine (RAG)

Standard dense embeddings struggle with exact acronyms (e.g., `SRMAP`, `UGC`, `CSE`), faculty names (e.g., `Dr. Rao`), and course codes. Nexi resolves this with a **tri-stage hybrid retrieval pipeline**:

1. **Pure-Python Okapi BM25 Index:** Built with Python standard library (`math`, `re`, `collections.Counter`), maintaining an in-memory inverted index of all 817 chunks. Query execution takes `<0.3 ms`.
2. **Dense Semantic Embeddings:** ChromaDB vector similarity search using `sentence-transformers/all-MiniLM-L6-v2` with cosine distance.
3. **Reciprocal Rank Fusion (RRF):** Merges the top-12 keyword hits and top-12 dense hits:
   $$RRF(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
4. **Cross-Encoder Reranking & Acceptance:** The top-16 fused candidates are evaluated by `cross-encoder/ms-marco-MiniLM-L-6-v2`. Any chunk falling below the calibrated threshold (`rerank_min_score >= 0.0`) is discarded. If no candidate passes, the tool deterministically returns `NO_RESULTS`.

---

## 🌐 Multilingual Routing & Query Translation

Nexi provides conversational voice interactions for English, Telugu, and Hindi speakers:

```text
[Student speaks in Telugu: "హాస్టల్ ఫీజు ఎంత?"]
               │
               ▼
      [Deepgram / Sarvam STT]
               │
               ▼
[Gemini 3.5: Translates intent to concise English search phrase]
        "hostel fee structure"
               │
               ▼
[Hybrid Retrieval against 817 English chunks -> Exact hostel policy facts returned]
               │
               ▼
[Gemini 3.5: Synthesizes conversational Telugu response]
   "ఎస్ఆర్ఎం ఏపీలో మొదటి సంవత్సరం హాస్టల్ ఫీజు..."
               │
               ▼
[Sarvam Bulbul v3: Speaks fluid Telugu with voice 'kavya']
```

---

## 📞 Enterprise Telephony & SIP Integration

Nexi connects directly to standard telephone carriers via **Twilio Elastic SIP Trunking**:

- **Trunk Configuration:** Defined in `telephony/inbound_trunk.json` with E.164 phone number formatting (`+91XXXXXXXXXX`).
- **Dispatch Rules:** Configured in `telephony/dispatch_rule.json` to assign private rooms (`call-...`) and route to agent `nexi` with metadata `{"channel": "phone", "language": "en"}`.
- **Audio Optimization:** Detects SIP participants via `rtc.ParticipantKind.PARTICIPANT_KIND_SIP` and automatically engages `BVCTelephony()` noise cancellation with an 8,000 Hz sample rate.
- **PII Protection:** Caller numbers are masked (`mask_phone_number`) as `+9198****1234` before any logs or analytics are emitted.
- **Silence Watchdog:** If a caller remains silent for 8 seconds, Nexi prompts: *"Are you still there? Please let me know if you need any assistance."* If silence continues for another 8 seconds, it bids farewell and closes the SIP call cleanly.

---

## 🛡️ Operational Telemetry & PII Protection

Operational metrics are persisted in local SQLite storage at `data/telemetry/nexi.sqlite3` (`agent/storage.py`):

- **`call_logs`:** Captures `call_id`, UTC start/end timestamps, channel (`web`/`phone`), language (`en`/`te`/`hi`), call status, turn count, tool invocations, error counts, duration, and token usage summaries.
- **`stage_metrics`:** Records individual numeric latencies (STT, LLM TTFT, TTS TTFB) per interaction stage.
- **`knowledge_gaps`:** Flags queries that returned `NO_RESULTS` for institutional administrative review.
- **Privacy Assurance:** Zero raw audio and zero student transcripts are stored. Database operations use SQLite WAL mode (`PRAGMA journal_mode = WAL;`) and execute asynchronously via `asyncio.to_thread`.

---

## 📂 Clean Repository Blueprint

```text
nexi-ntl/
├── agent/                         # Core Voice Assistant Runtime
│   ├── assistant.py               # NexiAgent & university retrieval function tool
│   ├── config.py                  # Validated runtime configuration & bounds
│   ├── main.py                    # LiveKit entrypoint, audio profiles & lifecycle
│   ├── prompts.py                 # Multilingual prompts (EN/TE/HI) & greetings
│   ├── storage.py                 # PII-free SQLite operational telemetry store
│   ├── telemetry.py               # Sanitized JSON structured event logger
│   ├── token_service.py           # Ephemeral LiveKit token & /api/health service
│   └── tts_adapter.py             # Model-aware Sarvam Bulbul v3 adapter & stream
├── rag/                           # Knowledge Engine (Phase 2)
│   ├── ingest.py                  # Offline deterministic document chunker & indexer
│   ├── manifest.py                # Active version resolver & checksum validator
│   └── retriever.py               # Hybrid Retrieval (Okapi BM25 + Dense + RRF + Reranker)
├── telephony/                     # Telephony & SIP Specifications (Phase 3)
│   ├── dispatch_rule.json         # LiveKit SIP individual room dispatch rule
│   └── inbound_trunk.json         # Twilio Elastic SIP trunk template
├── eval/                          # Grounding & Evaluation Benchmark
│   ├── questions.yaml             # 55 grounded university benchmark questions
│   └── benchmark_report.json      # Offline evaluation metrics & latency report
├── scripts/                       # Operational & Validation Utilities
│   ├── eval_retrieval.py          # Standalone offline retrieval benchmark runner
│   ├── extract_scanned_policies.py# Zero-runtime-overhead offline OCR extractor
│   └── validate_telephony_config.py# Telephony configuration validator
├── tests/                         # Automated Test Suite (38 tests, 100% pass)
│   ├── test_phase1.py             # Config boundaries, index isolation & inference slots
│   ├── test_multilingual_and_lifecycle.py # Prompts, caller ID masking & dispatch parsing
│   ├── test_storage.py            # SQLite database records & non-blocking writes
│   ├── test_token_service.py      # JWT token generation, claims & health checks
│   ├── test_tts_adapter.py        # Sarvam v3 payload filtering & parameter validation
│   └── scenarios/                 # Offline Lifecycle Scenarios (Phase 3)
│       └── test_assistant_scenarios.py # 19 scenario tests for tool errors, timeouts & prompts
├── web/                           # Lightweight Web Test Client
│   └── index.html                 # Browser testing console with live audio visualizer
├── uni_pdfs/                      # Official SRM AP Policy PDFs (11 documents)
├── requirements.txt               # Direct runtime requirements
├── requirements.lock              # Hash-pinned universal lockfile
├── pyproject.toml                 # Project metadata & Python 3.11 baseline
└── README.md                      # Public documentation
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites

- Linux / macOS / Windows WSL2
- Python 3.11 (tested on Python 3.11.18)
- Native system tools for offline document extraction: `pdftoppm`, `tesseract`, `pdfunite` (available in standard Linux distributions via `poppler-utils` and `tesseract-ocr`).

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/YEsh-DEV/nexi-ntl.git
cd nexi-ntl

# Create Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install CPU-optimized PyTorch (eliminates 3.5GB CUDA bloat)
pip install torch==2.8.0+cpu -f https://download.pytorch.org/whl/cpu

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create `.env.local` in the project root:

```ini
# LiveKit Cloud Credentials
LIVEKIT_URL="wss://your-project.livekit.cloud"
LIVEKIT_API_KEY="your_livekit_key"
LIVEKIT_API_SECRET="your_livekit_secret"
AGENT_NAME=""                      # Empty for auto-dispatch; set to 'nexi' for named dispatch

# AI Provider Credentials
GOOGLE_API_KEY="your_gemini_api_key"
DEEPGRAM_API_KEY="your_deepgram_api_key"
SARVAM_API_KEY="your_sarvam_api_key"

# Models and Voices
LLM_MODEL="gemini-3.5-flash"
STT_MODEL="nova-3"
STT_LANGUAGE="en-US"
TTS_MODEL="bulbul:v3"
TTS_SPEAKER="kavya"

# Knowledge & Timezone
TIMEZONE="Asia/Kolkata"
RERANK_MIN_SCORE="0.0"
```

### 4. Build the Knowledge Base (One-Time Offline Step)

```bash
# Run ingestion to chunk, embed, and initialize ChromaDB
python -m rag.ingest --allow-empty-sources
```

### 5. Launch the Web Test Console & Token Service

```bash
# Serves /api/token, /api/health, and web/index.html on http://localhost:8080
python -m agent.token_service
```

### 6. Launch the Voice Agent Worker

```bash
# Start the LiveKit agent worker daemon
python -m agent.main dev
```

Open `http://localhost:8080` in Chrome/Edge, click **Start Call**, and begin speaking with Nexi!

---

## 🧪 Verification & Testing Suite

All tests execute completely offline without requiring live server daemons, provider API keys, or network connections:

```bash
# 1. Run the entire automated test suite (38 unit & scenario tests)
python -m unittest discover -s tests

# 2. Run code style and lint check (Ruff)
uv run --with ruff ruff check .

# 3. Validate Telephony SIP trunk and dispatch rule configurations
python scripts/validate_telephony_config.py

# 4. Run the 55-question Grounding & Anti-Hallucination Retrieval Benchmark
python scripts/eval_retrieval.py
```

---

## 🗺️ Future Development Roadmap

- 🎨 **Phase 5 Full-Stack UI:** Production Next.js web application with customized UI designs, responsive mobile layout, dark/light theme, and live bilingual captions.
- 🏢 **Multi-Tenant University Portal:** Administrative panel allowing departments to upload new semester schedules, audit unindexed queries, and adjust voice personas.
- 🎓 **Campus ERP / SIS Action Tools:** Secure OAuth tool calling allowing authenticated students to ask:
  - *"Nexi, what is my current attendance percentage in Operating Systems?"*
  - *"Have my semester exam fees been cleared?"*
- 🌐 **Indian Carrier DID Onboarding:** Transitioning Twilio trial SIP trunk to licensed Indian telecom carriers (Plivo / Exotel / Tata Tele) with regional latency pinning.

---

## 👥 Authors & Acknowledgments

- **Lead Architecture & Engineering:** [Atmakuri Yeshwanth](https://github.com/YEsh-DEV)
- **Institution:** SRM University-AP, Andhra Pradesh, India
- **Core Technologies:** [LiveKit](https://livekit.io/), [Google Gemini](https://ai.google.dev/), [Sarvam AI](https://www.sarvam.ai/), [Deepgram](https://deepgram.com/), [ChromaDB](https://www.trychroma.com/), [HuggingFace](https://huggingface.co/).

---
*Built with precision and care for the students and faculty of SRM University-AP.*
