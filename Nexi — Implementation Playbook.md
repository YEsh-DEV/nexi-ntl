# Nexi — Implementation Playbook

*Step-by-step plan for SRM AP University's voice assistant: fix the core agent, add Twilio telephony, add Sarvam multilingual support, build the frontend, then grow the product.*

## 0. H**ow to use this document**

- Work through the phases in order. Each phase has a goal, concrete tasks and a **Done when** checklist. Do not start a phase until the previous one passes its checklist.
- Phases 1 and 2 are the foundation. Telephony and multilingual support on top of a buggy agent only multiply the bugs.
- **Version warning.** The LiveKit Agents API changes between minor versions (for example `RoomInputOptions` vs newer room options, or `WorkerOptions` vs `AgentServer`). The snippets here target the 1.x `AgentSession` style. Run `pip show livekit-agents`, pin the version, and check each snippet against the docs for your version before pasting.
- Snippets use single quotes inside Python so they are easy to copy. They are skeletons showing structure and the key calls, not final production code. We refine them file by file together.

## 1. Target architecture

```
 Web student ──► Next.js frontend ──► /api/token ──┐
                                                    ▼
 Phone student ──► Twilio number ──► Elastic SIP ──► LiveKit Cloud SIP ──► Room (one per call)
                                                    │
                                                    ▼
                                      Nexi agent worker (Python)
                          Silero VAD · STT · Gemini LLM · Sarvam TTS
                                                    │ function tool
                                                    ▼
                                      Retriever (Chroma + BM25 + reranker)
                                                    ▲
                                 ingest.py (offline) ◄── PDFs / JSON
```

**Design principles**

1. **One agent, two channels.** The channel (`web` or `phone`) is passed as job metadata. Only the prompt, TTS sample rate and noise cancellation differ.
2. **Heavy models load once per worker process** in `prewarm`, never inside a call.
3. **Ingestion is offline.** Building the vector database is a separate script, never part of the call path.
4. **The framework owns the conversation loop.** We do not re-implement STT → LLM → TTS by hand.
5. **Everything configurable through environment variables.** No hardcoded paths or keys.

**Target repository layout**

```
nexi/
├─ agent/
│  ├─ main.py          # prewarm + entrypoint only
│  ├─ assistant.py     # Agent class and its tools
│  ├─ prompts.py       # prompt builder per channel and language
│  └─ config.py        # all settings from env
├─ rag/
│  ├─ ingest.py        # build or rebuild the database (run manually)
│  ├─ chunking.py      # PDF and JSON chunkers
│  └─ retriever.py     # hybrid search + reranker + threshold
├─ scripts/
│  ├─ setup_sip.py     # creates SIP trunk and dispatch rule once
│  └─ eval.py          # runs the question set
├─ eval/questions.yaml
├─ frontend/           # Next.js app (Phase 5)
├─ data/pdfs, data/json
├─ .env.example
├─ requirements.txt    # pinned versions
└─ README.md
```

**Technology decisions**

| Component | Choice | Reason |
| --- | --- | --- |
| Realtime layer | LiveKit Cloud | Handles WebRTC and SIP for you, so no SIP server to run |
| STT | Deepgram (English), Sarvam Saarika (Indian languages, Phase 4) | Best fit per language |
| LLM | Gemini 2.5 Flash | Fast and cheap, with good tool calling |
| TTS | Sarvam Bulbul | Natural Indian voices, 8 kHz output for phone |
| Vector store | Chroma now, pgvector later | Chroma is fine until the Spring Boot phase |
| Telephony (test) | Twilio trial | Free to test the whole flow |
| Frontend | Next.js + LiveKit React components | Fastest route to a polished UI |

## 2. Phase 0 — Preparation (half a day)

**Goal:** a clean, reproducible starting point.

**Tasks**

1. `git init` and commit your current code as the baseline. Create a branch `refactor-core`.
2. Run `pip freeze > requirements.lock` and keep it. Then create a clean `requirements.txt` listing only what you import, with pinned versions (`livekit-agents[...]==x.y.z`).
3. Create accounts and keys: LiveKit Cloud (project URL, API key, secret), Deepgram, Google AI Studio, Sarvam, and Twilio (trial; used in Phase 3).
4. Install the LiveKit CLI (`lk`) and run `lk cloud auth` to link your project.
5. Create `.env.example` (committed) and `.env.local` (git-ignored) using the variables below.
6. Copy your PDFs and JSON files into `data/pdfs` and `data/json` so nothing depends on `E:/` paths.

| Variable | Purpose |
| --- | --- |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit connection |
| `DEEPGRAM_API_KEY` | STT |
| `GOOGLE_API_KEY` | Gemini |
| `SARVAM_API_KEY` | TTS and STT. **Uppercase**: the plugin reads this exact name, and lowercase breaks on Linux |
| `AGENT_NAME` | Leave **empty** in Phases 1 to 2 (automatic dispatch). Set to `nexi` in Phase 3 |
| `PDF_DIR`, `JSON_DIR`, `CHROMA_DIR` | Data locations |
| `RERANK_MIN_SCORE` | Relevance threshold (tuned in Phase 2) |

**Done when:** the repo builds from a clean virtualenv, secrets are not committed, and `lk` can list your project.

## 3. Phase 1 — Fix the core voice agent (2 to 3 days)

**Goal:** a small, stable agent that answers by text-grounded retrieval, with the framework doing the conversation work.

### 3.1 Bug register (everything to fix)

| # | Where | Problem | Fix |
| --- | --- | --- | --- |
| 1 | clean.py | Two answer paths: `AgentSession` answers on its own, and `process_query` calls `generate_reply` again. Replies can be duplicated | Delete `process_query` and the transcription handler entirely |
| 2 | clean.py | `session.on('transcription')` is not a real `AgentSession` event, so the handler is dead or erroring | Not needed after fix 1. If you ever need transcripts, use `user_input_transcribed` and check `is_final` |
| 3 | clean.py | `generate_reply()` returns a `SpeechHandle`, not a string, so the `isinstance(response, str)` check is always false | Removed with fix 1 |
| 4 | clean.py | Awaiting the handle waits for speech playout, so the 10 s and 6 s timeouts measured speaking time, not generation time | Removed with fix 1. Measure latency with metrics (fix 15) |
| 5 | clean.py | Fallback strings were returned but never spoken | Use `session.say()` inside an `error` event handler |
| 6 | clean.py | Log says 4 s while the code uses 6 s | Removed with fix 1 |
| 7 | clean.py | Embedding model, reranker, Chroma and Silero load inside every job | Load once in `prewarm` and keep in `proc.userdata` |
| 8 | rag\_engine.py | Database building runs inside the call path, and two jobs can rebuild at once | Move to `rag/ingest.py`, run manually |
| 9 | rag\_engine.py | `Chroma.from_documents` on an existing folder appends, so rebuilds duplicate chunks | Delete the collection before re-ingesting |
| 10 | rag\_engine.py | Reranker always returns `top_n` documents, so the 'no results' path never fires | Add a score threshold and return `NO_RESULTS` when nothing passes |
| 11 | tools.py | `ToolError` raised inside `try` is caught by `except Exception` and replaced with a generic message | Add `except ToolError: raise` first, or restructure |
| 12 | clean.py | `WorkerOptions` signature-filter hack silently drops `reconnect_attempts`, `reconnect_interval` and `room_join_timeout`, which are not valid options | Remove the hack, pass only valid options, pin the version |
| 13 | clean.py | `auto_subscribe` is passed to `RoomInputOptions` but belongs to `ctx.connect()`, and is the default anyway | Remove |
| 14 | clean.py | `start_thinking()` and `stop_thinking()` are called on `BackgroundAudioPlayer`, but the player plays the thinking sound automatically when started with the session | Remove the calls. Optionally keep `BackgroundAudioPlayer(thinking_sound=...)` and `start(room, agent_session)` |
| 15 | clean.py | No latency visibility | Subscribe to `metrics_collected` and log per-stage latency |
| 16 | clean.py | Cleanup looks for `end`, `stop` or `close` methods, but the method is `aclose()`. Also a `while True` health loop keeps the job alive | Let the entrypoint return. Use `ctx.add_shutdown_callback()` for cleanup |
| 17 | clean.py | `os.getenv('sarvam_api_key')` is lowercase | Let the plugin read `SARVAM_API_KEY` |
| 18 | livekit\_session\_manager.py | Never imported. `AgentSession` already keeps chat history | Delete the file |
| 19 | tools.py | `extract_key_entities`, `enhance_query` and `SearchMetrics` are never called | Delete (or wire `enhance_query` in deliberately later) |
| 20 | clean.py | Unused imports (`JobRequest`, `Optional`), duplicate `import inspect` | Delete |
| 21 | clean.py / prompt | Greeting text differs between the prompt and the code, and the greeting needs an LLM call and a timeout | Use `session.say('...')` for a fixed greeting |
| 22 | tools.py | `classify_query` uses substring matching, so 'dr' matches 'address' and 'how' matches 'show'. It only tweaks `k` and `top_n` | Delete it. A reranker plus a fixed `k` is simpler and as good |
| 23 | tools.py | Tool returns `search_params`, `query_type` and `sources`. They cost tokens and invite the model to mention sources aloud | Return only the trimmed context string |
| 24 | prompt | No current date, so 'when is the next holiday?' cannot be answered | Inject today's date when the session starts |
| 25 | clean.py | `interim_results=False` on Deepgram hurts turn handling responsiveness | Set `interim_results=True` |
| 26 | clean.py | No turn-detection model despite the comment saying so | Add LiveKit's turn detector (see main.py below) |
| 27 | rag\_engine.py | `get_relevant_documents` is deprecated | Use `invoke` |
| 28 | rag\_engine.py | Hardcoded `E:/` paths | Read from env through `config.py` |
| 29 | rag\_engine.py | JSON handling loses data (details in Phase 2) | Redo chunking in Phase 2 |

### 3.2 Delete first

Before writing anything new, remove: `process_query`, `_processing_query`, `_active_sessions`, `get_health_stats`, the transcription handler, the manual retry loop around `ctx.connect()`, the signature-introspection code, the health-monitor loop and `livekit_session_manager.py`. The old `clean.py` shrinks from about 450 lines to about 80.

### 3.3 New files

**`agent/config.py`**

```python
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env.local')


@dataclass(frozen=True)
class Settings:
    agent_name: str = os.getenv('AGENT_NAME', '')  # empty = automatic dispatch
    pdf_dir: Path = Path(os.getenv('PDF_DIR', './data/pdfs'))
    json_dir: Path = Path(os.getenv('JSON_DIR', './data/json'))
    chroma_dir: Path = Path(os.getenv('CHROMA_DIR', './data/chroma'))
    embed_model: str = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    rerank_model: str = os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    rerank_min_score: float = float(os.getenv('RERANK_MIN_SCORE', '0.0'))
    search_timeout: float = float(os.getenv('SEARCH_TIMEOUT', '6'))


settings = Settings()
```

**`agent/prompts.py`**

```python
from datetime import date

BASE = '''You are Nexi, the voice assistant of SRM AP University.

HOW YOU WORK
- For any question about the university, call search_university_info first.
- Answer ONLY from what the tool returns. Never invent facts.
- If the tool returns NO_RESULTS, say you do not have that information and suggest contacting the university office.
- Speak only about SRM AP University, not other SRM campuses.

STYLE
- Warm and friendly, like a helpful senior student.
- Plain spoken sentences only. No lists, markdown, symbols or links.
- Say numbers, dates and times the way a person speaks them.
- Never mention tools, searching or context.

Today's date is {today}.
'''

WEB_EXTRA = '- Keep answers under 80 words unless the student asks for detail.'
PHONE_EXTRA = (
    '- This is a phone call. Keep answers under 50 words.\n'
    '- Spell out email addresses and phone numbers slowly, in small groups.\n'
    '- Offer to repeat or give more detail instead of reading long lists.'
)


def build_prompt(channel: str) -> str:
    extra = PHONE_EXTRA if channel == 'phone' else WEB_EXTRA
    return BASE.format(today=date.today().strftime('%A, %d %B %Y')) + '\n' + extra
```

**`agent/assistant.py`**

```python
import asyncio
import logging

from livekit.agents import Agent, RunContext, ToolError, function_tool

from config import settings
from prompts import build_prompt

log = logging.getLogger('nexi.agent')


class NexiAgent(Agent):
    def __init__(self, retriever, channel: str = 'web') -> None:
        super().__init__(instructions=build_prompt(channel))
        self._retriever = retriever

    @function_tool
    async def search_university_info(self, context: RunContext, query: str) -> str:
        '''Look up facts about SRM AP University: fees, hostels, calendar, faculty, rules, admissions.

        Args:
            query: the student's question, rewritten as a short standalone search phrase.
        '''
        if not query or not query.strip():
            raise ToolError('Ask the student to repeat the question.')
        try:
            text = await asyncio.wait_for(
                self._retriever.search(query.strip()), timeout=settings.search_timeout
            )
        except asyncio.TimeoutError:
            log.warning('search timeout: %s', query)
            raise ToolError('Search is slow right now. Ask the student to try again.')
        except Exception:
            log.exception('search failed: %s', query)
            raise ToolError('Search failed. Apologise briefly and suggest contacting the office.')
        return text or 'NO_RESULTS'
```

The retriever contract is simple: `async def search(query) -> str | None`. In Phase 1 wrap your existing `search_unified_knowledge` in it. In Phase 2 it is replaced by the new retriever.

**`agent/main.py`**

```python
import asyncio
import json
import logging

from livekit import rtc
from livekit.agents import (AgentSession, JobContext, JobProcess, MetricsCollectedEvent,
                            RoomInputOptions, WorkerOptions, cli, metrics)
from livekit.plugins import deepgram, google, noise_cancellation, sarvam, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from assistant import NexiAgent
from config import settings
from rag.retriever import build_retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
for noisy in ('urllib3', 'httpx', 'httpcore'):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger('nexi')

GREETING = {'web': 'Hi! I am Nexi, your SRM AP assistant. How can I help you today?',
            'phone': 'Hello, this is Nexi, the SRM AP University assistant. How can I help you?'}


def prewarm(proc: JobProcess):
    # Runs once per worker process, before any call arrives.
    proc.userdata['vad'] = silero.VAD.load()
    proc.userdata['retriever'] = build_retriever()


def pick_noise_cancellation(params):
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return noise_cancellation.BVCTelephony()
    return noise_cancellation.BVC()


def read_channel(ctx: JobContext) -> str:
    try:
        return json.loads(ctx.job.metadata or '{}').get('channel', 'web')
    except json.JSONDecodeError:
        log.warning('bad job metadata: %r', ctx.job.metadata)
        return 'web'


async def entrypoint(ctx: JobContext):
    channel = read_channel(ctx)
    is_phone = channel == 'phone'

    session = AgentSession(
        vad=ctx.proc.userdata['vad'],
        stt=deepgram.STT(model='nova-3', language='en-IN', interim_results=True),
        llm=google.LLM(model='gemini-2.5-flash', temperature=0.2),
        tts=sarvam.TTS(target_language_code='en-IN', model='bulbul:v2', speaker='anushka',
                       speech_sample_rate=8000 if is_phone else 24000),
        turn_detection=MultilingualModel(),
    )

    usage = metrics.UsageCollector()

    @session.on('metrics_collected')
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    @session.on('error')
    def _on_error(ev):
        log.error('session error: %s', ev.error)
        if not getattr(ev.error, 'recoverable', True):
            asyncio.create_task(session.say(
                'Sorry, I am having technical trouble. Please contact the university office.'))

    async def _log_usage():
        log.info('usage: %s', usage.get_summary())

    ctx.add_shutdown_callback(_log_usage)

    await session.start(
        room=ctx.room,
        agent=NexiAgent(ctx.proc.userdata['retriever'], channel),
        room_input_options=RoomInputOptions(noise_cancellation=pick_noise_cancellation),
    )
    await ctx.connect()
    await session.say(GREETING[channel])
    # No while-loop. The job lives until the room closes.


if __name__ == '__main__':
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name=settings.agent_name,
    ))
```

**Things to verify against your installed version** (I could not confirm these for your exact release):

- Whether `session.start` expects `room_input_options` or a newer room-options argument.
- Whether the `deepgram.STT` model name `nova-3` supports `en-IN` in your plugin version. Fall back to `nova-2` if not.
- The Gemini plugin's option for reducing the thinking budget. Lower thinking means lower latency for voice.
- Run `python main.py download-files` once to fetch the turn-detector and noise models.
- If your version uses `AgentServer` and `@server.rtc_session()` instead of `WorkerOptions`, keep the same body and only change the wrapper.

### 3.4 Run and test

1. Terminal-only test: `python main.py console`. Speak to it. No browser or phone needed.
2. Room test: `python main.py dev`, then open the LiveKit Agents Playground and connect to your project. This works because `AGENT_NAME` is empty.
3. Test sentences: a fee question, a hostel question, a date question, an off-topic question ('who won the cricket match?'), and an interruption in the middle of an answer.
4. Read the metrics log lines. Note the end-of-utterance delay, LLM time to first token and TTS time to first byte. These are your baseline numbers.

### 3.5 Done when

- Each question produces exactly one spoken answer, with no duplicates.
- Interrupting the agent works.
- The first greeting is identical every time.
- Off-topic questions get the 'I do not have that information' answer.
- No `while True` loop, no unused files, no hardcoded paths.
- Per-stage latency numbers are logged and saved in your notes as the baseline.

## 4. Phase 2 — Fix the knowledge base (3 to 4 days)

**Goal:** the right chunk comes back for real student questions, and the agent says 'I do not know' when it should.

### 4.1 What is wrong today

- **Information loss.** A person profile keeps only name, title and department, so email, phone, bio and office are dropped. Events lose end dates and descriptions.
- **Nested data is flattened.** `_is_person_profile` matches any dict with `name` plus `department`. A department object that holds a `faculty` list becomes one short line, and its children are never processed.
- **Fragments without context.** Every scalar becomes its own tiny chunk, such as 'Fee: 50000', with no idea which course or year it belongs to.
- **Dense-only search.** MiniLM is weak on names and exact terms such as 'Dr. Rao' or a fee code.
- **No 'I do not know'.** The reranker always returns something.
- **Rebuild duplicates data**, and rebuilding runs inside the call path.

### 4.2 New chunking rules

1. **One chunk per record**, not per value. A faculty member, an event or an FAQ item is one chunk containing all of its scalar fields.
2. **Prefix every chunk with its path**, for example `faculty.json | Faculty > CSE > Dr Rao: designation: Professor; email: ...`. The path is what lets a search for 'CSE professors' find the chunk.
3. **Never flatten children into the parent.** Scalars of a dict go in its own chunk. Nested dicts and lists are walked separately with a longer path.
4. **Keep chunks between 200 and 1000 characters.** Split larger records, and merge tiny sibling records.
5. **PDFs:** keep a splitter at 600 to 900 characters with 100 overlap, but store the page number and the file name. Add a short heading to each chunk when the PDF has headings.
6. **Metadata on every chunk:** `source`, `source_type`, `path` and `language`. The `language` field matters in Phase 4.

**`rag/chunking.py` (JSON part)**

```python
SCALARS = (str, int, float, bool)


def nice(key: str) -> str:
    return key.replace('_', ' ')


def json_to_chunks(data, source: str) -> list[str]:
    chunks: list[str] = []

    def walk(node, path: tuple):
        label = ' > '.join(path)
        if isinstance(node, dict):
            facts = {k: v for k, v in node.items() if isinstance(v, SCALARS) and str(v).strip()}
            if facts:
                body = '; '.join(nice(k) + ': ' + str(v) for k, v in facts.items())
                chunks.append(source + ' | ' + label + ': ' + body)
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, path + (nice(k),))
        elif isinstance(node, list):
            plain = [str(x) for x in node if isinstance(x, SCALARS)]
            if plain:
                chunks.append(source + ' | ' + label + ': ' + ', '.join(plain))
            for i, item in enumerate(node):
                if isinstance(item, (dict, list)):
                    name = ''
                    if isinstance(item, dict):
                        name = str(item.get('name') or item.get('title') or item.get('event') or '')
                    walk(item, path + (name or 'item ' + str(i + 1),))

    walk(data, (nice(source),))
    return [c for c in chunks if 40 <= len(c)]
```

Add a size guard that splits any chunk above 1000 characters at a sentence or semicolon boundary.

### 4.3 Offline ingestion (`rag/ingest.py`)

**Behaviour**

- Command: `python -m rag.ingest --rebuild`. Stop all workers first (Windows locks the Chroma files).
- With `--rebuild` it deletes the whole Chroma folder first, which fixes the duplicate-chunk bug.
- Loads PDFs from `PDF_DIR` and JSON from `JSON_DIR`, chunks them, embeds, and writes to the `nexi` collection.
- Writes `manifest.json` next to the database: chunk counts per source, embedding model name and timestamp. The agent logs this manifest at startup so you always know which data it is running on.
- Prints a summary and warns about files that produced zero chunks.

```python
import argparse, json, logging, shutil, time
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from rag.chunking import load_pdf_chunks, load_json_chunks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('ingest')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()
    if args.rebuild and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    docs: list[Document] = load_pdf_chunks(settings.pdf_dir) + load_json_chunks(settings.json_dir)
    if not docs:
        raise SystemExit('No documents found. Check PDF_DIR and JSON_DIR.')
    emb = HuggingFaceEmbeddings(model_name=settings.embed_model,
                                encode_kwargs={'normalize_embeddings': True})
    Chroma.from_documents(docs, emb, collection_name='nexi',
                          persist_directory=str(settings.chroma_dir))
    manifest = {'chunks': len(docs), 'embed_model': settings.embed_model,
                'built_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    (settings.chroma_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    log.info('ingested %s chunks', len(docs))


if __name__ == '__main__':
    main()
```

### 4.4 Hybrid retriever with a relevance threshold (`rag/retriever.py`)

**Pipeline:** vector search (top 12) and BM25 keyword search (top 12), merged with reciprocal rank fusion, reranked by the cross-encoder, then the best 3 to 4 chunks that pass a minimum score are returned. If none pass, return `None` and the tool answers `NO_RESULTS`.

```python
import asyncio
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from config import settings


def rrf(result_lists, k: int = 60):
    scores, by_text = {}, {}
    for lst in result_lists:
        for rank, doc in enumerate(lst):
            by_text[doc.page_content] = doc
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (k + rank + 1)
    return [by_text[t] for t in sorted(scores, key=scores.get, reverse=True)]


class Retriever:
    def __init__(self):
        emb = HuggingFaceEmbeddings(model_name=settings.embed_model,
                                    encode_kwargs={'normalize_embeddings': True})
        self.store = Chroma(collection_name='nexi', embedding_function=emb,
                            persist_directory=str(settings.chroma_dir))
        raw = self.store.get(include=['documents', 'metadatas'])
        docs = [Document(page_content=t, metadata=m or {})
                for t, m in zip(raw['documents'], raw['metadatas'])]
        self.bm25 = BM25Retriever.from_documents(docs, k=12)
        self.reranker = CrossEncoder(settings.rerank_model)

    def _search_sync(self, query: str):
        vec = self.store.similarity_search(query, k=12)
        kw = self.bm25.invoke(query)
        merged = rrf([vec, kw])[:16]
        if not merged:
            return None
        scores = self.reranker.predict([(query, d.page_content) for d in merged])
        ranked = sorted(zip(merged, scores), key=lambda p: p[1], reverse=True)[:4]
        good = [d for d, s in ranked if s >= settings.rerank_min_score]
        if not good:
            return None
        return '\n\n'.join(d.page_content for d in good)[:1800]

    async def search(self, query: str):
        return await asyncio.to_thread(self._search_sync, query)


def build_retriever() -> Retriever:
    return Retriever()
```

The voice answer never needs source file names, so they are not returned. Keep the context under about 1800 characters: more text only slows the LLM and tempts it to read too much aloud.

**Tuning the threshold.** The cross-encoder scores are raw numbers, not percentages. Run your eval set (4.5), print the top score for answerable questions and for unanswerable ones, and set `RERANK_MIN_SCORE` between the two groups. Do not guess the number.

### 4.5 Evaluation set

Build `eval/questions.yaml` with **40 to 60 real student questions**: 30 answerable (fees, hostel, mess timings, exam dates, faculty, admission steps, library, transport), 10 unanswerable or off-topic, and 5 tricky ones (names with different spellings, abbreviations such as CSE, questions with two parts).

```yaml
- q: 'What is the hostel fee for first year?'
  must_contain: ['hostel', 'fee']
  answerable: true
- q: 'Who won the cricket match yesterday?'
  answerable: false
```

`scripts/eval.py` runs each question through the retriever (no audio) and reports: **retrieval hit rate** (the expected keywords appear in the returned context), **refusal accuracy** on unanswerable questions, and **latency p50 and p95**. Targets for this phase: hit rate at least 85 percent, refusal accuracy 100 percent on the off-topic set, retrieval p95 under 800 ms on your machine.

Save the results in a table in the README. You will rerun the same script after every later change (multilingual, new embeddings), which proves that nothing got worse.

### 4.6 Done when

- Re-running ingestion never creates duplicates, and the manifest shows the right counts.
- JSON people, events and FAQ items keep all their fields and have path context.
- The eval hit rate and refusal targets are met and recorded.
- The agent starts without building anything.

## 5. Phase 3 — Telephony with Twilio (2 to 3 days)

**Goal:** a student dials a phone number and talks to Nexi, with the same brain and the same answers as the web version.

### 5.1 How one call flows

1. A caller dials the Twilio number.
2. Twilio's Elastic SIP trunk forwards the call to your LiveKit Cloud SIP address.
3. LiveKit checks the inbound trunk (is this number allowed?) and finds the dispatch rule.
4. LiveKit creates a room named `call-...`, adds the caller as a SIP participant, and dispatches the agent named `nexi` with metadata `channel: phone`.
5. Your worker picks up the job, builds the phone version of the session (8 kHz voice, telephony noise cancellation, short answers) and greets the caller.

The three things you create **once** are the Twilio trunk, the LiveKit inbound trunk and the dispatch rule. Do not create rules per call.

### 5.2 What the Twilio free trial allows (read before you start)

Based on Twilio's current documentation. Trial rules change, so confirm them on Twilio's trial and SIP-trunk limits pages.

- You get a free trial balance and an auto-assigned phone number, and a trunk you can configure for real test calls.
- **Trial accounts can only place or receive calls with verified numbers.** Inbound calls to a trial number must come from a number you have verified in that account (verify your own mobile first, in the Twilio console).
- A trial project is limited to **one Elastic SIP trunk**. Trial numbers are limited too, and you cannot buy extra numbers on trial subaccounts.
- Trial units expire after 30 days or when used up. Upgrading removes the restrictions.

**What this means for you in Ind****ia**

- The trial number is most likely a US number. Calling it from your Indian mobile is an international call, so check your carrier's charges, or use an ISD pack for the test.
- Only phones you verified can call it, which is fine for testing and demos but not for real students. It is a **proof of concept**, which is exactly what we need now.
- For the real deployment you move to an Indian carrier (section 5.7). The agent code does not change at all.
- If the trial limits block you, a small paid top-up on the same account lifts them. That is cheaper than losing days.

### 5.3 Step-by-step setup

**Step 1. Switch to named dispatch.** In `.env.local` set `AGENT_NAME=nexi` and restart the worker. From now on the worker only joins rooms that explicitly ask for `nexi`. The web frontend must therefore request the agent too (Phase 5 does this in the token). Until then, test the web version by turning `AGENT_NAME` empty again.

**Step 2. Twilio account.** Sign up, verify your email and your mobile number, and note the trial number.

**Step 3. Twilio Elastic SIP trunk.** In the Twilio console create an Elastic SIP trunk. Under **Origination**, add a URI `sip:YOUR-LIVEKIT-SIP-HOST` with priority 10 and weight 10. Under **Numbers**, attach your Twilio number to the trunk.

Important: copy the SIP host exactly as shown in your LiveKit Cloud project settings. A community report shows that using a host different from the project's current one lets the call reach LiveKit's edge but never match your trunk, so no room is created.

**Step 4. LiveKit inbound trunk.** Create `inbound-trunk.json`:

```json
{
  "trunk": {
    "name": "twilio-inbound",
    "numbers": ["+1XXXXXXXXXX"]
  }
}
```

Then run `lk sip inbound create inbound-trunk.json`. Use the Twilio number in full E.164 format. A missing `+` or country code is a common reason the trunk does not match. Note the trunk ID in the response.

**Step 5. Dispatch rule.** Create `dispatch-rule.json`:

```json
{
  "dispatch_rule": {
    "rule": { "dispatch_rule_individual": { "room_prefix": "call-" } },
    "name": "nexi-phone",
    "room_config": {
      "agents": [{
        "agent_name": "nexi",
        "metadata": "{\"channel\": \"phone\"}"
      }]
    }
  }
}
```

Run `lk sip dispatch create dispatch-rule.json`. The individual rule gives each caller a private room. Because `trunk_ids` is omitted the rule matches all inbound trunks. Add your trunk ID there once you have more than one.

**Step 6. Place the call.** Start the worker (`python main.py dev`), then call the Twilio number from your **verified** mobile. Watch the LiveKit Cloud dashboard: Telephony shows the call, Sessions shows the room, and your worker log shows the job.

**Step 7. Troubleshooting**

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Call fails immediately, nothing in LiveKit | Origination URI wrong, number not attached to the trunk, or caller not verified (trial) | Recheck steps 2 and 3, and verify your mobile |
| Reaches LiveKit but no room | Trunk number mismatch or SIP host mismatch | Use exact E.164 and the exact project SIP host |
| Room exists, agent never joins | `agent_name` in the dispatch rule and `AGENT_NAME` differ, or the worker is not running | Make both `nexi` and check the worker registered |
| Agent speaks but caller hears garbled or fast audio | Sample rate mismatch | Make sure the phone session uses 8 kHz TTS |
| Poor recognition | Web noise cancellation used for a phone line | Confirm `BVCTelephony` is chosen for SIP participants |
| A Twilio trial announcement plays first | Trial behaviour | Expected on trial, goes away after upgrade |

### 5.4 Code changes for the phone channel

Most of it already exists from Phase 1: the channel metadata, the 8 kHz TTS, the telephony noise cancellation and the short phone prompt. Add the following.

1. **Caller identity in logs.** When the channel is `phone`, read the caller number from the SIP participant attributes (`sip.phoneNumber`) after `ctx.wait_for_participant()`. Log only a masked or hashed version, never the full number, and do not wait for a participant in `console` mode.
2. **Spoken digits.** The prompt already tells the model to read numbers slowly in groups. Test it with an email and a phone number answer.
3. **Human handoff.** A tool such as `transfer_to_office` that transfers the call to the admissions or accounts number using the SIP transfer API (`ctx.api.sip.transfer_sip_participant(...)` with a `tel:` destination). Transfer needs a trunk that supports SIP REFER: LiveKit's docs say Twilio's programmable-voice method does not, so stay on **Elastic SIP Trunking**, which this plan uses. Verify the exact request fields in your SDK version.
4. **Silence handling.** If the caller says nothing for about 8 seconds, ask 'Are you still there?', and after a second silence say goodbye and end the call.
5. **Goodbye.** When the caller says goodbye or thanks, say a short farewell and end the session instead of waiting.

### 5.5 Phone conversation rules

- Greeting under 10 seconds. State that it is an AI assistant.
- Answers under about 50 words. Offer more detail instead of long lists.
- Confirm unclear numbers by repeating them back.
- Never read out personal data. At this stage Nexi only knows public information anyway.

### 5.6 Show the result (portfolio evidence)

- Screen-record a call with the LiveKit dashboard open next to it.
- Screenshot the telephony log, room and the per-stage latency lines.
- Put the call flow diagram and a latency table (web vs phone) in the README.
- Note the limits honestly: trial number, verified callers only.

### 5.7 Moving from the trial to a real number

When the university wants real students to call:

1. **Use an in-country carrier.** LiveKit works with any standards-compliant SIP trunk, but India needs a licensed Indian carrier. Candidates with LiveKit guides or integrations include Plivo, Twilio, Exotel and Edesy. Expect KYC: business registration, proof of address and a description of the use case.
2. **Enable region pinning in LiveKit** for calls to or from Indian numbers. The Plivo and LiveKit docs say calls fail without it.
3. **Easiest route for a university:** keep the existing helpline number and forward it to the new Indian DID, or ask the telecom team for a SIP trunk from the existing exchange. Students keep dialing the number they know.
4. **Start the KYC paperwork early.** It takes days and is the only part of this project that can stall you.
5. **Compliance:** say that the call is AI-assisted, say it if the call is recorded, and keep outbound calling for later because of DND and consent rules.

### 5.8 Done when

- A verified phone can call the Twilio number and Nexi answers within a few seconds.
- Phone audio is clear, answers are short, and numbers are spoken naturally.
- The same question gives the same answer on web and phone.
- Hanging up ends the job cleanly (no orphan sessions in the worker log).
- The demo evidence from 5.6 is saved.

## 6. Phase 4 — Multilingual support with Sarvam (3 to 4 days)

**Goal:** a student can talk to Nexi in English, Telugu or Hindi (and mixed speech), on both web and phone, and gets the answer in the same language.

### 6.1 What Sarvam gives you (verify against the current docs when you start)

- **STT:** `saarika:v2.5` for transcription, and `saaras:v3`, a multi-mode model that also offers translation and code-mix output. Older Saarika versions are deprecated. Language can be passed as a BCP-47 code (for example `te-IN`, `hi-IN`, `en-IN`), and the API can auto-detect when the language is left unknown. Check whether your LiveKit plugin version exposes that option.
- **TTS:** `bulbul:v2` and `bulbul:v3`. **Speaker names are not interchangeable** between versions, so pick the model first and then the speaker from that model's catalog. Supported output rates include 8000, 16000, 22050 and 24000 Hz.
- **Phone audio:** if you use 8 kHz, the sample rate must match everywhere (the connection and every audio chunk), otherwise the output is garbled.
- **Plugin behaviour to measure:** some versions of the LiveKit Sarvam STT plugin send audio as a batch request after the speaker finishes, not as a stream. That adds latency compared with Deepgram. Measure it with the metrics log and compare before choosing.

### 6.2 Language strategy (build in two steps)

**Step 4a: explicit language choice (reliable, build first)**

- **Web:** a language selector on the landing page (English, తెలుగు, हिन्दी). The choice goes into the token's agent metadata as `language`.
- **Phone:** a short spoken menu after the greeting: 'For English, say English or press 1. తెలుగు కోసం 2 నొక్కండి. हिन्दी के लिए 3 दबाएँ.' Listen for the DTMF digit through the room's DTMF event (verify the event name in your SDK version) or for the spoken language word.
- Default to English if nothing is chosen within a few seconds.

**Step 4b: automatic detection (build after 4a works)**

- Let STT auto-detect the language on the first utterance, then switch the voice. Treat this as experimental and test it on real accents before trusting it.
- Code-mixed speech ('hostel fee entha?') is how students really talk. Sarvam's code-mix option and Bulbul's handling of mixed text are the reason to use it. Test with at least 10 such sentences.

### 6.3 How to switch language inside the agent

There are two workable options. Pick one and keep it consistent.

- **Option A (recommended): one Agent class instance per language.** An `Agent` can carry its own `stt` and `tts`, which override the session defaults. Build `NexiAgent(language='te')` with a Telugu STT and a Telugu TTS voice. To change language, give the LLM a small tool `switch_language(language)` that returns a new `NexiAgent` for that language. The framework hands the conversation over and keeps the chat history. This is explicit and easy to test.
- **Option B: update options on the running session.** Some plugins support `update_options(...)` on STT and TTS. If your Sarvam plugin version supports it, this is less code, but the behaviour depends on the plugin, so test it before relying on it.

Keep a small table in `config.py`:

```python
LANGUAGES = {
    'en': {'name': 'English', 'code': 'en-IN', 'speaker': 'anushka'},
    'te': {'name': 'Telugu', 'code': 'te-IN', 'speaker': 'anushka'},
    'hi': {'name': 'Hindi', 'code': 'hi-IN', 'speaker': 'anushka'},
}
```

The speaker values above are placeholders. Replace them with real speaker names from the catalog of the Bulbul model you chose.

### 6.4 Retrieval for other languages (cheap first, heavy later)

- **Cheapest and best first step: query translation.** Your documents are English. Change the tool docstring so the LLM must pass `query` as a short **English** search phrase, translating from Telugu or Hindi when needed. Retrieval then stays exactly as tested in Phase 2, with no re-embedding. The LLM answers in the student's language.
- **Later, only if you add Telugu or Hindi documents:** switch to a multilingual embedding model (for example BGE-M3 or multilingual-E5) and a multilingual reranker, then re-run `ingest.py --rebuild`. Use the `language` metadata added in Phase 2 to filter if needed.

### 6.5 Prompt changes

- Add a line: 'Reply in the language the student speaks. Keep names, course codes and numbers as they are. Use simple spoken Telugu or Hindi, not formal written style.'
- Keep the 'no lists, no symbols' rule. Symbols are read badly by TTS in every language.
- Keep answers short on phone, where an error costs more than on web.
- Inject the language name into the prompt through `build_prompt(channel, language)`.

### 6.6 Testing multilingual quality

1. Record 20 short utterances per language (use friends, different accents, a phone and a laptop mic). Write down what was said and what STT returned. Count errors by hand.
2. Add 20 Telugu, 20 Hindi and 10 code-mixed questions to `eval/questions.yaml` with a `lang` field. Re-run the eval, and report hit rate per language.
3. Listen to 10 TTS answers per language. Check numbers, dates, names and abbreviations such as CSE. Note anything read wrongly and add pronunciation hints to the prompt (for example, write 'C S E').
4. Compare Deepgram and Sarvam STT latency on English. Keep whichever gives the better balance, and write it in the README.

### 6.7 Done when

- Telugu, Hindi and English work on web with the selector, and on phone with the spoken or DTMF menu.
- The answer language always matches the chosen language.
- Per-language eval numbers are recorded.
- Language switching in the middle of a call works without losing the conversation.

## 7. Phase 5 — Frontend (3 to 4 days)

**Goal:** a clean, mobile-friendly website where a student can talk to Nexi, see what it says, and also find the phone number.

### 7.1 Stack

Next.js (App Router) with TypeScript and Tailwind, plus `livekit-client` and `@livekit/components-react`. LiveKit also publishes a React starter for voice agents that you can start from. Check its current repository and copy its structure, then restyle it.

### 7.2 Screens and components

1. **Landing page:** university branding, a large **Talk to Nexi** button, a language selector, suggested question chips (fees, hostel, exam dates, faculty, admissions), and a card showing the phone number with a tap-to-call link. Add a small privacy note: 'This is an AI assistant. Voice is processed to answer your question.'
2. **Call view:** a live audio visualizer, a clear state label (Listening, Thinking, Speaking), mute and end buttons, a scrolling transcript, and a row of follow-up question chips.
3. **Error states:** microphone blocked (explain how to allow it), connection failed (retry button), agent did not join within about 10 seconds (friendly message and the phone number as a fallback), and server busy.
4. **Design:** dark and light themes, a large readable font, and a proper font for Telugu and Hindi text in the transcript (Noto Sans Telugu and Noto Sans Devanagari). Always show captions, which also helps accessibility.

### 7.3 Token server (the part people forget)

The browser must never see your LiveKit secret. A server route issues short-lived tokens and also **requests the agent**, which is required because the worker now uses a named agent.

```ts
// app/api/token/route.ts
import { AccessToken } from 'livekit-server-sdk';
import { RoomAgentDispatch, RoomConfiguration } from '@livekit/protocol';
import { randomUUID } from 'crypto';

export async function POST(req: Request) {
  const { language = 'en' } = await req.json().catch(() => ({}));
  const room = 'web-' + randomUUID();
  const identity = 'student-' + randomUUID().slice(0, 8);

  const at = new AccessToken(process.env.LIVEKIT_API_KEY!, process.env.LIVEKIT_API_SECRET!, {
    identity,
    ttl: '10m',
  });
  at.addGrant({ room, roomJoin: true, canPublish: true, canSubscribe: true });
  at.roomConfig = new RoomConfiguration({
    agents: [new RoomAgentDispatch({
      agentName: 'nexi',
      metadata: JSON.stringify({ channel: 'web', language }),
    })],
  });

  return Response.json({ token: await at.toJwt(), url: process.env.LIVEKIT_URL, room });
}
```

Add a per-IP rate limit to this route, because every call costs real money in STT, LLM and TTS minutes.

### 7.4 Call view skeleton

```tsx
'use client';
import { LiveKitRoom, RoomAudioRenderer, BarVisualizer, useVoiceAssistant } from '@livekit/components-react';
import { useState } from 'react';

function Call() {
  const { state, audioTrack } = useVoiceAssistant();
  return (
    <div className='flex flex-col items-center gap-6'>
      <BarVisualizer state={state} trackRef={audioTrack} barCount={7} />
      <p className='text-lg capitalize'>{state}</p>
    </div>
  );
}

export default function Page() {
  const [conn, setConn] = useState<{ token: string; url: string } | null>(null);
  async function start() {
    const r = await fetch('/api/token', { method: 'POST', body: JSON.stringify({ language: 'en' }) });
    setConn(await r.json());
  }
  if (!conn) return <button onClick={start}>Talk to Nexi</button>;
  return (
    <LiveKitRoom token={conn.token} serverUrl={conn.url} connect audio onDisconnected={() => setConn(null)}>
      <Call />
      <RoomAudioRenderer />
    </LiveKitRoom>
  );
}
```

The transcript panel uses the transcription hooks from the components library (check the current hook names in its docs). Start with the visualizer and state label, and add the transcript once the call works.

### 7.5 Cost and abuse protection

- Limit each web session to about 5 minutes. Enforce it in the agent: after the limit, say goodbye and call `ctx.shutdown()`.
- Rate limit the token route per IP, and add a simple captcha (for example Cloudflare Turnstile) before the call starts if you publish the site publicly.
- Keep `LIVEKIT_API_SECRET` only in server environment variables, never in client code.
- Deploy the frontend on Vercel (or any Node host) with environment variables set in the dashboard.

### 7.6 Done when

- A new visitor can start a call in two clicks and hear Nexi greet them.
- The visualizer, state label, mute, end and language selector all work, on desktop and on a phone browser.
- The transcript shows both sides correctly, including Telugu and Hindi characters.
- Error states are handled, and the phone number is visible as a fallback.

## 8. Phase 6 — Quality, observability and deployment (3 to 4 days)

**Goal:** make it measurable, testable and runnable by someone else.

### 8.1 Observability

- **Structured logs:** one JSON line per event with `call_id`, `channel`, `language` and the stage name. Never log full phone numbers or full transcripts of private information.
- **Call summary at shutdown:** duration, number of turns, tool calls, `NO_RESULTS` count, errors, and token and audio usage from the usage collector. Write it to a `call_logs` table (SQLite for now, PostgreSQL later).
- **Latency dashboard:** end-of-utterance delay, LLM time to first token, TTS time to first byte, and total response time, with p50 and p95 per channel. A small Grafana panel or even a notebook chart is enough.
- **Unanswered question log:** every time the tool returns `NO_RESULTS`, store the question. This becomes the admin's to-do list: add the missing document, re-run ingestion, ask again. It is the most valuable feature for a real deployment.

### 8.2 Tests and CI

- **Unit tests (pytest):** the JSON chunker (nested people, events, FAQs), the prompt builder (date and channel), the RRF merge and threshold logic.
- **Retrieval regression test:** run `scripts/eval.py` in CI against a small checked-in data sample, and fail the build if the hit rate drops.
- **GitHub Actions:** lint (ruff), tests, and the eval job on every pull request.

### 8.3 Packaging and deployment

- **Dockerfile for the worker:** a slim Python image, install pinned requirements, run `python main.py download-files` at build time so models are baked in, and start with `python main.py start`.
- **Data:** mount the Chroma folder as a volume, or bake a built database into a separate image layer. Run ingestion as a one-off job.
- **docker-compose:** worker plus a volume for data. The frontend is deployed separately on Vercel.
- **Hosting choices:** LiveKit Cloud's own agent hosting if available for your account, or a small VM. Each concurrent call runs in its own process with preloaded models, so measure RAM per call before choosing machine size.
- **Secrets:** environment variables or the platform's secret store. Never commit `.env.local`.

### 8.4 Capacity test

Open several concurrent sessions (browser tabs in the playground, then a scripted test if you want), and record CPU, RAM and p95 latency at 1, 5 and 10 concurrent calls on your machine. Put the numbers in the README as 'Supports N concurrent calls at p95 under X seconds on \<machine>'. One real number is worth more on a resume than a list of technologies.

### 8.5 Done when

- Every call produces a summary row, and the latency dashboard shows real data.
- CI is green on tests and the retrieval eval.
- Someone else can clone the repo, follow the README and run the agent with Docker.
- The capacity numbers are recorded.

## 9. Phase 7 — Growth roadmap (after Phase 6 is solid)

Each item below is independent. Build them one at a time, in the order of the table in 9.7. Do not start any of them until Phases 1 to 6 pass their checklists.

### 9.1 Spring Boot Campus Services API (the Java showcase)

**Why it exists:** the voice agent stays in Python, because LiveKit Agents runs on Python. Spring Boot becomes the transactional backend that owns real student data and business rules. This is the standard split in industry: a Python AI runtime in front of a Java business layer.

**What it owns**

- Student profile lookup, fee dues, attendance, timetable and exam schedule.
- Complaint tickets and admission leads.
- Call logs and analytics (replacing the SQLite table from Phase 6).
- Authentication, roles (student, staff, admin) and audit logs.

**Suggested stack:** Spring Boot 3 with Java 21, Spring Web, Spring Security (JWT), Spring Data JPA, PostgreSQL, Flyway migrations, Redis for caching and rate limits, Resilience4j (circuit breaker, retry, rate limiter), Micrometer with Prometheus, Testcontainers for integration tests, OpenAPI docs.

**Build order**

1. Project skeleton, Postgres, Flyway migration for students, tickets, leads and call logs.
2. Read endpoints with seed data: `GET /students/{id}/fees`, `/attendance`, `/timetable`.
3. Service-to-service authentication: the agent uses a signed service token. Students never receive it.
4. Write endpoints: `POST /tickets`, `POST /leads`, `POST /call-logs`.
5. Circuit breaker and timeouts on every outbound call. The agent must still answer politely if this API is down.
6. Agent tools that call the API: `get_my_fee_status`, `create_complaint`, `capture_admission_lead`.
7. Tests, OpenAPI page, and a Docker Compose service.

**Student verification (required before any personal data is spoken)**

1. Recognise the caller's number from the SIP attributes. Treat it as a hint only, because numbers can be spoofed.
2. Ask for the student ID and a PIN entered with the keypad (DTMF), or send a one-time code by SMS to the registered number.
3. The API returns a short-lived verified session. The agent holds it only for that call.
4. Personal-data tools refuse to run without a verified session. Log every access in the audit table.
5. On web, use the university's login (OAuth/OIDC) instead of a PIN.

### 9.2 MCP (agent to tools)

**Idea:** instead of hardcoding tools in the Python agent, expose each capability as an MCP server. Any agent, including a future second one, can use them without code changes.

- `knowledge-mcp`: the RAG search tool (Python).
- `student-mcp`: fees, attendance, timetable (a thin MCP layer in front of the Spring Boot API; Spring AI has MCP server support, so check its current docs).
- `ticket-mcp`: create and track complaints.
- `calendar-mcp`: academic calendar and events.

**How to build it**

1. First wrap only `knowledge-mcp`, and make the agent call it through MCP instead of in-process. Compare latency with the in-process version, and write the difference in the README.
2. Check how your installed LiveKit Agents version attaches MCP servers (the option name and transport have changed between versions).
3. Keep the in-process tool as a fallback, because a network hop adds latency to every voice turn.
4. Add authentication between the agent and each MCP server.

**Honest trade-off:** MCP adds a hop to every tool call, and voice is latency sensitive. The real benefit is reuse and clean boundaries, so measure before you commit.

### 9.3 A2A (agent to agent)

**Idea:** Nexi becomes a front-desk agent that delegates to specialist agents, each published with an agent card that describes its skills.

- Specialists: Admissions, Accounts and Fees, Hostel, Exams, Placements.
- Each specialist can be its own service with its own knowledge base and its own tools, and can even be written in another language.
- Nexi decides which specialist should answer, sends the task, and speaks the result.

**How to build it**

1. First implement the simple in-process version with LiveKit agent handoff, where each specialist is an `Agent` subclass and a tool returns the next agent. Measure it.
2. Then split one specialist (for example Admissions) into a separate service that speaks the A2A protocol, with an agent card describing its skills. Nexi calls it as a tool.
3. Compare both versions: latency, complexity and failure behaviour. Writing this comparison is the real showcase.

**Honest trade-off:** for a single university, A2A is overkill. Build it as an architecture exercise after everything else works, and say so in the README.

### 9.4 Admin portal

- Dashboard: calls per day, channels, languages, average latency, top questions.
- **Knowledge gaps page:** the unanswered questions from Phase 6, with a button to add an FAQ answer or upload a document.
- Upload triggers re-ingestion automatically and shows the new chunk counts.
- Ticket and lead management for staff.
- Role based access, backed by the Spring Boot API.

### 9.5 Actions and outreach

- **Follow-up message after the call:** WhatsApp or SMS with the fee link, the form link or the call summary.
- **Human handoff with context:** transfer to the right office and pass a short summary so the student does not repeat themselves.
- **Outbound calls:** fee reminders, exam notices and event invitations. This needs student consent, DND and TRAI compliance, an opt-out, and calling-hour rules. Treat it as a compliance project first and a technical one second.
- **Feedback:** ask for a one-to-five rating at the end of each call and store it.

### 9.6 Platform and scale

- **PostgreSQL with pgvector** instead of Chroma, so relational and vector data live in one database.
- **Redis cache** for repeated questions such as fee structure, with a short expiry.
- **Event pipeline:** call events go to a queue (Redis Streams or RabbitMQ), and workers handle transcripts, analytics and CRM sync outside the call path.
- **OpenTelemetry traces** across agent, API and database, with a Grafana dashboard.
- **Kubernetes** with autoscaling workers, only after the Docker setup is stable.
- **Multi-tenancy:** make university name, data folders, languages, prompts and phone numbers configuration, so a second campus is a config change, not a code change.

### 9.7 Suggested order

| Order | Item | Effort | Why this order |
| --- | --- | --- | --- |
| 1 | Spring Boot API with verified student mode | Medium | Biggest real-world value and the Java story |
| 2 | Admin portal with the knowledge-gap loop | Medium | Turns a demo into a product |
| 3 | Follow-up messages, tickets, human handoff | Low | Quick wins that show real actions |
| 4 | MCP servers | Low to medium | Current, easy to explain, and measurable |
| 5 | pgvector, Redis, event pipeline | Medium | Scale story |
| 6 | A2A specialists | Medium | Architecture showcase |
| 7 | Outbound campaigns | Medium | Needs compliance first |
| 8 | Kubernetes and multi-tenancy | High | Only once everything else is stable |

## 10. Schedule

**Normal pace (about 6 weeks, part time)**

| Week | Work | Output |
| --- | --- | --- |
| 1 | Phase 0 and Phase 1 | Clean, stable agent with baseline latency |
| 2 | Phase 2 | New ingestion, hybrid retrieval, eval results |
| 3 | Phase 3 | Working Twilio call and a demo recording |
| 4 | Phase 4 | Telugu, Hindi and English |
| 5 | Phase 5 | Polished web frontend |
| 6 | Phase 6 | Metrics, tests, CI, Docker, capacity numbers |
| 7 onward | Phase 7 items | Spring Boot first |

**Fast track (about 3 weeks, if you need a demo soon)**

1. Days 1 to 3: Phase 0 and Phase 1.
2. Days 4 to 6: Phase 2 without hybrid search (chunking, ingestion, threshold, a small eval set).
3. Days 7 to 9: Phase 3 with the Twilio trial, plus the demo recording.
4. Days 10 to 14: a simple Phase 5 frontend (landing page, call view, token route).
5. Days 15 to 21: Phase 4 for Telugu and Hindi with the language selector only, and a README with diagrams.

Start the Indian-carrier KYC paperwork (5.7) on day 1 in either plan, because it runs in parallel and takes days.

## 11. Master checklist

- [ ] Phase 0: repo, pinned requirements, `.env.example`, accounts and keys, `lk` CLI linked
- [ ] Phase 1: 29 bugs fixed, old code deleted, baseline latency saved
- [ ] Phase 2: new chunking, offline ingestion, hybrid retrieval, threshold tuned, eval at target
- [ ] Phase 3: Twilio trunk, LiveKit inbound trunk, dispatch rule, successful call, demo recording
- [ ] Phase 4: language selector and phone menu, language switching, per-language eval
- [ ] Phase 5: landing page, call view, token route, error states, deployed
- [ ] Phase 6: call summaries, unanswered-question log, tests, CI, Docker, capacity numbers
- [ ] Phase 7: Spring Boot API and student verification
- [ ] README: architecture diagram, call flow, latency table, eval table, limits, demo link

## 12. Risks and how to handle them

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Twilio trial limits (verified callers only, one trunk, 30-day units) | Cannot let real students call | Use it for proof of concept only, top up if blocked, plan the Indian carrier early |
| Indian carrier KYC delay | Real number takes days or weeks | Start paperwork on day 1 |
| LiveKit Agents API changes between versions | Snippets break | Pin versions, read the changelog before upgrading |
| Sarvam STT latency in the plugin | Slow turns | Measure early, keep Deepgram for English if it is clearly faster |
| Wrong answers (hallucination) | Students get false fee or date information | Threshold, answer only from context, eval set in CI, unanswered log |
| Personal data spoken to the wrong person | Privacy breach | Verification before any personal data, audit log |
| Cost of voice minutes | Unexpected bills | Call time limits, rate limits, usage logging |
| Over-scoping | Nothing finished | Follow the phase gates and finish Phase 6 before Phase 7 |

## 13. How to present this on a resume and in interviews

- One sentence of the problem: a multilingual voice assistant that answers university questions over the web and the phone.
- Three numbers from your own measurements: p95 response latency, retrieval hit rate on your eval set, and concurrent calls supported.
- Three design decisions you can defend: a framework-owned conversation loop instead of a custom one, offline ingestion with hybrid retrieval and a relevance threshold, and a Python AI runtime backed by a Spring Boot services layer.
- Be honest about trade-offs (MCP adds a network hop, A2A is overkill for one university). Interviewers trust candidates who say this.
- Keep a 2-minute demo video: a phone call, the web UI, the Telugu answer and the dashboard.

## 14. Working agreement for us

1. We go phase by phase. You paste the current file or error, and I give the next exact change.
2. For every snippet that depends on a library version, tell me the installed version and I will adapt it.
3. After each phase, run its **Done when** list and send me the results before we move on.

## 15. Sources used for this plan

- LiveKit telephony overview, inbound trunks and dispatch rules: docs.livekit.io/telephony
- LiveKit inbound calls with Twilio and the Twilio SIP trunk quickstart: docs.livekit.io/telephony/accepting-calls/inbound-twilio
- LiveKit region pinning and the Plivo guide (India requirement): docs.livekit.io/telephony/features/region-pinning
- LiveKit Sarvam plugin reference and TTS guide: docs.livekit.io/agents/models/tts/sarvam
- Twilio free-trial limitations and SIP trunking scale and limits: twilio.com/docs/sip-trunking/scale-and-limits
- Sarvam notes on building for India and 8 kHz telephony: docs.sarvam.ai
- LiveKit community threads on Indian numbers and the Twilio host mismatch issue: community.livekit.io

Trial rules, plugin options and provider prices change often. Re-check any of them on the provider's own page before relying on it.
