# Phase 0 and Phase 1 implementation record

Date: 2026-11-11. Status: implementation prepared; execution and acceptance testing deferred at the user's request. No API keys are available for verification.

## Delivered

- Explicit validated settings, a complete `.env.example`, Python 3.11 baseline, UTF-8 direct requirements, and a hash-pinned universal dependency lock.
- Kept LiveKit 1.2.6 and matching speech/provider plugins; added the missing Google and Sarvam plugins. Version-specific signatures were read from the installed source and published wheels, without importing/running providers.
- A single `AgentSession` reply path, supported transcription observation, fixed greeting, interim STT, turn detection, stage metrics, usage summaries, bounded cleanup, and one best-effort fatal-error notification.
- Embedding/reranking/VAD setup in prewarm. Turn detector facade is created in the job because LiveKit 1.2.6 requires its job inference executor; the framework manages the detector inference model separately.
- Offline bootstrap builds a new index version, preserves prior versions, writes a manifest/source coverage report, and atomically replaces `CURRENT`. Runtime never invokes ingestion or creates a missing collection.
- Dense retrieval with explicit raw-logit reranking, provisional rejection, complete-chunk context packing, and one outstanding inference task per job process. Timed-out work retains its slot until its underlying thread finishes.
- Provider/query content excluded from application logs. Per-call counters and usage are logged; database persistence/dashboard remain Phase 6.
- Removed the unused standalone session manager, obsolete tool helpers, static prompt, and competing reply loop. Root `clean.py` and `rag_engine.py` remain small CLI compatibility launchers.
- Added a stdlib offline regression file for later execution; no tests were run.

## Preserved inputs

The original pre-refactor application files, requirements, README, and ignore rules were preserved in the ignored `.local-baseline/phase-0-input.zip`. This is a local recovery snapshot, not a deployed artifact; it excludes environment files/secrets. The historical requirements were separately decoded into `docs/requirements.pre-phase0.txt` for reference, not installation.

No commit, branch change, key-file edit, provider provisioning, deployment, model execution, or ingestion was performed. The implementation playbook is unchanged. PDFs remain in `uni_pdfs` to avoid unnecessary moves/duplicate copies; `PDF_DIR` makes this portable. `data/json` is empty by design.

## Deliberate simplifications

- Used Chroma's public collection API and SentenceTransformers directly; retained only LangChain's text splitter for bootstrap chunking. No redundant LangChain store wrapper or speculative retriever interface.
- Provider construction and short lifecycle callbacks stay in `agent/main.py`; separate factory/lifecycle packages are unnecessary until later phases add branching behavior.
- No background thinking sound, custom reconnect loop, custom history manager, BM25, OCR, phone provisioning, multilingual handoff, frontend, or durable analytics in this phase.
- The bootstrap traverses JSON without dropping children, but full record semantics, temporal filtering, OCR quality and hybrid retrieval remain Phase 2.
- No account login or project-linking step can be completed without credentials. No baseline performance or quality numbers are claimed.

## Dependency preparation versus testing

Package metadata and API source were read. `uv pip compile` resolved and wrote `requirements.lock` with hashes for the Python 3.11 baseline; it did not install dependencies or run application code. Successful resolution is not evidence of a successful clean install or compatible provider behavior. The old SDK baseline and present provider model availability must be exercised when testing resumes.

Version references:

- [Google plugin 1.2.6](https://pypi.org/project/livekit-plugins-google/1.2.6/)
- [Sarvam plugin 1.2.6](https://pypi.org/project/livekit-plugins-sarvam/1.2.6/)
- [LiveKit Agents 1.2.6](https://pypi.org/project/livekit-agents/1.2.6/)

The current interpreter on this workstation is not the targeted Python 3.11 environment. The existing Windows virtualenv was not reused or modified. A clean install is deferred.

## Required later validation, in order

1. Install the lock in a fresh Python 3.11 virtual environment; verify dependency imports and platform audio prerequisites.
2. Run the saved offline regression checks and lint after testing is authorized. Add/fix tests from any failures rather than treating source review as a pass.
3. Download LiveKit plugin assets explicitly, then build the offline knowledge artifact. With today's scanned PDFs, default bootstrap should refuse incomplete extraction; review the coverage report before explicitly allowing partial development data.
4. Supply provider keys; verify the selected STT, Gemini, Sarvam model/voice/rate combinations and LiveKit account features.
5. Exercise console and room scenarios: one response, greeting once, interruption, multi-turn context, missing evidence, timeout, busy inference, provider failure, participant departure and cleanup.
6. Confirm no model/index construction happens during calls; ensure models are cached under the same account that runs the worker.
7. Capture real endpointing delay, LLM TTFT, TTS TTFB and tool timing. Do not label elapsed full speech playout as model latency.
8. Review the phase gates before Phase 2. Phase 0/1 are **implemented but not acceptance-verified**, not production-ready.

## Known limits to carry forward

- Five source PDFs lacked substantive extracted text during the earlier assessment. OCR/replacements are outstanding. Some other sources are dated 2024/2025 and cannot establish current university facts.
- `RERANK_MIN_SCORE=0.0` is an uncalibrated raw-logit development default. Evaluation and refusal quality are not proven.
- Grounding currently uses the prompt and retrieval evidence gate; model compliance needs behavioral testing. This is not a hallucination guarantee.
- A provider failure can make even the error announcement impossible; failure speech is best effort and bounded.
- SDK turn-final transcript counts are a proxy for user turns. Accurate turn correlation and durable latency dashboards belong in Phase 6.
- This phase includes channel-specific prompt/audio profiles, not validated phone-call termination, SIP transfers, or public-launch cost controls. Do not expose it as an unattended public service.
- Local embedded Chroma can update internal bookkeeping even when this code only queries. Stop workers before switching active versions; do not assume a filesystem read-only mount works without testing it.
- Keep old index versions until a replacement is verified. `CURRENT` can be restored to a prior version only if its embedding fingerprint matches the configuration.

## Source inventory carried from the assessment

| Source | Current finding | Next action |
| --- | --- | --- |
| Internship policy | No substantive extracted text | OCR or accessible replacement |
| Financial assistance policy | No substantive extracted text | OCR or replacement |
| Seed grant policy | No substantive extracted text | OCR or replacement |
| Academic programmes ordinances | No substantive extracted text | OCR or replacement |
| Student code of conduct | No substantive extracted text | OCR or replacement |
| Academic calendar AY 2025–26 | Text present; dated | Review extraction and source freshness |
| May 2025 endterm instructions | Text present; historical | Keep historical scope explicit |
| Hostel policy | Text present | Review version/authority |
| Attendance policy | Text present, artifacts | Review numerical conditions |
| Student handbook 2024 | Text present | Review age and mixed image pages |
| UGC proforma/appendices | Text present | Review relevance, dates and tables |
| JSON corpus | Not supplied | Use real approved records when available |
