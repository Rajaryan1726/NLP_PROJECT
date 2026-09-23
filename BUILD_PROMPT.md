# Build Prompt: Saaraansh (FactIndic)

You are building a complete course project from scratch in this folder (`C:\NLP PROJECT`, Windows). Read this whole file before writing any code. Build it phase by phase (Section 12), and after each phase run it and confirm it works before moving on.

## 1. What this project is

**Saaraansh** is a factuality-aware summarizer for Hindi and Hindi-English code-mixed text, built for the university course CSET 346 (NLP), Project 6 "FactIndic: Factuality-Aware Summarization for Indic and Code-Mixed Languages".

The assignment requires:
- Summarize Hindi / English / code-mixed / code-switched / noisy text with an LLM.
- Preserve facts, entities, numbers, dates, events and relationships. Never add unsupported information.
- Compare a **standard prompt** with a **factuality-aware prompt**.
- **Verify** whether each summary is factually supported by the source.
- Analyze how **code-mixing** affects summary quality (monolingual Hindi vs code-mixed).
- Report Precision, Recall, F1, average summary length, compression ratio and inference time.
- Deliverables: GitHub source code, an IEEE-style research paper, presentation slides. So the evaluation outputs must be paper-ready (tables + charts).

**Output language rule:** every summary is written in **Hinglish** (Hindi in Roman/Latin script, natural code-mixed style, e.g. "Sarkar ne nayi scheme launch ki hai"). Never Devanagari, never pure English. Input can be Devanagari Hindi, English, or code-mixed.

## 2. Scope

In scope: summarization, claim-level factuality verification with retrieval + reranking + NLI + entity/number matching, corrective retry loop, Mem0 correction memory, MongoDB, Qdrant, full React UI, evaluation scripts.

Out of scope: Neo4j / knowledge graphs, user authentication, payments. The "Upgrade" button and "PRO" badge in the design are visual only.

## 3. Tech stack

**Backend:** Python 3 + FastAPI + Uvicorn.
- **LLM calls:** `litellm` only, never the raw `openai` SDK. The model comes from `MODEL_NAME` in `.env` (default `gpt-4o-mini`). Switching later to a local Llama must only need `MODEL_NAME=ollama/llama3`. All LLM calls go through one module, `llm_service.py`.
- **Embeddings:** `sentence-transformers` with `paraphrase-multilingual-mpnet-base-v2` (768 dims), run locally.
- **Reranker:** `sentence_transformers.CrossEncoder` with a multilingual model: `BAAI/bge-reranker-v2-m3`. If it is too slow on CPU, fall back to `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Put the model name in `.env`. Do not use English-only ms-marco models, because the evidence is Hindi.
- **NLI:** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (multilingual, includes Hindi).
- **NER:** simple and robust. LLM-based extraction of entities/numbers/dates as JSON, plus regex for numbers and dates.
- **Vector DB:** Qdrant. **Database:** MongoDB (`motor` async driver). **Memory:** Mem0 (`mem0ai`, hosted, `MEM0_API_KEY`).

**Frontend:** React + Vite + Tailwind CSS + **shadcn/ui** + **Vercel AI Elements** (built on shadcn, for the AI message / streaming / actions UI). Font: "Plus Jakarta Sans". JavaScript is fine, TypeScript optional.

**Infra:** `docker-compose.yml` with MongoDB and Qdrant, both with `restart: unless-stopped` (without it, a stopped container silently breaks everything with vague "fetch failed" errors). First check whether Docker is installed. If not, tell me and use MongoDB Atlas free tier + Qdrant Cloud free tier instead, via `.env` URLs.

## 4. Secrets

- An old `.env` with my OpenAI key is saved at `C:\NLP PROJECT\old_openai.env`. Move its OpenAI key into `backend/.env` (rename the variable to `OPENAI_API_KEY` if needed), then delete `old_openai.env`. **Never print, log or echo the key.**
- Create `backend/.env.example` with every variable and placeholder values.
- Add `.env` to `.gitignore` before any git commit.
- Variables: `MODEL_NAME`, `OPENAI_API_KEY`, `MONGO_URI`, `QDRANT_URL`, `QDRANT_API_KEY` (optional), `MEM0_API_KEY`, `EMBED_MODEL`, `RERANK_MODEL`, `NLI_MODEL`, `FACTUALITY_THRESHOLD=7`, `MAX_RETRIES=2`.

## 5. The pipeline (core logic)

For each request `{ text, length: short|medium|long, style: paragraph|bullets, compare: bool, session_id }`:

1. **Preprocess:** clean whitespace, normalize Unicode, convert Devanagari digits (०-९) to ASCII. Detect language type from script ratio: `hindi` (mostly Devanagari), `english` (mostly Latin, English words), `code_mixed` (a real mix, or romanized Hindi). A regex/heuristic is enough.
2. **Chunk + index:** split the source into chunks of 2-3 sentences (split on `।`, `.`, `?`, `!`). Embed and upsert into Qdrant collection `factindic_chunks` with payload `{doc_id, chunk_id, text, start, end}`. **Every** Qdrant read must filter by `doc_id`.
3. **Memory recall (Mem0):** search Mem0 for past corrections for this `session_id` relevant to the text. Inject hits into the prompt as soft rules ("Previously a claim about X was marked wrong because Y"). Important: check the installed `mem0ai` version's exact `search()` signature. Newer versions require `filters={"user_id": ...}` instead of a top-level `user_id`. Write a test that proves `search()` returns what `add()` stored. Do not wrap it in a try/except that silently returns `[]`; log the error.
4. **Generate** the Hinglish summary (streamed). Two prompt templates in `prompts.py`:
   - **Standard:** "Summarize this text in Hinglish (Roman script), {length}, as {style}."
   - **Factuality-aware:** first list the key facts (entities, numbers, dates, events) from the source, then write the summary using only those facts. Keep every number, date and name exactly. Never add information not in the source. Write in Hinglish (Roman script), never Devanagari. Handle code-mixed phrases correctly.
   When `compare=true`, run both. Otherwise use the factuality-aware prompt.
5. **Atomic claim decomposition:** the LLM splits the summary into short independent factual claims (JSON list).
6. **Evidence retrieval + reranking, per claim:** Qdrant top-10 (filtered by `doc_id`), then the cross-encoder reranks to the top-3.
7. **Verification, per claim:**
   - **Language normalization first (important):** claims are romanized Hinglish and the evidence may be Devanagari Hindi. Multilingual NLI works badly across scripts and on romanized Hindi. So translate the claim and its top-3 evidence chunks into English with the LLM before NLI, and cache the translations.
   - NLI (premise = evidence, hypothesis = claim) gives `entailed | neutral | contradicted` plus a confidence.
   - If no chunk passes a minimum rerank score, mark the claim `unsupported` without calling NLI (abstention).
   - **Entity/number/date matching:** extract entities, numbers and dates from source and summary, normalize them (`2 crore` = 20,000,000, `4,500 crore`, lakh, Devanagari digits, common date formats), and compare. Summary values missing from the source are mismatches.
8. **Score:** `claim_score` = entailed claims / total claims, `entity_score` = matched facts / facts in summary. `factuality_score = 10 * (0.6*claim_score + 0.4*entity_score)`.
9. **Corrective retry loop:** if `factuality_score < FACTUALITY_THRESHOLD`, regenerate with a corrective prompt that lists the failed claims and their best evidence, at most `MAX_RETRIES` times. If it still fails, return the best attempt with `low_confidence=true`. Never fail silently.
10. **Save** everything to MongoDB, and return the summary, score, claims (with verdict, confidence, evidence chunk text, chunk id), entity mismatches, stats and attempt count.
11. **Correction write-back:** when the user marks a claim wrong in the UI, call Mem0 `add()` with the correction for that `session_id`.

Use an **embedding cache** in MongoDB (`text_hash -> vector`) so identical chunks are never re-embedded. Load all local models **once** at startup, not per request.

## 6. Data model (MongoDB)

- `documents`: `{_id, session_id, source_text, language_type, word_count, created_at}`
- `summary_attempts`: `{_id, doc_id, attempt_number, prompt_type (standard|factuality_aware|corrective), summary, factuality_score, claim_score, entity_score, low_confidence, inference_time_ms, created_at, claims: [{claim_text, verdict, nli_confidence, evidence_chunk_id, evidence_text, rerank_score}], entity_mismatches: [...]}`
- `corrections`: `{_id, session_id, doc_id, claim_text, user_note, created_at}` (also written to Mem0)
- `embedding_cache`: `{_id: text_hash, vector, model_name}`
- `evaluation_runs`: saved results of the evaluation script

No login. The frontend makes a random `session_id`, keeps it in localStorage, and sends it with every request. It is also the Mem0 `user_id`.

## 7. API (FastAPI)

- `POST /api/summarize`: Server-Sent Events stream with event types `status` (phase text such as "Generating", "Verifying claims", "Improving summary"), `token` (summary text as it streams), `attempt` (a retry started, so the UI should clear the old text), `result` (final JSON: summary, factuality_score, low_confidence, claims, entity_mismatches, stats `{words_in, words_out, compression_ratio, inference_time_ms}`, doc_id, attempt_id), `compare_result` (the standard-prompt summary and its score, when `compare=true`), `error`.
- `POST /api/corrections`: `{session_id, doc_id, claim_text, note}`, stores in MongoDB and Mem0.
- `GET /api/history?session_id=...`: last 20 documents with a title snippet and time.
- `GET /api/history/{doc_id}`: full saved result.
- `DELETE /api/history?session_id=...`: clear history.
- `GET /api/stats`: total documents summarized, average factuality score, average inference time (for the stat cards).
- `GET /api/health`: checks MongoDB, Qdrant, the LLM config and that the models are loaded.
- Validation: empty text gives 400. More than ~5,000 words gives 400 with a clear message. LLM or database failures give a clear error, never a raw stack trace.

## 8. Frontend

**Design reference:** `C:\NLP PROJECT\Main@1x.png` (open and look at it). Premium dark SaaS look, dark by default with a working light/dark toggle. Near-black background `#08080A`, cards `#131317` with `rgba(255,255,255,0.08)` borders and an inset top highlight, violet accent `#A78BFA` with a gradient to `#D8B4FE`, a soft violet glow at the top of the page, rounded 16-18px cards.

Layout (keep it): top bar (logo "सा", "Saaraansh", PRO badge, model pill showing the real `MODEL_NAME`, theme toggle, Upgrade button), a strip of 4 stat cards (real numbers from `/api/stats`), then 3 columns: History, Input, Output.

Fix these problems visible in the reference image:
- Text wrapping: "Hindi + Hinglish" overlaps its subtitle, and "Paste Text", "Upload .txt", "Bullet Points", "Code-Mixed Input Detected", "97% factual", "gpt-4o-mini" and the stat labels break onto two lines. Use `whitespace-nowrap` and proper widths.
- The Input and Output columns have large empty space at the bottom. Fill the Output column with the **Claim Verification panel** below.

Features:
1. Paste text, or upload a `.txt` file.
2. A "Code-Mixed Input Detected" badge (from the backend's `language_type`, with a quick client-side guess while typing).
3. Summary length (Short / Medium / Long) and style (Paragraph / Bullets) as shadcn toggle groups.
4. A "Compare Standard vs Factuality-Aware" switch. When on, show both summaries side by side with their scores.
5. Summarize button with loading states driven by the SSE `status` events.
6. Streaming output in an AI Elements message/response component, with a live cursor.
7. Stats line: words in → words out, % compressed, time.
8. A factuality badge (score out of 10 or as %), green, amber or red. A "Low confidence" warning when `low_confidence` is true.
9. **Claim Verification panel:** one row per claim with a verdict chip (Supported / Not sure / Contradicted / Unsupported). Hovering or expanding shows the evidence chunk from the source. Each row has a "Mark wrong" action that opens a small note input and calls `/api/corrections`.
10. Highlight the sentence in the Input text that a selected claim's evidence came from.
11. Copy, Regenerate, thumbs up/down actions (AI Elements actions).
12. History sidebar from `/api/history`. Clicking an item reloads its full result. "Clear history" works.
13. Toasts for errors (shadcn `sonner`).

## 9. Evaluation (needed for the paper)

Write `backend/eval/`:
- **Datasets:** (a) Hindi news: the `hindi` split of `csebuetnlp/xlsum` (HuggingFace), a sample of about 100 articles. (b) Code-mixed: take about 50 of those articles and make code-mixed versions with an LLM (a strict prompt that must keep every fact, number and name), and save them to `eval/data/`. Document clearly in the README that the code-mixed set is synthetic. That is a limitation to state in the paper.
- **Experiments:** standard vs factuality-aware prompt, on monolingual Hindi and on code-mixed input. Also with vs without the retry loop.
- **Metrics:** claim-level precision/recall/F1 against a small hand-labelled subset (make a simple CSV labelling template, about 30 summaries, which I will fill in), average factuality score, entity/number mismatch rate, average summary length, compression ratio, average inference time. ROUGE against the XL-Sum reference summaries as an extra metric (mention that the references are Devanagari while ours are Hinglish, so compare after transliteration, or report it with that caveat).
- **Outputs:** CSV + JSON results, and matplotlib charts (PNG) ready for the IEEE paper: prompt comparison, Hindi vs code-mixed, score distribution, error categories (number mismatch / entity swap / negation drop / added info / code-mix misread).
- Make it resumable and cheap: cache LLM outputs, and add a `--limit` flag.

## 10. Project structure

```
C:\NLP PROJECT\
  docker-compose.yml
  README.md          (setup, run commands, architecture, how to switch to local Llama)
  .gitignore
  backend/
    main.py            FastAPI app + routes
    config.py          settings from .env
    llm_service.py     every LiteLLM call (generate, stream, decompose, translate, extract)
    prompts.py         all prompt templates
    preprocessing.py   cleaning, language detection, chunking, number normalization
    vector_store.py    Qdrant + embeddings + embedding cache
    reranker.py        cross-encoder
    verifier.py        claims, NLI, entity matching, scoring
    pipeline.py        the full flow + corrective retry loop
    memory_service.py  Mem0
    db.py              MongoDB
    eval/              evaluation scripts + data + results
    tests/             pytest: preprocessing, number normalization, scoring, a Mem0 round trip
    requirements.txt
    .env.example
  frontend/
    (Vite + React + Tailwind + shadcn/ui + AI Elements)
```

## 11. Quality rules

- Keep the code simple and readable. This is a student project that I have to explain to my professor, so no unnecessary abstractions.
- Short comments only where the reason is not obvious.
- Log each pipeline phase with its time, so the stats are real.
- Everything must work on Windows. Give PowerShell commands.
- No fake numbers anywhere in the UI. Every number comes from the backend.

## 12. Build order (confirm each phase works before continuing)

1. **Infra + skeleton:** docker-compose, check Docker, FastAPI app with `/api/health`, config, `.env` from `old_openai.env`, MongoDB and Qdrant connections.
2. **Summarization:** `llm_service` with LiteLLM, both prompts, Hinglish output, SSE streaming. Test with a real Hindi paragraph and a code-mixed one.
3. **Verification:** preprocessing, chunking, Qdrant, reranker, claims, translation normalization, NLI, entity matching, scoring. Unit tests for number normalization and scoring.
4. **Retry loop + Mem0 + MongoDB persistence + history/stats/corrections APIs.**
5. **Frontend:** the full UI matched to the design, wired to the real API.
6. **Evaluation scripts** + charts + README.

At the end, give me: the exact commands to start everything, a short demo script (which texts to paste to show each feature to my professor), and the one line to change in `.env` to switch from OpenAI to a local Ollama Llama model.
