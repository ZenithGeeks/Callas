# orchestrator

The LangGraph layer. It wires the modules together for one call turn or one text message. It also offers the nlp-core models as tools to an "analyst" agent that explains a message on the vLLM server. The drawing of the graphs is in [`docs/diagrams/langgraph.md`](../../docs/diagrams/langgraph.md). To draw it again after you change a graph:

```
python -m orchestrator.show_graph --out docs/diagrams/langgraph.md
```

Run it from the repo root with `PYTHONPATH=modules`, or after `pip install -e .`. Without `--out` it prints to the terminal.

## Graphs

Call turn (`graph.build_turn_graph`):

```
START -> remember -> analyze -> decide --auto_reply--> reply -> speak -> END
                                   |
                                   +--otherwise-----> escalate -> END
```

- remember: cleans the turn text (`clean_turn`) and adds it to `CallState`.
- analyze: runs NER, summarizer and classifier from the registry on `CallState.context()` and returns that text as `transcript`. The entities go into `entities_seen`.
- decide: `tts_voice_agent.decision.decide`.
- reply: `safe_reply` (leak check, template fallback) with the injected LLM. It returns `reply_text` and `template_used`; the `Decision` from decide is not changed. The reply is added to call memory as an `agent` turn, so the next reply can follow on.
- speak: `Speaker.synthesize_bytes`. The node exists only when a speaker is injected.
- escalate: records the reason. Nothing else.

Text message (`graph.build_message_graph`): `START -> analyze -> verdict -> END`. No reply and no memory. The verdict level is `safe`, `suspicious` (scam label with confidence below 0.7, or unknown type) or `scam`. The 0.7 is `ESCALATE_THRESHOLD` in `tts_voice_agent/constants.py`.

Every node records its run time in `state["timings"]`.

`sessions.CallSessions` is the entry point: `start`, `on_turn`, `stream_turn` (per-node updates for server-sent events; use `to_jsonable` on them), `end`, `simulate_call(audio_path)` and `analyze_message(text)`.

### Graph design choices

- `graph.Runtime` holds the per-run dependencies (llm, speaker, writer, judge, registry names, call states) and is the graphs' `context_schema`. Nodes read it as `runtime.context`. Every invoke or stream passes `context=runtime.context()`, which is the `Runtime` itself.
- The analyze node has `RetryPolicy(max_attempts=2, initial_interval=0.3, backoff_factor=2.0, retry_on=LLMUnavailable)`. It retries once, and only when the LLM server is unreachable. Reply is not retried because `safe_reply` already falls back to a template.
- There is no node cache and no checkpointer. A cache would keep call text after the call ends. That breaks the rule that call memory is deleted at the end.
- `route_after_decide` is a plain function on `add_conditional_edges`. Tests prove it reads only `Decision.auto_reply`.

## Where the safety rules live

The list of rules is in the root [README](../../README.md#safety-rules). In this module:

- Only `decide()` chooses between escalate and reply. No LLM is involved in routing.
- All speech text comes from `safe_reply`. `speak` only voices `reply_text`.
- Call memory is the `CallState` dict in `Runtime.states`, per call id, in process memory. `CallSessions.end()` deletes it.
- The only LLM is the vLLM server at `VLLM_BASE_URL`.
- The analyst agent (`agents.explain`) gives advisory text for the UI. It never feeds `decide()`.

Tests for these are in `tests/test_orch_graph.py` (routing, sticky escalation, leak fallback).

## DSPy reply and leak judge

Two optional switches in `.env`. Both need `pip install -e ".[dspy]"`. The defaults keep the old behaviour.

| Setting | Values | What it does |
|---|---|---|
| `REPLY_ENGINE` | `prompt` (default), `dspy` | `dspy` writes the reply with a DSPy `ChainOfThought` program (`llm_serving/dspy_programs.py`) |
| `LEAK_JUDGE` | `off` (default), `on` | `on` asks a DSPy judge whether the reply leaks anything from the call |

Rules that hold for both:

- The reply LLM only sees the redacted context. `tts_voice_agent.redact.redact_context` swaps entities seen in the call, digits, spoken numbers and links for `[LABEL]`, `[NUMBER]` and `[LINK]`.
- The reply is blocked when `leaks_sensitive(reply)` or the judge blocks it. The rule check runs first. The judge runs only when the rules pass, and it can only add blocks. If the judge raises, times out, or does not clearly answer "no leak", the reply is blocked and a template is used.
- The chain-of-thought text is discarded inside the writer. It is never returned, logged, or put in graph state or events. Logs carry only lengths, timings and reason codes (`judge_block`, `judge_error`).

Optimize the writer offline against your own vLLM server. Never run it in tests.

```
python -m llm_serving.dspy_optimize --out modules/llm_serving/compiled/reply.dspy.json
python -m llm_serving.dspy_optimize --out reply.dspy.json --trainset my_contexts.jsonl
```

The default trainset is a small synthetic Thai set. The script prints the dev score before and after. `DSPyReplyWriter` loads `llm_serving/compiled/reply.dspy.json` on start if it exists. That file is in git.

## Add a model to nlp-core

The three parts are protocols in `nlp_core/interfaces.py`. The registry is `nlp_core/registry.py`.

| Kind | Method | Variable | Built-in names (default first) |
|---|---|---|---|
| `ner` | `extract(text) -> list[Entity]` | `NLP_NER` | `rules` |
| `classifier` | `predict(text) -> (label_scam, label_type, confidence)` | `NLP_CLASSIFIER` | `llm`, `phayathaibert` |
| `summarizer` | `summarize(text) -> str` | `NLP_SUMMARIZER` | `llm` |

### The PhayaThaiBERT classifier

`NLP_CLASSIFIER=phayathaibert` runs the models the notebook trains (`nlp_core/phayathaibert.py`). It needs the `ml` extra and `models/phayathaibert-scam-gate`, which is not in git; `models/phayathaibert-scam-type` is used when it is there. `NLP_MODELS_DIR` points at another folder. The web backend loads the model when it starts, so a missing model stops startup with a message that says how to get it.

The models learned from the call so far, written one way (the notebook's step 6b), so `predict` rewrites the text it gets into that shape first:

- Speakers become letters: `A` and Callas' own replies (`agent`) are the person called, `A`; any other speaker (`B`, the live page's `Caller`, a third voice) is the caller, `B`. Lines in a row from one letter are joined.
- A text with no `X:` lines (a text message) is one caller line. Lines without a speaker, such as the summary of older turns, are left out.
- Each line is normalized and the sensitive values the entity rules find become `[PHONE]`, `[MONEY]`, ...
- The oldest lines are dropped until the text fits the model's 256 tokens. The newest line always stays.

On the test split this rebuilds every row exactly as the notebook did and gives the same macro-F1. The gate's top probability is the confidence. Below 0.60 the answer is `(True, None, 0.0)`; the scam type is left empty below 0.50 (the notebook's starting values, `GATE_MIN_CONFIDENCE` and `TYPE_MIN_CONFIDENCE` in `phayathaibert.py`).

Known limits:

- The entity rules find phone numbers, bank accounts, OTPs, links and amounts. The training rows also masked LINE IDs, card and ID numbers, birth dates and addresses, which stay as written here.
- The model is trained on generated calls. On hand-written ones it leans on the masked values: a fake-police call that asks for "all your money" with no amount or account scored 0.0003 scam, the same call with `[MONEY]` and `[BANK_ACCOUNT]` 0.997. A text message is read as one caller line, which the model never saw in training.

### Add a model

Steps to connect another one:

1. Write a class with the method above. For a classifier, `label_type` is one of `SCAM_TYPES` or `None`. If the model is unsure, return `(True, None, 0.0)`. Never return "safe" when unsure.
2. Register it, and make sure that module is imported before the first call (for example from `nlp_core/__init__.py`):

   ```python
   from nlp_core.registry import register

   register("classifier", "phayathaibert", lambda: MyClassifier("models/phayathaibert-scam-gate"))
   ```
3. Set `NLP_CLASSIFIER=phayathaibert` in `.env`. A real environment variable wins over `.env`. If both are empty the built-in default is used (`registry.default_name`).
4. Run `python -m pytest modules -q`. An unknown name raises a `KeyError` that lists the registered names.

The tools in `tools.py` and the graphs pick the new model up through the registry. Nothing else changes.

## vLLM for the analyst agent

Tool calling needs these flags. The parser depends on your model; see the vLLM docs.

```
vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser>
```

Without them the server answers 400 and `explain()` raises `ToolCallingUnavailable`. The VLLM_* settings are in the root [README](../../README.md#settings). The agent uses `langchain` (`create_agent`), `langgraph`, `langchain-core` and `langchain-openai`.

## Live calls

`orchestrator.live.LiveCall(sessions, transcriber=None)` runs one live (microphone) call. Audio goes into `asr_diarization.streaming.LiveTranscriber`. It makes partial text and closed turns (energy VAD, pythaiasr streaming ASR, model loaded once per process). `ASR_MODEL` must be a streaming model: `typhoon_asr` or `typhoon_nemotron_asr`.

Each closed turn goes through the turn graph on one worker thread, in order, so slow LLM or TTS work never blocks audio intake. After a turn closes, `LiveCall` waits `REPLY_DELAY_S` (0.7 s). If the same speaker starts again inside that time, the turns are merged and share one verdict. Callas' replies are added to call memory (see the reply node above).

Client to server (WebSocket):

- binary frames: PCM16 little-endian, mono, 16 kHz
- text JSON: `{"type":"pause"}` (drop audio; the browser pauses the mic while the reply plays), `{"type":"resume"}`, `{"type":"end"}`

Server to client (JSON, one per message):

| type | fields |
| --- | --- |
| `ready` | `call_id` |
| `partial` | `text` (running text of the open turn) |
| `turn` | `speaker`, `start`, `end` (seconds since start), `text` |
| `node` | `turn`, `node`, `data` (graph update; audio shown as `{"audio_bytes": n}`) |
| `verdict` | `turn`, `label_scam`, `label_type`, `confidence`, `escalated`, `reason`, `entities` (`text`, `label`, `start`, `end`), `summary` |
| `reply` | `turn`, `text`, `template_used`, `audio_wav_b64` (null if no speaker or TTS failed). Not sent when the call is escalated |
| `error` | `turn`, `message` (exception type only, never call text). The call stays alive |
| `ended` | `call_id`. Always last. Call memory is deleted |

The WebSocket loop in `backend/live_ws.py` calls `feed(bytes)` for each binary frame, `poll_events()` to drain events, and `end()` on close.

Tuning constants in `asr_diarization/streaming.py`: `VAD_RMS_THRESHOLD` (0.015; about 2 to 3 times your room noise RMS), `SILENCE_CLOSE_S` (0.8), `MIN_TURN_S` (0.5), `MAX_TURN_S` (15.0), `PARTIAL_EVERY_S` (0.5).
