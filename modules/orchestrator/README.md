# orchestrator

LangGraph layer that wires the modules together for one call turn or one text message. It also
gives nlp-core as swappable tools, and a tool-calling "analyst" agent on the vLLM server.

## Graphs

Call turn (`graph.build_turn_graph`):

```
START -> remember -> analyze -> decide --auto_reply--> reply -> speak -> END
                                   |
                                   +--otherwise-----> escalate -> END
```

- remember: `CallState.add(turn)`
- analyze: `analyze_with_tools(CallState.context())` (NER + summarizer + classifier from the
  registry), then the entities go into `entities_seen`
- decide: `tts_voice_agent.decision.decide`
- reply: `safe_reply` (leak check, template fallback) using the injected LLM
- speak: `Speaker.synthesize_bytes`. Left out of the graph when no speaker is injected
- escalate: no-op that records the reason

Text message (`graph.build_message_graph`): `START -> analyze -> verdict -> END`. No reply, no memory.
The verdict level is `safe`, `suspicious` (scam label with low or unknown confidence) or `scam`.

Every node records its run time in `state["timings"]`.

`sessions.CallSessions` is the entry point: `start`, `on_turn`, `stream_turn` (per-node updates,
for server-sent events; use `to_jsonable` on them), `end`, `simulate_call(audio_path)`,
`analyze_message(text)`.

## Where the safety rules live

- Only `decide()` chooses between escalate and reply. The edge after `decide` is
  `graph.route_after_decide`, plain Python reading `Decision.auto_reply`. No LLM is involved in routing.
- All speech text comes from `safe_reply`. `speak` only voices `reply_text`.
- Call memory is the `CallState` dict in `Runtime.states`, per call id, in process memory.
  `CallSessions.end()` deletes it. The graphs have no checkpointer, so nothing is persisted.
- The only LLM is the vLLM server at `VLLM_BASE_URL`.
- The analyst agent (`agents.explain`) is advisory text for the UI. It never feeds `decide()`.

Tests for these are in `tests/test_orch_graph.py` (routing, sticky escalation, leak fallback).

## DSPy reply and leak judge

Two optional switches in `.env` (both default to the old behaviour). They need the extra:
`pip install -e ".[dspy]"`.

| env var | values | what it does |
|---------|--------|--------------|
| `REPLY_ENGINE` | `prompt` (default), `dspy` | `dspy` writes the reply with a DSPy `ChainOfThought` program (`llm_serving/dspy_programs.py`) on the vLLM server |
| `LEAK_JUDGE` | `off` (default), `on` | `on` asks a DSPy judge whether the reply leaks anything from the call |

Rules that hold for both:

- The reply LLM (prompt or DSPy) only sees the redacted context: `tts_voice_agent.redact.redact_context`
  swaps entities seen in the call, digits, spoken numbers and links for `[LABEL]`, `[NUMBER]`, `[LINK]`.
- Final decision: `blocked = leaks_sensitive(reply) OR judge_blocks`. The rule check stays the hard
  gate and runs first; the judge runs only when the rules pass and can only add blocks. If the judge
  raises, times out, or does not answer a clear "no leak", the reply is blocked and a template is used.
- Chain-of-thought reasoning is discarded inside the writer. It is never returned, logged, or put in
  graph state or events. Logs carry only lengths, timings and reason codes (`judge_block`, `judge_error`).
- The LLM still never decides warn vs reply; that is `decide()`.

Optimize the writer offline, against your own vLLM server (never run in tests):

```
python -m llm_serving.dspy_optimize --out modules/llm_serving/compiled/reply.dspy.json
python -m llm_serving.dspy_optimize --out reply.dspy.json --trainset my_contexts.jsonl
```

The default trainset is a small synthetic Thai set. It prints the dev score before and after, and
`DSPyReplyWriter` loads `llm_serving/compiled/reply.dspy.json` on start if it exists.

## Add your WangchanBERTa model

The three parts are protocols in `nlp_core/interfaces.py`:

| kind | method | env var | default |
|------|--------|---------|---------|
| `ner` | `extract(text) -> list[Entity]` | `NLP_NER` | `rules` |
| `classifier` | `predict(text) -> (label_scam, label_type, confidence)` | `NLP_CLASSIFIER` | `llm` |
| `summarizer` | `summarize(text) -> str` | `NLP_SUMMARIZER` | `llm` |

Classifier steps (NER and summarizer are the same):

1. Write a class in `nlp_core/wangchanberta.py` with a `predict` method. `label_type` is one of
   `SCAM_TYPES` or `None`. If the model is unsure, return `(True, None, 0.0)`; never return "safe"
   when unsure.
2. Register it, in the same file:

   ```python
   from nlp_core.registry import register

   register("classifier", "wangchanberta", lambda: WangchanbertaClassifier("path/or/hub-id"))
   ```

   Make sure the module is imported at startup (for example from `nlp_core/__init__.py`).
3. In `.env` set `NLP_CLASSIFIER=wangchanberta`.
4. Run `python -m pytest modules -q`. An unknown name raises a `KeyError` that lists the
   registered names.

The tools in `tools.py` and the graph pick the new model up through the registry; nothing else changes.

## vLLM for the analyst agent

Tool calling needs these flags (the parser depends on your model, see the vLLM docs):

```
vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser>
```

Without them the server answers 400 and `explain()` raises `ToolCallingUnavailable`.

Env vars (read through `callas_settings`): `VLLM_BASE_URL`, `VLLM_MODEL` (empty = first model the
server lists), `VLLM_API_KEY`, `VLLM_TIMEOUT_S`, plus `NLP_NER`, `NLP_CLASSIFIER`, `NLP_SUMMARIZER`.

Python dependencies: `langgraph`, `langchain-core`, `langchain-openai`.
`langgraph.prebuilt.create_react_agent` is deprecated in favour of `langchain.agents.create_agent`
(package `langchain`, not installed here); `agents.py` is the only place to change when moving.

## Live calls

`orchestrator.live.LiveCall` runs one live (microphone) call. Audio goes into
`asr_diarization.streaming.LiveTranscriber`, which makes partial text and closed turns (energy VAD,
pythaiasr streaming ASR, model loaded once per process). Each closed turn goes through the same turn
graph on ONE worker thread, in order, so slow LLM or TTS work never blocks audio intake. Callas'
replies are not added to call memory.

Client to server (WebSocket):

- binary frames: PCM16 little-endian, mono, 16 kHz
- text JSON: `{"type":"pause"}` (drop audio; the browser pauses the mic while the reply plays),
  `{"type":"resume"}`, `{"type":"end"}`

Server to client (JSON, one per message):

| type | fields |
| --- | --- |
| `ready` | `call_id` |
| `partial` | `text` (running text of the open turn) |
| `turn` | `speaker`, `start`, `end` (seconds since start), `text` |
| `node` | `turn`, `node`, `data` (graph update; audio shown as its byte length) |
| `verdict` | `turn`, `label_scam`, `label_type`, `confidence`, `escalated`, `reason`, `entities` (`text`, `label`, `start`, `end`), `summary` |
| `reply` | `turn`, `text`, `template_used`, `audio_wav_b64` (null if no speaker or TTS failed). Not sent when the call is escalated |
| `error` | `turn`, `message` (exception type only, never call text); the call stays alive |
| `ended` | `call_id` (always last; call memory is deleted) |

The WebSocket loop calls `feed(bytes)` for each binary frame, `poll_events()` to drain events, and
`end()` on close.

Tuning constants (`asr_diarization/streaming.py`): `VAD_RMS_THRESHOLD` (0.015; about 2-3x your room
noise RMS), `SILENCE_CLOSE_S` (0.8), `MIN_TURN_S` (0.5), `MAX_TURN_S` (15.0), `PARTIAL_EVERY_S` (0.5).
