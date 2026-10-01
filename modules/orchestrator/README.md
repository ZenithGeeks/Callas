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
