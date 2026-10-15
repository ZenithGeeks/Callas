"""Per-call sessions. CallState lives only in this object's dict and is deleted by end()."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from typing import Any

from asr_diarization.types import Turn
from tts_voice_agent.memory import CallState

from .graph import Runtime, build_message_graph, build_turn_graph


def to_jsonable(obj: Any) -> Any:
    """Make a graph update JSON-safe (for server-sent events). Audio becomes its byte length."""
    if isinstance(obj, bytes):
        return {"audio_bytes": len(obj)}
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    return obj


class CallSessions:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._turn_graph = build_turn_graph(runtime)
        self._message_graph = build_message_graph(runtime)

    @property
    def _states(self) -> dict[str, CallState]:
        return self.runtime.states

    def start(self, call_id: str | None = None) -> str:
        call_id = call_id or uuid.uuid4().hex
        if call_id in self._states:
            raise ValueError(f"call {call_id!r} already started")
        self._states[call_id] = CallState()
        return call_id

    def _turn_input(self, call_id: str, turn: Turn) -> dict:
        if call_id not in self._states:
            raise KeyError(f"unknown call {call_id!r}; call start() first")
        return {"call_id": call_id, "turn": turn, "timings": {}}

    def on_turn(self, call_id: str, turn: Turn) -> dict:
        """Run one turn through the graph and return the final state dict."""
        return self._turn_graph.invoke(
            self._turn_input(call_id, turn), context=self.runtime.context()
        )

    def stream_turn(self, call_id: str, turn: Turn) -> Iterator[dict]:
        """Yield {"node": name, "update": {...}} as each node finishes."""
        for chunk in self._turn_graph.stream(
            self._turn_input(call_id, turn), context=self.runtime.context(), stream_mode="updates"
        ):
            for node, update in chunk.items():
                yield {"node": node, "update": update}

    def end(self, call_id: str) -> None:
        """Delete the call's memory. Safe to call twice."""
        state = self._states.pop(call_id, None)
        if state is not None:
            state.end()

    def simulate_call(self, audio_path: str) -> Iterator[dict]:
        """Transcribe a recording, then feed its turns one by one. Yields node updates.

        Each item also carries call_id and turn_index. The session is ended even if the
        caller stops reading early.
        """
        from asr_diarization import pipeline

        transcript = pipeline.run(audio_path)
        call_id = self.start()
        try:
            for i, turn in enumerate(transcript.turns):
                for item in self.stream_turn(call_id, turn):
                    yield {"call_id": call_id, "turn_index": i, **item}
        finally:
            self.end(call_id)

    def analyze_message(self, text: str) -> dict:
        """Verdict for one text message. No reply and no memory."""
        out = self._message_graph.invoke(
            {"text": text, "timings": {}}, context=self.runtime.context()
        )
        return {"verdict": out["verdict"], "nlp": out["nlp"], "timings": out["timings"]}
