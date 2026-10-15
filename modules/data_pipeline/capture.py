"""Record what each agent did during a call, for the /studio page.

`Capture` is a DSPy callback. It records, for every agent call, which module ran, which model,
how long it took, and the prompt and answer of each LM call. The web page shows these under
each message.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from dspy.utils.callback import BaseCallback

from .agents import SIGNATURES, Agents


@dataclass
class LmCall:
    """One request to the model: the messages sent and the text that came back."""

    model: str = ""
    messages: list[dict] = field(default_factory=list)
    answer: str = ""
    error: str = ""


@dataclass
class ModuleRun:
    """One agent call (a DSPy program run), with the LM calls it made."""

    agent: str  # scam_caller, receiver, judge, ... (the name in agents.yaml)
    module: str  # the DSPy class, e.g. ChainOfThought
    started: float
    seconds: float = 0.0
    error: str = ""
    lm_calls: list[LmCall] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.lm_calls[-1].model if self.lm_calls else ""


class Capture(BaseCallback):
    """DSPy callback that records agent calls. Install it with `dspy.context(callbacks=[cap])`.

    Only the outermost module of a call is kept (ChainOfThought, not the Predict inside it).
    One Capture serves one call, where agents run one at a time, so an LM call belongs to the
    one open module. `take()` hands over what finished since the last time.
    """

    def __init__(self, agents: Agents | None = None) -> None:
        self._names = {id(getattr(agents, n)): n for n in SIGNATURES} if agents else {}
        self._lock = threading.Lock()
        self._open: dict[str, ModuleRun] = {}  # module call id -> run (outermost only)
        self._lm: dict[str, LmCall] = {}
        self._done: list[ModuleRun] = []

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        with self._lock:
            if self._open:  # a module inside the agent's program
                return
            name = self._names.get(id(instance), type(instance).__name__)
            self._open[call_id] = ModuleRun(name, type(instance).__name__, time.perf_counter())

    def on_module_end(self, call_id: str, outputs: Any, exception: BaseException | None = None):
        with self._lock:
            run = self._open.pop(call_id, None)
            if run is None:
                return
            run.seconds = round(time.perf_counter() - run.started, 3)
            run.error = type(exception).__name__ if exception else ""
            self._done.append(run)

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        with self._lock:
            if not self._open:
                return
            messages = inputs.get("messages")
            if not messages:
                messages = [{"role": "user", "content": str(inputs.get("prompt", ""))}]
            call = LmCall(
                model=str(getattr(instance, "model", "")),
                messages=[
                    {"role": m.get("role", ""), "content": _as_text(m.get("content"))}
                    for m in messages
                ],
            )
            self._lm[call_id] = call
            next(iter(self._open.values())).lm_calls.append(call)

    def on_lm_end(self, call_id: str, outputs: Any, exception: BaseException | None = None):
        with self._lock:
            call = self._lm.pop(call_id, None)
            if call is None:
                return
            if exception is not None:
                call.error = type(exception).__name__
            else:
                call.answer = _answer_text(outputs)

    def take(self) -> list[ModuleRun]:
        with self._lock:
            done, self._done = self._done, []
            return done


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _answer_text(outputs: Any) -> str:
    """The model's reply as text. DSPy gives a list of strings, or of dicts with a "text" key."""
    items = outputs if isinstance(outputs, list) else [outputs]
    parts = [i.get("text", "") if isinstance(i, dict) else str(i) for i in items]
    return "\n".join(p for p in parts if p)
