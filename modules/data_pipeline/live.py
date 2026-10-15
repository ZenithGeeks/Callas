"""Run one call and report it step by step, for the /studio page.

`stream_call` does what `simulate.run_call` does, but drives the same LangGraph graph with
`graph.stream(stream_mode="updates")` and yields an Event for every step the browser should
show: "thinking" before an agent starts, then "plan", "turn" or "verdict" with the saved record.
The call is saved by the graph nodes exactly as in a command line run.

`Capture` is a DSPy callback. It records, for every agent call, which module ran, which model,
how long it took, and the prompt and answer of each LM call. The web page shows these under
each message. `fake_agents` builds real DSPy programs on a scripted LM, so demo mode runs the
whole stack (prompt, parsing, callbacks) without a model server.
"""

from __future__ import annotations

import itertools
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dspy.utils import DummyLM
from dspy.utils.callback import BaseCallback

from .agents import SIGNATURES, Agents
from .config import Config, ReceiverProfile, Scenario
from .fill import SlotFiller
from .scenarios import sample_config
from .simulate import CALL_GRAPH, SimContext, _open_record, call_id, is_model_error, next_step
from .store import Store

__all__ = ["CallSpec", "Capture", "Event", "ModuleRun", "fake_agents", "stream_call"]


# ---- what the DSPy callbacks collect ---------------------------------------------------------


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
    `take()` hands over what finished since the last time.
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


# ---- one call, step by step ------------------------------------------------------------------


class CallConflict(ValueError):
    """The call cannot be run: it is finished, or it was started with other settings."""


@dataclass
class CallSpec:
    """What the studio form asks for."""

    call_type: str
    seed: int
    scenario: Scenario | None = None  # None: the one the seed picks
    receiver: ReceiverProfile | None = None  # None: the one the seed picks
    receiver_name: str = ""  # the name the caller hears; "" keeps the one the seed picks

    @property
    def record_id(self) -> str:
        return call_id(self.call_type, self.seed, self.receiver)


@dataclass
class Event:
    kind: str  # thinking | plan | turn | verdict
    agent: str  # who is working or worked: planner, scam_caller, legit_caller, receiver, judge
    record: dict
    index: int | None = None  # turn: 0-based position
    modules: list[ModuleRun] = field(default_factory=list)


def stream_call(
    spec: CallSpec,
    store: Store,
    agents: Agents,
    config: Config,
    voices: Any,
    capture: Capture,
    cancel: threading.Event | None = None,
) -> Iterator[Event]:
    """Run (or resume) the call and yield what happens. Stops early when `cancel` is set; the
    call stays in the store as unfinished and a command line run can resume it."""
    call = sample_config(spec.call_type, spec.seed, config, spec.receiver)
    if spec.scenario is not None:
        call = replace(call, scenario=spec.scenario)
    record = _open_record(store, call, config, agents, spec.receiver)
    generation = record["generation"]
    if generation["status"] != "in_progress":
        raise CallConflict(f"{record['id']} is already finished")
    call, name = _resume_overrides(spec, call, generation)
    generation["call"]["receiver_name"] = name  # the overrides stay with the stored call
    filler = SlotFiller(call, config.details)
    if name:
        filler.values["RECEIVER_NAME"] = name
    context = SimContext(agents, config, call, store, voices, filler)
    caller = "scam_caller" if call.label_scam else "legit_caller"

    def whose_turn(rec: dict) -> Event:
        step = next_step({"record": rec}, SimpleNamespace(context=context))
        agent = {"caller": caller, "receiver": "receiver", "finish": "judge"}[step]
        return Event("thinking", agent, rec)

    for index, turn in enumerate(generation["turns"]):  # a resumed call: show what is there
        yield Event("turn", turn["agent"], record, index)
    pending = Event("thinking", "planner", record) if _needs_plan(record) else whose_turn(record)
    yield pending
    written = len(generation["turns"])

    updates = CALL_GRAPH.stream(
        {"record": record},
        context=context,
        config={"recursion_limit": call.max_turns + 5},
        stream_mode="updates",
    )
    for update in updates:
        ((node, result),) = update.items()
        record = result["record"]
        modules = capture.take()
        if node == "plan":
            if record["generation"]["briefing"] is not None:
                yield Event("plan", "planner", record, modules=modules)
        elif node == "finish":
            yield Event("verdict", "judge", record, modules=modules)
            return
        elif len(record["generation"]["turns"]) == written:
            # the turn node ended the call early (a prompt that does not fit): no turn was added
            yield Event("ended", pending.agent, record, written, modules)
            continue  # the finish step follows at once
        else:
            written = index = len(record["generation"]["turns"])
            yield Event("turn", node_agent(record, index - 1), record, index - 1, modules)
        if cancel is not None and cancel.is_set():
            return
        pending = whose_turn(record)
        yield pending


def _resume_overrides(spec: CallSpec, call: Any, generation: dict) -> tuple[Any, str]:
    """The call and the receiver name to use. A call with no turns yet takes what the form asked
    for. A call being resumed keeps the scenario and name it was started with: the form may leave
    them empty but may not ask for others, because the turns already written used the old ones."""
    stored = generation["call"]
    if not generation["turns"] and generation["briefing"] is None:
        return call, spec.receiver_name
    scenario = Scenario.model_validate(stored["scenario"])
    name = stored.get("receiver_name", "")
    if spec.scenario is not None and spec.scenario.key != scenario.key:
        raise CallConflict(
            f"{spec.record_id} was started with scenario {scenario.key}. Resume it with that "
            "scenario or none, or pick another seed."
        )
    if spec.receiver_name and spec.receiver_name != name:
        raise CallConflict(
            f"{spec.record_id} was started with another receiver name. Resume it with that "
            "name or none, or pick another seed."
        )
    return replace(call, scenario=scenario), name


def node_agent(record: dict, index: int) -> str:
    return record["generation"]["turns"][index]["agent"]


def _needs_plan(record: dict) -> bool:
    return bool(record["label_scam"]) and record["generation"]["briefing"] is None


def mark_model_error(store: Store, record_id: str, exc: BaseException) -> None:
    """Same as the command line run: a model error that survived the retries rejects the call
    (llm_error:...). Any other error leaves the call unfinished so it can be resumed."""
    if not is_model_error(exc):
        return
    record = store.load(record_id)
    if record is not None and record["generation"]["status"] == "in_progress":
        record["generation"]["status"] = "rejected"
        record["generation"]["reasons"] = [f"llm_error:{type(exc).__name__}"]
        store.save(record)


# ---- demo mode: a scripted LM and silent voices ----------------------------------------------

FAKE_MODEL = "scripted-demo"

_PLAN = ["สร้างความน่าเชื่อถือ", "บอกว่าบัญชีมีปัญหา", "ขอรหัสจากผู้รับสาย", "เร่งให้รีบทำตาม"]
_SCAM_LINES = [
    ("สวัสดีครับ ผมโทรจาก{ORG}ครับ ไม่ทราบว่าใช่คุณ{RECEIVER_NAME}ใช่ไหมครับ", False),
    ("ตอนนี้บัญชีของคุณมีรายการผิดปกติครับ ต้องรีบตรวจสอบ", False),
    ("รบกวนบอกรหัส{OTP}ที่เพิ่งส่งเข้ามือถือด้วยนะครับ", True),
    ("ถ้าไม่ยืนยันตอนนี้ บัญชีจะถูกระงับทันทีนะครับ", True),
    ("โอนเงินมาที่บัญชี{BANK_ACCOUNT}เพื่อความปลอดภัยครับ", True),
]
_LEGIT_LINES = [
    "สวัสดีครับ ผมโทรจาก{ORG}ครับ ไม่ทราบว่าใช่คุณ{RECEIVER_NAME}ใช่ไหมครับ",
    "โทรมาแจ้งว่าพัสดุของคุณจะถึงพรุ่งนี้ช่วงบ่ายครับ",
    "สะดวกให้ส่งที่บ้านตามที่อยู่เดิมใช่ไหมครับ",
    "ขอบคุณมากครับ ขอให้มีวันที่ดีนะครับ",
]
_REPLIES = [
    "สวัสดีค่ะ ใช่ค่ะ มีอะไรหรือเปล่าคะ",
    "ตกใจเลยค่ะ รายการอะไรเหรอคะ",
    "เดี๋ยวนะคะ ทำไมต้องขอรหัสด้วยคะ",
    "ไม่สบายใจเลยค่ะ ขอโทรกลับไปถามที่ธนาคารเองดีกว่า",
    "ไม่ให้ค่ะ แล้วจะวางสายแล้วนะคะ",
]
_LEGIT_REPLIES = ["สวัสดีค่ะ ใช่ค่ะ", "ได้ค่ะ ขอบคุณที่แจ้งนะคะ", "สะดวกค่ะ ที่เดิมเลยค่ะ", "ค่ะ สวัสดีค่ะ"]


class ScriptedLM(DummyLM):
    """DummyLM that waits `delay` seconds per call, so the page visibly streams."""

    def __init__(self, answers: list[dict], delay: float = 0.0) -> None:
        super().__init__(itertools.cycle(answers))
        self.model = FAKE_MODEL
        self.delay = delay

    def __call__(self, prompt=None, messages=None, **kwargs):
        time.sleep(self.delay)
        return super().__call__(prompt=prompt, messages=messages, **kwargs)


def fake_agents(config: Config, call_type: str, delay: float = 0.0) -> Agents:
    """Real DSPy programs (instructions from the config) on scripted LMs. The same lines repeat
    if the call is long. Asking starts on the caller's second turn."""
    scam = call_type != "benign"
    caller_answers = (
        [
            dict(reasoning=f"step {i + 1} of the plan", line=line, asks_now=asks, end_call=False)
            for i, (line, asks) in enumerate(_SCAM_LINES)
        ]
        if scam
        else [
            dict(reasoning="keep it short and friendly", line=line, end_call=False)
            for line in _LEGIT_LINES
        ]
    )
    reply_lines = _REPLIES if scam else _LEGIT_REPLIES
    receiver_answers = [
        dict(reasoning="answer like this person would", line=line, end_call=False)
        for line in reply_lines
    ]
    judge = dict(
        reasoning="read the whole call against the intended type",
        label_holds=True,
        scam_start_turn=2 if scam else -1,
        natural_thai=4,
        giveaway=False,
        summary="ผู้โทรขอรหัสจากผู้รับสาย" if scam else "ผู้โทรแจ้งเรื่องการส่งพัสดุ",
        issues="none",
    )
    plan = dict(reasoning="a short plan that builds up to the ask", plan=_PLAN)
    answers = {
        "scam_caller": caller_answers,
        "legit_caller": caller_answers,
        "receiver": receiver_answers,
        "planner": [plan],
        "judge": [judge],
        "pattern_extractor": [
            dict(reasoning="-", relevant=False, summary="", tactics=[], phrases=[])
        ],
    }
    lms = {name: ScriptedLM(answers[name], delay) for name in SIGNATURES}
    return Agents(config, lms=lms)


class SilentVoices:
    """Stands in for tts.Voices in demo mode: writes a quiet WAV about as long as the line."""

    def speak(self, text: str, voice: str, path: Path) -> float:
        seconds = max(0.5, round(len(text) * 0.07, 2))
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(bytes(2 * int(16000 * seconds)))
        return seconds


def next_free_seed(call_type: str, taken_ids: set[str], receiver: ReceiverProfile | None) -> int:
    """The smallest seed whose call id is not in the store yet."""
    for seed in itertools.count():
        if call_id(call_type, seed, receiver) not in taken_ids:
            return seed
    raise AssertionError("unreachable")
