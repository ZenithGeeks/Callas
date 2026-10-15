"""DemoLiveCall: a scripted live call with the same interface as orchestrator.live.LiveCall.

Interface: feed(bytes), pause(), resume(), poll_events() -> list[dict], end().

It ignores the microphone audio and plays a fixed call on a clock that stops while paused
(the browser pauses while Callas' own voice is playing). Turns 1 to 3 are normal and Callas
answers after turn 3 with a short silent reply; turn 4 is the scam and everything after it is
held as escalated, with no reply.
"""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import asdict

from . import demo

PARTIAL_AT = (0.3, 0.6, 0.9)  # share of the turn text heard at each partial
SECONDS_PER_PARTIAL = 0.7
REPLY_AFTER_TURN = 3

Script = list[tuple[float, dict]]  # (clock time, event), in the order poll_events sends them


class DemoLiveCall:
    """A fake LiveCall. Read __init__ and poll_events first, then _build_script."""

    def __init__(self, call_id: str | None = None, speed: float = 1.0) -> None:
        self.call_id = call_id or uuid.uuid4().hex
        self.bytes_fed = 0
        self._speed = speed
        self._paused = False
        self._ended = False
        self._ended_sent = False
        self._clock = 0.0  # seconds of call played so far; stops while paused
        self._last = time.monotonic()
        self._script = self._build_script()
        self._next = 0  # index of the next script event to send

    def feed(self, data: bytes) -> None:
        self.bytes_fed += len(data)

    def pause(self) -> None:
        self._tick()
        self._paused = True

    def resume(self) -> None:
        self._tick()
        self._paused = False

    def poll_events(self) -> list[dict]:
        """The script events whose time has come, once each. After end(), only "ended"."""
        self._tick()
        if self._ended:
            if self._ended_sent:
                return []
            self._ended_sent = True
            return [{"type": "ended", "call_id": self.call_id}]
        events = []
        while self._next < len(self._script) and self._script[self._next][0] <= self._clock:
            events.append(self._script[self._next][1])
            self._next += 1
        return events

    def end(self) -> None:
        self._ended = True

    def _tick(self) -> None:
        now = time.monotonic()
        if not self._paused and not self._ended:
            self._clock += (now - self._last) * self._speed
        self._last = now

    def _build_script(self) -> Script:
        """The whole call as timed events: partials, the finished turn, then its decision."""
        script: Script = [(0.0, {"type": "ready", "call_id": self.call_id})]
        ner = demo.DemoNER()
        heard: list[str] = []
        entities: dict[tuple[str, str], dict] = {}  # first sighting of each entity
        escalated = False
        clock = 0.5
        for number, (speaker, start, text) in enumerate(demo.DEMO_TURNS, start=1):
            partials, clock = _partial_events(text, clock)
            script += partials
            clock += SECONDS_PER_PARTIAL
            turn = {
                "type": "turn",
                "turn": number,
                "speaker": speaker,
                "start": start,
                "end": demo.turn_end(number - 1),
                "text": text,
            }
            script.append((clock, turn))

            heard.append(f"{speaker}: {text}")
            call_so_far = "\n".join(heard)
            for entity in ner.extract(call_so_far):
                entities.setdefault((entity.text, entity.label), asdict(entity))
            label = demo.classify_call(call_so_far)
            is_scam = label != "normal"
            reason = _decision_reason(is_scam, already_escalated=escalated)
            escalated = escalated or is_scam
            verdict = {
                "type": "verdict",
                "turn": number,
                "label_scam": is_scam,
                "label_type": label if is_scam else None,
                "confidence": 1.0,
                "escalated": escalated,
                "reason": reason,
                "entities": list(entities.values()),
                "summary": demo.summarize_call(call_so_far),
            }
            script += [
                (clock, _node_event(number, "remember", 0.0)),
                (clock, _node_event(number, "analyze", 0.02)),
                (
                    clock,
                    _node_event(number, "decide", 0.0, auto_reply=not escalated, reason=reason),
                ),
                (clock, verdict),
            ]
            if number == REPLY_AFTER_TURN and not escalated:
                script += _reply_events(number, clock)
            elif escalated:
                script.append((clock, _node_event(number, "escalate", 0.0)))
            clock += demo.TURN_GAP
        return script


def _partial_events(text: str, clock: float) -> tuple[Script, float]:
    """Growing partial transcripts of one turn, and the clock after the last one."""
    events: Script = []
    for share in PARTIAL_AT:
        clock += SECONDS_PER_PARTIAL
        heard = text[: max(1, int(len(text) * share))]
        events.append((clock, {"type": "partial", "text": heard}))
    return events, clock


def _decision_reason(is_scam: bool, already_escalated: bool) -> str:
    """The decide node's reason for this turn."""
    if already_escalated:
        return "call already escalated"
    if is_scam:
        return "high-confidence scam, escalate to user"
    return "benign"


def _node_event(number: int, node: str, seconds: float, **extra) -> dict:
    """A graph node finishing, as the real LiveCall reports it."""
    return {"type": "node", "turn": number, "node": node, "data": {"seconds": seconds, **extra}}


def _reply_events(number: int, clock: float) -> Script:
    """Callas answers with a short silent WAV, a moment after the reply nodes finish."""
    reply = {
        "type": "reply",
        "turn": number,
        "text": demo.DEMO_REPLY,
        "template_used": False,
        "audio_wav_b64": base64.b64encode(demo.SILENT_WAV).decode("ascii"),
    }
    return [
        (clock, _node_event(number, "reply", 0.4)),
        (clock, _node_event(number, "speak", 2.9)),
        (clock + 0.05, reply),
    ]
