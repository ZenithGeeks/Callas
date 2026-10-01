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

TURN_GAP = 0.4  # seconds of silence before the caller starts the next turn
PARTIAL_AT = (0.3, 0.6, 0.9)  # share of the turn text heard at each partial
SECONDS_PER_PARTIAL = 0.7
SCAM_TURN = 4
REPLY_AFTER_TURN = 3


class DemoLiveCall:
    def __init__(self, call_id: str | None = None, speed: float = 1.0) -> None:
        self.call_id = call_id or uuid.uuid4().hex
        self.bytes_fed = 0
        self._speed = speed
        self._paused = False
        self._ended = False
        self._ended_sent = False
        self._clock = 0.0
        self._last = time.monotonic()
        self._script = self._build_script()
        self._next = 0

    # ---- LiveCall interface -----------------------------------------------------------

    def feed(self, data: bytes) -> None:
        self.bytes_fed += len(data)

    def pause(self) -> None:
        self._tick()
        self._paused = True

    def resume(self) -> None:
        self._tick()
        self._paused = False

    def poll_events(self) -> list[dict]:
        self._tick()
        out: list[dict] = []
        if self._ended:
            if not self._ended_sent:
                self._ended_sent = True
                out.append({"type": "ended", "call_id": self.call_id})
            return out
        while self._next < len(self._script) and self._script[self._next][0] <= self._clock:
            out.append(self._script[self._next][1])
            self._next += 1
        return out

    def end(self) -> None:
        self._ended = True

    # ---- script -----------------------------------------------------------------------

    def _tick(self) -> None:
        now = time.monotonic()
        if not self._paused and not self._ended:
            self._clock += (now - self._last) * self._speed
        self._last = now

    def _build_script(self) -> list[tuple[float, dict]]:
        script: list[tuple[float, dict]] = [(0.0, {"type": "ready", "call_id": self.call_id})]
        ner = demo.DemoNER()
        seen: dict[tuple[str, str], dict] = {}
        context: list[str] = []
        escalated = False
        t = 0.5
        audio = base64.b64encode(demo.SILENT_WAV).decode("ascii")
        for number, (speaker, start, text) in enumerate(demo.DEMO_TURNS, start=1):
            for share in PARTIAL_AT:
                t += SECONDS_PER_PARTIAL
                script.append(
                    (t, {"type": "partial", "text": text[: max(1, int(len(text) * share))]})
                )
            t += SECONDS_PER_PARTIAL
            end = (
                demo.DEMO_TURNS[number][1] - TURN_GAP
                if number < len(demo.DEMO_TURNS)
                else demo.DEMO_DURATION
            )
            script.append(
                (
                    t,
                    {
                        "type": "turn",
                        "turn": number,
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "text": text,
                    },
                )
            )
            context.append(f"{speaker}: {text}")
            joined = "\n".join(context)
            for e in ner.extract(joined):
                seen.setdefault((e.text, e.label), asdict(e))
            label = demo._classify(joined)
            scam = label != "normal"
            reason = "benign"
            if scam and not escalated:
                reason = "high-confidence scam, escalate to user"
            elif escalated:
                reason = "call already escalated"
            escalated = escalated or scam
            script.append(
                (t, {"type": "node", "turn": number, "node": "remember", "data": {"seconds": 0.0}})
            )
            script.append(
                (t, {"type": "node", "turn": number, "node": "analyze", "data": {"seconds": 0.02}})
            )
            script.append(
                (
                    t,
                    {
                        "type": "node",
                        "turn": number,
                        "node": "decide",
                        "data": {"seconds": 0.0, "auto_reply": not escalated, "reason": reason},
                    },
                )
            )
            script.append(
                (
                    t,
                    {
                        "type": "verdict",
                        "turn": number,
                        "label_scam": scam,
                        "label_type": None if not scam else label,
                        "confidence": 1.0,
                        "escalated": escalated,
                        "reason": reason,
                        "entities": list(seen.values()),
                        "summary": demo._summary(joined),
                    },
                )
            )
            if number == REPLY_AFTER_TURN and not escalated:
                script.append(
                    (t, {"type": "node", "turn": number, "node": "reply", "data": {"seconds": 0.4}})
                )
                script.append(
                    (t, {"type": "node", "turn": number, "node": "speak", "data": {"seconds": 2.9}})
                )
                script.append(
                    (
                        t + 0.05,
                        {
                            "type": "reply",
                            "turn": number,
                            "text": demo.DEMO_REPLY,
                            "template_used": False,
                            "audio_wav_b64": audio,
                        },
                    )
                )
            elif escalated:
                script.append(
                    (
                        t,
                        {
                            "type": "node",
                            "turn": number,
                            "node": "escalate",
                            "data": {"seconds": 0.0},
                        },
                    )
                )
            t += TURN_GAP
        return script
