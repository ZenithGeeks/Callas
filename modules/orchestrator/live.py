"""One live call: audio in, JSON events out. Closed turns go through the turn graph in order."""

from __future__ import annotations

import base64
import logging
import queue
import threading

from asr_diarization.streaming import LiveTranscriber
from asr_diarization.types import Turn

from .sessions import CallSessions, to_jsonable

log = logging.getLogger(__name__)

END_TIMEOUT_S = 30.0  # how long end() waits for queued turns


class LiveCall:
    def __init__(self, sessions: CallSessions, transcriber: LiveTranscriber | None = None) -> None:
        self.sessions = sessions
        self.transcriber = transcriber or LiveTranscriber()
        self.call_id = sessions.start()
        self._events: queue.Queue[dict] = queue.Queue()
        self._turns: queue.Queue[tuple[int, Turn] | None] = queue.Queue()
        self._turn_no = 0
        self._paused = False
        self._ended = False
        self._lock = threading.Lock()
        self._events.put({"type": "ready", "call_id": self.call_id})
        self._worker = threading.Thread(target=self._run, name="live-call", daemon=True)
        self._worker.start()

    def feed(self, pcm: bytes) -> None:
        if self._paused or self._ended:
            return
        self._handle(self.transcriber.feed(pcm))

    def _handle(self, events: list[dict]) -> None:
        for ev in events:
            self._events.put(ev)
            if ev["type"] == "turn":
                self._turn_no += 1
                turn = Turn(ev["speaker"], ev["start"], ev["end"], ev["text"])
                self._turns.put((self._turn_no, turn))

    def pause(self) -> None:
        """Drop audio until resume(); an open turn is closed first."""
        if self._ended:
            return
        if not self._paused:
            self._handle(self.transcriber.flush())
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def poll_events(self, max_items: int = 100) -> list[dict]:
        out: list[dict] = []
        while len(out) < max_items:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def end(self) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
        self._handle(self.transcriber.flush())
        self._turns.put(None)
        self._worker.join(END_TIMEOUT_S)
        if self._worker.is_alive():
            log.warning("live call worker still busy at end")
        self.sessions.end(self.call_id)
        self._events.put({"type": "ended", "call_id": self.call_id})

    def _run(self) -> None:
        while True:
            item = self._turns.get()
            if item is None:
                return
            n, turn = item
            try:
                self._process(n, turn)
            except Exception as exc:  # keep the call alive; never put call text in the message
                log.warning("turn %d failed (%s)", n, type(exc).__name__)
                self._events.put(
                    {"type": "error", "turn": n, "message": f"{type(exc).__name__}: turn failed"}
                )

    def _process(self, n: int, turn: Turn) -> None:
        put = self._events.put
        nlp = None
        reply: dict | None = None
        audio: bytes | None = None
        for item in self.sessions.stream_turn(self.call_id, turn):
            node, update = item["node"], item["update"]
            put({"type": "node", "turn": n, "node": node, "data": to_jsonable(update)})
            if node == "analyze":
                nlp = update["nlp"]
            elif node == "decide":
                decision = update["decision"]
                put(
                    {
                        "type": "verdict",
                        "turn": n,
                        "label_scam": nlp.label_scam,
                        "label_type": nlp.label_type,
                        "confidence": nlp.confidence,
                        "escalated": not decision.auto_reply,
                        "reason": decision.reason,
                        "entities": [to_jsonable(e) for e in nlp.entities],
                        "summary": nlp.summary,
                    }
                )
            elif node == "reply":
                reply = {"text": update["reply_text"], "template_used": update["template_used"]}
            elif node == "speak":
                audio = update.get("audio")
        if reply is not None:
            put(
                {
                    "type": "reply",
                    "turn": n,
                    "text": reply["text"],
                    "template_used": reply["template_used"],
                    "audio_wav_b64": base64.b64encode(audio).decode("ascii") if audio else None,
                }
            )
