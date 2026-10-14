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
REPLY_DELAY_S = 0.7  # extra quiet after a turn closes; a turn starting inside it is merged


class LiveCall:
    def __init__(self, sessions: CallSessions, transcriber: LiveTranscriber | None = None) -> None:
        self.sessions = sessions
        self.transcriber = transcriber or LiveTranscriber()
        self.call_id = sessions.start()
        self._events: queue.Queue[dict] = queue.Queue()
        self._turns: queue.Queue[tuple[int, Turn] | None] = queue.Queue()
        self._turn_no = 0
        self._held: tuple[int, Turn | None] | None = None
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

    def _next_turn(self) -> tuple[int, Turn, list[int]] | None:
        """Next turn to process. Waits REPLY_DELAY_S for the caller to go on and merges that."""
        item, self._held = (self._held, None) if self._held else (self._turns.get(), None)
        if item is None or item[1] is None:
            return None
        n, turn = item
        merged: list[int] = []  # numbers of the turns folded into this one
        while True:
            try:
                nxt = self._turns.get(timeout=REPLY_DELAY_S)
            except queue.Empty:
                return n, turn, merged
            if nxt is None:
                self._held = (-1, None)  # process this turn first, then stop
                return n, turn, merged
            if nxt[1].speaker != turn.speaker:
                self._held = nxt  # not a continuation; handled on the next call
                return n, turn, merged
            text = f"{turn.text} {nxt[1].text}"
            merged.append(n)
            n, turn = nxt[0], Turn(turn.speaker, turn.start, nxt[1].end, text)

    def _run(self) -> None:
        while (item := self._next_turn()) is not None:
            n, turn, merged = item
            try:
                self._process(n, turn, merged)
            except Exception as exc:  # keep the call alive; never put call text in the message
                log.warning("turn %d failed (%s)", n, type(exc).__name__)
                self._events.put(
                    {"type": "error", "turn": n, "message": f"{type(exc).__name__}: turn failed"}
                )

    def _process(self, n: int, turn: Turn, merged: list[int]) -> None:
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
                verdict = {
                    "type": "verdict",
                    "label_scam": nlp.label_scam,
                    "label_type": nlp.label_type,
                    "confidence": nlp.confidence,
                    "escalated": not decision.auto_reply,
                    "reason": decision.reason,
                    "entities": [to_jsonable(e) for e in nlp.entities],
                    "summary": nlp.summary,
                }
                for turn_no in (*merged, n):  # folded turns share the merged turn's verdict
                    put({**verdict, "turn": turn_no})
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
