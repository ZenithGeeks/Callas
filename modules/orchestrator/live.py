"""One live call: audio in, JSON events out. Closed turns go through the turn graph in order."""

from __future__ import annotations

import base64
import logging
import queue
import threading

from asr_diarization.streaming import LiveTranscriber
from asr_diarization.types import Turn
from nlp_core.types import NLPResult
from tts_voice_agent.decision import Decision

from .sessions import CallSessions, to_jsonable

log = logging.getLogger(__name__)

END_TIMEOUT_S = 30.0  # how long end() waits for queued turns
REPLY_DELAY_S = 0.7  # extra quiet after a turn closes; a turn starting inside it is merged
STOP_MARKER = (-1, None)  # put in LiveCall._held when end() arrived while turns were merging


class LiveCall:
    """One live call. Audio goes in with feed(); JSON events come out of poll_events().

    A single worker thread runs the closed turns through the turn graph in order, so slow LLM
    or TTS work never blocks audio intake.
    """

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
        """Add PCM16 mono 16 kHz audio. Ignored while paused or after end()."""
        if self._paused or self._ended:
            return
        self._handle(self.transcriber.feed(pcm))

    def _handle(self, events: list[dict]) -> None:
        """Queue transcriber events for the client, and each closed turn for the worker."""
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
        """Take audio again after pause()."""
        self._paused = False

    def poll_events(self, max_items: int = 100) -> list[dict]:
        """Return up to `max_items` queued events without waiting."""
        out: list[dict] = []
        while len(out) < max_items:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def end(self) -> None:
        """Close the open turn, wait for queued turns, delete call memory. Safe to call twice."""
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

    # _next_turn returns (number, turn, folded numbers), or None when the call is over.
    # Queue items: (number, turn), or None from end(). If end() arrives while merging, the
    # current turn is returned first and STOP_MARKER is held so the next call returns None.
    def _next_turn(self) -> tuple[int, Turn, list[int]] | None:
        """Next turn to process. Waits REPLY_DELAY_S for the caller to go on and merges that."""
        item = self._held or self._turns.get()  # a held turn goes first
        self._held = None
        if item is None or item[1] is None:  # end() sentinel, or the held stop marker
            return None
        n, turn = item
        merged: list[int] = []  # numbers of the turns folded into this one
        while True:
            try:
                following = self._turns.get(timeout=REPLY_DELAY_S)
            except queue.Empty:
                return n, turn, merged
            if following is None:
                self._held = STOP_MARKER  # process this turn first, then stop
                return n, turn, merged
            following_n, following_turn = following
            if following_turn.speaker != turn.speaker:
                self._held = following  # not a continuation; handled on the next call
                return n, turn, merged
            text = f"{turn.text} {following_turn.text}"
            merged.append(n)
            n, turn = following_n, Turn(turn.speaker, turn.start, following_turn.end, text)

    def _run(self) -> None:
        """Worker thread: run turns in order until end(). A failed turn is reported, not fatal."""
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
        """Run one turn through the graph and queue its events: node, verdict, reply."""
        put = self._events.put
        nlp = None
        reply_update: dict | None = None
        audio: bytes | None = None
        for item in self.sessions.stream_turn(self.call_id, turn):
            node, update = item["node"], item["update"]
            put({"type": "node", "turn": n, "node": node, "data": to_jsonable(update)})
            if node == "analyze":
                nlp = update["nlp"]
            elif node == "decide":
                verdict = _verdict_event(nlp, update["decision"])
                for turn_no in (*merged, n):  # folded turns share the merged turn's verdict
                    put({**verdict, "turn": turn_no})
            elif node == "reply":
                reply_update = update
            elif node == "speak":
                audio = update.get("audio")
        if reply_update is not None:
            put(
                {
                    "type": "reply",
                    "turn": n,
                    "text": reply_update["reply_text"],
                    "template_used": reply_update["template_used"],
                    "audio_wav_b64": base64.b64encode(audio).decode("ascii") if audio else None,
                }
            )


def _verdict_event(nlp: NLPResult, decision: Decision) -> dict:
    """The verdict event for a turn, without its turn number."""
    return {
        "type": "verdict",
        "label_scam": nlp.label_scam,
        "label_type": nlp.label_type,
        "confidence": nlp.confidence,
        "escalated": not decision.auto_reply,
        "reason": decision.reason,
        "entities": [to_jsonable(e) for e in nlp.entities],
        "summary": nlp.summary,
    }
