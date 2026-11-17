"""Demo mode (CALLAS_FAKE=1): scripted parts so the whole UI runs with no models or GPU.

Nothing here is used in real mode. The scripted answers go through the real orchestrator
graphs: a FakeLLM behind the registry's LLM classifier and summarizer, a fake speech-to-text
pipeline, and a fake speaker that returns a short silent WAV.
"""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace

from asr_diarization.types import Transcript, Turn
from llm_serving.fake import FakeLLM
from nlp_core import registry
from nlp_core.entities import extract_entities
from nlp_core.registry import LLMClassifier, LLMSummarizer
from nlp_core.types import Entity

DEMO_FILENAME = "call_parcel_police_07.wav"

# (speaker, start seconds, text). Like any recording, A is the person being called
# (views.CALLED_PERSON) and B is the caller. Generated calls in the dataset use the opposite
# letters (dataset.GENERATED_CALLER).
DEMO_TURNS: tuple[tuple[str, float, str], ...] = (
    ("B", 0.0, "สวัสดีครับ ผมโทรจากบริษัทขนส่งนะครับ มีพัสดุส่งถึงคุณครับ"),
    ("A", 6.0, "ค่ะ พัสดุอะไรคะ ไม่ได้สั่งอะไรไว้เลย"),
    ("B", 11.0, "เป็นพัสดุจากต่างประเทศครับ ตรวจพบของผิดกฎหมายอยู่ในกล่อง"),
    (
        "B",
        19.0,
        "ผมจะโอนสายให้ร้อยตำรวจโทวิชัย กรุณาโอนเงิน 50,000 บาท เข้าบัญชี 123-4-56789-0 เพื่อตรวจสอบครับ",
    ),
    ("A", 33.0, "แล้วต้องทำยังไงต่อคะ"),
    ("B", 37.0, "โทรกลับเบอร์ 081-234-5678 ภายในวันนี้ ไม่อย่างนั้นจะถูกดำเนินคดีครับ"),
    ("A", 49.0, "ขอคิดดูก่อนนะคะ"),
)
DEMO_DURATION = 58.0
TURN_GAP = 0.4  # seconds of silence between the end of a turn and the start of the next

DEMO_REPLY = "ขอบคุณที่โทรมาค่ะ ขณะนี้ไม่สะดวกคุยค่ะ"

# Literal entities the interim rule NER cannot find (no digits): the demo NER adds them.
_LITERALS: tuple[tuple[str, str], ...] = (
    ("บริษัทขนส่ง", "ORG"),
    ("ร้อยตำรวจโทวิชัย", "PERSON"),
    ("รหัส OTP", "OTP"),
)


def silent_wav(seconds: float = 1.4, rate: int = 16_000) -> bytes:
    """A mono 16-bit WAV file of silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


SILENT_WAV = silent_wav()


def turn_end(index: int) -> float:
    """When turn `index` (0-based) ends: TURN_GAP before the next turn, or the call's end."""
    if index + 1 < len(DEMO_TURNS):
        return DEMO_TURNS[index + 1][1] - TURN_GAP
    return DEMO_DURATION


def fake_pipeline_run(audio_path: str, *args, **kwargs) -> Transcript:
    """Stand-in for asr_diarization.pipeline.run: the same 7-turn call for any file."""
    turns = [
        Turn(speaker, start, turn_end(i), text)
        for i, (speaker, start, text) in enumerate(DEMO_TURNS)
    ]
    return Transcript(turns=turns, model="demo", model_version="scripted")


class FakeSpeaker:
    """Stands in for tts_voice_agent.speech.Speaker: every reply is a short silent WAV."""

    def synthesize_bytes(self, text: str, check_leaks: bool = True) -> bytes:
        return SILENT_WAV


class DemoNER:
    """Interim rule NER plus a few literal Thai names and organisations."""

    def extract(self, text: str) -> list[Entity]:
        found = extract_entities(text)
        for literal, label in _LITERALS:
            start = text.find(literal)
            if start < 0:
                continue
            end = start + len(literal)
            if all(end <= e.start or start >= e.end for e in found):
                found.append(Entity(literal, label, start, end))
        return sorted(found, key=lambda e: e.start)


def classify_call(context: str) -> str:
    """The scam type the demo finds in the call so far, or "normal"."""
    if "OTP" in context:
        return "phishing_otp"
    if "โอนเงิน" in context or "ร้อยตำรวจ" in context:
        return "fake_authority"
    return "normal"


def summarize_call(context: str) -> str:
    """The Thai summary the demo gives for the call so far."""
    if "OTP" in context:
        return "ข้อความอ้างว่าบัญชีถูกระงับ ให้ยืนยันตัวตนผ่านลิงก์ หรือส่งรหัส OTP กลับไป"
    if "ดำเนินคดี" in context:
        return (
            "ผู้โทรอ้างว่าเป็นบริษัทขนส่ง แล้วโอนสายให้ผู้ที่อ้างว่าเป็นตำรวจ "
            "ขอให้โอนเงินเข้าบัญชีเพื่อ “ตรวจสอบ” และขู่ว่าจะถูกดำเนินคดี"
        )
    if "โอนเงิน" in context:
        return "ผู้โทรอ้างว่าเป็นบริษัทขนส่ง แล้วโอนสายให้ผู้ที่อ้างว่าเป็นตำรวจ ขอให้โอนเงินเข้าบัญชี"
    return "ผู้โทรอ้างว่าเป็นบริษัทขนส่ง และแจ้งว่ามีพัสดุส่งถึง"


def make_llm() -> FakeLLM:
    """The scripted LLM behind the demo classifier and summarizer."""
    return FakeLLM({"classify": classify_call, "summary": summarize_call, "reply": DEMO_REPLY})


def register_demo_parts(llm: FakeLLM) -> None:
    """Register the demo NER, classifier and summarizer under the name "demo"."""
    registry.register("ner", "demo", DemoNER)
    registry.register("classifier", "demo", lambda: LLMClassifier(llm))
    registry.register("summarizer", "demo", lambda: LLMSummarizer(llm))


def fake_agent(text: str) -> SimpleNamespace:
    """Looks like a langgraph agent: invoke({"messages": ...}) -> {"messages": [msg]}."""
    if "OTP" in text:
        why = (
            "ข้อความอ้างว่าบัญชีถูกระงับเพื่อเร่งให้รีบทำ ให้กดลิงก์ที่ไม่ใช่เว็บไซต์ธนาคาร "
            "และขอรหัส OTP ซึ่งธนาคารไม่เคยขอทางข้อความ"
        )
    elif classify_call(text) != "normal":
        why = "ข้อความนี้มีลักษณะของการหลอกลวง เช่น การขอให้โอนเงินหรือให้รีบดำเนินการ"
    else:
        why = "ไม่พบสัญญาณของการหลอกลวงในข้อความนี้ ไม่มีการขอเงิน รหัส หรือลิงก์"
    msg = SimpleNamespace(content=why)
    return SimpleNamespace(invoke=lambda _inputs: {"messages": [msg]})
