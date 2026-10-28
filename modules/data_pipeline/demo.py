"""Demo mode: a scripted LM and silent voices, so the studio runs without a model server.

`fake_agents` builds real DSPy programs on a scripted LM, so demo mode still runs the whole
stack: prompt, parsing and callbacks. `SilentVoices` stands in for tts.Voices.
"""

from __future__ import annotations

import itertools
import time
import wave
from pathlib import Path

from dspy.utils import DummyLM

from .agents import SIGNATURES, Agents
from .config import Config

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
    if the call is long. Asking starts on the caller's third turn (scam_start_turn 2)."""
    answers = _scripted_answers(scam=call_type != "benign")
    lms = {name: ScriptedLM(answers[name], delay) for name in SIGNATURES}
    return Agents(config, lms=lms)


def _scripted_answers(scam: bool) -> dict[str, list[dict]]:
    """The answers each agent gives in turn, as the fields of its signature plus `reasoning`."""
    if scam:
        caller_answers = [
            dict(reasoning=f"step {i + 1} of the plan", line=line, asks_now=asks, end_call=False)
            for i, (line, asks) in enumerate(_SCAM_LINES)
        ]
    else:
        caller_answers = [
            dict(reasoning="keep it short and friendly", line=line, end_call=False)
            for line in _LEGIT_LINES
        ]
    receiver_answers = [
        dict(reasoning="answer like this person would", line=line, end_call=False)
        for line in (_REPLIES if scam else _LEGIT_REPLIES)
    ]
    judge = dict(
        reasoning="read the whole call against the intended type",
        scam_clear=True,
        scam_clarity_reason="The scripted transcript makes it clear whether there is a scam.",
        label_holds=True,
        label_reason="The scripted transcript follows its intended call type.",
        language_realistic=True,
        language_realism_reason="The demo fixture is concise and internally consistent.",
        setup_holds=True,
        setup_reason="The scripted caller and receiver follow their selected scenario and persona.",
        scam_start_turn=2 if scam else -1,
        giveaway=False,
        summary="ผู้โทรขอรหัสจากผู้รับสาย" if scam else "ผู้โทรแจ้งเรื่องการส่งพัสดุ",
        issues="none",
    )
    plan = dict(reasoning="a short plan that builds up to the ask", plan=_PLAN)
    extractor = dict(reasoning="-", relevant=False, summary="", tactics=[], phrases=[])
    return {
        "scam_caller": caller_answers,
        "legit_caller": caller_answers,
        "receiver": receiver_answers,
        "planner": [plan],
        "judge": [judge],
        "pattern_extractor": [extractor],
    }


class SilentVoices:
    """Stands in for tts.Voices in demo mode: writes a quiet WAV about as long as the line."""

    def speak(self, text: str, voice: str, path: Path) -> float:
        """Write the quiet WAV to `path` and return its length in seconds."""
        seconds = max(0.5, round(len(text) * 0.07, 2))
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(bytes(2 * int(16000 * seconds)))
        return seconds
