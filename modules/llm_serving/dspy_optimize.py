"""Offline optimizer for the DSPy reply writer. Run by hand against your own vLLM server:

    python -m llm_serving.dspy_optimize --out modules/llm_serving/compiled/reply.dspy.json
    python -m llm_serving.dspy_optimize --out reply.dspy.json --trainset my_contexts.jsonl

It runs BootstrapFewShot over redacted call contexts and saves the compiled program, which
DSPyReplyWriter loads on start. Trainset rows are JSON lines: {"call_context": "...",
"verdict": "benign"}. The metric is code, not a model: a reply scores 1.0 only if it is
non-empty, short, mostly Thai and passes the same rule leak check as production.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dspy

from tts_voice_agent.constants import MAX_REPLY_CHARS
from tts_voice_agent.leak import leaks_sensitive
from tts_voice_agent.spelling import looks_misspelled

from .dspy_programs import COMPILED_REPLY, WriteReply, build_lm

MIN_THAI_SHARE = 0.8
MAX_BOOTSTRAPPED_DEMOS = 3
DEV_SIZE = 4

# SYNTHETIC: invented, already redacted, benign call contexts. No real call text.
SYNTHETIC_CONTEXTS: tuple[str, ...] = (
    "A: สวัสดีครับ มีพัสดุของคุณ [PERSON] มาส่ง แต่ไม่มีคนอยู่บ้าน",
    "A: ผมเป็นพนักงานส่งของ จะนำพัสดุไปส่งที่บ้านช่วงบ่าย",
    "A: พัสดุตีกลับมาที่ [ORG] ครับ ช่วยโทรกลับด้วย",
    "A: แม่ฝากบอกว่าให้โทรกลับหน่อย มีเรื่องที่บ้าน",
    "A: ป้าโทรมาถามว่าวันอาทิตย์จะมากินข้าวที่บ้านไหม",
    "A: น้องชายอยากคุยด้วย เรื่องไปเที่ยวด้วยกัน",
    "A: โทรมาจากคลินิกฟัน ขอเลื่อนนัดตรวจฟัน",
    "A: ทางโรงเรียนโทรมาแจ้งเรื่องประชุมผู้ปกครองสัปดาห์หน้า",
    "A: ช่างแอร์โทรมานัดวันเข้าไปล้างแอร์ที่บ้าน",
    "A: สนใจประกันสุขภาพไหมคะ มีโปรโมชั่นพิเศษเดือนนี้",
    "A: ขอเสนอสินค้าใหม่ เครื่องกรองน้ำ ลดราคาพิเศษ",
    "A: โทรมาสำรวจความพึงพอใจการใช้บริการอินเทอร์เน็ต ใช้เวลาไม่นานค่ะ",
    "A: ร้านอาหารโทรมายืนยันการจองโต๊ะวันเสาร์นี้",
    "A: เพื่อนเก่าโทรมาชวนไปงานเลี้ยงรุ่น ฝากให้โทรกลับ",
)


def load_trainset(path: Path | None) -> list[dspy.Example]:
    if path is None:
        rows = [{"call_context": context, "verdict": "benign"} for context in SYNTHETIC_CONTEXTS]
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [
        dspy.Example(
            call_context=row["call_context"], verdict=row.get("verdict", "benign")
        ).with_inputs("call_context", "verdict")
        for row in rows
    ]


def is_thai_enough(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    thai = sum(1 for char in letters if "฀" <= char <= "๿")
    return thai / len(letters) >= MIN_THAI_SHARE


def reply_metric(example, pred, trace=None) -> float:
    """1.0 only for a usable reply: non-empty, short, Thai, and clean by the rule check."""
    text = (getattr(pred, "reply", "") or "").strip()
    ok = (
        bool(text)
        and len(text) <= MAX_REPLY_CHARS
        and is_thai_enough(text)
        and not leaks_sensitive(text, [])
        and not looks_misspelled(text)
    )
    return 1.0 if ok else 0.0


def optimize(trainset: list[dspy.Example], out: Path, lm=None) -> tuple[float, float]:
    """Compile on the trainset (minus a dev slice), save to `out`, return (before, after) as
    dspy.Evaluate scores (percent, 0 to 100)."""
    if len(trainset) <= DEV_SIZE:
        raise ValueError(
            f"need more than {DEV_SIZE} examples (the last {DEV_SIZE} are the dev set)"
        )
    train, dev = trainset[:-DEV_SIZE], trainset[-DEV_SIZE:]
    lm = lm or build_lm()
    with dspy.context(lm=lm):
        student = dspy.ChainOfThought(WriteReply)
        evaluate = dspy.Evaluate(devset=dev, metric=reply_metric, num_threads=1)
        before = float(evaluate(student).score)
        tuned = dspy.BootstrapFewShot(
            metric=reply_metric, max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS
        ).compile(student, trainset=train)
        after = float(evaluate(tuned).score)
    out.parent.mkdir(parents=True, exist_ok=True)
    tuned.save(str(out))
    return before, after


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=COMPILED_REPLY)
    parser.add_argument("--trainset", type=Path, default=None, help="JSONL; default: synthetic set")
    args = parser.parse_args(argv)
    before, after = optimize(load_trainset(args.trainset), args.out)
    print(f"dev score before: {before:.1f}%")
    print(f"dev score after:  {after:.1f}%")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
