"""Fixed fallback replies.

TEMPLATES NEED TEAM SIGN-OFF (spec: "Ask first: the backup templates"). They must promise
nothing, ask nothing of substance, and never confirm who answered or that the owner is there.
"""

from __future__ import annotations

from nlp_core.types import SCAM_TYPES, NLPResult

from .leak import leaks_sensitive

TEMPLATES: tuple[str, ...] = (
    "ขณะนี้ไม่สะดวกรับสายค่ะ กรุณาติดต่อกลับภายหลังค่ะ",
    "ขออภัยค่ะ ตอนนี้ยังไม่สะดวกคุยค่ะ",
    "ขอบคุณที่โทรมาค่ะ ขณะนี้ไม่สะดวกคุยค่ะ",
    "ขออภัยค่ะ ไม่สะดวกรับสายในขณะนี้ค่ะ",
)

for _t in TEMPLATES:
    assert not leaks_sensitive(_t, []), f"template fails leak check: {_t!r}"


def pick_template(result: NLPResult) -> str:
    """Deterministic: chosen by label_type (no randomness)."""
    t = result.label_type
    idx = SCAM_TYPES.index(t) % len(TEMPLATES) if t in SCAM_TYPES else 0
    return TEMPLATES[idx]
