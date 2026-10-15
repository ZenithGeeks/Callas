"""Facts about now that a reply may state: the date and time in Thai."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_DAYS = "จันทร์ อังคาร พุธ พฤหัสบดี ศุกร์ เสาร์ อาทิตย์".split()
_MONTHS = (
    "มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม"
).split()


def current_facts(now: datetime | None = None) -> str:
    """Today's date and time in Thai (Bangkok time, Buddhist year), for the reply prompt."""
    now = now or datetime.now(ZoneInfo("Asia/Bangkok"))
    return (
        f"วันนี้วัน{_DAYS[now.weekday()]}ที่ {now.day} {_MONTHS[now.month - 1]} "
        f"{now.year + 543} เวลา {now:%H:%M} น."
    )


def fact_numbers(now: datetime | None = None) -> set[str]:
    """Numbers a reply may contain: those of current_facts(), allowing the minute to tick over."""
    now = now or datetime.now(ZoneInfo("Asia/Bangkok"))
    return {
        n
        for delta in (-1, 0, 1)
        for n in re.findall(r"\d+", current_facts(now + timedelta(minutes=delta)))
    }
