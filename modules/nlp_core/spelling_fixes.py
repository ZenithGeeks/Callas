"""Thai words the simulator's model garbles, each checked by a person: wrong form -> right form.

Used by nlp_core.text.fix_thai_spelling after PyThaiNLP normalize, so the wrong forms here are
already normalized (no doubled marks). A wrong form is replaced wherever it appears, so it must
never occur inside a real word: คัณ (in คัณฑมาลา), ที้ (in เที้ยร), ถิง (in เถิง) and ล๊อค
(in บล๊อค) are left out for that reason, and short forms such as คณ, ไม, ให or หนุ่ cannot be
listed at all. test_text checks both rules against the PyThaiNLP dictionary.

Found in data/synthetic/calls.db: the Thai tokens still outside the dictionary after normalize.
"""

from __future__ import annotations

SPELLING_FIXES: dict[str, str] = {
    "คณุ": "คุณ",
    "คุ้ณ": "คุณ",
    "ใหคณุ": "ให้คุณ",
    "ใหคณ": "ให้คุณ",
    "ดวย": "ด้วย",
    "ถุก": "ถูก",
    "คร่ะ": "ค่ะ",
    "ให่": "ให้",
    "แลว้": "แล้ว",
    "กลบั": "กลับ",
    "ร้จัก": "รู้จัก",
    "พัดุ": "พัสดุ",
    "พัสสดุ": "พัสดุ",
    "พสดุ": "พัสดุ",
    "ปลอดภยั": "ปลอดภัย",
    "เปลี่น": "เปลี่ยน",
    "ขึ่น": "ขึ้น",
    "คดี่": "คดี",
    "ระหว่ง": "ระหว่าง",
    "งั้ณ": "งั้น",
    "งั้บ": "งั้น",
    "เดี๋ยย": "เดี๋ยว",
    "ที๋": "ที",
    "กั๋น": "กัน",
    "มั่ย": "มั้ย",
    "บตรประชน": "บัตรประชาชน",
    "ไซตต์": "ไซต์",
    "แอปพิเคชัน": "แอปพลิเคชัน",
    "ล๊อก": "ล็อก",
}
