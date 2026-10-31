"""Transcript text cleanup shared by the web system and the classifier notebook.

normalize_text is the exact function the notebook trains with, so a model sees the same form
of text in training and in the web system. autocorrect_text is optional (AUTOCORRECT setting);
its last step, sentence repair, has its own setting (SPELL_REPAIR).
"""

from __future__ import annotations

import math
import re
import unicodedata
from functools import cache, lru_cache
from pathlib import Path

from pythainlp.util import normalize as thai_normalize

_TITLES = frozenset({"คุณ", "นาย", "นาง", "นางสาว", "พี่", "น้อง", "ดร", "ผู้กอง", "จ่า"})
# Thai letters, vowels and tone marks only: no digits (Thai or Arabic), Latin, URLs, symbols
# or the repeat mark ๆ (U+0E46).
_THAI_WORD = re.compile("[ก-ฺเ-ๅ็-๎]+")
# A vowel or tone mark written above or below a letter (ั ิ ี ึ ื ุ ู ฺ ็ ่ ้ ๊ ๋ ์ ...).
_MARK = re.compile("[ัิ-ฺ็-๎]")
# A consonant typed twice (ดด, มม) is a doubled letter, not a misplaced mark; รร is real Thai.
_DOUBLED = re.compile(r"([ก-ฮ])\1")
# Two different vowels on one letter (เงิืน): one was typed by mistake. Two of the same (เงิิน)
# are already removed by normalize_text.
_STACK = re.compile("([ัิีึื็])(?!\\1)[ัิีึื็]|([ฺุู])(?!\\2)[ฺุู]")
# The vowels and tone marks a typo can leave out: above or below a letter (คณ for คุณ).
_ADDABLE = "ัิีึืุู็่้๊๋์"
# Sentence repair (_repair): a mark may be swapped for another of its group (ได่ to ได้) as one
# edit; a span with more candidates than _REPAIR_MAX is left alone; a fix must beat the runner-up
# by 3x and the unchanged span by 10x (log probabilities).
_SWAP_GROUPS = ("ัิีึื็", "ุู", "่้๊๋")
_REPAIR_MAX = 8
_MARGIN = math.log(3)
_STRICT_MARGIN = math.log(10)
_TONE = re.compile("[่้๊๋]")
_CONSONANT = re.compile("[ก-ฮ]")
_ENGINE = "newmm"  # PyThaiNLP word tokenizer for a line (_words)
_COMMON = 1000  # a short piece may only become one of this many most common words (TNC)
# The masked language model that picks between stack fixes (กลับ or กลีบ), if torch is installed.
_MLM_TOKENIZER = "airesearch/wangchanberta-base-att-spm-uncased"
_MLM_MODEL = "bookpanda/wangchanberta-base-att-spm-uncased-masking"
# Loanwords the calls use that no PyThaiNLP word list has, so they are not "corrected".
_LOANWORDS = Path(__file__).resolve().parent / "loanwords.txt"


def normalize_text(text: str) -> str:
    """NFC, PyThaiNLP normalize (fixes doubled tone marks) and single spaces between words."""
    # "。" (ideographic full stop) turns up in some ASR output; treat it as a space
    text = unicodedata.normalize("NFC", text).replace("。", " ")
    return " ".join(thai_normalize(text).split())


@cache
def _dictionary() -> frozenset[str]:
    """The words a fix may produce: PyThaiNLP's main word list."""
    from pythainlp.corpus import thai_words

    return frozenset(thai_words())


@cache
def _loanwords() -> frozenset[str]:
    lines = _LOANWORDS.read_text(encoding="utf-8").splitlines()
    return frozenset(w for w in (line.strip() for line in lines) if w and not w.startswith("#"))


@cache
def _known() -> frozenset[str]:
    """Words left alone: the dictionary, the loanwords and slang in PyThaiNLP's Volubilis and ICU
    lists (พลัส, บ๊าย, เฮ้) and loanwords.txt. Fixes still produce only dictionary words."""
    from pythainlp.corpus import thai_icu_words, thai_volubilis_words

    return _dictionary() | thai_volubilis_words() | thai_icu_words() | _loanwords()


@cache
def _known_trie():
    """The known words for the tokenizer, so a loanword stays one piece (แอป, not แอ ป)."""
    from pythainlp.util import Trie

    return Trie(_known())


@cache
def _common_words() -> frozenset[str]:
    """The most common Thai words, by their count in the Thai National Corpus."""
    from pythainlp.corpus.tnc import word_freqs

    ranked = sorted(word_freqs(), key=lambda pair: -pair[1])
    return frozenset(word for word, _ in ranked[:_COMMON])


@cache
def _names() -> frozenset[str]:
    from pythainlp.corpus import thai_family_names, thai_female_names, thai_male_names

    return frozenset(thai_family_names() | thai_female_names() | thai_male_names())


@lru_cache(maxsize=4096)
def _fix_word(word: str) -> str:
    """The word with one misplaced vowel or tone mark moved back (คณุ to คุณ), else unchanged.

    Only marks move, so a fix never turns a word into a different-sounding one; the result must
    be the one dictionary word made that way, or split into fewer words, all known. Words
    PyThaiNLP's spell() would guess are left as they are: on stored calls its guesses were
    wrong far more often than right (โอนเงนิ to ถอนเงิน, แอป to แผน).
    """
    if len(word) < 3:  # one or two letters have too many near words to pick from
        return word
    if any(m.group(1) != "ร" for m in _DOUBLED.finditer(word)):  # ก็ดดี, not a moved mark
        return word
    from pythainlp.tokenize import word_tokenize

    words = _dictionary()
    pieces = len(word_tokenize(word, engine="newmm", keep_whitespace=False))

    def is_better(spelling: str) -> bool:
        if spelling in words:
            return True
        split = word_tokenize(spelling, engine="newmm", keep_whitespace=False)
        return len(split) < pieces and all(len(p) > 1 and p in words for p in split)

    swaps = {
        word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        for i in range(len(word) - 1)
        if _MARK.match(word[i]) or _MARK.match(word[i + 1])
    }
    found = {s for s in swaps if is_better(s)}
    return found.pop() if len(found) == 1 else word


def _stack_fixes(word: str) -> list[str]:
    """spell()'s suggestions that are the word with one of its two stacked vowels removed, most
    common first."""
    from pythainlp.spell import spell

    dropped = {
        word[: m.start() + k] + word[m.start() + k + 1 :]
        for m in _STACK.finditer(word)
        for k in (0, 1)
    }
    return [s for s in spell(word) if s in dropped]


@cache
def _mlm():
    """A scorer for how well a word fits between two strings, or None without torch/transformers."""
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_MLM_TOKENIZER)
        model = AutoModelForMaskedLM.from_pretrained(_MLM_MODEL).eval()
    except (ImportError, OSError):  # not installed, or the model cannot be downloaded
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def ids(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"] if s else []

    def score(left: str, word: str, right: str) -> float:
        # each of the word's tokens is masked in turn; the sum of their log probabilities
        before, middle, after = ids(left[-200:]), ids(word), ids(right[:200])
        seq = [tokenizer.cls_token_id, *before, *middle, *after, tokenizer.sep_token_id]
        start = 1 + len(before)
        batch = [
            seq[: start + k] + [tokenizer.mask_token_id] + seq[start + k + 1 :]
            for k in range(len(middle))
        ]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor(batch, device=device)).logits
            logp = torch.log_softmax(logits, -1)
        return float(sum(logp[k, start + k, t] for k, t in enumerate(middle)))

    return score


def _added_mark_fixes(word: str) -> list[str]:
    """spell()'s suggestions that are the word with one vowel or tone mark added (คณ to คุณ),
    most common first. One letter has too many near words, so it gets none. A word that already
    has a tone mark (ก้ for ก็, ต้ว for ตัว) has a wrong mark, not a missing one: adding one makes
    another word (กี้, ติ้ว)."""
    if len(word) < 2 or _TONE.search(word):
        return []
    from pythainlp.spell import spell

    added = {word[:i] + mark + word[i:] for i in range(1, len(word) + 1) for mark in _ADDABLE}
    fixes = [s for s in spell(word) if s in added]
    # A short piece (two consonants) is often a cut-off part of a longer word, and adding a mark
    # makes some rare word (วล to วลี, สท to สูท); the words a short typo really means are common
    # ones (คณ to คุณ, ไม to ไม่). A longer piece (พัสด) has only one sensible fix.
    if len(_CONSONANT.findall(word)) >= 3:
        return fixes
    return [s for s in fixes if s in _common_words()]


def _pick_fix(fixes: list[str], left: str, right: str) -> str:
    """The one fix, else the one the MLM scores highest in the sentence, else the most common."""
    score = _mlm() if len(fixes) > 1 else None
    if score is None:
        return fixes[0]
    return max(fixes, key=lambda f: score(left, f, right))


def _fix_stacks(text: str) -> str:
    """Remove the wrong one of two different vowels stacked on a letter (เงิืน to เงิน)."""
    if not _STACK.search(text):
        return text
    tokens = _words(text)
    fixed: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not _STACK.search(token):
            fixed.append(token)
            i += 1
            continue
        # the piece alone first, then together with its Thai neighbours (the tokenizer may cut it)
        before = fixed[-1] if fixed and _THAI_WORD.fullmatch(fixed[-1]) else ""
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ""
        after = next_token if _THAI_WORD.fullmatch(next_token) else ""
        for prev, nxt in (("", ""), (before, after)):
            fixes = _stack_fixes(prev + token + nxt)
            if fixes:
                break
        if not fixes:
            fixed.append(token)
            i += 1
            continue
        if prev:
            fixed.pop()
        right = "".join(tokens[i + 2 if nxt else i + 1 :])
        fixed.append(_pick_fix(fixes, "".join(fixed), right))
        i += 2 if nxt else 1
    return "".join(fixed)


def autocorrect_text(text: str, mode: str | None = None, repair: bool | None = None) -> str:
    """Fix two different vowels stacked on a letter, then misplaced vowels and tone marks in Thai
    words that are not known, then (repair) the marks of words still unknown, scored in their
    sentence. Leaves digits, English, names, URLs and known loanwords.

    mode is "pythainlp" or "off"; None reads the AUTOCORRECT setting, which is "off" by default.
    repair None reads the SPELL_REPAIR setting, which is "off" by default.
    """
    if mode is None:
        from callas_settings import get_settings

        mode = get_settings().autocorrect
    if mode != "pythainlp":
        return text
    text = _fix_stacks(text)
    text = "".join(_correct_tokens(_words(text)))
    if repair is None:
        from callas_settings import get_settings

        repair = get_settings().spell_repair == "on"
    return _repair(text) if repair else text


def _words(text: str) -> list[str]:
    """The line cut into words, spaces kept. A dictionary engine (newmm) gets the known words, so a
    loanword stays one piece; a neural one (attacut, deepcut) cannot take a word list."""
    from pythainlp.tokenize import word_tokenize

    if _ENGINE in ("newmm", "longest", "mm"):
        return word_tokenize(text, custom_dict=_known_trie(), engine=_ENGINE, keep_whitespace=True)
    return word_tokenize(text, engine=_ENGINE, keep_whitespace=True)


def _correct_tokens(tokens: list[str]) -> list[str]:
    """Fix each unknown Thai token: alone first, else together with its Thai neighbours."""
    known, names = _known(), _names()
    # `corrected` is the output so far. The previous Thai token is taken back out of it so it
    # can be corrected together with the unknown one; `i` skips the next token only when the
    # corrected word already includes it.
    corrected: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # a name the lists do not know is cut into several pieces after its title
        after_title = any(t in _TITLES for t in corrected[-4:])
        is_unknown_word = (
            bool(_THAI_WORD.fullmatch(token)) and token not in known and token not in names
        )
        if not is_unknown_word or after_title:
            corrected.append(token)
            i += 1
            continue

        # The piece alone first (ให้|คณุ|นะคะ), so its neighbours are never rewritten.
        alone = _fix_word(token)
        if alone != token:
            corrected.append(alone)
            i += 1
            continue

        # A misspelled word can be cut into pieces ("ขอบ" "ค" "นุ"), so try the unknown piece
        # together with the Thai pieces next to it.
        before = corrected.pop() if corrected and _THAI_WORD.fullmatch(corrected[-1]) else ""
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ""
        after = next_token if _THAI_WORD.fullmatch(next_token) else ""
        candidate = before + token + after
        fixed = _fix_word(candidate)
        if fixed != candidate:
            corrected.append(fixed)
            i += 2 if after else 1
            continue

        # Last, a vowel or tone mark left out (คณ to คุณ). With its neighbours first: the tokenizer
        # may have cut the word (กง|วล), and a fixed piece alone would be wrong (วล to วลี).
        # The piece alone only when its Thai neighbours are words: an unknown neighbour means the
        # word was cut into more pieces (บ|ตร), and fixing one of them makes another word (ตรู).
        whole = all(not p or p in known or p in names for p in (before, after))
        for prev, nxt in ((before, after), ("", "")) if whole else ((before, after),):
            fixes = _added_mark_fixes(prev + token + nxt)
            if fixes:
                break
        if not fixes:
            corrected.extend(piece for piece in (before, token) if piece)
            i += 1
            continue
        if before and not prev:
            corrected.append(before)
        right = "".join(tokens[i + 2 if nxt else i + 1 :])
        corrected.append(_pick_fix(fixes, "".join(corrected), right))
        i += 2 if nxt else 1
    return corrected


def _unknown(token: str) -> bool:
    return bool(_THAI_WORD.fullmatch(token)) and token not in _known() and token not in _names()


def _repair(text: str) -> str:
    """Fix the marks of Thai words still unknown, one sentence at a time (see _repair_sentence).
    Off without torch, and costs nothing on a line with no unknown word."""
    if not any(_unknown(t) for t in _words(text)) or _mlm() is None:
        return text
    from pythainlp.tokenize import sent_tokenize

    return "".join(
        _repair_sentence(s) for s in sent_tokenize(text, engine="crfcut", keep_whitespace=True)
    )


def _repair_sentence(sentence: str) -> str:
    """Each run of unknown pieces, with one Thai neighbour on each side (ให|คณ), is replaced by the
    mark edit of it the MLM scores best in the sentence, only if it beats the runner-up by 3x and
    the unchanged span by 10x. Pieces after a title (a name no list has) are left alone."""
    score = _mlm()
    tokens = _words(sentence)
    i = 0
    while i < len(tokens):
        if not _unknown(tokens[i]) or any(t in _TITLES for t in tokens[max(0, i - 4) : i]):
            i += 1
            continue
        j = i
        while j + 1 < len(tokens) and _unknown(tokens[j + 1]):
            j += 1
        lo = i - 1 if i > 0 and _THAI_WORD.fullmatch(tokens[i - 1]) else i
        hi = j + 1 if j + 1 < len(tokens) and _THAI_WORD.fullmatch(tokens[j + 1]) else j
        span = "".join(tokens[lo : hi + 1])
        fixes = _repair_candidates(span)
        if not fixes or len(fixes) > _REPAIR_MAX:
            i = j + 1
            continue
        left, right = "".join(tokens[:lo]), "".join(tokens[hi + 1 :])
        scored = sorted(((score(left, c, right), c) for c in [*fixes, span]), reverse=True)
        (best_score, best), (runner_up, _) = scored[0], scored[1]
        original = next(s for s, c in scored if c == span)
        if (
            best != span
            and best_score - runner_up >= _MARGIN
            and best_score - original >= _STRICT_MARGIN
        ):
            tokens[lo : hi + 1] = [best]
            i = lo + 1
        else:
            i = j + 1
    return "".join(tokens)


def _repair_candidates(span: str) -> list[str]:
    """The span's one-mark edits that cut into words, else its two-mark edits. A short span (two
    consonants or fewer) may only become common words: it is often a cut-off piece."""
    from pythainlp.tokenize import word_tokenize

    words = _common_words() if len(_CONSONANT.findall(span)) <= 2 else _dictionary()

    def is_words(s: str) -> bool:
        return all(p in words for p in word_tokenize(s, engine="newmm", keep_whitespace=False))

    one = _mark_edits(span)
    fixes = [s for s in one if is_words(s)]
    if fixes:
        return fixes
    two = {s2 for s in one for s2 in _mark_edits(s) if s2 != span}
    return [s for s in two if is_words(s)]


def _mark_edits(s: str) -> set[str]:
    """s with one vowel or tone mark added, removed, swapped for another of its group (ที้ to
    ที่) or moved past its neighbour."""
    edits = {s[:i] + mark + s[i:] for i in range(1, len(s) + 1) for mark in _ADDABLE}
    for i, c in enumerate(s):
        if c in _ADDABLE:
            edits.add(s[:i] + s[i + 1 :])
            for group in _SWAP_GROUPS:
                if c in group:
                    edits.update(s[:i] + m + s[i + 1 :] for m in group.replace(c, ""))
        if i + 1 < len(s) and (c in _ADDABLE or s[i + 1] in _ADDABLE):
            edits.add(s[:i] + s[i + 1] + c + s[i + 2 :])
    edits.discard(s)
    return edits
