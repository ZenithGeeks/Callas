"""The fine-tuned PhayaThaiBERT scam classifier from notebooks/scam_classifier_phayathaibert.ipynb.

Registered as "phayathaibert" in nlp_core.registry. Select it with NLP_CLASSIFIER=phayathaibert.
It needs the `ml` extra (torch, transformers) and the models the notebook saves, which are not in
git: models/phayathaibert-scam-gate (required) and models/phayathaibert-scam-type (optional).
NLP_MODELS_DIR points at another folder that holds them.

The models learned from rows of the notebook's step 6b: the call so far, one "A: ..." or
"B: ..." line per turn (A the person called, B the caller), with sensitive values masked as
"[PHONE]", "[MONEY]", ... `model_row` rewrites the text the app sends into that shape.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from pathlib import Path

from .entities import extract_entities
from .text import normalize_text

GATE_DIR_NAME = "phayathaibert-scam-gate"
TYPE_DIR_NAME = "phayathaibert-scam-type"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Starting values from the notebook (step 14 and the start of part B).
GATE_MIN_CONFIDENCE = 0.60  # below this the gate is unsure: (True, None, 0.0)
TYPE_MIN_CONFIDENCE = 0.50  # below this the scam type is left as None

# The letters the models know. In the app A is the person called (backend.views.CALLED_PERSON).
CALLED = "A"
CALLER = "B"
# Callas speaks for the person called, so its replies are that person's lines. Any other
# speaker (B, the live page's "Caller", a third voice found by diarization) is the caller side.
CALLED_SPEAKERS = frozenset({"A", "agent"})

# The values the notebook masks (step 1), the same set as backend.views.SENSITIVE_LABELS.
SENSITIVE_LABELS = frozenset(
    {"PHONE", "BANK_ACCOUNT", "OTP", "MONEY", "URL", "LINE_ID", "CARD_NUMBER", "ID_CARD",
     "BIRTH_DATE", "ADDRESS"}
)  # fmt: skip

# "A: ...", "agent: ...", "Caller: ...". ASCII only, so a Thai word before a colon is not a speaker.
_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s?(.*)$")


def mask(text: str) -> str:
    """Replace each sensitive value found by the entity rules with "[LABEL]"."""
    spans = sorted(
        (e for e in extract_entities(text) if e.label in SENSITIVE_LABELS),
        key=lambda e: e.start,
        reverse=True,
    )
    after = len(text)  # values are masked from the end, so earlier offsets stay valid
    for e in spans:
        if e.end > after:  # overlaps a value already masked
            continue
        text = f"{text[: e.start]}[{e.label}]{text[e.end :]}"
        after = e.start
    return text


def turns_of(text: str) -> list[tuple[str, str]]:
    """The text as (letter, line text) pairs, the way the models were trained.

    A line "speaker: text" becomes one turn. Lines without a speaker (the summary of older turns
    that call memory puts first) are left out: the models never saw one, and they only ever read
    the newest lines of a long call. A text with no speaker line at all (a text message) is one
    caller turn. Turns of the same letter in a row are joined, as in the notebook's cleaning.
    """
    turns: list[tuple[str, str]] = []
    lines = [m for line in text.splitlines() if (m := _LINE.match(line.strip()))]
    if not lines:
        pairs = [(CALLER, text)]
    else:
        pairs = [(CALLED if m[1] in CALLED_SPEAKERS else CALLER, m[2]) for m in lines]
    for letter, line in pairs:
        line = mask(normalize_text(line))
        if not line:
            continue
        if turns and turns[-1][0] == letter:
            turns[-1] = (letter, f"{turns[-1][1]} {line}")
        else:
            turns.append((letter, line))
    return turns


def model_row(text: str, count_tokens: Callable[[str], int], max_tokens: int) -> str:
    """The row the models read for `text`: its turns as "X: line" lines, the oldest dropped until
    `count_tokens` of the row is at most `max_tokens`. The newest line always stays: the models
    were trained on the call so far, and the newest line is the one a row's label is about."""
    lines = [f"{letter}: {line}" for letter, line in turns_of(text)]
    first = 0
    while first < len(lines) - 1 and count_tokens("\n".join(lines[first:])) > max_tokens:
        first += 1
    return "\n".join(lines[first:])


def to_model_text(text: str, space_token: str | None) -> str:
    """The notebook's to_model_text: every run of whitespace, newlines between turns too,
    becomes one space, and that space becomes the space token."""
    text = " ".join(text.split())
    return text.replace(" ", space_token) if space_token else text


def answer(
    gate: dict[str, float],
    types: dict[str, float] | None,
    gate_min: float = GATE_MIN_CONFIDENCE,
    type_min: float = TYPE_MIN_CONFIDENCE,
) -> tuple[bool, str | None, float]:
    """(label_scam, label_type, confidence) from the gate's and the type model's probabilities
    by label. An unsure gate gives (True, None, 0.0): never "safe" when unsure."""
    label, confidence = max(gate.items(), key=lambda kv: kv[1])
    if confidence < gate_min:
        return True, None, 0.0
    if label != "scam":
        return False, None, confidence
    label_type = None
    if types:
        best, p = max(types.items(), key=lambda kv: kv[1])
        label_type = best if p >= type_min else None
    return True, label_type, confidence


class PhayaThaiBertClassifier:
    """nlp_core ScamClassifier on the notebook's gate model, and its type model if there is one."""

    def __init__(self, models_dir: Path | str | None = None, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        models = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        gate_dir, type_dir = models / GATE_DIR_NAME, models / TYPE_DIR_NAME
        if not (gate_dir / "config.json").exists():
            raise FileNotFoundError(
                f"no PhayaThaiBERT gate model in {gate_dir}. Run "
                "notebooks/scam_classifier_phayathaibert.ipynb to train it, or set NLP_MODELS_DIR "
                "to the folder that holds it."
            )
        pre = json.loads((gate_dir / "preprocessing.json").read_text(encoding="utf-8"))
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(gate_dir)
        self.max_length = int(pre["max_length"])
        space = pre.get("space_token")
        self.space_token = space if space and space in self.tokenizer.get_vocab() else None
        self.gate = self._load(AutoModelForSequenceClassification, gate_dir)
        self.type = (
            self._load(AutoModelForSequenceClassification, type_dir)
            if (type_dir / "config.json").exists()
            else None
        )
        self._lock = threading.Lock()  # one model, called from web requests and call workers

    def _load(self, auto, path: Path):
        return auto.from_pretrained(path).to(self.device).eval()

    def count_tokens(self, row: str) -> int:
        return len(self.tokenizer(to_model_text(row, self.space_token))["input_ids"])

    def row(self, text: str) -> str:
        """The row the models read for `text` (see model_row)."""
        return model_row(text, self.count_tokens, self.max_length)

    def _probabilities(self, model, row: str) -> dict[str, float]:
        inputs = self.tokenizer(
            to_model_text(row, self.space_token),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            probs = model(**inputs).logits.float().softmax(dim=-1)[0].tolist()
        return {model.config.id2label[i]: p for i, p in enumerate(probs)}

    def predict(self, text: str) -> tuple[bool, str | None, float]:
        row = self.row(text)
        if not row:
            return True, None, 0.0  # nothing to read: unsure, never safe
        with self._lock:
            gate = self._probabilities(self.gate, row)
            scam_likely = max(gate, key=gate.get) == "scam"
            types = self._probabilities(self.type, row) if self.type and scam_likely else None
        return answer(gate, types)


def from_settings() -> PhayaThaiBertClassifier:
    """The classifier on NLP_MODELS_DIR, or models/ at the repo root when that is empty."""
    from callas_settings import get_settings

    return PhayaThaiBertClassifier(get_settings().nlp_models_dir.strip() or None)
