"""The fine-tuned scam classifiers from notebooks/scam_classifier_phayathaibert.ipynb.

The notebook saves a gate and a type model per encoder in models/<encoder>-scam-gate and
models/<encoder>-scam-type. Any gate can go with any type (registry.py registers every pair).
Each model reads the call memory text rewritten into the rows it learned from, as its own
preprocessing.json describes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection
from functools import cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .entities import extract_entities
from .registry import model_dir
from .text import normalize_text

# The notebook's TYPE_MIN_CONFIDENCE (first cell of part B) and GATE_MIN_CONFIDENCE (step 14).
# Retune them there, then copy them here.
GATE_MIN_CONFIDENCE = 0.60
TYPE_MIN_CONFIDENCE = 0.50

# Speaker names in the call memory text: the live transcriber calls the person on the line
# "Caller" (asr_diarization.streaming) and the turn graph calls Callas's own replies "agent".
# Any other name, such as a diarizer's "A" or "B", is passed on unchanged.
SPEAKER_ROLES = {"Caller": "caller", "agent": "receiver"}
PREPROCESSING_KEYS = ("row_text", "speaker_letters", "masked_labels", "space_token", "max_length")
_LINE = re.compile(r"^(\S+): (.*)$")


class FineTunedModel:
    """One saved model with its own tokenizer and the row format it was trained on."""

    def __init__(self, model_dir: Path, device: str) -> None:
        self.pre = read_preprocessing(model_dir)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model = model.to(device).eval()

    def top_label(self, text: str) -> tuple[str, float]:
        """The most likely label for the call memory text, and its probability."""
        row = model_text(text, self.pre, self._count_tokens)
        probabilities = self._probabilities(row)
        best = int(probabilities.argmax())
        return self.model.config.id2label[best], float(probabilities[best])

    def _count_tokens(self, row: str) -> int:
        return len(self.tokenizer(row)["input_ids"])

    @torch.inference_mode()
    def _probabilities(self, row: str) -> torch.Tensor:
        inputs = self.tokenizer(
            row, truncation=True, max_length=self.pre["max_length"], return_tensors="pt"
        ).to(self.device)
        return self.model(**inputs).logits.float().softmax(dim=-1)[0].cpu()


class FineTunedClassifier:
    """A scam gate model and a scam type model, each fed the text it was trained on."""

    def __init__(self, gate_dir: Path, type_dir: Path) -> None:
        self.gate = _load(gate_dir)
        # The notebook skips the type model when fewer than 2 types have enough data.
        self.type = _load(type_dir) if type_dir.exists() else None
        # The first forward pass sets up CUDA (about 5 s); do it here, not on a live turn.
        self.predict("Caller: สวัสดีครับ")

    @classmethod
    def from_encoders(cls, gate: str, type_: str) -> FineTunedClassifier:
        """The saved gate model of encoder `gate` and type model of encoder `type_`."""
        return cls(model_dir(gate, "gate"), model_dir(type_, "type"))

    def predict(self, text: str) -> tuple[bool, str | None, float]:
        """Return (label_scam, label_type, confidence); (True, None, 0.0) when unsure."""
        gate_label, confidence = self.gate.top_label(text)
        if confidence < GATE_MIN_CONFIDENCE:
            return True, None, 0.0
        is_scam = gate_label == "scam"
        label_type = None
        if is_scam and self.type is not None:
            type_label, type_confidence = self.type.top_label(text)
            if type_confidence >= TYPE_MIN_CONFIDENCE:
                label_type = type_label
        return is_scam, label_type, confidence


@cache
def _load(path: Path) -> FineTunedModel:
    """Each model loads once, however many gate and type pairs share it."""
    return FineTunedModel(path, "cuda" if torch.cuda.is_available() else "cpu")


def read_preprocessing(model_dir: Path) -> dict:
    """The model's preprocessing.json. Raises when the model cannot read the live call text."""
    path = model_dir / "preprocessing.json"
    retrain = "Run notebooks/scam_classifier_phayathaibert.ipynb to train and save the models."
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. {retrain}")
    pre = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in PREPROCESSING_KEYS if key not in pre]
    if missing:
        raise ValueError(f"{path} has no {missing}: an older notebook saved it. {retrain}")
    if pre["row_text"] != "call_so_far":
        raise ValueError(
            f"{path}: row_text is {pre['row_text']!r}, but the live call sends the call so far. "
            f'Set ROW_TEXT = "call_so_far" in the notebook and retrain.'
        )
    return pre


def model_text(text: str, pre: dict, count_tokens: Callable[[str], int]) -> str:
    """The call memory text as one training row (notebook step 6b, ROW_TEXT "call_so_far").

    Each "speaker: words" line gets its role letter and its sensitive values masked. Other lines
    are left out: the summary of older turns can start inside a line ("ler: ..."). The oldest
    lines are cut until the row fits max_length; the newest line, the one the label is about,
    stays.
    """
    lines = []
    for match in filter(None, map(_LINE.match, text.splitlines())):
        speaker, words = match.groups()
        role = SPEAKER_ROLES.get(speaker)
        if role is None and len(speaker) != 1:  # not a role name, not a diarizer letter
            continue
        letter = pre["speaker_letters"][role] if role else speaker
        lines.append(f"{letter}: {mask(normalize_text(words), pre['masked_labels'])}")
    row = _encoded("\n".join(lines), pre)
    while len(lines) > 1 and count_tokens(row) > pre["max_length"]:
        lines.pop(0)
        row = _encoded("\n".join(lines), pre)
    return row


def mask(text: str, labels: Collection[str]) -> str:
    """Replace each sensitive value the entity rules find with "[LABEL]", as notebook step 1."""
    for entity in sorted(extract_entities(text), key=lambda e: e.start, reverse=True):
        if entity.label in labels:
            text = text[: entity.start] + f"[{entity.label}]" + text[entity.end :]
    return text


def _encoded(text: str, pre: dict) -> str:
    """The notebook's Backbone.encode: one space per whitespace run, lowercased for an uncased
    vocabulary, then each space written as the space token."""
    text = " ".join(text.split())
    # Models saved before the lowercase key existed were all cased.
    if pre.get("lowercase", False):
        text = text.lower()
    return text.replace(" ", pre["space_token"]) if pre["space_token"] else text
