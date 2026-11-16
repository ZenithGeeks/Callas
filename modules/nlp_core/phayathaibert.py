"""The fine-tuned PhayaThaiBERT scam classifier, registered as "phayathaibert".

Set NLP_CLASSIFIER=phayathaibert to use it. It loads the gate and type models that
notebooks/scam_classifier_phayathaibert.ipynb saves in models/, and rewrites the call memory
text into the rows the models learned from, as each folder's preprocessing.json describes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .entities import extract_entities
from .text import normalize_text

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
GATE_DIR = MODELS_DIR / "phayathaibert-scam-gate"
TYPE_DIR = MODELS_DIR / "phayathaibert-scam-type"

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


class PhayaThaiBertClassifier:
    """Scam gate and scam type models, fed the same text the notebook trained them on."""

    def __init__(self, gate_dir: Path = GATE_DIR, type_dir: Path = TYPE_DIR) -> None:
        self.pre = read_preprocessing(gate_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(gate_dir)
        self.gate = self._load(gate_dir)
        # The notebook skips the type model when fewer than 2 types have enough data.
        self.type = None
        if type_dir.exists():
            if read_preprocessing(type_dir) != self.pre:
                raise ValueError(f"{type_dir} and {gate_dir} were saved by different notebook runs")
            self.type = self._load(type_dir)
        # The first forward pass sets up CUDA (about 5 s); do it here, not on a live turn.
        self.predict("Caller: สวัสดีครับ")

    def predict(self, text: str) -> tuple[bool, str | None, float]:
        """Return (label_scam, label_type, confidence); (True, None, 0.0) when unsure."""
        row = model_text(text, self.pre, self._count_tokens)
        gate = self._probabilities(self.gate, row)
        confidence = float(gate.max())
        if confidence < GATE_MIN_CONFIDENCE:
            return True, None, 0.0
        is_scam = self.gate.config.id2label[int(gate.argmax())] == "scam"
        label_type = None
        if is_scam and self.type is not None:
            types = self._probabilities(self.type, row)
            if float(types.max()) >= TYPE_MIN_CONFIDENCE:
                label_type = self.type.config.id2label[int(types.argmax())]
        return is_scam, label_type, confidence

    def _load(self, model_dir: Path):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return model.to(self.device).eval()

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    @torch.inference_mode()
    def _probabilities(self, model, row: str) -> torch.Tensor:
        inputs = self.tokenizer(
            row, truncation=True, max_length=self.pre["max_length"], return_tensors="pt"
        ).to(self.device)
        return model(**inputs).logits.float().softmax(dim=-1)[0].cpu()


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
    row = _spaced("\n".join(lines), pre["space_token"])
    while len(lines) > 1 and count_tokens(row) > pre["max_length"]:
        lines.pop(0)
        row = _spaced("\n".join(lines), pre["space_token"])
    return row


def mask(text: str, labels: Collection[str]) -> str:
    """Replace each sensitive value the entity rules find with "[LABEL]", as notebook step 1."""
    for entity in sorted(extract_entities(text), key=lambda e: e.start, reverse=True):
        if entity.label in labels:
            text = text[: entity.start] + f"[{entity.label}]" + text[entity.end :]
    return text


def _spaced(text: str, space_token: str | None) -> str:
    """The notebook's to_model_text: one space for every run of whitespace, then the token."""
    text = " ".join(text.split())
    return text.replace(" ", space_token) if space_token else text
