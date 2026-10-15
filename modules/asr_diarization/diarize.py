"""Speaker diarization via pythaiasr's ONNX models (offline, no Hugging Face token at run time)."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass

from callas_settings import get_settings

log = logging.getLogger(__name__)

# Setting value (DIARIZER) -> pythaiasr model name.
DIARIZER_MODELS = {
    "nemotron": "nemotron-3-diarization",  # NVIDIA Nemotron-3, int8 ONNX, speaker cache
    "pyannote_segmentation": "pyannote_segmentation",  # pyannote-seg-3.0, 10 s windows
}

_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


@dataclass
class Segment:
    speaker: str  # "A", "B", ...
    start: float
    end: float


def diarize(
    audio_path: str,
    model: str | None = None,
    max_gap: float = 0.5,
    **kwargs,
) -> list[Segment]:
    """Return speaker segments for the file; model comes from settings unless overridden.

    Args:
        audio_path: The file to split by speaker.
        model: A DIARIZER_MODELS key or a pythaiasr model name. Default: the DIARIZER setting.
        max_gap: Seconds of silence below which two segments of one speaker are joined.
        **kwargs: Passed on to the pythaiasr model's diarize().
    """
    model = resolve_model(model)
    started = time.perf_counter()
    raw_segments = get_diarizer(model).diarize(audio_path, **kwargs)
    segments = relabel_and_merge(raw_segments, max_gap=max_gap)
    log.info(
        "diarized model=%s segments=%d speakers=%d in %.2fs",
        model,
        len(segments),
        len({s.speaker for s in segments}),
        time.perf_counter() - started,
    )
    return segments


def resolve_model(model: str | None = None) -> str:
    """Pythaiasr model name: the override (key or raw name) if given, else the DIARIZER setting."""
    choice = model or get_settings().diarizer
    return DIARIZER_MODELS.get(choice, choice)


def get_diarizer(model_name: str):
    """Load the pythaiasr Diarization model once per model name and reuse it."""
    with _cache_lock:
        if model_name in _cache:
            return _cache[model_name]
        from pythaiasr import Diarization

        started = time.perf_counter()
        diarizer = Diarization(model=model_name)
        _cache[model_name] = diarizer
        log.info("loaded diarizer model=%s in %.2fs", model_name, time.perf_counter() - started)
        return diarizer


def relabel_and_merge(raw: list[dict], max_gap: float = 0.5) -> list[Segment]:
    """Sort, rename speaker ids to A/B/... by first appearance, merge same-speaker neighbours."""
    labels: dict[str, str] = {}
    merged: list[Segment] = []
    for item in sorted(raw, key=lambda s: (float(s["start"]), float(s["end"]))):
        speaker = labels.setdefault(str(item["speaker"]), label_for(len(labels)))
        start, end = float(item["start"]), float(item["end"])
        previous = merged[-1] if merged else None
        if previous and previous.speaker == speaker and start - previous.end <= max_gap:
            previous.end = max(previous.end, end)
        else:
            merged.append(Segment(speaker, start, end))
    return merged


def label_for(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA ..."""
    label = ""
    while True:
        index, letter = divmod(index, 26)
        label = chr(ord("A") + letter) + label
        index -= 1  # after Z comes AA, not BA: there is no zero digit
        if index < 0:
            return label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speaker diarization; prints segments as JSON.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default=None, help="nemotron or pyannote_segmentation")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    segments = diarize(args.audio, model=args.model)
    print(json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
