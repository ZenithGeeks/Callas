"""Speaker diarization via pythaiasr's ONNX models (offline, no Hugging Face token at run time)."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass

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
    """Return speaker segments for the file; model comes from settings unless overridden."""
    model = resolve_model(model)
    engine = get_diarizer(model)
    t0 = time.perf_counter()
    raw = engine.diarize(audio_path, **kwargs)
    segments = relabel_and_merge(raw, max_gap=max_gap)
    log.info(
        "diarized model=%s segments=%d speakers=%d in %.2fs",
        model,
        len(segments),
        len({s.speaker for s in segments}),
        time.perf_counter() - t0,
    )
    return segments


def resolve_model(model: str | None = None) -> str:
    """Pythaiasr model name: the override (key or raw name) if given, else the DIARIZER setting."""
    from callas_settings import get_settings

    choice = model or get_settings().diarizer
    return DIARIZER_MODELS.get(choice, choice)


def get_diarizer(model_name: str):
    """Load the pythaiasr Diarization model once per model name and reuse it."""
    with _cache_lock:
        inst = _cache.get(model_name)
        if inst is None:
            from pythaiasr import Diarization

            t0 = time.perf_counter()
            inst = Diarization(model=model_name)
            _cache[model_name] = inst
            log.info("loaded diarizer model=%s in %.2fs", model_name, time.perf_counter() - t0)
        return inst


def relabel_and_merge(raw: list[dict], max_gap: float = 0.5) -> list[Segment]:
    """Sort, rename speaker ids to A/B/... by first appearance, merge same-speaker neighbours."""
    labels: dict[str, str] = {}
    out: list[Segment] = []
    for seg in sorted(raw, key=lambda s: (float(s["start"]), float(s["end"]))):
        spk = labels.setdefault(str(seg["speaker"]), label_for(len(labels)))
        start, end = float(seg["start"]), float(seg["end"])
        if out and out[-1].speaker == spk and start - out[-1].end <= max_gap:
            out[-1].end = max(out[-1].end, end)
        else:
            out.append(Segment(spk, start, end))
    return out


def label_for(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA ..."""
    label = ""
    n = index
    while True:
        label = chr(ord("A") + n % 26) + label
        n = n // 26 - 1
        if n < 0:
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
