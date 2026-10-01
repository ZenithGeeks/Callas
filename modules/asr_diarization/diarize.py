"""Speaker diarization via pythaiasr's ONNX models (offline, no Hugging Face token at run time)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass

log = logging.getLogger(__name__)

# The spec's pyannote-segmentation-3.0, ported to ONNX and bundled via pythaiasr.
DEFAULT_DIARIZE_MODEL = "pyannote_segmentation"


@dataclass
class Segment:
    speaker: str  # "A", "B", ...
    start: float
    end: float


def label_for(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA ..."""
    label = ""
    n = index
    while True:
        label = chr(ord("A") + n % 26) + label
        n = n // 26 - 1
        if n < 0:
            return label


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


def diarize(
    audio_path: str,
    model: str = DEFAULT_DIARIZE_MODEL,
    max_gap: float = 0.5,
    **kwargs,
) -> list[Segment]:
    """Return speaker segments for the file; the pythaiasr model is cached across calls."""
    from pythaiasr import diarize as _diarize

    t0 = time.perf_counter()
    raw = _diarize(audio_path, model=model, **kwargs)
    segments = relabel_and_merge(raw, max_gap=max_gap)
    log.info(
        "diarized model=%s segments=%d speakers=%d in %.2fs",
        model,
        len(segments),
        len({s.speaker for s in segments}),
        time.perf_counter() - t0,
    )
    return segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speaker diarization; prints segments as JSON.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default=DEFAULT_DIARIZE_MODEL)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    segments = diarize(args.audio, model=args.model)
    print(json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
