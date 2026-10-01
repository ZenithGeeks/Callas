"""Diarize, then transcribe each speaker segment. Dependencies are injectable for tests."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from asr_diarization.backends import SAMPLE_RATE, ASRBackend
from asr_diarization.types import Transcript, Turn

log = logging.getLogger(__name__)

Diarizer = Callable[[str], list]  # audio_path -> list of objects with speaker/start/end
AudioLoader = Callable[[str], np.ndarray]  # audio_path -> mono float32 at SAMPLE_RATE


@lru_cache(maxsize=1)
def get_default_pipeline() -> tuple[ASRBackend, Diarizer]:
    """Build the real backend and diarizer once per process."""
    from asr_diarization.backends import PyThaiASR
    from asr_diarization.diarize import diarize

    return PyThaiASR(), diarize


def run(
    audio_path: str,
    backend: ASRBackend | None = None,
    diarizer: Diarizer | None = None,
    audio_loader: AudioLoader | None = None,
) -> Transcript:
    if backend is None or diarizer is None:
        default_backend, default_diarizer = get_default_pipeline()
        backend = backend or default_backend
        diarizer = diarizer or default_diarizer
    if audio_loader is None:
        from asr_diarization.backends import load_audio

        audio_loader = load_audio

    t0 = time.perf_counter()
    audio = audio_loader(audio_path)
    segments = diarizer(audio_path)

    turns: list[Turn] = []
    if not segments:
        log.warning("diarization found no segments in %s; using one turn for speaker A", audio_path)
        text = backend.transcribe_array(audio, SAMPLE_RATE).strip()
        if text:
            turns.append(Turn("A", 0.0, len(audio) / SAMPLE_RATE, text))
    else:
        for seg in segments:
            chunk = audio[int(seg.start * SAMPLE_RATE) : int(seg.end * SAMPLE_RATE)]
            text = backend.transcribe_array(chunk, SAMPLE_RATE).strip() if len(chunk) else ""
            if text:
                turns.append(Turn(seg.speaker, round(seg.start, 3), round(seg.end, 3), text))

    transcript = Transcript(turns=turns, model=backend.name, model_version=backend.version)
    log.info(
        "transcribed %s model=%s version=%s turns=%d in %.2fs",
        audio_path,
        transcript.model,
        transcript.model_version,
        len(turns),
        time.perf_counter() - t0,
    )
    return transcript


def to_json_dict(transcript: Transcript) -> dict:
    return {
        "model": transcript.model,
        "model_version": transcript.model_version,
        "speaker_turns": [asdict(t) for t in transcript.turns],
    }


def write_json(transcript: Transcript, out_path: str) -> None:
    Path(out_path).write_text(
        json.dumps(to_json_dict(transcript), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thai ASR + speaker split to transcript JSON.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True, help="transcript.json path")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    transcript = run(args.audio)
    write_json(transcript, args.out)
    sys.stdout.reconfigure(encoding="utf-8")
    print(transcript.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
