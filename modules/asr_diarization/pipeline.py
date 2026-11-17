"""Diarize a file, then transcribe each speaker segment. Dependencies can be passed in for tests."""

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

from asr_diarization import backends
from asr_diarization.backends import SAMPLE_RATE, ASRBackend
from asr_diarization.diarize import Segment, diarize
from asr_diarization.types import Transcript, Turn

log = logging.getLogger(__name__)

Diarizer = Callable[[str], list[Segment]]  # audio_path -> speaker segments
AudioLoader = Callable[[str], np.ndarray]  # audio_path -> mono float32 at SAMPLE_RATE


@lru_cache(maxsize=1)
def get_default_backend() -> ASRBackend:
    """Build the real ASR backend once per process. Loading the model is the slow part."""
    return backends.PyThaiASR()


def run(
    audio_path: str,
    backend: ASRBackend | None = None,
    diarizer: Diarizer | None = None,
    audio_loader: AudioLoader | None = None,
) -> Transcript:
    """Transcribe one audio file into speaker turns.

    Args:
        audio_path: The file to transcribe.
        backend: ASR engine. Default: the pythaiasr model from the ASR_MODEL setting.
        diarizer: Splits the file by speaker. Default: the model from the DIARIZER setting.
        audio_loader: Reads the file into a 16 kHz mono array. Default: pythaiasr's loader.
    """
    backend = backend or get_default_backend()
    diarizer = diarizer or diarize
    # Look the loader up on the module at call time, so a patched backends.load_audio is used.
    audio_loader = audio_loader or backends.load_audio

    started = time.perf_counter()
    with backends.readable(audio_path) as path:  # .m4a and the like become a WAV first
        audio = audio_loader(path)
        segments = diarizer(path)
    if segments:
        turns = _turns_from_segments(audio, segments, backend)
    else:
        log.warning("diarization found no segments in %s; using one turn for speaker A", audio_path)
        turns = _single_turn(audio, backend)

    transcript = Transcript(turns=turns, model=backend.name, model_version=backend.version)
    log.info(
        "transcribed %s model=%s version=%s turns=%d in %.2fs",
        audio_path,
        transcript.model,
        transcript.model_version,
        len(turns),
        time.perf_counter() - started,
    )
    return transcript


def _turns_from_segments(
    audio: np.ndarray, segments: list[Segment], backend: ASRBackend
) -> list[Turn]:
    """Run ASR on each segment's audio. Segments with no audio or no words give no turn."""
    turns: list[Turn] = []
    for segment in segments:
        chunk = audio[int(segment.start * SAMPLE_RATE) : int(segment.end * SAMPLE_RATE)]
        if len(chunk) == 0:
            continue
        text = backend.transcribe_array(chunk, SAMPLE_RATE).strip()
        if text:
            turns.append(
                Turn(
                    speaker=segment.speaker,
                    start=round(segment.start, 3),
                    end=round(segment.end, 3),
                    text=text,
                )
            )
    return turns


def _single_turn(audio: np.ndarray, backend: ASRBackend) -> list[Turn]:
    """Fallback when diarization finds nobody: the whole file is one turn by speaker A."""
    text = backend.transcribe_array(audio, SAMPLE_RATE).strip()
    if not text:
        return []
    end = round(len(audio) / SAMPLE_RATE, 3)  # same rounding as _turns_from_segments
    return [Turn(speaker="A", start=0.0, end=end, text=text)]


def to_json_dict(transcript: Transcript) -> dict:
    """The transcript in the speaker_turns schema of the data pipeline."""
    return {
        "model": transcript.model,
        "model_version": transcript.model_version,
        "speaker_turns": [  # the schema has no "raw": ASR output is the raw text
            {k: v for k, v in asdict(t).items() if k != "raw"} for t in transcript.turns
        ],
    }


def write_json(transcript: Transcript, out_path: str) -> None:
    """Save the transcript as UTF-8 JSON, keeping Thai text readable."""
    Path(out_path).write_text(
        json.dumps(to_json_dict(transcript), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """Transcribe --audio, save the JSON to --out and print the text."""
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
