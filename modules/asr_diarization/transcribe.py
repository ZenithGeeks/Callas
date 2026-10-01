"""CLI: transcribe one audio file with a single ASR model (no diarization)."""

from __future__ import annotations

import argparse
import logging
import sys

from asr_diarization.backends import PyThaiASR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thai speech-to-text; prints the text.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default=None, help="default: ASR_MODEL setting (typhoon_asr)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    backend = PyThaiASR(model=args.model)
    text = backend.transcribe(args.audio)
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
