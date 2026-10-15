"""CLI: python -m tts_voice_agent.respond --nlp-result result.json --out reply.wav"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from nlp_core.types import Entity, NLPResult

from .decision import decide
from .memory import CallState
from .reply import safe_reply
from .speech import Speaker
from .templates import pick_template


def load_result(path: str) -> NLPResult:
    """Read an NLPResult from a JSON file. Missing fields get safe defaults."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return NLPResult(
        entities=[Entity(**e) for e in data.get("entities", [])],
        summary=data.get("summary", ""),
        label_scam=bool(data.get("label_scam", False)),
        label_type=data.get("label_type"),
        confidence=float(data.get("confidence", 0.0)),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Prints the decision as JSON and returns the exit code."""
    p = argparse.ArgumentParser(description="Decide and, for benign calls, speak a reply.")
    p.add_argument("--nlp-result", required=True, help="NLPResult JSON file")
    p.add_argument("--out", required=True, help="output WAV path")
    p.add_argument("--llm", action="store_true", help="write the reply with the LLM")
    args = p.parse_args(argv)

    result = load_result(args.nlp_result)
    state = CallState()
    state.add_entities(result.entities)
    state.older_summary = result.summary
    decision = decide(result, state)

    if decision.auto_reply:
        if args.llm:
            try:
                from llm_serving.client import LLMClient
            except ImportError as exc:
                print(f"error: --llm needs llm_serving.client.LLMClient ({exc})", file=sys.stderr)
                return 2
            decision.reply_text, decision.template_used = safe_reply(result, state, LLMClient())
        else:
            decision.reply_text, decision.template_used = pick_template(result), True
        Speaker().synthesize(decision.reply_text, args.out)

    print(json.dumps(asdict(decision)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
