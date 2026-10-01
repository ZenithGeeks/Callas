"""Check the LLM server: python -m llm_serving.health. Never prints the API key."""

from __future__ import annotations

import sys
import time

from .client import LLMClient, LLMUnavailable, parse_label

SAMPLE = "สวัสดีครับ พรุ่งนี้เจอกันที่ร้านกาแฟนะครับ"


def main() -> int:
    try:
        client = LLMClient()
        print(f"base url: {client.base_url}")
        models = client.list_models()
        print(f"models: {', '.join(models) if models else '(none)'}")
        print(f"using model: {client.model}")
        t0 = time.perf_counter()
        raw = client.complete("classify", SAMPLE)
        latency = time.perf_counter() - t0
    except LLMUnavailable as e:
        print(f"LLM server unreachable: {e}", file=sys.stderr)
        return 1
    print(f"classify({SAMPLE!r}) -> {raw!r} (parsed: {parse_label(raw)}) in {latency:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
