"""DSPy programs for the voice reply: a reply writer and an optional leak judge.

Both run on the local vLLM server (an OpenAI-compatible endpoint) and both only ever see the
REDACTED call context. Needs the optional extra: pip install -e ".[dspy]".

Safety notes:
  * The writer is a ChainOfThought program. The `reasoning` field it produces is dropped
    here: `write()` returns `prediction.reply` and nothing else, and nothing is logged.
  * The judge can only add blocks (see tts_voice_agent.reply); `blocks()` answers True for
    anything that is not a clear "no leak".
  * `dspy.context(lm=...)` is used per call, never `dspy.configure`, so threads do not
    interfere with each other.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import dspy
import openai

from callas_settings import get_settings

from .client import TEMPERATURE, LLMClient, LLMUnavailable

log = logging.getLogger("llm_serving")

COMPILED_REPLY = Path(__file__).resolve().parent / "compiled" / "reply.dspy.json"

# Connection trouble, in the forms DSPy 3.4 can raise it (its own lm15 errors, the OpenAI
# client's, and the builtins).
_UNAVAILABLE: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    dspy.lm15.TransportError,
    dspy.lm15.TimeoutError,
    dspy.lm15.ServerError,
)


# Chain of thought needs room for the reasoning as well as the reply. With the 80-token
# reply budget the output was cut off, failed to parse, and every reply became a template.
# The reply text itself is still capped by MAX_REPLY_CHARS in safe_reply.
DSPY_MAX_TOKENS = 512


def build_lm(max_tokens: int | None = None, temperature: float | None = None) -> dspy.LM:
    """dspy.LM on the vLLM server. Same model resolution as LLMClient (empty VLLM_MODEL means
    the first served model, which needs the server to be up)."""
    s = get_settings()
    model = LLMClient().model
    return dspy.LM(
        f"openai/{model}",
        api_base=s.vllm_base_url,
        api_key=s.vllm_api_key.get_secret_value() or "EMPTY",
        temperature=TEMPERATURE["reply"] if temperature is None else temperature,
        max_tokens=DSPY_MAX_TOKENS if max_tokens is None else max_tokens,
        timeout=s.vllm_timeout_s,
        num_retries=0,
        cache=False,
    )


class WriteReply(dspy.Signature):
    """Write the spoken reply of an automatic answering assistant for a person who cannot take
    a phone call. The call context is redacted: placeholders such as [NUMBER], [PERSON] or
    [LINK] stand for details that must never be repeated."""

    call_context: str = dspy.InputField(desc="redacted Thai text of the call so far")
    verdict: str = dspy.InputField(desc='short English verdict of the call, for example "benign"')
    reply: str = dspy.OutputField(
        desc=(
            "short polite neutral Thai reply, 1-2 sentences, with no numbers, names, links or "
            "places, no promises, no questions to the caller, never confirming who answered"
        )
    )


class JudgeLeak(dspy.Signature):
    """Decide whether a reply to a phone caller leaks anything. Be strict: when unsure, say true."""

    reply: str = dspy.InputField(desc="the Thai reply that would be spoken to the caller")
    call_context: str = dspy.InputField(desc="redacted Thai text of the call")
    leaks: bool = dspy.OutputField(
        desc=(
            "true if the reply repeats, hints at, or confirms ANY specific detail from the call, "
            "addresses anyone by name or title, mentions an organisation, or asks for anything"
        )
    )


def _timed_call(name: str, context: str, fn):
    t0 = time.perf_counter()
    try:
        return fn()
    except _UNAVAILABLE as exc:
        raise LLMUnavailable(f"cannot reach LLM server for {name}: {type(exc).__name__}") from exc
    finally:
        # lengths and timings only, never call text or reply text
        log.info(
            "dspy call=%s context_chars=%d latency_s=%.3f",
            name,
            len(context),
            time.perf_counter() - t0,
        )


class DSPyReplyWriter:
    """write(context, result) -> reply text. `context` must already be redacted."""

    def __init__(
        self,
        program: dspy.Module | None = None,
        lm: Any | None = None,
        compiled_path: Path | None = None,
    ) -> None:
        self._lm = lm
        if program is None:
            program = dspy.ChainOfThought(WriteReply)
            path = compiled_path or COMPILED_REPLY
            if path.exists():
                try:
                    program.load(str(path))
                except Exception as exc:  # a broken optimizer file must not stop the call guard
                    log.warning(
                        "could not load %s (%s), using uncompiled program",
                        path.name,
                        type(exc).__name__,
                    )
        self.program = program

    @property
    def lm(self):
        if self._lm is None:
            self._lm = build_lm()
        return self._lm

    def write(self, context: str, result: Any = None) -> str:
        # `result` is accepted for the writer interface; the verdict is fixed because the
        # reply is only ever written for calls the decision step judged benign.
        def run():
            with dspy.context(lm=self.lm):
                return self.program(call_context=context, verdict="benign")

        pred = _timed_call("reply", context, run)
        return (getattr(pred, "reply", "") or "").strip()  # reasoning is dropped here


class DSPyLeakJudge:
    """blocks(reply, context) -> True if the reply must not be spoken. Only a real False
    from the model means "no leak"; every other outcome is a block."""

    def __init__(self, lm: Any | None = None) -> None:
        self._lm = lm
        self.program = dspy.Predict(JudgeLeak)

    @property
    def lm(self):
        if self._lm is None:
            self._lm = build_lm(temperature=0.0)
        return self._lm

    def blocks(self, reply: str, context: str) -> bool:
        def run():
            with dspy.context(lm=self.lm):
                return self.program(reply=reply, call_context=context)

        try:
            pred = _timed_call("judge", context, run)
        except LLMUnavailable:
            raise
        except Exception as exc:  # unparsable output, server error: block
            log.warning("leak judge failed (%s), blocking", type(exc).__name__)
            return True
        return getattr(pred, "leaks", True) is not False
