"""DSPy programs for the voice reply: DSPyReplyWriter and DSPyLeakJudge.

Both run on the vLLM server and only see the REDACTED call context. Needs the optional
extra: pip install -e ".[dspy]". Read the two signatures, then the two classes.
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


def build_lm(
    max_tokens: int | None = None, temperature: float | None = None, model: str | None = None
) -> dspy.LM:
    """Build a dspy.LM on the vLLM server.

    Without `model` it resolves the model like LLMClient: an empty VLLM_MODEL means the first
    served model, so the server must be up.
    """
    s = get_settings()
    model = model or LLMClient().model
    return dspy.LM(
        f"openai/{model}",
        api_base=s.vllm_base_url,
        api_key=s.vllm_api_key.get_secret_value() or "EMPTY",
        temperature=TEMPERATURE["reply"] if temperature is None else temperature,
        max_tokens=DSPY_MAX_TOKENS if max_tokens is None else max_tokens,
        timeout=s.vllm_timeout_s,
        num_retries=0,
        cache=False,
        **({"extra_body": s.llm_extra_body()} if s.llm_extra_body() else {}),
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


def _run_program(
    name: str, program: dspy.Module, lm: Any, context: str, **inputs: str
) -> dspy.Prediction:
    """Run a DSPy program on `lm`, turn connection errors into LLMUnavailable, log the timing.

    `context` is only measured for the log line. It is also the call_context input.
    """
    started = time.perf_counter()
    try:
        # dspy.context per call, never dspy.configure: threads must not share one global LM
        with dspy.context(lm=lm):
            return program(call_context=context, **inputs)
    except _UNAVAILABLE as error:
        raise LLMUnavailable(
            f"cannot reach LLM server for {name}: {type(error).__name__}"
        ) from error
    finally:
        # lengths and timings only, never call text or reply text
        log.info(
            "dspy call=%s context_chars=%d latency_s=%.3f",
            name,
            len(context),
            time.perf_counter() - started,
        )


def _load_reply_program(compiled_path: Path | None) -> dspy.Module:
    """Build the reply program, with the optimizer's demos when the compiled file loads."""
    program = dspy.ChainOfThought(WriteReply)
    path = compiled_path or COMPILED_REPLY
    if not path.exists():
        return program
    try:
        program.load(str(path))
    except Exception as error:  # a broken optimizer file must not stop the call guard
        log.warning(
            "could not load %s (%s), using uncompiled program", path.name, type(error).__name__
        )
    return program


class DSPyReplyWriter:
    """write(context, result) -> reply text. `context` must already be redacted.

    The ChainOfThought `reasoning` field is dropped: `write()` returns the reply text only.
    """

    def __init__(
        self,
        program: dspy.Module | None = None,
        lm: Any | None = None,
        compiled_path: Path | None = None,
    ) -> None:
        self._lm = lm
        self.program = program if program is not None else _load_reply_program(compiled_path)

    @property
    def lm(self):
        """The language model, built on first use."""
        if self._lm is None:
            # Temperature 0: sampling at 0.3 produced garbled Thai spelling on qwen3.8.
            self._lm = build_lm(temperature=0.0)
        return self._lm

    def write(self, context: str, result: Any = None) -> str:
        """Return the spoken reply for the redacted call context."""
        # `result` is accepted for the writer interface; the verdict is fixed because the
        # reply is only ever written for calls the decision step judged benign.
        prediction = _run_program("reply", self.program, self.lm, context, verdict="benign")
        return (getattr(prediction, "reply", "") or "").strip()  # reasoning is dropped here


class DSPyLeakJudge:
    """blocks(reply, context) -> True if the reply must not be spoken. Only a real False
    from the model means "no leak"; every other outcome is a block."""

    def __init__(self, lm: Any | None = None) -> None:
        self._lm = lm
        # Chain of thought: the judge reasons about the reply against the redacted call before
        # answering. Its reasoning is dropped like the writer's; only `leaks` is read.
        self.program = dspy.ChainOfThought(JudgeLeak)

    @property
    def lm(self):
        """The language model, built on first use."""
        if self._lm is None:
            self._lm = build_lm(temperature=0.0)
        return self._lm

    def blocks(self, reply: str, context: str) -> bool:
        """Return True when the reply must not be spoken (see the class docstring)."""
        try:
            prediction = _run_program("judge", self.program, self.lm, context, reply=reply)
        except LLMUnavailable:
            raise
        except Exception as exc:  # unparsable output, server error: block
            log.warning("leak judge failed (%s), blocking", type(exc).__name__)
            return True
        return getattr(prediction, "leaks", True) is not False
