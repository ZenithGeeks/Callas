"""Build the orchestrator Runtime once, at startup.

Real mode: LLMClient + Speaker + the default nlp-core registry.
Demo mode (env CALLAS_FAKE=1): scripted parts from backend.demo, no models and no GPU.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from asr_diarization import pipeline as asr_pipeline
from orchestrator import CallSessions, Runtime

from . import demo

_local = threading.local()


def demo_enabled() -> bool:
    """True when CALLAS_FAKE is set to 1, true, yes or on."""
    return os.environ.get("CALLAS_FAKE", "").strip().lower() in {"1", "true", "yes", "on"}


def last_transcription() -> tuple[Any, float] | None:
    """(Transcript, seconds) of the last pipeline.run on this thread, or None.

    _recording() stores it in a thread-local, so call this on the thread that ran ASR.
    """
    return getattr(_local, "last", None)


def _recording(run):
    """Wrap pipeline.run so the page can show the turn text, times and ASR duration.

    CallSessions.simulate_call only yields node updates, which do not carry the turn text.
    """

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        transcript = run(*args, **kwargs)
        _local.last = (transcript, time.perf_counter() - t0)
        return transcript

    wrapper._callas_wrapped = run
    return wrapper


@dataclass
class AppRuntime:
    """The orchestrator Runtime plus what the web app needs around it."""

    mode: str
    runtime: Runtime
    sessions: CallSessions
    original_run: Any
    _agent: Any = field(default=None, repr=False)

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    def llm_reachable(self) -> bool | None:
        """True or False for the real LLM server. None in demo mode (nothing to reach)."""
        if self.is_demo:
            return None
        from llm_serving.client import LLMClient

        try:
            LLMClient(timeout=3.0).list_models()
        except Exception:
            return False
        return True

    def explain(self, text: str) -> str:
        """Advisory Thai text from orchestrator.agents.explain. May raise."""
        from orchestrator import agents

        if self.is_demo:
            return agents.explain(text, agent=demo.fake_agent(text))
        if self._agent is None:
            self._agent = agents.build_analyst_agent()  # not cached if this raises
        return agents.explain(text, agent=self._agent)

    def models(self) -> list[tuple[str, str]]:
        """(feature, model) for every model a call goes through, as the settings choose them."""
        if self.is_demo:
            return [("Every feature", "scripted demo parts, no models")]
        from callas_settings import get_settings
        from nlp_core import registry

        s = get_settings()
        llm = s.vllm_model or "vLLM (VLLM_MODEL not set)"

        def part(kind: str, name: str | None) -> str:
            name = name or registry.default_name(kind)
            return {"llm": llm, "rules": "rules (regex)"}.get(name, name)

        classifier = self.runtime.classifier or registry.default_name("classifier")
        pair = registry.fine_tuned_pair(classifier)
        if pair:
            gate, type_ = (f"fine-tuned {registry.FINE_TUNED[e]}" for e in pair)
        else:
            gate = type_ = part("classifier", classifier)
        spelling = {"off": "off", "pythainlp": "PyThaiNLP"}[s.autocorrect]
        if s.autocorrect != "off" and s.spell_repair == "on":
            spelling += " + WangchanBERTa repair"
        return [
            ("Speech to text", s.asr_model),
            ("Speaker split (recordings)", s.diarizer),
            ("Spelling fix", spelling),
            ("Entities", part("ner", self.runtime.ner)),
            ("Scam gate", gate),
            ("Scam type", type_),
            ("Summary", part("summarizer", self.runtime.summarizer)),
            ("Reply", f"{llm} ({s.reply_engine})"),
            ("Leak judge", f"{llm} (DSPy)" if s.leak_judge == "on" else "rules only"),
            ("Voice", f"PyThaiTTS {s.tts_pretrained}, speaker {s.tts_speaker}"),
        ]

    def close(self) -> None:
        """End every call still in memory and put the original ASR function back."""
        for call_id in list(self.runtime.states):
            self.sessions.end(call_id)
        asr_pipeline.run = self.original_run


def build(force_demo: bool | None = None) -> AppRuntime:
    """Build the runtime for demo mode (CALLAS_FAKE, or force_demo) or real mode."""
    is_demo = demo_enabled() if force_demo is None else force_demo
    original = getattr(asr_pipeline.run, "_callas_wrapped", asr_pipeline.run)
    if is_demo:
        llm = demo.make_llm()
        demo.register_demo_parts(llm)
        runtime = Runtime(
            llm=llm, speaker=demo.FakeSpeaker(), ner="demo", classifier="demo", summarizer="demo"
        )
        asr_pipeline.run = _recording(demo.fake_pipeline_run)
    else:
        from llm_serving.client import LLMClient
        from nlp_core import registry
        from tts_voice_agent.speech import Speaker

        # Build the classifier now: a fine-tuned model loads once at startup, and a missing or
        # outdated one stops the server here instead of failing the first live turn.
        registry.get("classifier")
        runtime = Runtime(llm=LLMClient(), speaker=Speaker())
        asr_pipeline.run = _recording(original)
    return AppRuntime("demo" if is_demo else "real", runtime, CallSessions(runtime), original)
