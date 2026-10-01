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
    return os.environ.get("CALLAS_FAKE", "").strip().lower() in {"1", "true", "yes", "on"}


def last_transcription() -> tuple[Any, float] | None:
    """(Transcript, seconds) of the last pipeline.run on this thread, or None."""
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

    def close(self) -> None:
        for call_id in list(self.runtime.states):
            self.sessions.end(call_id)
        asr_pipeline.run = self.original_run


def build(force_demo: bool | None = None) -> AppRuntime:
    is_demo = demo_enabled() if force_demo is None else force_demo
    original = getattr(asr_pipeline.run, "_callas_wrapped", asr_pipeline.run)
    if is_demo:
        llm = demo.make_llm()
        demo.register_demo_parts(llm)
        rt = Runtime(
            llm=llm, speaker=demo.FakeSpeaker(), ner="demo", classifier="demo", summarizer="demo"
        )
        asr_pipeline.run = _recording(demo.fake_pipeline_run)
    else:
        from llm_serving.client import LLMClient
        from tts_voice_agent.speech import Speaker

        rt = Runtime(llm=LLMClient(), speaker=Speaker())
        asr_pipeline.run = _recording(original)
    return AppRuntime("demo" if is_demo else "real", rt, CallSessions(rt), original)
