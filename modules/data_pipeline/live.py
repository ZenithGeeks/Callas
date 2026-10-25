"""The names the /studio page uses, gathered in one place.

The code lives in three modules:

    streaming.py  stream_call runs one call and yields an Event per step
    capture.py    Capture, the DSPy callback that records each agent's prompt and answer
    demo.py       fake_agents and SilentVoices, for demo mode without a model server

backend/studio.py imports `data_pipeline.live`, so the names stay importable from here.
"""

from .capture import Capture, LmCall, ModuleRun
from .demo import FAKE_MODEL, ScriptedLM, SilentVoices, fake_agents
from .streaming import (
    CallConflict,
    CallSpec,
    Event,
    mark_model_error,
    next_free_seed,
    stream_call,
)

__all__ = [
    "FAKE_MODEL",
    "CallConflict",
    "CallSpec",
    "Capture",
    "Event",
    "LmCall",
    "ModuleRun",
    "ScriptedLM",
    "SilentVoices",
    "fake_agents",
    "mark_model_error",
    "next_free_seed",
    "stream_call",
]
