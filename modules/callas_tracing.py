"""Local tracing with Arize Phoenix. Call `setup_tracing` once at startup.

Spans go only to a Phoenix server on this machine. Unless `call_text` is True, the
OpenInference instrumentors hide inputs and outputs, so real calls leave no text in a trace.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from callas_settings import get_settings

log = logging.getLogger("callas.tracing")

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
_started = False


def setup_tracing(project: str, *, call_text: bool) -> None:
    """Send LangGraph and DSPy traces to the local Phoenix server when TRACING=phoenix.

    Args:
        project: Phoenix project name, for example "callas-web".
        call_text: Keep prompts and answers in the traces. False hides them.

    Raises:
        ValueError: PHOENIX_COLLECTOR_ENDPOINT is not on this machine.
    """
    global _started
    settings = get_settings()
    if settings.tracing == "off" or _started:
        return
    endpoint = settings.phoenix_collector_endpoint
    if urlparse(endpoint).hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"PHOENIX_COLLECTOR_ENDPOINT={endpoint!r} is not on this machine; "
            f"use a host from {_LOOPBACK_HOSTS} so call text stays local"
        )
    try:
        from openinference.instrumentation import TraceConfig
        from openinference.instrumentation.dspy import DSPyInstrumentor
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        log.warning("TRACING=phoenix but the trace extra is missing: pip install -e '.[trace]'")
        return

    provider = register(
        project_name=project,
        endpoint=endpoint.rstrip("/") + "/v1/traces",  # register() does not add the path
        protocol="http/protobuf",  # the gRPC port listens on every interface; use HTTP only
        verbose=False,
    )
    config = TraceConfig(hide_inputs=not call_text, hide_outputs=not call_text)
    LangChainInstrumentor().instrument(tracer_provider=provider, config=config)
    DSPyInstrumentor().instrument(tracer_provider=provider, config=config)
    _started = True
    log.info(
        "tracing %s to %s (call text %s)", project, endpoint, "kept" if call_text else "hidden"
    )
