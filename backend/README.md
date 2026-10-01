# Callas web demo

FastAPI + Jinja templates + htmx + Tailwind CSS (standalone CLI). No Node, no CDN at runtime.

## Run

From the repo root, with the project venv:

```
# demo mode: scripted Thai answers, fake ASR, silent TTS. No models, no GPU.
CALLAS_FAKE=1 python -m uvicorn backend.main:app --port 8077

# real mode: needs the vLLM server (VLLM_BASE_URL in .env) and the asr/tts extras
python -m uvicorn backend.main:app --port 8077
```

Open http://localhost:8077/. `GET /healthz` reports `mode` (`demo` or `real`) and whether the LLM
server is reachable; it never returns settings or keys. The microphone needs https or localhost.

Demo-only knobs: `CALLAS_FAKE_DELAY` (seconds between streamed turns, default 0.45) and
`CALLAS_LIVE_SPEED` (speed-up of the scripted live call, default 1).

## Pages

- `/message` check a text message. `POST /message/check` returns the htmx partial.
- `/call` simulate a call from a recording. `POST /call/simulate` returns the result shell;
  `GET /call/{run_id}/events` is a server-sent event stream.
- `/live` live microphone call over `WS /ws/live`.

### How the simulated call updates

The shell opens the stream with the htmx `sse` extension (`sse-connect`, `sse-close="done"`). The
server renders each changing part (`stages`, `verdict`, `strip`, `transcript`, `entities`,
`summary`, `log`) as an HTML partial (`templates/partials/call_regions.html`) and sends it as an
event of the same name; elements with `sse-swap="<name>"` replace their contents. A region is only
sent again when its HTML changed, so the SCAM wipe-in and the pen circles play once. A worker thread
runs `CallSessions.simulate_call`; `runtime.py` wraps `asr_diarization.pipeline.run` to capture the
turn text, times and ASR duration (the node updates do not carry them). A finished run can be
replayed from a stored snapshot (htmx reconnects).

### Live call

`live.js` -> `getUserMedia` -> AudioWorklet (`pcm-worklet.js`, 16 kHz PCM16, 100 ms frames) ->
WebSocket. On a `reply` event it sends `{"type":"pause"}`, plays the audio, then
`{"type":"resume"}`. The server code talks to a `LiveCall` (feed, pause, resume, poll_events, end).
`orchestrator.live.LiveCall(sessions)` is used when it exists and demo mode is off (the constructor
signature is assumed; change `live_ws.make_live_call` if it differs). Otherwise `DemoLiveCall`
(`live_demo.py`) plays a scripted call: three normal turns, a spoken reply, the scam at turn 4.

## Build the CSS

The compiled `static/css/app.css` is committed. To rebuild after changing templates, JS or
`static/src/app.css`, from `backend/`:

```
tailwindcss -i static/src/app.css -o static/css/app.css --minify
```

`tailwindcss` is the standalone binary that `pytailwindcss` installs (v4; it reads
`tailwind.config.js` through `@config`). Design tokens are CSS custom properties on `:root`.

## Tests

```
CALLAS_FAKE=1 python -m pytest backend -q
```

## Fonts and vendored scripts

`static/fonts`: Kanit 500/600/700/900 and Sarabun 400/400 italic/600/700, Thai and Latin subsets
(woff2 from @fontsource 5.3.0, SIL OFL). `static/js`: htmx 2 and htmx-ext-sse 2 (minified).
