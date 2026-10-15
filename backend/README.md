# Callas web demo

FastAPI, Jinja templates, htmx and Tailwind CSS (standalone CLI). No Node and no CDN at runtime. Endpoints are thin wrappers over the orchestrator; model code lives in `modules/`. For settings, see the root [README](../README.md#settings). `main.py` builds the app. Each page has its own router module: `basics.py`, `message.py`, `call.py`, `live.py`, `review.py` (the `/dataset` page) and `studio.py`.

## Run

From the repo root, with the project venv:

```
# demo mode: scripted Thai answers, fake ASR, silent TTS. No models, no GPU.
CALLAS_FAKE=1 python -m uvicorn backend.main:app --port 8077

# real mode: needs the vLLM server (VLLM_BASE_URL in .env) and the asr and tts extras
python -m uvicorn backend.main:app --port 8077
```

Open http://localhost:8077/. It redirects to `/message`. `GET /healthz` returns `status`, `mode` (`demo` or `real`), `llm_reachable` and `live_call`. It never returns settings or keys. The microphone needs https or localhost.

Demo-only settings: `CALLAS_FAKE_DELAY` (seconds between streamed turns, default 0.45) and `CALLAS_LIVE_SPEED` (speed-up of the scripted live call, default 1).

## Pages

- `/message`: check a text message (up to 2000 characters). `POST /message/check` returns the htmx partial.
- `/call`: simulate a call from a recording. `POST /call/simulate` returns the result shell. `GET /call/{run_id}/events` is a server-sent event stream.
- `/live`: live microphone call over `WS /ws/live`.
- `/dataset`: review simulated calls. It reads the store written by `python -m data_pipeline.synthesize`: `data/synthetic/calls.db`, or the file in `CALLAS_DATASET_DB`. Audio is in an `audio/` folder next to the file. If the file does not exist, the page says so. Review status is `pending`, `approved` or `rejected`, with an optional note. Review posts must come from the page itself (htmx request from the same host).
- `/studio`: make a simulated call and watch it written live (see below).

### Studio

`POST /studio/start` runs the simulator graph in a background thread. `GET /studio/{run}/events` streams its steps as server-sent events. The call is saved to the dataset store (`CALLAS_DATASET_DB`), like a command line run. One run at a time per server. Closing the tab does not stop the run. Stop works between steps, so a stuck model call holds the run until the model server times out. A stopped call stays unfinished. It continues from where it stopped when you press Start with the same seed (same scenario and name, or both empty). Demo mode (`CALLAS_FAKE=1`) uses scripted agents on a scripted LM.

### How the simulated call updates

The shell opens the stream with the htmx `sse` extension. The server renders each changing part (`stages`, `verdict`, `strip`, `transcript`, `entities`, `summary`, `log`) as an HTML partial from `templates/partials/call_regions.html`. It sends the partial as an event of the same name. Elements with `sse-swap="<name>"` replace their contents. A part is sent again only when its HTML changed. A finished run is kept so that a reconnect replays the final picture.

`runtime.py` wraps `asr_diarization.pipeline.run` to capture the turn text, times and ASR duration, because the graph node updates do not carry them.

### Live call

`live.js` reads the microphone with `getUserMedia` and an AudioWorklet (`pcm-worklet.js`, 16 kHz PCM16, 100 ms frames) and sends the audio over the WebSocket. On a `reply` event it sends `{"type":"pause"}`, plays the audio, then sends `{"type":"resume"}`. The message format is in [`modules/orchestrator/README.md`](../modules/orchestrator/README.md#live-calls).

In real mode the server uses `orchestrator.live.LiveCall`. In demo mode it uses `DemoLiveCall` (`live_demo.py`): three normal turns, a spoken reply, and the scam at turn 4. The choice is made in `live_ws.make_live_call`.

## Build the CSS

The compiled `static/css/app.css` is committed. Rebuild it after you change templates, JS or `static/src/app.css`. Run this from `backend/`:

```
tailwindcss -i static/src/app.css -o static/css/app.css --minify
```

`tailwindcss` is the standalone v4 binary. The `pytailwindcss` package installs it. It is not in `pyproject.toml`. The build reads `tailwind.config.js` through `@config`. Design tokens are CSS custom properties on `:root`.

## Tests

```
CALLAS_FAKE=1 python -m pytest backend -q
```

## Fonts and scripts

`static/fonts` has Kanit and Sarabun (Thai and Latin subsets, woff2 from @fontsource 5.3.0, SIL OFL). `static/js` has htmx 2 and htmx-ext-sse 2, minified.
