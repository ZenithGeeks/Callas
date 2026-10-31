# Callas

A guard for Thai scam calls and messages.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Language: Thai](https://img.shields.io/badge/language-Thai-blue)

Callas checks Thai phone calls and text messages for scams. It turns a call into text, splits it by speaker, and checks every turn. If the call looks normal, Callas can answer with a short, safe voice reply. If the call looks like a scam, Callas stops and warns the user.

Callas works in Thai only. It does not connect to LINE, Messenger, Instagram or a real phone line. You use it through a web demo.

Status: the full pipeline and the web demo work. The scam check and the summary still use the LLM. A fine-tuned PhayaThaiBERT scam classifier exists (see [`notebooks/README.md`](notebooks/README.md)), but it is not connected to the pipeline yet.

## Where to read more

- [`docs/how-it-works.md`](docs/how-it-works.md): one scam call, step by step, with the NLP technique at each step.
- [`docs/spec.html`](docs/spec.html): the full design.
- [`modules/orchestrator/README.md`](modules/orchestrator/README.md): the graphs, DSPy, how to add a model, live calls.
- [`backend/README.md`](backend/README.md): the web pages, demo mode, the CSS build.
- [`modules/data_pipeline/README.md`](modules/data_pipeline/README.md): the simulator that makes labelled Thai scam calls.
- [`docs/scam-wiki/README.md`](docs/scam-wiki/README.md): the scam notes the simulator reads.
- [`notebooks/README.md`](notebooks/README.md): the classifier notebook (baseline and PhayaThaiBERT).
- Diagrams in `docs/diagrams/`: [LangGraph](docs/diagrams/langgraph.md), [decision flow](docs/diagrams/decision-flow.png), [call sequence](docs/diagrams/call-sequence.png), [architecture](docs/diagrams/architecture.png), [data schema](docs/diagrams/data-schema.png). The `.eraser` sources are edited in [Eraser](https://app.eraser.io/workspace/15jDoAO4Rlj2bQ5kTYqg).

## Quick start

You need Python 3.10 or newer. The scam check, summary and reply need an LLM served with vLLM on a machine you control.

```bash
# 1. Install (pick the extras you need)
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash. PowerShell: .venv\Scripts\Activate.ps1
                                  # macOS or Linux: source .venv/bin/activate
                                  # Or use conda instead of the three lines above:
                                  #   conda create -n callas python=3.11 -y && conda activate callas
pip install -e ".[asr,tts,dev]"   # add ,dspy for the DSPy reply writer, ,scrape for the scraping notebook

# 2. Settings
cp .env.example .env              # then fill in VLLM_BASE_URL and VLLM_MODEL

# 3. Check the LLM server
python -m llm_serving.health

# 4. Run the web demo
uvicorn backend.main:app --port 8077
```

Open http://localhost:8077. The extras in `pyproject.toml` are `asr`, `tts`, `dspy`, `data` (the simulator), `ml` (the notebook) and `dev`.

Demo mode needs no models and no GPU. Every page then runs on a scripted call and fake models. It is the fastest way to work on the UI.

```bash
CALLAS_FAKE=1 uvicorn backend.main:app --port 8077
```

The first real run downloads the speech models. They go to `~/pythaiasr-data` (or `PYTHAIASR_DATA_DIR`). After that they work offline.

## The web pages

| Page | What it does |
|---|---|
| `/message` | Check one text message. Shows the verdict, entities, summary and an explanation |
| `/call` | Simulate a call from a recording (.wav, .mp3, .m4a, .ogg, up to 20 MB) |
| `/live` | Live call with your microphone. Text appears while the caller speaks |
| `/dataset` | Review simulated calls from the data_pipeline store. Approve or reject each call |
| `/studio` | Make one simulated call and watch the agents write it. See [`backend/README.md`](backend/README.md#studio) |

`/dataset` reads `data/synthetic/calls.db`. Set `CALLAS_DATASET_DB` to read another file. Make the store with `python -m data_pipeline.synthesize` (see [`modules/data_pipeline/README.md`](modules/data_pipeline/README.md)).

To train the classifier on approved simulated calls, export them with `python -m data_pipeline.synthesize --export-notebook dataset/scam_dataset_generated.csv`. The notebook reads that file. See [`notebooks/README.md`](notebooks/README.md#generated-data).

## How it works

```text
Call audio ──► speech to text ──► speaker split ──► call memory ──► nlp-core ──► decide()
Live mic   ──► streaming text, a turn ends after 0.8 s of silence ─┘               ├─ scam   ──► warn the user
                                                                                   └─ normal ──► masked context ──► LLM reply ──► leak check ──► voice
Text message ──► nlp-core ──► verdict
```

nlp-core finds entities, writes a summary and gives the scam label. The LLM never decides whether to warn or reply. Fixed rules in `decide()` do that. Each call turn runs as a LangGraph graph; see [`modules/orchestrator/README.md`](modules/orchestrator/README.md).

## Project layout

```text
modules/
  asr_diarization/   speech to text (Typhoon ASR) and speaker split (Nemotron-3), file and live
  nlp_core/          entities, summary, scam check, and the registry where models plug in
  llm_serving/       LLM client for vLLM, prompts, DSPy reply writer and leak judge
  tts_voice_agent/   call memory, decide(), leak check, reply templates, voice (PyThaiTTS)
  orchestrator/      LangGraph graphs, call sessions, live calls, the explaining agent
  data_pipeline/     simulated scam and normal calls for the dataset
  callas_settings.py reads .env
backend/             FastAPI web demo: Jinja templates, HTMX, Tailwind, live call over WebSocket
notebooks/           classifier training notebook
scripts/             eval scripts (eval_autocorrect.py)
eval/results/        scores written by the eval scripts
data/processed/      the fixed train, valid and test split (data/raw and data/synthetic are not in git)
docs/                spec, how-it-works, diagrams, scam wiki, eval notes, design images
```

Each module has a `tests/` folder.

## Settings

All settings come from `.env` (copy `.env.example`). Never commit `.env`. The defaults are in [`modules/callas_settings.py`](modules/callas_settings.py).

| Setting | Default | What it does |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Your vLLM server |
| `VLLM_MODEL` | empty | Model name. Empty means the first model the server lists |
| `VLLM_API_KEY` | `EMPTY` | Key for the vLLM server, if it needs one |
| `VLLM_TIMEOUT_S` | `30` | Seconds before an LLM call gives up |
| `VLLM_THINKING` | `false` | Keep `false`. Thinking models use the whole token budget on hidden reasoning and return no answer |
| `ASR_MODEL` | `typhoon_asr` | Speech-to-text model for recordings and the live mic. `typhoon_nemotron_asr` is the other choice; see `.env.example` |
| `PYTHAIASR_DATA_DIR` | `~/pythaiasr-data` | Where pythaiasr keeps downloaded models. Not in `.env` by default |
| `DIARIZER` | `nemotron` | Speaker split: `nemotron` or `pyannote_segmentation` |
| `AUTOCORRECT` | `off` | Fix misspelled Thai words in each turn: `off` or `pythainlp`. Keep `off`; see [`docs/eval/autocorrect.md`](docs/eval/autocorrect.md) |
| `SPELL_REPAIR` | `off` | Last autocorrect step: fix the vowel and tone marks of words still unknown, scored in their sentence by WangchanBERTa: `on` or `off`. See [`docs/sentence-level-benchmark.md`](docs/sentence-level-benchmark.md) |
| `TTS_PRETRAINED` | `fastthaig2p` | Voice engine (PyThaiTTS) |
| `TTS_SPEAKER` | `thai_som` | Voice |
| `REPLY_ENGINE` | `prompt` | How the reply is written: `prompt` or `dspy` |
| `LEAK_JUDGE` | `off` | `on` adds a DSPy leak judge after the leak rules. It can only block more replies |
| `NLP_NER`, `NLP_CLASSIFIER`, `NLP_SUMMARIZER` | empty (built-in: `rules`, `llm`, `llm`) | Which model fills each nlp-core slot, by registry name. A real environment variable wins over `.env` |
| `TRACING` | `off` | `phoenix` sends traces to a local Phoenix server; see [Tracing](#tracing) |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Where traces go. Must be localhost, 127.0.0.1 or ::1, or startup fails |
| `TRACE_CALL_TEXT` | `off` | `on` keeps prompts and answers in web traces; see [Tracing](#tracing) |
| `CALLAS_FAKE` | not set | `1` runs the web demo with fake models |
| `CALLAS_DATASET_DB` | `data/synthetic/calls.db` | The store the `/dataset` page reads |

The explaining agent on the message page needs vLLM started with `--enable-auto-tool-choice --tool-call-parser <parser for your model>`. Without it the page still works and says the explanation is not available.

## Tracing

Traces show which graph node was slow and what each DSPy call was sent. They go only to a Phoenix server on your own machine. Nothing is sent to a cloud tracer.

Start the Phoenix server outside the app venv. The full server upgrades `openai`, which DSPy and LiteLLM use, so run it with `uvx`. `PHOENIX_HOST=127.0.0.1` keeps the UI, which has no login, off the network:

```
PHOENIX_HOST=127.0.0.1 uvx --from arize-phoenix phoenix serve
```

Then install the client parts, turn tracing on and start the app:

```
pip install -e .[trace]
TRACING=phoenix uvicorn backend.main:app
```

Open http://localhost:6006 and pick the project `callas-web` (web app) or `callas-simulator` (`python -m data_pipeline.synthesize`). Setup refuses a `PHOENIX_COLLECTOR_ENDPOINT` whose host is not localhost, 127.0.0.1 or ::1. Spans go over HTTP to port 6006. The gRPC port 4317 is not used.

What a trace keeps:

- Simulator runs: full prompts and answers, because the calls are made up.
- Web app (message page, recordings, live calls): inputs and outputs are hidden and show as `__REDACTED__`. You still see the node tree, timings, token counts and model names. This keeps the rule that call text is deleted when the call ends.
- `TRACE_CALL_TEXT=on` keeps the text in web traces too. Use it only for local debugging with made-up input. Phoenix then stores that text on disk until you delete it.

## How to contribute

| Task | Where |
|---|---|
| Connect a trained scam, entity or summary model | `modules/nlp_core/`. Steps in [`modules/orchestrator/README.md`](modules/orchestrator/README.md) |
| Dataset and simulator | `modules/data_pipeline/`. See its [README](modules/data_pipeline/README.md) |
| Train or compare classifiers | [`notebooks/`](notebooks/README.md) |
| Eval scripts and report numbers | `scripts/`, results in `eval/results/`. Every number in the report must come from a script run on `data/processed/test.csv` |
| Web UI | `backend/`. Use demo mode. Design images are in `docs/design/` |

Before you open a pull request:

```bash
pytest -q                 # all tests must pass
ruff check .              # lint
ruff format modules backend
```

- Work on a branch, not on `main`. Open a pull request and link the issue.
- Add a test for every new rule or bug fix. Check that the test fails when you break the code it protects.
- Keep `backend/` endpoints thin. Model code belongs in `modules/`.
- Do not commit `.env`, API keys, model files, audio files or data under `data/raw/`.
- The reply prompt (the highest `modules/llm_serving/prompts/reply.v<N>.txt`, now `reply.v3.txt`) and the reply templates (`modules/tts_voice_agent/templates.py`) need team sign-off before you change them.

## Safety rules

These are tested. Do not remove them.

- Call text stays on our machine. The LLM runs on our own vLLM server, never through an outside API.
- The LLM never decides whether to reply or warn. Fixed rules in `decide()` do that.
- Before the LLM writes a reply, numbers, names and links from the call are masked.
- A reply never repeats a phone number, bank detail, OTP, name, link or anything else from the call. If it does, Callas uses a fixed template instead.
- Once a call is marked as a scam, it stays marked until the call ends.
- Call memory is deleted when the call ends. Uploaded recordings are deleted when the run ends, and at server start and stop.
- The dataset never names a real private person as a scammer or victim.

## Evaluation

Every number in the report comes from a script run on the test set and is saved in `eval/results/`. The list of measures is in [`docs/spec.html`](docs/spec.html). Today there is one script, `scripts/eval_autocorrect.py`. Its write-up is [`docs/eval/autocorrect.md`](docs/eval/autocorrect.md).

## Open items

- Team sign-off on the reply prompt and the reply templates.
- Connect the fine-tuned PhayaThaiBERT classifier to nlp-core. It is not registered yet.
- Eval scripts for the other measures in the spec.
- Tune the DSPy reply writer: `python -m llm_serving.dspy_optimize` once `.env` points at vLLM.
- Test the silence threshold for live calls on a real microphone (`VAD_RMS_THRESHOLD` in `modules/asr_diarization/streaming.py`).
- Speaker labels in live calls: there is no streaming speaker split yet, so live turns are labelled "Caller".
- Which open-source LLM to serve with vLLM is still open; see the llm-serving section of [`docs/spec.html`](docs/spec.html).
