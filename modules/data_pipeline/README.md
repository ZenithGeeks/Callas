# data_pipeline: simulated calls

Makes labelled Thai calls for the dataset. DSPy ChainOfThought agents act out each call turn
by turn:

| Agent | Role |
|---|---|
| `planner` | before a scam call, writes the caller's step-by-step plan from the scenario and the newest notes in the scam wiki |
| `scam_caller` | plays a scammer of one of the 5 spec types, following the plan, paced so the ask comes at a set turn |
| `legit_caller` | plays an honest caller (bank, courier, hospital, family); some are hard cases that mention OTPs or accounts |
| `receiver` | answers the phone as one person (age, job, region, family, phone habits, what they know about scams); never told whether the call is a scam |
| `judge` | reads the finished call, checks it shows what it should, writes the reference summary |
| `pattern_extractor` | turns a news article into a scam wiki note (`python -m data_pipeline.patterns add`) |

The label comes from the config, never from a model. Each turn is spoken with PyThaiTTS
(VachanaTTS2, a different voice per speaker), and the turns are joined into one WAV with real
start and end times in `speaker_turns`.

## How a call runs

Each call is a LangGraph graph (`simulate.build_graph`). The drawing below comes from
`CALL_GRAPH.get_graph().draw_mermaid()`:

```mermaid
graph TD;
	__start__([start]) --> plan;
	plan -.-> caller;
	plan -.-> receiver;
	plan -.-> finish;
	caller -.-> receiver;
	caller -.-> finish;
	receiver -.-> caller;
	receiver -.-> finish;
	finish --> __end__([end]);
```

- `plan`: scam calls only, writes the plan once.
- `caller` / `receiver`: add one turn each and save the call.
- The dotted edges are one routing function, `next_step`: whose turn it is, or `finish` when
  someone hung up or the call reached its length.
- `finish`: builds the text, joins the audio, asks the judge, sets accepted or rejected.
- Only errors that come from the model are retried by LangGraph's `RetryPolicy` (waits of 5, 15
  and 45 s): a reply DSPy cannot parse, and rate limits or 5xx answers (`simulate.is_model_error`).
  A bug or a store error fails at once and leaves the call unfinished. A server that is down
  stops the run.
- `plan` can lead to `caller`, `receiver` or `finish`, because a resumed call may be waiting for
  either speaker. A turn is built before it is added to the record, so a retry after a failed
  save never adds it twice.

The agents, the store and the config reach the nodes as LangGraph run context
(`context_schema`), not as state. The graph has no checkpointer: every node saves the call to
the SQLite store, so a stopped run resumes from the last saved turn.

## Run

```
python -m data_pipeline.synthesize --type all --n 20               # 20 calls per type
python -m data_pipeline.synthesize --type romance --n 200 --workers 4
python -m data_pipeline.synthesize --type all --n 5 --no-tts         # text only, faster
python -m data_pipeline.synthesize --check-config --config my_cfg/   # check a config, no LLM
python -m data_pipeline.synthesize --export data/synthetic/calls.jsonl --approved-only
```

Press Ctrl-C to stop: queued calls are cancelled, calls already running stop after their
current turn, and the command prints `stopped by user; rerun to resume` and exits with a non-zero
code. Every turn is saved, so the same command resumes. A rerun also skips finished calls, says how
many it skipped, and warns if any were made with a different config than the current one.

Do not run two commands that write the same database at the same time (two `synthesize` runs
with the same `--db`, say). Calls are not locked, so both would work on the same call and write
over each other's turns. Give each run its own `--db`.

Calls go into one SQLite file, `data/synthetic/calls.db` (gitignored; `--db` picks another),
with audio in `data/synthetic/audio/`. Run from the repo root. `--export` writes the accepted
calls as JSON lines in the spec's schema, for build_dataset; `--approved-only` keeps the ones a
person approved. Calls from an older run that wrote one JSON file per call load with
`--import-json <folder>`.

Review the calls at `/dataset` in the web demo: the call reads as a chat, each message plays on
its own, the agents' reasoning and the caller's plan are one click away, and approve or reject is
saved in the store. Set `CALLAS_DATASET_DB` to review another database.

## Export for the classifier notebook

```
python -m data_pipeline.synthesize --export-notebook dataset/scam_dataset_generated.csv
python -m data_pipeline.synthesize --export-notebook out.csv --store other/calls.db --text asr
```

This writes the csv that `notebooks/scam_classifier_phayathaibert.ipynb` reads next to the
hand-written `dataset/scam_dataset.csv` (the notebook looks for `dataset/scam_dataset_generated.csv`).
It reads the store without changing it, needs no config and no LLM, and prints how many rows it
wrote and how many calls it skipped for each reason. The `--export` option above is a different
file: JSON lines in the spec schema.

Each row is one call, with the columns the notebook needs: `dialogue` (turns written as
`ผู้โทร: ... ผู้รับ: ...` on one line), `scam_class` (`Scam` or `Not a Scam`) and `scam_type` (one of
the five spec names, empty for a normal call). It adds `source` (`simulated`), `group_hint` (call
type and scenario key, for example `romance/engineer_abroad`), `scam_start_turn` (the 0-based turn
where the scam starts, empty for a normal call) and `call_id`. If the store has the leak columns
`receiver_leaked` and `first_leak_turn`, they are written too; if not, they are left out. The file
is utf-8 with a byte order mark, like every csv in the notebook.

Only calls the simulator accepted and a person approved on the `/dataset` page are written. A
call is skipped, and counted under a reason, when the notebook's cleaning would change what was
reviewed: a turn that contains a speaker tag, text in square brackets, a stage note such as
`(เสียงร้องไห้)`, the English words Scam, Phishing or Impersonation, the word ผู้โทร in a caller turn
(the notebook cuts the rest of the turn as narration), an empty turn, two turns in a row from one
speaker, fewer than two turns, a scam type outside the five, or (with `--text asr`) a turn with
no transcript. The rules are copied from the
notebook, and a test fails if the notebook's version changes.

`--text asr` writes the ASR transcript of each turn's audio instead of the written line, so the
model trains on text like the one it reads in production. The export keeps its transcripts in
`eval/results/asr_cache/typhoon_asr.export.json` (or the folder in `CALLAS_ASR_CACHE`). It is not
the cache of `scripts/eval_autocorrect.py`: each entry here also holds a hash of the turn's
written text and audio file name, and an entry whose hash differs is ignored. So a call that you
regenerate under the same id is transcribed again. A turn that is not cached is transcribed with
pythaiasr when it is installed and the audio file is there, and the result is added to the
cache. If that is not possible, the call is skipped as `asr_incomplete`: a call with some turns
written and some spoken would not match what the classifier reads in production. The
transcripts go through the same checks as the written text.

In the notebook, rows with the same `group_hint` always land in one split. With
`KEEP_ORIGINAL_TEST = True` every hand-written row keeps the split it has in
`data/processed/{train,valid,test}.csv`. A generated row follows the hand-written rows of its
group, is dropped if that split is test, and only groups that are new go to train and valid.
Without a generated file the notebook gives back `data/processed` unchanged.

One limit to know: all calls of a scenario stay in one split. Each type has 4 to 6
scenarios, so the type model sees few distinct scenarios, and a type can have too few samples
in valid or test to be scored. Add scenarios in `config/scenarios.yaml` before you generate
more calls of the same type.

## Choose who answers

By default each call samples a receiver (see `receivers` in `config/scenarios.yaml`), and
`profile_share` of calls use one of the people written out under `profiles` (these keep the plain
call id). To have every call
in a run answered by one person:

```
python -m data_pipeline.synthesize --type romance --n 5 --receiver teacher_cm
python -m data_pipeline.synthesize --type all --n 2 --receiver-file my_person.yaml
```

`--receiver` takes the `key` of a profile in `scenarios.yaml`. `--receiver-file` takes a YAML file
with the same fields as a profile:

```yaml
key: nurse_kk
gender: female
age: 41
occupation: nurse on night shifts
education: bachelor's degree
region: Khon Kaen
family: lives with two teenage children
digital_literacy: uses mobile banking every day
scam_awareness: reads police warnings on Facebook
temperament: "skeptical: asks who is calling"
ending: ends the call politely
speech: polite, a little tired
```

A call with a receiver chosen this way gets its own id (`romance_nurse_kk_00003`), so it never
replaces the sampled call for the same seed. A sampled call keeps the plain id
(`romance_00003`) even when the sampler drew one of the written-out profiles. The `key` in a
`--receiver-file` must not be a key already used in `scenarios.yaml`. The chosen receiver is
remembered as a short hash in the call (`generation.call.receiver_override`); a half-done call
will not resume with a different receiver, so use the same `--receiver` or `--receiver-file`
again, or delete the call. The receiver agent never sees the `key`.

## Change how the agents behave

Everything the agents are told lives in `config/`:

| File | What it holds |
|---|---|
| `agents.yaml` | each agent's instructions and field descriptions, model, temperature and max_tokens; caller pacing text; receiver persona and when it may hang up; judge's description of each call type; TTS voices |
| `scenarios.yaml` | the scam and normal scenarios, the receiver pools (jobs, temperaments, endings), call length and pacing numbers |
| `details.yaml` | names, banks, made-up companies, link words and amounts that fill the slots |

To try a variant without touching the default:

1. Copy the folder: `cp -r modules/data_pipeline/config my_cfg`
2. Edit the YAML.
3. Check it: `python -m data_pipeline.synthesize --check-config --config my_cfg`
4. Generate into its own folder: `--config my_cfg --out data/synthetic/my_cfg`

The loader refuses a config that would break the run: an unknown key, a renamed field under
`fields`, a slot that does not exist (`{ADDRESS}`), a missing scam type, a gender with fewer than
two different voices or names, a pacing or stage text that `str.format` cannot fill (only `{turn}`
and `{turns}` exist; stage texts only `{turn}`; write a literal brace as `{{`), a receiver age range
that is not `[low, high]` within 10-100, or a reveal turn that no call can fit. The last rule:
the shortest call a reveal turn can get (`2 * reveal + 4` turns, within `max_turns`) has the caller's
last own turn at `(turns - 1) // 2`, and the ask must come one turn before that at the latest. Every record stores the config name and a hash of its content, and a run will
not resume a call that was started with a different config.

Before you edit:

- Field names under `fields:` are fixed (the code reads them). Change the text, not the keys.
- Agents must write details as slots: `{CALLER_NAME} {RECEIVER_NAME} {ORG} {PHONE}
  {BANK_ACCOUNT} {OTP} {URL} {MONEY}`. Code fills them with made-up values, which gives exact
  entity spans and keeps real numbers out of the data.
- Seeds pick scenarios by list position. Adding a scenario in the middle of a list changes which
  call a seed makes; the config hash keeps old and new records apart.
- Temperature: qwen3.8 garbled Thai spelling at 0.3, so speakers run at 0.0. Variety comes from
  the sampled scenario, receiver, plan and pacing.

## Checks before a call is accepted

A call is marked `rejected` (it stays on disk and shows on `/dataset`) when:

- it has fewer turns than `calls.min_turns`
- an agent wrote an unknown slot (`unknown_slot:NAME`)
- a line has a number the agent wrote itself (`number_without_slot`): 4 or more digits in a row,
  or digit groups joined by `,` `.` `-` or a space with 3-4 digits after the first group
  (`50,000`, `5 000`, `081.234.5678`). Small numbers (`3`, `15`) and times (`10.30`) are fine
- a slot is left unfilled, so `{` or `}` is in the line (`stray_brace`: `{otp}`, `{OTP2}`, `{{OTP}}`)
- a line has a line break or starts with `A:` or `B:` (`line_break`)
- a tone mark comes before a vowel
- the judge says it does not show the intended call, or that it names the label outright
- a scam call has no scam move
- the judge finds a scam in a normal call

The scam start turn is the caller agent's own flag. A flag on a turn before the call's reveal
turn is ignored (the plan forbids asking then). When the caller did not flag one, the judge's
answer is used if it points at a caller turn, and `generation.scam_start_from` says which.

If the model keeps failing after the retries (an unparsable answer, rate limits), the call is
marked `rejected` with `llm_error:<Name>`. Any other error (a bug, a store error) leaves the call
unfinished: the command prints the call id and the error, counts it as `error` in the summary,
exits with code 1, and the next run resumes the call.

The judge's 1-5 "sounds natural" score is shown but does not reject calls: in testing it
reported spelling errors in correctly spelled words.

## Install

Needs the `dspy` and `data` extras. VachanaTTS2 depends on `onnxruntime`; on a machine with
`onnxruntime-gpu`, install it without dependencies so pip does not replace the GPU build:

```
pip install -e ".[dspy,data]"
pip install --no-deps vachanatts==0.0.7 ssg==0.0.8
```

VachanaTTS2 downloads its four voices (about 240 MB) into `./voices/` of the folder you run
from, the first time each is used. Run from the repo root so there is one copy; `voices/` is
gitignored.

## Scam wiki

The pattern notes the scam caller's planner reads are kept as Markdown, one page per scam type, in
`docs/scam-wiki/`. The pages are the source of truth and the SQLite store is a copy.

```
python -m data_pipeline.patterns sync                                    # pages -> store
python -m data_pipeline.patterns add --type romance --file article.txt   # LLM note -> page -> store
python -m data_pipeline.patterns list --type romance
python -m data_pipeline.patterns disable --type romance 2                # sets "active: no"
```

Run `sync` after you edit a page. It replaces the store's notes for each type with the page's
notes, so a deleted note disappears from the store. The page format and the rules for notes are in
`docs/scam-wiki/README.md`; the code is in `wiki.py`.
