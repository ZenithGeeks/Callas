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
| `details.yaml` | names, banks, made-up companies, link words, amounts and address parts that fill the slots |

To try a variant without touching the default:

1. Copy the folder: `cp -r modules/data_pipeline/config my_cfg`
2. Edit the YAML.
3. Check it: `python -m data_pipeline.synthesize --check-config --config my_cfg`
4. Generate into its own folder: `--config my_cfg --out data/synthetic/my_cfg`

Every record stores the config hash, which is a hash of the parsed YAML, so any change to
`agents.yaml` (the `context` and `speech` settings too) gives a new hash. A call left unfinished with
the old config is not resumed with the new one: finish it with the old config or delete it. An older
config folder with no `addresses` in `details.yaml` still loads and uses the built-in parts.

The loader refuses a config that would break the run: an unknown key, a renamed field under
`fields`, a slot that does not exist (`{HOUSE_NUMBER}`), a missing scam type, a gender with fewer than
two different voices or names, a pacing or stage text that `str.format` cannot fill (only `{turn}`
and `{turns}` exist; stage texts only `{turn}`; write a literal brace as `{{`), a receiver age range
that is not `[low, high]` within 10-100, or a reveal turn that no call can fit. The last rule:
the shortest call a reveal turn can get (`2 * reveal + 4` turns, within `max_turns`) has the caller's
last own turn at `(turns - 1) // 2`, and the ask must come one turn before that at the latest. Every record stores the config name and a hash of its content, and a run will
not resume a call that was started with a different config.

Before you edit:

- Field names under `fields:` are fixed (the code reads them). Change the text, not the keys.
- Agents must write details as slots: `{CALLER_NAME} {RECEIVER_NAME} {ORG} {PHONE}
  {BANK_ACCOUNT} {OTP} {URL} {MONEY}` and the receiver's own `{RECEIVER_PHONE}
  {RECEIVER_ACCOUNT} {ID_CARD} {BIRTH_DATE} {ADDRESS}`. Code fills them with made-up values,
  which gives exact entity spans and keeps real numbers out of the data.
- Seeds pick scenarios by list position. Adding a scenario in the middle of a list changes which
  call a seed makes; the config hash keeps old and new records apart.
- Temperature: qwen3.8 garbled Thai spelling at 0.3, so speakers run at 0.0. Variety comes from
  the sampled scenario, receiver, plan and pacing.

## Receiver leaks

A real victim sometimes gives personal details away, and those are the calls a scam classifier
most needs to see. The receiver may do this when the person it plays would (its temperament, how
much it knows about scams, how it uses its phone); it is never told whether the call is a scam.
It can only give details through its own slots, so no real data appears:

| Slot | Value | Entity label |
|---|---|---|
| `{OTP}` | 6 random digits | `OTP` |
| `{RECEIVER_PHONE}` | a made-up mobile number | `PHONE` |
| `{RECEIVER_ACCOUNT}` | a made-up bank account | `BANK_ACCOUNT` |
| `{ID_CARD}` | 13 random digits as `1-2345-67890-12-3` (the check digit is not computed, so it is never a valid ID number) | `ID_CARD` |
| `{BIRTH_DATE}` | day, month and Buddhist-era year, e.g. `12 มีนาคม 2511`, exactly the receiver's age on 1 July 2025 | `BIRTH_DATE` |
| `{ADDRESS}` | built from the made-up parts under `addresses` in `details.yaml` | `ADDRESS` |

The receiver's values are drawn after all the other values, so adding them did not change the
values of existing calls. The receiver reads its own name and details back as slots in the
conversation, so it recognises them.

Leaks are found by code from the slots a receiver line uses, never reported by the model.
`{RECEIVER_NAME}` does not count: saying your name when you answer is not a leak. A caller cannot know
the receiver's own details: the callers' prompts do not list those slots, and a caller turn that
uses one gets the problem `caller_used_receiver_slot`, which rejects the call like any other slot
problem. Only receiver turns count as leaks. Each receiver turn gets
`leaked`, the labels it gave away in order of first use (`["OTP", "ID_CARD"]`). When the call
finishes, `generation.receiver_leaked` has the sorted labels of the whole call and
`generation.first_leak_turn` the 0-based index of the first leaking turn (`null` if none). The
store keeps them in new columns (`calls.receiver_leaked`, `calls.first_leak_turn`,
`turns.leaked`); an older `calls.db` gets the columns added when it is opened. `--export` writes
`receiver_leaked` and `first_leak_turn` at the top level of each record. `/dataset` shows a red
"Leaked: OTP, ID CARD" tag under a receiver message that leaked.

New entity labels: `ID_CARD`, `BIRTH_DATE` and `ADDRESS` (in `nlp_core.types.ENTITY_LABELS`).
`PHONE`, `BANK_ACCOUNT`, `OTP` and `PERSON` already existed.

## How numbers are spoken

VachanaTTS reads `158-5-96045-2` as one number with minus signs. So the voice is not given the
stored text: `spoken.py` builds a spoken version from the turn's own entity spans and keeps it on
the turn as `spoken_text` (shown to reviewers; the stored `text` and its spans keep the digits,
because the digits are the labels).

| Label | Said as |
|---|---|
| `PHONE`, `BANK_ACCOUNT`, `ID_CARD` | digit by digit per group: `158-5-96045-2` is "หนึ่งห้าแปด ขีด ห้า ขีด เก้าหกศูนย์สี่ห้า ขีด สอง" |
| `OTP` | digit by digit, no dashes |
| `MONEY` | place value: `49,500 บาท` is "สี่หมื่นเก้าพันห้าร้อยบาท" |
| `BIRTH_DATE` | day and year as numbers, month as written |
| `ADDRESS` | house and soi numbers by place value (`45` is สี่สิบห้า, `12/3` is สิบสองทับสาม); a 5-digit postcode digit by digit |
| `PERSON`, `ORG`, `URL` | as written |

People do not always say "ขีด"; often they only pause between groups. In the audio a pause
is a comma (", "), because VachanaTTS barely pauses on a plain space (4 digit groups: 3.41 s with
spaces, 3.29 s with nothing, 3.55 s with commas on `th_f_1`; 3.17 s, 3.11 s and 3.80 s on
`th_m_1`). Each call picks one style
from its seed and keeps it (`generation.speech_style`, `dash` or `pause`). The share of `dash`
calls is `speech.say_dash_share` in `agents.yaml` (0 to 1, default 0.5). Text outside the spans is
never changed.

## Context window

qwen3.8 on the gateway has a 100K-token window. `context` in `agents.yaml` sets
`window_tokens: 100000` and `safety_margin_tokens: 2000`. Each agent's prompt may use
`window_tokens - lm.max_tokens - safety_margin_tokens`; the loader refuses a config where an
agent's `max_tokens` plus the margin is not below the window, and names the agent.

The prompt is measured, not guessed: `budget.py` formats the exact messages DSPy sends (the
ChatAdapter, with the `reasoning` field ChainOfThought adds) and estimates their size as UTF-8
bytes / 3, rounded up. That is about one token per Thai character, the worst case; it over-counts
English, which is the safe side. When a prompt is over budget:

- speakers (both callers and the receiver) keep the first 2 turns and as many of the newest turns
  as fit; the turns between are replaced by one line, `(... N earlier turns not shown ...)`
- the planner drops the oldest pattern notes until it fits (`generation.briefing.notes_used`
  lists the ones it used)
- if the prompt still does not fit (a huge newest turn, or 3 turns or fewer), the model is not
  asked: the call ends `rejected` with the reason `too_long_for_context`
- the judge must read the whole call, so a call that does not fit is not judged: it is marked
  `rejected` with the reason `too_long_for_judge`, and the judge program is not called
- `patterns add` cuts the article to the longest start that fits (never more than
  `MAX_ARTICLE_CHARS`)

Every turn stores the estimated size of its prompt as `prompt_tokens` (and the plan and the judge
store theirs). A normal call is a few thousand tokens, so none of this changes a normal run. The
live call's `CONTEXT_TOKEN_BUDGET` counts PyThaiNLP words, not model tokens (see
`tts_voice_agent/constants.py`).

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
