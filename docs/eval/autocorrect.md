# Transcript autocorrect: what it does to scores and ASR errors

Closes the checks listed in issue #30. The step measured is `AUTOCORRECT=pythainlp` in
`nlp_core.text.autocorrect_text`, applied to each turn after `normalize_text`.

Reproduce with `python scripts/eval_autocorrect.py` (about 70 to 100 s on an RTX 5070 Ti with the ASR
transcripts cached, about 2.5 minutes the first time). Results are written to
`eval/results/autocorrect.json`. The script needs the two trained models in `models/`
(not in git) and the simulated-call store (a `calls.db` with `audio/` beside it)
given by `--store <calls.db>` or the env var `CALLAS_DATASET_DB` (default `data/synthetic/calls.db`; make one with `python -m data_pipeline.synthesize`). Fixed seed 42, batch inference on the GPU.

## Recommendation

Keep `AUTOCORRECT=off`.

- On 187 ASR mistakes (Typhoon ASR) it fixed 0. It changed 3 turns, and in all 3 the new word was
  still wrong. The other ASR model: 0 fixed, 4 changed, all still wrong.
- It did not break a correct word in this data, but only because it almost never fires. The
  errors in the transcripts are real Thai words (wrong word, not wrong spelling), and a
  dictionary check cannot see those.
- The error rates did not move: CER 0.1076 to 0.1078, TER 0.1699 to 0.1695.
- It costs time: 1.5 s on the first call (loading the dictionary), 0.18 s per turn on average and
  0.83 s at p95 with a cold word cache, with a worst turn of 6.9 s.
- The known bad fix from PR #35 still happens (see below). In a scam call "transfer" turning into
  "withdraw" changes the meaning.
- Scam gate, scam type and rule entity results were identical with it on and off, on the test
  set and on the ASR text of 20 calls (small n).

The step may be worth another look only with a context-aware corrector, and only if the same
checks (this script) show fixed words well above broken or still-wrong words.

## Method

**Gate and type.** `data/processed/test.csv` has 55 dialogues. Each turn is normalized, then
(autocorrect on) corrected one turn at a time, as the web system does, then rebuilt as
`A: ...` / `B: ...` lines. With autocorrect off the rebuilt text matches the `text` column exactly
(the script asserts it). Tokenization follows the notebook and each model's `preprocessing.json`:
spaces become `<_>`, max length 256. Gate scores use all 55 dialogues. The type model has two
classes (`fake_authority`, `phishing_otp`), so it is scored on the 39 test dialogues of those two
types.

**Entities.** The rule NER (`nlp_core.entities`) is run on every turn with autocorrect off and on.
The `entities` column of `test.csv` (and of `train.csv` and `valid.csv`) is empty in every row,
so there is no gold to score against; the table compares the two runs with each other. Gold
spans would refer to raw text and shift when a word changes length, so entities are matched by
(label, text) with Thai digits, spaces, commas and dashes removed, not by offset.

Entity scores against real gold come from the simulated calls instead. Their `turns.entities`
column has spans for 9 entities the rules can find (4 MONEY, 3 BANK_ACCOUNT, 1 OTP, 1 URL).
The ASR text is scored against those, matched by (label, text).

**ASR.** 124 turns from 20 simulated calls (`calls.db`, one WAV per turn). Each WAV is
transcribed with `pythaiasr.ASR(model=..., device="auto")` for `typhoon_asr` and
`typhoon_nemotron_asr`. Reference and ASR text both go through `normalize_text`, then spaces and
punctuation are removed (the ASR writes none and places spaces freely, and Thai does not need
them), then both are cut with PyThaiNLP `newmm`. Tokens are aligned with `difflib.SequenceMatcher`.
Every non-equal block is one mistake. CER is the character edit distance over the joined text,
TER (token error rate, newmm tokens, not whitespace words) the token edit distance, both over the whole corpus.

To score the autocorrect, the same ASR text is corrected and re-aligned. A mistake is
**fixed** when its tokens were replaced and now match the reference, **changed but still wrong**
when replaced by another wrong word, and **left** when untouched (this includes words the ASR
dropped). **Broken** is a place where tokens that matched the reference were changed.

**Latency.** Time of `autocorrect_text` per turn on the 124 ASR outputs. "Cold" is after clearing
the word cache (the cache is `lru_cache` on `_fix_word`; the dictionary stays loaded). "Warm" is
a second pass.

## Results

### Scam gate and type (test.csv, 55 dialogues)

| Task | Autocorrect | n | Macro-F1 | Accuracy |
| --- | --- | --- | --- | --- |
| Gate | off | 55 | 0.952 | 0.964 |
| Gate | on | 55 | 0.952 | 0.964 |
| Type | off | 39 | 0.862 | 0.923 |
| Type | on | 39 | 0.862 | 0.923 |

The model input is built exactly as in the notebook (`to_model_text`: all whitespace, newlines
between turns included, becomes the `<_>` token). The script checks that the "off" rows equal the
phayathaibert test scores recorded in `models/scam_classifier_metrics.json` (gate F1 0.9521,
accuracy 0.9636; type F1 0.8615, accuracy 0.9231) and stops with an error if they do not.

Autocorrect changed 0 of 55 test dialogues, so the scores cannot differ. This matches PR #35
(0 of 247 turns changed): the test set is clean written Thai, so on this text autocorrect does
nothing. It says nothing about ASR text; the next table does.

### Gate and type on ASR text of whole calls

Each of the 20 simulated calls is rebuilt from its ASR turns (`A:` / `B:` lines, each turn
normalized, then corrected for the "on" row) and labelled from the store (`label_scam`,
`label_type`). The type model is scored on calls of its two types only. n is small, so one call
moves a score by several points.

| ASR model | Task | n | Autocorrect | Macro-F1 | Accuracy | Calls changed |
| --- | --- | --- | --- | --- | --- | --- |
| typhoon_asr | Gate | 20 | off | 0.804 | 0.900 | 3 of 20 |
| typhoon_asr | Gate | 20 | on | 0.804 | 0.900 | 3 of 20 |
| typhoon_asr | Type | 7 | off | 0.857 | 0.857 | 2 of 7 |
| typhoon_asr | Type | 7 | on | 0.857 | 0.857 | 2 of 7 |
| typhoon_nemotron_asr | Gate | 20 | off | 0.804 | 0.900 | 4 of 20 |
| typhoon_nemotron_asr | Gate | 20 | on | 0.804 | 0.900 | 4 of 20 |
| typhoon_nemotron_asr | Type | 7 | off | 0.708 | 0.714 | 0 of 7 |
| typhoon_nemotron_asr | Type | 7 | on | 0.708 | 0.714 | 0 of 7 |

Autocorrect changed 3 and 4 of the 20 call texts and no prediction changed, so on this sample
correction did not lower the gate or the type scores. It did not raise them either. With 20
calls this can only rule out a large drop.

### Rule entities

On `test.csv` the rules found 45 entities with autocorrect off and the same 45 with it on; 0 of
55 dialogues differ. There is no gold column to give precision or recall there.

On the ASR text, against the 9 gold entities in the simulated calls:

| Text | Precision | Recall | F1 |
| --- | --- | --- | --- |
| ASR, autocorrect off (both ASR models) | 0 | 0 (0 of 9) | 0 |
| ASR, autocorrect on (both ASR models) | 0 | 0 (0 of 9) | 0 |

Per label (BANK_ACCOUNT, MONEY, OTP, URL) the result is 0 for each, with and without autocorrect.
The cause is not spelling. The ASR writes numbers as Thai words ("85,000" came out as
แปดสิบห้าศูนย์, "158-5-96045-2" as หนึ่งร้อยห้าสิบแปดลบห้าลบเก้าหมื่นหกพันสี่สิบห้าลบสอง, the
URL as ATPSREAFunceUREห้าสิบเน็ต), and the rules only read digits. This is a bigger problem for
entity extraction than any spelling mistake, and autocorrect does not touch it. Only 9 gold
entities exist, so the exact figures are coarse.

### ASR errors (124 turns, 2,578 reference words)

| Model | CER | TER | CER after autocorrect | TER after autocorrect |
| --- | --- | --- | --- | --- |
| typhoon_asr | 0.108 | 0.170 | 0.108 | 0.170 |
| typhoon_nemotron_asr | 0.122 | 0.215 | 0.123 | 0.213 |

| Model | Mistakes | Fixed | Changed, still wrong | Left | Correct words broken | Turns changed |
| --- | --- | --- | --- | --- | --- | --- |
| typhoon_asr | 187 | 0 | 3 | 184 | 0 | 3 of 124 |
| typhoon_nemotron_asr | 234 | 0 | 4 | 230 | 0 | 4 of 124 |

The seven changes it made:

| Reference | ASR | After autocorrect | Call, turn |
| --- | --- | --- | --- |
| ค่าน้ำ | ขนาม | ขนาด | fake_authority_00001, 2 |
| ดิฉันนภัสสร | ดีฉันผสอน | ดีผันผ่อน | fake_authority_00002, 1 |
| พลัสอินเวสต์โทรมาแจ้ง... | ภาพInvestTomorrowแจ๊ค... | ...แจ๊ด... | benign_00002, 0 |
| อ้าว 5,000 | เอ้าห้าศูนย์ | เจ้าหน้า | lottery_00002, 3 (nemotron) |
| นภัสสร | พัทสอน | พังพอน | romance_00000, 0 (nemotron) |
| อ้าวผม | เอ้าผม | เข้าผม | romance_00002, 4 (nemotron) |
| ศุลกากร | ศูนย์กรากรณ์ | ศูนย์กลาง | romance_teacher_cm_00050, 6 (nemotron) |

Most remaining mistakes are real words in the wrong place (homophones and near sounds), names,
numbers read as words, English brand names, and variant spellings (แพ็คเกจ for แพ็กเกจ, โปรโมชั่น
for โปรโมชัน, เช็ค for เช็ก). A spell checker that only knows which words exist cannot fix the
first four, and the last is not an error.

### 24 ASR mistakes (typhoon_asr)

| # | Reference | ASR | Call, turn |
| --- | --- | --- | --- |
| 1 | มีธุระ | มิธุรัตน์ | benign_00000, 0 |
| 2 | โกลบอล | โกนบ่อน | benign_00001, 0 |
| 3 | แอป | แอร์ | benign_00001, 2 |
| 4 | ฐานข้อมูณลูกค้า | ฐานข้อมูลค้า | benign_00002, 2 |
| 5 | ว่าที่ | กวาดที่ | benign_00002, 4 |
| 6 | กรมศุลกากร | กรมศูนย์ลากร | fake_authority_00000, 1 |
| 7 | ตัวตน | ทุกต้น | fake_authority_00000, 1 |
| 8 | ค่าน้ำ | ขนาม | fake_authority_00001, 2 |
| 9 | 85,000 | แปดสิบห้าศูนย์ | fake_authority_00001, 2 |
| 10 | แอปขัดข้อง | Accuscon | fake_authority_00001, 4 |
| 11 | แอปดูเอง | Adduan | fake_authority_00001, 5 |
| 12 | แอดไลน์ | แปดไลท์ | fake_authority_00001, 7 |
| 13 | https://refund-secure50.net | ATPSREAFunceUREห้าสิบเน็ต | fake_authority_00001, 8 |
| 14 | นภัสสร | พระสอน | fake_authority_00002, 0 |
| 15 | อย่าโทร | อยากถูก | fake_authority_00002, 4 |
| 16 | ลับ | รัก | fake_authority_00002, 4 |
| 17 | 158-5-96045-2 | หนึ่งร้อยห้าสิบแปดลบห้าลบเก้าหมื่นหกพันสี่สิบห้าลบสอง | fake_authority_00002, 4 |
| 18 | เจ้าหน้า | เจ้าน้า | fake_authority_00002, 6 |
| 19 | ดิฉัน | ติฉัน | fake_authority_00002, 7 |
| 20 | ทรัพย์ทวีแคปปิตอล | Subviแคบติร | investment_00000, 0 |
| 21 | 30% | สามสิบ | investment_00000, 0 |
| 22 | เคารพ | เข้ารบ | investment_00000, 2 |
| 23 | กี่ | ขีด | investment_00000, 3 |
| 24 | หรอก | หลอก | investment_00001, 5 |

The full list (every mistake, with the status after autocorrect) is in
`eval/results/autocorrect.json` under `asr.models.<model>.cases`.

### Latency (per turn, 124 ASR outputs)

| Model output | Mean | p95 | Max | Warm cache mean |
| --- | --- | --- | --- | --- |
| typhoon_asr | 0.18 s | 0.83 s | 6.9 s | 0.0004 s |
| typhoon_nemotron_asr | 0.09 s | 0.77 s | 2.1 s | 0.0003 s |

The first call in a process took 1.5 s (dictionary load). Each new unknown word costs about
0.4 s to 0.5 s in `pythainlp.spell.correct`, which is what drives p95 and max. A live turn
should not wait that long. Timings vary by a few percent between runs.

## Known bad fix from PR #35, re-checked

`pythainlp.spell.correct("โอนเงนิ")` returns `ถอนเงิน` and `autocorrect_text("โอนเงนิ", "pythainlp")`
still returns `ถอนเงิน`: a typo of "transfer money" becomes "withdraw money". Not fixed by the
guards added in PR #35 (those cover shortening, names and digits, not a wrong near word).

## Caveats

- The audio is TTS. It is clean speech with no line noise, codec loss, overlap or accents. Real
  phone recordings will have higher CER and TER, and more misspelled or invented words, so the
  number of mistakes a corrector could in principle fix is probably larger than here. The result
  that matters is the fixed-to-broken ratio, and that was 0 fixed in 187 here.
- 124 turns from 20 calls, written by a language model, is a small sample. The reference text is
  the exact text sent to the TTS engine.
- The type model is scored on 39 dialogues of two types, and the gate on 55, so one
  dialogue moves macro-F1 by several points. On the ASR calls (20 for the gate, 7 for the type)
  the sample is smaller still. The identical test scores come from 0 changed dialogues.
- The reference text uses variant spellings the ASR does not (แพ็กเกจ, โปรโมชัน). Those count as
  mistakes here and inflate the mistake count slightly.
- Entity recall is 0 for both settings, so this evaluation cannot show whether autocorrect helps
  or hurts entities on ASR text. Reading numbers written as words is a separate fix.
