# nlp_core

Text work shared by the web system, the classifier notebook and the data pipeline.

| File | What it does |
|---|---|
| `text.py` | `normalize_text` (also used by the classifier notebook) and the optional `autocorrect_text` |
| `numbers.py` | `spoken_numbers_to_digits`: Thai number words back to digits (issue #53) |
| `entities.py` | `extract_entities`: rule-based phone, bank account, OTP, URL and money |
| `registry.py` | Swappable NER, classifier and summarizer, chosen by `NLP_NER`, `NLP_CLASSIFIER`, `NLP_SUMMARIZER` |
| `parts.py` | The built-in parts: `RulesNER`, `LLMClassifier`, `LLMSummarizer` (also importable from `registry.py`) |
| `finetuned.py` | `FineTunedClassifier`: the notebook's gate and type models, each fed the live call the way the notebook wrote its rows. Registry names `best` (WangchanBERTa gate, PhayaThaiBERT type), `phayathaibert`, `wangchanberta`, `bge_m3`, or any `<gate>+<type>` pair such as `bge_m3+phayathaibert` |
| `interfaces.py` | The three protocols a fine-tuned model has to implement |
| `analyzer.py` | `Analyzer`: rule NER plus one LLM for the summary and the scam label |
| `types.py` | `Entity`, `NLPResult`, `SCAM_TYPES`, `ENTITY_LABELS` |

Each turn is cleaned in `tts_voice_agent.memory.clean_turn`:
`normalize_text`, then `spoken_numbers_to_digits`, then `autocorrect_text`.

## How spoken numbers are read

The ASR writes numbers the way they are said, and the rule NER (`entities.py`) only reads digits.
`spoken_numbers_to_digits` turns such words back into digits. How a number is read depends on
what it is, so the rule follows that. Everything that is not a number below is left as it is.

1. Identifiers (phone, bank account, OTP, ID card) are read digit by digit, with ขีด for a
   dash: "หนึ่งห้าแปดขีดห้าขีดเก้าหกศูนย์สี่ห้าขีดสอง" is 158-5-96045-2. Digit words are
   ศูนย์ หนึ่ง สอง/โท สาม สี่ ห้า หก เจ็ด แปด เก้า (เอ็ด as 1). Speakers also pause between
   groups or run everything together, so spaces between digit words are dropped
   ("หนึ่งห้าแปด ห้า" is 1585) and only ขีด/ลบ becomes "-". At least `MIN_DIGITS` (4) digit
   words in a row, or groups joined by ขีด/ลบ with 6 digits or more in all. "สองคน",
   "สามสี่วัน" and "ห้าลบสอง" stay. Known trade-off: a run of 4 or more digit words converts even
   before a counter ("สองสามสี่ห้าวัน" becomes "2345 วัน"); an extra entity is safer than a
   missed phone number.
2. Money is read with place value (สิบ ร้อย พัน หมื่น แสน ล้าน, ยี่สิบ, สิบเอ็ด): it is
   converted when a unit follows (บาท, ดอลลาร์, ...) or a money word comes right before it
   (เงิน โอน ยอด ค่า ราคา จำนวน) and no counter follows (คน ปี วัน เดือน ครั้ง ชั้น ตัว ชิ้น ข้อ
   ที่: "จำนวนสามสิบคน" stays). "โอนหนึ่งร้อยห้าสิบแปดบาท" is "โอน 158 บาท". A bare multiplier
   counts as 1 ("ล้านบาท", "สามร้อยสิบบาท" = 310). A single digit word before a unit ("ห้าบาท")
   converts too, but เอ็ด alone does not. Two numbers split by a space ("สอง สามร้อยบาท")
   are never merged: that would invent an amount, so the pair stays words. A range written
   with a hyphen ("ห้าสิบ-หกสิบบาท") converts neither half.
3. A place-value number with no money context stays words: "สามสิบคน", "สิบวัน", "ร้อยละห้า".
   The one exception is a code: after รหัส / OTP / โค้ด a place-value value of 1000 or more
   converts, as an ASR may read an OTP like a number.
4. Low priority fallback: place-value groups joined by ลบ/ขีด ("หนึ่งร้อยห้าสิบแปดลบห้าลบ...",
   the form our own TTS produced) need 3 groups and 6 digits, or 8 digits.
5. English digit words ("zero eight one ...") with at least 4 in a row.

A number word only counts when it is a whole word for PyThaiNLP's newmm tokenizer, so
หนึ่งเดียว, ร้อยละ, สามารถ, สามี, ห้าง, ร้อยตำรวจ, สิบเอก, พันธุ์, โทร, ปริญญาโท are never read
as numbers. Text with no number word is returned without being tokenized.

## Add your own model

See "Add a model to nlp-core" in [`../orchestrator/README.md`](../orchestrator/README.md#add-a-model-to-nlp-core). That section also explains why the PhayaThaiBERT models are not connected yet.
