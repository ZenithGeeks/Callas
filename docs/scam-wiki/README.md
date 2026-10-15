# Scam wiki

One page per scam type holds the notes about how that scam works right now. The pages are the
source of truth. The scam-call simulator reads them through the SQLite store, so a change on a
page does nothing until you sync it.

Pages: `fake_authority.md`, `romance.md`, `investment.md`, `phishing_otp.md`, `lottery.md`.

## Page format

```
# Phishing / OTP

Any text here, before the first "##", is ignored.

## One-line summary of the pattern
- source: URL or name of where you read it (may be empty)
- published: 2026-01-31 (YYYY-MM-DD, may be empty)
- active: yes

Tactics:
- What the scammer does, one tactic per line.

Phrases:
- A typical Thai phrase, with slots like {OTP} {URL} {MONEY} {BANK_ACCOUNT}
```

Rules:

- Each `## ` heading starts one note. The heading is the note's summary.
- A note needs a summary and at least one tactic. `Phrases:` and the three `- key:` lines are optional.
- `active: no` keeps the note on the page but stops the simulator from using it.
- Keys are not case sensitive. Bullets can start with `-` or `*`.
- One line per bullet. Do not wrap a bullet onto a second line.
- Write tactics only. No real names, phone numbers, account numbers or links. Use the slots
  `{CALLER_NAME} {RECEIVER_NAME} {ORG} {PHONE} {BANK_ACCOUNT} {OTP} {URL} {MONEY}` in phrases.
- If a page cannot be read, `sync` stops and names the file and line. Nothing is changed in the
  store when any page fails.

## Workflow

Edit a page by hand, then run from the repo root:

```
python -m data_pipeline.patterns sync
```

`sync` makes the store match the pages exactly. A note you delete from a page is deleted from the
store. It prints the number of notes and active notes for each type.

To add a note from an article with the LLM extractor:

```
python -m data_pipeline.patterns add --type romance --file article.txt --source https://... --published 2026-01-31
```

This appends a note to the page and syncs that type. Read the new note on the page, fix it if
needed, and commit the page. A scraper can use the same function, `wiki.add_note_to_page`.

Other commands: `list --type romance` shows the numbered notes on a page. `disable --type romance 2`
sets `active: no` on note 2 and syncs.

The seed notes give `general public warnings (seed example)` as their source. Replace them as
real notes come in.
