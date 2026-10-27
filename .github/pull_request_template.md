<!--
Title: type(#issue): lower-case summary, e.g. fix(#99): keep the scam flag after the caller hangs up
Delete these comments and any section that does not apply.
-->

Closes #

## What and why

<!-- What changes for a user or a teammate, and why. One short paragraph. -->

## How I checked it

<!-- Tick only what you ran. Paste the counts. -->

- [ ] `pytest -q`: N passed
- [ ] `ruff check .` clean, `ruff format modules backend` run
- [ ] Tried it by hand: what you did and what you saw

## Prompt, agent or judge change

<!-- Only if a prompt, DSPy module, agent or the judge changed. -->

- Prompt size before and after, in tokens:
- Results before and after, from a script that writes to `eval/results/`:
- A few generated lines or calls, before and after:

## Checklist

- [ ] One change per PR. Unrelated work goes in its own PR.
- [ ] Branched from `main`. If it builds on another open PR, name it and the merge order.
- [ ] A bug fix or new rule has a test that fails without it.
- [ ] No `.env`, keys, model files, audio, `data/raw/` or `data/synthetic/`.
- [ ] Call text does not go to an outside API ([safety rules](https://github.com/ZenithGeeks/Callas/blob/main/README.md#safety-rules)).
- [ ] Line endings stay LF. No whole-file diffs.
- [ ] If the reply prompt or `tts_voice_agent/templates.py` changed, the team signed off.
