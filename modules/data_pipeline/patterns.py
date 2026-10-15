"""Scam pattern notes: what real scams of each type look like right now, for the scam caller.

The notes live in a Markdown wiki, one page per scam type (docs/scam-wiki/). People read and edit
the pages in GitHub. `sync` copies them into the SQLite store, and the store is what the planner
reads. The flow has three steps. Only the first is left for the team to build.

1. Get article text. (Team: a scraper for news and police warning sites. Anything that ends
   with plain text works: a file, stdin, or a function call.)
2. The pattern extractor (agents.ExtractScamPattern, a DSPy ChainOfThought whose text is in
   config/agents.yaml) reads one article and writes a note: a short
   summary, the tactics, and typical phrases. `add` appends the note to the type's wiki page and
   syncs that type into the store.
3. Before each scam call, `PlanScamCall` (in agents.py) reads the newest notes for that type and
   writes the caller's plan for the call. The scam caller sees that plan on every turn.

    python -m data_pipeline.patterns add --type romance --file article.txt --source https://...
    python -m data_pipeline.patterns sync
    python -m data_pipeline.patterns list --type romance
    python -m data_pipeline.patterns disable --type romance 2

The wiki is the source of truth: `sync` makes the store match the pages, so a note deleted from a
page is deleted from the store. To stop using a note, set `active: no` on its page (or run
`disable`, which changes only that line) and run `sync`.

Notes describe tactics, never real people: the extractor is told to drop names, numbers, accounts
and links, and a note keeps only the summary, tactics and phrases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dspy

from nlp_core.types import SCAM_TYPES

from .budget import cut_article, prompt_tokens
from .store import DEFAULT_DB, Store
from .wiki import DEFAULT_WIKI, add_note_to_page, read_wiki, set_active, sync

# Upper bound on the article, about 4k tokens of Thai. The context window (budget.py) can cut
# it shorter, never longer. Longer articles are cut, not skipped.
MAX_ARTICLE_CHARS = 12_000


def render_notes(notes: list[dict]) -> str:
    """Notes as plain text for the planner's prompt."""
    if not notes:
        return "(no recent notes for this scam type)"
    blocks = []
    for n in notes:
        lines = [f"- {n['summary']}"]
        lines += [f"  tactic: {t}" for t in n["tactics"]]
        lines += [f"  phrase: {p}" for p in n["phrases"]]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def extract(
    article: str, scam_type: str, program: dspy.Module, lm, budget: int | None = None
) -> dict | None:
    """Run the extractor on one article. Returns the note fields, or None if not relevant.

    `budget` is the prompt budget in tokens (budget.prompt_budget); the article is cut to the
    longest start that fits, and to MAX_ARTICLE_CHARS at most."""
    signature = getattr(getattr(program, "predict", None), "signature", None)
    if budget is not None and signature is not None:

        def fits(part: str) -> bool:
            inputs = {"scam_type": scam_type, "article": part}
            return prompt_tokens(signature, inputs) <= budget

        article = cut_article(article, fits, MAX_ARTICLE_CHARS)
    with dspy.context(lm=lm):
        pred = program(scam_type=scam_type, article=article[:MAX_ARTICLE_CHARS])
    if not pred.relevant:
        return None
    return {
        "summary": pred.summary.strip(),
        "tactics": [t.strip() for t in pred.tactics if t.strip()],
        "phrases": [p.strip() for p in pred.phrases if p.strip()],
    }


# ---- command line -----------------------------------------------------------------------------


def _add(args, store: Store) -> int:
    from llm_serving.dspy_programs import build_lm

    from .agents import configured
    from .budget import prompt_budget
    from .config import load_config

    settings = load_config(args.config).agents.pattern_extractor
    program = dspy.ChainOfThought(configured("pattern_extractor", settings))
    lm = build_lm(
        model=settings.lm.model,
        max_tokens=settings.lm.max_tokens,
        temperature=settings.lm.temperature,
    )
    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    config = load_config(args.config)
    note = extract(text, args.type, program, lm, prompt_budget(config, "pattern_extractor"))
    if note is None:
        print("not relevant to this scam type; nothing stored")
        return 1
    note = {**note, "source": args.source, "published": args.published, "active": True}
    path = add_note_to_page(args.wiki, args.type, note)
    sync(store, args.wiki, (args.type,))
    print(f"added to {path.as_posix()} and synced: {note['summary']}")
    return 0


def _sync(args, store: Store) -> int:
    for scam_type, n in sync(store, args.wiki).items():
        print(f"{scam_type}: {n['notes']} notes, {n['active']} active")
    return 0


def _list(args, store: Store) -> int:
    notes = read_wiki(args.wiki)[args.type]
    for number, n in enumerate(notes[: args.limit], start=1):
        state = "" if n["active"] else " (inactive)"
        out = (
            f"[{number}] {n['published'] or '-'} {n['source'] or '-'}{state}\n{render_notes([n])}\n"
        )
        sys.stdout.buffer.write(out.encode("utf-8"))
    return 0


def _disable(args, store: Store) -> int:
    summary = set_active(args.wiki, args.type, args.number, False)
    sync(store, args.wiki, (args.type,))
    print(f"note {args.number} of {args.type} set to 'active: no' and synced: {summary}")
    return 0


def _add_options(parser: argparse.ArgumentParser, given: bool) -> None:
    def default(value):
        return value if given else argparse.SUPPRESS

    parser.add_argument("--db", type=Path, default=default(DEFAULT_DB))
    parser.add_argument(
        "--wiki", type=Path, default=default(DEFAULT_WIKI), help="folder with the wiki pages"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default(None),
        help="config folder (default: data_pipeline/config)",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    # The options work before or after the command. After it they only override when given.
    _add_options(p, given=True)
    _add_options(common, given=False)
    sub = p.add_subparsers(dest="command", required=True)

    add = sub.add_parser(
        "add", parents=[common], help="extract a note from article text and add it to the wiki"
    )
    add.add_argument("--type", required=True, choices=SCAM_TYPES)
    add.add_argument("--file", help="UTF-8 text file; reads stdin when left out")
    add.add_argument("--source", default="", help="where the article came from (URL or name)")
    add.add_argument("--published", default=None, help="article date, YYYY-MM-DD")

    sub.add_parser("sync", parents=[common], help="make the store's notes match the wiki pages")

    show = sub.add_parser("list", parents=[common], help="show the notes on a wiki page, numbered")
    show.add_argument("--type", required=True, choices=SCAM_TYPES)
    show.add_argument("--limit", type=int, default=20)

    off = sub.add_parser(
        "disable", parents=[common], help="set 'active: no' on a note of a wiki page"
    )
    off.add_argument("--type", required=True, choices=SCAM_TYPES)
    off.add_argument("number", type=int, help="the note's number, as shown by `list`")

    args = p.parse_args(argv)
    handlers = {"add": _add, "sync": _sync, "list": _list, "disable": _disable}
    store = Store(args.db)
    try:
        return handlers[args.command](args, store)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
