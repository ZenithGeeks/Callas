from pathlib import Path

import pytest

from data_pipeline import patterns
from data_pipeline.store import Store
from data_pipeline.wiki import (
    WikiError,
    add_note_to_page,
    parse_page,
    read_wiki,
    render_page,
    set_active,
    sync,
)
from nlp_core.types import SCAM_TYPES

SEED_FOLDER = Path(__file__).resolve().parents[3] / "docs" / "scam-wiki"

ODD_PAGE = """# Romance
Intro text that the parser ignores.
- a bullet in the intro is ignored too


## Partner asks for money for a flight
   - SOURCE:   a police warning
- Published: 2026-02-03


-   ACTIVE: No

tactics:
* builds trust for weeks


*  asks for a ticket fee
PHRASES:
* ส่งเงิน {MONEY} มาให้หน่อยนะ

## Second note
- active: yes
Tactics:
- one tactic
"""


def note(summary="A summary", **extra):
    base = {
        "summary": summary,
        "source": "",
        "published": None,
        "active": True,
        "tactics": ["tactic one"],
        "phrases": [],
    }
    return {**base, **extra}


def test_parse_page_with_odd_spacing_and_star_bullets():
    first, second = parse_page(ODD_PAGE)
    assert first == {
        "summary": "Partner asks for money for a flight",
        "source": "a police warning",
        "published": "2026-02-03",
        "active": False,
        "tactics": ["builds trust for weeks", "asks for a ticket fee"],
        "phrases": ["ส่งเงิน {MONEY} มาให้หน่อยนะ"],
    }
    assert second["summary"] == "Second note"
    assert second["source"] == "" and second["published"] is None and second["active"] is True
    assert second["phrases"] == []


def test_render_then_parse_round_trips():
    notes = [
        note("First", source="https://example.org", published="2026-01-31", phrases=["สวัสดี {ORG}"]),
        note("Second", active=False, tactics=["a", "b"]),
    ]
    text = render_page("Phishing / OTP", notes, intro="Some intro.")
    assert text.startswith("# Phishing / OTP\n")
    assert "\r" not in text
    assert parse_page(text) == notes


def test_note_without_tactics_names_file_and_line():
    text = "# T\n\n## Fine\nTactics:\n- x\n\n## Broken\n- source: s\n"
    with pytest.raises(WikiError, match=r"romance\.md:7: .*no tactics"):
        parse_page(text, "romance.md")


def test_note_without_summary_names_file_and_line():
    with pytest.raises(WikiError, match=r"p\.md:3: .*no summary"):
        parse_page("# T\n\n##\nTactics:\n- x\n", "p.md")


def test_bad_active_value_names_line():
    with pytest.raises(WikiError, match=r"p\.md:2: active must be yes or no"):
        parse_page("## S\n- active: maybe\nTactics:\n- x\n", "p.md")


def test_bad_date_names_line():
    with pytest.raises(WikiError, match=r"p\.md:2: published must be"):
        parse_page("## S\n- published: 3 Feb\nTactics:\n- x\n", "p.md")


def test_add_note_to_page_creates_then_appends(tmp_path):
    path = add_note_to_page(tmp_path, "romance", note("First"))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Romance\n")
    assert [n["summary"] for n in parse_page(text)] == ["First"]

    add_note_to_page(tmp_path, "romance", note("Second", phrases=["p"]))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Romance\n")
    assert [n["summary"] for n in parse_page(text)] == ["First", "Second"]
    assert b"\r" not in path.read_bytes()


def test_add_note_to_page_rejects_note_without_tactics(tmp_path):
    with pytest.raises(WikiError):
        add_note_to_page(tmp_path, "romance", note(tactics=[]))
    assert not (tmp_path / "romance.md").exists()


def test_sync_makes_store_match_pages_and_drops_deleted_notes(tmp_path):
    store = Store(tmp_path / "calls.db")
    wiki = tmp_path / "wiki"
    add_note_to_page(wiki, "lottery", note("One", published="2026-01-01"))
    add_note_to_page(wiki, "lottery", note("Two", published="2026-02-01", phrases=["p"]))
    store.add_note("lottery", "Old note only in the store", ["t"], [])
    store.add_note("romance", "Unrelated type", ["t"], [])

    counts = sync(store, wiki, ("lottery",))
    assert counts == {"lottery": {"notes": 2, "active": 2}}
    assert [n["summary"] for n in store.notes("lottery")] == ["Two", "One"]
    assert store.notes("lottery")[0]["phrases"] == ["p"]
    assert len(store.notes("romance")) == 1  # types not synced are left alone

    page = wiki / "lottery.md"
    page.write_text(render_page("Lottery", [note("Two")]), encoding="utf-8")
    assert sync(store, wiki, ("lottery",))["lottery"]["notes"] == 1
    assert [n["summary"] for n in store.notes("lottery")] == ["Two"]


def test_inactive_notes_are_stored_but_not_returned(tmp_path):
    store = Store(tmp_path / "calls.db")
    wiki = tmp_path / "wiki"
    add_note_to_page(wiki, "investment", note("On"))
    add_note_to_page(wiki, "investment", note("Off", active=False))

    counts = sync(store, wiki)
    assert counts["investment"] == {"notes": 2, "active": 1}
    assert [n["summary"] for n in store.notes("investment")] == ["On"]
    rows = store._conn().execute("SELECT active FROM pattern_notes ORDER BY id").fetchall()
    assert [r["active"] for r in rows] == [1, 0]


def test_sync_with_a_broken_page_changes_nothing(tmp_path):
    store = Store(tmp_path / "calls.db")
    wiki = tmp_path / "wiki"
    add_note_to_page(wiki, "lottery", note("Kept"))
    sync(store, wiki)
    (wiki / "romance.md").write_text("## No tactics here\n", encoding="utf-8")
    with pytest.raises(WikiError, match=r"romance\.md:1"):
        sync(store, wiki)
    assert [n["summary"] for n in store.notes("lottery")] == ["Kept"]


def test_sync_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sync(Store(tmp_path / "calls.db"), tmp_path / "nope")


def test_set_active_changes_only_that_line(tmp_path):
    path = tmp_path / "romance.md"
    path.write_text(ODD_PAGE, encoding="utf-8")
    set_active(tmp_path, "romance", 2, False)
    set_active(tmp_path, "romance", 1, True)
    before, after = ODD_PAGE.splitlines(), path.read_text(encoding="utf-8").splitlines()
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    assert changed == [("-   ACTIVE: No", "-   ACTIVE: yes"), ("- active: yes", "- active: no")]
    assert [n["active"] for n in parse_page("\n".join(after))] == [True, False]


def test_cli_sync_and_disable(tmp_path, capsys):
    wiki, db = tmp_path / "wiki", tmp_path / "calls.db"
    add_note_to_page(wiki, "phishing_otp", note("Only"))
    base = ["--db", str(db), "--wiki", str(wiki)]
    assert patterns.main([*base, "sync"]) == 0
    out = capsys.readouterr().out
    assert "phishing_otp: 1 notes, 1 active" in out
    assert "lottery: 0 notes, 0 active" in out
    assert patterns.main([*base, "disable", "--type", "phishing_otp", "1"]) == 0
    assert Store(db).notes("phishing_otp") == []
    assert "active: no" in (wiki / "phishing_otp.md").read_text(encoding="utf-8")


def test_seeded_pages_parse():
    wiki = read_wiki(SEED_FOLDER)
    assert set(wiki) == set(SCAM_TYPES)
    for scam_type, notes in wiki.items():
        assert 1 <= len(notes) <= 2, scam_type
        for n in notes:
            assert n["tactics"] and n["active"] and n["published"] is None
            assert n["source"] == "general public warnings (seed example)"
