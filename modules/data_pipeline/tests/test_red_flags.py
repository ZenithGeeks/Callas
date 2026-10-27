"""red_flags.digest builds a short, blind-safe list of scam warning signs for the receiver.

It is the union of one active note per scam-type page, so every receiver reads the same thing and
never learns this call's own label.
"""

from data_pipeline import red_flags


def test_the_real_wiki_gives_a_short_non_empty_digest():
    lines = red_flags.digest().splitlines()
    assert lines, "the shipped wiki should yield some warning signs"
    assert len(lines) <= red_flags.MAX_FLAGS
    assert all(line.startswith("- ") for line in lines)


def test_the_digest_is_the_same_for_every_call_so_the_receiver_stays_blind():
    # This is the blind-to-label guarantee: digest() takes no call, so every receiver reads the
    # identical union of all types and nothing tells it which kind of call this is. (A plain word
    # like "investment" may appear inside a general warning; that is not a per-call signal.)
    assert red_flags.digest() == red_flags.digest()


def test_the_digest_carries_no_internal_type_key():
    # the snake_case page keys are identifiers, never prose, so they must not leak into the digest
    lowered = red_flags.digest().lower()
    assert "phishing_otp" not in lowered
    assert "fake_authority" not in lowered


def test_a_missing_wiki_folder_gives_an_empty_digest_not_an_error():
    # advisory, not essential: a wiki problem must not break a run
    assert red_flags.digest("/no/such/folder") == ""


def test_one_active_summary_per_page_skipping_inactive_sorted(tmp_path):
    (tmp_path / "phishing_otp.md").write_text(
        "# P\n\n## Zeta active phishing note\nTactics:\n- t\n\n"
        "## Alpha inactive note\n- active: no\nTactics:\n- t\n",
        encoding="utf-8",
    )
    (tmp_path / "lottery.md").write_text(
        "# L\n\n## Beta active lottery note\nTactics:\n- t\n", encoding="utf-8"
    )
    # one summary per page (PER_TYPE=1), the inactive note skipped, result sorted
    assert red_flags.digest(tmp_path) == "- Beta active lottery note\n- Zeta active phishing note"
