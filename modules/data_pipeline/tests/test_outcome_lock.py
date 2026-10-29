"""The receiver's scam outcome is locked 50/50 by seed, and the persona is drawn to fit it — a
wary person for a "safe" call, a gullible one for "scammed" — so the person, not a forced
instruction, leads to the outcome. No call is rejected for its outcome; hand-written profiles are
left unlocked. The outcome is a stored label, hidden from the receiver's prompt.
"""

from data_pipeline.config import load_config
from data_pipeline.scenarios import sample_config

CONFIG = load_config()


def test_outcome_follows_seed_parity_for_pool_built_receivers():
    # the lock is 50/50 by seed parity over pool-built receivers (even -> scammed, odd -> safe);
    # hand-written profiles are unlocked and skipped
    scammed = safe = 0
    for seed in range(1, 201):
        r = sample_config("phishing_otp", seed, CONFIG).receiver
        if r.key:
            continue
        assert r.scam_outcome == ("scammed" if seed % 2 == 0 else "safe")
        scammed += r.scam_outcome == "scammed"
        safe += r.scam_outcome == "safe"
    assert scammed > 0 and safe > 0


def test_a_pool_built_receiver_persona_fits_its_outcome():
    recv = CONFIG.scenarios.receivers
    wary_a, wary_t = set(recv.wary_scam_awareness), set(recv.wary_temperaments)
    checked = 0
    for seed in range(1, 160):
        r = sample_config("phishing_otp", seed, CONFIG).receiver
        if r.key:  # a hand-written profile is left unlocked, so skip
            continue
        checked += 1
        if r.scam_outcome == "safe":
            assert r.scam_awareness in wary_a and r.temperament in wary_t
        else:
            assert r.scam_awareness not in wary_a and r.temperament not in wary_t
    assert checked, "expected some pool-built receivers"


def test_a_forced_outcome_overrides_parity_and_skips_profiles():
    # the studio can force an outcome; it then holds for any seed and type, always on a pool-built
    # receiver (never an unlocked profile), with a persona that fits
    recv = CONFIG.scenarios.receivers
    wary_a = set(recv.wary_scam_awareness)
    for forced, want_wary in (("scammed", False), ("safe", True)):
        for seed in range(1, 40):
            r = sample_config("phishing_otp", seed, CONFIG, outcome=forced).receiver
            assert not r.key  # forced -> pool-built, never a profile
            assert r.scam_outcome == forced
            assert (r.scam_awareness in wary_a) == want_wary


def test_hand_written_profiles_are_left_unlocked():
    # a profile is a fixed persona with its own nature; the 50/50 lock is not forced onto it
    profiles = [
        r for s in range(1, 80) if (r := sample_config("phishing_otp", s, CONFIG).receiver).key
    ]
    assert profiles and all(r.scam_outcome == "" for r in profiles)


def test_scam_outcome_is_hidden_from_the_receiver_prompt():
    # the persona drawn to fit the outcome carries the behaviour; the label itself is not shown
    r = next(
        r for s in range(1, 80) if not (r := sample_config("phishing_otp", s, CONFIG).receiver).key
    )
    assert r.scam_outcome and "scam_outcome" not in r.model_dump(exclude={"key", "scam_outcome"})
