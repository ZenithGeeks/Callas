"""How long a receiver stays on a cold call should mirror why people get scammed: someone who
does not know the pattern, trusts easily, or is overconfident keeps talking; a scam-aware,
skeptical person cuts the call early. stay_on_line_for encodes that."""

from data_pipeline.config import ReceiverProfile, load_config
from data_pipeline.turns import stay_on_line_for

CONFIG = load_config()
RECEIVER = CONFIG.agents.receiver
FLOOR = RECEIVER.stay_on_line


def _persona(temperament: str, scam_awareness: str) -> ReceiverProfile:
    return ReceiverProfile(
        key="",
        gender="female",
        age=50,
        occupation="x",
        education="x",
        region="x",
        family="x",
        digital_literacy="x",
        scam_awareness=scam_awareness,
        temperament=temperament,
        ending="x",
        speech="x",
    )


def test_trusting_unaware_stays_longer_than_skeptical_aware():
    unaware = stay_on_line_for(
        RECEIVER,
        _persona(
            "trusting: believes people who sound official", "has never heard of call-center scams"
        ),
    )
    aware = stay_on_line_for(
        RECEIVER,
        _persona("skeptical: asks who is calling", "knows the police warnings and checks numbers"),
    )
    assert unaware > aware


def test_overconfident_person_engages_more_than_a_careful_one():
    overconfident = stay_on_line_for(
        RECEIVER,
        _persona(
            "trusting: polite to officials",
            "has seen TV warnings but believes she would notice a scam",
        ),
    )
    careful = stay_on_line_for(
        RECEIVER,
        _persona("skeptical: refuses on the phone", "was scammed before and is now very careful"),
    )
    assert overconfident > careful


def test_everyone_stays_at_least_the_configured_floor():
    for temperament, awareness in [
        ("skeptical: refuses on the phone", "was scammed before and is now very careful"),
        ("busy: wants the call over", "knows the police warnings and checks numbers"),
        ("trusting: follows instructions", "has never heard of call-center scams"),
    ]:
        assert stay_on_line_for(RECEIVER, _persona(temperament, awareness)) >= FLOOR
