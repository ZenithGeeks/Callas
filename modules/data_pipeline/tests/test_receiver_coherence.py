"""The sampled receiver (the victim) must be a believable person: their age, job and household
must not contradict each other. Independent sampling once gave calls like a 20-year-old who
"lives with grown children" or a 78-year-old "university student"; coherent life-stage sampling
fixes that. The rules below flag only undeniable contradictions, so variety is kept.
"""

from data_pipeline.config import CALL_TYPES, load_config
from data_pipeline.scenarios import ReceiverProfile, sample_config

CONFIG = load_config()


def incoherences(r: ReceiverProfile) -> list[str]:
    """Undeniable age-vs-life contradictions in one receiver. Empty list means coherent."""
    problems = []
    family = r.family.lower()
    occupation = r.occupation.lower()
    if ("grown child" in family or "grandchild" in family) and r.age < 40:
        problems.append(f"age {r.age} but family={r.family!r}")
    if "elderly parent" in family and r.age > 65:
        problems.append(f"age {r.age} but family={r.family!r}")
    if "university student" in occupation and r.age > 30:
        problems.append(f"age {r.age} but occupation={r.occupation!r}")
    if "retired" in occupation and r.age < 55:
        problems.append(f"age {r.age} but occupation={r.occupation!r}")
    return problems


def test_sampled_receivers_are_coherent():
    bad = []
    for call_type in CALL_TYPES:
        for seed in range(500):
            receiver = sample_config(call_type, seed, CONFIG).receiver
            for problem in incoherences(receiver):
                bad.append(f"{call_type}:{seed} {problem}")
    assert not bad, f"{len(bad)} incoherent receivers, e.g. " + "; ".join(bad[:5])
