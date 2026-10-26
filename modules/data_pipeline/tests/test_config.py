import shutil

import pytest
import yaml

from data_pipeline.agents import SIGNATURES, configured
from data_pipeline.config import DEFAULT_DIR, ConfigError, load_config
from data_pipeline.synthesize import main


@pytest.fixture
def cfg_dir(tmp_path):
    d = tmp_path / "team_cfg"
    shutil.copytree(DEFAULT_DIR, d)
    return d


def edit(folder, name, change):
    path = folder / name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_default_config_loads_and_matches_signatures():
    c = load_config()
    for name in SIGNATURES:
        configured(name, getattr(c.agents, name))
    assert c.name == "config" and len(c.hash) == 12


def test_instructions_and_descriptions_come_from_yaml(cfg_dir):
    edit(cfg_dir, "agents.yaml", lambda d: d["receiver"].update(instructions="BE A CAT"))
    edit(cfg_dir, "agents.yaml", lambda d: d["receiver"]["fields"].update(line="meow only"))
    sig = configured("receiver", load_config(cfg_dir).agents.receiver)
    assert sig.instructions == "BE A CAT"
    assert sig.output_fields["line"].json_schema_extra["desc"] == "meow only"


def test_renamed_field_is_rejected(cfg_dir):
    def rename(d):
        d["judge"]["fields"]["verdict"] = d["judge"]["fields"].pop("label_holds")

    edit(cfg_dir, "agents.yaml", rename)
    with pytest.raises(ConfigError, match="missing.*label_holds"):
        configured("judge", load_config(cfg_dir).agents.judge)


def test_unknown_key_is_rejected(cfg_dir):
    edit(cfg_dir, "agents.yaml", lambda d: d["receiver"].update(mood="grumpy"))
    with pytest.raises(ConfigError, match="mood"):
        load_config(cfg_dir)


def test_unknown_slot_in_scenario_is_rejected(cfg_dir):
    edit(cfg_dir, "scenarios.yaml", lambda d: d["legit"][0].update(goal="visit {HOUSE_NUMBER}"))
    with pytest.raises(ConfigError, match="HOUSE_NUMBER"):
        load_config(cfg_dir)


def test_scam_types_are_fixed(cfg_dir):
    edit(cfg_dir, "scenarios.yaml", lambda d: d["scams"].pop("romance"))
    with pytest.raises(ConfigError, match="scams needs exactly"):
        load_config(cfg_dir)


def test_one_voice_per_gender_is_not_enough(cfg_dir):
    edit(cfg_dir, "agents.yaml", lambda d: d["tts"]["voices"].update(male=["th_m_1"]))
    with pytest.raises(ConfigError, match="at least 2 voices"):
        load_config(cfg_dir)


def test_hash_follows_content_not_comments(cfg_dir):
    before = load_config(cfg_dir).hash
    path = cfg_dir / "details.yaml"
    path.write_text("# a new comment\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_config(cfg_dir).hash == before
    edit(cfg_dir, "details.yaml", lambda d: d["banks"].append("ธนาคารใหม่"))
    assert load_config(cfg_dir).hash != before


def test_check_config_cli(cfg_dir, capsys):
    assert main(["--check-config", "--config", str(cfg_dir)]) == 0
    assert "team_cfg ok" in capsys.readouterr().out
    edit(cfg_dir, "agents.yaml", lambda d: d["judge"]["intended"].pop("benign"))
    assert main(["--check-config", "--config", str(cfg_dir)]) == 2
    assert "intended" in capsys.readouterr().err


# ---- checks made when the config loads -------------------------------------------------------


@pytest.mark.parametrize("name", ["before", "start", "push", "last"])
@pytest.mark.parametrize("text", ["turn {tern} of {turns}", "oops {", "empty {}", "stray }"])
def test_pacing_text_must_format(cfg_dir, name, text):
    edit(cfg_dir, "agents.yaml", lambda d: d["scam_caller"]["pacing"].update({name: text}))
    with pytest.raises(ConfigError, match=f"pacing.{name}"):
        load_config(cfg_dir)


@pytest.mark.parametrize("name", ["stay", "free"])
@pytest.mark.parametrize("text", ["turn {nope}", "of {turns}", "oops {"])
def test_stage_text_must_format(cfg_dir, name, text):
    edit(cfg_dir, "agents.yaml", lambda d: d["receiver"]["stage"].update({name: text}))
    with pytest.raises(ConfigError, match=f"stage.{name}"):
        load_config(cfg_dir)


def test_a_pacing_text_with_both_fields_and_an_escaped_brace_loads(cfg_dir):
    text = "turn {turn} of {turns} {{literal}}"
    edit(cfg_dir, "agents.yaml", lambda d: d["scam_caller"]["pacing"].update(before=text))
    assert load_config(cfg_dir).agents.scam_caller.pacing.before == text


@pytest.mark.parametrize("age", [[5, 30], [40, 30], [20, 120], [9, 100]])
def test_receiver_age_range_is_checked(cfg_dir, age):
    edit(cfg_dir, "scenarios.yaml", lambda d: d["receivers"].update(age=age))
    with pytest.raises(ConfigError, match="receivers.age"):
        load_config(cfg_dir)


def test_receiver_age_range_edges_load(cfg_dir):
    edit(cfg_dir, "scenarios.yaml", lambda d: d["receivers"].update(age=[10, 100]))
    assert load_config(cfg_dir).scenarios.receivers.age == (10, 100)


def test_two_names_must_be_different(cfg_dir):
    edit(cfg_dir, "details.yaml", lambda d: d["names"].update(male=["สมชาย", "สมชาย"]))
    with pytest.raises(ConfigError, match="names.male needs at least 2 different"):
        load_config(cfg_dir)


def test_two_voices_must_be_different(cfg_dir):
    edit(cfg_dir, "agents.yaml", lambda d: d["tts"]["voices"].update(male=["th_m_1", "th_m_1"]))
    with pytest.raises(ConfigError, match="at least 2 voices"):
        load_config(cfg_dir)


def test_reveal_turn_must_leave_room_to_ask(cfg_dir):
    # an 18-turn call gives the caller own turns 0-8; its last two are "make the ask now",
    # so the latest reveal is turn 7. Reveal 8 can never be followed by an ask.
    edit(cfg_dir, "scenarios.yaml", lambda d: d["calls"].update(reveal_turn_fast=[0, 8]))
    with pytest.raises(ConfigError, match=r"reveal_turn_fast has 8.*latest reveal turn there: 7"):
        load_config(cfg_dir)
    edit(cfg_dir, "scenarios.yaml", lambda d: d["calls"].update(reveal_turn_fast=[0, 7]))
    load_config(cfg_dir)


def test_reveal_turn_is_checked_against_a_short_max_turns(cfg_dir):
    def shorter(d):
        d["calls"].update(max_turns=[6, 8], reveal_turn_slow=[3], reveal_turn_fast=[0, 1])

    edit(cfg_dir, "scenarios.yaml", shorter)
    with pytest.raises(ConfigError, match=r"reveal_turn_slow has 3"):
        load_config(cfg_dir)
