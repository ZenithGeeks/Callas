"""Measure what the optional transcript autocorrect (AUTOCORRECT=pythainlp) does. Issue #30.

    python scripts/eval_autocorrect.py            # every part, writes eval/results/autocorrect.json
    python scripts/eval_autocorrect.py gate       # scam gate and type on data/processed/test.csv
    python scripts/eval_autocorrect.py entities   # rule NER on test.csv, autocorrect off and on
    python scripts/eval_autocorrect.py asr        # ASR mistakes, fixes, breaks, latency

Needs the trained models in models/ (not in git), the simulated-call store (--store) and pythaiasr.
Parts run alone only update their own keys in the JSON file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "modules")]

SEED = 42
DEFAULT_STORE = Path("D:/text_analytics/Callas-sim/data/synthetic")
DEFAULT_MODELS = next(  # models/ is not in git; a git worktree finds the main checkout's copy
    (
        p / "models"
        for p in [ROOT, *ROOT.parents]
        if (p / "models" / "phayathaibert-scam-gate").exists()
    ),
    ROOT / "models",
)
RESULT = ROOT / "eval" / "results" / "autocorrect.json"
ASR_MODELS = ("typhoon_asr", "typhoon_nemotron_asr")
RULE_LABELS = ("BANK_ACCOUNT", "MONEY", "OTP", "PHONE", "URL")  # what nlp_core.entities finds
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# ---------------------------------------------------------------- pure helpers (unit tested)


def edit_distance(a: list | str, b: list | str) -> int:
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rates(pairs: list[tuple[list[str], list[str]]], chars: bool) -> float:
    """Corpus error rate: total edits over total reference length. chars=True gives CER."""
    edits = total = 0
    for ref, hyp in pairs:
        r, h = ("".join(ref), "".join(hyp)) if chars else (ref, hyp)
        edits += edit_distance(r, h)
        total += len(r)
    return edits / total if total else 0.0


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int, int, int, int]]:
    """Token alignment as (tag, ref_start, ref_end, hyp_start, hyp_end); tag is equal,
    replace, delete (ASR dropped reference words) or insert (ASR added words)."""
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    return [(t, i1, i2, j1, j2) for t, i1, i2, j1, j2 in sm.get_opcodes()]


def mistakes(ref: list[str], hyp: list[str]) -> list[dict]:
    """Every place the ASR tokens differ from the reference, with hyp token positions."""
    return [
        {"ref": "".join(ref[i1:i2]), "asr": "".join(hyp[j1:j2]), "kind": t, "h1": j1, "h2": j2}
        for t, i1, i2, j1, j2 in align(ref, hyp)
        if t != "equal"
    ]


def correctness(ref: list[str], hyp: list[str]) -> list[bool]:
    """For each hyp token: True when it sits in an equal block of the alignment with ref."""
    ok = [False] * len(hyp)
    for t, _i1, _i2, j1, j2 in align(ref, hyp):
        if t == "equal":
            ok[j1:j2] = [True] * (j2 - j1)
    return ok


def score_correction(ref: list[str], asr: list[str], fixed: list[str]) -> dict:
    """What a correction did to one turn, over the ASR mistakes in it.

    fixed:   mistakes whose ASR tokens were all replaced by tokens that now match the reference
    changed_wrong: mistakes whose tokens were replaced but still do not match the reference
    left:    mistakes the correction did not touch (including words the ASR dropped)
    broken:  places where the correction changed ASR tokens that already matched the reference
    """
    out = {"fixed": [], "changed_wrong": [], "left": [], "broken": []}
    asr_ok = correctness(ref, asr)
    fixed_ok = correctness(ref, fixed)
    blocks = [b for b in align(asr, fixed) if b[0] != "equal"]
    for m in mistakes(ref, asr):
        if m["h1"] == m["h2"]:  # the ASR dropped words; correction cannot restore them
            out["left"].append(m)
            continue
        touched = [b for b in blocks if b[1] < m["h2"] and b[2] > m["h1"]]
        covered = sum(min(b[2], m["h2"]) - max(b[1], m["h1"]) for b in touched)
        if not touched:
            out["left"].append(m)
        elif covered == m["h2"] - m["h1"] and all(all(fixed_ok[b[3] : b[4]]) for b in touched):
            out["fixed"].append({**m, "now": "".join(fixed[touched[0][3] : touched[-1][4]])})
        else:
            out["changed_wrong"].append(
                {**m, "now": "".join(fixed[touched[0][3] : touched[-1][4]])}
            )
    for _t, i1, i2, j1, j2 in blocks:
        if i1 < i2 and all(asr_ok[i1:i2]):
            out["broken"].append({"was": "".join(asr[i1:i2]), "now": "".join(fixed[j1:j2])})
    return out


def prf(gold: list[tuple], pred: list[tuple]) -> dict:
    """Precision, recall and F1 over multisets of hashable items."""
    g, p = Counter(gold), Counter(pred)
    tp = sum((g & p).values())
    prec = tp / sum(p.values()) if p else 0.0
    rec = tp / sum(g.values()) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp,
        "fp": sum(p.values()) - tp,
        "fn": sum(g.values()) - tp,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def entity_key(label: str, text: str) -> tuple[str, str]:
    """(label, text) with Thai digits as Arabic and spaces, commas and dashes removed."""
    t = text.translate(THAI_DIGITS)
    return label, "".join(c for c in t if c not in " ,-")


def prf_by_label(gold: list[tuple], pred: list[tuple]) -> dict:
    out = {"overall": prf(gold, pred)}
    for label in sorted({k[0] for k in gold + pred}):
        out[label] = prf([k for k in gold if k[0] == label], [k for k in pred if k[0] == label])
    return out


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else 0.0


# ---------------------------------------------------------------- text and models


_first_call: dict = {}


def autocorrect(text: str) -> str:
    """nlp_core autocorrect with the pythainlp mode; the very first call is timed."""
    from nlp_core.text import autocorrect_text

    t0 = time.perf_counter()
    out = autocorrect_text(text, mode="pythainlp")
    if "s" not in _first_call:
        _first_call["s"] = time.perf_counter() - t0
    return out


_PUNCT = re.compile(r"[\s?!\"'()“”‘’…]+|\.{2,}")


def comparable(text: str) -> str:
    """Text for comparing ASR with the reference: normalized, with spaces and punctuation
    removed. The ASR writes no punctuation and places spaces freely, and Thai needs none, so
    they would show up as mistakes that are not spelling."""
    from nlp_core.text import normalize_text

    return _PUNCT.sub("", normalize_text(text))


def tokens(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    return [t for t in word_tokenize(comparable(text), engine="newmm") if t.strip()]


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything() -> None:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def render(turns: list[dict], fix: bool) -> str:
    """The text the model reads: one 'X: turn' line per turn, each turn normalized, and
    corrected when fix is set (the web system corrects each turn, not the whole call)."""
    from nlp_core.text import normalize_text

    lines = []
    for t in turns:
        text = normalize_text(t["text"])
        lines.append(f"{t['speaker']}: {autocorrect(text) if fix else text}")
    return "\n".join(lines)


def predict(model_dir: Path, texts: list[str], batch: int = 16) -> list[int]:
    """Argmax label ids, tokenized like the notebook (space -> <_>, max_length from
    preprocessing.json)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pre = json.loads((model_dir / "preprocessing.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device()).eval()
    space = pre["space_token"] if pre["space_token"] in tok.get_vocab() else None
    out: list[int] = []
    for i in range(0, len(texts), batch):
        chunk = [t.replace(" ", space) if space else t for t in texts[i : i + batch]]
        enc = tok(
            chunk, truncation=True, max_length=pre["max_length"], padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out += model(**enc).logits.argmax(-1).tolist()
    del model
    return out


def classification_scores(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "n": len(y_true),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
    }


# ---------------------------------------------------------------- parts


def load_test():
    import pandas as pd

    df = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    df["turns"] = df["speaker_turns"].map(json.loads)
    return df


def run_gate(models: Path) -> dict:
    seed_everything()
    df = load_test()
    off = [render(t, False) for t in df["turns"]]
    on = [render(t, True) for t in df["turns"]]
    assert off == list(df["text"]), "turn text does not rebuild test.csv text"
    changed = sum(a != b for a, b in zip(off, on, strict=True))
    res: dict = {"texts": len(df), "texts_changed_by_autocorrect": changed}

    gate_dir = models / "phayathaibert-scam-gate"
    names = json.loads((gate_dir / "config.json").read_text(encoding="utf-8"))["id2label"]
    y = ["scam" if s else "not_scam" for s in df["label_scam"]]
    res["gate"] = {}
    for name, texts in (("off", off), ("on", on)):
        pred = [names[str(i)] for i in predict(gate_dir, texts)]
        res["gate"][name] = classification_scores(y, pred)

    type_dir = models / "phayathaibert-scam-type"
    tnames = json.loads((type_dir / "config.json").read_text(encoding="utf-8"))["id2label"]
    keep = df["label_type"].isin(set(tnames.values())).to_numpy()
    yt = list(df["label_type"][keep])
    res["type"] = {"labels": sorted(tnames.values())}
    for name, texts in (("off", off), ("on", on)):
        sub = [t for t, k in zip(texts, keep, strict=True) if k]
        res["type"][name] = classification_scores(
            yt, [tnames[str(i)] for i in predict(type_dir, sub)]
        )
    res["type"]["note"] = "scored on test rows whose label_type is one of the model's labels"
    return res


def run_entities() -> dict:
    from nlp_core.entities import extract_entities

    df = load_test()
    res = {"texts": len(df), "gold_entities_in_test_csv": 0}
    found = {"off": [], "on": []}
    differ = 0
    for turns in df["turns"]:
        a = [entity_key(e.label, e.text) for t in turns for e in extract_entities(t["text"])]
        b = [
            entity_key(e.label, e.text)
            for t in turns
            for e in extract_entities(autocorrect(t["text"]))
        ]
        found["off"] += a
        found["on"] += b
        differ += Counter(a) != Counter(b)
    gold = [json.loads(x) for x in df["entities"]]
    res["gold_entities_in_test_csv"] = sum(len(g) for g in gold)
    res["found_off"] = dict(Counter(k[0] for k in found["off"]))
    res["found_on"] = dict(Counter(k[0] for k in found["on"]))
    res["texts_with_different_entities"] = differ
    res["vs_each_other"] = prf_by_label(found["off"], found["on"])
    res["note"] = (
        "The entities column of test.csv is empty for every row, so there is no gold to score "
        "against. The table compares the entities found with autocorrect on to those found "
        "with it off (off treated as the reference)."
    )
    return res


def transcribe(store: Path, rows: list, model: str, cache: Path) -> dict[str, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [r for r in rows if f"{r['call_id']}/{r['idx']}" not in have]
    if todo:
        from pythaiasr import ASR

        asr = ASR(model=model, device="auto")
        for r in todo:
            have[f"{r['call_id']}/{r['idx']}"] = str(asr(str(store / r["audio"])))
        cache.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
    return have


def run_asr(store: Path, models: tuple[str, ...], cache_dir: Path, limit: int | None) -> dict:
    from nlp_core.entities import extract_entities
    from nlp_core.text import _fix_word, normalize_text

    con = sqlite3.connect(f"file:{(store / 'calls.db').as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [
        dict(r)
        for r in con.execute(
            "select call_id, idx, speaker, text, entities, audio from turns order by call_id, idx"
        )
    ]
    con.close()
    if limit:
        rows = rows[:limit]
    res: dict = {"turns": len(rows), "calls": len({r["call_id"] for r in rows}), "models": {}}
    for model in models:
        hyp = transcribe(store, rows, model, cache_dir / f"{model}.json")
        pairs, cases, tally = [], [], Counter()
        ent_gold, ent_off, ent_on = [], [], []
        times_cold, times_warm = [], []
        broken = []
        _fix_word.cache_clear()
        for r in rows:
            key = f"{r['call_id']}/{r['idx']}"
            ref_text, asr_text = normalize_text(r["text"]), normalize_text(hyp[key])
            ref, asr = tokens(ref_text), tokens(asr_text)
            pairs.append((ref, asr))
            t0 = time.perf_counter()
            fixed_text = autocorrect(asr_text)
            times_cold.append(time.perf_counter() - t0)
            fixed = tokens(fixed_text)
            sc = score_correction(ref, asr, fixed)
            for name in ("fixed", "changed_wrong", "left"):
                tally[name] += len(sc[name])
                for m in sc[name]:
                    cases.append(
                        {
                            "call": r["call_id"],
                            "turn": r["idx"],
                            "status": name,
                            "ref": m["ref"],
                            "asr": m["asr"],
                            "after": m.get("now", m["asr"]),
                        }
                    )
            tally["broken"] += len(sc["broken"])
            broken += [{"call": r["call_id"], "turn": r["idx"], **b} for b in sc["broken"]]
            tally["turns_changed"] += fixed_text != asr_text
            ent_gold += [
                entity_key(e["label"], e["text"])
                for e in json.loads(r["entities"])
                if e["label"] in RULE_LABELS
            ]
            ent_off += [entity_key(e.label, e.text) for e in extract_entities(asr_text)]
            ent_on += [entity_key(e.label, e.text) for e in extract_entities(fixed_text)]
        for r in rows:  # second pass: every word already cached
            t0 = time.perf_counter()
            autocorrect(normalize_text(hyp[f"{r['call_id']}/{r['idx']}"]))
            times_warm.append(time.perf_counter() - t0)
        mistakes_total = tally["fixed"] + tally["changed_wrong"] + tally["left"]
        res["models"][model] = {
            "cer": error_rates(pairs, chars=True),
            "wer": error_rates(pairs, chars=False),
            "cer_after_autocorrect": None,
            "reference_words": sum(len(r) for r, _ in pairs),
            "mistakes": mistakes_total,
            "fixed": tally["fixed"],
            "changed_but_still_wrong": tally["changed_wrong"],
            "left": tally["left"],
            "correct_words_broken": tally["broken"],
            "turns_changed_by_autocorrect": tally["turns_changed"],
            "latency_s": {
                "cold_words": latency(times_cold),
                "warm_cache": latency(times_warm),
            },
            "entities": {
                "gold_in_store": len(ent_gold),
                "asr_off": prf_by_label(ent_gold, ent_off),
                "asr_autocorrect": prf_by_label(ent_gold, ent_on),
            },
            "cases": cases,
            "broken": broken,
        }
        # corpus rates after correction, same alignment settings
        after = []
        for r, (ref, _asr) in zip(rows, pairs, strict=True):
            after.append(
                (ref, tokens(autocorrect(normalize_text(hyp[f"{r['call_id']}/{r['idx']}"]))))
            )
        res["models"][model]["cer_after_autocorrect"] = error_rates(after, chars=True)
        res["models"][model]["wer_after_autocorrect"] = error_rates(after, chars=False)
    res["first_autocorrect_call_s"] = _first_call.get("s")
    res["known_bad_fix"] = known_bad_fix()
    return res


def known_bad_fix() -> dict:
    """PR #35: PyThaiNLP turned the typo of "transfer money" into "withdraw money"."""
    from pythainlp.spell import correct

    from nlp_core.text import autocorrect_text

    typo = "โอนเงนิ"
    return {
        "typo": typo,
        "pythainlp_correct": correct(typo),
        "autocorrect_text": autocorrect_text(typo, mode="pythainlp"),
    }


def latency(times: list[float]) -> dict:
    return {
        "mean": statistics.fmean(times),
        "p95": percentile(times, 0.95),
        "max": max(times),
        "turns": len(times),
    }


# ---------------------------------------------------------------- output


def show(res: dict) -> None:
    if "gate" in res:
        print(
            f"\nGate and type on test.csv ({res['texts']} texts, "
            f"{res['texts_changed_by_autocorrect']} changed by autocorrect)"
        )
        print(f"{'task':<6}{'autocorrect':<13}{'n':>4}{'macro-F1':>10}{'accuracy':>10}")
        for task in ("gate", "type"):
            for mode in ("off", "on"):
                s = res[task][mode]
                print(f"{task:<6}{mode:<13}{s['n']:>4}{s['macro_f1']:>10.3f}{s['accuracy']:>10.3f}")
    if "entities" in res:
        e = res["entities"]
        print(
            f"\nRule NER on test.csv: gold entities {e['gold_entities_in_test_csv']}, "
            f"found off {sum(e['found_off'].values())}, on {sum(e['found_on'].values())}, "
            f"texts that differ {e['texts_with_different_entities']}"
        )
    if "asr" in res:
        print(f"\nASR on {res['asr']['turns']} turns of {res['asr']['calls']} calls")
        print(
            f"{'model':<22}{'CER':>7}{'WER':>7}{'mist.':>7}{'fixed':>7}{'wrong':>7}{'left':>6}"
            f"{'broke':>7}{'mean s':>8}{'p95 s':>7}{'max s':>7}"
        )
        for name, m in res["asr"]["models"].items():
            lat = m["latency_s"]["cold_words"]
            print(
                f"{name:<22}{m['cer']:>7.3f}{m['wer']:>7.3f}{m['mistakes']:>7}{m['fixed']:>7}"
                f"{m['changed_but_still_wrong']:>7}{m['left']:>6}{m['correct_words_broken']:>7}"
                f"{lat['mean']:>8.3f}{lat['p95']:>7.3f}{lat['max']:>7.3f}"
            )
        print(f"first autocorrect call: {res['asr']['first_autocorrect_call_s']:.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("part", nargs="?", default="all", choices=["all", "gate", "entities", "asr"])
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--asr-models", nargs="+", default=list(ASR_MODELS))
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "eval" / "results" / "asr_cache",
        help="ASR transcripts are kept here so a rerun skips the audio",
    )
    ap.add_argument("--limit", type=int, help="only the first N turns (quick check)")
    ap.add_argument("--out", type=Path, default=RESULT)
    args = ap.parse_args()

    t0 = time.perf_counter()
    res = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    if args.part in ("all", "asr"):  # first, so the first-call cost is not hidden by other parts
        res["asr"] = run_asr(args.store, tuple(args.asr_models), args.cache_dir, args.limit)
    if args.part in ("all", "gate"):
        res.update(run_gate(args.models_dir))
    if args.part in ("all", "entities"):
        res["entities"] = run_entities()
    res["wall_time_s"] = {args.part: time.perf_counter() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    show(res)
    print(f"\nwall time {time.perf_counter() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
