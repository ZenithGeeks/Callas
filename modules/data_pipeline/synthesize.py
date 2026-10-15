"""Generate simulated calls into the SQLite store (--db), with each call's audio in an
audio/ folder next to it (each turn is spoken right after it is written; --no-tts skips this).

    python -m data_pipeline.synthesize --type phishing_otp --n 200
    python -m data_pipeline.synthesize --type all --n 20 --config my_cfg/ --db data/synthetic/my.db
    python -m data_pipeline.synthesize --check-config --config my_config/
    python -m data_pipeline.synthesize --export data/synthetic/calls.jsonl --approved-only
    python -m data_pipeline.synthesize --import-json data/synthetic/dialogues
    python -m data_pipeline.synthesize --type romance --n 5 --receiver teacher_cm
    python -m data_pipeline.synthesize --type all --n 2 --receiver-file my_person.yaml

What each agent is told, its model and temperature, the scenarios and the voices all come from
the config folder (default: data_pipeline/config/; see its YAML files). Models run on the vLLM
server in .env (VLLM_BASE_URL). Seeds give the same calls every run with the same config;
finished calls are skipped, unfinished ones are resumed. Do not run two commands that write the
same --db at the same time: calls are not locked, so both would work on the same call.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from callas_tracing import setup_tracing

from llm_serving.client import LLMUnavailable

from .agents import SIGNATURES, Agents, configured
from .config import CALL_TYPES, Config, ConfigError, ReceiverProfile, load_config
from .simulate import call_id, is_model_error, run_call
from .store import DEFAULT_DB, Store, import_json_folder
from .tts import Voices
from .wiki import DEFAULT_WIKI, WikiError
from .wiki import sync as sync_wiki


def main(argv: list[str] | None = None) -> int:
    """The command line. Returns the exit code: 0 done, 1 some calls hit an error, 2 bad input."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.export_notebook:  # reads the store only: no config, no LLM (see export.py)
        from .export import run_export

        return run_export(args.export_notebook, args.store or args.db, args.text)

    try:
        config = _load_checked_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.check_config:
        print(f"config {config.name} ok (hash {config.hash})")
        return 0
    store = Store(args.db)
    if args.import_json:
        imported, skipped = import_json_folder(store, args.import_json)
        print(f"imported {imported} calls, skipped {skipped} already in {args.db}")
        return 0
    if args.export:
        written = _export(store, args.export, args.approved_only)
        print(f"wrote {written} calls to {args.export}")
        return 0

    try:
        receiver = _chosen_receiver(args, config)
    except (ConfigError, ValueError, OSError) as exc:
        print(f"receiver error: {exc}", file=sys.stderr)
        return 2

    if args.type is None:
        parser.error("--type is required")
    if args.wiki.is_dir():  # the planner reads notes from the store; bring them up to date
        try:
            sync_wiki(store, args.wiki)
        except WikiError as exc:
            print(f"wiki error: {exc}", file=sys.stderr)
            return 2
    call_types = CALL_TYPES if args.type == "all" else (args.type,)
    if any(call_type not in CALL_TYPES for call_type in call_types):
        parser.error(f"--type must be one of {CALL_TYPES} or all")

    setup_tracing("callas-simulator", call_text=True)  # simulated calls are made up
    jobs = [
        (call_type, seed)
        for call_type in call_types
        for seed in range(args.start_seed, args.start_seed + args.n)
    ]
    counts = _run_jobs(jobs, args, config, store, receiver)
    if counts is None:
        return 2

    print(f"config {config.name} (hash {config.hash})")
    errors = 0
    for call_type in call_types:
        accepted, rejected = counts[(call_type, "accepted")], counts[(call_type, "rejected")]
        failed = counts[(call_type, "error")]
        errors += failed
        print(f"{call_type}: {accepted} accepted, {rejected} rejected, {failed} error")
    if errors:
        print(f"{errors} calls hit an error and were left unfinished; rerun to resume them")
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--type", help=f"one of {', '.join(CALL_TYPES)}, or all")
    parser.add_argument("--n", type=int, default=10, help="calls per type")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"default: {DEFAULT_DB}")
    parser.add_argument("--workers", type=int, default=4, help="calls generated at the same time")
    parser.add_argument("--config", type=Path, help="config folder (default: data_pipeline/config)")
    parser.add_argument(
        "--check-config", action="store_true", help="only check the config and exit (no LLM)"
    )
    parser.add_argument(
        "--no-tts", action="store_true", help="text only: skip speaking each turn (faster)"
    )
    parser.add_argument(
        "--export", type=Path, help="write finished calls as JSON lines (spec schema) and exit"
    )
    parser.add_argument(
        "--approved-only", action="store_true", help="with --export: only calls a human approved"
    )
    parser.add_argument(
        "--export-notebook",
        type=Path,
        metavar="CSV",
        help="write the approved calls as the csv the classifier notebook reads, and exit",
    )
    parser.add_argument(
        "--store", type=Path, help="with --export-notebook: the calls.db to read (default --db)"
    )
    parser.add_argument(
        "--text",
        choices=("script", "asr"),
        default="script",
        help="with --export-notebook: the written text, or the ASR transcript of each turn's audio",
    )
    parser.add_argument(
        "--import-json", type=Path, help="load per-call JSON files from an older run and exit"
    )
    parser.add_argument(
        "--wiki",
        type=Path,
        default=DEFAULT_WIKI,
        help=f"scam pattern pages synced into the store before a run (default: {DEFAULT_WIKI})",
    )
    who = parser.add_mutually_exclusive_group()
    who.add_argument(
        "--receiver",
        metavar="KEY",
        help="every call is answered by this profile from scenarios.yaml",
    )
    who.add_argument(
        "--receiver-file",
        type=Path,
        metavar="YAML",
        help="every call is answered by the one person described in this file",
    )
    return parser


def _chosen_receiver(args: argparse.Namespace, config: Config) -> ReceiverProfile | None:
    """The person picked by hand with --receiver or --receiver-file, or None to sample."""
    if args.receiver:
        profiles = {p.key: p for p in config.scenarios.receivers.profiles}
        if args.receiver not in profiles:
            raise ConfigError(
                f"no receiver profile {args.receiver!r}; scenarios.yaml has {sorted(profiles)}"
            )
        return profiles[args.receiver]
    if args.receiver_file:
        import yaml

        data = yaml.safe_load(args.receiver_file.read_text(encoding="utf-8"))
        profile = ReceiverProfile.model_validate(data)
        if not profile.key:
            raise ConfigError(
                f"{args.receiver_file}: add a key (lowercase and _), e.g. key: aunt_noi"
            )
        if profile.key in {p.key for p in config.scenarios.receivers.profiles}:
            raise ConfigError(
                f"{args.receiver_file}: key {profile.key!r} is already a profile in "
                "scenarios.yaml; pick another key, or use --receiver to choose that profile"
            )
        if profile.gender not in config.scenarios.receivers.genders:
            raise ConfigError(
                f"{args.receiver_file}: gender must be one of {config.scenarios.receivers.genders}"
            )
        return profile
    return None


def _export(store: Store, path: Path, approved_only: bool) -> int:
    """Finished calls as one JSON object per line, without the generation and review blocks.

    Reads the store's dicts, not CallRecords: this is an edge, and converting every call to a
    CallRecord and back made the export about 2.5 times slower.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for record in store.records():
            generation = record["generation"]
            if generation["status"] != "accepted":
                continue
            if approved_only and record["review"]["status"] != "approved":
                continue
            plain = {k: v for k, v in record.items() if k not in ("generation", "review")}
            plain["receiver_leaked"] = generation.get("receiver_leaked", [])
            plain["first_leak_turn"] = generation.get("first_leak_turn")
            out.write(json.dumps(plain, ensure_ascii=False) + "\n")
            written += 1
    return written


def _load_checked_config(folder: Path | None) -> Config:
    """Load the config and check that each agent's fields match the signatures in agents.py."""
    config = load_config(folder)
    for name in SIGNATURES:  # field names must match the code
        configured(name, getattr(config.agents, name))
    return config


def _run_jobs(
    jobs: list[tuple[str, int]],
    args: argparse.Namespace,
    config: Config,
    store: Store,
    receiver: ReceiverProfile | None = None,
):
    """Run every (call type, seed) job. Returns counts of (call type, status), or None if the
    LLM server, the config or Ctrl-C stopped the run."""
    counts: Counter = Counter()
    todo = _skip_finished(jobs, config, store, receiver, counts)
    agents = Agents(config)
    voices = None if args.no_tts else Voices()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        pool.submit(_run_one, call_type, seed, store, agents, config, voices, receiver): (
            call_type,
            seed,
        )
        for call_type, seed in todo
    }
    try:
        for done, future in enumerate(as_completed(futures), 1):
            call_type, seed = futures[future]
            call_id_used, status = future.result()
            counts[(call_type, status)] += 1
            print(f"[{done}/{len(todo)}] {call_id_used} {status}", flush=True)
    except KeyboardInterrupt:
        # Leaving a `with` block would wait for every queued call; cancel the queue instead.
        # Calls already running finish their current turn; every turn is saved, so a rerun resumes.
        pool.shutdown(wait=False, cancel_futures=True)
        print("stopped by user; rerun to resume", file=sys.stderr)
        return None
    except (LLMUnavailable, ConfigError) as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        print(f"stopped: {exc}", file=sys.stderr)
        return None
    pool.shutdown()
    return counts


def _skip_finished(
    jobs: list[tuple[str, int]],
    config: Config,
    store: Store,
    receiver: ReceiverProfile | None,
    counts: Counter,
) -> list[tuple[str, int]]:
    """The jobs that still have work to do. Finished calls are counted by their status and
    reported, with a warning for any made with another config than this one."""
    todo, skipped, other_config = [], 0, 0
    for call_type, seed in jobs:
        record = store.load_call(call_id(call_type, seed, receiver))
        if record is None or record.generation.status == "in_progress":
            todo.append((call_type, seed))
            continue
        skipped += 1
        counts[(call_type, record.generation.status)] += 1
        other_config += record.generation.config_hash != config.hash
    if skipped:
        print(f"skipped {skipped} calls that were already finished")
    if other_config:
        print(
            f"warning: {other_config} of the skipped calls were made with a different config "
            f"than the current one ({config.hash}); delete them to remake them",
            file=sys.stderr,
        )
    return todo


def _run_one(
    call_type: str,
    seed: int,
    store: Store,
    agents: Agents,
    config: Config,
    voices: Voices | None,
    receiver: ReceiverProfile | None = None,
) -> tuple[str, str]:
    """Run one call. Returns (call id, status).

    Only a model error that survived the retries marks the call `rejected` (llm_error:...).
    Any other error leaves the call in progress, so a rerun resumes it, and counts as "error".
    """
    wanted_id = call_id(call_type, seed, receiver)
    try:
        record = run_call(call_type, seed, store, agents, config, voices=voices, receiver=receiver)
    except (LLMUnavailable, ConfigError):
        raise
    except Exception as exc:
        if not is_model_error(exc):
            print(f"{wanted_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return wanted_id, "error"
        record = store.load_call(wanted_id)
        if record is None:
            return wanted_id, "error"
        reasons = [f"llm_error:{type(exc).__name__}"]
        generation = replace(record.generation, status="rejected", reasons=reasons)
        store.save_call(replace(record, generation=generation))
        return wanted_id, "rejected"
    return record["id"], record["generation"]["status"]


if __name__ == "__main__":
    raise SystemExit(main())
