"""Label scam posts with Qwen through the university's OpenAI-compatible API.

Two labelling passes, each run over the scraped dataset (data/antifakenews_scam.csv):

1. Scam type pass (LABEL_TASK, every post). Two fields:
     scam_type              one of the five Callas scam types, or "" when no scam is described
     method_channel         where the scam happens, from the fixed channel list, comma separated

2. Scenario pass (SCENARIO_TASK, only posts with a scam_type). The fields the call simulator
   (modules/data_pipeline) needs to turn a post into a scam wiki note or a scenario:
     call_suitable           call / convertible / no: is it, or could it be, a phone scam
                             (always "call" when method_channel lists โทรศัพท์)
     summary                 one English line naming the pattern (the wiki note's heading)
     caller_role             who the scammer claims to be, in English (scenario `role`)
     goal                    what the scammer wants by the end, in English with slots (scenario `goal`)
     tactics                 the scam's steps in order, in English, " | " separated
     phrases                 Thai lines the post quotes from the scammer, with slots, " | " separated
     scenario_key            the matching scenario in scenarios.yaml, or "new"
     scenario_group          a short name shared by posts with the same new situation
   Code adds source_url (the post's link).

How a pass runs: posts are sent in batches, one request after another. Each post in a batch is
tagged with its post_id, e.g. [78875]; the model answers {"posts": [{"post_id": 78875, ...}, ...]}
and the ids map the answers back to the rows. A bad or cut-off reply splits the batch in half;
posts missing from a reply are asked once more; posts that still fail get empty fields and
llm_error=True.

Every finished answer is appended to data/llm_<task>_checkpoint.jsonl, so a run that stops early
(crash, Ctrl+C) resumes where it left off. The file is deleted once every post is labelled. Delete
it by hand after editing the prompts, or the old answers are reused.

The prompts are text files in prompts/ (lines starting with # are comments and are not sent):
  label_system.txt / label_user.txt        scam type pass
  scenario_system.txt / scenario_user.txt  scenario pass
Connection settings come from the repo-root .env (never from code):
  QWEN_BASE_URL=https://.../v1
  QWEN_API_KEY=...
  QWEN_MODEL=...

Use it on the scraped dataset, without scraping again:
  python llm_labeler.py                    scam type pass, saved back into the dataset
  python llm_labeler.py --test             scam type pass on the first batch, only printed
  python llm_labeler.py --scenario         scenario pass, saved back into the dataset
  python llm_labeler.py --scenario --test  scenario pass on the first batch, only printed
In the notebook: from llm_labeler import label_dataset, label_scenarios_dataset
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")  # repo-root .env; variables already set in the shell win

# the call simulator's slot names, so the scenario pass only writes slots the simulator knows
sys.path.insert(0, str(HERE.parent / "modules"))
from data_pipeline.config import SLOT_NAMES  # noqa: E402

MAX_CONTENT_CHARS = 3000  # long fact-check posts state the claim first, so the start is enough
PROMPT_DIR = HERE / "prompts"
SCENARIOS_YAML = HERE.parent / "modules" / "data_pipeline" / "config" / "scenarios.yaml"


def load_prompt(name):
    """Read prompts/<name>.txt without its # comment lines."""
    raw = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("#")).strip()


def fill(template, **values):
    """Replace {key} placeholders with str.replace, so braces in a post cannot break the prompt."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


@dataclass
class Task:
    """One labelling pass: its prompts, the fields it answers and how one answer is cleaned."""

    name: str  # also names the checkpoint file
    system_prompt: str  # built once, the same for every post
    user_prompt: str  # one post; its {placeholders} are the lower-cased column names
    columns: list  # dataset columns put into the user prompt
    fields: list  # answer fields
    clean: Callable  # raw answer item -> {field: text}
    tokens_per_post: int  # answer budget per post
    batch_size: int  # posts per request

    @property
    def checkpoint(self):
        return HERE / "data" / f"llm_{self.name}_checkpoint.jsonl"


# ---- pass 1: scam type -------------------------------------------------------------------------

SCAM_TYPES = {"fake_authority", "romance", "investment", "phishing_otp", "lottery"}
CHANNELS = {
    "เพจเฟซบุ๊ก", "กลุ่มเฟซบุ๊ก", "Messenger", "TikTok", "LINE", "SMS", "โทรศัพท์", "เว็บไซต์", "อีเมล",
    "เอกสาร", "แอปพลิเคชัน", "โซเชียลมีเดีย", "พอดแคสต์", "QR Code", "Google Form",
}
FIELDS = ["scam_type", "method_channel"]


def clean(item):
    """Keep only allowed values for one post's scam type answer."""
    out = {field: str(item.get(field) or "").strip() for field in FIELDS}

    if out["scam_type"] not in SCAM_TYPES:
        out["scam_type"] = ""
    channels = [c.strip() for c in out["method_channel"].split(",")]
    out["method_channel"] = ", ".join(c for c in channels if c in CHANNELS)
    if not out["scam_type"]:  # no scam -> no method
        out["method_channel"] = ""
    return out


LABEL_TASK = Task(
    name="labels",
    # the whole definitions file goes into the instructions
    system_prompt=fill(load_prompt("label_system"),
                       definitions=(HERE / "scam_type_definitions.md").read_text(encoding="utf-8")),
    user_prompt=load_prompt("label_user"),
    columns=["post_id", "title", "content"],
    fields=FIELDS,
    clean=clean,
    tokens_per_post=100,  # a scam type and a few channel names
    batch_size=50,
)


# ---- pass 2: scenario fields -------------------------------------------------------------------

SCENARIO_FIELDS = [
    "call_suitable", "summary", "caller_role", "goal", "tactics", "phrases", "scenario_key", "scenario_group",
]
SUITABLE = {"call", "convertible", "no"}
LIST_SEP = " | "  # tactics and phrases are lists; a csv cell holds them joined with this

# real numbers, links and LINE IDs the model left in its answer -> the simulator's slots
TO_SLOTS = [
    # a link; [\w-]+(?:\.[\w-]+)* takes every part of the domain, e.g. thaipoliceonline.go.th
    (re.compile(r"https?://\S+|www\.\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|co|ly|me|th|info|xyz)\b\S*"), "{URL}"),
    (re.compile(r"[\w-]+\.\{URL\}"), "{URL}"),  # a domain an earlier run cut in half: thaipoliceonline.{URL}
    (re.compile(r"\d[\d,.]*\s*(?:บาท|baht)", re.IGNORECASE), "{MONEY}"),  # 50,000 บาท
    (re.compile(r"\b0\d{1,2}[- ]?\d{3}[- ]?\d{3,4}\b"), "{PHONE}"),  # 081-234-5678, 02 123 4567
    # a LINE ID such as @กองคลัง; there is no LINE slot. Not \w: Thai vowel marks are not word characters
    (re.compile(r"@[^\s,()\"']{3,}"), "{URL}"),
]
SLOT = re.compile(r"\{([A-Za-z_]+)\}")


def to_slots(text):
    """Replace real numbers, links and LINE IDs with slots, and unwrap slots the simulator lacks."""
    for pattern, slot in TO_SLOTS:
        text = pattern.sub(slot, text)
    # {LINE_ID} is not a simulator slot: keep the word, drop the braces, so the line still reads
    return SLOT.sub(lambda m: m.group(0) if m.group(1) in SLOT_NAMES else m.group(1), text)


def as_list(value):
    """The model may answer a list or one string; always give a list of non-empty strings."""
    items = value if isinstance(value, list) else str(value or "").split("|")
    return [str(item).strip() for item in items if str(item).strip()]


def load_scenarios():
    """Existing scam scenarios from the simulator's scenarios.yaml: {scam_type: [scenario dict]}."""
    return yaml.safe_load(SCENARIOS_YAML.read_text(encoding="utf-8"))["scams"]


SCENARIOS = load_scenarios()
SCENARIO_KEYS = {s["key"] for scenarios in SCENARIOS.values() for s in scenarios}


def clean_scenario(item):
    """Keep only allowed values for one post's scenario answer, with slots in place of numbers."""
    out = {}
    for field in ["summary", "caller_role", "goal"]:
        out[field] = to_slots(str(item.get(field) or "").strip())
    out["tactics"] = LIST_SEP.join(to_slots(t) for t in as_list(item.get("tactics")))
    out["phrases"] = LIST_SEP.join(to_slots(p) for p in as_list(item.get("phrases")))

    suitable = str(item.get("call_suitable") or "").strip().lower()
    out["call_suitable"] = suitable if suitable in SUITABLE else ""

    key = str(item.get("scenario_key") or "").strip()
    out["scenario_key"] = key if key in SCENARIO_KEYS else "new"
    group = re.sub(r"[^a-z0-9_]+", "_", str(item.get("scenario_group") or "").strip().lower()).strip("_")
    out["scenario_group"] = out["scenario_key"] if out["scenario_key"] != "new" else group
    return {field: out[field] for field in SCENARIO_FIELDS}


def scenario_list_text():
    """The existing scenarios as prompt text, one line each: type, key, role and goal."""
    lines = []
    for scam_type, scenarios in SCENARIOS.items():
        for s in scenarios:
            lines.append(f"- {scam_type} / {s['key']}: {s['role']}; wants: {' '.join(s['goal'].split())}")
    return "\n".join(lines)


SCENARIO_TASK = Task(
    name="scenarios",
    system_prompt=fill(load_prompt("scenario_system"),
                       scenarios=scenario_list_text(), slots=" ".join("{" + s + "}" for s in SLOT_NAMES)),
    user_prompt=load_prompt("scenario_user"),
    columns=["post_id", "scam_type", "ORG", "title", "content"],
    fields=SCENARIO_FIELDS,
    clean=clean_scenario,
    tokens_per_post=700,  # eleven fields, two of them lists
    batch_size=20,
)


# ---- running a pass ----------------------------------------------------------------------------

def get_client():
    return OpenAI(base_url=os.environ["QWEN_BASE_URL"], api_key=os.environ["QWEN_API_KEY"], timeout=600, max_retries=2)


def ask(client, task, user_text, n_posts):
    """One chat call. Asks for JSON mode, and drops it if the server does not support it."""
    request = dict(
        model=os.environ["QWEN_MODEL"],
        messages=[{"role": "system", "content": task.system_prompt}, {"role": "user", "content": user_text}],
        temperature=0,
        max_tokens=task.tokens_per_post * n_posts,  # the answer grows with the number of posts
        # Qwen 3.x thinks before answering; turn it off so the tokens go to the JSON answer
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    try:
        reply = client.chat.completions.create(**request, response_format={"type": "json_object"})
    except Exception as error:  # some servers reject response_format; retry without it
        if "response_format" not in str(error).lower():
            raise
        reply = client.chat.completions.create(**request)
    choice = reply.choices[0]
    if not choice.message.content:
        raise ValueError(f"empty reply (finish_reason={choice.finish_reason})")
    if choice.finish_reason == "length":  # hit max_tokens, so the JSON is cut off
        raise ValueError("reply cut off at max_tokens")
    return choice.message.content


def parse_reply(task, raw):
    """Turn the model's reply into {post_id: answer} for the posts it answered."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate text or <think> blocks around the JSON
    if not match:
        raise ValueError(f"no JSON in reply: {raw[:300]!r}")
    items = json.loads(match.group(0)).get("posts", [])
    return {str(item.get("post_id", "")).strip(): task.clean(item) for item in items}


def prompt_values(post):
    """A post's values as prompt text, with the content cut to MAX_CONTENT_CHARS."""
    values = {key: str(value) for key, value in post.items()}
    values["content"] = values["content"][:MAX_CONTENT_CHARS]
    return values


def label_batch(client, task, posts, retry=True):
    """Label a list of posts (dicts of the task's columns, lower-cased) in one request.
    Returns {post_id: answer}.

    - Bad or cut-off reply (ValueError): the batch is split in half and each half is asked on its
      own, so one broken reply does not lose the answers of the whole batch.
    - Connection error, timeout or rate limit (the client already retried twice): no answers.
    Posts still without an answer are asked once more (retry); after that they get empty fields
    and llm_error=True.
    """
    # one user message holding all the posts, each tagged with its post_id
    message = "\n\n".join(fill(task.user_prompt, **prompt_values(post)) for post in posts)
    try:
        answers = parse_reply(task, ask(client, task, message, len(posts)))
    except ValueError as error:
        if len(posts) > 1:
            print(f"bad reply for {len(posts)} posts ({error}); splitting the batch")
            half = len(posts) // 2
            return {**label_batch(client, task, posts[:half], retry),
                    **label_batch(client, task, posts[half:], retry)}
        print(f"bad reply: {error}")
        answers = {}
    except Exception as error:
        print(f"request failed: {error}")
        answers = {}

    results = {post["post_id"]: {**answers[str(post["post_id"])], "llm_error": False}
               for post in posts if str(post["post_id"]) in answers}
    missing = [post for post in posts if post["post_id"] not in results]
    if missing and retry:
        results.update(label_batch(client, task, missing, retry=False))  # ask once more for the posts left out
    else:
        for post in missing:
            print(f"failed: [{post['post_id']}] {str(post['title'])[:60]}")
            results[post["post_id"]] = {**dict.fromkeys(task.fields, ""), "llm_error": True}
    return results


def load_checkpoint(task):
    """Answers saved by an earlier run that stopped early, as {post_id (str): answer}."""
    if not task.checkpoint.exists():
        return {}
    saved = {}
    for line in task.checkpoint.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            saved[item["post_id"]] = item["answer"]
        except (ValueError, KeyError):  # a line half-written when the run was killed
            pass
    return saved


def label_all(df, task=LABEL_TASK):
    """Label every row of df with one task. Returns the task's fields + llm_error, same index."""
    client = get_client()
    labels = load_checkpoint(task)  # keyed by str(post_id), so int and str ids both match
    if labels:
        print(f"resuming: {len(labels)} answers found in {task.checkpoint.name}")
    # one dict per post, keys = the task's columns lower-cased ("ORG" -> {org} in the prompt)
    rows = df[task.columns].rename(columns=str.lower).to_dict("records")
    posts = [post for post in rows if str(post["post_id"]) not in labels]

    task.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with task.checkpoint.open("a", encoding="utf-8") as checkpoint:
        for start in range(0, len(posts), task.batch_size):
            for post_id, answer in label_batch(client, task, posts[start:start + task.batch_size]).items():
                labels[str(post_id)] = answer
                if not answer["llm_error"]:  # failed posts are not saved, so the next run asks again
                    checkpoint.write(json.dumps({"post_id": str(post_id), "answer": answer}, ensure_ascii=False) + "\n")
            checkpoint.flush()  # on disk now, even if the next batch crashes
            print(f"labelled {min(start + task.batch_size, len(posts))}/{len(posts)}")

    # put the answers back in the order of df's rows, matched by post_id
    result = pd.DataFrame([labels[str(post_id)] for post_id in df["post_id"]], index=df.index)
    if not result["llm_error"].any():
        task.checkpoint.unlink()  # everything labelled; the next run starts fresh
    return result


# ---- scenario columns ----------------------------------------------------------------------------

POST_URL = "https://www.antifakenewscenter.com/?p={}"


def phrase_in_post(phrase, content):
    """True when every piece of the phrase between slots appears in the post (spaces ignored).
    The review sheet in scrapping.ipynb uses it to drop made-up quotes."""
    squashed = re.sub(r"\s+", "", content)
    pieces = [re.sub(r"\s+", "", piece) for piece in SLOT.split(phrase)[::2]]  # [::2] drops slot names
    return all(piece in squashed for piece in pieces if len(piece) >= 3)


def add_scenario_columns(df):
    """Run the scenario pass on the posts that have a scam_type and add its columns to df.

    Posts with no scam_type get empty scenario fields. Also adds source_url (every post).
    Returns df with the new columns.
    """
    scams = df[df["scam_type"] != ""]
    labels = label_all(scams, SCENARIO_TASK)
    for field in SCENARIO_FIELDS:
        df[field] = ""
        df.loc[scams.index, field] = labels[field]
    # a post whose method_channel lists the phone is a phone scam, whatever the second pass said
    phone = (df["scam_type"] != "") & df["method_channel"].str.contains("โทรศัพท์")
    df.loc[phone, "call_suitable"] = "call"
    df["source_url"] = [POST_URL.format(post_id) for post_id in df["post_id"]]
    print(f"{labels['llm_error'].sum()} of {len(scams)} scam posts failed to label")
    return df


# ---- the dataset -------------------------------------------------------------------------------

DATASET = HERE / "data" / "antifakenews_scam.csv"
def read_dataset(path=DATASET):
    """The scraped dataset as a dataframe; everything as text so values are saved back unchanged."""
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def save_dataset(df, path=DATASET):
    """Save the dataset as UTF-8 with BOM (so Excel shows the Thai text), dates as DD/MM/YYYY."""
    path = Path(path)
    if pd.api.types.is_datetime64_any_dtype(df["date"]):  # the notebook keeps dates as datetimes
        df = df.assign(date=df["date"].dt.strftime("%d/%m/%Y"))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def label_dataset(path=DATASET):
    """Read the scraped dataset, label every post's scam type with Qwen, and save it back."""
    df = read_dataset(path)
    labels = label_all(df)
    df[FIELDS] = labels[FIELDS]
    save_dataset(df, path)
    print(f"saved {len(df)} posts to {path}; {labels['llm_error'].sum()} failed to label")
    return df


def label_scenarios_dataset(path=DATASET):
    """Read the labelled dataset, add the scenario columns with Qwen, and save it back."""
    df = add_scenario_columns(read_dataset(path))
    save_dataset(df, path)
    print(f"saved {len(df)} posts to {path}")
    return df


if __name__ == "__main__":
    scenario = "--scenario" in sys.argv
    if "--test" in sys.argv:
        # --test: label the first batch and print it (nothing is saved)
        df = read_dataset()
        if scenario:
            sample = df[df["scam_type"] != ""].head(SCENARIO_TASK.batch_size)
            labels = label_all(sample, SCENARIO_TASK)
            columns = ["post_id", "scam_type", "title"]
        else:
            sample = df.head(LABEL_TASK.batch_size)
            labels = label_all(sample)
            columns = ["post_id", "title"]
        print(pd.concat([sample[columns], labels], axis=1).to_string())
    elif scenario:
        label_scenarios_dataset()
    else:
        label_dataset()
