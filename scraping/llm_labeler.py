"""Label scam posts with Qwen through the university's OpenAI-compatible API.

For each post (title + content) the model returns four fields:
  scam_type              one of the five Callas scam types, or "" when no scam is described
  method_channel         where the scam happens, from the fixed channel list, comma separated
  method_scammer_action  the story or offer used to hook the victim (short Thai phrase)
  method_victim_action   what the victim is pushed to do (short Thai phrase)

Posts are sent in batches of BATCH_SIZE (50) per request, one request after another. Each post in
a batch is tagged with its post_id, e.g. [78875]; the model answers
{"posts": [{"post_id": 78875, ...}, ...]} and the ids map the answers back to the rows. A bad or
cut-off reply splits the batch in half; posts missing from a reply are asked once more; posts that
still fail get empty fields and llm_error=True.

Every finished answer is appended to data/llm_labels_checkpoint.jsonl, so a run that stops early
(crash, Ctrl+C) resumes where it left off. The file is deleted once every post is labelled. Delete
it by hand after editing the prompts, or the old answers are reused.

The prompts are text files in prompts/: label_system.txt (instructions, with the whole of
scam_type_definitions.md put in place of {definitions}) and label_user.txt (one tagged post, with
{post_id}, {title} and {content}). Editing those files changes how posts are labelled. Connection settings come from the repo-root .env (never from code):
  QWEN_BASE_URL=https://.../v1
  QWEN_API_KEY=...
  QWEN_MODEL=...

Use it on the scraped dataset (data/antifakenews_scam.csv), without scraping again:
  python llm_labeler.py          label every post and save the labels back into the dataset
  python llm_labeler.py --test   label the first batch and only print it
In the notebook the same thing is: from llm_labeler import label_dataset; label_dataset()
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")  # repo-root .env; variables already set in the shell win

SCAM_TYPES = {"fake_authority", "romance", "investment", "phishing_otp", "lottery"}
CHANNELS = {
    "เพจเฟซบุ๊ก", "กลุ่มเฟซบุ๊ก", "Messenger", "TikTok", "LINE", "SMS", "โทรศัพท์", "เว็บไซต์", "อีเมล",
    "เอกสาร", "แอปพลิเคชัน", "โซเชียลมีเดีย", "พอดแคสต์", "QR Code", "Google Form",
}
FIELDS = ["scam_type", "method_channel", "method_scammer_action", "method_victim_action"]
MAX_CONTENT_CHARS = 3000  # long fact-check posts state the claim first, so the start is enough
BATCH_SIZE = 50  # posts per request
TOKENS_PER_POST = 250  # answer budget per post (four short Thai fields)

PROMPT_DIR = HERE / "prompts"


def load_prompt(name):
    """Read prompts/<name>.txt without its # comment lines."""
    raw = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("#")).strip()


def fill(template, **values):
    """Replace {key} placeholders with str.replace, so braces in a post cannot break the prompt."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


# the system prompt is the same for every post, so it is built once
DEFINITIONS = (HERE / "scam_type_definitions.md").read_text(encoding="utf-8")
SYSTEM_PROMPT = fill(load_prompt("label_system"), definitions=DEFINITIONS)
USER_PROMPT = load_prompt("label_user")


def get_client():
    return OpenAI(base_url=os.environ["QWEN_BASE_URL"], api_key=os.environ["QWEN_API_KEY"], timeout=600, max_retries=2)


def ask(client, user_text, n_posts):
    """One chat call. Asks for JSON mode, and drops it if the server does not support it."""
    request = dict(
        model=os.environ["QWEN_MODEL"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}],
        temperature=0,
        max_tokens=TOKENS_PER_POST * n_posts,  # the answer grows with the number of posts
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


def clean(item):
    """Keep only allowed values for one post's answer."""
    out = {field: str(item.get(field) or "").strip() for field in FIELDS}

    if out["scam_type"] not in SCAM_TYPES:
        out["scam_type"] = ""
    channels = [c.strip() for c in out["method_channel"].split(",")]
    out["method_channel"] = ", ".join(c for c in channels if c in CHANNELS)
    if not out["scam_type"]:  # no scam -> no method
        out.update(method_channel="", method_scammer_action="", method_victim_action="")
    return out


def parse_reply(raw):
    """Turn the model's reply into {post_id: answer} for the posts it answered."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate text or <think> blocks around the JSON
    if not match:
        raise ValueError(f"no JSON in reply: {raw[:300]!r}")
    items = json.loads(match.group(0)).get("posts", [])
    return {str(item.get("post_id", "")).strip(): clean(item) for item in items}


def label_batch(client, posts, retry=True):
    """Label a list of (post_id, title, content) in one request. Returns {post_id: answer}.

    - Bad or cut-off reply (ValueError): the batch is split in half and each half is asked on its
      own, so one broken reply does not lose the answers of the whole batch.
    - Connection error, timeout or rate limit (the client already retried twice): no answers.
    Posts still without an answer are asked once more (retry); after that they get empty fields
    and llm_error=True.
    """
    # one user message holding all the posts, each tagged with its post_id
    message = "\n\n".join(
        fill(USER_PROMPT, post_id=str(post_id), title=title, content=content[:MAX_CONTENT_CHARS])
        for post_id, title, content in posts
    )
    try:
        answers = parse_reply(ask(client, message, len(posts)))
    except ValueError as error:
        if len(posts) > 1:
            print(f"bad reply for {len(posts)} posts ({error}); splitting the batch")
            half = len(posts) // 2
            return {**label_batch(client, posts[:half], retry), **label_batch(client, posts[half:], retry)}
        print(f"bad reply: {error}")
        answers = {}
    except Exception as error:
        print(f"request failed: {error}")
        answers = {}

    results = {post_id: {**answers[str(post_id)], "llm_error": False}
               for post_id, _, _ in posts if str(post_id) in answers}
    missing = [post for post in posts if post[0] not in results]
    if missing and retry:
        results.update(label_batch(client, missing, retry=False))  # ask once more for the posts left out
    else:
        for post_id, title, _ in missing:
            print(f"failed: [{post_id}] {title[:60]}")
            results[post_id] = {**dict.fromkeys(FIELDS, ""), "llm_error": True}
    return results


CHECKPOINT = HERE / "data" / "llm_labels_checkpoint.jsonl"


def load_checkpoint():
    """Answers saved by an earlier run that stopped early, as {post_id (str): answer}."""
    if not CHECKPOINT.exists():
        return {}
    saved = {}
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            saved[item["post_id"]] = item["answer"]
        except (ValueError, KeyError):  # a line half-written when the run was killed
            pass
    return saved


def label_all(df, batch_size=BATCH_SIZE):
    """Label every row of df (needs post_id, title, content). Returns FIELDS + llm_error, same index."""
    client = get_client()
    labels = load_checkpoint()  # keyed by str(post_id), so int and str ids both match
    if labels:
        print(f"resuming: {len(labels)} answers found in {CHECKPOINT.name}")
    posts = [post for post in zip(df["post_id"], df["title"], df["content"]) if str(post[0]) not in labels]

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as checkpoint:
        for start in range(0, len(posts), batch_size):
            for post_id, answer in label_batch(client, posts[start:start + batch_size]).items():
                labels[str(post_id)] = answer
                if not answer["llm_error"]:  # failed posts are not saved, so the next run asks again
                    checkpoint.write(json.dumps({"post_id": str(post_id), "answer": answer}, ensure_ascii=False) + "\n")
            checkpoint.flush()  # on disk now, even if the next batch crashes
            print(f"labelled {min(start + batch_size, len(posts))}/{len(posts)}")

    # put the answers back in the order of df's rows, matched by post_id
    result = pd.DataFrame([labels[str(post_id)] for post_id in df["post_id"]], index=df.index)
    if not result["llm_error"].any():
        CHECKPOINT.unlink()  # everything labelled; the next run starts fresh
    return result


DATASET = HERE / "data" / "antifakenews_scam.csv"


def read_dataset(path=DATASET):
    """The scraped dataset as a dataframe; everything as text so values are saved back unchanged."""
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def label_dataset(path=DATASET):
    """Read the scraped dataset, label every post with Qwen, and save it back to the same file."""
    df = read_dataset(path)
    labels = label_all(df)
    df[FIELDS] = labels[FIELDS]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} posts to {path}; {labels['llm_error'].sum()} failed to label")
    return df


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # python llm_labeler.py --test: label the first batch and print it (nothing is saved)
        sample = read_dataset().head(BATCH_SIZE)
        labels = label_all(sample)
        print(pd.concat([sample[["post_id", "title"]], labels], axis=1).to_string())
    else:
        # python llm_labeler.py: label the whole scraped dataset and save it back
        label_dataset()
