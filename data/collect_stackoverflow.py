"""Parse Stack Overflow data dump (Posts.xml) for git-tagged Q&A pairs.

The SO data dump is large (~80 GB compressed).  This script expects you to
have already downloaded and extracted the Posts.xml file from the dump.  It
streams through the XML incrementally so memory usage stays bounded.

Records kept:
  - Tagged with [git] (and optionally related tags)
  - Score >= 5

Each output record:
  {"text": ..., "source": "stackoverflow/<post_id>", "type": "qa",
   "question_title": ..., "question_body": ..., "answer_body": ...}

Output: data/raw/stackoverflow.jsonl
"""

import argparse
import html as html_module
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import TextIO

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def clean_html(raw_html: str) -> str:
    """Strip HTML tags and decode entities, returning plain text.

    Uses BeautifulSoup if available, otherwise falls back to a regex
    approach.
    """
    if not raw_html:
        return ""

    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw_html, "html.parser")
        # Preserve code blocks with markers
        for code_tag in soup.find_all("code"):
            code_tag.string = f"`{code_tag.get_text()}`"
        for pre_tag in soup.find_all("pre"):
            pre_tag.string = f"\n```\n{pre_tag.get_text()}\n```\n"
        text = soup.get_text(separator="\n")
    else:
        # Regex fallback
        text = re.sub(r"<pre><code>(.*?)</code></pre>", r"\n```\n\1\n```\n", raw_html, flags=re.DOTALL)
        text = re.sub(r"<code>(.*?)</code>", r"`\1`", raw_html, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)

    text = html_module.unescape(text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _has_git_tag(tags_str: str) -> bool:
    """Return True if the SO tags string contains a git-related tag."""
    if not tags_str:
        return False
    # Tags look like: <git><github><merge>
    tag_list = re.findall(r"<([^>]+)>", tags_str)
    git_tags = {"git", "github", "git-merge", "git-rebase", "git-branch",
                "git-commit", "git-push", "git-pull", "git-stash",
                "git-checkout", "git-log", "git-diff", "git-reset",
                "git-remote", "git-submodules", "git-tag", "gitignore",
                "git-bash", "git-flow", "git-cherry-pick", "git-revert",
                "git-config", "git-clone", "git-fetch", "git-bisect"}
    return bool(set(tag_list) & git_tags)


def parse_posts_xml(
    xml_path: str,
    min_score: int = 5,
) -> list[dict]:
    """Incrementally parse Posts.xml and pair questions with their best answer.

    PostTypeId == 1  -> Question
    PostTypeId == 2  -> Answer

    We do two passes:
      1. Collect qualifying questions (git-tagged, score >= min_score).
      2. Collect answers, match to questions via ParentId, keep the
         accepted answer or highest-scored answer.
    """
    print(f"Pass 1: collecting git-tagged questions with score >= {min_score} ...")
    questions: dict[str, dict] = {}  # id -> {title, body, score, accepted_answer_id}

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue
        post_type = elem.get("PostTypeId", "")
        if post_type == "1":  # Question
            tags = elem.get("Tags", "")
            score = int(elem.get("Score", "0"))
            if _has_git_tag(tags) and score >= min_score:
                post_id = elem.get("Id", "")
                questions[post_id] = {
                    "title": elem.get("Title", ""),
                    "body": elem.get("Body", ""),
                    "score": score,
                    "accepted_answer_id": elem.get("AcceptedAnswerId", ""),
                }
        elem.clear()  # free memory

    print(f"  Found {len(questions)} qualifying questions.")

    if not questions:
        return []

    # Build a set of question ids for fast lookup
    question_ids = set(questions.keys())
    # Also collect the set of accepted answer ids for quick checking
    accepted_ids = {q["accepted_answer_id"] for q in questions.values() if q["accepted_answer_id"]}

    print("Pass 2: collecting answers ...")
    # answer_id -> {parent_id, body, score}
    answers: dict[str, dict] = {}

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue
        post_type = elem.get("PostTypeId", "")
        if post_type == "2":  # Answer
            parent_id = elem.get("ParentId", "")
            if parent_id in question_ids:
                ans_id = elem.get("Id", "")
                ans_score = int(elem.get("Score", "0"))
                # Keep answer if it is accepted or has higher score than current best
                existing = answers.get(parent_id)
                is_accepted = ans_id in accepted_ids
                if existing is None:
                    answers[parent_id] = {
                        "id": ans_id,
                        "body": elem.get("Body", ""),
                        "score": ans_score,
                        "is_accepted": is_accepted,
                    }
                else:
                    # Prefer accepted, then higher score
                    if is_accepted and not existing["is_accepted"]:
                        answers[parent_id] = {
                            "id": ans_id,
                            "body": elem.get("Body", ""),
                            "score": ans_score,
                            "is_accepted": True,
                        }
                    elif not existing["is_accepted"] and ans_score > existing["score"]:
                        answers[parent_id] = {
                            "id": ans_id,
                            "body": elem.get("Body", ""),
                            "score": ans_score,
                            "is_accepted": False,
                        }
        elem.clear()

    print(f"  Matched answers for {len(answers)} questions.")

    # Build records
    records: list[dict] = []
    for qid, q in questions.items():
        ans = answers.get(qid)
        if ans is None:
            continue  # skip questions without answers

        q_title = clean_html(q["title"])
        q_body = clean_html(q["body"])
        a_body = clean_html(ans["body"])

        if not a_body or len(a_body) < 30:
            continue

        # Combine into a training-friendly text block
        text = f"Question: {q_title}\n\n{q_body}\n\nAnswer:\n{a_body}"

        records.append({
            "text": text,
            "source": f"stackoverflow/{qid}",
            "type": "qa",
            "question_title": q_title,
            "question_body": q_body,
            "answer_body": a_body,
        })

    print(f"  Produced {len(records)} Q&A records.")
    return records


def write_jsonl(records: list[dict], output_path: str) -> None:
    """Write records to a JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Stack Overflow Posts.xml for git-tagged Q&A pairs.",
    )
    parser.add_argument(
        "input_xml",
        type=str,
        help="Path to the Stack Overflow Posts.xml file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to write stackoverflow.jsonl (default: data/raw)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=5,
        help="Minimum question score to include (default: 5)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_xml):
        parser.error(f"File not found: {args.input_xml}")

    records = parse_posts_xml(args.input_xml, min_score=args.min_score)
    output_path = os.path.join(args.output_dir, "stackoverflow.jsonl")
    write_jsonl(records, output_path)


if __name__ == "__main__":
    main()
