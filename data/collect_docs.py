"""Collect and parse real git documentation sources into JSONL format.

Sources:
  - Pro Git book (progit2): AsciiDoc files
  - Git man pages (git/git): Documentation/*.txt
  - tldr-pages: pages/common/git-*.md
  - GitHub Docs: markdown files

Each record: {"text": ..., "source": ..., "type": ...}
Output: data/raw/docs.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_repo(url: str, dest: str, shallow: bool = True) -> None:
    """Clone a git repository.  Uses --depth 1 by default for speed."""
    if os.path.isdir(dest):
        print(f"  [skip] {dest} already exists")
        return
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [url, dest]
    print(f"  Cloning {url} ...")
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _iter_files(root: str, glob_pattern: str) -> Generator[Path, None, None]:
    """Yield all files matching *glob_pattern* under *root*."""
    yield from Path(root).rglob(glob_pattern)


def _strip_asciidoc(text: str) -> str:
    """Rough AsciiDoc-to-plaintext conversion.

    Removes common AsciiDoc markup so the resulting text is usable for
    language-model training without excessive noise.
    """
    # Remove image macros
    text = re.sub(r"image::?[^\[]*\[[^\]]*\]", "", text)
    # Remove include directives
    text = re.sub(r"^include::.*$", "", text, flags=re.MULTILINE)
    # Remove block delimiters (----, ====, ....)
    text = re.sub(r"^[\-=\.~\*\+]{4,}\s*$", "", text, flags=re.MULTILINE)
    # Remove anchor links  [[...]]
    text = re.sub(r"\[\[.*?\]\]", "", text)
    # Remove inline formatting (bold, italic, mono) but keep content
    text = re.sub(r"(?<!\w)[*_`]+([^*_`]+?)[*_`]+(?!\w)", r"\1", text)
    # Remove cross-reference macros <<label,text>> -> text
    text = re.sub(r"<<[^,>]+,([^>]+)>>", r"\1", text)
    text = re.sub(r"<<[^>]+>>", "", text)
    # Remove role/option markers  [source,...] lines
    text = re.sub(r"^\[.*?\]\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    """Rough Markdown-to-plaintext conversion."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove images  ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Convert links  [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"(\*{1,2}|_{1,2})(.+?)\1", r"\2", text)
    # Remove inline code backticks (keep content)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Source parsers
# ---------------------------------------------------------------------------

def collect_progit(clone_dir: str) -> list[dict]:
    """Parse Pro Git book AsciiDoc files."""
    repo_url = "https://github.com/progit/progit2.git"
    dest = os.path.join(clone_dir, "progit2")
    _clone_repo(repo_url, dest)

    records: list[dict] = []
    # The book chapters live under book/<chapter>/sections/*.asc (and ch*.asc)
    for path in sorted(_iter_files(dest, "*.asc")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cleaned = _strip_asciidoc(raw)
        if len(cleaned) < 100:
            continue
        records.append({
            "text": cleaned,
            "source": f"progit2/{path.relative_to(dest)}",
            "type": "book",
        })
    print(f"  Pro Git: {len(records)} documents")
    return records


def collect_git_manpages(clone_dir: str) -> list[dict]:
    """Parse git man-page sources (Documentation/*.txt)."""
    repo_url = "https://github.com/git/git.git"
    dest = os.path.join(clone_dir, "git")
    _clone_repo(repo_url, dest)

    docs_dir = os.path.join(dest, "Documentation")
    records: list[dict] = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cleaned = _strip_asciidoc(raw)
        if len(cleaned) < 80:
            continue
        records.append({
            "text": cleaned,
            "source": f"git/Documentation/{path.name}",
            "type": "manpage",
        })
    print(f"  Git man pages: {len(records)} documents")
    return records


def collect_tldr(clone_dir: str) -> list[dict]:
    """Parse tldr-pages for git-related commands."""
    repo_url = "https://github.com/tldr-pages/tldr.git"
    dest = os.path.join(clone_dir, "tldr")
    _clone_repo(repo_url, dest)

    records: list[dict] = []
    # pages/common/git-*.md and pages/common/git.md
    for pattern in ("pages/common/git-*.md", "pages/common/git.md"):
        for path in sorted(Path(dest).glob(pattern)):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            cleaned = _strip_markdown(raw)
            if len(cleaned) < 20:
                continue
            records.append({
                "text": cleaned,
                "source": f"tldr/{path.relative_to(dest)}",
                "type": "tldr",
            })
    print(f"  tldr-pages: {len(records)} documents")
    return records


def collect_github_docs(clone_dir: str) -> list[dict]:
    """Parse GitHub Docs markdown for git-related content."""
    repo_url = "https://github.com/github/docs.git"
    dest = os.path.join(clone_dir, "github-docs")
    _clone_repo(repo_url, dest)

    # GitHub docs keep content under content/ directory
    content_root = os.path.join(dest, "content")
    if not os.path.isdir(content_root):
        # Fallback: search entire repo for markdown
        content_root = dest

    git_keywords = re.compile(
        r"\b(git\b|commit|branch|merge|rebase|pull request|push|fetch|clone|"
        r"checkout|stash|cherry.pick|reset|revert|diff|log|remote|repository)\b",
        re.IGNORECASE,
    )

    records: list[dict] = []
    for path in sorted(_iter_files(content_root, "*.md")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Only keep git-relevant files
        if not git_keywords.search(raw[:2000]):
            continue
        cleaned = _strip_markdown(raw)
        if len(cleaned) < 100:
            continue
        # Strip YAML front matter
        cleaned = re.sub(r"^---\n.*?\n---\n?", "", cleaned, count=1, flags=re.DOTALL)
        cleaned = cleaned.strip()
        if len(cleaned) < 100:
            continue
        records.append({
            "text": cleaned,
            "source": f"github-docs/{path.relative_to(dest)}",
            "type": "github_docs",
        })
    print(f"  GitHub Docs: {len(records)} documents")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect git documentation sources into JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to write docs.jsonl (default: data/raw)",
    )
    parser.add_argument(
        "--clone-dir",
        type=str,
        default=None,
        help="Directory for cloned repos (default: a temp directory)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Use a persistent temp dir if none supplied so reruns don't re-clone
    if args.clone_dir is None:
        clone_dir = os.path.join(tempfile.gettempdir(), "model-nano-sources")
    else:
        clone_dir = args.clone_dir
    os.makedirs(clone_dir, exist_ok=True)
    print(f"Clone directory: {clone_dir}")

    all_records: list[dict] = []

    collectors = [
        ("Pro Git book", collect_progit),
        ("Git man pages", collect_git_manpages),
        ("tldr-pages", collect_tldr),
        ("GitHub Docs", collect_github_docs),
    ]

    for name, fn in collectors:
        print(f"\n--- {name} ---")
        try:
            records = fn(clone_dir)
            all_records.extend(records)
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")

    # Write JSONL
    output_path = os.path.join(args.output_dir, "docs.jsonl")
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_records)} records to {output_path}")
    print("Breakdown:")
    from collections import Counter
    for doc_type, count in Counter(r["type"] for r in all_records).most_common():
        print(f"  {doc_type}: {count}")


if __name__ == "__main__":
    main()
