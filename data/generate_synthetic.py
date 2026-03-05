"""Generate synthetic git training data in ChatML instruction-pair format.

Three generation strategies:
  1. Seed expansion   -- rephrase common git commands in varied natural language
  2. Error scenarios  -- common git errors with diagnosis and fix
  3. Flag combinatorics -- systematic command + flag pairings

Output format (one JSON object per line):
  {"messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
  ]}

Output: data/raw/synthetic.jsonl
"""

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import NamedTuple

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DataConfig

# Use centralized system prompt from config
SYSTEM_PROMPT = DataConfig.system_prompt


# ═══════════════════════════════════════════════════════════════════════════
# 1. Seed Commands
# ═══════════════════════════════════════════════════════════════════════════

class SeedCommand(NamedTuple):
    command: str
    description: str


SEED_COMMANDS: list[SeedCommand] = [
    SeedCommand("git init", "Initialize a new Git repository"),
    SeedCommand("git clone <url>", "Clone a remote repository"),
    SeedCommand("git add .", "Stage all changes"),
    SeedCommand("git add <file>", "Stage a specific file"),
    SeedCommand("git commit -m '<message>'", "Commit staged changes with a message"),
    SeedCommand("git status", "Show the working tree status"),
    SeedCommand("git log", "Show the commit history"),
    SeedCommand("git log --oneline", "Show compact commit history"),
    SeedCommand("git log --graph --oneline --all", "Show a visual branch graph"),
    SeedCommand("git diff", "Show unstaged changes"),
    SeedCommand("git diff --staged", "Show staged changes"),
    SeedCommand("git diff HEAD~1", "Show changes since the previous commit"),
    SeedCommand("git branch", "List all local branches"),
    SeedCommand("git branch <name>", "Create a new branch"),
    SeedCommand("git branch -d <name>", "Delete a branch"),
    SeedCommand("git checkout <branch>", "Switch to a branch"),
    SeedCommand("git checkout -b <branch>", "Create and switch to a new branch"),
    SeedCommand("git switch <branch>", "Switch branches (modern command)"),
    SeedCommand("git switch -c <branch>", "Create and switch to a new branch (modern)"),
    SeedCommand("git merge <branch>", "Merge a branch into the current branch"),
    SeedCommand("git rebase <branch>", "Rebase current branch onto another branch"),
    SeedCommand("git rebase -i HEAD~<n>", "Interactive rebase of the last n commits"),
    SeedCommand("git pull", "Fetch and merge changes from the remote"),
    SeedCommand("git pull --rebase", "Fetch and rebase instead of merge"),
    SeedCommand("git push", "Push commits to the remote"),
    SeedCommand("git push -u origin <branch>", "Push and set upstream tracking"),
    SeedCommand("git push --force-with-lease", "Force push safely"),
    SeedCommand("git fetch", "Download objects and refs from a remote"),
    SeedCommand("git fetch --all", "Fetch from all remotes"),
    SeedCommand("git remote -v", "List remote repositories"),
    SeedCommand("git remote add origin <url>", "Add a remote repository"),
    SeedCommand("git stash", "Stash uncommitted changes"),
    SeedCommand("git stash pop", "Apply and remove the latest stash"),
    SeedCommand("git stash list", "List all stashes"),
    SeedCommand("git stash apply", "Apply a stash without removing it"),
    SeedCommand("git cherry-pick <hash>", "Apply a specific commit to current branch"),
    SeedCommand("git reset HEAD <file>", "Unstage a file"),
    SeedCommand("git reset --soft HEAD~1", "Undo last commit, keep changes staged"),
    SeedCommand("git reset --hard HEAD~1", "Undo last commit and discard changes"),
    SeedCommand("git revert <hash>", "Create a new commit that undoes a previous commit"),
    SeedCommand("git tag <name>", "Create a lightweight tag"),
    SeedCommand("git tag -a <name> -m '<msg>'", "Create an annotated tag"),
    SeedCommand("git show <ref>", "Show details of a commit or tag"),
    SeedCommand("git blame <file>", "Show who changed each line of a file"),
    SeedCommand("git bisect start", "Start binary search for a bug-introducing commit"),
    SeedCommand("git clean -fd", "Remove untracked files and directories"),
    SeedCommand("git config --global user.name '<name>'", "Set your Git username globally"),
    SeedCommand("git config --global user.email '<email>'", "Set your Git email globally"),
    SeedCommand("git rm <file>", "Remove a file from tracking and working tree"),
    SeedCommand("git mv <old> <new>", "Rename/move a tracked file"),
    SeedCommand("git reflog", "Show reference log of HEAD changes"),
]

# Natural language question templates for seed expansion
_QUESTION_TEMPLATES: list[str] = [
    "How do I {action}?",
    "What's the git command to {action}?",
    "I want to {action}. What should I run?",
    "Can you show me how to {action} in git?",
    "What is the command for {action_gerund}?",
    "How can I {action} using git?",
    "Tell me the git command to {action}.",
    "I need to {action}. Help?",
    "Show me how to {action} with git.",
    "What git command lets me {action}?",
]


def _action_from_description(desc: str) -> str:
    """Convert a description like 'Show the commit history' -> 'show the commit history'."""
    return desc[0].lower() + desc[1:]


def _gerund_from_description(desc: str) -> str:
    """Very simple gerund form: 'Show ...' -> 'showing ...'."""
    words = desc.split()
    verb = words[0].lower()
    if verb.endswith("e") and not verb.endswith("ee"):
        verb = verb[:-1] + "ing"
    elif len(verb) >= 3 and verb[-1] not in "aeiou" and verb[-2] in "aeiou" and verb[-3] not in "aeiou":
        verb = verb + verb[-1] + "ing"
    else:
        verb = verb + "ing"
    return " ".join([verb] + words[1:]).lower()


def _answer_for_seed(seed: SeedCommand) -> str:
    """Generate a detailed answer for a seed command."""
    return (
        f"You can {_action_from_description(seed.description)} with:\n\n"
        f"```\n{seed.command}\n```\n\n"
        f"This command will {_action_from_description(seed.description)}."
    )


def generate_seed_expansion(rng: random.Random, count: int) -> list[dict]:
    """Generate varied natural language phrasings for seed commands."""
    records: list[dict] = []
    for _ in range(count):
        seed = rng.choice(SEED_COMMANDS)
        template = rng.choice(_QUESTION_TEMPLATES)
        action = _action_from_description(seed.description)
        action_gerund = _gerund_from_description(seed.description)
        question = template.format(action=action, action_gerund=action_gerund)
        answer = _answer_for_seed(seed)
        records.append(_make_chatml(question, answer))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# 2. Error Scenarios
# ═══════════════════════════════════════════════════════════════════════════

class ErrorScenario(NamedTuple):
    error_message: str
    diagnosis: str
    fix: str


ERROR_SCENARIOS: list[ErrorScenario] = [
    ErrorScenario(
        error_message="CONFLICT (content): Merge conflict in <file>",
        diagnosis=(
            "This means Git could not automatically merge changes because both "
            "branches modified the same lines in the same file."
        ),
        fix=(
            "1. Open the conflicting file and look for conflict markers "
            "(<<<<<<< HEAD, =======, >>>>>>> branch).\n"
            "2. Edit the file to resolve the conflict by choosing or combining changes.\n"
            "3. Stage the resolved file: `git add <file>`\n"
            "4. Complete the merge: `git commit`"
        ),
    ),
    ErrorScenario(
        error_message="You are in 'detached HEAD' state.",
        diagnosis=(
            "A detached HEAD means you are not on any branch -- you've checked "
            "out a specific commit, tag, or remote ref directly. Commits made here "
            "will be orphaned when you switch away."
        ),
        fix=(
            "To keep your work, create a new branch from this point:\n\n"
            "```\ngit checkout -b <new-branch-name>\n```\n\n"
            "Or, to simply return to an existing branch:\n\n"
            "```\ngit checkout main\n```"
        ),
    ),
    ErrorScenario(
        error_message="error: failed to push some refs to '<remote>'",
        diagnosis=(
            "The remote has commits that your local branch doesn't have. Git "
            "refuses to push because it would overwrite those commits."
        ),
        fix=(
            "Pull the remote changes first, then push:\n\n"
            "```\ngit pull --rebase\ngit push\n```\n\n"
            "If you are certain you want to overwrite the remote:\n\n"
            "```\ngit push --force-with-lease\n```"
        ),
    ),
    ErrorScenario(
        error_message="fatal: not a git repository (or any of the parent directories): .git",
        diagnosis=(
            "You are running a git command outside of any Git repository. There "
            "is no .git directory in this folder or any parent folder."
        ),
        fix=(
            "Navigate to your project directory, or initialize a new repository:\n\n"
            "```\ngit init\n```"
        ),
    ),
    ErrorScenario(
        error_message="error: Your local changes to the following files would be overwritten by merge",
        diagnosis=(
            "You have uncommitted changes in files that would be modified by the "
            "merge/pull. Git refuses to proceed to avoid data loss."
        ),
        fix=(
            "Either commit your changes first:\n\n"
            "```\ngit add .\ngit commit -m 'Save work in progress'\n```\n\n"
            "Or stash them:\n\n"
            "```\ngit stash\ngit pull\ngit stash pop\n```"
        ),
    ),
    ErrorScenario(
        error_message="fatal: refusing to merge unrelated histories",
        diagnosis=(
            "The two branches have no common ancestor, which typically happens "
            "when you try to merge a freshly-created remote repo (with its own "
            "initial commit) into a local repo."
        ),
        fix=(
            "Allow the merge with:\n\n"
            "```\ngit pull origin main --allow-unrelated-histories\n```"
        ),
    ),
    ErrorScenario(
        error_message="error: pathspec '<file>' did not match any file(s) known to git",
        diagnosis=(
            "The file you specified does not exist in the working tree or the index. "
            "This can happen if you have a typo or the file was never tracked."
        ),
        fix=(
            "Double-check the filename and path. Use `git status` to see tracked "
            "and untracked files. If the file exists but isn't tracked:\n\n"
            "```\ngit add <file>\n```"
        ),
    ),
    ErrorScenario(
        error_message="fatal: The current branch <branch> has no upstream branch.",
        diagnosis=(
            "Your local branch is not tracking any remote branch, so `git push` "
            "doesn't know where to push."
        ),
        fix=(
            "Set the upstream with:\n\n"
            "```\ngit push -u origin <branch>\n```"
        ),
    ),
    ErrorScenario(
        error_message="error: cannot delete branch '<branch>' checked out at '<path>'",
        diagnosis=(
            "You cannot delete the branch you are currently on."
        ),
        fix=(
            "Switch to a different branch first, then delete:\n\n"
            "```\ngit checkout main\ngit branch -d <branch>\n```"
        ),
    ),
    ErrorScenario(
        error_message="warning: LF will be replaced by CRLF in <file>",
        diagnosis=(
            "Git is converting line endings because your core.autocrlf setting "
            "is enabled. This is common on Windows."
        ),
        fix=(
            "To suppress the warning, configure your preferred line-ending behaviour:\n\n"
            "```\n# On Windows (convert to CRLF on checkout, LF on commit):\n"
            "git config --global core.autocrlf true\n\n"
            "# On macOS/Linux (only convert CRLF to LF on commit):\n"
            "git config --global core.autocrlf input\n```"
        ),
    ),
    ErrorScenario(
        error_message="fatal: bad object HEAD",
        diagnosis=(
            "The HEAD reference is corrupted or points to a non-existent object. "
            "This usually indicates repository corruption."
        ),
        fix=(
            "Try to recover with:\n\n"
            "```\ngit fsck --full\n```\n\n"
            "If the reflog is intact:\n\n"
            "```\ngit reflog\ngit reset --hard <last-good-commit>\n```"
        ),
    ),
    ErrorScenario(
        error_message="error: you need to resolve your current index first",
        diagnosis=(
            "A previous merge or rebase left unresolved conflicts. Git won't let "
            "you switch branches or perform other operations until they are resolved."
        ),
        fix=(
            "Resolve the conflicts and commit, or abort the merge/rebase:\n\n"
            "```\n# To abort the merge:\n"
            "git merge --abort\n\n"
            "# To abort the rebase:\n"
            "git rebase --abort\n```"
        ),
    ),
    ErrorScenario(
        error_message="fatal: 'origin' does not appear to be a git repository",
        diagnosis=(
            "The remote URL for 'origin' is missing or invalid."
        ),
        fix=(
            "Check your remotes and fix the URL:\n\n"
            "```\ngit remote -v\ngit remote set-url origin <correct-url>\n```"
        ),
    ),
    ErrorScenario(
        error_message="error: The following untracked working tree files would be overwritten by checkout",
        diagnosis=(
            "The branch you're switching to contains files that exist as untracked "
            "files in your working directory. Git won't overwrite them."
        ),
        fix=(
            "Remove or rename the conflicting files, or commit them first:\n\n"
            "```\ngit stash --include-untracked\ngit checkout <branch>\ngit stash pop\n```"
        ),
    ),
    ErrorScenario(
        error_message="fatal: unable to access 'https://...': Could not resolve host: github.com",
        diagnosis=(
            "Git cannot reach the remote server. This is usually a network issue, "
            "DNS resolution problem, or proxy misconfiguration."
        ),
        fix=(
            "Check your internet connection and DNS settings. If behind a proxy:\n\n"
            "```\ngit config --global http.proxy http://proxy:port\n```\n\n"
            "To unset a proxy:\n\n"
            "```\ngit config --global --unset http.proxy\n```"
        ),
    ),
]

# Question templates for error scenarios
_ERROR_QUESTION_TEMPLATES: list[str] = [
    "I got this error: \"{error}\". What does it mean and how do I fix it?",
    "Git shows: \"{error}\". Help!",
    "What does this git error mean? \"{error}\"",
    "How do I resolve: \"{error}\"?",
    "I'm getting \"{error}\" when running git. What should I do?",
]


def generate_error_scenarios(rng: random.Random, count: int) -> list[dict]:
    """Generate error diagnosis + fix pairs."""
    records: list[dict] = []
    for _ in range(count):
        scenario = rng.choice(ERROR_SCENARIOS)
        template = rng.choice(_ERROR_QUESTION_TEMPLATES)
        question = template.format(error=scenario.error_message)
        answer = f"**Error:** `{scenario.error_message}`\n\n**Diagnosis:**\n{scenario.diagnosis}\n\n**Fix:**\n{scenario.fix}"
        records.append(_make_chatml(question, answer))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# 3. Flag Combinatorics
# ═══════════════════════════════════════════════════════════════════════════

# command -> list of (flag_combo, description)
FLAG_COMBOS: dict[str, list[tuple[str, str]]] = {
    "git log": [
        ("--oneline", "Show each commit on a single line"),
        ("--oneline --graph", "Show a visual ASCII graph of branches"),
        ("--oneline --graph --all", "Show all branches in a visual graph"),
        ("-1", "Show only the most recent commit"),
        ("-n 5", "Show the last 5 commits"),
        ("--stat", "Show file change statistics per commit"),
        ("--patch", "Show the full diff for each commit"),
        ("--author='<name>'", "Filter commits by author name"),
        ("--since='2 weeks ago'", "Show commits from the last 2 weeks"),
        ("--grep='<pattern>'", "Search commit messages for a pattern"),
        ("--oneline --all --decorate", "Show all branches with ref names"),
        ("--format='%h %s (%an, %ar)'", "Custom format: short hash, subject, author, relative date"),
    ],
    "git diff": [
        ("--staged", "Show staged (cached) changes"),
        ("--cached", "Alias for --staged"),
        ("--name-only", "Show only the names of changed files"),
        ("--stat", "Show a summary of changes per file"),
        ("--word-diff", "Show word-level diff instead of line-level"),
        ("HEAD~1", "Show changes since the previous commit"),
        ("<branch1>..<branch2>", "Show differences between two branches"),
        ("--no-index <file1> <file2>", "Compare two files outside a repo"),
    ],
    "git commit": [
        ("-m '<message>'", "Commit with an inline message"),
        ("-am '<message>'", "Stage tracked files and commit with a message"),
        ("--amend", "Amend the previous commit"),
        ("--amend --no-edit", "Amend without changing the commit message"),
        ("--allow-empty -m '<message>'", "Create an empty commit"),
        ("-S", "GPG-sign the commit"),
        ("--fixup <hash>", "Create a fixup commit for use with rebase --autosquash"),
    ],
    "git branch": [
        ("-a", "List all branches (local and remote)"),
        ("-r", "List remote branches"),
        ("-d <name>", "Delete a merged branch"),
        ("-D <name>", "Force-delete a branch (even unmerged)"),
        ("-m <old> <new>", "Rename a branch"),
        ("--merged", "List branches merged into the current branch"),
        ("--no-merged", "List branches not yet merged"),
        ("-vv", "Show branches with upstream tracking info"),
    ],
    "git stash": [
        ("push -m '<message>'", "Stash with a descriptive message"),
        ("pop", "Apply and remove the latest stash"),
        ("apply", "Apply the latest stash without removing it"),
        ("list", "List all stashes"),
        ("drop", "Remove the latest stash"),
        ("show -p", "Show the diff of the latest stash"),
        ("branch <name>", "Create a branch from a stash"),
        ("push --include-untracked", "Stash including untracked files"),
        ("push --keep-index", "Stash but keep staged changes"),
    ],
    "git reset": [
        ("HEAD <file>", "Unstage a specific file"),
        ("--soft HEAD~1", "Undo last commit, keep changes staged"),
        ("--mixed HEAD~1", "Undo last commit, keep changes unstaged"),
        ("--hard HEAD~1", "Undo last commit, discard all changes (destructive)"),
        ("--hard origin/main", "Reset branch to match remote main"),
    ],
    "git clean": [
        ("-n", "Dry run: show what would be removed"),
        ("-f", "Remove untracked files"),
        ("-fd", "Remove untracked files and directories"),
        ("-fX", "Remove only ignored files"),
        ("-fdx", "Remove all untracked and ignored files and directories"),
    ],
    "git remote": [
        ("-v", "List remotes with URLs"),
        ("add <name> <url>", "Add a new remote"),
        ("remove <name>", "Remove a remote"),
        ("rename <old> <new>", "Rename a remote"),
        ("set-url <name> <url>", "Change a remote's URL"),
        ("show <name>", "Show detailed information about a remote"),
    ],
    "git tag": [
        ("-l", "List all tags"),
        ("-l 'v1.*'", "List tags matching a pattern"),
        ("-a <name> -m '<msg>'", "Create an annotated tag"),
        ("-d <name>", "Delete a local tag"),
        ("-n", "List tags with their messages"),
    ],
    "git fetch": [
        ("--all", "Fetch from all remotes"),
        ("--prune", "Remove stale remote-tracking branches"),
        ("--tags", "Fetch all tags from the remote"),
        ("origin <branch>", "Fetch a specific branch from origin"),
    ],
    "git push": [
        ("-u origin <branch>", "Push and set upstream tracking"),
        ("--force-with-lease", "Force push safely (checks remote hasn't changed)"),
        ("--tags", "Push all tags to the remote"),
        ("origin --delete <branch>", "Delete a remote branch"),
        ("--dry-run", "Simulate push without actually pushing"),
    ],
    "git rebase": [
        ("-i HEAD~<n>", "Interactive rebase of last n commits"),
        ("--onto <base> <old> <new>", "Rebase a range of commits onto a new base"),
        ("--continue", "Continue after resolving conflicts"),
        ("--abort", "Abort the rebase and restore original state"),
        ("--skip", "Skip the current conflicting commit"),
        ("--autosquash", "Auto-reorder fixup/squash commits"),
    ],
    "git cherry-pick": [
        ("<hash>", "Apply a specific commit"),
        ("<hash1>..<hash2>", "Apply a range of commits"),
        ("--no-commit <hash>", "Apply changes without committing"),
        ("--abort", "Abort cherry-pick in progress"),
        ("-x <hash>", "Append '(cherry picked from commit ...)' to message"),
    ],
}

_FLAG_QUESTION_TEMPLATES: list[str] = [
    "How do I use `{cmd} {flags}`?",
    "What does `{cmd} {flags}` do?",
    "Explain the command: {cmd} {flags}",
    "When would I use `{cmd} {flags}`?",
]


def generate_flag_combinatorics(rng: random.Random, count: int) -> list[dict]:
    """Systematically pair git commands with their flag combinations."""
    # Flatten all combos
    all_combos: list[tuple[str, str, str]] = []
    for cmd, flag_list in FLAG_COMBOS.items():
        for flags, desc in flag_list:
            all_combos.append((cmd, flags, desc))

    records: list[dict] = []
    for _ in range(count):
        cmd, flags, desc = rng.choice(all_combos)
        template = rng.choice(_FLAG_QUESTION_TEMPLATES)
        question = template.format(cmd=cmd, flags=flags)
        full_cmd = f"{cmd} {flags}"
        answer = (
            f"`{full_cmd}`\n\n"
            f"{desc}.\n\n"
            f"Usage:\n```\n{full_cmd}\n```"
        )
        records.append(_make_chatml(question, answer))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_chatml(user_content: str, assistant_content: str) -> dict:
    """Create a ChatML-formatted record."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic git training data in ChatML format.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to write synthetic.jsonl (default: data/raw)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Total examples to generate, split 40/20/40 across strategies (overrides individual counts)",
    )
    parser.add_argument(
        "--count-seed",
        type=int,
        default=2000,
        help="Number of seed-expansion examples (default: 2000)",
    )
    parser.add_argument(
        "--count-errors",
        type=int,
        default=1000,
        help="Number of error-scenario examples (default: 1000)",
    )
    parser.add_argument(
        "--count-flags",
        type=int,
        default=2000,
        help="Number of flag-combinatorics examples (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # --count splits total 40% seed / 20% errors / 40% flags
    if args.count is not None:
        args.count_seed   = int(args.count * 0.40)
        args.count_errors = int(args.count * 0.20)
        args.count_flags  = int(args.count * 0.40)

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating synthetic git training data ...")

    print(f"  Seed expansion: {args.count_seed} examples")
    seed_data = generate_seed_expansion(rng, args.count_seed)

    print(f"  Error scenarios: {args.count_errors} examples")
    error_data = generate_error_scenarios(rng, args.count_errors)

    print(f"  Flag combinatorics: {args.count_flags} examples")
    flag_data = generate_flag_combinatorics(rng, args.count_flags)

    all_data = seed_data + error_data + flag_data
    rng.shuffle(all_data)

    output_path = os.path.join(args.output_dir, "synthetic.jsonl")
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in all_data:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(all_data)
    print(f"\nWrote {total} records to {output_path}")
    print(f"  Seed expansion:     {len(seed_data)}")
    print(f"  Error scenarios:    {len(error_data)}")
    print(f"  Flag combinatorics: {len(flag_data)}")


if __name__ == "__main__":
    main()
