"""Git repo context detection for model-nano CLI."""

import subprocess
import re


class GitContext:
    """Detect and provide current git repository context."""

    # Patterns that indicate destructive git operations
    DESTRUCTIVE_PATTERNS = [
        r"push\s+.*--force",
        r"push\s+.*-f\b",
        r"reset\s+--hard",
        r"clean\s+-f",
        r"clean\s+.*-fd",
        r"checkout\s+--\s+\.",
        r"branch\s+-D\b",
        r"branch\s+.*--delete\s+--force",
        r"rebase\s+.*--force",
        r"stash\s+drop",
        r"stash\s+clear",
        r"reflog\s+expire",
        r"reflog\s+delete",
        r"gc\s+--prune",
        r"filter-branch",
    ]

    def __init__(self):
        self.is_git_repo = False
        self.branch = None
        self.status = None
        self.recent_commits = []
        self.remotes = []
        self._modified_count = 0
        self._staged_count = 0
        self._untracked_count = 0
        self._detect()

    def _run_git(self, *args: str) -> str | None:
        """Run a git command and return stdout, or None on failure."""
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _detect(self):
        """Auto-detect git state using subprocess calls to git."""
        # Check if we're in a git repo
        check = self._run_git("rev-parse", "--is-inside-work-tree")
        if check != "true":
            self.is_git_repo = False
            return

        self.is_git_repo = True

        # Get current branch name
        branch = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        self.branch = branch if branch else "(detached)"

        # Get short status (modified/staged/untracked counts)
        status_output = self._run_git("status", "--porcelain")
        if status_output is not None:
            self.status = status_output
            for line in status_output.splitlines():
                if not line or len(line) < 2:
                    continue
                index_status = line[0]
                worktree_status = line[1]
                if index_status in ("M", "A", "D", "R", "C"):
                    self._staged_count += 1
                if worktree_status in ("M", "D"):
                    self._modified_count += 1
                if index_status == "?" and worktree_status == "?":
                    self._untracked_count += 1
        else:
            self.status = ""

        # Get last 5 commit messages (oneline)
        log_output = self._run_git("log", "--oneline", "-5", "--no-decorate")
        if log_output:
            self.recent_commits = log_output.splitlines()
        else:
            self.recent_commits = []

        # Get remote names
        remotes_output = self._run_git("remote")
        if remotes_output:
            self.remotes = remotes_output.splitlines()
        else:
            self.remotes = []

    def summary(self) -> str:
        """Return a concise context string to include in prompts."""
        if not self.is_git_repo:
            return "Not in a git repository"

        parts = [f"Branch: {self.branch}"]
        parts.append(f"Modified: {self._modified_count}")
        parts.append(f"Staged: {self._staged_count}")
        parts.append(f"Untracked: {self._untracked_count}")

        if self.recent_commits:
            # Extract just the message part (after the short hash)
            first_commit = self.recent_commits[0]
            # Format is "abc1234 commit message"
            msg = first_commit.split(" ", 1)[1] if " " in first_commit else first_commit
            parts.append(f"Last commit: {msg}")

        if self.remotes:
            parts.append(f"Remotes: {', '.join(self.remotes)}")

        return " | ".join(parts)

    def prompt_context(self) -> str:
        """Return a fuller context block suitable for embedding in a model prompt."""
        if not self.is_git_repo:
            return "The user is NOT inside a git repository."

        lines = [
            f"Git repository context:",
            f"  Branch: {self.branch}",
            f"  Modified files: {self._modified_count}",
            f"  Staged files: {self._staged_count}",
            f"  Untracked files: {self._untracked_count}",
        ]

        if self.recent_commits:
            lines.append("  Recent commits:")
            for commit in self.recent_commits:
                lines.append(f"    - {commit}")

        if self.remotes:
            lines.append(f"  Remotes: {', '.join(self.remotes)}")

        return "\n".join(lines)

    def is_destructive(self, command: str) -> bool:
        """Check if a git command is destructive (force-push, reset --hard, etc)."""
        # Normalize whitespace
        normalized = " ".join(command.strip().split())
        # Strip leading 'git ' if present
        if normalized.startswith("git "):
            normalized = normalized[4:]

        for pattern in self.DESTRUCTIVE_PATTERNS:
            if re.search(pattern, normalized):
                return True
        return False
