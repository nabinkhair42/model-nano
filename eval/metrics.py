"""Evaluation metrics for model-nano git command generation."""

from __future__ import annotations

import re
import shlex


def _normalize(text: str) -> str:
    """Strip, lowercase, and collapse runs of whitespace to a single space."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def exact_match(predicted: str, expected: str | list[str]) -> bool:
    """Check if predicted command matches any expected command exactly (after normalization).

    Args:
        predicted: The model's generated command string.
        expected: A single expected string or a list of acceptable alternatives.

    Returns:
        True if the normalized prediction matches any normalized expected string.
    """
    norm_pred = _normalize(predicted)
    if isinstance(expected, str):
        expected = [expected]
    return any(norm_pred == _normalize(e) for e in expected)


# ---------------------------------------------------------------------------
# Git command parsing
# ---------------------------------------------------------------------------

# Flags that are known to take a following value argument.
_VALUE_FLAGS: set[str] = {
    "-n", "--max-count", "-m", "--message", "-b", "-c", "--count",
    "--author", "--since", "--until", "--after", "--before",
    "--format", "--pretty", "--grep", "--depth", "--branch",
    "-o", "--output", "-C", "--config", "--set-upstream",
    "--track", "-u",
}


def parse_git_command(cmd: str) -> dict:
    """Parse a git command into components.

    Returns:
        {
            "base": "git log",          # the git sub-command
            "flags": {"--oneline": True, "-n": "5"},
            "args": ["main"],           # positional arguments
        }
    """
    cmd = cmd.strip()
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        # Fallback for malformed quoting
        tokens = cmd.split()

    base_parts: list[str] = []
    flags: dict[str, str | bool] = {}
    args: list[str] = []

    i = 0
    # Consume 'git' and the sub-command (e.g. 'git', 'log')
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            break
        base_parts.append(tok)
        i += 1
        # After 'git <subcommand>' stop collecting base
        if len(base_parts) >= 2:
            break

    base = " ".join(base_parts)

    # Parse remaining tokens into flags and positional args
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--":
            # Everything after -- is positional
            args.extend(tokens[i + 1:])
            break
        if tok.startswith("--"):
            if "=" in tok:
                key, val = tok.split("=", 1)
                flags[key] = val
            elif tok in _VALUE_FLAGS and i + 1 < len(tokens):
                flags[tok] = tokens[i + 1]
                i += 1
            else:
                flags[tok] = True
        elif tok.startswith("-") and len(tok) >= 2:
            # Could be a short flag or combined short flags (e.g. -am)
            # Check for '-<letter><value>' patterns like '-n5'
            if len(tok) > 2 and tok[1:2].isalpha() and not tok[2:3].isalpha():
                # Pattern like -n5
                key = tok[:2]
                val = tok[2:]
                flags[key] = val
            elif len(tok) > 2 and tok[1:].isalpha():
                # Combined flags like -am  -> -a, -m
                # But -m always takes a value, so split carefully
                j = 1
                while j < len(tok):
                    short_flag = f"-{tok[j]}"
                    if short_flag in _VALUE_FLAGS:
                        # Rest of combined string, or next token, is the value
                        rest = tok[j + 1:]
                        if rest:
                            flags[short_flag] = rest
                        elif i + 1 < len(tokens):
                            flags[short_flag] = tokens[i + 1]
                            i += 1
                        else:
                            flags[short_flag] = True
                        break
                    else:
                        flags[short_flag] = True
                    j += 1
            elif tok[:2] in _VALUE_FLAGS:
                if i + 1 < len(tokens):
                    flags[tok] = tokens[i + 1]
                    i += 1
                else:
                    flags[tok] = True
            else:
                flags[tok] = True
        else:
            args.append(tok)
        i += 1

    return {"base": base, "flags": flags, "args": args}


# ---------------------------------------------------------------------------
# Semantic equivalence rules
# ---------------------------------------------------------------------------

# Maps of (base_command, flag) -> canonical form so that synonyms collapse.
_FLAG_ALIASES: dict[str, str] = {
    "--staged": "--cached",
    "--no-edit": "--no-edit",
}

# Short flag -> long flag canonical mappings per subcommand
_SHORT_TO_LONG: dict[str, dict[str, str]] = {
    "git log": {"-n": "-n", "--max-count": "-n"},
    "git commit": {"-a": "-a", "--all": "-a", "-m": "-m", "--message": "-m"},
    "git branch": {"-d": "-d", "--delete": "-d", "-D": "-D", "--force-delete": "-D"},
    "git push": {"-u": "-u", "--set-upstream": "-u"},
    "git checkout": {"-b": "-b"},
    "git switch": {"-c": "-c", "--create": "-c"},
    "git diff": {},
    "git remote": {"-v": "-v", "--verbose": "-v"},
    "git stash": {},
}

# Base-command synonyms: map alternative forms to a canonical base
_BASE_SYNONYMS: dict[str, str] = {
    "git switch": "git checkout",
    "git restore": "git checkout",
}

# Flags that make two different base commands equivalent when present
# e.g. 'git checkout -b X' == 'git switch -c X'
_CROSS_COMMAND_EQUIVALENCES: list[tuple[dict, dict]] = [
    # git checkout -b <branch> == git switch -c <branch>
    (
        {"base": "git checkout", "required_flags": {"-b"}},
        {"base": "git switch", "required_flags": {"-c"}},
    ),
]


def _canonicalize(parsed: dict) -> tuple[str, dict[str, str | bool], list[str]]:
    """Produce a canonical representation of a parsed git command."""
    base = parsed["base"].lower().strip()
    flags = dict(parsed["flags"])
    args = list(parsed["args"])

    # Canonicalize flag aliases (--staged -> --cached, etc.)
    new_flags: dict[str, str | bool] = {}
    for k, v in flags.items():
        canon_key = _FLAG_ALIASES.get(k, k)
        # Also apply per-subcommand short/long mappings
        sub_map = _SHORT_TO_LONG.get(base, {})
        canon_key = sub_map.get(canon_key, canon_key)
        new_flags[canon_key] = v

    return base, new_flags, args


def command_equivalence(predicted: str, expected: str) -> bool:
    """Check if two git commands are semantically equivalent.

    Handles common equivalences such as:
    - git log -5            == git log -n 5
    - git commit -am "msg"  == git commit -a -m "msg"
    - git checkout -b br    == git switch -c br
    - git diff --staged     == git diff --cached

    Args:
        predicted: The model's generated command.
        expected: The reference command.

    Returns:
        True if the commands are semantically equivalent.
    """
    p_parsed = parse_git_command(predicted)
    e_parsed = parse_git_command(expected)

    p_base, p_flags, p_args = _canonicalize(p_parsed)
    e_base, e_flags, e_args = _canonicalize(e_parsed)

    # Direct match after canonicalization
    if p_base == e_base and p_flags == e_flags and p_args == e_args:
        return True

    # Check cross-command equivalences (e.g. checkout -b == switch -c)
    for rule_a, rule_b in _CROSS_COMMAND_EQUIVALENCES:
        # Try both orderings: (predicted matches A and expected matches B) or vice-versa
        for (check_base_1, check_flags_1, check_args_1), (check_base_2, check_flags_2, check_args_2), ra, rb in [
            ((p_base, p_flags, p_args), (e_base, e_flags, e_args), rule_a, rule_b),
            ((e_base, e_flags, e_args), (p_base, p_flags, p_args), rule_a, rule_b),
        ]:
            if check_base_1 == ra["base"] and check_base_2 == rb["base"]:
                # Verify required flags are present
                if ra["required_flags"].issubset(set(check_flags_1.keys())) and \
                   rb["required_flags"].issubset(set(check_flags_2.keys())):
                    # Build comparable flag/arg sets by removing the distinguishing flags
                    f1 = {k: v for k, v in check_flags_1.items() if k not in ra["required_flags"]}
                    f2 = {k: v for k, v in check_flags_2.items() if k not in rb["required_flags"]}
                    # Collect the values of the distinguishing flags (e.g. branch name)
                    vals_1 = [check_flags_1[f] for f in ra["required_flags"] if f in check_flags_1]
                    vals_2 = [check_flags_2[f] for f in rb["required_flags"] if f in check_flags_2]
                    if f1 == f2 and check_args_1 == check_args_2 and vals_1 == vals_2:
                        return True

    # Handle git log -<N> == git log -n <N>
    if p_base == e_base == "git log":
        p_n = p_flags.pop("-n", None)
        e_n = e_flags.pop("-n", None)
        # Look for bare -<number> which parse_git_command might store as a flag
        for flags_dict, n_ref in [(p_flags, "p_n"), (e_flags, "e_n")]:
            for k in list(flags_dict.keys()):
                if re.match(r"^-\d+$", k):
                    val = k[1:]  # strip the dash
                    if n_ref == "p_n" and p_n is None:
                        p_n = val
                    elif n_ref == "e_n" and e_n is None:
                        e_n = val
                    del flags_dict[k]
        if p_n is not None and e_n is not None:
            if str(p_n) == str(e_n) and p_flags == e_flags and p_args == e_args:
                return True

    return False


# ---------------------------------------------------------------------------
# Response quality scoring
# ---------------------------------------------------------------------------

def response_quality(predicted: str, expected: str) -> float:
    """Score explanation quality using simple heuristics.

    Scoring criteria (each contributes to the final 0.0-1.0 score):
    - Term overlap: what fraction of key terms from the expected answer appear
      in the prediction (weight: 0.6).
    - Length penalty: moderate-length responses score higher; very short or
      very long responses are penalized (weight: 0.2).
    - Structure bonus: presence of code blocks, bullet points, or numbered
      lists indicates a well-structured answer (weight: 0.2).

    Args:
        predicted: The model's generated explanation.
        expected: The reference explanation.

    Returns:
        A float between 0.0 and 1.0.
    """
    if not predicted or not predicted.strip():
        return 0.0

    pred_lower = predicted.lower()
    exp_lower = expected.lower()

    # --- Term overlap (0.6 weight) ---
    # Extract significant terms (3+ chars, skip common stop words)
    stop_words = {
        "the", "and", "for", "you", "can", "are", "this", "that", "with",
        "from", "will", "your", "has", "have", "was", "were", "been", "not",
        "but", "they", "all", "also", "its", "use", "using",
    }
    exp_terms = {
        w for w in re.findall(r"[a-z]{3,}", exp_lower) if w not in stop_words
    }
    if exp_terms:
        matches = sum(1 for t in exp_terms if t in pred_lower)
        term_score = min(matches / len(exp_terms), 1.0)
    else:
        term_score = 0.5  # neutral when expected has no key terms

    # --- Length penalty (0.2 weight) ---
    pred_words = len(predicted.split())
    exp_words = max(len(expected.split()), 1)
    ratio = pred_words / exp_words
    if ratio < 0.3:
        length_score = ratio / 0.3  # too short
    elif ratio > 3.0:
        length_score = max(0.0, 1.0 - (ratio - 3.0) / 5.0)  # too long
    else:
        length_score = 1.0

    # --- Structure bonus (0.2 weight) ---
    structure_score = 0.0
    if re.search(r"```", predicted):
        structure_score += 0.4
    if re.search(r"^[\-\*]\s", predicted, re.MULTILINE):
        structure_score += 0.3
    if re.search(r"^\d+[\.\)]\s", predicted, re.MULTILINE):
        structure_score += 0.3
    # Even a plain paragraph gets partial credit
    if predicted.strip() and structure_score == 0.0:
        structure_score = 0.3
    structure_score = min(structure_score, 1.0)

    # Weighted combination
    score = 0.6 * term_score + 0.2 * length_score + 0.2 * structure_score
    return round(min(max(score, 0.0), 1.0), 4)
