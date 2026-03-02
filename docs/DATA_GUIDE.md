# Data Guide

How data is collected, processed, and formatted for model-nano training.

---

## Data Sources Overview

model-nano uses a hybrid dataset combining real documentation with synthetic data.

### Real Data Sources

| Source | Script | Description | Format |
|--------|--------|-------------|--------|
| Pro Git book | `collect_docs.py` | CC-licensed git textbook | AsciiDoc -> plain text |
| Git man pages | `collect_docs.py` | Official reference for 168 commands | AsciiDoc -> plain text |
| tldr-pages | `collect_docs.py` | Community-maintained command examples | Markdown -> plain text |
| GitHub Docs | `collect_docs.py` | Workflows, API, guides | Markdown -> plain text |
| Stack Overflow | `collect_stackoverflow.py` | Community Q&A (score >= 5) | XML -> Q&A pairs |

### Synthetic Data Sources

| Strategy | Script | Description |
|----------|--------|-------------|
| Seed expansion | `generate_synthetic.py` | 50 seed commands x 10+ query variations |
| Error scenarios | `generate_synthetic.py` | 15 common errors with diagnosis + fix |
| Flag combinatorics | `generate_synthetic.py` | 15 commands x common flag combinations |

---

## Data Collection

### collect_docs.py

```bash
python data/collect_docs.py [--output-dir data/raw] [--clone-dir /tmp/model-nano-sources]
```

Clones four git repositories and extracts text:

1. **Pro Git book** (`progit/progit2`)
   - Finds all `*.asc` files recursively
   - Strips AsciiDoc markup (images, includes, cross-references, formatting)
   - Each chapter/section becomes a separate record

2. **Git man pages** (`git/git`)
   - Parses `Documentation/*.txt` files
   - Same AsciiDoc stripping
   - Each man page becomes a record

3. **tldr-pages** (`tldr-pages/tldr`)
   - Parses `pages/common/git-*.md`
   - Strips Markdown formatting
   - Each page is a compact command reference

4. **GitHub Docs** (`github/docs`)
   - Searches `content/` for git-related Markdown
   - Strips YAML front matter and Markdown formatting
   - Filters by git/GitHub keywords

Output format (JSONL):
```json
{"text": "cleaned text content", "source": "progit", "type": "documentation"}
```

### collect_stackoverflow.py

```bash
python data/collect_stackoverflow.py --input /path/to/Posts.xml
```

Parses the Stack Overflow data dump XML (streaming, memory-efficient):

1. **Pass 1**: Collect questions with `[git]` tag and score >= 5
2. **Pass 2**: Match answers (prefer accepted, then highest score)
3. **Clean**: Strip HTML with BeautifulSoup, preserve code blocks

Output format (JSONL):
```json
{
  "question_title": "How to undo git add before commit?",
  "question_body": "I mistakenly added files...",
  "answer_body": "You can use git reset HEAD <file>...",
  "text": "combined Q&A text",
  "source": "stackoverflow",
  "type": "qa"
}
```

Note: The SO data dump is large (~100 GB). Download from Archive.org and extract `Posts.xml`.

### generate_synthetic.py

```bash
python data/generate_synthetic.py [--count 5000] [--seed 42]
```

Three generation strategies:

**1. Seed Expansion**

50 hardcoded seed commands, each with:
- The git command and expected output
- 10+ natural language query templates:
  - "How do I {action}?"
  - "What's the command to {action}?"
  - "I need to {action}"
  - "Show me how to {action}"
  - etc.

Example seed:
```
Command: git stash
Action: temporarily save uncommitted changes
Explanation: Stashes changes in the working directory for later use.
```

**2. Error Scenarios**

15 common git error scenarios:
- Merge conflicts
- Detached HEAD state
- Push rejected (non-fast-forward)
- Permission denied (publickey)
- Fatal: not a git repository
- etc.

Each generates a diagnosis + fix instruction pair.

**3. Flag Combinatorics**

15 git commands with their common flag combinations:
- `git log`: `--oneline`, `--graph`, `--all`, `-n`, `--author`, `--since`, `--grep`
- `git diff`: `--staged`, `--stat`, `--name-only`, `--word-diff`
- `git commit`: `-m`, `-a`, `--amend`, `--no-edit`
- etc.

Systematically pairs each flag/combination with a question template.

Output format (JSONL):
```json
{
  "messages": [
    {"role": "system", "content": "You are a Git expert..."},
    {"role": "user", "content": "How do I see the commit history as one line per commit?"},
    {"role": "assistant", "content": "git log --oneline\n\nThis shows each commit..."}
  ]
}
```

---

## Data Preparation

```bash
python data/prepare_dataset.py [--raw-dir data/raw] [--output-dir data]
```

### Pipeline

1. **Load**: Reads all JSONL files from `data/raw/`
   - `docs.jsonl` — raw documentation text
   - `stackoverflow.jsonl` — Q&A pairs
   - `synthetic.jsonl` — instruction pairs

2. **Deduplicate**: Normalizes text (lowercase, collapse whitespace) and removes exact matches. Typically removes ~40% duplicates from synthetic data.

3. **Format**: Converts records to tokenizable text
   - **Synthetic** (has `messages` field): Already in ChatML format, formatted with `<|im_start|>` / `<|im_end|>` tags
   - **Stack Overflow** (has Q&A fields): Wrapped in ChatML with system prompt
   - **Documentation** (raw text): Used as-is for pre-training

4. **Tokenize**: Batch-encodes all text using the trained BPE tokenizer. Long sequences are split into `max_seq_len` chunks.

5. **Shuffle & Split**: Random shuffle with fixed seed, split 95% train / 5% validation.

6. **Save**: Writes uint16 numpy arrays as raw `.bin` files, loadable via `np.memmap` for memory-efficient training.

### Output Files

```
data/train.bin    # Training tokens (uint16 numpy array)
data/val.bin      # Validation tokens (uint16 numpy array)
```

### Verifying the Dataset

Check file sizes and token counts:
```bash
python -c "
import numpy as np
train = np.fromfile('data/train.bin', dtype=np.uint16)
val = np.fromfile('data/val.bin', dtype=np.uint16)
print(f'Train: {len(train):,} tokens ({len(train)*2/1e6:.1f} MB)')
print(f'Val:   {len(val):,} tokens ({len(val)*2/1e6:.1f} MB)')
"
```

Decode some tokens to verify:
```bash
python -c "
import numpy as np
from tokenizers import Tokenizer
tok = Tokenizer.from_file('tokenizer/tokenizer.json')
data = np.fromfile('data/train.bin', dtype=np.uint16)
print(tok.decode(data[:100].tolist()))
"
```

---

## ChatML Format

All instruction data uses the ChatML template:

```
<|im_start|>system
You are a Git expert. Provide precise, correct git commands and explanations.<|im_end|>
<|im_start|>user
{user query}<|im_end|>
<|im_start|>assistant
{model response}<|im_end|>
```

### Token IDs

| Token | ID |
|-------|-----|
| `<|im_start|>` | 0 |
| `<|im_end|>` | 1 |
| `<|pad|>` | 2 |

### Loss Masking

During SFT, only the assistant's response tokens contribute to the loss:

```
<|im_start|>system\n...<|im_end|>   -> mask = 0 (don't train)
<|im_start|>user\n...<|im_end|>     -> mask = 0 (don't train)
<|im_start|>assistant\n...<|im_end|> -> mask = 1 (train on this)
```

---

## Data Quality Tips

### What Makes Good Training Data

1. **Accuracy**: Every git command must be correct and produce the described effect
2. **Diversity**: Cover common and uncommon commands, simple and complex workflows
3. **Natural phrasing**: Users ask in many ways — "undo commit", "revert last change", "take back my commit"
4. **Complete explanations**: Don't just give the command — explain what it does and why

### Common Pitfalls

1. **Outdated commands**: Avoid deprecated syntax (e.g., `git checkout` vs `git switch`/`git restore`)
2. **Platform-specific**: Commands should work across Linux, macOS, Windows
3. **Missing context**: Some commands need context (branch name, remote name) — template with placeholders
4. **Duplicates**: Deduplication is important — redundant data wastes training budget

### Extending the Dataset

To add more training data:

1. Create a new JSONL file in `data/raw/` with records matching the format
2. Re-run `python data/prepare_dataset.py` to rebuild the dataset
3. The pipeline will automatically merge, dedup, and tokenize

For ChatML instruction data:
```json
{"messages": [
  {"role": "system", "content": "You are a Git expert..."},
  {"role": "user", "content": "your question"},
  {"role": "assistant", "content": "your answer"}
]}
```

For raw text data:
```json
{"text": "your documentation text", "source": "custom", "type": "documentation"}
```
