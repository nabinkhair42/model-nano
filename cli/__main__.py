"""Entry point for the git-nano CLI tool."""

import sys

import click
from rich.console import Console

console = Console()

# Subcommand names for disambiguation
_SUBCOMMANDS = {"chat", "explain"}


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to the best available option."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _load_engine(model: str | None, tokenizer: str | None, device: str):
    """Load the inference engine, handling missing dependencies gracefully.

    Returns the engine instance, or None if loading fails.
    """
    resolved_device = _resolve_device(device)

    # Default paths if not specified
    if model is None:
        model = "checkpoints/model.pt"
    if tokenizer is None:
        tokenizer = "tokenizer/tokenizer.json"

    try:
        from inference.engine import InferenceEngine

        engine = InferenceEngine(model, tokenizer, resolved_device)
        return engine
    except FileNotFoundError as e:
        console.print(
            f"[yellow]Model/tokenizer not found: {e}[/yellow]\n"
            f"[dim]Looked for model at: {model}[/dim]\n"
            f"[dim]Looked for tokenizer at: {tokenizer}[/dim]\n"
            f"[dim]Train or download a model first. See README for instructions.[/dim]"
        )
        return None
    except ImportError as e:
        console.print(
            f"[yellow]Missing dependency: {e}[/yellow]\n"
            f"[dim]Install with: pip install model-nano[/dim]"
        )
        return None
    except Exception as e:
        console.print(f"[red]Failed to load engine: {e}[/red]")
        return None


def _read_piped_input() -> str | None:
    """Read piped stdin if available. Returns the text or None."""
    if sys.stdin.isatty():
        return None
    try:
        data = sys.stdin.read()
        return data.strip() if data.strip() else None
    except Exception:
        return None


class NaturalOrderGroup(click.Group):
    """A Click group that allows free-form query arguments alongside subcommands.

    Click normally cannot mix variadic arguments with subcommands. This custom
    group resolves the ambiguity: if the first non-option argument matches a
    known subcommand name, it is dispatched as a subcommand; otherwise all
    remaining arguments are collected as the one-shot query.
    """

    def parse_args(self, ctx, args):
        """Override to separate query words from subcommands."""
        # Collect any leading options (--model, etc.) first, then look at positionals
        # We let Click's default parsing handle subcommand detection.
        # If the first positional arg is NOT a known subcommand, stash everything
        # as the query and prevent Click from looking for subcommands.

        # Find where positional args start (skip options and their values)
        positional_start = 0
        i = 0
        option_names = set()
        for param in self.params:
            if isinstance(param, click.Option):
                option_names.update(param.opts)
                option_names.update(param.secondary_opts)

        while i < len(args):
            if args[i] in option_names:
                i += 2  # skip option and its value
            elif args[i].startswith("-"):
                # Could be --help or unknown flag
                break
            else:
                break
            positional_start = i

        positional_start = i

        # Check if the first positional is a subcommand
        if positional_start < len(args) and args[positional_start] in _SUBCOMMANDS:
            # It's a subcommand - let Click handle it normally
            return super().parse_args(ctx, args)

        # Not a subcommand - everything after options is the query
        # We need to extract query words and store them, then let Click parse
        # just the options part
        query_words = []
        clean_args = []
        i = 0
        while i < len(args):
            if args[i] in option_names and i + 1 < len(args):
                clean_args.append(args[i])
                clean_args.append(args[i + 1])
                i += 2
            elif args[i].startswith("-"):
                clean_args.append(args[i])
                i += 1
            else:
                query_words.append(args[i])
                i += 1

        # Store the query for retrieval in the command callback
        ctx.ensure_object(dict)
        ctx.obj["_query_words"] = query_words

        return super().parse_args(ctx, clean_args)


@click.group(cls=NaturalOrderGroup, invoke_without_command=True)
@click.option("--model", "-m", default=None, help="Path to model checkpoint")
@click.option("--tokenizer", "-t", default=None, help="Path to tokenizer")
@click.option("--device", "-d", default="auto", help="Device (auto/cuda/cpu)")
@click.pass_context
def main(ctx, model, tokenizer, device):
    """git-nano: AI-powered git assistant.

    \b
    Usage:
      git-nano "undo last commit"     # One-shot mode
      git-nano                        # Interactive REPL
      git status | git-nano explain   # Pipe mode
    """
    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["tokenizer"] = tokenizer
    ctx.obj["device"] = device

    # If a subcommand was invoked, let it handle things
    if ctx.invoked_subcommand is not None:
        return

    # Retrieve query words stashed by NaturalOrderGroup
    query_words = ctx.obj.get("_query_words", [])
    query_str = " ".join(query_words).strip()

    # Check for piped input
    piped = _read_piped_input()

    if query_str:
        # One-shot mode with explicit query
        engine = _load_engine(model, tokenizer, device)
        if engine is None:
            sys.exit(1)

        from cli.context import GitContext
        from cli.oneshot import run_oneshot

        context = GitContext()

        # If there's also piped input, prepend it as context
        if piped:
            full_query = f"Given this output:\n{piped}\n\n{query_str}"
        else:
            full_query = query_str

        run_oneshot(full_query, engine, context)

    elif piped:
        # Pipe-only mode: explain piped input
        engine = _load_engine(model, tokenizer, device)
        if engine is None:
            sys.exit(1)

        from cli.context import GitContext
        from cli.oneshot import run_oneshot

        context = GitContext()
        run_oneshot(f"Explain this git output:\n{piped}", engine, context)

    else:
        # No query and no piped input -> interactive mode
        engine = _load_engine(model, tokenizer, device)
        if engine is None:
            sys.exit(1)

        from cli.interactive import run_interactive

        run_interactive(engine)


@main.command()
@click.pass_context
def chat(ctx):
    """Start interactive chat mode."""
    model = ctx.obj.get("model")
    tokenizer = ctx.obj.get("tokenizer")
    device = ctx.obj.get("device", "auto")

    engine = _load_engine(model, tokenizer, device)
    if engine is None:
        sys.exit(1)

    from cli.interactive import run_interactive

    run_interactive(engine)


@main.command()
@click.argument("text", nargs=-1)
@click.pass_context
def explain(ctx, text):
    """Explain piped git output or a concept.

    \b
    Examples:
      git status | git-nano explain
      git-nano explain "detached HEAD"
      git diff | git-nano explain "what changed"
    """
    model = ctx.obj.get("model")
    tokenizer = ctx.obj.get("tokenizer")
    device = ctx.obj.get("device", "auto")

    engine = _load_engine(model, tokenizer, device)
    if engine is None:
        sys.exit(1)

    from cli.context import GitContext
    from cli.oneshot import run_oneshot

    context = GitContext()
    text_str = " ".join(text).strip() if text else ""

    # Check for piped input
    piped = _read_piped_input()

    if piped and text_str:
        query = f"Explain this git output:\n{piped}\n\nAdditional context: {text_str}"
    elif piped:
        query = f"Explain this git output:\n{piped}"
    elif text_str:
        query = f"Explain: {text_str}"
    else:
        console.print("[yellow]Provide text to explain or pipe git output.[/yellow]")
        console.print("[dim]Example: git status | git-nano explain[/dim]")
        sys.exit(1)

    run_oneshot(query, engine, context)


if __name__ == "__main__":
    main()
