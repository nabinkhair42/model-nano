"""Interactive REPL chat mode for model-nano CLI."""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

from cli.context import GitContext

console = Console()


def _build_system_prompt(context: GitContext | None) -> str:
    """Build the system prompt including git context."""
    parts = [
        "You are a Git expert. Provide precise, correct git commands and explanations.",
        "Respond concisely. When a git command is appropriate, include it in a code block.",
    ]
    if context and context.is_git_repo:
        parts.append("")
        parts.append(context.prompt_context())
    return "\n".join(parts)


def _build_conversation_prompt(
    system: str, history: list[dict[str, str]], user_msg: str
) -> str:
    """Build a full conversation prompt from system, history, and new user message."""
    lines = [system, ""]
    for turn in history:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
        lines.append("")
    lines.append(f"User: {user_msg}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _print_welcome(context: GitContext | None):
    """Print welcome banner with git context."""
    welcome_lines = ["[bold cyan]git-nano[/bold cyan] interactive mode"]
    welcome_lines.append("[dim]Type your git questions. Commands: /quit, /clear, /context, /help[/dim]")

    if context and context.is_git_repo:
        welcome_lines.append("")
        welcome_lines.append(f"[green]{context.summary()}[/green]")
    elif context and not context.is_git_repo:
        welcome_lines.append("")
        welcome_lines.append("[yellow]Not in a git repository. Context features limited.[/yellow]")

    console.print(
        Panel(
            "\n".join(welcome_lines),
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def _handle_slash_command(
    cmd: str, context: GitContext, history: list[dict[str, str]]
) -> bool:
    """Handle slash commands. Returns True if the REPL should continue, False to quit."""
    cmd_lower = cmd.strip().lower()

    if cmd_lower in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        return False

    elif cmd_lower == "/clear":
        history.clear()
        console.clear()
        _print_welcome(context)
        console.print("[dim]Conversation cleared.[/dim]")
        console.print()
        return True

    elif cmd_lower == "/context":
        # Refresh and display git context
        new_context = GitContext()
        if new_context.is_git_repo:
            console.print(
                Panel(
                    new_context.prompt_context(),
                    title="[bold cyan]Git Context[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        else:
            console.print("[yellow]Not in a git repository.[/yellow]")
        console.print()
        # Update the context reference fields
        context.is_git_repo = new_context.is_git_repo
        context.branch = new_context.branch
        context.status = new_context.status
        context.recent_commits = new_context.recent_commits
        context.remotes = new_context.remotes
        context._modified_count = new_context._modified_count
        context._staged_count = new_context._staged_count
        context._untracked_count = new_context._untracked_count
        return True

    elif cmd_lower in ("/help", "/h"):
        help_text = (
            "[bold]Commands:[/bold]\n"
            "  [cyan]/quit[/cyan]    - Exit the REPL\n"
            "  [cyan]/clear[/cyan]   - Clear conversation history\n"
            "  [cyan]/context[/cyan] - Refresh and show git context\n"
            "  [cyan]/help[/cyan]    - Show this help message"
        )
        console.print(
            Panel(help_text, title="[bold cyan]Help[/bold cyan]", border_style="cyan")
        )
        console.print()
        return True

    else:
        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
        console.print("[dim]Type /help for available commands.[/dim]")
        console.print()
        return True


def run_interactive(engine):
    """Run interactive chat REPL.

    Features:
    - Persistent conversation context
    - Show git context at start
    - Commands: /quit, /clear, /context, /help
    - Rich formatting for responses
    """
    # Bail out if stdin is not a terminal
    if not sys.stdin.isatty():
        console.print("[red]Interactive mode requires a terminal (stdin is not a TTY).[/red]")
        sys.exit(1)

    # Detect git context
    context = GitContext()

    # Build system prompt
    system = _build_system_prompt(context)

    # Conversation history
    history: list[dict[str, str]] = []

    _print_welcome(context)

    while True:
        # Read user input
        try:
            user_input = console.input("[bold cyan]>>> [/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            should_continue = _handle_slash_command(user_input, context, history)
            if not should_continue:
                break
            continue

        # Build prompt with conversation history
        prompt = _build_conversation_prompt(system, history, user_input)

        # Generate response
        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            try:
                response = engine.generate(prompt, max_new_tokens=256, temperature=0.0)
            except Exception as e:
                console.print(f"[red]Engine error: {e}[/red]")
                console.print()
                continue

        # Display response
        console.print()
        try:
            # Try rendering as markdown for nice code block formatting
            md = Markdown(response)
            console.print(
                Panel(
                    md,
                    border_style="green",
                    padding=(1, 2),
                )
            )
        except Exception:
            # Fallback to plain text
            console.print(
                Panel(
                    response,
                    border_style="green",
                    padding=(1, 2),
                )
            )
        console.print()

        # Add to conversation history (keep last 10 turns to stay within context window)
        history.append({"user": user_input, "assistant": response})
        if len(history) > 10:
            history = history[-10:]
