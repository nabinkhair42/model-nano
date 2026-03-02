"""One-shot query mode for model-nano CLI."""

import re
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt

from cli.context import GitContext

console = Console()


def _extract_git_command(response: str) -> str | None:
    """Extract a git command from the model response.

    Looks for lines starting with 'git ' or fenced code blocks containing git commands.
    Returns the first git command found, or None.
    """
    # Try fenced code block first: ```\ngit ...\n```
    fenced = re.findall(r"```(?:bash|sh|shell)?\s*\n(git\s+[^\n]+)\n```", response)
    if fenced:
        return fenced[0].strip()

    # Try inline code: `git ...`
    inline = re.findall(r"`(git\s+[^`]+)`", response)
    if inline:
        return inline[0].strip()

    # Try bare lines starting with 'git '
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.startswith("git ") and len(stripped) < 200:
            return stripped

    return None


def _extract_explanation(response: str, command: str | None) -> str:
    """Extract the explanation portion of the response (everything except the command)."""
    if command is None:
        return response.strip()

    # Remove the command line and surrounding code fences from the explanation
    explanation = response
    # Remove fenced code blocks containing the command
    explanation = re.sub(
        r"```(?:bash|sh|shell)?\s*\n" + re.escape(command) + r"\s*\n```",
        "",
        explanation,
    )
    # Remove inline code containing the command
    explanation = explanation.replace(f"`{command}`", "")
    # Remove bare command line
    explanation = explanation.replace(command, "")
    # Clean up whitespace
    explanation = re.sub(r"\n{3,}", "\n\n", explanation).strip()
    return explanation


def _execute_command(command: str) -> None:
    """Execute a git command and display its output."""
    console.print()
    console.print(f"[dim]$ {command}[/dim]")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout:
            console.print(result.stdout, end="")
        if result.stderr:
            console.print(f"[yellow]{result.stderr}[/yellow]", end="")
        if result.returncode != 0:
            console.print(f"[red]Exit code: {result.returncode}[/red]")
    except subprocess.TimeoutExpired:
        console.print("[red]Command timed out (30s limit)[/red]")


def _copy_to_clipboard(text: str) -> bool:
    """Attempt to copy text to system clipboard. Returns True on success."""
    # Try xclip, xsel, pbcopy (macOS), or clip.exe (WSL) in order
    clipboard_commands = [
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["pbcopy"],
        ["clip.exe"],
    ]
    for cmd in clipboard_commands:
        try:
            proc = subprocess.run(
                cmd, input=text, text=True, capture_output=True, timeout=5
            )
            if proc.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _build_prompt(query: str, context: GitContext | None) -> str:
    """Build the full prompt including system context and user query."""
    parts = [
        "You are a Git expert. Provide precise, correct git commands and explanations.",
        "Respond concisely. If a git command is appropriate, include it.",
    ]
    if context and context.is_git_repo:
        parts.append("")
        parts.append(context.prompt_context())

    parts.append("")
    parts.append(f"User: {query}")
    parts.append("Assistant:")
    return "\n".join(parts)


def run_oneshot(query: str, engine, context: GitContext = None):
    """Run a single query and display result with suggest-then-confirm UX.

    Display format:
    +-------------------------------------+
    | git reset --soft HEAD~3             |
    |                                     |
    | Resets last 3 commits, keeps staged |
    +-------------------------------------+
    [Enter] Execute  [e] Edit  [c] Copy  [q] Cancel

    If command is destructive, show warning.
    """
    is_piped = not sys.stdin.isatty()

    # Generate response from the engine
    prompt = _build_prompt(query, context)

    with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
        try:
            response = engine.generate(prompt, max_new_tokens=256, temperature=0.0)
        except Exception as e:
            console.print(f"[red]Engine error: {e}[/red]")
            return

    # Extract command and explanation
    command = _extract_git_command(response)
    explanation = _extract_explanation(response, command)

    # Build panel content
    panel_parts = []
    if command:
        panel_parts.append(f"[bold green]{command}[/bold green]")
    if explanation:
        if command:
            panel_parts.append("")
        panel_parts.append(f"[white]{explanation}[/white]")

    panel_content = "\n".join(panel_parts) if panel_parts else response

    # For piped input, just output the result without interactive prompts
    if is_piped:
        if command:
            print(command)
        if explanation:
            print(explanation)
        return

    # Display the panel
    console.print()
    console.print(
        Panel(
            panel_content,
            title="[bold cyan]git-nano[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # If no executable command found, we're done
    if not command:
        return

    # Check for destructive commands
    ctx = context or GitContext()
    if ctx.is_destructive(command):
        console.print(
            Panel(
                f"[bold red]WARNING: This is a destructive operation![/bold red]\n"
                f"[yellow]{command}[/yellow]\n\n"
                f"This command may cause irreversible data loss.",
                border_style="red",
                title="[bold red]Destructive Command[/bold red]",
            )
        )

    # Interactive prompt
    console.print(
        "[dim]  [Enter] Execute  [e] Edit  [c] Copy  [q] Cancel[/dim]"
    )

    try:
        choice = Prompt.ask(
            "[bold]Action[/bold]",
            choices=["", "e", "c", "q"],
            default="",
            show_choices=False,
            show_default=False,
        )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Cancelled.[/dim]")
        return

    if choice == "" or choice is None:
        # Execute
        if ctx.is_destructive(command):
            confirm = Prompt.ask(
                "[bold red]Are you sure? Type 'yes' to confirm[/bold red]",
                default="no",
            )
            if confirm.lower() != "yes":
                console.print("[dim]Cancelled.[/dim]")
                return
        _execute_command(command)

    elif choice == "e":
        # Edit the command before executing
        try:
            edited = Prompt.ask("[bold]Edit command[/bold]", default=command)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Cancelled.[/dim]")
            return

        if edited.strip():
            edited = edited.strip()
            if ctx.is_destructive(edited):
                console.print(
                    "[bold red]WARNING: Edited command is destructive![/bold red]"
                )
                confirm = Prompt.ask(
                    "[bold red]Are you sure? Type 'yes' to confirm[/bold red]",
                    default="no",
                )
                if confirm.lower() != "yes":
                    console.print("[dim]Cancelled.[/dim]")
                    return
            _execute_command(edited)
        else:
            console.print("[dim]Empty command, cancelled.[/dim]")

    elif choice == "c":
        # Copy to clipboard
        if _copy_to_clipboard(command):
            console.print("[green]Copied to clipboard![/green]")
        else:
            console.print(
                f"[yellow]Clipboard not available. Command:[/yellow] {command}"
            )

    elif choice == "q":
        console.print("[dim]Cancelled.[/dim]")
