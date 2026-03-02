"""Benchmark runner for model-nano evaluation suite."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from eval.metrics import command_equivalence, exact_match, response_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_command(text: str) -> str:
    """Best-effort extraction of the first git/gh command from model output.

    The model may return prose around the command.  We look for lines that
    start with ``git `` or ``gh `` (possibly inside a fenced code block).
    """
    # Try fenced code blocks first
    code_blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    for block in code_blocks:
        for line in block.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith(("git ", "gh ")):
                return stripped

    # Fallback: scan all lines for lines that start with a git/gh command
    for line in text.splitlines():
        stripped = line.strip().lstrip("$ ").strip()
        if stripped.startswith(("git ", "gh ")):
            return stripped

    # Search for a git/gh command embedded anywhere in the text
    match = re.search(r"((?:git|gh)\s[^\n]+)", text)
    if match:
        return match.group(1).strip()

    # Last resort: return the whole text trimmed
    return text.strip()


def _score_command(predicted_raw: str, expected: str | list[str]) -> float:
    """Score a command-type test case.  Returns 1.0 for pass, 0.0 for fail."""
    predicted = _extract_command(predicted_raw)

    # 1. Exact match (handles list of alternatives)
    if exact_match(predicted, expected):
        return 1.0

    # 2. Semantic equivalence against each alternative
    alternatives = expected if isinstance(expected, list) else [expected]
    for alt in alternatives:
        if command_equivalence(predicted, alt):
            return 1.0

    return 0.0


def _score_explanation(predicted_raw: str, expected: str) -> float:
    """Score an explanation-type test case.  Returns 0.0-1.0."""
    return response_quality(predicted_raw, expected)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class Benchmark:
    """Load test cases, run inference, and produce a scored report."""

    def __init__(self, engine, test_cases_path: str = "eval/test_cases.json"):
        """Load test cases and store the inference engine reference.

        Args:
            engine: An ``InferenceEngine`` instance with a ``.generate()`` method.
            test_cases_path: Path to the JSON file with test cases.
        """
        self.engine = engine
        path = Path(test_cases_path)
        if not path.exists():
            raise FileNotFoundError(f"Test cases file not found: {path}")
        with open(path) as f:
            self.test_cases: list[dict] = json.load(f)

    # ------------------------------------------------------------------ run

    def run(self, verbose: bool = False) -> dict:
        """Run all test cases and return structured results.

        Returns:
            {
                "categories": {
                    "<category>": {
                        "total": int,
                        "passed": int,
                        "scores": [float, ...],
                        "details": [
                            {"id": int, "query": str, "predicted": str,
                             "expected": ..., "score": float},
                            ...
                        ],
                    },
                    ...
                },
                "overall": {"total": int, "passed": int, "accuracy": float},
                "elapsed_seconds": float,
            }
        """
        console = Console()
        categories: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "passed": 0, "scores": [], "details": []}
        )

        start = time.perf_counter()

        for i, case in enumerate(self.test_cases):
            case_id = case["id"]
            category = case["category"]
            query = case["query"]
            expected = case["expected"]
            case_type = case["type"]

            # Generate model response
            prompt = self.engine.format_prompt(
                query,
                system_prompt="You are a Git expert. Provide precise, correct git commands and explanations.",
            )
            predicted = self.engine.generate(
                prompt, max_new_tokens=256, temperature=0.0
            )

            # Score
            if case_type == "command":
                score = _score_command(predicted, expected)
            else:
                score = _score_explanation(predicted, expected)

            passed = score >= 0.5  # threshold for "pass"

            cat = categories[category]
            cat["total"] += 1
            if passed:
                cat["passed"] += 1
            cat["scores"].append(score)
            cat["details"].append(
                {
                    "id": case_id,
                    "query": query,
                    "predicted": predicted,
                    "expected": expected,
                    "score": score,
                }
            )

            if verbose:
                status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                console.print(
                    f"  [{i + 1:>3}/{len(self.test_cases)}] {status}  "
                    f"(score={score:.2f})  {query[:60]}"
                )

        elapsed = time.perf_counter() - start

        total = sum(c["total"] for c in categories.values())
        total_passed = sum(c["passed"] for c in categories.values())

        return {
            "categories": dict(categories),
            "overall": {
                "total": total,
                "passed": total_passed,
                "accuracy": total_passed / total if total else 0.0,
            },
            "elapsed_seconds": round(elapsed, 2),
        }

    # ------------------------------------------------------------- report

    def print_report(self, results: dict) -> None:
        """Print a formatted benchmark report using Rich tables.

        Args:
            results: The dict returned by :meth:`run`.
        """
        console = Console()
        console.print()

        table = Table(
            title="model-nano Evaluation Report",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            min_width=50,
        )
        table.add_column("Category", style="bold", min_width=18)
        table.add_column("Accuracy", justify="right", min_width=10)
        table.add_column("Cases", justify="right", min_width=8)

        # Sort categories in a canonical order for readability
        category_order = [
            "basic",
            "branching",
            "history",
            "remote",
            "stash",
            "config",
            "error_recovery",
            "github_cli",
        ]
        seen = set()
        ordered_cats = []
        for cat in category_order:
            if cat in results["categories"]:
                ordered_cats.append(cat)
                seen.add(cat)
        for cat in sorted(results["categories"]):
            if cat not in seen:
                ordered_cats.append(cat)

        for cat in ordered_cats:
            data = results["categories"][cat]
            pct = (data["passed"] / data["total"] * 100) if data["total"] else 0.0
            if pct >= 80:
                pct_style = "green"
            elif pct >= 50:
                pct_style = "yellow"
            else:
                pct_style = "red"
            table.add_row(
                cat,
                f"[{pct_style}]{pct:5.1f}%[/{pct_style}]",
                f"{data['passed']}/{data['total']}",
            )

        # Overall row
        overall = results["overall"]
        overall_pct = overall["accuracy"] * 100
        if overall_pct >= 80:
            ov_style = "green"
        elif overall_pct >= 50:
            ov_style = "yellow"
        else:
            ov_style = "red"
        table.add_section()
        table.add_row(
            "[bold]OVERALL[/bold]",
            f"[bold {ov_style}]{overall_pct:5.1f}%[/bold {ov_style}]",
            f"[bold]{overall['passed']}/{overall['total']}[/bold]",
        )

        console.print(table)
        console.print(f"\n  Elapsed: {results['elapsed_seconds']:.1f}s\n")

    # ---------------------------------------------------- failure details

    def print_failures(self, results: dict) -> None:
        """Print details about each failed test case for debugging.

        Args:
            results: The dict returned by :meth:`run`.
        """
        console = Console()
        failures = []
        for cat, data in results["categories"].items():
            for detail in data["details"]:
                if detail["score"] < 0.5:
                    failures.append((cat, detail))

        if not failures:
            console.print("[green]All test cases passed![/green]\n")
            return

        console.print(f"\n[bold red]Failed cases ({len(failures)}):[/bold red]\n")
        for cat, detail in failures:
            console.print(f"  [bold]#{detail['id']}[/bold] [{cat}] {detail['query']}")
            console.print(f"    Expected : {detail['expected']}")
            predicted_short = detail["predicted"][:120].replace("\n", " ")
            console.print(f"    Predicted: {predicted_short}")
            console.print(f"    Score    : {detail['score']:.2f}")
            console.print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the benchmark from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the model-nano evaluation benchmark."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint (e.g. checkpoints/model.pt).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/tokenizer.json",
        help="Path to the tokenizer file (default: tokenizer/tokenizer.json).",
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default="eval/test_cases.json",
        help="Path to the test cases JSON file (default: eval/test_cases.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case pass/fail during the run.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print details about each failed test case after the report.",
    )
    args = parser.parse_args()

    # Import here so the module can be imported without torch installed
    from inference.engine import InferenceEngine

    console = Console()
    console.print(f"\n[bold]Loading model:[/bold] {args.model}")
    console.print(f"[bold]Tokenizer:    [/bold] {args.tokenizer}")
    console.print(f"[bold]Device:       [/bold] {args.device}")
    console.print(f"[bold]Test cases:   [/bold] {args.test_cases}\n")

    engine = InferenceEngine(args.model, args.tokenizer, args.device)

    bench = Benchmark(engine, test_cases_path=args.test_cases)
    results = bench.run(verbose=args.verbose)
    bench.print_report(results)

    if args.show_failures:
        bench.print_failures(results)


if __name__ == "__main__":
    main()
