"""
CricketIQ — Compute Ball Outcome Distributions
================================================
Aggregates phase-specific ball outcome probabilities from the deliveries
table and saves them for use by the Monte Carlo simulator.

Run:
    python scripts/compute_distributions.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.simulation.distributions import (
    compute_distributions,
    save_distributions,
    DISTRIBUTIONS_PATH,
    RUN_OUTCOMES,
)
from src.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("scripts.compute_distributions")
console = Console()


def main() -> None:
    console.print("\n[bold]CricketIQ — Ball Outcome Distributions[/bold]")
    console.print("Computing from deliveries table...\n")

    dist = compute_distributions()

    # Pretty print
    table = Table(title="Phase Distributions", show_lines=True)
    table.add_column("Phase",      style="cyan",  justify="left")
    table.add_column("N Balls",    style="bold",  justify="right")
    table.add_column("Dot %",      justify="right")
    table.add_column("1s %",       justify="right")
    table.add_column("2s %",       justify="right")
    table.add_column("4s %",       justify="right")
    table.add_column("6s %",       justify="right")
    table.add_column("Wicket %",   style="red",   justify="right")

    for phase in ["powerplay", "middle", "death"]:
        if phase not in dist:
            continue
        d     = dist[phase]
        probs = d["run_probs"]
        table.add_row(
            phase.title(),
            f"{d['n_deliveries']:,}",
            f"{probs[0]*100:.1f}%",   # 0 (dot)
            f"{probs[1]*100:.1f}%",   # 1
            f"{probs[2]*100:.1f}%",   # 2
            f"{probs[4]*100:.1f}%",   # 4
            f"{probs[5]*100:.1f}%",   # 6
            f"{d['wicket_prob']*100:.2f}%",
        )
    console.print(table)

    save_distributions(dist)
    console.print(f"\n[green]✓ Distributions saved to {DISTRIBUTIONS_PATH}[/green]")
    console.print("[dim]Load in sim: MonteCarloSimulator.load()[/dim]\n")


if __name__ == "__main__":
    main()
