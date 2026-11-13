"""Kaggle-only figures: game outcomes and White's win rate by rating gap.

Originally the team's first eda.py; renamed so the shared eda.py can
summarize both datasets.

    python kaggle_plots.py                  # reads data/games.csv
    python kaggle_plots.py path/to/games.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIGURE_DIR = Path("figures")


def plot_outcomes(games: pd.DataFrame, folder: Path) -> Path:
    """Bar chart of White wins, Black wins and draws."""
    outcomes = games["winner"].value_counts().reindex(
        ["white", "black", "draw"])
    outcomes.index = ["White win", "Black win", "Draw"]
    ax = outcomes.plot.bar(rot=0)
    ax.set_title("Game outcomes in the Kaggle chess dataset")
    ax.set_ylabel("Number of games")
    plt.tight_layout()
    chart_path = folder / "game_outcomes.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path


def win_rate_by_rating_gap(games: pd.DataFrame) -> pd.DataFrame:
    """White's win rate and game count per rating-difference bin.

    Draws count in the totals, so the rate is wins / all games.
    """
    rating_gap = games["white_rating"] - games["black_rating"]
    groups = pd.cut(
        rating_gap,
        bins=[-float("inf"), -400, -200, -100, 100, 200, 400, float("inf")],
        labels=[
            "≤−401", "−400 to −201", "−200 to −101", "−100 to +99",
            "+100 to +199", "+200 to +399", "≥+400",
        ],
        right=False,
    )
    white_won = games["winner"].eq("white")
    return white_won.groupby(groups, observed=True).agg(["mean", "size"])


def plot_win_rate_by_rating_gap(games: pd.DataFrame, folder: Path) -> Path:
    """Bar chart of White's win rate by rating difference."""
    summary = win_rate_by_rating_gap(games)
    ax = (summary["mean"] * 100).plot.bar(rot=25, figsize=(9, 5))
    ax.set_title("White's win rate by rating difference")
    ax.set_xlabel("White's rating minus Black's rating")
    ax.set_ylabel("Games won by White (%)")
    ax.set_ylim(0, 100)
    ax.bar_label(
        ax.containers[0],
        labels=[f"n={count:,}" for count in summary["size"]],
        padding=3,
    )
    plt.tight_layout()
    chart_path = folder / "white_win_rate_by_rating_gap.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path


def main() -> None:
    """Load the Kaggle games and save both charts to figures/."""
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/games.csv")
    games = pd.read_csv(csv_path)
    rows, columns = games.shape
    print(f"Games: {rows:,} | Columns: {columns}")
    print("\nMissing values by column:")
    print(games.isna().sum())
    print("\nGame outcomes:")
    print(games["winner"].value_counts(dropna=False))
    FIGURE_DIR.mkdir(exist_ok=True)
    for path in (plot_outcomes(games, FIGURE_DIR),
                 plot_win_rate_by_rating_gap(games, FIGURE_DIR)):
        print(f"Chart saved to {path}")


if __name__ == "__main__":
    main()
