"""Print a first summary of the chess games dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """Load the games and summarize missing values and outcomes."""
    csv_path = Path(__file__).with_name("games.csv")
    games = pd.read_csv(csv_path)

    rows, columns = games.shape
    print(f"Games: {rows:,} | Columns: {columns}")
    print("\nMissing values by column:")
    print(games.isna().sum())
    print("\nGame outcomes:")
    print(games["winner"].value_counts(dropna=False))

    outcomes = games["winner"].value_counts().reindex(["white", "black", "draw"])
    outcomes.index = ["White win", "Black win", "Draw"]
    ax = outcomes.plot.bar(rot=0)
    ax.set_title("Game outcomes in the Kaggle chess dataset")
    ax.set_ylabel("Number of games")
    plt.tight_layout()
    chart_path = csv_path.with_name("game_outcomes.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nChart saved to {chart_path}")
    plot_win_rate_by_rating_gap(games, csv_path.parent)


def plot_win_rate_by_rating_gap(games: pd.DataFrame, folder: Path) -> None:
    """Plot White's win rate by rating difference, counting draws in totals."""
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
    summary = white_won.groupby(groups, observed=True).agg(["mean", "size"])
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
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
