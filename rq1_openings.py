"""RQ1 EDA: openings and results for evenly matched players.

Research question: Among evenly matched players (rating gap < 100),
which openings give White or Black the best results, and does this
differ between Beginner (<1200) and Expert (2000+) games?

Works on the shared schema from data_loader.py, so it runs on either
dataset:

    python rq1_openings.py                                  # Lichess
    python rq1_openings.py --kaggle data/games.csv --min-games 30

Openings need at least --min-games games (default 50) in a rating group
to be ranked or tested.

Prints the RQ1 summaries and the chi-square assumption checks, and
saves the RQ1 figures to figures/.
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

import data_loader as dl
import eda_utils as eu

MAX_GAP = 100
GROUP_ORDER = ["Beginner", "Intermediate", "Advanced", "Expert"]
OUTCOME_ORDER = ["White win", "Draw", "Black win"]
DECISIVE = ["White win", "Black win"]
COMPARED_GROUPS = ["Beginner", "Expert"]
FIGURE_DIR = "figures"


def rating_group(avg_rating: float) -> str:
    """Return the rating group for an average rating.

    Beginner is below 1200, Intermediate is 1200-1599, Advanced is
    1600-1999 and Expert is 2000 or above.
    """
    if avg_rating < 1200:
        return "Beginner"
    elif avg_rating < 1600:
        return "Intermediate"
    elif avg_rating < 2000:
        return "Advanced"
    else:
        return "Expert"


def outcome_label(result: str) -> str:
    """Turn a PGN result ("1-0", "0-1", "1/2-1/2") into a readable label."""
    if result == "1-0":
        return "White win"
    elif result == "0-1":
        return "Black win"
    else:
        return "Draw"


def prepare_rq1(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the games RQ1 uses and add its columns.

    Keeps evenly matched games (rating gap below MAX_GAP) that have an
    opening family, then adds rating_group and outcome columns.
    """
    is_even = df["rating_gap"] < MAX_GAP
    has_opening = df["opening_family"].notna()
    games = df[is_even & has_opening].copy()

    games["rating_group"] = games["avg_rating"].apply(rating_group)
    games["outcome"] = games["result"].apply(outcome_label)
    return games


def opening_results(games: pd.DataFrame, min_games: int) -> pd.DataFrame:
    """Return the result shares for each rating group and opening family.

    Each row has the number of games and the share of White wins,
    draws and Black wins. Openings with fewer than min_games games in
    a rating group are dropped so rare openings do not rank high by
    chance.
    """
    counts = pd.crosstab([games["rating_group"], games["opening_family"]],
                         games["outcome"])
    counts = counts.reindex(columns=OUTCOME_ORDER, fill_value=0)

    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0)
    shares["games"] = totals

    enough_games = shares["games"] >= min_games
    return shares[enough_games]


def top_openings(results: pd.DataFrame, group: str, outcome: str,
                 n: int = 5) -> pd.DataFrame:
    """Return the n openings with the highest share of the given outcome
    (e.g. "White win") in one rating group.
    """
    group_results = results.loc[group]
    ranked = group_results.sort_values(outcome, ascending=False)
    return ranked.head(n)


def wilson_interval(successes: int, n: int,
                    z: float = 1.96) -> tuple[float, float]:
    """Return the 95% Wilson confidence interval for a proportion."""
    p = successes / n
    denominator = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    margin = margin / denominator
    return (center - margin, center + margin)


def contingency_table(games: pd.DataFrame, group: str,
                      min_games: int) -> pd.DataFrame:
    """Return the opening family x (White win, Black win) table of counts
    for one rating group, using only openings with at least min_games
    games in that group.

    Draws are left out of the table because they are rare (about 4% of
    games), which makes their expected counts too small for the
    chi-square test. The draw rate is reported separately.
    """
    group_games = games[games["rating_group"] == group]
    table = pd.crosstab(group_games["opening_family"],
                        group_games["outcome"])
    table = table.reindex(columns=OUTCOME_ORDER, fill_value=0)

    enough_games = table.sum(axis=1) >= min_games
    return table[enough_games][DECISIVE]


def expected_count_check(table: pd.DataFrame) -> dict:
    """Check the chi-square assumption that every expected count is at
    least 5. Returns the smallest expected count, the share of cells
    below 5, and whether the assumption holds.
    """
    expected = chi2_contingency(table)[3]
    cells_below_5 = (expected < 5).sum()
    share_below_5 = cells_below_5 / expected.size

    return {
        "min_expected": expected.min(),
        "share_below_5": share_below_5,
        "assumption_met": cells_below_5 == 0,
    }


def repeat_player_share(games: pd.DataFrame) -> float:
    """Return the share of games whose White player appears in more
    than one game. A high share weakens the "independent games"
    assumption of the chi-square test.
    """
    games_per_player = games["white_player"].value_counts()
    repeat_players = games_per_player[games_per_player > 1].index
    is_repeat = games["white_player"].isin(repeat_players)
    return is_repeat.mean()


def plot_games_per_group(games: pd.DataFrame, path: str) -> None:
    """Bar chart of the number of evenly matched games per rating group."""
    sns.catplot(x="rating_group", kind="count", data=games,
                order=GROUP_ORDER, color="b")
    plt.title("Evenly matched games per rating group")
    plt.xlabel("Rating group (average rating)")
    plt.ylabel("Number of games")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_top_openings(games: pd.DataFrame, path: str, n: int = 10) -> None:
    """Bar chart of the n most common opening families."""
    top = games["opening_family"].value_counts().head(n).index
    top_games = games[games["opening_family"].isin(top)]

    sns.catplot(y="opening_family", kind="count", data=top_games,
                order=top, color="b")
    plt.title(f"{n} most common opening families (evenly matched games)")
    plt.xlabel("Number of games")
    plt.ylabel("Opening family")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_outcomes_by_group(games: pd.DataFrame, path: str) -> None:
    """Bar chart of the share of White wins, draws and Black wins in
    each rating group.
    """
    shares = pd.crosstab(games["rating_group"], games["outcome"],
                         normalize="index")
    shares = shares.reindex(index=GROUP_ORDER, columns=OUTCOME_ORDER)
    shares = shares.stack().rename("share").reset_index()

    sns.catplot(x="rating_group", y="share", hue="outcome", kind="bar",
                data=shares, order=GROUP_ORDER, hue_order=OUTCOME_ORDER)
    plt.title("Game outcomes by rating group (evenly matched games)")
    plt.xlabel("Rating group (average rating)")
    plt.ylabel("Share of games")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def print_rq1_report(games: pd.DataFrame, min_games: int) -> None:
    """Print the RQ1 summaries, top openings and assumption checks."""
    summary = eu.summarize(games, ["avg_rating", "rating_gap"],
                           ["rating_group", "outcome", "opening_family"])
    eu.print_summary("RQ1 games (rating gap < %d)" % MAX_GAP, summary)

    results = opening_results(games, min_games)
    groups_present = results.index.get_level_values(0).unique()
    for group in COMPARED_GROUPS:
        if group not in groups_present:
            print("\n%s: no opening has %d+ games" % (group, min_games))
            continue
        for outcome in ["White win", "Black win"]:
            print("\nTop openings for %s in %s games:" % (outcome, group))
            print(top_openings(results, group, outcome)
                  [[outcome, "games"]])

    print("\nChi-square assumption checks (expected counts >= 5):")
    for group in COMPARED_GROUPS:
        table = contingency_table(games, group, min_games)
        if len(table) < 2:
            print("  %s: fewer than 2 openings with %d+ games"
                  % (group, min_games))
            continue
        check = expected_count_check(table)
        print("  %s: %d openings, min expected %.1f, %.0f%% of cells "
              "below 5, assumption met: %s"
              % (group, len(table), check["min_expected"],
                 check["share_below_5"] * 100, check["assumption_met"]))

    print("\nShare of games by a repeat White player: %.1f%%"
          % (repeat_player_share(games) * 100))


def main():
    """Load the data, print the RQ1 EDA and save the RQ1 figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lichess", default="data/lichess_clean.parquet",
                        help="clean Lichess parquet from eda.py")
    parser.add_argument("--kaggle", help="use the Kaggle games.csv instead")
    parser.add_argument("--min-games", type=int, default=50,
                        help="minimum games per opening and group")
    args = parser.parse_args()

    if args.kaggle:
        df = dl.load_kaggle(args.kaggle)
        name = "kaggle"
    else:
        df = pd.read_parquet(args.lichess)
        name = "lichess"

    sns.set_theme()
    games = prepare_rq1(df)
    print_rq1_report(games, args.min_games)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot_games_per_group(
        games, os.path.join(FIGURE_DIR, f"rq1_{name}_games_per_group.png"))
    plot_top_openings(
        games, os.path.join(FIGURE_DIR, f"rq1_{name}_top_openings.png"))
    plot_outcomes_by_group(
        games, os.path.join(FIGURE_DIR, f"rq1_{name}_outcomes_by_group.png"))
    print("\nFigures saved to", FIGURE_DIR)


if __name__ == "__main__":
    main()
