"""RQ2 EDA: are upsets more common in bullet, blitz, or rapid games?

An upset is a game won by the lower-rated player. We only compare
games between players within 200 rating points, split into gap bins,
so a difference between time controls is not just a difference in how
evenly matched the players were.

    python rq2_time_control.py

Reads data/games.csv and data/lichess_clean.parquet (see README) and
saves figures to figures/.
"""
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

import data_loader as dl
import eda_utils as eu

matplotlib.use("Agg")

TIME_CONTROLS = ["bullet", "blitz", "rapid"]
OUTCOMES = ["higher wins", "upset", "draw"]
GAP_BINS = [0, 50, 100, 200]
GAP_LABELS = ["1-50", "51-100", "101-200"]
MAX_GAP = 200
FIGURE_DIR = "figures"


def filter_steps(df):
    """Apply the RQ2 filters one at a time.

    Returns (filtered DataFrame, list of (step, rows left) pairs) so the
    report can say how many games each filter removed.
    """
    steps = [("all games", len(df))]
    df = df[df["time_control"].isin(TIME_CONTROLS)]
    steps.append(("bullet/blitz/rapid only", len(df)))
    df = df[df["rating_gap"] > 0]
    steps.append(("ratings not equal", len(df)))
    df = df[df["rating_gap"] <= MAX_GAP]
    steps.append(("rating gap <= %d" % MAX_GAP, len(df)))
    return df.copy(), steps


def label_outcome(row):
    """Return "higher wins", "upset" or "draw" for one game."""
    if row["result"] == "1/2-1/2":
        return "draw"
    white_won = row["result"] == "1-0"
    if white_won == (row["higher_rated_color"] == "white"):
        return "higher wins"
    return "upset"


def prepare_rq2(df):
    """Filter games and add higher_rated_color, outcome and gap_bin."""
    df, _ = filter_steps(df)
    df["higher_rated_color"] = (
        (df["white_elo"] > df["black_elo"])
        .map({True: "white", False: "black"})
    )
    df["outcome"] = df.apply(label_outcome, axis=1)
    df["gap_bin"] = pd.cut(df["rating_gap"], bins=GAP_BINS,
                           labels=GAP_LABELS)
    return df.reset_index(drop=True)


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (center - half, center + half)


def outcome_rates(df):
    """Games, outcome rates and upset-rate CI per time control and bin."""
    rows = []
    for (tc, gap), group in df.groupby(["time_control", "gap_bin"],
                                       observed=True):
        n = len(group)
        counts = group["outcome"].value_counts()
        upsets = int(counts.get("upset", 0))
        low, high = wilson_ci(upsets, n)
        rows.append({
            "time_control": tc, "gap_bin": gap, "games": n,
            "higher_win_rate": counts.get("higher wins", 0) / n,
            "upset_rate": upsets / n,
            "draw_rate": counts.get("draw", 0) / n,
            "upset_ci_low": low, "upset_ci_high": high,
        })
    return pd.DataFrame(rows)


def upset_test(df):
    """Chi-square test of upset vs not-upset across time controls.

    H0: within the given games, the upset rate is the same in every
    time control. Also checks the test's assumption that every
    expected count is at least 5.
    """
    table = pd.crosstab(df["time_control"], df["outcome"] == "upset")
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, p_value, dof, expected = chi2_contingency(table)
    return {
        "chi2": round(float(chi2), 3), "p_value": float(p_value),
        "dof": int(dof), "min_expected": round(float(expected.min()), 1),
        "expected_ok": bool((expected >= 5).all()),
    }


def tests_by_gap_bin(df):
    """Run upset_test() within each gap bin and on all bins together."""
    results = {}
    for gap in GAP_LABELS:
        results[gap] = upset_test(df[df["gap_bin"] == gap])
    results["all bins"] = upset_test(df)
    return results


def repeat_players(df):
    """Summarize how often the same player appears (independence check)."""
    players = pd.concat([df["white_player"], df["black_player"]]).dropna()
    if players.empty:
        return None
    per_player = players.value_counts()
    return {
        "games": len(df),
        "unique_players": len(per_player),
        "median_games_per_player": float(per_player.median()),
        "max_games_one_player": int(per_player.max()),
        "share_players_one_game": float((per_player == 1).mean()),
    }


def plot_games_per_time_control(df, name, folder):
    """Bar chart of the number of RQ2 games in each time control."""
    counts = df["time_control"].value_counts().reindex(
        TIME_CONTROLS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_title("%s: games per time control (gap 1-200)" % name)
    ax.set_xlabel("Time control")
    ax.set_ylabel("Number of games")
    for i, value in enumerate(counts.values):
        ax.annotate("%d" % value, (i, value), ha="center", va="bottom")
    return save(fig, folder, name, "games_per_time_control")


def plot_gap_histogram(df, name, folder):
    """Overlaid histograms of rating gap, one per time control."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for tc in TIME_CONTROLS:
        gaps = df.loc[df["time_control"] == tc, "rating_gap"]
        if gaps.empty:
            continue
        ax.hist(gaps, bins=range(0, MAX_GAP + 10, 10), density=True,
                histtype="step", linewidth=2, label=tc)
    ax.set_title("%s: rating gap distribution by time control" % name)
    ax.set_xlabel("Rating gap (points)")
    ax.set_ylabel("Share of games (density)")
    ax.legend(title="Time control")
    return save(fig, folder, name, "rating_gap_histogram")


def plot_upset_rates(rates, name, folder):
    """Grouped bars of upset rate by gap bin and time control, with CIs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for i, tc in enumerate(TIME_CONTROLS):
        part = rates[rates["time_control"] == tc].set_index("gap_bin")
        part = part.reindex(GAP_LABELS)
        x = [j + (i - 1) * width for j in range(len(GAP_LABELS))]
        errors = [part["upset_rate"] - part["upset_ci_low"],
                  part["upset_ci_high"] - part["upset_rate"]]
        ax.bar(x, part["upset_rate"] * 100, width, label=tc,
               yerr=[e * 100 for e in errors], capsize=3)
    ax.set_xticks(range(len(GAP_LABELS)))
    ax.set_xticklabels(GAP_LABELS)
    ax.set_title("%s: upset rate by rating gap and time control" % name)
    ax.set_xlabel("Rating gap bin (points)")
    ax.set_ylabel("Upset rate (%), 95% Wilson CI")
    ax.legend(title="Time control")
    return save(fig, folder, name, "upset_rate_by_gap")


def save(fig, folder, name, kind):
    """Save a figure as figures/rq2_<dataset>_<kind>.png and close it."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "rq2_%s_%s.png" % (name.lower(), kind))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_dataset(name, df, folder=FIGURE_DIR):
    """Print every RQ2 EDA result for one dataset and save its figures."""
    print("\n" + "=" * 60 + "\nRQ2: %s\n" % name + "=" * 60)
    _, steps = filter_steps(df)
    print("Rows left after each filter:")
    for step, rows in steps:
        print("  %-28s %d" % (step, rows))
    rq2 = prepare_rq2(df)
    columns = ["time_control", "rating_gap", "gap_bin",
               "higher_rated_color", "outcome", "result"]
    print("\nMissing data in RQ2 columns:\n", eu.missing_report(rq2[columns]))
    summary = eu.summarize(rq2, ["rating_gap"],
                           ["time_control", "gap_bin",
                            "higher_rated_color", "outcome"])
    eu.print_summary(name + " (RQ2 games)", summary)
    rates = outcome_rates(rq2)
    print("\nOutcome rates by time control and gap bin:\n",
          rates.round(3).to_string(index=False))
    print("\nChi-square tests (H0: equal upset rate across time "
          "controls):")
    for gap, result in tests_by_gap_bin(rq2).items():
        print("  %-9s %s" % (gap, result))
    print("\nRepeat players (independence check):", repeat_players(rq2))
    for path in (plot_games_per_time_control(rq2, name, folder),
                 plot_gap_histogram(rq2, name, folder),
                 plot_upset_rates(rates, name, folder)):
        print("Saved", path)
    return rq2, rates


def main():
    """Run the RQ2 EDA on the Kaggle and Lichess data."""
    kaggle, lichess = dl.load_both()
    run_dataset("Kaggle", kaggle)
    run_dataset("Lichess", lichess)


if __name__ == "__main__":
    main()
