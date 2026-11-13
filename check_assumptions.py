"""Setup step 4: look at real rows to confirm the loader's assumptions.

Usage:
    python check_assumptions.py --kaggle data/games.csv
    python check_assumptions.py --lichess data/<month>.pgn.zst

Read the printed output and confirm:
  * Kaggle increment_code base time is in minutes (else change
    base_unit in data_loader.load_kaggle).
  * Every tag in data_loader.LICHESS_TAGS shows up for Lichess games.
"""
import argparse
from collections import Counter

import pandas as pd

import data_loader as dl

KAGGLE_NEEDED = ["id", "rated", "winner", "increment_code",
                 "white_rating", "black_rating", "opening_name",
                 "white_id", "black_id"]
SURVEY_TAGS = ["TimeControl", "Termination", "WhiteTitle", "Result"]


def missing_columns(df, needed):
    """Return the needed column names that are not in df."""
    return [column for column in needed if column not in df.columns]


def kaggle_unit_check(df):
    """Count time-control categories if the base is minutes vs seconds."""
    counts = {}
    for label, unit in [("minutes", 60), ("seconds", 1)]:
        durations = df["increment_code"].map(
            lambda tc: dl.parse_time_control(tc, base_unit=unit))
        counts[label] = durations.map(
            dl.classify_time_control).value_counts(dropna=False)
    return counts


def tag_survey(lines, max_games=1000):
    """Count every header tag, and the values of SURVEY_TAGS.

    Looks at the first max_games games only.
    """
    tags = Counter()
    values = {tag: Counter() for tag in SURVEY_TAGS}
    games = 0
    for line in lines:
        match = dl.TAG_PATTERN.match(line.strip())
        if not match:
            continue
        tag, value = match.groups()
        if tag == "Event":
            if games == max_games:
                break
            games += 1
        tags[tag] += 1
        if tag in values:
            values[tag][value] += 1
    return {"games": games, "tags": tags, "values": values}


def missing_tags(survey):
    """Return the loader's tags that never appeared in the survey."""
    return [tag for tag in dl.LICHESS_TAGS if survey["tags"][tag] == 0]


def check_kaggle(path):
    """Print what to look at in the Kaggle file."""
    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    print("Missing needed columns:", missing_columns(df, KAGGLE_NEEDED))
    shown = [c for c in ["increment_code", "created_at", "last_move_at",
                         "rated", "winner", "opening_name"]
             if c in df.columns]
    print("\nFirst rows:\n", df[shown].head(10))
    print("\nMost common increment_code:\n",
          df["increment_code"].value_counts().head(10))
    print("\nTime-control counts under each unit assumption:")
    for label, counts in kaggle_unit_check(df).items():
        print("\nbase in %s:\n%s" % (label, counts))
    if {"created_at", "last_move_at"} <= set(df.columns):
        df["seconds"] = (df["last_move_at"] - df["created_at"]) / 1000
        top = df["increment_code"].value_counts().head(8).index
        print("\nMedian real game length (s) by code, for comparison:\n",
              df[df["increment_code"].isin(top)]
              .groupby("increment_code")["seconds"].median())
    print("\nSample opening names:\n",
          df["opening_name"].drop_duplicates().head(10).tolist())


def check_lichess(path):
    """Print what to look at in the Lichess file."""
    with dl.open_pgn(path) as lines:
        survey = tag_survey(lines)
    print("Games surveyed:", survey["games"])
    print("Tags seen:", dict(survey["tags"]))
    print("Loader tags never seen:", missing_tags(survey))
    for tag, counts in survey["values"].items():
        print("\n%s (top 10):" % tag, counts.most_common(10))


def main():
    """Run the requested checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", help="path to Kaggle games.csv")
    parser.add_argument("--lichess", help="path to a Lichess .pgn[.zst]")
    args = parser.parse_args()
    if not args.kaggle and not args.lichess:
        parser.print_help()
        return
    if args.kaggle:
        check_kaggle(args.kaggle)
    if args.lichess:
        check_lichess(args.lichess)


if __name__ == "__main__":
    main()
