"""Shared EDA entry point: load, summarize and sanity-check the data.

Either dataset can be given on its own, so you can run it as soon as
one file has been downloaded:

    python eda.py --kaggle data/games.csv
    python eda.py --lichess data/<month>.pgn.zst --max-games 200000

The cleaned Lichess sample is saved to data/lichess_clean.parquet so
teammates can load it without re-parsing the big file.
"""
import argparse
import os

import pandas as pd

import data_loader as dl
import eda_utils as eu

NUMERIC = ["white_elo", "black_elo", "avg_rating", "rating_gap",
           "rating_difference"]
CATEGORICAL = ["result", "time_control", "opening_family"]


def drop_rate(raw_count, clean_count):
    """Return the fraction of raw games removed by cleaning."""
    if raw_count == 0:
        return 0.0
    return 1 - clean_count / raw_count


def sanity_checks(df):
    """Return a list of problems found in a cleaned DataFrame.

    An empty list means every check passed.
    """
    if df.empty:
        return ["no rows"]
    problems = []
    if list(df.columns) != dl.COMMON_COLUMNS:
        problems.append("columns differ from the common schema")
    if not df["result"].isin(dl.VALID_RESULTS).all():
        problems.append("invalid result values")
    ratings = pd.concat([df["white_elo"], df["black_elo"]])
    if ratings.min() < 100 or ratings.max() > 4000:
        problems.append("ratings outside the range 100-4000")
    if (df["rating_gap"] < 0).any():
        problems.append("negative rating gap")
    if df["time_control"].notna().sum() == 0:
        problems.append("no bullet/blitz/rapid games found")
    return problems


def report(name, df):
    """Print the standard summaries and sanity checks for one dataset."""
    summary = eu.summarize(df, NUMERIC, CATEGORICAL)
    eu.print_summary(name, summary)
    problems = sanity_checks(df)
    print("\nSanity checks:", "all passed" if not problems else problems)


def run_kaggle(path):
    """Load, report on and return the clean Kaggle data."""
    df = dl.load_kaggle(path)
    report("Kaggle", df)
    return df


def run_lichess(path, max_games, out_dir):
    """Parse, report on, save and return the clean Lichess sample."""
    raw = dl.load_lichess_raw(path, max_games)
    df = dl.clean_lichess(raw)
    rate = drop_rate(len(raw), len(df))
    print("Lichess: kept %d of %d games (%.1f%% dropped)"
          % (len(df), len(raw), rate * 100))
    report("Lichess sample", df)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lichess_clean.parquet")
    df.to_parquet(out_path)
    print("Saved", out_path)
    return df


def main():
    """Parse command-line arguments and run the requested datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", help="path to Kaggle games.csv")
    parser.add_argument("--lichess", help="path to a Lichess .pgn[.zst]")
    parser.add_argument("--max-games", type=int, default=200000,
                        help="raw Lichess games to read (default 200000)")
    parser.add_argument("--out-dir", default="data",
                        help="where to save the clean parquet file")
    args = parser.parse_args()
    if not args.kaggle and not args.lichess:
        parser.print_help()
        return
    if args.kaggle:
        run_kaggle(args.kaggle)
    if args.lichess:
        run_lichess(args.lichess, args.max_games, args.out_dir)


if __name__ == "__main__":
    main()
