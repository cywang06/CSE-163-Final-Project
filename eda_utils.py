"""Shared EDA helpers used by every research question.

summarize() answers the standard EDA questions for one dataset: its
size, missing data, a seven-number summary for numeric columns, and
value counts for categorical columns.
"""
import pandas as pd

NUMERIC_STATS = ["mean", "std", "min", "25%", "50%", "75%", "max"]


def missing_report(df):
    """Return a DataFrame with the count and percent missing per column."""
    counts = df.isna().sum()
    return pd.DataFrame({
        "missing": counts,
        "percent": (counts / len(df) * 100).round(2),
    })


def numeric_summary(df, columns):
    """Seven-number summary (mean, std, min, Q1, median, Q3, max)."""
    return df[columns].describe().loc[NUMERIC_STATS].T


def categorical_summary(df, column):
    """Unique values of a column with their counts (missing included)."""
    return df[column].value_counts(dropna=False)


def summarize(df, numeric_columns, categorical_columns):
    """Collect all standard EDA summaries for one dataset in a dict."""
    return {
        "shape": df.shape,
        "missing": missing_report(df),
        "numeric": numeric_summary(df, numeric_columns),
        "categorical": {
            column: categorical_summary(df, column)
            for column in categorical_columns
        },
    }


def print_summary(name, summary):
    """Print the output of summarize() in a readable form."""
    rows, columns = summary["shape"]
    print("=== %s: %d rows x %d columns ===" % (name, rows, columns))
    print("\nMissing data:\n", summary["missing"])
    print("\nNumeric summary:\n", summary["numeric"])
    for column, counts in summary["categorical"].items():
        print("\nValue counts for %s:\n" % column, counts.head(15))
