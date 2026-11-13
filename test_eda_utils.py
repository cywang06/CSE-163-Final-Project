"""Tests for eda_utils.py using a tiny DataFrame checked by hand."""
import pandas as pd

import eda_utils as eu


def small_df():
    """Four games; one missing rating and one missing time control."""
    return pd.DataFrame({
        "white_elo": [1000.0, 1200.0, 1400.0, None],
        "time_control": ["blitz", "blitz", None, "rapid"],
    })


def test_missing_report():
    report = eu.missing_report(small_df())
    assert report.loc["white_elo", "missing"] == 1
    assert report.loc["white_elo", "percent"] == 25.0
    assert report.loc["time_control", "missing"] == 1


def test_numeric_summary():
    summary = eu.numeric_summary(small_df(), ["white_elo"])
    assert list(summary.columns) == eu.NUMERIC_STATS
    assert summary.loc["white_elo", "mean"] == 1200.0
    assert summary.loc["white_elo", "min"] == 1000.0
    assert summary.loc["white_elo", "50%"] == 1200.0
    assert summary.loc["white_elo", "max"] == 1400.0


def test_categorical_summary():
    counts = eu.categorical_summary(small_df(), "time_control")
    assert counts["blitz"] == 2
    assert counts["rapid"] == 1
    assert counts.sum() == 4        # the missing value is counted too


def test_summarize_shape():
    summary = eu.summarize(small_df(), ["white_elo"], ["time_control"])
    assert summary["shape"] == (4, 2)
    assert "time_control" in summary["categorical"]


def main():
    """Run every test function in this file without pytest."""
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)


if __name__ == "__main__":
    main()
