"""Tests for rq1_openings.py using tiny DataFrames checked by hand."""
import pandas as pd

import rq1_openings as rq1


def small_games():
    """Six games in the shared schema, worked out by hand.

    Two have a rating gap of 100 or more and one has no opening, so
    prepare_rq1 should keep only the first three.
    """
    return pd.DataFrame({
        "white_elo": [1000, 1150, 2100, 1500, 1000, 1800],
        "black_elo": [1050, 1100, 2050, 1700, 1100, 1800],
        "result": ["1-0", "0-1", "1/2-1/2", "1-0", "1-0", "0-1"],
        "opening_family": ["Sicilian Defense", "Sicilian Defense",
                           "French Defense", "Italian Game",
                           "Sicilian Defense", None],
        "avg_rating": [1025.0, 1125.0, 2075.0, 1600.0, 1050.0, 1800.0],
        "rating_gap": [50, 50, 50, 200, 100, 0],
        "white_player": ["a", "b", "a", "c", "d", "e"],
    })


def test_rating_group_boundaries():
    assert rq1.rating_group(1199) == "Beginner"
    assert rq1.rating_group(1200) == "Intermediate"
    assert rq1.rating_group(1599) == "Intermediate"
    assert rq1.rating_group(1600) == "Advanced"
    assert rq1.rating_group(1999.5) == "Advanced"
    assert rq1.rating_group(2000) == "Expert"


def test_outcome_label():
    assert rq1.outcome_label("1-0") == "White win"
    assert rq1.outcome_label("0-1") == "Black win"
    assert rq1.outcome_label("1/2-1/2") == "Draw"


def test_prepare_rq1_filters():
    games = rq1.prepare_rq1(small_games())
    # gap 200 and gap exactly 100 are dropped; the missing opening too
    assert len(games) == 3
    assert list(games["rating_group"]) == ["Beginner", "Beginner",
                                           "Expert"]
    assert list(games["outcome"]) == ["White win", "Black win", "Draw"]


def test_opening_results_shares_and_min_games():
    games = rq1.prepare_rq1(small_games())
    results = rq1.opening_results(games, min_games=1)
    sicilian = results.loc[("Beginner", "Sicilian Defense")]
    assert sicilian["games"] == 2
    assert sicilian["White win"] == 0.5
    assert sicilian["Black win"] == 0.5
    assert sicilian["Draw"] == 0.0

    # With min_games=2, the French Defense (1 game) is dropped
    results = rq1.opening_results(games, min_games=2)
    assert len(results) == 1


def test_top_openings():
    results = pd.DataFrame({
        "rating_group": ["Expert", "Expert", "Expert"],
        "opening_family": ["A", "B", "C"],
        "White win": [0.4, 0.6, 0.5],
        "games": [10, 10, 10],
    }).set_index(["rating_group", "opening_family"])
    top = rq1.top_openings(results, "Expert", "White win", n=2)
    assert list(top.index) == ["B", "C"]


def test_wilson_interval():
    # Hand-computed: 50 successes out of 100 gives about (0.404, 0.596)
    low, high = rq1.wilson_interval(50, 100)
    assert abs(low - 0.4038) < 0.001
    assert abs(high - 0.5962) < 0.001
    # With 0 successes the interval starts at 0 (up to rounding error)
    low, high = rq1.wilson_interval(0, 10)
    assert abs(low) < 1e-9
    assert 0 < high < 1


def test_expected_count_check():
    big = pd.DataFrame({"White win": [50, 40], "Draw": [10, 10],
                        "Black win": [40, 50]}, index=["A", "B"])
    assert rq1.expected_count_check(big)["assumption_met"]

    small = pd.DataFrame({"White win": [3, 2], "Draw": [1, 0],
                          "Black win": [2, 3]}, index=["A", "B"])
    check = rq1.expected_count_check(small)
    assert not check["assumption_met"]
    assert check["min_expected"] < 5


def test_contingency_table_min_games():
    games = rq1.prepare_rq1(small_games())
    table = rq1.contingency_table(games, "Beginner", min_games=1)
    # Draws are left out of the chi-square table
    assert list(table.columns) == ["White win", "Black win"]
    assert table.loc["Sicilian Defense", "White win"] == 1
    assert table.loc["Sicilian Defense", "Black win"] == 1


def test_repeat_player_share():
    games = rq1.prepare_rq1(small_games())
    # Player "a" plays White in 2 of the 3 kept games
    assert abs(rq1.repeat_player_share(games) - 2 / 3) < 1e-9


def main():
    """Run every test function in this file without pytest."""
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)


if __name__ == "__main__":
    main()
