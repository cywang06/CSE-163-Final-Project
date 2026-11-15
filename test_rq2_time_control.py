"""Tests for rq2_time_control.py using small hand-checked DataFrames."""
import os
import tempfile

import pandas as pd

import data_loader as dl
import rq2_time_control as rq2


def games(rows):
    """Build a DataFrame in the shared schema from (w, b, result, tc)."""
    df = pd.DataFrame(rows, columns=["white_elo", "black_elo", "result",
                                     "time_control"])
    df["opening_family"] = "Italian Game"
    df["white_player"] = ["w%d" % i for i in range(len(df))]
    df["black_player"] = ["b%d" % i for i in range(len(df))]
    return dl.add_rating_columns(df)


def small():
    """Eight games; four survive the RQ2 filters."""
    return games([
        (1550, 1500, "1-0", "blitz"),        # higher (white) wins, gap 50
        (1500, 1551, "1-0", "blitz"),        # upset, gap 51
        (1700, 1500, "1/2-1/2", "bullet"),   # draw, gap 200
        (1500, 1600, "1-0", "rapid"),        # upset, gap 100
        (1500, 1500, "1-0", "blitz"),        # equal ratings: dropped
        (1500, 1701, "0-1", "blitz"),        # gap 201: dropped
        (1500, 1400, "1-0", None),           # classical: dropped
        (1500, 1450, "0-1", "classical"),    # not a valid tc: dropped
    ])


def test_filter_steps():
    df, steps = rq2.filter_steps(small())
    assert [rows for _, rows in steps] == [8, 6, 5, 4]
    assert len(df) == 4


def test_prepare_rq2_labels():
    df = rq2.prepare_rq2(small())
    assert df["higher_rated_color"].tolist() == [
        "white", "black", "white", "black"]
    assert df["outcome"].tolist() == [
        "higher wins", "upset", "draw", "upset"]


def test_gap_bin_boundaries():
    df = rq2.prepare_rq2(small())
    assert df["gap_bin"].astype(str).tolist() == [
        "1-50", "51-100", "101-200", "51-100"]


def test_wilson_ci():
    low, high = rq2.wilson_ci(5, 10)
    assert abs(low - 0.2366) < 1e-3
    assert abs(high - 0.7634) < 1e-3
    low, high = rq2.wilson_ci(0, 20)
    assert low == 0 or abs(low) < 1e-12
    assert 0 < high < 0.2
    assert all(pd.isna(v) for v in rq2.wilson_ci(0, 0))


def test_outcome_rates():
    rates = rq2.outcome_rates(rq2.prepare_rq2(small()))
    blitz = rates[(rates["time_control"] == "blitz")
                  & (rates["gap_bin"] == "51-100")].iloc[0]
    assert blitz["games"] == 1
    assert blitz["upset_rate"] == 1.0
    assert rates["games"].sum() == 4


def big_table(bullet_upsets, blitz_upsets, n=100):
    """n games each of bullet and blitz with the given upset counts."""
    rows = []
    for tc, upsets in [("bullet", bullet_upsets), ("blitz", blitz_upsets)]:
        rows += [(1500, 1560, "1-0", tc)] * upsets
        rows += [(1560, 1500, "1-0", tc)] * (n - upsets)
    return rq2.prepare_rq2(games(rows))


def test_upset_test_detects_difference():
    result = rq2.upset_test(big_table(50, 10))
    assert result["p_value"] < 0.001
    assert result["expected_ok"]
    same = rq2.upset_test(big_table(30, 30))
    assert same["p_value"] > 0.5


def test_upset_test_expected_count_check():
    result = rq2.upset_test(big_table(1, 0, n=6))
    assert not result["expected_ok"]
    assert result["min_expected"] < 5


def test_upset_test_needs_two_groups():
    one_tc = games([(1550, 1500, "1-0", "blitz"),
                    (1500, 1550, "1-0", "blitz")])
    assert rq2.upset_test(rq2.prepare_rq2(one_tc)) is None


def test_drop_small_time_controls():
    df = big_table(10, 10, n=40)
    extra = rq2.prepare_rq2(games([(1550, 1500, "1-0", "rapid")] * 2))
    kept, small = rq2.drop_small_time_controls(pd.concat([df, extra]))
    assert small == ["rapid"]
    assert set(kept["time_control"]) == {"bullet", "blitz"}
    assert len(kept) == 80


def test_repeat_players():
    df = games([(1550, 1500, "1-0", "blitz")] * 3)
    df["white_player"] = ["ann", "ann", "bo"]
    df["black_player"] = ["cy", "dee", "ann"]
    summary = rq2.repeat_players(df)
    assert summary["unique_players"] == 4
    assert summary["max_games_one_player"] == 3
    assert summary["share_players_one_game"] == 0.75


def test_figures_are_saved():
    df = big_table(30, 20)
    rates = rq2.outcome_rates(df)
    with tempfile.TemporaryDirectory() as folder:
        paths = [rq2.plot_games_per_time_control(df, "Test", folder),
                 rq2.plot_gap_histogram(df, "Test", folder),
                 rq2.plot_upset_rates(rates, "Test", folder)]
        assert all(os.path.exists(path) for path in paths)
        assert os.path.basename(paths[2]) == "rq2_test_upset_rate_by_gap.png"


def main():
    """Run every test function in this file without pytest."""
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("passed:", test.__name__)


if __name__ == "__main__":
    main()
