"""Tests for eda.py, check_assumptions.py and data_loader.load_both."""
import io
import os
import tempfile

import pandas as pd

import check_assumptions as ca
import data_loader as dl
import eda
import test_data_loader as tdl


def clean_sample():
    """The four cleaned games from the shared PGN fixture."""
    games = list(dl.read_pgn_headers(io.StringIO(tdl.sample_pgn())))
    return dl.clean_lichess(pd.DataFrame(games))


def test_drop_rate():
    assert abs(eda.drop_rate(10, 9) - 0.1) < 1e-9
    assert eda.drop_rate(4, 4) == 0.0
    assert eda.drop_rate(0, 0) == 0.0


def test_sanity_checks_pass_on_clean_data():
    assert eda.sanity_checks(clean_sample()) == []


def test_sanity_checks_catch_problems():
    df = clean_sample()
    bad = df.copy()
    bad.loc[0, "white_elo"] = 9000
    assert "ratings outside the range 100-4000" in eda.sanity_checks(bad)
    bad = df.copy()
    bad.loc[0, "result"] = "2-0"
    assert "invalid result values" in eda.sanity_checks(bad)
    assert eda.sanity_checks(df.iloc[0:0]) == ["no rows"]
    bad = df.drop(columns=["avg_rating"])
    assert eda.sanity_checks(bad) == [
        "columns differ from the common schema"]


def test_missing_columns():
    df = pd.DataFrame({"id": [1], "rated": [True]})
    assert ca.missing_columns(df, ["id", "winner"]) == ["winner"]
    assert ca.missing_columns(df, ["id", "rated"]) == []


def test_kaggle_unit_check():
    df = pd.DataFrame({"increment_code": ["15+2", "1+0"]})
    counts = ca.kaggle_unit_check(df)
    assert counts["minutes"]["rapid"] == 1      # 900 + 80 = 980
    assert counts["minutes"]["bullet"] == 1     # 60
    assert "rapid" not in counts["seconds"].index   # 15 + 80 = 95


def test_tag_survey():
    survey = ca.tag_survey(io.StringIO(tdl.sample_pgn()))
    assert survey["games"] == 9
    assert survey["tags"]["Event"] == 9
    assert survey["values"]["WhiteTitle"]["BOT"] == 1
    assert survey["values"]["TimeControl"]["300+3"] == 6
    assert ca.missing_tags(survey) == ["BlackTitle", "ECO"]
    short = ca.tag_survey(io.StringIO(tdl.sample_pgn()), max_games=3)
    assert short["games"] == 3
    assert short["tags"]["Event"] == 3


def test_load_both_round_trip():
    raw = pd.DataFrame({
        "id": ["x1"], "rated": [True], "winner": ["white"],
        "white_id": ["ann"], "black_id": ["bo"],
        "increment_code": ["10+0"], "white_rating": [1500],
        "black_rating": [1400], "opening_name": ["Italian Game"],
    })
    with tempfile.TemporaryDirectory() as folder:
        csv_path = os.path.join(folder, "games.csv")
        parquet_path = os.path.join(folder, "lichess.parquet")
        raw.to_csv(csv_path, index=False)
        clean_sample().to_parquet(parquet_path)
        kaggle, lichess = dl.load_both(csv_path, parquet_path)
    assert list(kaggle.columns) == list(lichess.columns)
    assert len(kaggle) == 1
    assert len(lichess) == 4


def main():
    """Run every test function in this file without pytest."""
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("passed:", test.__name__)


if __name__ == "__main__":
    main()
