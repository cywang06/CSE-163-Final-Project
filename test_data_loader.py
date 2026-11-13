"""Tests for data_loader.py using tiny hand-checked fixtures."""
import io
import os
import tempfile

import pandas as pd

import data_loader as dl


def make_game(site, white="1800", black="1750", result="1-0",
              tc="300+3", opening="Sicilian Defense: Najdorf Variation",
              termination="Normal", white_title=None):
    """Build the PGN text of one game with the tags we care about."""
    lines = [
        '[Event "Rated Blitz game"]',
        '[Site "https://lichess.org/%s"]' % site,
        '[White "w_%s"]' % site,
        '[Black "b_%s"]' % site,
        '[Result "%s"]' % result,
        '[WhiteElo "%s"]' % white,
        '[BlackElo "%s"]' % black,
    ]
    if white_title:
        lines.append('[WhiteTitle "%s"]' % white_title)
    lines += [
        '[Opening "%s"]' % opening,
        '[TimeControl "%s"]' % tc,
        '[Termination "%s"]' % termination,
        '',
        '1. e4 c5 2. Nf3 d6 1-0',
        '',
    ]
    return "\n".join(lines)


def sample_pgn():
    """Nine raw games; only four should survive cleaning."""
    games = [
        make_game("a1"),                                  # kept, blitz
        make_game("a1"),                                  # duplicate
        make_game("b2", white_title="BOT"),               # bot
        make_game("c3", white="?"),                       # no rating
        make_game("d4", termination="Abandoned"),         # abandoned
        make_game("e5", tc="60+0", opening="Italian Game"),   # bullet
        make_game("f6", tc="600+0", result="1/2-1/2"),    # rapid draw
        make_game("g7", tc="1800+0"),                     # classical
        make_game("h8", result="*"),                      # unfinished
    ]
    return "\n".join(games)


def test_parse_time_control():
    assert dl.parse_time_control("300+3") == 420
    assert dl.parse_time_control("60+0") == 60
    assert dl.parse_time_control("15+2", base_unit=60) == 980
    assert dl.parse_time_control("-") is None
    assert dl.parse_time_control("1/86400") is None
    assert dl.parse_time_control(None) is None


def test_classify_boundaries():
    assert dl.classify_time_control(179) == "bullet"
    assert dl.classify_time_control(180) == "blitz"
    assert dl.classify_time_control(479) == "blitz"
    assert dl.classify_time_control(480) == "rapid"
    assert dl.classify_time_control(1499) == "rapid"
    assert dl.classify_time_control(1500) is None
    assert dl.classify_time_control(None) is None


def test_opening_family():
    name = "Sicilian Defense: Najdorf Variation"
    assert dl.opening_family(name) == "Sicilian Defense"
    assert dl.opening_family("Italian Game") == "Italian Game"
    assert dl.opening_family("?") is None
    assert dl.opening_family(None) is None


def test_read_pgn_headers_counts_games():
    games = list(dl.read_pgn_headers(io.StringIO(sample_pgn())))
    assert len(games) == 9
    assert games[0]["WhiteElo"] == "1800"
    limited = list(dl.read_pgn_headers(io.StringIO(sample_pgn()), 3))
    assert len(limited) == 3


def test_clean_lichess():
    games = list(dl.read_pgn_headers(io.StringIO(sample_pgn())))
    df = dl.clean_lichess(pd.DataFrame(games))
    assert len(df) == 4
    assert list(df.columns) == dl.COMMON_COLUMNS
    assert df["time_control"].tolist()[:3] == ["blitz", "bullet", "rapid"]
    assert pd.isna(df["time_control"].iloc[3])   # classical game
    assert df["rating_difference"].iloc[0] == 50
    assert df["rating_gap"].iloc[0] == 50
    assert df["avg_rating"].iloc[0] == 1775
    assert df["opening_family"].iloc[1] == "Italian Game"
    assert df["result"].tolist() == ["1-0", "1-0", "1/2-1/2", "1-0"]
    assert df["white_player"].iloc[0] == "w_a1"


def test_load_kaggle():
    raw = pd.DataFrame({
        "id": ["x1", "x2", "x3", "x1"],
        "white_id": ["ann", "bo", "cy", "ann"],
        "black_id": ["dee", "ed", "fay", "dee"],
        "rated": [True, False, True, True],
        "winner": ["white", "black", "draw", "white"],
        "increment_code": ["15+2", "5+0", "1+0", "15+2"],
        "white_rating": [1500, 1600, 1400, 1500],
        "black_rating": [1450, 1650, 1400, 1450],
        "opening_name": ["Scandinavian Defense: Main Line",
                         "Italian Game", "Italian Game", "x"],
    })
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "kaggle.csv")
        raw.to_csv(path, index=False)
        df = dl.load_kaggle(path)
    assert len(df) == 2          # unrated and duplicate removed
    assert df["time_control"].tolist() == ["rapid", "bullet"]
    assert df["result"].tolist() == ["1-0", "1/2-1/2"]
    assert df["opening_family"].iloc[0] == "Scandinavian Defense"
    assert df["black_player"].tolist() == ["dee", "fay"]


def main():
    """Run every test function in this file without pytest."""
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)


if __name__ == "__main__":
    main()
