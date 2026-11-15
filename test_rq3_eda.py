"""Check RQ3 outputs using a small, hand-checked Kaggle sample."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib
import pandas as pd

import kaggle_plots
import rq3_eda


def test_rq3_outputs() -> None:
    """Check that the report uses cleaned games and generates both charts."""
    records = [
        ("a", True, "white", 1700, 1500, "Italian Game: Variation", "10+0"),
        ("b", True, "black", 1500, 1500, "Italian Game", "10+0"),
        ("c", True, "draw", 1300, 1500, "French Defense", "30+0"),
        ("a", True, "white", 2000, 1500, "Sicilian Defense", "10+0"),
        ("d", False, "black", 2100, 1500, "Sicilian Defense", "10+0"),
    ]
    columns = [
        "id", "rated", "winner", "white_rating", "black_rating",
        "opening_name", "increment_code",
    ]
    raw = pd.DataFrame(records, columns=columns)
    raw = raw.assign(white_id="white-player", black_id="black-player")

    with TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "data").mkdir()
        raw.to_csv(root / "data" / "games.csv", index=False)

        with patch.object(rq3_eda, "ROOT", root):
            rq3_eda.main()

        output = root / "figures" / "rq3"
        ratings = pd.read_csv(output / "rating_summary.csv", index_col=0)
        assert ratings.loc["White rating", "Mean"] == 1500
        assert ratings.loc["Rating difference", "Mean"] == 0
        assert ratings.loc["Rating difference", "Min"] == -200
        assert ratings.loc["Rating difference", "Max"] == 200

        openings = pd.read_csv(
            output / "opening_family_counts.csv", index_col=0
        )["Games"].to_dict()
        assert openings == {"Italian Game": 2, "French Defense": 1}

        time_controls = pd.read_csv(
            output / "time_control_counts.csv", index_col=0
        )["Games"].to_dict()
        assert time_controls == {"rapid": 2, "Unclassified": 1}

        charts = ("game_outcomes.png", "white_win_rate_by_rating_gap.png")
        for name in charts:
            assert (output / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_rating_gap_win_rate_includes_draws() -> None:
    """A win and a draw in the same group give a 50 percent win rate."""
    games = pd.DataFrame({
        "white_rating": [1500, 1500, 1900],
        "black_rating": [1500, 1500, 1500],
        "winner": ["white", "draw", "black"],
    })
    rates = kaggle_plots.win_rate_by_rating_gap(games)
    assert rates["size"].tolist() == [2, 1]
    assert rates["mean"].tolist() == [0.5, 0.0]


def main() -> None:
    """Run the RQ3 tests without requiring pytest."""
    for test in (test_rq3_outputs, test_rating_gap_win_rate_includes_draws):
        test()
        print("passed:", test.__name__)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
