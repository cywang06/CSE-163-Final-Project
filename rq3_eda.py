"""Regenerate the RQ3 figures and tables with the team's Kaggle loader."""

from pathlib import Path

import data_loader
import kaggle_plots


ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Load the cleaned games and save the RQ3 results."""
    folder = ROOT / "figures" / "rq3"
    folder.mkdir(parents=True, exist_ok=True)

    games = data_loader.load_kaggle(ROOT / "data" / "games.csv")
    columns = ["white_elo", "black_elo", "rating_difference"]
    ratings = games[columns].describe().drop(index="count").T.round(2)
    ratings.index = ["White rating", "Black rating", "Rating difference"]
    ratings.columns = ["Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"]
    ratings.to_csv(folder / "rating_summary.csv", index_label="Variable")
    for column in ("opening_family", "time_control"):
        counts = games[column].fillna("Unclassified").value_counts()
        counts.rename("Games").to_csv(folder / f"{column}_counts.csv")

    plot_data = games.rename(columns={
        "white_elo": "white_rating",
        "black_elo": "black_rating",
    }).assign(winner=games["result"].map({
        "1-0": "white", "0-1": "black", "1/2-1/2": "draw",
    }))
    assert plot_data["winner"].notna().all()
    rates = kaggle_plots.win_rate_by_rating_gap(plot_data)
    assert rates["size"].sum() == len(games)
    kaggle_plots.plot_outcomes(plot_data, folder)
    kaggle_plots.plot_win_rate_by_rating_gap(plot_data, folder)
    print(games.shape)
    print(plot_data["winner"].value_counts())
    print(f"Saved results to {folder}")


if __name__ == "__main__":
    main()
