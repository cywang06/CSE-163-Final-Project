# CSE 163 Chess EDA

## Kaggle analysis

Download and extract `games.csv` from the [Chess Game Dataset](https://www.kaggle.com/datasets/datasnaek/chess). Place it next to `eda.py`.

```sh
python -m pip install pandas matplotlib
python eda.py
```

The script prints dataset and outcome summaries and saves two PNG charts beside the script.

## Lichess data preparation

Download the October 2014 standard rated games archive from the [Lichess database](https://database.lichess.org/#standard_games). With `zstd` and Python available, run this from the repository folder after placing the archive there:

```sh
zstd -dc lichess_db_standard_rated_2014-10.pgn.zst | python prepare_lichess.py > lichess_2014-10.csv
```

The converter saves one row per game with ratings, opening, result, time control, and other PGN headers. Keep the downloaded archives and generated data outside version control.
