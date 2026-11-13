# Chess EDA (CSE 163 final project, Part 2)

Analyzing how openings, ratings, and time controls shape win rates in
online chess. Datasets: Kaggle [`datasnaek/chess`](https://www.kaggle.com/datasets/datasnaek/chess)
(`games.csv`) and the October 2014 Lichess standard rated games file
(`lichess_db_standard_rated_2014-10.pgn.zst`) from the
[Lichess database](https://database.lichess.org/#standard_games).

**Data is not stored in this repo.** Download it yourself, or get
`games.csv` and `lichess_clean.parquet` from the team Drive folder, and
put them in `data/`.

## Files

| File | What it does |
|---|---|
| `data_loader.py` | Streams and cleans the Lichess PGN, loads the Kaggle CSV, and converts both to one shared schema |
| `eda_utils.py` | `summarize()`: shape, missing data, seven-number summary, value counts |
| `eda.py` | Runs the loaders and summaries, runs sanity checks, and saves the clean Lichess sample |
| `check_assumptions.py` | Prints real rows to confirm the loader's assumptions (setup step 4) |
| `kaggle_plots.py` | Kaggle figures: game outcomes, White's win rate by rating gap (saved to `figures/`) |
| `test_*.py` | Tests on tiny hand-made fixtures |

## Setup order

1. `pip install -r requirements.txt`
2. Put the two downloaded files in `data/` (the whole folder is
   git-ignored):
   - `data/games.csv` (Kaggle)
   - `data/lichess_db_standard_rated_2014-10.pgn.zst` (Lichess;
     **do not unzip it**, the loader reads it directly)
3. Confirm the assumptions on the real files and read the output:
   ```
   python check_assumptions.py --kaggle data/games.csv
   python check_assumptions.py --lichess data/lichess_db_standard_rated_2014-10.pgn.zst
   ```
   - **Kaggle:** the `increment_code` base time should be in minutes
     (`15+2` = 15 minutes). Compare the two "time-control counts" tables
     and the "median real game length" table. If the base is actually in
     seconds, change `base_unit=60` to `base_unit=1` in
     `data_loader.load_kaggle`.
   - **Lichess:** "Loader tags never seen" should be empty or only
     `WhiteTitle` / `BlackTitle`. Those tags only appear for titled
     players and bots, so they can be absent from the first 1000 games.
     If any other tag is listed, tell the team before going on.
4. Run the shared EDA. It prints the summaries and saves
   `data/lichess_clean.parquet`:
   ```
   python eda.py --kaggle data/games.csv
   python eda.py --lichess data/lichess_db_standard_rated_2014-10.pgn.zst --max-games 200000
   ```
5. Check the printed numbers and post them in the team chat:
   - both datasets show 10 columns;
   - "Sanity checks: all passed";
   - the Lichess "dropped" percentage is roughly 3-10%;
   - bullet, blitz, and rapid all appear in `time_control`.
6. Upload `data/games.csv` and `data/lichess_clean.parquet` to the team
   Drive folder. **Do not commit any data files.** Teammates download
   those two files into their own `data/` folder.

## Gate: everyone must be able to run this

```python
import data_loader as dl
kaggle, lichess = dl.load_both()   # data/games.csv + data/lichess_clean.parquet
print(kaggle.shape, lichess.shape)
```

## Shared columns (both datasets)

`white_elo`, `black_elo`, `result` (`1-0`, `0-1`, `1/2-1/2`),
`time_control` (`bullet` / `blitz` / `rapid` / empty for others),
`opening_family` (opening name before the first colon), `avg_rating`,
`rating_gap` (absolute), `rating_difference` (White minus Black),
`white_player`, `black_player` (usernames, for checking repeat players).

## Notes and limits

- The Lichess sample is the **first N games** of the month, so it covers
  only the first part of that month. State this in the report.
- Classical games and games with no opening are kept, with an empty
  `time_control` or `opening_family`. RQ2 should drop empty
  `time_control`; RQ1 should drop empty `opening_family`.
- Do not use `WhiteRatingDiff` / `BlackRatingDiff` for prediction (RQ3);
  they reveal the result. They are not loaded.

## Tests and style

```
python test_data_loader.py
python test_eda_utils.py
python test_eda_and_checks.py
python kaggle_plots.py
flake8 *.py
```
