"""Shared loading and cleaning code for the chess EDA.

Both datasets (Lichess PGN sample and Kaggle CSV) are converted to one
common schema so the same EDA functions can run on either of them:

    white_elo, black_elo, result ("1-0", "0-1", "1/2-1/2"),
    time_control ("bullet" / "blitz" / "rapid" / None),
    opening_family, avg_rating, rating_gap, rating_difference,
    white_player, black_player

Player names let us check how often the same player repeats, which
matters for the "independent games" assumption of the chi-square test.
"""
import io
import re

import pandas as pd

TAG_PATTERN = re.compile(r'^\[(\w+) "(.*)"\]$')
LICHESS_TAGS = [
    "Event", "Site", "White", "Black", "Result", "WhiteElo", "BlackElo",
    "WhiteTitle", "BlackTitle", "Opening", "ECO", "TimeControl",
    "Termination",
]
VALID_RESULTS = ["1-0", "0-1", "1/2-1/2"]
COMMON_COLUMNS = [
    "white_elo", "black_elo", "result", "time_control",
    "opening_family", "avg_rating", "rating_gap", "rating_difference",
    "white_player", "black_player",
]


def parse_time_control(tc, base_unit=1):
    """Return the estimated game duration in seconds, or None.

    tc looks like "300+3" (base + increment). The estimate is
    base * base_unit + 40 * increment. Lichess PGN gives the base in
    seconds (base_unit=1); check the Kaggle units before reusing this.
    Untimed or correspondence values such as "-" return None.
    """
    if not isinstance(tc, str) or "+" not in tc:
        return None
    base, increment = tc.split("+", 1)
    if not (base.isdigit() and increment.isdigit()):
        return None
    return int(base) * base_unit + 40 * int(increment)


def classify_time_control(duration):
    """Map an estimated duration (seconds) to bullet, blitz or rapid.

    Anything of 1500 seconds or more (or missing) returns None.
    """
    if duration is None or pd.isna(duration):
        return None
    if duration < 180:
        return "bullet"
    if duration < 480:
        return "blitz"
    if duration < 1500:
        return "rapid"
    return None


def opening_family(name):
    """Keep only the part of an opening name before the first colon."""
    if not isinstance(name, str) or name.strip() in ("", "?"):
        return None
    return name.split(":")[0].strip()


def read_pgn_headers(lines, max_games=None):
    """Yield one dict of header tags per game from an iterable of lines.

    Move text is skipped, so memory use stays small even for huge
    files. Stops after max_games games if it is given.
    """
    game = {}
    count = 0
    for line in lines:
        match = TAG_PATTERN.match(line.strip())
        if not match:
            continue
        tag, value = match.groups()
        if tag == "Event" and game:
            yield game
            count += 1
            game = {}
            if max_games is not None and count >= max_games:
                return
        if tag in LICHESS_TAGS:
            game[tag] = value
    if game and (max_games is None or count < max_games):
        yield game


def open_pgn(path):
    """Open a .pgn or .pgn.zst file as a stream of text lines."""
    if str(path).endswith(".zst"):
        import zstandard
        raw = open(path, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, encoding="utf-8")


def add_rating_columns(df):
    """Add avg_rating, rating_difference (White - Black), rating_gap."""
    df = df.copy()
    df["avg_rating"] = (df["white_elo"] + df["black_elo"]) / 2
    df["rating_difference"] = df["white_elo"] - df["black_elo"]
    df["rating_gap"] = df["rating_difference"].abs()
    return df


def clean_lichess(raw):
    """Clean a DataFrame of raw Lichess header tags.

    Removes bot games, missing or "?" ratings, unfinished or abandoned
    games and duplicate games (same Site URL). Games whose time control
    is not bullet/blitz/rapid are kept with time_control = None.
    """
    df = raw.reindex(columns=LICHESS_TAGS).copy()
    df["white_elo"] = pd.to_numeric(df["WhiteElo"], errors="coerce")
    df["black_elo"] = pd.to_numeric(df["BlackElo"], errors="coerce")
    is_bot = (df["WhiteTitle"] == "BOT") | (df["BlackTitle"] == "BOT")
    df = df[~is_bot]
    df = df[df["Result"].isin(VALID_RESULTS)]
    df = df[df["Termination"] != "Abandoned"]
    df = df.dropna(subset=["white_elo", "black_elo"])
    df = df.drop_duplicates(subset="Site")
    df = df.rename(columns={"Result": "result", "White": "white_player",
                            "Black": "black_player"})
    df["time_control"] = (
        df["TimeControl"].map(parse_time_control)
        .map(classify_time_control)
    )
    df["opening_family"] = df["Opening"].map(opening_family)
    df = add_rating_columns(df)
    return df[COMMON_COLUMNS].reset_index(drop=True)


def load_lichess_raw(path, max_games):
    """Stream up to max_games raw games from a PGN file (no cleaning)."""
    with open_pgn(path) as lines:
        games = list(read_pgn_headers(lines, max_games))
    return pd.DataFrame(games)


def load_lichess_sample(path, max_games):
    """Stream up to max_games raw games from a PGN file and clean them."""
    return clean_lichess(load_lichess_raw(path, max_games))


def load_both(kaggle_csv="data/games.csv",
              lichess_parquet="data/lichess_clean.parquet"):
    """Load the clean Kaggle and Lichess DataFrames (the team gate)."""
    return load_kaggle(kaggle_csv), pd.read_parquet(lichess_parquet)


def load_kaggle(path):
    """Load the Kaggle chess CSV into the common schema.

    Keeps rated games only. increment_code (e.g. "15+2") is assumed to
    be minutes + seconds; verify this on a few rows.
    """
    df = pd.read_csv(path)
    df = df[df["rated"].astype(str).str.lower() == "true"]
    df = df.drop_duplicates(subset="id")
    result_map = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}
    out = pd.DataFrame({
        "white_elo": df["white_rating"],
        "black_elo": df["black_rating"],
        "result": df["winner"].map(result_map),
        "opening_family": df["opening_name"].map(opening_family),
        "white_player": df["white_id"],
        "black_player": df["black_id"],
    })
    durations = df["increment_code"].map(
        lambda tc: parse_time_control(tc, base_unit=60))
    out["time_control"] = durations.map(classify_time_control)
    out = out.dropna(subset=["white_elo", "black_elo", "result"])
    out = add_rating_columns(out)
    return out[COMMON_COLUMNS].reset_index(drop=True)
