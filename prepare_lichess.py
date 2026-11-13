"""Extract game headers from a Lichess PGN stream into a CSV."""

import csv
import re
import sys

FIELDS = (
    "Site", "UTCDate", "UTCTime", "White", "Black", "WhiteElo", "BlackElo",
    "Opening", "ECO", "Result", "TimeControl", "Termination",
)
TAG = re.compile(r'^\[(\w+) "(.*)"\]$')


def main() -> None:
    """Read decompressed PGN from stdin and write one CSV row per game."""
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    game = {}
    for line in sys.stdin:
        if not line.startswith("["):
            continue
        match = TAG.fullmatch(line.rstrip())
        if match is None:
            continue
        name, value = match.groups()
        if name == "Event":
            if game:
                writer.writerow(game)
            game = {}
        elif name in FIELDS:
            game[name] = value
    if game:
        writer.writerow(game)


if __name__ == "__main__":
    main()
