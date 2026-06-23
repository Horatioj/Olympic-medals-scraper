"""
Verify medal-shaped tallies using only Olympedia-derived `position` strings from the
event-level CSV (never the `medal` column).

Rules
-----
- Split each ``position`` cell on ``|`` into **segments** (one athlete/phase per segment).
- **Strict (default):** a segment counts only if it has **no letters** (so heat text like
  ``1 h1 r1/4`` or ``5 h4 r2/4`` is ignored) and the whole trimmed segment is an
  optional leading ``=`` plus digits only, e.g. ``1``, ``=3``, `` 2 ``.
  This aligns better with ``aggregate_olympedia_results.py`` scoring, which already
  treats compound non-pure ranks as participation (``1/9``), not podium ``1/k``.
- Count gold/silver/bronze when that rank is ``1``, ``2``, or ``3``.

**Note:** ``aggregate_olympedia_results.py`` does **not** rewrite the ``position``
text in the CSV; it only applies strict logic when computing **scores**. Medal-style
verification is done in **this** script (and ``diagnose_medal_positions.py``).

This matches IOC-style counting for most sports but can differ slightly from official
IOC tables due to reallocations, data quirks, heat vs final duplication on source pages,
or non-numeric placeholders (DQ, DNS, '7 h3 r2/5', …) that never contribute.

Ties: IOC does **not** award "silver + bronze for two athletes both in 1st". Joint wins
typically mean multiple gold medals; our rule counts each NOC's ``1`` tokens separately.

To inspect shared ``1``s on the same ``result_url`` (same Olympic competition row on
Olympedia), run: ``python diagnose_medal_positions.py --year YYYY --season Summer|Winter \\
    --show-shared-gold`` (optionally ``--noc AUS`` to list that team's medal-bearing rows).

Usage
-----
python verify_medal_counts_from_positions.py
python verify_medal_counts_from_positions.py --event-csv data/olympedia_event_level_2000_2024.csv \\
    --countries USA CHN GBR NOR --years 2024 2020 2012
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


_STRICT_SEGMENT_RANK = re.compile(r"^\s*=?\s*(\d+)\s*$")


def placements_from_position(position: str | float) -> list[int]:
    """
    Numeric ranks suitable for IOC-style medal counting from ``position``.

    Ignores segments that contain letters (heats: ``1 h1 r1/4``, ``DNS``, etc.).
    Only pure ``=?\d+`` segments count, so one cell can still list multiple finalists
    as ``1 | 2`` (two segments, both numeric).
    """
    if pd.isna(position) or str(position).strip() == "":
        return []
    ranks: list[int] = []
    for part in str(position).split("|"):
        chunk = part.strip()
        if not chunk or re.search(r"[A-Za-z]", chunk):
            continue
        match = _STRICT_SEGMENT_RANK.match(chunk)
        if match:
            ranks.append(int(match.group(1)))
    return ranks


def tally_medals(placements: list[int]) -> tuple[int, int, int]:
    gold = sum(1 for r in placements if r == 1)
    silver = sum(1 for r in placements if r == 2)
    bronze = sum(1 for r in placements if r == 3)
    return gold, silver, bronze


def summarize(
    df: pd.DataFrame,
    *,
    nocs: list[str],
    years: list[int] | None,
    seasons: list[str] | None,
) -> pd.DataFrame:
    rows = df.copy()
    placements = rows["position"].map(placements_from_position)
    rows["gold"] = placements.map(lambda vals: tally_medals(vals)[0])
    rows["silver"] = placements.map(lambda vals: tally_medals(vals)[1])
    rows["bronze"] = placements.map(lambda vals: tally_medals(vals)[2])

    if seasons:
        rows = rows[rows["season"].isin(seasons)]

    totals = rows.groupby(["participating_noc", "year", "season"], dropna=False)[
        ["gold", "silver", "bronze"]
    ].sum()
    totals["total_medal_analog"] = totals["gold"] + totals["silver"] + totals["bronze"]

    if years is not None:
        totals = totals[totals.index.get_level_values("year").isin([str(y) for y in years])]

    if nocs:
        idx = totals.index.get_level_values("participating_noc").isin(nocs)
        totals = totals[idx]

    return totals.sort_values(["year", "season", "gold"], ascending=[True, True, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument(
        "--event-csv",
        type=Path,
        default=Path("data/olympedia_event_level_2000_2024.csv"),
    )
    parser.add_argument("--countries", nargs="*", default=["USA", "CHN", "GBR", "NOR", "AUS", "NED"])
    parser.add_argument("--years", nargs="*", type=int, default=[2024, 2020, 2012])
    parser.add_argument(
        "--seasons",
        nargs="*",
        default=None,
        help="Subset of Summer or Winter (default: all rows for selected years).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.event_csv, dtype=str)
    print(
        summarize(
            df,
            nocs=[c.upper() for c in args.countries],
            years=args.years,
            seasons=args.seasons,
        ).to_string()
    )


if __name__ == "__main__":
    main()
