"""
Explore why position-based medal counts can differ slightly from official IOC tables.

Olympic ties (high level)
-------------------------

Ties do **not** mean "the second athlete who is also ranked 1st gets bronze".

- **Joint 1st** → often **two gold medals**, **no silver**, and bronze goes to the next finisher(s), depending on the sport's rules—not a downgrade of the tied pair to silver/bronze.
- **Joint 3rd** → often **two bronze** medals (= two ``3`` places in results).

Counter logic (same as verify_medal_counts_from_positions.py)
-------------------------------------------------------------
- Each numeric token ``1`` / ``2`` / ``3`` (after stripping ``=``) in a country's
  ``position`` cell counts as **one** gold/silver/bronze **for that country**.
  If two NOCs each have a ``1`` in the same competition, that is **two golds**—which
  matches joint gold on an official medal table.

Common reasons for trivial gaps vs IOC
-------------------------------------
- Tokens that are integers but are **heat / round** placements, not final podium ranks.
- **IOC reallocations** after the Games; Olympedia may not match every revision.
- **Composite** positions: only pure integer tokens count; strings like ``7 h3 r2/5`` do not add medals.

Uses verify_medal_counts_from_positions helpers.

Usage
-----
python diagnose_medal_positions.py --year 2000 --season Summer --noc AUS
python diagnose_medal_positions.py --year 2022 --season Winter --noc GER --show-shared-gold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from verify_medal_counts_from_positions import placements_from_position, tally_medals


def row_medal_counts(position: object) -> tuple[int, int, int]:
    return tally_medals(placements_from_position(position))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect position strings and shared-rank patterns vs IOC-style medal logic."
    )
    parser.add_argument("--event-csv", type=Path, default=Path("data/olympedia_event_level_2000_2024.csv"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--season", type=str, required=True, choices=["Summer", "Winter"])
    parser.add_argument("--noc", type=str, default="", help="Focus one NOC, e.g. AUS")
    parser.add_argument(
        "--show-shared-gold",
        action="store_true",
        help="List competitions (result_url) where 2+ NOCs have a rank-1 token (shared-gold pattern).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="How many medal-bearing rows to list for --noc.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.event_csv, dtype=str)
    subset = df[(df["year"] == str(args.year)) & (df["season"] == args.season)].copy()
    if not len(subset):
        print("No rows for that year/season.")
        return

    print(
        "\nTies: IOC does not turn 'second equal 1st' into bronze. "
        "Shared 1st → multiple golds in the same competition is normal.\n"
    )

    if args.show_shared_gold:
        rows = []
        for url, grp in subset.groupby("result_url", dropna=False):
            if not url or str(url).strip() == "":
                continue
            nocs_with_gold = []
            for _, r in grp.iterrows():
                g, _, _ = row_medal_counts(r["position"])
                if g > 0:
                    nocs_with_gold.append((str(r["participating_noc"]), str(r["position"]), g))
            if len(nocs_with_gold) >= 2:
                rows.append((url, len(nocs_with_gold), nocs_with_gold))
        print(f"Competitions with 2+ NOCs having at least one rank-1 token ({args.year} {args.season}): {len(rows)}")
        for url, n, data in rows[:40]:
            print(f"\n  {url}  ({n} NOCs with gold token)")
            for noc, pos, g in data[:12]:
                print(f"    {noc}  position={repr(pos)}  gold_tokens={g}")
        if len(rows) > 40:
            print(f"\n  ... {len(rows) - 40} more")
        print()

    if args.noc:
        noc = args.noc.upper()
        one = subset[subset["participating_noc"] == noc].copy()
        one[["g", "s", "b"]] = one["position"].apply(
            lambda p: pd.Series(row_medal_counts(p))
        )
        medal_rows = one[(one["g"] > 0) | (one["s"] > 0) | (one["b"] > 0)].copy()
        medal_rows["gsb_total"] = medal_rows["g"] + medal_rows["s"] + medal_rows["b"]
        medal_rows = medal_rows.sort_values(["g", "s", "b"], ascending=False)
        tg, ts, tb = int(medal_rows["g"].sum()), int(medal_rows["s"].sum()), int(medal_rows["b"].sum())
        print(f"{noc} {args.year} {args.season}: position-based G={tg} S={ts} B={tb} (rows with any medal token: {len(medal_rows)})\n")
        cols = ["sport", "event", "position", "g", "s", "b", "result_url"]
        print(medal_rows.head(args.top)[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
