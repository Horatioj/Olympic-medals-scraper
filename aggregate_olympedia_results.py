"""
Aggregate Olympedia athlete-level results into event-level and national-team datasets.

Inputs are produced by scrape_olympedia_results.py. This script:
1. Optionally drops programmed contests that did not award official Olympic medals (see NON_MEDAL_PROGRAMME).
2. Builds one country/event/result row with athlete counts, female proportions, and scores.
3. Builds one national-team/Games row with athlete counts and normalized performance scores.

Scores use ``parse_rank`` on the whole ``position`` cell (compound heat strings do not yield
pure rank ``1/k`` → participation ``1/9`` only). Medal-style tallies **from placement text**
instead use ``verify_medal_counts_from_positions.py`` (strict per-``|`` segments, ignores
segments with letters).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pycountry


NON_MEDAL_SCORE = 1 / 9

HOSTS = {
    ("1996 Summer Olympics", 1996, "Summer"): ("United States", "USA"),
    ("1998 Winter Olympics", 1998, "Winter"): ("Japan", "JPN"),
    ("2000 Summer Olympics", 2000, "Summer"): ("Australia", "AUS"),
    ("2002 Winter Olympics", 2002, "Winter"): ("United States", "USA"),
    ("2004 Summer Olympics", 2004, "Summer"): ("Greece", "GRC"),
    ("2006 Winter Olympics", 2006, "Winter"): ("Italy", "ITA"),
    ("2008 Summer Olympics", 2008, "Summer"): ("China", "CHN"),
    ("2010 Winter Olympics", 2010, "Winter"): ("Canada", "CAN"),
    ("2012 Summer Olympics", 2012, "Summer"): ("United Kingdom", "GBR"),
    ("2014 Winter Olympics", 2014, "Winter"): ("Russia", "RUS"),
    ("2016 Summer Olympics", 2016, "Summer"): ("Brazil", "BRA"),
    ("2018 Winter Olympics", 2018, "Winter"): ("South Korea", "KOR"),
    ("2020 Summer Olympics", 2020, "Summer"): ("Japan", "JPN"),
    ("2022 Winter Olympics", 2022, "Winter"): ("China", "CHN"),
    ("2024 Summer Olympics", 2024, "Summer"): ("France", "FRA"),
}

# ---------------------------------------------------------------------------
# Exclusions: contests that appear on Olympedia/Games sites but did not award
# official Olympic medals. Extend this list when you find more cases.
#
# Background (web):
# - IOC "demonstration sports" historically: https://en.wikipedia.org/wiki/Demonstration_sport
# - 2008 Wushu was a sanctioned parallel tournament, not IOC medal programme:
#   https://en.wikipedia.org/wiki/2008_Beijing_Wushu_Tournament
#
# Tuple: (games_year or None), season ("Summer"|"Winter"), sport string as in
# the athlete CSV/Olympedia, event_substrings or None to drop entire sport rows.
NON_MEDAL_PROGRAMME: list[tuple[int | None, str, str, tuple[str, ...] | None]] = [
    (2008, "Summer", "Wushu", None),
]

# Olympedia/NOC codes are not always ISO-3166 alpha-3 codes.
NOC_TO_ISO3 = {
    "AIN": "RUS",
    "AHO": "ANT",
    "ANT": "ATG",
    "ARU": "ABW",
    "ASA": "ASM",
    "BAH": "BHS",
    "BAN": "BGD",
    "BAR": "BRB",
    "BER": "BMU",
    "BIZ": "BLZ",
    "BOH": "",
    "BRN": "BHR",
    "BRU": "BRN",
    "BUL": "BGR",
    "BUR": "BFA",
    "CAM": "KHM",
    "CAY": "CYM",
    "CGO": "COG",
    "CHA": "TCD",
    "CHI": "CHL",
    "CIV": "CIV",
    "CRC": "CRI",
    "CRO": "HRV",
    "DEN": "DNK",
    "ESA": "SLV",
    "EUN": "",
    "FIJ": "FJI",
    "GER": "DEU",
    "GRE": "GRC",
    "GUA": "GTM",
    "GUI": "GIN",
    "GUM": "GUM",
    "HAI": "HTI",
    "INA": "IDN",
    "IRI": "IRN",
    "ISV": "VIR",
    "IVB": "VGB",
    "KOS": "XKX",
    "LAT": "LVA",
    "LBA": "LBY",
    "LIB": "LBN",
    "MAD": "MDG",
    "MAS": "MYS",
    "MAW": "MWI",
    "MDA": "MDA",
    "MGL": "MNG",
    "MON": "MCO",
    "MRI": "MUS",
    "MTN": "MRT",
    "MYA": "MMR",
    "NED": "NLD",
    "NGR": "NGA",
    "NIG": "NER",
    "OMA": "OMN",
    "PAR": "PRY",
    "PHI": "PHL",
    "PLE": "PSE",
    "POR": "PRT",
    "PRK": "PRK",
    "PUR": "PRI",
    "ROC": "RUS",
    "ROT": "",
    "RSA": "ZAF",
    "SAM": "WSM",
    "SCG": "",
    "SIN": "SGP",
    "SLO": "SVN",
    "SOL": "SLB",
    "SRI": "LKA",
    "SUI": "CHE",
    "TAN": "TZA",
    "TCH": "",
    "TGA": "TON",
    "TPE": "TWN",
    "UAE": "ARE",
    "URU": "URY",
    "VAN": "VUT",
    "VIE": "VNM",
    "VIN": "VCT",
    "YUG": "",
    "ZIM": "ZWE",
}

COUNTRY_NAME_TO_ISO3 = {
    "Bolivia": "BOL",
    "Chinese Taipei": "TWN",
    "Congo": "COG",
    "Czech Republic": "CZE",
    "Democratic Republic of the Congo": "COD",
    "Great Britain": "GBR",
    "Hong Kong, China": "HKG",
    "Iran": "IRN",
    "Kosovo": "XKX",
    "Laos": "LAO",
    "Moldova": "MDA",
    "North Korea": "PRK",
    "Palestine": "PSE",
    "Russia": "RUS",
    "South Korea": "KOR",
    "Syria": "SYR",
    "Tanzania": "TZA",
    "United States": "USA",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def row_matches_non_medal_programme(year_value: object, season: object, sport: object, event: object) -> bool:
    """True if this row should be omitted from IOC-style aggregates."""
    year_text = clean_text(year_value)
    try:
        year_int = int(float(year_text)) if year_text else None
    except ValueError:
        year_int = None

    season_n = clean_text(season)
    sport_n = clean_text(sport)
    event_n = clean_text(event)

    for yr, seas, sport_rule, event_filters in NON_MEDAL_PROGRAMME:
        if yr is not None and year_int != yr:
            continue
        if clean_text(seas).lower() != season_n.lower():
            continue
        if sport_rule.lower() != sport_n.lower():
            continue
        if event_filters is None:
            return True
        if any(pat.lower() in event_n.lower() for pat in event_filters):
            return True
    return False


def drop_non_medal_programme(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = df.apply(
        lambda r: row_matches_non_medal_programme(r["year"], r["season"], r["sport"], r["event"]),
        axis=1,
    )
    removed = int(mask.sum())
    return df.loc[~mask].copy(), removed


def country_to_iso3(country: str, noc: str = "") -> str:
    noc = clean_text(noc).upper()
    if noc in NOC_TO_ISO3:
        return NOC_TO_ISO3[noc]
    if len(noc) == 3 and pycountry.countries.get(alpha_3=noc):
        return noc

    country = clean_text(country)
    if country in COUNTRY_NAME_TO_ISO3:
        return COUNTRY_NAME_TO_ISO3[country]

    try:
        return pycountry.countries.lookup(country).alpha_3
    except LookupError:
        return noc if len(noc) == 3 else ""


def host_for_game(game: str, year: int, season: str) -> tuple[str, str]:
    return HOSTS.get((clean_text(game), int(year), clean_text(season)), ("", ""))


def is_host_team(team_iso3: str, team_noc: str, host_iso3: str, year: int) -> bool:
    if team_iso3 and team_iso3 == host_iso3:
        return True

    # Treat Hong Kong, China as a host-country team for Olympics hosted by China.
    if host_iso3 == "CHN" and int(year) in {2008, 2022} and team_noc == "HKG":
        return True

    return False


def parse_rank(position: str) -> int | None:
    """
    Parsed overall place only when the whole cell is a single integer rank.

    Examples that return a rank: ``3``, ``=9``, ``=17``.
    Examples that return None (not a pure final/simple rank token):
    ``DQ``, ``DNS``, ``7 h3 r2/5``, ``DQ final``.
    Those None cases are scored as participation (see ``score_from_position``).
    """
    position = clean_text(position)
    match = re.fullmatch(r"=?\s*(\d+)\s*", position)
    if not match:
        return None
    return int(match.group(1))


def score_from_position(position: str) -> float:
    """
    Map one country-entry ``position`` string to raw score contribution.

    * Pure numeric ranks ``1``..``8`` (optional leading ``=``) → ``1/rank``.
    * Any other non-empty placement (DQ, DNS, DNF, heat strings like ``7 h3 r2/5``, ``NM``, …)
      counts as participation → ``NON_MEDAL_SCORE`` (currently ``1/9``).
    * Rank ``9`` and above counts as participation → ``1/9`` as before.

    Edit **here** (and ``NON_MEDAL_SCORE`` at module top) to change behaviour.
    """
    position = clean_text(position)
    if not position:
        return NON_MEDAL_SCORE

    rank = parse_rank(position)
    if rank is None:
        # Non-numeric or compound strings: treated as competed, not podium rank scoring.
        return NON_MEDAL_SCORE
    if rank >= 9:
        return NON_MEDAL_SCORE
    return 1 / rank


def normalize_sex(value: str) -> str:
    value = clean_text(value)
    if value in {"Male", "Female"}:
        return value
    return ""


def gender_score_weights(event_gender: str, female_proportion: float | None) -> tuple[float, float]:
    if event_gender == "Women":
        return 1.0, 0.0
    if event_gender == "Men":
        return 0.0, 1.0
    if event_gender == "Mixed":
        return 0.5, 0.5

    if female_proportion is not None and pd.notna(female_proportion):
        female_weight = float(female_proportion)
        return female_weight, 1 - female_weight

    return 0.5, 0.5


def join_unique(values: pd.Series) -> str:
    unique_values = []
    seen = set()
    for value in values:
        text = clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique_values.append(text)
    return " | ".join(unique_values)


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["athlete_id"] = df["athlete_id"].astype(str).replace("nan", pd.NA)
    df["sex"] = df["sex"].map(normalize_sex)
    df["team_iso3"] = [
        country_to_iso3(country, noc)
        for country, noc in zip(df["country"], df["noc"], strict=False)
    ]

    host_info = [host_for_game(game, year, season) for game, year, season in zip(df["game"], df["year"], df["season"], strict=False)]
    df["host_country"] = [item[0] for item in host_info]
    df["host_iso3"] = [item[1] for item in host_info]
    df["is_host_country_team"] = [
        is_host_team(team_iso3, noc, host_iso3, year)
        for team_iso3, noc, host_iso3, year in zip(df["team_iso3"], df["noc"], df["host_iso3"], df["year"], strict=False)
    ]
    return df


def make_event_level(df: pd.DataFrame) -> pd.DataFrame:
    entry_cols = [
        "game",
        "year",
        "season",
        "host_country",
        "host_iso3",
        "sport",
        "event",
        "event_gender",
        "result_url",
        "country",
        "team_iso3",
        "noc",
        "position",
        "medal",
        "team_or_entry",
    ]

    entry_rows = []
    for keys, group in df.groupby(entry_cols, dropna=False, sort=True):
        record = dict(zip(entry_cols, keys, strict=False))
        athlete_ids = group["athlete_id"].dropna().astype(str)
        total_athletes = athlete_ids.nunique()
        female_athletes = group.loc[group["sex"] == "Female", "athlete_id"].dropna().astype(str).nunique()
        male_athletes = group.loc[group["sex"] == "Male", "athlete_id"].dropna().astype(str).nunique()
        known_gender_athletes = female_athletes + male_athletes
        female_proportion = female_athletes / total_athletes if total_athletes else pd.NA

        event_score = score_from_position(record["position"])
        female_weight, male_weight = gender_score_weights(record["event_gender"], female_proportion)

        record.update(
            {
                "participating_country": record.pop("country"),
                "participating_country_iso3": record.pop("team_iso3"),
                "participating_noc": record.pop("noc"),
                "number_of_athletes": total_athletes,
                "number_of_female_athletes": female_athletes,
                "number_of_male_athletes": male_athletes,
                "number_of_known_gender_athletes": known_gender_athletes,
                "female_proportion": female_proportion,
                "rank": parse_rank(record["position"]),
                "event_score_raw": event_score,
                "event_score_female_raw": event_score * female_weight,
                "event_score_male_raw": event_score * male_weight,
            }
        )
        entry_rows.append(record)

    entry_df = pd.DataFrame(entry_rows)
    country_event_cols = [
        "game",
        "year",
        "season",
        "host_country",
        "host_iso3",
        "sport",
        "event",
        "event_gender",
        "result_url",
        "participating_country",
        "participating_country_iso3",
        "participating_noc",
    ]

    rows = []
    for keys, group in entry_df.groupby(country_event_cols, dropna=False, sort=True):
        record = dict(zip(country_event_cols, keys, strict=False))
        total_athletes = int(group["number_of_athletes"].sum())
        female_athletes = int(group["number_of_female_athletes"].sum())
        male_athletes = int(group["number_of_male_athletes"].sum())
        known_gender_athletes = female_athletes + male_athletes
        female_proportion = female_athletes / total_athletes if total_athletes else pd.NA
        ranks = group["rank"].dropna()

        record.update(
            {
                "position": join_unique(group["position"]),
                "medal": join_unique(group["medal"]),
                "team_or_entries": join_unique(group["team_or_entry"]),
                "number_of_entries": int(len(group)),
                "number_of_athletes": total_athletes,
                "number_of_female_athletes": female_athletes,
                "number_of_male_athletes": male_athletes,
                "number_of_known_gender_athletes": known_gender_athletes,
                "female_proportion": female_proportion,
                "best_rank": int(ranks.min()) if not ranks.empty else pd.NA,
                "event_score_raw": group["event_score_raw"].sum(),
                "event_score_female_raw": group["event_score_female_raw"].sum(),
                "event_score_male_raw": group["event_score_male_raw"].sum(),
            }
        )
        rows.append(record)

    event_df = pd.DataFrame(rows)
    event_df = event_df.sort_values(
        ["year", "season", "sport", "event", "best_rank", "participating_country_iso3"],
        na_position="last",
    )
    return event_df


def make_national_team_level(df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    athlete_counts = (
        df.groupby(["game", "year", "season", "host_country", "host_iso3", "country", "team_iso3", "noc"], dropna=False)
        .agg(
            total_number_of_athletes=("athlete_id", "nunique"),
            number_of_female_athletes=("athlete_id", lambda values: df.loc[values.index[df.loc[values.index, "sex"] == "Female"], "athlete_id"].nunique()),
            number_of_male_athletes=("athlete_id", lambda values: df.loc[values.index[df.loc[values.index, "sex"] == "Male"], "athlete_id"].nunique()),
        )
        .reset_index()
    )

    score_sums = (
        event_df.groupby(["game", "year", "season", "participating_country", "participating_country_iso3", "participating_noc"], dropna=False)
        .agg(
            score_total_raw=("event_score_raw", "sum"),
            score_female_raw=("event_score_female_raw", "sum"),
            score_male_raw=("event_score_male_raw", "sum"),
        )
        .reset_index()
    )

    game_totals = (
        event_df.groupby(["game", "year", "season"], dropna=False)
        .agg(game_total_score_raw=("event_score_raw", "sum"))
        .reset_index()
    )

    national_df = athlete_counts.merge(
        score_sums,
        left_on=["game", "year", "season", "country", "team_iso3", "noc"],
        right_on=["game", "year", "season", "participating_country", "participating_country_iso3", "participating_noc"],
        how="left",
    ).merge(game_totals, on=["game", "year", "season"], how="left")

    national_df["score_total"] = national_df["score_total_raw"] / national_df["game_total_score_raw"]
    national_df["score_female"] = national_df["score_female_raw"] / national_df["game_total_score_raw"]
    national_df["score_male"] = national_df["score_male_raw"] / national_df["game_total_score_raw"]
    national_df["proportion_female_athletes"] = (
        national_df["number_of_female_athletes"] / national_df["total_number_of_athletes"]
    )
    national_df["proportion_male_athletes"] = (
        national_df["number_of_male_athletes"] / national_df["total_number_of_athletes"]
    )
    national_df["is_host_country_team"] = [
        is_host_team(team_iso3, noc, host_iso3, year)
        for team_iso3, noc, host_iso3, year in zip(
            national_df["team_iso3"],
            national_df["noc"],
            national_df["host_iso3"],
            national_df["year"],
            strict=False,
        )
    ]

    national_df = national_df.rename(
        columns={
            "country": "national_olympic_team",
            "team_iso3": "national_iso3",
            "noc": "national_noc",
            "game": "participated_olympic_game",
        }
    )
    national_df = national_df[
        [
            "national_olympic_team",
            "national_iso3",
            "national_noc",
            "participated_olympic_game",
            "year",
            "season",
            "host_country",
            "host_iso3",
            "is_host_country_team",
            "total_number_of_athletes",
            "number_of_male_athletes",
            "number_of_female_athletes",
            "proportion_male_athletes",
            "proportion_female_athletes",
            "score_total",
            "score_female",
            "score_male",
            "score_total_raw",
            "score_female_raw",
            "score_male_raw",
            "game_total_score_raw",
        ]
    ].sort_values(["year", "season", "national_iso3"])
    return national_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Olympedia athlete-level results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/olympedia_results_2000_2024_gender_mixed_bios.csv"),
    )
    parser.add_argument(
        "--event-output",
        type=Path,
        default=Path("data/olympedia_event_level_2000_2024.csv"),
    )
    parser.add_argument(
        "--national-output",
        type=Path,
        default=Path("data/olympedia_national_team_level_2000_2024.csv"),
    )
    parser.add_argument(
        "--keep-non-medal-programme",
        action="store_true",
        help="Keep disciplines listed in NON_MEDAL_PROGRAMME (e.g. 2008 Wushu); default drops them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, dtype=str)

    if not args.keep_non_medal_programme:
        df, removed = drop_non_medal_programme(df)
        if removed:
            print(
                f"Excluded {removed:,} athlete-level rows matching NON_MEDAL_PROGRAMME "
                f"(IOC non-medal / parallel contests). "
                "Use --keep-non-medal-programme to retain them.",
                flush=True,
            )

    df = add_basic_fields(df)

    event_df = make_event_level(df)
    national_df = make_national_team_level(df, event_df)

    args.event_output.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(args.event_output, index=False, encoding="utf-8-sig")
    national_df.to_csv(args.national_output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(event_df):,} rows to {args.event_output}")
    print(f"Wrote {len(national_df):,} rows to {args.national_output}")


if __name__ == "__main__":
    main()
