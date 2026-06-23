"""
Scrape Olympedia athlete-event results for Summer/Winter Olympic Games.

The script writes one row per athlete-event result. It scrapes country-by-Games
results pages, which are much smaller and more direct than visiting every
athlete biography page first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.olympedia.org"
COUNTRIES_URL = urljoin(BASE_URL, "/countries")
EDITIONS_URL = urljoin(BASE_URL, "/editions")
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; academic Olympic results scraper)"
}
MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


@dataclass(frozen=True)
class Country:
    noc: str
    name: str
    url: str


@dataclass(frozen=True)
class ResultPage:
    noc: str
    country: str
    edition_id: str
    game: str
    year: int
    season: str
    game_start_date: str
    url: str


@dataclass(frozen=True)
class GameInfo:
    edition_id: str
    game: str
    year: int
    season: str
    start_date: str


@dataclass(frozen=True)
class AthleteBio:
    athlete_id: str
    sex: str
    born: str
    birth_date: str
    height_cm: str
    weight_kg: str


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.html"


def get_html(
    url: str,
    cache_dir: Path,
    *,
    refresh_cache: bool = False,
    max_retries: int = 5,
    timeout: int = 60,
    delay: float = 0.5,
) -> str | None:
    cache_file = cache_path(cache_dir, url)
    if cache_file.exists() and not refresh_cache:
        return cache_file.read_text(encoding="utf-8", errors="replace")

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=timeout,
            )
            if response.status_code == 200:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(response.text, encoding="utf-8")
                if delay:
                    time.sleep(delay)
                return response.text

            last_error = f"HTTP {response.status_code}"
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 30 * attempt
            else:
                wait_seconds = 5 * attempt
        except requests.RequestException as error:
            last_error = str(error)
            wait_seconds = 5 * attempt

        print(f"Fetch failed ({last_error}); retrying in {wait_seconds}s: {url}", flush=True)
        time.sleep(wait_seconds)

    print(f"Skipping after {max_retries} failed attempts: {url} ({last_error})", flush=True)
    return None


def soup_from_html(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def parse_year_and_season(game: str) -> tuple[int | None, str]:
    match = re.search(r"\b(18|19|20)\d{2}\b", game)
    year = int(match.group(0)) if match else None
    season = "Summer" if "Summer" in game else "Winter" if "Winter" in game else ""
    return year, season


def extract_athlete_id(url: str) -> str:
    match = re.search(r"/athletes/(\d+)", url)
    return match.group(1) if match else ""


def extract_edition_id(url: str) -> str:
    match = re.search(r"/editions/(\d+)", url)
    return match.group(1) if match else ""


def parse_calendar_date(value: str, default_year: int | None = None) -> date | None:
    value = clean_text(value)
    match = re.search(
        r"\b(\d{1,2})\s+("
        + "|".join(MONTHS)
        + r")(?:\s+((?:18|19|20)\d{2}))?\b",
        value,
    )
    if not match:
        return None

    day = int(match.group(1))
    month = MONTHS[match.group(2)]
    year = int(match.group(3)) if match.group(3) else default_year
    if year is None:
        return None

    try:
        return date(year, month, day)
    except ValueError:
        return None


def date_to_string(value: date | None) -> str:
    return value.isoformat() if value else ""


def calculate_age_on_date(birth_date: str, reference_date: str) -> str:
    if not birth_date or not reference_date:
        return ""

    born = date.fromisoformat(birth_date)
    reference = date.fromisoformat(reference_date)
    age = reference.year - born.year
    if (reference.month, reference.day) < (born.month, born.day):
        age -= 1
    return str(age)


def parse_measurements(value: str) -> tuple[str, str]:
    height_cm = ""
    weight_kg = ""

    height_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", value)
    if height_match:
        height_cm = height_match.group(1)

    weight_match = re.search(r"(\d+(?:\.\d+)?)\s*kg", value)
    if weight_match:
        weight_kg = weight_match.group(1)

    return height_cm, weight_kg


def infer_event_gender(event: str) -> str:
    """
    Infer the event gender category from Olympedia event names.

    This is enough for most country/team gender-share analysis because most
    Olympic events are explicitly labelled Men, Women, or Mixed.
    """
    event = clean_text(event)
    if re.search(r"(?:^|,\s*)Mixed\b", event):
        return "Mixed"
    if re.search(r"(?:^|,\s*)Men\b", event):
        return "Men"
    if re.search(r"(?:^|,\s*)Women\b", event):
        return "Women"
    return "Open/Unknown"


def infer_sex_from_event_gender(event_gender: str) -> str:
    if event_gender == "Men":
        return "Male"
    if event_gender == "Women":
        return "Female"
    return ""


def collect_countries(cache_dir: Path, refresh_cache: bool) -> list[Country]:
    html = get_html(COUNTRIES_URL, cache_dir, refresh_cache=refresh_cache)
    if html is None:
        raise RuntimeError(f"Could not fetch {COUNTRIES_URL}")

    soup = soup_from_html(html)
    table_body = soup.find("tbody")
    if table_body is None:
        raise RuntimeError("Could not find countries table on Olympedia")

    countries: list[Country] = []
    for row in table_body.find_all("tr"):
        cells = row.find_all("td")
        links = row.find_all("a", href=True)
        if not cells or len(links) < 2:
            continue
        if not row.find(attrs={"class": "glyphicon glyphicon-ok"}):
            continue

        noc = clean_text(cells[0].get_text())
        if noc == "MIX":
            continue

        countries.append(
            Country(
                noc=noc,
                name=clean_text(links[1].get_text()),
                url=urljoin(BASE_URL, links[1]["href"]),
            )
        )

    return countries


def collect_games_info(cache_dir: Path, refresh_cache: bool) -> dict[str, GameInfo]:
    html = get_html(EDITIONS_URL, cache_dir, refresh_cache=refresh_cache)
    if html is None:
        raise RuntimeError(f"Could not fetch {EDITIONS_URL}")

    soup = soup_from_html(html)
    games: dict[str, GameInfo] = {}
    for table in soup.find_all("table")[:2]:
        heading = table.find_previous(["h1", "h2", "h3", "h4"])
        season = clean_text(heading.get_text()) if heading else ""
        if season not in {"Summer", "Winter"}:
            continue

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            links = row.find_all("a", href=True)
            if len(cells) < 5 or not links:
                continue

            year_text = clean_text(cells[1].get_text())
            if not year_text.isdigit():
                continue

            year = int(year_text)
            edition_id = extract_edition_id(links[0]["href"])
            if not edition_id:
                continue

            opened = clean_text(cells[4].get_text())
            start_date = parse_calendar_date(opened, default_year=year)
            games[edition_id] = GameInfo(
                edition_id=edition_id,
                game=f"{year} {season} Olympics",
                year=year,
                season=season,
                start_date=date_to_string(start_date),
            )

    return games


def collect_result_pages(
    country: Country,
    cache_dir: Path,
    *,
    start_year: int,
    end_year: int,
    seasons: set[str],
    games_info: dict[str, GameInfo],
    refresh_cache: bool,
    delay: float,
) -> list[ResultPage]:
    html = get_html(country.url, cache_dir, refresh_cache=refresh_cache, delay=delay)
    if html is None:
        return []

    soup = soup_from_html(html)
    table_body = soup.find("tbody")
    if table_body is None:
        return []

    pages: list[ResultPage] = []
    for row in table_body.find_all("tr"):
        cells = row.find_all("td")
        links = row.find_all("a", href=True)
        if not cells or len(links) < 2:
            continue

        game = clean_text(cells[0].get_text())
        year, season = parse_year_and_season(game)
        if year is None or season not in seasons or not (start_year <= year <= end_year):
            continue

        result_link = next(
            (link for link in links if f"/countries/{country.noc}/editions/" in link.get("href", "")),
            links[-1],
        )
        edition_id = extract_edition_id(result_link["href"])
        game_info = games_info.get(edition_id)
        pages.append(
            ResultPage(
                noc=country.noc,
                country=country.name,
                edition_id=edition_id,
                game=game,
                year=year,
                season=season,
                game_start_date=game_info.start_date if game_info else "",
                url=urljoin(BASE_URL, result_link["href"]),
            )
        )

    return pages


def parse_country_results_page(page: ResultPage, html: str) -> list[dict[str, str | int]]:
    soup = soup_from_html(html)
    table = soup.find("table")
    if table is None:
        return []

    records: list[dict[str, str | int]] = []
    current_sport = ""
    current_event = ""
    current_result_url = ""
    current_position = ""
    current_medal = ""

    for row in table.find_all("tr"):
        cells = row.find_all("td", recursive=False)
        if not cells:
            continue

        if len(cells) == 1:
            current_sport = clean_text(cells[0].get_text())
            current_event = ""
            current_result_url = ""
            current_position = ""
            current_medal = ""
            continue

        if len(cells) >= 4:
            event_cell, athlete_cell, position_cell, medal_cell = cells[:4]
            event = clean_text(event_cell.get_text())
            position = clean_text(position_cell.get_text())
            medal = clean_text(medal_cell.get_text())

            if event:
                current_event = event
                current_position = position
                current_medal = medal
                result_link = event_cell.find("a", href=re.compile(r"^/results/"))
                current_result_url = urljoin(BASE_URL, result_link["href"]) if result_link else ""
            else:
                position = position or current_position
                medal = medal or current_medal
        elif len(cells) >= 2 and current_event:
            # Team events such as relays often put the team result on one row
            # and the athlete links on a following short colspan row.
            athlete_cell = cells[1]
            position = current_position
            medal = current_medal
        else:
            continue

        athlete_links = athlete_cell.find_all("a", href=re.compile(r"^/athletes/"))
        if not athlete_links:
            continue

        team_or_entry = clean_text(athlete_cell.get_text(" "))
        for athlete_link in athlete_links:
            athlete_url = urljoin(BASE_URL, athlete_link["href"])
            records.append(
                {
                    "athlete_id": extract_athlete_id(athlete_url),
                    "athlete_name": clean_text(athlete_link.get_text()),
                    "athlete_url": athlete_url,
                    "noc": page.noc,
                    "country": page.country,
                    "edition_id": page.edition_id,
                    "year": page.year,
                    "season": page.season,
                    "game": page.game,
                    "game_start_date": page.game_start_date,
                    "sport": current_sport,
                    "event": current_event,
                    "event_gender": infer_event_gender(current_event),
                    "team_or_entry": team_or_entry,
                    "position": position,
                    "medal": medal,
                    "result_url": current_result_url,
                    "source_url": page.url,
                }
            )

    return records


def scrape_result_page(
    page: ResultPage,
    cache_dir: Path,
    *,
    refresh_cache: bool,
    delay: float,
) -> list[dict[str, str | int]]:
    html = get_html(page.url, cache_dir, refresh_cache=refresh_cache, delay=delay)
    if html is None:
        return []
    return parse_country_results_page(page, html)


def parse_athlete_bio(athlete_id: str, html: str) -> AthleteBio:
    soup = soup_from_html(html)
    sex = ""
    born = ""
    birth_date = ""
    height_cm = ""
    weight_kg = ""

    for row in soup.select(".biodata tr"):
        header = clean_text(row.find("th").get_text()) if row.find("th") else ""
        value = clean_text(row.find("td").get_text(" ")) if row.find("td") else ""

        if header == "Sex":
            sex = value
        elif header == "Born":
            born = value
            birth_date = date_to_string(parse_calendar_date(value))
        elif header == "Measurements":
            height_cm, weight_kg = parse_measurements(value)

    return AthleteBio(
        athlete_id=athlete_id,
        sex=sex,
        born=born,
        birth_date=birth_date,
        height_cm=height_cm,
        weight_kg=weight_kg,
    )


def scrape_athlete_bio(
    athlete_url: str,
    cache_dir: Path,
    *,
    refresh_cache: bool,
    delay: float,
) -> AthleteBio:
    athlete_id = extract_athlete_id(athlete_url)
    html = get_html(athlete_url, cache_dir, refresh_cache=refresh_cache, delay=delay)
    if html is None:
        return AthleteBio(
            athlete_id=athlete_id,
            sex="",
            born="",
            birth_date="",
            height_cm="",
            weight_kg="",
        )
    return parse_athlete_bio(athlete_id, html)


def collect_athlete_bios(
    rows: list[dict[str, str | int]],
    cache_dir: Path,
    *,
    refresh_cache: bool,
    delay: float,
    workers: int,
    bio_scope: str,
) -> dict[str, AthleteBio]:
    if bio_scope == "none":
        return {}

    if bio_scope == "mixed":
        rows = [
            row
            for row in rows
            if row.get("event_gender") in {"Mixed", "Open/Unknown"}
        ]

    athlete_urls = sorted({str(row["athlete_url"]) for row in rows if row.get("athlete_url")})
    print(f"Scraping {len(athlete_urls)} unique athlete biography pages.", flush=True)

    bios: dict[str, AthleteBio] = {}
    if workers <= 1:
        for index, athlete_url in enumerate(athlete_urls, start=1):
            bio = scrape_athlete_bio(athlete_url, cache_dir, refresh_cache=refresh_cache, delay=delay)
            bios[bio.athlete_id] = bio
            if index % 100 == 0:
                print(f"Scraped {index}/{len(athlete_urls)} athlete bios.", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    scrape_athlete_bio,
                    athlete_url,
                    cache_dir,
                    refresh_cache=refresh_cache,
                    delay=delay,
                ): athlete_url
                for athlete_url in athlete_urls
            }
            for index, future in enumerate(as_completed(futures), start=1):
                athlete_url = futures[future]
                try:
                    bio = future.result()
                    bios[bio.athlete_id] = bio
                except Exception as error:
                    print(f"Error parsing {athlete_url}: {error}", flush=True)
                if index % 100 == 0:
                    print(f"Scraped {index}/{len(athlete_urls)} athlete bios.", flush=True)

    return bios


def enrich_rows_with_bios(
    rows: list[dict[str, str | int]],
    bios: dict[str, AthleteBio],
) -> list[dict[str, str | int]]:
    enriched_rows: list[dict[str, str | int]] = []
    for row in rows:
        athlete_id = str(row.get("athlete_id", ""))
        bio = bios.get(
            athlete_id,
            AthleteBio(
                athlete_id=athlete_id,
                sex="",
                born="",
                birth_date="",
                height_cm="",
                weight_kg="",
            ),
        )
        enriched_row = {
            **row,
            "sex": bio.sex or infer_sex_from_event_gender(str(row.get("event_gender", ""))),
            "born": bio.born,
            "birth_date": bio.birth_date,
            "age": calculate_age_on_date(bio.birth_date, str(row.get("game_start_date", ""))),
            "height_cm": bio.height_cm,
            "weight_kg": bio.weight_kg,
        }
        enriched_rows.append(enriched_row)

    return enriched_rows


def write_csv(rows: Iterable[dict[str, str | int]], output_path: Path) -> int:
    fieldnames = [
        "athlete_id",
        "athlete_name",
        "athlete_url",
        "sex",
        "born",
        "birth_date",
        "age",
        "height_cm",
        "weight_kg",
        "noc",
        "country",
        "edition_id",
        "year",
        "season",
        "game",
        "game_start_date",
        "sport",
        "event",
        "event_gender",
        "team_or_entry",
        "position",
        "medal",
        "result_url",
        "source_url",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Olympedia athlete-event results into a CSV file."
    )
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--season",
        action="append",
        choices=["Summer", "Winter"],
        help="Season to scrape. Repeat for both. Defaults to Summer and Winter.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bio-workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("raw_data/olympedia_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/olympedia_athlete_results_1996_2024.csv"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--skip-bio",
        action="store_true",
        help="Skip athlete biography pages, leaving sex, age, height, and weight blank.",
    )
    parser.add_argument(
        "--bio-scope",
        choices=["all", "mixed", "none"],
        default="all",
        help=(
            "Which athlete bios to scrape. 'mixed' only scrapes bios for Mixed/Open/Unknown "
            "events; Men/Women event rows infer sex from event names."
        ),
    )
    parser.add_argument(
        "--noc",
        action="append",
        help="Only scrape this NOC code. Repeat for multiple countries, e.g. --noc USA --noc AUS.",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=0,
        help="Only scrape the first N country-Games pages. Useful for testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = set(args.season or ["Summer", "Winter"])

    games_info = collect_games_info(args.cache_dir, args.refresh_cache)
    countries = collect_countries(args.cache_dir, args.refresh_cache)
    if args.noc:
        requested_nocs = {noc.upper() for noc in args.noc}
        countries = [country for country in countries if country.noc.upper() in requested_nocs]

    print(f"Found {len(countries)} countries.", flush=True)

    result_pages: list[ResultPage] = []
    for index, country in enumerate(countries, start=1):
        pages = collect_result_pages(
            country,
            args.cache_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            seasons=seasons,
            games_info=games_info,
            refresh_cache=args.refresh_cache,
            delay=args.delay,
        )
        result_pages.extend(pages)
        if index % 25 == 0:
            print(f"Checked {index}/{len(countries)} countries; found {len(result_pages)} result pages.", flush=True)

    result_pages = sorted(result_pages, key=lambda page: (page.year, page.season, page.noc))
    if args.limit_pages:
        result_pages = result_pages[: args.limit_pages]

    print(f"Scraping {len(result_pages)} country-Games result pages.", flush=True)

    all_rows: list[dict[str, str | int]] = []
    if args.workers <= 1:
        for index, page in enumerate(result_pages, start=1):
            all_rows.extend(
                scrape_result_page(page, args.cache_dir, refresh_cache=args.refresh_cache, delay=args.delay)
            )
            if index % 25 == 0:
                print(f"Scraped {index}/{len(result_pages)} pages; collected {len(all_rows)} rows.", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    scrape_result_page,
                    page,
                    args.cache_dir,
                    refresh_cache=args.refresh_cache,
                    delay=args.delay,
                ): page
                for page in result_pages
            }
            for index, future in enumerate(as_completed(futures), start=1):
                page = futures[future]
                try:
                    all_rows.extend(future.result())
                except Exception as error:
                    print(f"Error parsing {page.url}: {error}", flush=True)
                if index % 25 == 0:
                    print(f"Scraped {index}/{len(result_pages)} pages; collected {len(all_rows)} rows.", flush=True)

    bio_scope = "none" if args.skip_bio else args.bio_scope
    if bio_scope == "none":
        all_rows = enrich_rows_with_bios(all_rows, {})
    else:
        bios = collect_athlete_bios(
            all_rows,
            args.cache_dir,
            refresh_cache=args.refresh_cache,
            delay=args.delay,
            workers=args.bio_workers,
            bio_scope=bio_scope,
        )
        all_rows = enrich_rows_with_bios(all_rows, bios)

    row_count = write_csv(all_rows, args.output)
    print(f"Wrote {row_count} athlete-event rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
