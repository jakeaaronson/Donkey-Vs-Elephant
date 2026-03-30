"""
Fetch Congressional Record speeches from the Congress.gov API.

Retrieves daily Congressional Record issues, extracts individual articles
(speeches/remarks) from House and Senate sections, and saves raw JSON
to data/raw/ for downstream processing.

Requires a Congress.gov API key: https://api.congress.gov/sign-up/
Set it as the CONGRESS_GOV_API_KEY environment variable.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://api.congress.gov/v3"
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2024
RATE_LIMIT_DELAY = 0.5  # seconds between requests (API allows ~1000/hr)
PAGE_SIZE = 250  # max allowed by the API


class CongressAPI:
    """Client for the Congress.gov API focused on Congressional Record text."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "data/raw",
        rate_limit_delay: float = RATE_LIMIT_DELAY,
    ) -> None:
        self.api_key = api_key or os.environ.get("CONGRESS_GOV_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set CONGRESS_GOV_API_KEY or pass api_key=."
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key, "format": "json"}  # type: ignore[assignment]

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """Make a rate-limited GET request and return parsed JSON."""
        time.sleep(self.rate_limit_delay)
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get_paginated(self, url: str, result_key: str) -> list[dict]:
        """Follow pagination links to collect all results for a given key."""
        all_items: list[dict] = []
        params = {"limit": PAGE_SIZE, "offset": 0}

        while True:
            data = self._get(url, params=params)
            items = data.get(result_key, [])
            all_items.extend(items)

            next_url = data.get("pagination", {}).get("next")
            if not next_url or not items:
                break
            # The 'next' URL already includes query params; use it directly.
            url = next_url
            params = {}

        return all_items

    # ------------------------------------------------------------------
    # Congressional Record issues
    # ------------------------------------------------------------------

    def fetch_issues_for_year(self, year: int) -> list[dict]:
        """Get all daily Congressional Record issues for a given year."""
        url = f"{BASE_URL}/daily-congressional-record"
        all_issues: list[dict] = []
        offset = 0

        while True:
            data = self._get(url, {"limit": PAGE_SIZE, "offset": offset, "y": year})
            issues = data.get("dailyCongressionalRecord", [])
            all_issues.extend(issues)
            if not data.get("pagination", {}).get("next") or not issues:
                break
            offset += PAGE_SIZE

        logger.info("Year %d: found %d issues", year, len(all_issues))
        return all_issues

    def fetch_articles_for_issue(
        self, volume: int, issue: str
    ) -> list[dict]:
        """Fetch all articles (speeches) from a single issue."""
        url = f"{BASE_URL}/daily-congressional-record/{volume}/{issue}/articles"
        return self._get_paginated(url, "articles")

    def fetch_article_text(self, text_url: str) -> Optional[str]:
        """Download the formatted text (HTML) for a single article."""
        try:
            time.sleep(self.rate_limit_delay)
            resp = self.session.get(text_url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            logger.warning("Failed to fetch article text from %s: %s", text_url, exc)
            return None

    # ------------------------------------------------------------------
    # Full collection pipeline
    # ------------------------------------------------------------------

    def collect(
        self,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        fetch_text: bool = True,
    ) -> list[dict]:
        """
        Main entry point: iterate over years, issues, and articles.

        Saves one JSON file per year to data/raw/cr_{year}.json containing
        a list of article records with metadata and (optionally) full text.

        Returns the flat list of all article records collected.
        """
        all_records: list[dict] = []

        for year in range(start_year, end_year + 1):
            year_file = self.output_dir / f"cr_{year}.json"
            if year_file.exists():
                logger.info("Skipping year %d (already downloaded)", year)
                with open(year_file) as f:
                    all_records.extend(json.load(f))
                continue

            issues = self.fetch_issues_for_year(year)
            year_records: list[dict] = []

            for issue in tqdm(issues, desc=f"CR {year}", unit="issue"):
                volume = issue["volumeNumber"]
                issue_num = issue["issueNumber"]
                issue_date = issue.get("issueDate", "")

                try:
                    sections = self.fetch_articles_for_issue(volume, issue_num)
                except requests.HTTPError as exc:
                    logger.warning(
                        "Skipping issue %s/%s: %s", volume, issue_num, exc
                    )
                    continue

                for section in sections:
                    section_name = section.get("name", "")
                    # We only care about House and Senate floor proceedings
                    if not any(
                        chamber in section_name.lower()
                        for chamber in ("senate", "house")
                    ):
                        continue

                    for article in section.get("sectionArticles", []):
                        record = {
                            "volume": volume,
                            "issue": issue_num,
                            "date": issue_date,
                            "section": section_name,
                            "title": article.get("title", ""),
                            "start_page": article.get("startPage", ""),
                            "end_page": article.get("endPage", ""),
                        }

                        if fetch_text:
                            text_entries = article.get("text", [])
                            html_url = next(
                                (
                                    t["url"]
                                    for t in text_entries
                                    if t.get("type") == "Formatted Text"
                                ),
                                None,
                            )
                            if html_url:
                                record["text_url"] = html_url
                                record["text_html"] = self.fetch_article_text(html_url)

                        year_records.append(record)

            # Persist per-year so we can resume after interruption
            with open(year_file, "w") as f:
                json.dump(year_records, f, indent=2)
            logger.info(
                "Year %d: saved %d article records to %s",
                year,
                len(year_records),
                year_file,
            )

            all_records.extend(year_records)

        return all_records
