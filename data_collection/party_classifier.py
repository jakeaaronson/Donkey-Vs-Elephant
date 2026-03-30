"""
Classify Congressional Record speeches by party affiliation.

Uses the Congress.gov Members API to build a lookup table of
congressmember names to party, then matches speakers found in
article titles/text. Outputs Democratic and Republican speeches
as separate JSONL files for model training.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://api.congress.gov/v3"
PAGE_SIZE = 250
RATE_LIMIT_DELAY = 0.5

# Congress numbers that cover 2000-2024 (106th through 118th)
CONGRESS_RANGE = range(106, 119)


class MemberLookup:
    """Build and query a name-to-party mapping from the Members API."""

    def __init__(self, api_key: Optional[str] = None, cache_path: str = "data/raw/members.json") -> None:
        self.api_key = api_key or os.environ.get("CONGRESS_GOV_API_KEY")
        if not self.api_key:
            raise ValueError("No API key. Set CONGRESS_GOV_API_KEY.")
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key, "format": "json"}  # type: ignore[assignment]

        # Keyed by normalized last name -> list of (full_name, party)
        self._by_last_name: dict[str, list[tuple[str, str]]] = {}
        # Keyed by "LAST, FIRST" normalized -> party
        self._by_full_name: dict[str, str] = {}

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        time.sleep(RATE_LIMIT_DELAY)
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_all_members(self) -> list[dict]:
        """Download member data across all relevant congresses."""
        if self.cache_path.exists():
            logger.info("Loading cached member data from %s", self.cache_path)
            with open(self.cache_path) as f:
                return json.load(f)

        all_members: list[dict] = []
        seen_ids: set[str] = set()

        for congress in tqdm(CONGRESS_RANGE, desc="Fetching members", unit="congress"):
            url = f"{BASE_URL}/member/congress/{congress}"
            offset = 0

            while True:
                data = self._get(url, {"limit": PAGE_SIZE, "offset": offset})
                members = data.get("members", [])

                for m in members:
                    bio_id = m.get("bioguideId", "")
                    if bio_id and bio_id not in seen_ids:
                        seen_ids.add(bio_id)
                        all_members.append(m)

                if not data.get("pagination", {}).get("next") or not members:
                    break
                offset += PAGE_SIZE

        with open(self.cache_path, "w") as f:
            json.dump(all_members, f, indent=2)
        logger.info("Cached %d unique members to %s", len(all_members), self.cache_path)
        return all_members

    def build_index(self) -> None:
        """Build the name-to-party lookup from member data."""
        members = self.fetch_all_members()

        for m in members:
            party = m.get("partyName", "").strip()
            name = m.get("name", "").strip()
            if not party or not name:
                continue

            # API returns names as "Last, First" or "Last, First Middle"
            normalized = name.upper()
            self._by_full_name[normalized] = party

            last = name.split(",")[0].strip().upper()
            self._by_last_name.setdefault(last, []).append((name, party))

        logger.info(
            "Member index: %d full names, %d unique last names",
            len(self._by_full_name),
            len(self._by_last_name),
        )

    def lookup(self, speaker_name: str) -> Optional[str]:
        """
        Try to resolve a speaker name to a party.

        Handles formats commonly found in Congressional Record titles:
          - "Mr. SMITH" / "Mrs. PELOSI" / "Ms. WARREN"
          - "Mr. SMITH of Ohio"
          - "SMITH" (last name only)

        Returns 'Democratic', 'Republican', or None if ambiguous/unknown.
        """
        # Strip honorifics and state qualifiers
        cleaned = re.sub(
            r"^(Mr\.|Mrs\.|Ms\.|Miss|Dr\.|The SPEAKER|The PRESIDING OFFICER)\s*",
            "",
            speaker_name,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+of\s+\w+.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip().upper()

        if not cleaned:
            return None

        # Check full-name matches first (unlikely from CR titles, but covers edge cases)
        if cleaned in self._by_full_name:
            return self._by_full_name[cleaned]

        # Last-name lookup
        candidates = self._by_last_name.get(cleaned, [])
        if len(candidates) == 1:
            return candidates[0][1]

        # Multiple members share the last name -- can't resolve without more context
        if len(candidates) > 1:
            parties = {party for _, party in candidates}
            if len(parties) == 1:
                # All members with this last name are the same party
                return parties.pop()
            logger.debug(
                "Ambiguous last name '%s' maps to %d members across parties",
                cleaned,
                len(candidates),
            )

        return None


class PartyClassifier:
    """Split raw Congressional Record articles into party-labeled JSONL."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        output_dir: str = "data/processed",
        api_key: Optional[str] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lookup = MemberLookup(api_key=api_key, cache_path=str(self.raw_dir / "members.json"))

        self.stats: dict[str, int] = {
            "total": 0,
            "democratic": 0,
            "republican": 0,
            "other": 0,
            "unresolved": 0,
        }

    def _extract_speaker_from_title(self, title: str) -> Optional[str]:
        """
        Pull a speaker name from a CR article title.

        Common title patterns:
          - "TRIBUTE TO JOHN SMITH" (not a floor speech, skip)
          - "RECOGNIZING..." (not attributable, skip)
          - "Mr. SMITH" or "Mrs. PELOSI" at the start of the title
          - Sometimes the title IS the speaker name
        """
        # Match honorific + NAME patterns
        match = re.match(
            r"^((?:Mr\.|Mrs\.|Ms\.|Miss|Dr\.)\s+[A-Z][A-Z'-]+(?:\s+of\s+\w+)?)",
            title,
        )
        if match:
            return match.group(1)

        # Some titles are just an all-caps last name (Senate style)
        if re.match(r"^[A-Z][A-Z'-]+$", title.strip()):
            return title.strip()

        return None

    def _extract_plain_text(self, html: Optional[str]) -> str:
        """Strip HTML tags to get plain speech text."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator=" ", strip=True)

    def _load_raw_records(self) -> list[dict]:
        """Load all cr_*.json files from the raw directory."""
        records: list[dict] = []
        for path in sorted(self.raw_dir.glob("cr_*.json")):
            with open(path) as f:
                records.extend(json.load(f))
        return records

    def classify(self) -> dict[str, int]:
        """
        Run the full classification pipeline.

        Reads raw JSON records, resolves speakers to parties, and writes
        data/processed/democratic.jsonl and data/processed/republican.jsonl.

        Each output line is a JSON object with keys:
          - text: plain text of the speech
          - speaker: resolved speaker name
          - party: Democratic or Republican
          - date: date of the record
          - section: House or Senate section
        """
        self.lookup.build_index()
        records = self._load_raw_records()

        dem_path = self.output_dir / "democratic.jsonl"
        rep_path = self.output_dir / "republican.jsonl"

        with open(dem_path, "w") as dem_f, open(rep_path, "w") as rep_f:
            for record in tqdm(records, desc="Classifying", unit="article"):
                self.stats["total"] += 1
                title = record.get("title", "")
                speaker = self._extract_speaker_from_title(title)

                if not speaker:
                    self.stats["unresolved"] += 1
                    continue

                party = self.lookup.lookup(speaker)
                if not party:
                    self.stats["unresolved"] += 1
                    continue

                plain_text = self._extract_plain_text(record.get("text_html"))
                if not plain_text or len(plain_text) < 50:
                    # Skip very short entries (procedural noise)
                    self.stats["unresolved"] += 1
                    continue

                entry = {
                    "text": plain_text,
                    "speaker": speaker,
                    "party": party,
                    "date": record.get("date", ""),
                    "section": record.get("section", ""),
                }
                line = json.dumps(entry, ensure_ascii=False) + "\n"

                if party == "Democratic":
                    dem_f.write(line)
                    self.stats["democratic"] += 1
                elif party == "Republican":
                    rep_f.write(line)
                    self.stats["republican"] += 1
                else:
                    self.stats["other"] += 1

        logger.info("Classification stats: %s", self.stats)
        return self.stats
