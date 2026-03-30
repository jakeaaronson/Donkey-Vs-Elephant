"""
Load and process the Stanford Congressional Record dataset.

Downloads the hein-daily dataset (97th-114th Congress, 1981-2016) from
Stanford's servers, parses the pipe-delimited files, joins speeches with
speaker metadata (including party), and outputs JSONL files split by party.

Source: Gentzkow, Shapiro, & Taddy — Congressional Record for the 43rd-114th Congresses
https://data.stanford.edu/congress_text
"""

import csv
import json
import logging
import os
import sys

# Congressional speeches can be very long; raise the CSV field size limit
csv.field_size_limit(sys.maxsize)
import zipfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

DOWNLOAD_URL = "https://stacks.stanford.edu/file/druid:md374tz9962/hein-daily.zip"
ZIP_FILENAME = "hein-daily.zip"

# hein-daily covers 97th through 114th Congress
CONGRESS_RANGE = range(97, 115)


def download_dataset(output_dir: str | Path, force: bool = False) -> Path:
    """Download hein-daily.zip if not already present.

    Args:
        output_dir: Directory to save the zip file.
        force: Re-download even if file exists.

    Returns:
        Path to the downloaded zip file.
    """
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / ZIP_FILENAME

    if zip_path.exists() and not force:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:  # sanity check — full file is ~2.8 GB
            logger.info("Dataset already downloaded: %s (%.0f MB)", zip_path, size_mb)
            return zip_path
        else:
            logger.warning("Incomplete download detected (%.0f MB), re-downloading", size_mb)

    logger.info("Downloading hein-daily.zip (~2.8 GB) from Stanford...")
    resp = requests.get(DOWNLOAD_URL, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="hein-daily.zip"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Download complete: %s", zip_path)
    return zip_path


def extract_dataset(zip_path: str | Path, extract_dir: str | Path) -> Path:
    """Extract the zip file if not already extracted.

    Handles the nested directory structure (zip contains hein-daily/files).

    Returns:
        Path to the directory containing the actual data files.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)

    # Check if already extracted by looking for a known file
    # Handle both flat and nested extraction
    for candidate in [extract_dir, extract_dir / "hein-daily"]:
        if (candidate / "speeches_114.txt").exists():
            logger.info("Dataset already extracted in %s", candidate)
            return candidate

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s to %s ...", zip_path.name, extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # The zip nests files under hein-daily/ — return the actual data dir
    nested = extract_dir / "hein-daily"
    if nested.exists() and (nested / "speeches_114.txt").exists():
        logger.info("Extraction complete (nested under hein-daily/)")
        return nested

    logger.info("Extraction complete")
    return extract_dir


def load_speaker_map(extract_dir: Path, congress: int) -> dict[int, dict]:
    """Load the SpeakerMap for a given congress number.

    Returns:
        Dict mapping speech_id -> {party, lastname, firstname, chamber, state, gender}
    """
    filename = extract_dir / f"{congress}_SpeakerMap.txt"
    if not filename.exists():
        logger.warning("SpeakerMap not found: %s", filename)
        return {}

    mapping: dict[int, dict] = {}
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            try:
                speech_id = int(row["speech_id"])
            except (ValueError, KeyError):
                continue
            mapping[speech_id] = {
                "party": row.get("party", "").strip(),
                "lastname": row.get("lastname", "").strip(),
                "firstname": row.get("firstname", "").strip(),
                "chamber": row.get("chamber", "").strip(),
                "state": row.get("state", "").strip(),
                "gender": row.get("gender", "").strip(),
            }

    logger.debug("Congress %d: loaded %d speaker entries", congress, len(mapping))
    return mapping


def load_speeches(extract_dir: Path, congress: int) -> dict[int, str]:
    """Load speeches for a given congress number.

    Returns:
        Dict mapping speech_id -> speech text
    """
    filename = extract_dir / f"speeches_{congress}.txt"
    if not filename.exists():
        logger.warning("Speeches file not found: %s", filename)
        return {}

    speeches: dict[int, str] = {}
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            try:
                speech_id = int(row["speech_id"])
            except (ValueError, KeyError):
                continue
            text = row.get("speech", "").strip()
            if text:
                speeches[speech_id] = text

    logger.debug("Congress %d: loaded %d speeches", congress, len(speeches))
    return speeches


def process_congress(
    extract_dir: Path,
    congress: int,
    dem_file,
    rep_file,
    min_words: int = 50,
) -> dict[str, int]:
    """Process a single congress: join speeches with speaker map, write to party files.

    Args:
        extract_dir: Path containing extracted Stanford files.
        congress: Congress number (e.g. 114).
        dem_file: Open file handle for democratic.jsonl.
        rep_file: Open file handle for republican.jsonl.
        min_words: Minimum word count to include a speech.

    Returns:
        Stats dict with counts.
    """
    stats = {"total": 0, "democratic": 0, "republican": 0, "other": 0, "too_short": 0}

    speaker_map = load_speaker_map(extract_dir, congress)
    speeches = load_speeches(extract_dir, congress)

    if not speaker_map or not speeches:
        return stats

    for speech_id, text in speeches.items():
        stats["total"] += 1

        speaker = speaker_map.get(speech_id)
        if not speaker:
            stats["other"] += 1
            continue

        party = speaker["party"]
        if party not in ("D", "R"):
            stats["other"] += 1
            continue

        # Filter very short speeches (procedural noise)
        if len(text.split()) < min_words:
            stats["too_short"] += 1
            continue

        record = {
            "text": text,
            "speaker": f"{speaker['firstname']} {speaker['lastname']}".strip(),
            "party": "Democratic" if party == "D" else "Republican",
            "chamber": speaker["chamber"],
            "state": speaker["state"],
            "congress": congress,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"

        if party == "D":
            dem_file.write(line)
            stats["democratic"] += 1
        else:
            rep_file.write(line)
            stats["republican"] += 1

    return stats


def process_all(
    extract_dir: str | Path,
    output_dir: str | Path,
    congresses: Optional[range] = None,
    min_words: int = 50,
) -> dict[str, int]:
    """Process all congresses and write party-split JSONL files.

    Args:
        extract_dir: Path containing extracted Stanford files.
        output_dir: Where to write democratic.jsonl and republican.jsonl.
        congresses: Range of congress numbers to process. Defaults to 97-114.
        min_words: Minimum word count to include a speech.

    Returns:
        Aggregate stats dict.
    """
    extract_dir = Path(extract_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    congresses = congresses or CONGRESS_RANGE
    totals = {"total": 0, "democratic": 0, "republican": 0, "other": 0, "too_short": 0}

    dem_path = output_dir / "democratic.jsonl"
    rep_path = output_dir / "republican.jsonl"

    with open(dem_path, "w", encoding="utf-8") as dem_f, \
         open(rep_path, "w", encoding="utf-8") as rep_f:

        for congress in tqdm(list(congresses), desc="Processing congresses", unit="congress"):
            stats = process_congress(extract_dir, congress, dem_f, rep_f, min_words)
            for key in totals:
                totals[key] += stats[key]
            logger.info(
                "Congress %d: %d D, %d R, %d other, %d short",
                congress,
                stats["democratic"],
                stats["republican"],
                stats["other"],
                stats["too_short"],
            )

    logger.info("Total: %s", totals)
    logger.info("Output: %s, %s", dem_path, rep_path)
    return totals
