"""
CLI entry point for the Congressional Record data collection pipeline.

Uses the Stanford Congressional Record dataset (Gentzkow, Shapiro, & Taddy)
which provides pre-parsed speeches with speaker metadata and party affiliation
for the 97th-114th Congresses (1981-2016).

Usage:
    # Full pipeline: download, extract, process
    python -m data_collection.run_collection

    # Process only (assumes data already downloaded)
    python -m data_collection.run_collection --process-only

    # Download only
    python -m data_collection.run_collection --download-only

    # Custom congress range
    python -m data_collection.run_collection --start-congress 110 --end-congress 114
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from data_collection.stanford_loader import (
    CONGRESS_RANGE,
    download_dataset,
    extract_dataset,
    process_all,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and process Stanford Congressional Record dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-congress",
        type=int,
        default=CONGRESS_RANGE.start,
        help=f"First congress to process (default: {CONGRESS_RANGE.start})",
    )
    parser.add_argument(
        "--end-congress",
        type=int,
        default=CONGRESS_RANGE.stop - 1,
        help=f"Last congress to process (default: {CONGRESS_RANGE.stop - 1})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Minimum word count per speech (default: 50)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dataset, skip processing",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process existing data, skip download",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def print_stats(stats: dict[str, int]) -> None:
    """Print a summary table of processing results."""
    total = stats.get("total", 0)
    dem = stats.get("democratic", 0)
    rep = stats.get("republican", 0)
    other = stats.get("other", 0)
    too_short = stats.get("too_short", 0)

    print(f"\n{'='*60}")
    print(f"  Stanford Congressional Record — Processing Results")
    print(f"{'='*60}")
    print(f"  Total speeches found:      {total:>10,}")
    print(f"  Democratic speeches:       {dem:>10,}")
    print(f"  Republican speeches:       {rep:>10,}")
    print(f"  Other/unresolved:          {other:>10,}")
    print(f"  Filtered (too short):      {too_short:>10,}")
    if total > 0:
        classified = dem + rep
        print(f"  Classification rate:       {classified / total:>10.1%}")
    print(f"{'='*60}\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.download_only and args.process_only:
        print("Error: --download-only and --process-only are mutually exclusive.")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    stanford_dir = data_dir / "stanford"
    extract_dir = stanford_dir / "hein-daily"
    processed_dir = data_dir / "processed"

    # Download
    if not args.process_only:
        print(f"\n{'='*60}")
        print(f"  Downloading Stanford Congressional Record dataset")
        print(f"  Output: {stanford_dir}/")
        print(f"{'='*60}\n")

        start = time.time()
        zip_path = download_dataset(stanford_dir)
        elapsed = time.time() - start
        print(f"Download step complete in {elapsed:.0f}s")

        # Extract — extract_dataset returns the actual data directory
        print(f"\nExtracting to {extract_dir}/ ...")
        extract_dir = extract_dataset(zip_path, extract_dir)
    else:
        # When process-only, find the actual data dir
        nested = extract_dir / "hein-daily"
        if nested.exists() and (nested / "speeches_114.txt").exists():
            extract_dir = nested

    if args.download_only:
        print("Done (download only).")
        return

    # Process
    congress_range = range(args.start_congress, args.end_congress + 1)
    print(f"\n{'='*60}")
    print(f"  Processing congresses {args.start_congress}-{args.end_congress}")
    print(f"  Min words per speech: {args.min_words}")
    print(f"  Output: {processed_dir}/")
    print(f"{'='*60}\n")

    start = time.time()
    stats = process_all(
        extract_dir=extract_dir,
        output_dir=processed_dir,
        congresses=congress_range,
        min_words=args.min_words,
    )
    elapsed = time.time() - start

    print_stats(stats)
    print(f"Processing complete in {elapsed:.0f}s")

    # Print file sizes
    for name in ("democratic.jsonl", "republican.jsonl"):
        path = processed_dir / name
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {size_mb:.1f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
