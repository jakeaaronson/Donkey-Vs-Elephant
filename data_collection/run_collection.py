"""
CLI entry point for the Congressional Record data collection pipeline.

Usage:
    # Full pipeline: fetch raw data, then classify by party
    python -m data_collection.run_collection

    # Fetch only (skip classification)
    python -m data_collection.run_collection --fetch-only

    # Classify only (assumes raw data already exists)
    python -m data_collection.run_collection --classify-only

    # Custom year range
    python -m data_collection.run_collection --start-year 2020 --end-year 2024

    # Skip downloading full article text (metadata only, much faster)
    python -m data_collection.run_collection --no-text
"""

import argparse
import logging
import sys
import time

from data_collection.congress_api import CongressAPI, DEFAULT_END_YEAR, DEFAULT_START_YEAR
from data_collection.party_classifier import PartyClassifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and classify Congressional Record speeches by party.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"First year to collect (default: {DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"Last year to collect (default: {DEFAULT_END_YEAR})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Congress.gov API key (default: $CONGRESS_GOV_API_KEY)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory for raw JSON output (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory for classified JSONL output (default: data/processed)",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch raw data, skip classification",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Only classify existing raw data, skip fetching",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip downloading full article text (metadata only)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def run_fetch(args: argparse.Namespace) -> int:
    """Fetch raw Congressional Record data from the API."""
    print(f"\n{'='*60}")
    print(f"  Fetching Congressional Record: {args.start_year}-{args.end_year}")
    print(f"  Output: {args.raw_dir}/")
    print(f"  Full text: {'no' if args.no_text else 'yes'}")
    print(f"{'='*60}\n")

    api = CongressAPI(
        api_key=args.api_key,
        output_dir=args.raw_dir,
    )

    start = time.time()
    records = api.collect(
        start_year=args.start_year,
        end_year=args.end_year,
        fetch_text=not args.no_text,
    )
    elapsed = time.time() - start

    print(f"\nFetch complete: {len(records):,} articles in {elapsed:.0f}s")
    return len(records)


def run_classify(args: argparse.Namespace) -> dict[str, int]:
    """Classify fetched records by party affiliation."""
    print(f"\n{'='*60}")
    print(f"  Classifying speeches by party")
    print(f"  Input:  {args.raw_dir}/")
    print(f"  Output: {args.processed_dir}/")
    print(f"{'='*60}\n")

    classifier = PartyClassifier(
        raw_dir=args.raw_dir,
        output_dir=args.processed_dir,
        api_key=args.api_key,
    )

    start = time.time()
    stats = classifier.classify()
    elapsed = time.time() - start

    print(f"\nClassification complete in {elapsed:.0f}s")
    return stats


def print_stats(stats: dict[str, int]) -> None:
    """Print a summary table of classification results."""
    total = stats.get("total", 0)
    dem = stats.get("democratic", 0)
    rep = stats.get("republican", 0)
    other = stats.get("other", 0)
    unresolved = stats.get("unresolved", 0)
    classified = dem + rep

    print(f"\n{'='*60}")
    print(f"  Classification Results")
    print(f"{'='*60}")
    print(f"  Total articles processed:  {total:>8,}")
    print(f"  Democratic speeches:       {dem:>8,}")
    print(f"  Republican speeches:       {rep:>8,}")
    print(f"  Other party:               {other:>8,}")
    print(f"  Unresolved/skipped:        {unresolved:>8,}")
    if total > 0:
        print(f"  Classification rate:       {classified / total:>8.1%}")
    print(f"{'='*60}\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.fetch_only and args.classify_only:
        print("Error: --fetch-only and --classify-only are mutually exclusive.")
        sys.exit(1)

    stats: dict[str, int] = {}

    if not args.classify_only:
        run_fetch(args)

    if not args.fetch_only:
        stats = run_classify(args)
        print_stats(stats)

    print("Done.")


if __name__ == "__main__":
    main()
