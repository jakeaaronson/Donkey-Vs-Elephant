# data_collection

Fetches Congressional Record speeches from the [Congress.gov API](https://api.congress.gov/) and splits them into Democratic and Republican training sets.

## Setup

1. Get a free API key at https://api.congress.gov/sign-up/
2. Export it:
   ```bash
   export CONGRESS_GOV_API_KEY="your-key-here"
   ```

## Usage

Run the full pipeline (fetch + classify):

```bash
python -m data_collection.run_collection
```

### Options

| Flag | Description |
|------|-------------|
| `--start-year` / `--end-year` | Year range to collect (default: 2000-2024) |
| `--fetch-only` | Download raw data without classifying |
| `--classify-only` | Classify already-downloaded data |
| `--no-text` | Skip full article text (metadata only, much faster) |
| `--api-key` | Pass API key directly instead of env var |
| `-v` | Verbose/debug logging |

### Examples

```bash
# Quick test with a single year, no full text
python -m data_collection.run_collection --start-year 2024 --end-year 2024 --no-text

# Fetch everything, then classify separately
python -m data_collection.run_collection --fetch-only
python -m data_collection.run_collection --classify-only
```

## Output

```
data/
  raw/
    cr_2000.json      # Raw article records per year
    cr_2001.json
    ...
    members.json       # Cached member-to-party lookup
  processed/
    democratic.jsonl   # One speech per line, ready for training
    republican.jsonl
```

Each JSONL line contains:

```json
{
  "text": "Plain text of the speech...",
  "speaker": "Mr. SMITH",
  "party": "Republican",
  "date": "2024-01-15T05:00:00Z",
  "section": "Senate"
}
```

## How it works

1. **Fetch**: Iterates through daily Congressional Record issues via the API, downloading articles from House and Senate sections. Saves per-year JSON files so collection can resume after interruption.

2. **Classify**: Builds a name-to-party index from the Members API (congresses 106-118). Extracts speaker names from article titles using pattern matching on Congressional Record conventions (`Mr. LASTNAME`, `Mrs. LASTNAME of State`, etc.). Strips HTML to plain text and writes party-labeled JSONL.

## Rate limits

The API allows roughly 1,000 requests per hour. The scraper uses a 0.5s delay between requests by default. A full 2000-2024 collection with text will take several days -- use `--no-text` for a faster metadata-only pass first.
