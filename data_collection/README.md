# data_collection

Downloads and processes the [Stanford Congressional Record dataset](https://data.stanford.edu/congress_text) (Gentzkow, Shapiro, & Taddy) to produce party-labeled training data.

The Stanford dataset provides pre-parsed speeches from the 97th–114th Congresses (1981–2016) with speaker metadata including **party affiliation** already resolved — no fragile name-matching or API rate limits to deal with.

## Usage

Run the full pipeline (download + extract + process):

```bash
python -m data_collection.run_collection
```

The first run downloads `hein-daily.zip` (~2.8 GB) from Stanford's servers.

### Options

| Flag | Description |
|------|-------------|
| `--start-congress` / `--end-congress` | Congress range to process (default: 97-114) |
| `--min-words` | Minimum word count per speech (default: 50) |
| `--download-only` | Download and extract without processing |
| `--process-only` | Process already-downloaded data |
| `-v` | Verbose/debug logging |

### Examples

```bash
# Just the most recent congresses
python -m data_collection.run_collection --start-congress 110 --end-congress 114

# Download first, process later
python -m data_collection.run_collection --download-only
python -m data_collection.run_collection --process-only
```

## Output

```
data/
  stanford/
    hein-daily.zip         # Downloaded archive (2.8 GB)
    hein-daily/            # Extracted files
      speeches_097.txt     # Pipe-delimited speech text
      097_SpeakerMap.txt   # Speaker metadata with party
      ...
  processed/
    democratic.jsonl       # Dem speeches, ready for training
    republican.jsonl       # GOP speeches, ready for training
```

Each JSONL line contains:

```json
{
  "text": "Full text of the speech...",
  "speaker": "JOSEPH BIDEN",
  "party": "Democratic",
  "chamber": "S",
  "state": "DE",
  "congress": 98
}
```

## Data source

The Stanford dataset was compiled from the official Congressional Record using automated parsing of the printed/digitized volumes. Speaker attribution was resolved against the `congress-legislators` database with fuzzy matching on name, chamber, gender, state, and district.

- **Coverage**: 97th–114th Congress (1981–2016)
- **Parser accuracy**: 99.7% correct speech boundary detection (daily edition)
- **Speaker matching**: 92% name agreement rate (daily edition)
- **Citation**: Gentzkow, M., Shapiro, J.M. and Taddy, M., 2019. "Measuring Group Differences in High-Dimensional Choices: Method and Application to Congressional Speech." *Econometrica*, 87(4), pp.1307-1340.

## Legacy: Congress.gov API scraper

The original `congress_api.py` and `party_classifier.py` are retained for reference. They use the Congress.gov API to fetch recent data (post-2016), but are fragile due to:
- No inline text in API responses (requires separate HTML fetches)
- No speaker/party metadata in the API (requires name-matching heuristics)
- Aggressive rate limits on the API
