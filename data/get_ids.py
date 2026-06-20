import argparse
from pathlib import Path

import pandas as pd


def parse_ids(input_csv: Path, output_csv: Path, limit: int | None) -> None:
    """
    Reads the 'id' column from the input CSV, parses each entry into
    (owner/repo, pr_number), and writes unique rows to the output CSV.
    If limit is provided, only the first `limit` parsed rows are written.
    """
    try:
        df = pd.read_csv(input_csv, usecols=["id"])
    except Exception as e:
        print(f"Failed to read CSV '{input_csv}': {e}")
        return

    parsed_rows = []
    for row in df["id"]:
        try:
            parts = row.rsplit("_", 1)
            if len(parts) != 2:
                continue

            repo_full = parts[0].replace("_", "/")
            repo_name = "/".join(repo_full.split("/")[:2])  # owner/repo
            pr_number = int(parts[1])
            parsed_rows.append((repo_name, pr_number))
        except Exception:
            continue

    if not parsed_rows:
        print("No valid rows parsed.")
        return

    # De-dupe while preserving order
    seen = set()
    unique_rows: list[tuple[str, int]] = []
    for repo_name, pr_number in parsed_rows:
        key = (repo_name, pr_number)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(key)

    if limit is not None:
        unique_rows = unique_rows[:limit]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as f:
        f.write("repo_name,pr_number\n")
        for repo_name, pr_number in unique_rows:
            f.write(f"{repo_name},{pr_number}\n")

    print(f"Wrote {len(unique_rows)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse PR IDs from a CSV with an 'id' column into repo_name,pr_number rows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/test.pr_commits_20_400_100_0.5_nltk.csv"),
        help="Input CSV path containing an 'id' column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/parsed.csv"),
        help="Output CSV path to write repo_name,pr_number rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on rows to write after de-duplication.",
    )
    args = parser.parse_args()

    parse_ids(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
