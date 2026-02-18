"""Dune Analytics v1 query runner — research-only script.

Usage:
    python scripts/dune/run_query.py --query-id 12345 --output result.csv
    python scripts/dune/run_query.py --query-id 12345  # prints to stdout

Rate: 2,500 credits/month on free plan (~10-20 queries/month).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from typing import Any

import httpx

DUNE_API_URL = "https://api.dune.com/api/v1"
POLL_INTERVAL = 5  # seconds
MAX_POLLS = 120  # 10 minutes max


def execute_query(api_key: str, query_id: int) -> list[dict[str, Any]]:
    """Execute a Dune query and poll until completion.

    Args:
        api_key: Dune API key.
        query_id: Dune query ID.

    Returns:
        List of result rows (dict per row).

    Raises:
        RuntimeError: Query failed or timed out.
    """
    headers = {"X-Dune-API-Key": api_key}
    client = httpx.Client(timeout=30.0)

    # 1. Execute query
    print(f"Executing Dune query {query_id}...")
    resp = client.post(
        f"{DUNE_API_URL}/query/{query_id}/execute",
        headers=headers,
    )
    resp.raise_for_status()
    execution_id = resp.json()["execution_id"]
    print(f"Execution ID: {execution_id}")

    # 2. Poll for completion
    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status_resp = client.get(
            f"{DUNE_API_URL}/execution/{execution_id}/status",
            headers=headers,
        )
        status_resp.raise_for_status()
        state = status_resp.json().get("state", "")
        print(f"  Poll {i + 1}/{MAX_POLLS}: {state}")

        if state == "QUERY_STATE_COMPLETED":
            break
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            msg = f"Dune query {query_id} {state}"
            raise RuntimeError(msg)
    else:
        msg = f"Dune query {query_id} timed out after {MAX_POLLS * POLL_INTERVAL}s"
        raise RuntimeError(msg)

    # 3. Fetch results
    results_resp = client.get(
        f"{DUNE_API_URL}/execution/{execution_id}/results",
        headers=headers,
    )
    results_resp.raise_for_status()
    data = results_resp.json()

    rows: list[dict[str, Any]] = data.get("result", {}).get("rows", [])
    print(f"Got {len(rows)} rows")
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run a Dune Analytics query")
    parser.add_argument("--query-id", type=int, required=True, help="Dune query ID")
    parser.add_argument("--output", type=str, default="", help="Output CSV path (default: stdout)")
    parser.add_argument(
        "--api-key", type=str, default="", help="Dune API key (or DUNE_API_KEY env)"
    )
    args = parser.parse_args()

    import os

    api_key = args.api_key or os.environ.get("DUNE_API_KEY", "")
    if not api_key:
        print("Error: DUNE_API_KEY not set (use --api-key or env var)", file=sys.stderr)
        sys.exit(1)

    rows = execute_query(api_key, args.query_id)

    if not rows:
        print("No results returned.")
        return

    if args.output:
        with pathlib.Path(args.output).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
