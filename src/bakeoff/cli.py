"""Command line entry point.

Two verbs: `run` executes a config and writes results.json, `report` renders a
results.json that already exists. They are separate so a report can be
regenerated -- or its formatting fixed -- without paying for the run again.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from bakeoff.config import RunConfig
from bakeoff.report import load_results, render_markdown, write_readme_table
from bakeoff.runner import run, write_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bakeoff", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute a benchmark config")
    run_parser.add_argument("config", type=pathlib.Path)
    run_parser.add_argument("--out", type=pathlib.Path, required=True, help="results directory")
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="override the config's query cap, for a fast partial run",
    )

    report_parser = subparsers.add_parser("report", help="render an existing results.json")
    report_parser.add_argument("results", type=pathlib.Path)
    report_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    report_parser.add_argument(
        "--write-readme",
        action="store_true",
        help="splice the table into README.md between the generated-table markers",
    )
    report_parser.add_argument("--readme", type=pathlib.Path, default=pathlib.Path("README.md"))

    args = parser.parse_args(argv)

    if args.command == "run":
        config = RunConfig.from_file(args.config)
        if args.max_queries is not None:
            config.max_queries = args.max_queries
        payload = run(config, verbose=not args.quiet)
        path = write_results(payload, args.out)
        print(f"\nwrote {path}", file=sys.stderr)

        failed = [row for row in payload["results"] if row["status"] == "failed"]
        if failed:
            print(f"{len(failed)} system(s) failed", file=sys.stderr)
            return 1
        return 0

    payload = load_results(args.results)
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(payload))

    if args.write_readme:
        if write_readme_table(payload, args.readme):
            print(f"updated {args.readme}", file=sys.stderr)
        else:
            print(
                f"{args.readme} has no generated-table markers; left unchanged",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
