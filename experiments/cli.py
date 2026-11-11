"""Thin command line interface; --help and --dry-run never load models."""

import argparse
import json
from pathlib import Path

from experiments.config import ROOT, load_config
from experiments.run_store import new_run, output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run reproducible social-engineering experiments")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--mode", choices=("corrected-mvp", "investigation"))
    choice.add_argument("--config", type=Path, help="JSON experiment config")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--skip-text",
        action="store_true",
        default=None,
        help="Skip frozen DistilBERT references; TF-IDF still runs",
    )
    parser.add_argument("--prepare-only", action="store_true", default=None)
    parser.add_argument("--run-name", help="Fresh run identifier; preserves all previous runs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings without computing or writing",
    )
    args = parser.parse_args(argv)
    path = args.config or ROOT / "configs" / f"{args.mode or 'corrected-mvp'}.json"
    try:
        config = load_config(
            path,
            {
                k: getattr(args, k)
                for k in ("seeds", "epochs", "threads", "skip_text", "prepare_only")
            },
        )
        output = output_path(config, args.run_name)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": config.as_dict(),
                    "output": str(output),
                    "output_exists": output.exists(),
                },
                indent=2,
            )
        )
        return
    try:
        with new_run(output, config):
            from experiments.runner import run_suite

            run_suite(config, output)
    except FileExistsError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
