#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from PLred._sort_base import find_data_between


def main():
    parser = argparse.ArgumentParser(
        description="Copy PLred/SCExAO files within a filename timestamp range."
    )
    parser.add_argument("datadir", help="Directory containing files")
    parser.add_argument("outdir", help="Directory to copy files into")
    parser.add_argument("--start", required=True, help="Start time, HH:MM:SS")
    parser.add_argument("--end", required=True, help="End time, HH:MM:SS")
    parser.add_argument("--header", default="", help="Filename header before timestamp")
    parser.add_argument("--footer", default="", help="Filename footer after timestamp")
    parser.add_argument("--dry-run", action="store_true", help="Print only; do not copy")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    datadir = str(Path(args.datadir)) + "/"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_data_between(
        datadir,
        args.start,
        args.end,
        header=args.header,
        footer=args.footer,
    )

    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files.")

    for src in files:
        src = Path(src)
        dst = outdir / src.name

        if dst.exists() and not args.overwrite:
            print(f"SKIP exists: {dst}")
            continue

        print(f"{src} -> {dst}")
        if not args.dry_run:
            shutil.copy2(src, dst)

    print("Done.")


if __name__ == "__main__":
    main()