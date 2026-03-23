from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import dolphin_memory_engine as dme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe float fields at offsets inside live root-pointed objects."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["0x8040E108", "0x8040E12C"],
        help="Absolute addresses whose words are treated as object base pointers.",
    )
    parser.add_argument("--start-offset", type=lambda x: int(x, 0), default=0x0)
    parser.add_argument("--end-offset", type=lambda x: int(x, 0), default=0x80)
    parser.add_argument("--step", type=lambda x: int(x, 0), default=0x4)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("object_offset_probe.csv"),
    )
    return parser.parse_args()


def wait_for_hook() -> None:
    while True:
        if dme.is_hooked():
            return
        dme.hook()
        if dme.is_hooked():
            return
        print("Waiting for Dolphin memory hook...")
        time.sleep(1.0)


def main() -> None:
    args = parse_args()
    roots = [int(value, 0) for value in args.roots]
    offsets = list(range(args.start_offset, args.end_offset + 1, args.step))
    wait_for_hook()

    rows: list[list[object]] = []
    print("Sampling float fields from live root-pointed objects.")

    try:
        for sample_idx in range(1, args.samples + 1):
            for root in roots:
                base = int(dme.read_word(root))
                if base == 0:
                    print(f"sample={sample_idx} root={hex(root)} base=NULL")
                    continue

                print(f"sample={sample_idx} root={hex(root)} base={hex(base)}")
                for offset in offsets:
                    address = base + offset
                    try:
                        value = float(dme.read_float(address))
                    except Exception:
                        continue
                    rows.append(
                        [
                            sample_idx,
                            hex(root),
                            hex(base),
                            hex(offset),
                            hex(address),
                            value,
                        ]
                    )
            if sample_idx < args.samples:
                time.sleep(args.interval_seconds)
    finally:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sample",
                    "root",
                    "base",
                    "offset",
                    "address",
                    "value",
                ]
            )
            writer.writerows(rows)
        print(f"Wrote CSV: {args.output_csv}")

        if dme.is_hooked():
            dme.un_hook()


if __name__ == "__main__":
    main()
