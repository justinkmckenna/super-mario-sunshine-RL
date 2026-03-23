from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import dolphin_memory_engine as dme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect candidate absolute pointer roots and nearby values over time."
    )
    parser.add_argument(
        "--addresses",
        nargs="+",
        default=["0x8040E12C", "0x8040E108", "0x8041E108"],
    )
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("pointer_root_inspection.csv"),
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


def safe_read_word(address: int) -> int | None:
    try:
        return int(dme.read_word(address))
    except Exception:
        return None


def looks_like_ram_pointer(value: int | None) -> bool:
    if value is None:
        return False
    return 0x80000000 <= value <= 0x81800000


def main() -> None:
    args = parse_args()
    addresses = [int(value, 0) for value in args.addresses]
    wait_for_hook()

    rows: list[list[object]] = []
    print("Sampling candidate pointer roots over time.")

    try:
        for sample_idx in range(1, args.samples + 1):
            for address in addresses:
                value = safe_read_word(address)
                plus4 = safe_read_word(address + 4)
                plus8 = safe_read_word(address + 8)
                rows.append(
                    [
                        sample_idx,
                        hex(address),
                        "" if value is None else hex(value),
                        int(looks_like_ram_pointer(value)),
                        "" if plus4 is None else hex(plus4),
                        int(looks_like_ram_pointer(plus4)),
                        "" if plus8 is None else hex(plus8),
                        int(looks_like_ram_pointer(plus8)),
                    ]
                )
                print(
                    f"sample={sample_idx} addr={hex(address)} "
                    f"word={hex(value) if value is not None else 'ERR'} "
                    f"word+4={hex(plus4) if plus4 is not None else 'ERR'} "
                    f"word+8={hex(plus8) if plus8 is not None else 'ERR'}"
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
                    "address",
                    "word",
                    "word_is_ram_ptr",
                    "word_plus_4",
                    "word_plus_4_is_ram_ptr",
                    "word_plus_8",
                    "word_plus_8_is_ram_ptr",
                ]
            )
            writer.writerows(rows)
        print(f"Wrote CSV: {args.output_csv}")

        if dme.is_hooked():
            dme.un_hook()


if __name__ == "__main__":
    main()
