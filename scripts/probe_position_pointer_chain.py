from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import dolphin_memory_engine as dme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe suspected Mario position pointer-chain values into CSV."
    )
    parser.add_argument("--root-address", type=lambda x: int(x, 0), default=0x8041E108)
    parser.add_argument("--inner-offset", type=lambda x: int(x, 0), default=0x4FC)
    parser.add_argument("--offset-a", type=lambda x: int(x, 0), default=0x2C)
    parser.add_argument("--offset-b", type=lambda x: int(x, 0), default=0x48)
    parser.add_argument("--offset-c", type=lambda x: int(x, 0), default=0x4C)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("position_pointer_chain_probe.csv"),
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


def resolve_chain(root_address: int, inner_offset: int) -> tuple[int, int]:
    root_ptr = int(dme.read_word(root_address))
    if root_ptr == 0:
        raise RuntimeError(f"Root pointer at {hex(root_address)} is null.")
    inner_ptr = int(dme.read_word(root_ptr + inner_offset))
    if inner_ptr == 0:
        raise RuntimeError(
            f"Inner pointer at {hex(root_ptr + inner_offset)} is null."
        )
    return root_ptr, inner_ptr


def main() -> None:
    args = parse_args()
    wait_for_hook()

    rows: list[list[object]] = []
    print(
        "Sampling started. Move Mario around while this script records the "
        "candidate pointer-chain values."
    )

    try:
        for sample_idx in range(1, args.samples + 1):
            root_ptr, inner_ptr = resolve_chain(args.root_address, args.inner_offset)
            value_a = float(dme.read_float(inner_ptr + args.offset_a))
            value_b = float(dme.read_float(inner_ptr + args.offset_b))
            value_c = float(dme.read_float(inner_ptr + args.offset_c))
            rows.append(
                [
                    sample_idx,
                    hex(args.root_address),
                    hex(root_ptr),
                    hex(inner_ptr),
                    hex(inner_ptr + args.offset_a),
                    hex(inner_ptr + args.offset_b),
                    hex(inner_ptr + args.offset_c),
                    value_a,
                    value_b,
                    value_c,
                ]
            )
            print(
                f"sample={sample_idx} root={hex(root_ptr)} inner={hex(inner_ptr)} "
                f"a={value_a:.6f} b={value_b:.6f} c={value_c:.6f}"
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
                    "root_address",
                    "root_ptr",
                    "inner_ptr",
                    "address_a",
                    "address_b",
                    "address_c",
                    "value_a",
                    "value_b",
                    "value_c",
                ]
            )
            writer.writerows(rows)
        print(f"Wrote CSV: {args.output_csv}")

        if dme.is_hooked():
            dme.un_hook()


if __name__ == "__main__":
    main()
