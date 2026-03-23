from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import dolphin_memory_engine as dme


ROOT_A = 0x8040E108
ROOT_B = 0x8040E12C

ROOT_A_FIELDS = {
    "x_like_a": 0x10,
    "y_like_a": 0x14,
    "z_like_a": 0x18,
    "yaw_like_a": 0x34,
}

ROOT_B_FIELDS = {
    "x_like_b": 0x48,
    "y_like_b": 0x4C,
    "z_like_b": 0x50,
    "x_dup_b": 0x60,
    "z_dup_b": 0x68,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the most promising Sunshine position-like pointer-chain fields."
    )
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("position_named_probe.csv"),
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


def read_root_base(root: int) -> int:
    return int(dme.read_word(root))


def read_field(base: int, offset: int) -> float | None:
    if base == 0:
        return None
    try:
        return float(dme.read_float(base + offset))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    wait_for_hook()

    rows: list[dict[str, object]] = []
    print("Sampling named candidate position fields.")

    try:
        for sample_idx in range(1, args.samples + 1):
            base_a = read_root_base(ROOT_A)
            base_b = read_root_base(ROOT_B)

            row: dict[str, object] = {
                "sample": sample_idx,
                "root_a": hex(ROOT_A),
                "base_a": hex(base_a) if base_a else "",
                "root_b": hex(ROOT_B),
                "base_b": hex(base_b) if base_b else "",
            }

            for field_name, offset in ROOT_A_FIELDS.items():
                row[field_name] = read_field(base_a, offset)
            for field_name, offset in ROOT_B_FIELDS.items():
                row[field_name] = read_field(base_b, offset)

            rows.append(row)

            print(
                "sample={sample} "
                "base_a={base_a} x={x} y={y} z={z} yaw={yaw} "
                "base_b={base_b} xb={xb} yb={yb} zb={zb}".format(
                    sample=sample_idx,
                    base_a=row["base_a"] or "NULL",
                    x=row["x_like_a"],
                    y=row["y_like_a"],
                    z=row["z_like_a"],
                    yaw=row["yaw_like_a"],
                    base_b=row["base_b"] or "NULL",
                    xb=row["x_like_b"],
                    yb=row["y_like_b"],
                    zb=row["z_like_b"],
                )
            )

            if sample_idx < args.samples:
                time.sleep(args.interval_seconds)
    finally:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample",
            "root_a",
            "base_a",
            "x_like_a",
            "y_like_a",
            "z_like_a",
            "yaw_like_a",
            "root_b",
            "base_b",
            "x_like_b",
            "y_like_b",
            "z_like_b",
            "x_dup_b",
            "z_dup_b",
        ]
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.output_csv}")

        if dme.is_hooked():
            dme.un_hook()


if __name__ == "__main__":
    main()
