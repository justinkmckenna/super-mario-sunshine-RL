from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import dolphin_memory_engine as dme


@dataclass(slots=True)
class WatchEntry:
    label: str
    address: int
    value_type: str
    values: list[float | int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Dolphin memory watch addresses into CSV columns."
    )
    parser.add_argument("--watch-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of samples to record. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert JSON rows to CSV metadata columns; do not hook Dolphin.",
    )
    return parser.parse_args()


def load_watch_entries(path: Path) -> list[WatchEntry]:
    raw = path.read_text(encoding="utf-8")
    # Be tolerant of trailing commas in hand-edited JSON.
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    payload = json.loads(raw)

    if isinstance(payload, dict):
        rows = payload.get("watches", [])
    else:
        rows = payload

    if not isinstance(rows, list):
        raise ValueError("Watch JSON must be an array or contain a 'watches' array.")

    dedupe_counts: dict[str, int] = {}
    entries: list[WatchEntry] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not an object.")
        address_raw = row.get("address")
        value_type = str(row.get("type", "float")).lower()
        base_label = str(row.get("label") or f"addr_{index:03d}")

        if address_raw is None:
            raise ValueError(f"Row {index} is missing 'address'.")

        address = int(str(address_raw), 0)

        dedupe_counts[base_label] = dedupe_counts.get(base_label, 0) + 1
        suffix = dedupe_counts[base_label]
        label = base_label if suffix == 1 else f"{base_label}_{suffix:02d}"

        entries.append(
            WatchEntry(
                label=label,
                address=address,
                value_type=value_type,
                values=[],
            )
        )
    return entries


def read_value(address: int, value_type: str) -> float | int:
    if value_type == "byte":
        return int(dme.read_byte(address))
    if value_type == "word":
        return int(dme.read_word(address))
    if value_type == "float":
        return float(dme.read_float(address))
    if value_type == "double":
        return float(dme.read_double(address))
    raise ValueError(f"Unsupported type '{value_type}' at {hex(address)}.")


def write_csv(path: Path, entries: list[WatchEntry]) -> None:
    sample_count = max((len(entry.values) for entry in entries), default=0)
    headers = ["label", "address", "type"] + [
        f"value{i}" for i in range(1, sample_count + 1)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for entry in entries:
            row = [
                entry.label,
                hex(entry.address),
                entry.value_type,
            ] + entry.values
            writer.writerow(row)


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
    entries = load_watch_entries(args.watch_json)
    write_csv(args.output_csv, entries)
    print(f"Wrote metadata CSV: {args.output_csv}")

    if args.convert_only:
        return

    wait_for_hook()
    print("Hooked Dolphin memory. Sampling started.")

    sample_index = 0
    try:
        while True:
            for entry in entries:
                entry.values.append(read_value(entry.address, entry.value_type))

            sample_index += 1
            write_csv(args.output_csv, entries)
            print(f"Captured sample value{sample_index}")

            if args.samples > 0 and sample_index >= args.samples:
                break
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("Sampling interrupted by user.")
    finally:
        if dme.is_hooked():
            dme.un_hook()
        print(f"Final CSV written: {args.output_csv}")


if __name__ == "__main__":
    main()
