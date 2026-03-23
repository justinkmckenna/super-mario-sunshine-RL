from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import dolphin_memory_engine as dme


ROOT_A = 0x8040E108
ROOT_B = 0x8040E12C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Log Sunshine manual-race traces including progress, terminal flags, "
            "and candidate pointer-based position fields."
        )
    )
    parser.add_argument("--progress-address", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--finished-address", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--failed-address", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--finished-value", type=int, default=1)
    parser.add_argument("--failed-value", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("manual_race_trace.csv"),
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


def read_word(address: int) -> int:
    return int(dme.read_word(address))


def read_float(address: int) -> float:
    return float(dme.read_float(address))


def read_byte(address: int) -> int:
    return int(dme.read_byte(address))


def maybe_read_float(base: int, offset: int) -> float | None:
    if base == 0:
        return None
    try:
        return read_float(base + offset)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    wait_for_hook()

    input(
        "Load the pre-race savestate, get ready to control the race manually, "
        "then press Enter to start logging..."
    )

    rows: list[list[object]] = []
    start_ts = time.time()
    print("Manual race trace logging started.")

    try:
        while True:
            now = time.time()
            elapsed = now - start_ts

            base_a = read_word(ROOT_A)
            base_b = read_word(ROOT_B)

            progress = read_float(args.progress_address)
            finished_raw = read_byte(args.finished_address)
            failed_raw = read_byte(args.failed_address)
            finished = finished_raw == args.finished_value
            failed = failed_raw == args.failed_value

            x_like_a = maybe_read_float(base_a, 0x10)
            y_like_a = maybe_read_float(base_a, 0x14)
            z_like_a = maybe_read_float(base_a, 0x18)
            yaw_like_a = maybe_read_float(base_a, 0x34)

            x_like_b = maybe_read_float(base_b, 0x48)
            y_like_b = maybe_read_float(base_b, 0x4C)
            z_like_b = maybe_read_float(base_b, 0x50)
            x_dup_b = maybe_read_float(base_b, 0x60)
            z_dup_b = maybe_read_float(base_b, 0x68)

            rows.append(
                [
                    f"{elapsed:.3f}",
                    progress,
                    finished_raw,
                    failed_raw,
                    int(finished),
                    int(failed),
                    hex(base_a) if base_a else "",
                    x_like_a,
                    y_like_a,
                    z_like_a,
                    yaw_like_a,
                    hex(base_b) if base_b else "",
                    x_like_b,
                    y_like_b,
                    z_like_b,
                    x_dup_b,
                    z_dup_b,
                ]
            )

            if finished or failed:
                print(
                    f"[event] terminal detected at t={elapsed:.2f}s "
                    f"finished={finished} failed={failed}"
                )
                break
            if elapsed >= args.max_seconds:
                print("[warn] Logging timed out before terminal state.")
                break
            time.sleep(args.interval_seconds)
    finally:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "elapsed_s",
                    "progress",
                    "finished_raw",
                    "failed_raw",
                    "mission_finished",
                    "mission_failed",
                    "base_a",
                    "x_like_a",
                    "y_like_a",
                    "z_like_a",
                    "yaw_like_a",
                    "base_b",
                    "x_like_b",
                    "y_like_b",
                    "z_like_b",
                    "x_dup_b",
                    "z_dup_b",
                ]
            )
            writer.writerows(rows)
        print(f"Wrote CSV: {args.output_csv}")

        if dme.is_hooked():
            dme.un_hook()


if __name__ == "__main__":
    main()
