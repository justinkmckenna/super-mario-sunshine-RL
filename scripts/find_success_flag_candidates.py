from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import dolphin_memory_engine as dme
import numpy as np


DEFAULT_START = 0x80000000
DEFAULT_END = 0x81800000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find candidate mission-success byte addresses by comparing memory "
            "snapshots across multiple user-guided state transitions."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("basic", "win-only", "true-finish"),
        default="basic",
        help=(
            "basic: safe/success/reset cycles only; "
            "win-only: requires fail to remain baseline; "
            "true-finish: requires bad terminal to remain baseline."
        ),
    )
    parser.add_argument("--start", type=lambda x: int(x, 0), default=DEFAULT_START)
    parser.add_argument("--end", type=lambda x: int(x, 0), default=DEFAULT_END)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("success_flag_candidates.csv"),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Maximum number of rows to write.",
    )
    parser.add_argument(
        "--require-same-success-value",
        action="store_true",
        help="Keep only addresses where success/finish value matches across both cycles.",
    )
    parser.add_argument(
        "--require-same-finish-value",
        action="store_true",
        help="Alias for --require-same-success-value (kept for compatibility).",
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


def read_snapshot(start: int, size: int) -> np.ndarray:
    raw = dme.read_bytes(start, size)
    return np.frombuffer(raw, dtype=np.uint8)


def score_candidate(baseline: int, success: int) -> int:
    score = 0
    if baseline in (0, 1) and success in (0, 1):
        score += 3
    if baseline == 0 and success == 1:
        score += 3
    if baseline == 1 and success == 0:
        score += 2
    if success in (1, 2, 255):
        score += 1
    if abs(success - baseline) <= 4:
        score += 1
    return score


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    max_rows: int,
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows[:max_rows]:
            writer.writerow([row[c] for c in columns])


def _sort_rows(rows: list[dict[str, Any]], key1: str, key2: str) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda r: (
            r["score"],
            -abs(int(r[key1]) - int(r["baseline"])),
            -abs(int(r[key2]) - int(r["baseline"])),
        ),
        reverse=True,
    )
    return rows


def build_candidates_basic(
    start: int,
    baseline: np.ndarray,
    safe_1: np.ndarray,
    success_1: np.ndarray,
    reset_1: np.ndarray,
    safe_2: np.ndarray,
    success_2: np.ndarray,
    reset_2: np.ndarray,
    *,
    require_same_success_value: bool,
) -> list[dict[str, Any]]:
    mask = (
        (baseline == safe_1)
        & (baseline == safe_2)
        & (success_1 != baseline)
        & (success_2 != baseline)
        & (reset_1 == baseline)
        & (reset_2 == baseline)
    )
    if require_same_success_value:
        mask &= success_1 == success_2

    rows: list[dict[str, Any]] = []
    for idx in np.where(mask)[0].tolist():
        b = int(baseline[idx])
        u1 = int(success_1[idx])
        u2 = int(success_2[idx])
        score = max(score_candidate(b, u1), score_candidate(b, u2))
        if u1 == u2:
            score += 2
        if b == 0 and u1 == 1 and u2 == 1:
            score += 2
        rows.append(
            {
                "address": hex(start + idx),
                "baseline": b,
                "safe_1": int(safe_1[idx]),
                "success_1": u1,
                "reset_1": int(reset_1[idx]),
                "safe_2": int(safe_2[idx]),
                "success_2": u2,
                "reset_2": int(reset_2[idx]),
                "delta_success_1": u1 - b,
                "delta_success_2": u2 - b,
                "score": score,
            }
        )
    return _sort_rows(rows, "success_1", "success_2")


def build_candidates_with_negative_terminal(
    start: int,
    baseline: np.ndarray,
    safe_1: np.ndarray,
    negative_1: np.ndarray,
    success_1: np.ndarray,
    reset_1: np.ndarray,
    safe_2: np.ndarray,
    negative_2: np.ndarray,
    success_2: np.ndarray,
    reset_2: np.ndarray,
    *,
    require_same_success_value: bool,
    negative_label: str,
    success_label: str,
) -> list[dict[str, Any]]:
    mask = (
        (safe_1 == baseline)
        & (safe_2 == baseline)
        & (negative_1 == baseline)
        & (negative_2 == baseline)
        & (success_1 != baseline)
        & (success_2 != baseline)
        & (reset_1 == baseline)
        & (reset_2 == baseline)
    )
    if require_same_success_value:
        mask &= success_1 == success_2

    rows: list[dict[str, Any]] = []
    for idx in np.where(mask)[0].tolist():
        b = int(baseline[idx])
        u1 = int(success_1[idx])
        u2 = int(success_2[idx])
        score = max(score_candidate(b, u1), score_candidate(b, u2))
        if u1 == u2:
            score += 2
        if b == 0 and u1 == 1 and u2 == 1:
            score += 2
        rows.append(
            {
                "address": hex(start + idx),
                "baseline": b,
                "safe_1": int(safe_1[idx]),
                f"{negative_label}_1": int(negative_1[idx]),
                f"{success_label}_1": u1,
                "reset_1": int(reset_1[idx]),
                "safe_2": int(safe_2[idx]),
                f"{negative_label}_2": int(negative_2[idx]),
                f"{success_label}_2": u2,
                "reset_2": int(reset_2[idx]),
                f"delta_{success_label}_1": u1 - b,
                f"delta_{success_label}_2": u2 - b,
                "score": score,
            }
        )
    return _sort_rows(rows, f"{success_label}_1", f"{success_label}_2")


def main() -> None:
    args = parse_args()
    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start")
    require_same_success_value = bool(
        args.require_same_success_value or args.require_same_finish_value
    )

    size = args.end - args.start
    wait_for_hook()
    input("Load BLOOPER savestate at start (not terminal), then pause. Press Enter...")
    baseline = read_snapshot(args.start, size)
    print("Captured baseline.")

    if args.mode == "basic":
        input("Unpause, drive safely for ~3 seconds (do NOT finish), pause. Press Enter...")
        safe_1 = read_snapshot(args.start, size)
        print("Captured safe-driving snapshot #1.")

        input("Now finish the mission, pause immediately on success state. Press Enter...")
        success_1 = read_snapshot(args.start, size)
        print("Captured success snapshot #1.")

        input("Reload savestate back to start, pause immediately. Press Enter...")
        reset_1 = read_snapshot(args.start, size)
        print("Captured reset snapshot #1.")

        input("Unpause, drive safely for ~3 seconds again, pause. Press Enter...")
        safe_2 = read_snapshot(args.start, size)
        print("Captured safe-driving snapshot #2.")

        input("Finish mission again, pause immediately on success state. Press Enter...")
        success_2 = read_snapshot(args.start, size)
        print("Captured success snapshot #2.")

        input("Reload savestate back to start again, pause immediately. Press Enter...")
        reset_2 = read_snapshot(args.start, size)
        print("Captured reset snapshot #2.")

        rows = build_candidates_basic(
            args.start,
            baseline,
            safe_1,
            success_1,
            reset_1,
            safe_2,
            success_2,
            reset_2,
            require_same_success_value=require_same_success_value,
        )
    elif args.mode == "win-only":
        input("Drive safely ~3s (no fail/success), pause. Press Enter...")
        safe_1 = read_snapshot(args.start, size)
        print("Captured safe #1.")

        input("Intentionally fail, pause on fail state. Press Enter...")
        fail_1 = read_snapshot(args.start, size)
        print("Captured fail #1.")

        input("Reload start savestate, then finish successfully, pause on success. Press Enter...")
        success_1 = read_snapshot(args.start, size)
        print("Captured success #1.")

        input("Reload start savestate and pause. Press Enter...")
        reset_1 = read_snapshot(args.start, size)
        print("Captured reset #1.")

        input("Drive safely ~3s again, pause. Press Enter...")
        safe_2 = read_snapshot(args.start, size)
        print("Captured safe #2.")

        input("Intentionally fail again, pause on fail state. Press Enter...")
        fail_2 = read_snapshot(args.start, size)
        print("Captured fail #2.")

        input("Reload start savestate, finish successfully again, pause on success. Press Enter...")
        success_2 = read_snapshot(args.start, size)
        print("Captured success #2.")

        input("Reload start savestate and pause. Press Enter...")
        reset_2 = read_snapshot(args.start, size)
        print("Captured reset #2.")

        rows = build_candidates_with_negative_terminal(
            args.start,
            baseline,
            safe_1,
            fail_1,
            success_1,
            reset_1,
            safe_2,
            fail_2,
            success_2,
            reset_2,
            require_same_success_value=require_same_success_value,
            negative_label="fail",
            success_label="success",
        )
    else:
        input("Drive safely ~3s (no terminal), pause. Press Enter...")
        safe_1 = read_snapshot(args.start, size)
        print("Captured safe #1.")

        input("Trigger bad terminal (reverse/timeout-like), pause. Press Enter...")
        bad_1 = read_snapshot(args.start, size)
        print("Captured bad terminal #1.")

        input("Reload start savestate, then finish the track normally, pause on finish. Press Enter...")
        finish_1 = read_snapshot(args.start, size)
        print("Captured true finish #1.")

        input("Reload start savestate and pause. Press Enter...")
        reset_1 = read_snapshot(args.start, size)
        print("Captured reset #1.")

        input("Drive safely ~3s again (no terminal), pause. Press Enter...")
        safe_2 = read_snapshot(args.start, size)
        print("Captured safe #2.")

        input("Trigger bad terminal again, pause. Press Enter...")
        bad_2 = read_snapshot(args.start, size)
        print("Captured bad terminal #2.")

        input("Reload start savestate, finish the track again, pause on finish. Press Enter...")
        finish_2 = read_snapshot(args.start, size)
        print("Captured true finish #2.")

        input("Reload start savestate and pause. Press Enter...")
        reset_2 = read_snapshot(args.start, size)
        print("Captured reset #2.")

        rows = build_candidates_with_negative_terminal(
            args.start,
            baseline,
            safe_1,
            bad_1,
            finish_1,
            reset_1,
            safe_2,
            bad_2,
            finish_2,
            reset_2,
            require_same_success_value=require_same_success_value,
            negative_label="bad_terminal",
            success_label="true_finish",
        )

    columns = list(rows[0].keys()) if rows else ["address", "score"]
    write_csv(args.output_csv, rows, args.max_rows, columns)
    print(f"Found {len(rows)} candidates.")
    print(f"Wrote: {args.output_csv}")
    if rows:
        print("Top 10:")
        for row in rows[:10]:
            print(f"{row['address']} score={row['score']} row={row}")

    if dme.is_hooked():
        dme.un_hook()


if __name__ == "__main__":
    main()
