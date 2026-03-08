from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import dolphin_memory_engine as dme
import numpy as np


DEFAULT_START = 0x80000000
DEFAULT_END = 0x81800000


@dataclass(slots=True)
class Candidate:
    address: int
    baseline: int
    safe_1: int
    success_1: int
    reset_1: int
    safe_2: int
    success_2: int
    reset_2: int
    score: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find candidate mission-success flag byte addresses by comparing memory "
            "snapshots across baseline, safe-driving, success, and reset states."
        )
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
        help="Keep only addresses where success_1 equals success_2.",
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


def build_candidates(
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
) -> list[Candidate]:
    # Good success-flag shape:
    # - unchanged during both safe-driving windows
    # - changed on success in both finish windows
    # - returns to baseline after both savestate resets
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
    indices = np.where(mask)[0]

    candidates: list[Candidate] = []
    for idx in indices.tolist():
        b = int(baseline[idx])
        s1 = int(safe_1[idx])
        u1 = int(success_1[idx])
        r1 = int(reset_1[idx])
        s2 = int(safe_2[idx])
        u2 = int(success_2[idx])
        r2 = int(reset_2[idx])
        score = max(score_candidate(b, u1), score_candidate(b, u2))
        if u1 == u2:
            score += 2
        if b == 0 and u1 == 1 and u2 == 1:
            score += 2
        candidates.append(
            Candidate(
                address=start + idx,
                baseline=b,
                safe_1=s1,
                success_1=u1,
                reset_1=r1,
                safe_2=s2,
                success_2=u2,
                reset_2=r2,
                score=score,
            )
        )
    candidates.sort(
        key=lambda c: (
            c.score,
            -abs(c.success_1 - c.baseline),
            -abs(c.success_2 - c.baseline),
        ),
        reverse=True,
    )
    return candidates


def write_csv(path: Path, rows: list[Candidate], max_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "address",
                "baseline",
                "safe_1",
                "success_1",
                "reset_1",
                "safe_2",
                "success_2",
                "reset_2",
                "delta_1",
                "delta_2",
                "score",
            ]
        )
        for row in rows[:max_rows]:
            writer.writerow(
                [
                    hex(row.address),
                    row.baseline,
                    row.safe_1,
                    row.success_1,
                    row.reset_1,
                    row.safe_2,
                    row.success_2,
                    row.reset_2,
                    row.success_1 - row.baseline,
                    row.success_2 - row.baseline,
                    row.score,
                ]
            )


def main() -> None:
    args = parse_args()
    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start")

    size = args.end - args.start
    wait_for_hook()

    input("Load BLOOPER savestate at start (not finished), then pause. Press Enter...")
    baseline = read_snapshot(args.start, size)
    print("Captured baseline snapshot.")

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

    rows = build_candidates(
        args.start,
        baseline,
        safe_1,
        success_1,
        reset_1,
        safe_2,
        success_2,
        reset_2,
        require_same_success_value=args.require_same_success_value,
    )
    write_csv(args.output_csv, rows, args.max_rows)
    print(f"Found {len(rows)} candidates.")
    print(f"Wrote: {args.output_csv}")
    if rows:
        print("Top 10:")
        for row in rows[:10]:
            print(
                f"{hex(row.address)} baseline={row.baseline} "
                f"success1={row.success_1} success2={row.success_2} "
                f"reset1={row.reset_1} reset2={row.reset_2} score={row.score}"
            )

    if dme.is_hooked():
        dme.un_hook()


if __name__ == "__main__":
    main()
