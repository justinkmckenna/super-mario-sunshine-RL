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
    bad_terminal_1: int
    true_finish_1: int
    reset_1: int
    safe_2: int
    bad_terminal_2: int
    true_finish_2: int
    reset_2: int
    score: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find true finish-line flag candidates by distinguishing safe driving, "
            "bad terminal states, and true finish states across two cycles."
        )
    )
    parser.add_argument("--start", type=lambda x: int(x, 0), default=DEFAULT_START)
    parser.add_argument("--end", type=lambda x: int(x, 0), default=DEFAULT_END)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("true_finish_flag_candidates.csv"),
    )
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument(
        "--require-same-finish-value",
        action="store_true",
        help="Keep only candidates where finish value matches across both cycles.",
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


def score_finish(b: int, f: int) -> int:
    score = 0
    if b in (0, 1) and f in (0, 1):
        score += 3
    if b == 0 and f == 1:
        score += 3
    if abs(f - b) <= 4:
        score += 1
    return score


def build_candidates(
    start: int,
    baseline: np.ndarray,
    safe_1: np.ndarray,
    bad_1: np.ndarray,
    finish_1: np.ndarray,
    reset_1: np.ndarray,
    safe_2: np.ndarray,
    bad_2: np.ndarray,
    finish_2: np.ndarray,
    reset_2: np.ndarray,
    *,
    require_same_finish_value: bool,
) -> list[Candidate]:
    # Desired true-finish shape:
    # - unchanged during safe riding
    # - unchanged during known bad terminal
    # - changed only on true finish
    # - reset returns to baseline
    mask = (
        (safe_1 == baseline)
        & (safe_2 == baseline)
        & (bad_1 == baseline)
        & (bad_2 == baseline)
        & (finish_1 != baseline)
        & (finish_2 != baseline)
        & (reset_1 == baseline)
        & (reset_2 == baseline)
    )
    if require_same_finish_value:
        mask &= finish_1 == finish_2

    indices = np.where(mask)[0]
    results: list[Candidate] = []
    for idx in indices.tolist():
        b = int(baseline[idx])
        s1 = int(safe_1[idx])
        bt1 = int(bad_1[idx])
        tf1 = int(finish_1[idx])
        r1 = int(reset_1[idx])
        s2 = int(safe_2[idx])
        bt2 = int(bad_2[idx])
        tf2 = int(finish_2[idx])
        r2 = int(reset_2[idx])
        score = max(score_finish(b, tf1), score_finish(b, tf2))
        if tf1 == tf2:
            score += 2
        if b == 0 and tf1 == 1 and tf2 == 1:
            score += 2
        results.append(
            Candidate(
                address=start + idx,
                baseline=b,
                safe_1=s1,
                bad_terminal_1=bt1,
                true_finish_1=tf1,
                reset_1=r1,
                safe_2=s2,
                bad_terminal_2=bt2,
                true_finish_2=tf2,
                reset_2=r2,
                score=score,
            )
        )

    results.sort(
        key=lambda c: (c.score, -abs(c.true_finish_1 - c.baseline), -abs(c.true_finish_2 - c.baseline)),
        reverse=True,
    )
    return results


def write_csv(path: Path, rows: list[Candidate], max_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "address",
                "baseline",
                "safe_1",
                "bad_terminal_1",
                "true_finish_1",
                "reset_1",
                "safe_2",
                "bad_terminal_2",
                "true_finish_2",
                "reset_2",
                "delta_finish_1",
                "delta_finish_2",
                "score",
            ]
        )
        for row in rows[:max_rows]:
            writer.writerow(
                [
                    hex(row.address),
                    row.baseline,
                    row.safe_1,
                    row.bad_terminal_1,
                    row.true_finish_1,
                    row.reset_1,
                    row.safe_2,
                    row.bad_terminal_2,
                    row.true_finish_2,
                    row.reset_2,
                    row.true_finish_1 - row.baseline,
                    row.true_finish_2 - row.baseline,
                    row.score,
                ]
            )


def main() -> None:
    args = parse_args()
    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start")
    size = args.end - args.start

    wait_for_hook()
    input("Load BLOOPER savestate at start (not terminal), then pause. Press Enter...")
    baseline = read_snapshot(args.start, size)
    print("Captured baseline.")

    input("Drive safely for ~3s (no terminal), pause. Press Enter...")
    safe_1 = read_snapshot(args.start, size)
    print("Captured safe #1.")

    input("Trigger BAD terminal (reverse/timeout-like), pause. Press Enter...")
    bad_1 = read_snapshot(args.start, size)
    print("Captured bad terminal #1.")

    input("Reload start savestate, then finish the track normally, pause on finish. Press Enter...")
    finish_1 = read_snapshot(args.start, size)
    print("Captured true finish #1.")

    input("Reload start savestate and pause immediately. Press Enter...")
    reset_1 = read_snapshot(args.start, size)
    print("Captured reset #1.")

    input("Drive safely for ~3s again (no terminal), pause. Press Enter...")
    safe_2 = read_snapshot(args.start, size)
    print("Captured safe #2.")

    input("Trigger BAD terminal again, pause. Press Enter...")
    bad_2 = read_snapshot(args.start, size)
    print("Captured bad terminal #2.")

    input("Reload start savestate, finish the track again, pause on finish. Press Enter...")
    finish_2 = read_snapshot(args.start, size)
    print("Captured true finish #2.")

    input("Reload start savestate and pause immediately. Press Enter...")
    reset_2 = read_snapshot(args.start, size)
    print("Captured reset #2.")

    rows = build_candidates(
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
        require_same_finish_value=args.require_same_finish_value,
    )
    write_csv(args.output_csv, rows, args.max_rows)

    print(f"Found {len(rows)} candidates.")
    print(f"Wrote: {args.output_csv}")
    if rows:
        print("Top 10:")
        for row in rows[:10]:
            print(
                f"{hex(row.address)} baseline={row.baseline} "
                f"bad1={row.bad_terminal_1} finish1={row.true_finish_1} "
                f"bad2={row.bad_terminal_2} finish2={row.true_finish_2} score={row.score}"
            )

    if dme.is_hooked():
        dme.un_hook()


if __name__ == "__main__":
    main()
