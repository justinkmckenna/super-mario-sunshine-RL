from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import dolphin_memory_engine as dme

from sms_rl.config import EpisodeConfig


@dataclass(slots=True)
class SignalBindings:
    progress_address: int
    finished_address: int
    failed_address: int
    finished_value: int = 1
    failed_value: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor Dolphin memory signals and compute reward components using the "
            "current environment reward settings."
        )
    )
    parser.add_argument("--progress-address", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--finished-address", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--failed-address", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--finished-value", type=int, default=1)
    parser.add_argument("--failed-value", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=0.25)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="Monitoring timeout per run.",
    )
    parser.add_argument(
        "--expect",
        choices=("success", "fail", "either"),
        default="either",
        help="Expected terminal state for validation.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dolphin_signal_debug.csv"),
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


def read_progress(address: int) -> float:
    return float(dme.read_float(address))


def read_flag(address: int, expected: int) -> bool:
    return int(dme.read_byte(address)) == expected


def validate_expectation(expect: str, finished: bool, failed: bool) -> bool:
    if expect == "either":
        return finished or failed
    if expect == "success":
        return finished and not failed
    return failed and not finished


def main() -> None:
    args = parse_args()
    bindings = SignalBindings(
        progress_address=args.progress_address,
        finished_address=args.finished_address,
        failed_address=args.failed_address,
        finished_value=args.finished_value,
        failed_value=args.failed_value,
    )
    reward_cfg = EpisodeConfig()

    wait_for_hook()
    input("Load your savestate and pause at episode start, then press Enter...")

    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "elapsed_s",
                "progress",
                "progress_delta",
                "finished",
                "failed",
                "reward_progress",
                "reward_step_penalty",
                "reward_finish",
                "reward_fail",
                "reward_total",
            ]
        )

        start_ts = time.time()
        step_idx = 0
        last_progress = read_progress(bindings.progress_address)
        terminated = False
        last_finished = False
        last_failed = False

        print("Monitoring started.")
        print(
            f"Using progress=0x{bindings.progress_address:X}, "
            f"finished=0x{bindings.finished_address:X}, "
            f"failed=0x{bindings.failed_address:X}"
        )

        while True:
            now = time.time()
            elapsed = now - start_ts
            progress = read_progress(bindings.progress_address)
            finished = read_flag(bindings.finished_address, bindings.finished_value)
            failed = read_flag(bindings.failed_address, bindings.failed_value)
            progress_delta = max(0.0, progress - last_progress)

            reward_progress = progress_delta * reward_cfg.progress_reward_scale
            reward_step_penalty = reward_cfg.step_penalty
            reward_finish = reward_cfg.finish_reward if finished else 0.0
            reward_fail = reward_cfg.fail_reward if failed else 0.0
            reward_total = (
                reward_progress + reward_step_penalty + reward_finish + reward_fail
            )

            writer.writerow(
                [
                    step_idx,
                    f"{elapsed:.3f}",
                    f"{progress:.6f}",
                    f"{progress_delta:.6f}",
                    int(finished),
                    int(failed),
                    f"{reward_progress:.6f}",
                    f"{reward_step_penalty:.6f}",
                    f"{reward_finish:.6f}",
                    f"{reward_fail:.6f}",
                    f"{reward_total:.6f}",
                ]
            )
            handle.flush()

            if finished and not last_finished:
                print(f"[event] success flag set at t={elapsed:.2f}s step={step_idx}")
            if failed and not last_failed:
                print(f"[event] fail flag set at t={elapsed:.2f}s step={step_idx}")

            if finished or failed:
                terminated = True
                ok = validate_expectation(args.expect, finished, failed)
                if ok:
                    print("[ok] Terminal state matched expectation.")
                else:
                    print("[warn] Terminal state did not match expectation.")
                break

            if elapsed >= args.max_seconds:
                print("[warn] Timed out before terminal state.")
                break

            step_idx += 1
            last_progress = progress
            last_finished = finished
            last_failed = failed
            time.sleep(args.interval_seconds)

    if dme.is_hooked():
        dme.un_hook()

    print(f"CSV written: {output_path}")
    if not terminated:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
