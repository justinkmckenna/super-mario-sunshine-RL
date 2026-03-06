# Progress Address Selection (Blooper Surfing v0)

## Outcome

Selected interim progress binding:

- game: Super Mario Sunshine USA (`GMSE01`)
- address: `0x80FA50D4`
- type: `float`

This address is currently used as the v0 progress signal for environment bring-up.

## Why This Address

- Monotonic increase during the race in repeated samples.
- Resets consistently from the fixed savestate baseline.
- Stable across short and long capture windows.
- Duplicate candidate (`0x80FA5128`) showed the same behavior; only one is needed.

## Process Used

1. Fixed savestate created at race start:
   - Mario already mounted on blooper
   - timer at race start
2. Dolphin Cheat Search was used to narrow unknown values:
   - 32-bit float and 32-bit unsigned searches
   - repeated `is greater than` filters while moving briefly
   - savestate reload loops and additional filters to reduce candidate count
3. Candidate list was exported to `progress.json`.
4. Sample script captured candidates into CSV every second:
   - straight run (short)
   - aggressive steering run (short)
   - normal steering long run
   - slow 40-second run
   - fast 40-second run
5. Candidates were ranked by:
   - monotonicity
   - repeatability across runs
   - divergence between fast and slow runs

## Important Interpretation

- The selected signal appears time-like (or tightly time-correlated), not pure geometric track distance.
- This is acceptable for v0 environment and baseline validation.
- Reward design should still rely heavily on finish/fail signals and step penalties until stronger geometric progress is found.

## Data Files Used

Captured files in `C:\Users\justi\Downloads`:

- `progress_live_straight.csv`
- `progress_live_agressive_steering.csv`
- `progress_live_long_run.csv`
- `progress_live_long_run_slow.csv`
- `progress_live_long_run_fast.csv`

## Next Memory Targets

- `mission_finished` flag address
- `mission_failed` flag address

These should be event/flag-style values (not continuously increasing values).
