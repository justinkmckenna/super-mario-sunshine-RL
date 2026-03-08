# Success Flag Selection (Blooper Surfing v0)

> Status: superseded. This selection has been reset and is no longer used.
> Success address discovery will be repeated from the purple-blooper savestate.

## Outcome

Selected provisional mission-success binding:

- game: Super Mario Sunshine USA (`GMSE01`)
- address: `0x80426B33`
- type: `byte`
- success value: `1`

## Why This Address

- Repeatedly matched expected success-flag behavior:
  - baseline start: `0`
  - safe driving: unchanged
  - mission success: flips to `1`
  - savestate reload: returns to `0`
- Top-ranked in two independent success/reset cycles.
- Does not overlap with the selected fail-flag candidate set.

## Process Used

1. Started from fixed Blooper Surfing start savestate.
2. Collected two success cycles with scripted snapshot sampling:
   - baseline
   - safe #1
   - success #1
   - reset #1
   - safe #2
   - success #2
   - reset #2
3. Required consistent success value across both cycles.
4. Ranked candidates by deterministic flip/reset behavior and simple byte-flag heuristics.

## Notes

- This is a provisional v0 success flag for environment bring-up.
- If it proves unstable across additional runs, test nearby top-ranked candidates from the generated success CSV.
