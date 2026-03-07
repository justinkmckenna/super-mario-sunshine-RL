# Fail Flag Selection (Blooper Surfing v0)

## Outcome

Selected provisional mission-failed binding:

- game: Super Mario Sunshine USA (`GMSE01`)
- address: `0x804257D3`
- type: `byte`
- failed value: `1`

## Why This Address

- Repeatedly matched the expected fail-flag behavior:
  - baseline start: `0`
  - safe driving: unchanged
  - fail event: flips to `1`
  - savestate reload: returns to `0`
- Ranked in the top candidate group from two fail/reset cycles.

## Process Used

1. Started from fixed Blooper Surfing savestate.
2. Ran Cheat Search narrowing loops for fail-state changes.
3. Used scripted snapshot filtering with:
   - baseline
   - safe #1
   - fail #1
   - reset #1
   - safe #2
   - fail #2
   - reset #2
4. Required consistent fail flip behavior across both cycles.
5. Manually validated candidate in Dolphin memory viewer.

## Notes

- This is a provisional v0 fail flag and is good enough for environment bring-up.
- If regressions appear, fallback candidates should be tested from the same top-ranked set.
