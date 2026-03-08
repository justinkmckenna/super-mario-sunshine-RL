# Success Flag Selection (Purple Blooper Start)

## Outcome

Selected provisional mission-success binding:

- game: Super Mario Sunshine USA (`GMSE01`)
- address: `0x805F64C6`
- type: `byte`
- success value: `1`

## Validation Summary

Signal was validated with repeated `debug_dolphin_signals.py` runs:

- fail run #1: success remained `0`, fail flipped to `1` (pass)
- fail run #2: success remained `0`, fail flipped to `1` (pass)
- success run #1: success flipped to `1`, fail remained `0` (pass)

## Notes

- This selection is tied to the purple-blooper start savestate workflow.
- Progress and fail bindings used in the same validation:
  - progress: `0x80FA50D4` (`float`)
  - fail: `0x804257D3` (`byte`, value `1`)
