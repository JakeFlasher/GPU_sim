# Raw Ampere PMA record field sweep

This is a decoder-completeness investigation, not a test or verification
workflow. I reassembled the 8-byte Ampere PMA records from the retained raw
PMA packet blobs and inspected every bit outside the documented PC/stall/wave/
active fields.

## Data

Four retained blobs were swept:

* short NAK probe: 52 records;
* long NAK body: 416 records;
* long NAK PMA capture: 368 records;
* long variant: 1,309 records.

Total: **2,145 decoded samples** after packet reassembly. The older v1
coverage helper counted packet headers/padding as eight-byte records; its
counts are superseded by `coverage_decoded_v2.json`.

The current reverse-engineered public decoder maps bits 0--44 to PC, 45--49 to stall key,
50--55 to wave ID, and bit 62 to active/not-issued. The sweep found:

| field | observed values |
|---|---|
| bit 63 | always 0 |
| bits 61:60 | always `11` |
| bit 62 plus bits 61:60 | only `0x3` or `0x7` |

The two top-nibble values correspond to the active/not-issued pattern under
this decoder; this is not proof that the unassigned bits have no alternate
meaning under other stimuli. The machine-readable results are
`sweep_reassembled_v2.json` and `coverage_decoded_v2.json`.

## Interpretation

This does not prove that every PMA encoding on every SM86 driver is identical,
or that unassigned bits cannot carry latent information. It removes only the
specific escape hatch that the retained records visibly contain a route field.
More PMA samples can reduce sampling uncertainty; they cannot identify a
dynamic physical pipe without a new record format or an independent ordinal
join.
