# SM86 control-word field-layout correction

**Date:** 2026-09-30

## Correct SM70+/SM86 dependency layout

The earlier sensitivity reports decoded the 21-bit control value with an
SM120-style layout. The SM70+ encoder used by Mesa NAK (and the SM86 hardware
path) places fields as follows:

| control-code bits | field |
|---:|---|
| 0..3 | delay/stall count |
| 4 | yield |
| 5..7 | write scoreboard barrier |
| 8..10 | read scoreboard barrier |
| 11..16 | wait-on-barrier mask |
| 17..20 | reuse mask |

The authoritative encoder writes these fields at full instruction bits
105..108, 109, 110..112, 113..115, 116..121, and 122..125 respectively.
Sources: `src/nouveau/compiler/nak/sm70_encode.rs` and the retained CuAssembler
control-code decoder.

## Consequences for prior disposable mutations

* PC `0x210`, `0x207f1 -> 0x7f1`, changed **reuse-mask bit 2**, not a one-cycle
  stall field.
* PC `0x90`, `0x27ed -> 0x7ed`, cleared **wait-mask bit 2** (`0x2000`),
  while read-barrier bits remain unchanged; this is the isolated LDG wait.
* PC `0x20`, `0xfed -> 0xfe1`, changed the **delay/stall nibble** from 13 to 1,
  not an opcode-specific misc field.

The hardware observations remain valid as control-word sensitivity results,
and the wait-mask mutation is a field-isolated LDG dependency witness. The
reuse and delay mutations remain mechanism-local timing observations; none is a
physical pipe/slot witness.

The temporary rule-driven synthesis script used the incorrect SM120-style
packing and is not promoted as an independent SM86 control generator.
