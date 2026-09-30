# SM86 control-word LDG wait-mask sensitivity

**Corrected interpretation:** SM70+/SM86 control fields are documented in
`sm86_control_word_decode_correction_20260930.md`.

A disposable structured-loop image changed only the wait-mask bit 2 on the
first post-LDG consumer at PC `0x90`:

* reference: `ctrl=0x27ed`, wait mask `0x04`;
* patched: `ctrl=0x7ed`, wait mask `0x00`;
* read-barrier bits remain `0x7` in both images.

Five Driver runs deterministically returned `64,66,69,73,68,70,73,77` instead
of `0,2,5,9,4,6,9,13`, with a 1.881088-ms median.

Artifact: `results/alpha/sm86_control_word_sensitivity_20260930/ldg_consumer_rbar/`.

## Interpretation

This is a direct hardware dependency witness for the SM86 LDG wait-mask field:
removing the wait produces an early/stale consumer result. It still does not
identify the physical pipe, subunit, or issue slot servicing the load.
