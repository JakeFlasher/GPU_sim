# SM86 control-word sensitivity: reuse-field mutation

**Correction:** this report supersedes the earlier “back-edge stall” label.
The SM86 field layout is documented in
`sm86_control_word_decode_correction_20260930.md`.

A disposable structured-loop image changed only control-code bit 17 at PC
`0x210`: `ctrl=0x207f1` to `ctrl=0x7f1`. Under the SM70+/SM86 layout this is
reuse-mask bit 2, not the delay/stall field. Outputs stayed
`0,2,5,9,4,6,9,13`; the patched median was 2.092032 ms versus a
contemporaneous reference median of 2.025472 ms.

This remains a bounded reuse/control sensitivity observation. It is not an
isolated stall-field witness or a physical pipe/slot tag.
