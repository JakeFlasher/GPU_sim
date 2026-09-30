# SM86 control-word delay-field mutation — corrected interpretation

**Correction:** the earlier report called the low nibble “misc.” SM70+/SM86
control bits 0..3 are the delay/stall count. The patch `0xfed -> 0xfe1`
changed delay 13 to delay 1; it did not isolate an opcode-specific misc field.

The loop output remained `0,2,5,9,4,6,9,13`; five runs had a 2.065408-ms median
versus a contemporaneous 2.025472-ms reference. Retain this as a bounded delay-
field sensitivity result, not a misc-field result.
