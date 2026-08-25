# sm_86 adversarial atomic-alias and service probe

This successor removes compiler-known address layouts from the timed loop. A
volatile runtime index table and a CTA barrier resolve every lane address before
the first CLOCK sample, leaving one static cubin per operation and word width.

The campaign sweeps:

- returning `ATOMG` versus non-returning `RED`;
- 32-bit and 64-bit atomic add;
- alias degrees 1, 2, 4, 8, 16, and 32;
- grouped and scattered lane-to-word assignments;
- dense words and equal-work 128/256-byte word spacing; and
- cross-warp spacing 0, 128, 256, 4096, and 4100 bytes.

Final memory values and the order-independent sum of all returned old values
are deterministic correctness oracles. Long cubins run 1,024 generations;
trace cubins run 64.

Run from this directory:

```bash
make -j"$(nproc)"
../../../.venv/bin/python campaign.py static
../../../.venv/bin/python campaign.py correctness
../../../.venv/bin/python campaign.py trace-correctness
../../../.venv/bin/python campaign.py ncu
keys="$(../../../.venv/bin/python campaign.py expanded-keys)"
../../../.venv/bin/python campaign.py ncu --cells "$keys" --repeat-start 2 --repeats 2
../../../.venv/bin/python campaign.py source
../../../.venv/bin/python campaign.py return-path
../../../.venv/bin/python campaign.py nvbit
cargo build --release --manifest-path ../../../sim/Cargo.toml -p sim-run
../../../.venv/bin/python campaign.py simulator
```

The default output root is
`results/captures/sm86_atomic_alias_adversarial_20260818_v1`.
