# SM86 NCU demo benchmark — smoke v2

This is the four-row executable smoke gate captured from `GPUSim_main`
revision `08c7c9dfb1d93181f5b303220a7ebcf2b380b8c4`. It contains one NCU
application-replay measurement and two byte-identical simulator replays per
workload. The smoke report passed before the complete matrix was opened.

`evidence/raw_campaign.tar.zst` is a deterministic archive of the materialized
`evidence/raw` tree:

- SHA-256: `c6c2bf41f72b16eee9f5619797582bdabfa4f10eee69de4dbaa74c1bd0f86f6d`
- Compressed bytes: `519460`
- Decompressed tar bytes: `35461120`

From this directory, materialize it with:

```bash
zstd -dc evidence/raw_campaign.tar.zst | tar -xf - -C .
```

The smoke matrix was subsequently superseded by the eight-row, three-repeat
`sm86_ncu_demo_benchmark_20260901_full_v2` campaign; it remains retained as the
required pre-full executable report.
