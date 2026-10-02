# Live cache-policy contrast (2026-10-02)

This is a post-prediction live NCU intervention on two completed large SM86
metrics cases. It does not tune the simulator or claim held-out accuracy.

- Cases: `large_read_0` and `large_atom_2` from the R10.0 campaign.
- Three fresh NCU application-replay repetitions per cache policy.
- `cache-control=all` is the original post-prediction condition;
  `cache-control=none` is the live intervention. Clock control remained none.
- Raw NCU reports remain under the ignored R10 results tree.

The machine-readable result is [`cache_policy_contrast.json`](cache_policy_contrast.json).
The contrast is a live observation of endpoint sensitivity and repeatability;
NCU values are not fed into fitting or prediction selection, and no hardware
cache-causality claim is made.
