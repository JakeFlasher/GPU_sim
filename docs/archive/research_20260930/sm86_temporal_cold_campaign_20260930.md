# Temporal cold holdout (2026-09-30)

`results/alpha/full_3090ti_20260930_temporal_cold_live2/` is a fresh 15-row
campaign with two NCU repeats, two native repeats, one replay, and eight
warmups. It completed all rows on the RTX 3090 Ti/SM86 path.

The native-event/frontier crosswalk was calibrated only on the prior refined
cold campaign and scored this temporal holdout at **8.36% GPC / 8.08%
duration MAPE** over 15 rows. Every row received a candidate/fallback result.

The high-clock elasticity route is not applicable to this campaign: its
post-hoc NCU-local frequency was approximately 1.344 GHz, while elasticity was
calibrated around the 1.905-GHz regime. The NCU frequency was not used to route
or fit the prediction; it is retained only as post-hoc regime evidence. This
result strengthens the need for an independent clock-regime endpoint before
applying frequency response models.

No test, hash, or verification workflow was run.

