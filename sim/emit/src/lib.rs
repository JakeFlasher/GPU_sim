//! sim-emit: the NCU-metric emission layer (design §6.4).
//!
//! Takes a completed kernel replay's counter snapshot and produces a typed
//! vector of NCU metric PREDICTIONS, one per registry observation contract
//! (`tools/local_labels/metric_registry.json`). It is the ONLY sim crate that
//! names label-pipeline contracts: metric NAMES are the output contract and
//! are allowed here; label VALUES still never enter (Rule 1, enforced by
//! `eval/guards/test_targets_not_features_guard.py`).
//!
//! Shape (design proposal P3, "descriptor-table hybrid"): a compiled
//! [`DESCRIPTORS`] table is the single source of truth — one row per emitted
//! metric binds a registry key + NCU metric string + [`Unit`] + [`MetricKind`]
//! to a pure emit fn over the raw counters. A registry-lockstep test walks the
//! table against `metric_registry.json` so a descriptor can never drift from
//! the contract it claims to implement. Rate accounting lives HERE, never in
//! the cache counter structs (gate-2 altitude note c): `L1Counters`/
//! `L2Counters` hold raw counts; the emit fns compose them.
//!
//! Three states are kept distinct, never conflated into a silent zero:
//!   * a real zero — `Exact(0)` / `Fraction { value: Some(0.0), .. }`
//!     (accesses happened, none hit);
//!   * not-applicable — `Fraction { value: None, denominator: 0, .. }`
//!     (no accesses; NCU reports n/a — a legitimate scorable prediction);
//!   * not-produced-by-this-run — [`EmitOutcome::Unavailable`] (the replay's
//!     mem-mode lacks the counting unit, e.g. no L2 under `L1Const`).
//!
//! LDGSTS folding (the §6.4 op_ld surface): NCU books LDGSTS (cp.async) global
//! loads under `op_ld` on the frozen sm_86 bundle — verified against the
//! bundle's own captured counts, NOT the metric-name taxonomy (a witness
//! kernel with only LDGSTS loads and no ordinary loads shows all of its
//! `op_ld` requests attributed to the async copies, with the total-request
//! rollup balancing exactly against `op_ld` + `op_st`). So the op_ld surface
//! folds the READ-class bypass stream (`bypass_sectors - bypass_atom_sectors`,
//! the async-copy loads) into its denominator; the ATOM write-class subset
//! stays out (NCU books it under op_atom/op_red). The sim models LDGSTS as
//! pure bypass with no L1 residency, so those sectors contribute to the
//! denominator but never to the hit numerator — an honest v0 under-prediction
//! whose residual localizes the un-modeled LDGSTS-L1-residency effect.

use sim_arch::ArchId;
use sim_mem::{L1Counters, L2Counters};
use sim_sm::{DramStats, KernelReplayResult};

/// The exact counter fields the emission layer reads, borrowed. Decouples
/// emission from the full `KernelReplayResult` so a cell's counters — summed
/// across its kernel launches — can be emitted directly (rates compose from
/// summed numerators/denominators; they cannot be averaged post-hoc).
#[derive(Debug, Clone, Copy)]
pub struct CounterSnapshot<'a> {
    /// Deployment architecture proven by replay composition.
    pub arch: ArchId,
    /// `smsp__inst_executed.sum` source — total warp-instructions issued.
    pub instructions_issued: u64,
    /// L1TEX counters, present iff the replay attached an L1 (not `Flat`).
    pub l1: Option<&'a L1Counters>,
    /// L2 counters, present only on the shared-L2 path.
    pub l2: Option<&'a L2Counters>,
    /// DRAM read observer, present only on the shared-L2 path.
    pub dram: Option<&'a DramStats>,
}

impl<'a> CounterSnapshot<'a> {
    /// Borrow the emission-relevant counters out of one completed replay.
    pub fn from_replay(r: &'a KernelReplayResult) -> Self {
        Self {
            arch: r.arch,
            instructions_issued: r.instructions_issued,
            l1: r.l1.as_ref(),
            l2: r.l2.as_ref(),
            dram: r.dram.as_ref(),
        }
    }

    fn validate_provenance(&self) -> Result<(), EmitError> {
        let check = |counter: &'static str, got: Option<ArchId>| {
            let got = got.ok_or(EmitError::CounterArchitectureUnbound { counter })?;
            if got != self.arch {
                return Err(EmitError::CounterArchitectureMismatch {
                    counter,
                    expected: self.arch,
                    got,
                });
            }
            Ok(())
        };
        if let Some(l1) = self.l1 {
            check("L1 counters", l1.arch())?;
        }
        if let Some(l2) = self.l2 {
            check("L2 counters", l2.arch())?;
        }
        if let Some(dram) = self.dram {
            check("DRAM stats", dram.arch())?;
        }
        Ok(())
    }
}

/// A predicted metric value, shaped to its NCU contract. The numeric payload
/// carries num/den so a `None` rate is auditable in-band (None IFF den == 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricValue {
    /// An exact non-negative integer count (sectors, requests, instructions).
    Exact(u64),
    /// A `[0, 1]` fraction. `value` is `None` iff `denominator == 0` (NCU
    /// reports the metric as n/a — never a fabricated 0.0).
    Fraction {
        value: Option<f64>,
        numerator: u64,
        denominator: u64,
    },
}

impl MetricValue {
    /// Fail-closed `[0, 1]` fraction constructor. `value` is `None` iff
    /// `den == 0`; a `num > den` fraction is a counter-arithmetic bug and is
    /// surfaced typed, never clamped into range.
    fn fraction(key: &'static str, num: u64, den: u64) -> Result<Self, EmitError> {
        if num > den {
            return Err(EmitError::RateExceedsUnity { key, num, den });
        }
        Ok(Self::Fraction {
            value: (den != 0).then(|| num as f64 / den as f64),
            numerator: num,
            denominator: den,
        })
    }
}

/// The NCU-facing unit of a metric (the registry `unit` field). Only the
/// variants v0 uses are declared; cycles/ns/ratio arrive with their metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// A raw non-negative count.
    Count,
    /// A percentage in NCU's own rendering; stored as a `[0, 1]` fraction
    /// (registry `parser_transform = percent_to_fraction`).
    Pct,
}

/// The numeric shape a descriptor promises to emit — asserted against the
/// returned [`MetricValue`] so a wrong-variant slip fails loudly, and pinned
/// against the registry `metric_class`/`valid_max` by the lockstep test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// Emits [`MetricValue::Exact`].
    Count,
    /// Emits [`MetricValue::Fraction`].
    Fraction,
}

/// A hardware counting unit a metric needs to be predictable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemUnit {
    /// The L1TEX cache (`KernelReplayResult.l1`).
    L1Tex,
    /// The L2 slices (`KernelReplayResult.l2`).
    Lts,
    /// The DRAM read observer (`KernelReplayResult.dram`).
    Dram,
}

/// Typed reason a metric was NOT predicted by THIS run — distinct from a
/// value-level `None` (which means "predicted n/a: no accesses"). The harness
/// excludes such a cell with a reason rather than scoring a null.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unavailable {
    /// This descriptor has no reviewed counting semantics on the replay architecture.
    ArchitectureUnsupported { arch: ArchId },
    /// The replay's mem-mode did not attach the counting unit this metric
    /// reads (e.g. no L2 counters under `Flat`/`L1Const`).
    MemModeLacksUnit { needs: MemUnit },
}

/// Per-metric outcome: a genuine prediction, or a typed not-produced reason.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmitOutcome {
    Predicted(MetricValue),
    Unavailable(Unavailable),
}

/// One emitted metric, carrying its registry key + NCU identifier so the eval
/// join (downstream, outside `sim/`) can match against frozen labels by key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricPrediction {
    /// Deployment architecture of the replay that produced this outcome.
    pub arch: ArchId,
    /// The registry key — the eval JOIN key.
    pub key: &'static str,
    /// The NCU metric identifier this predicts.
    pub ncu_metric: &'static str,
    pub unit: Unit,
    pub outcome: EmitOutcome,
}

/// A compiled row of the emission table: the binding from a sim quantity to
/// its registry contract. [`DESCRIPTORS`] is the single source of truth the
/// lockstep test pins against `metric_registry.json`.
pub struct MetricDescriptor {
    pub key: &'static str,
    pub ncu_metric: &'static str,
    pub unit: Unit,
    pub kind: MetricKind,
    /// Architectures with reviewed counter semantics for this descriptor.
    pub available_on: &'static [ArchId],
    pub emit: fn(&CounterSnapshot) -> Result<EmitOutcome, EmitError>,
}

/// Fail-closed emission faults. Benign "this run's mem-mode lacks the unit" is
/// NOT an error — it is an [`Unavailable`] outcome inside the vector, so a
/// Flat-mode run still emits the instruction count while marking the cache
/// rates typed-unavailable (the Carded::Gap / unsupported_memory_path idiom).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EmitError {
    #[error(
        "metric {key}: rate numerator {num} exceeds denominator {den} — a >1.0 fraction is a \
         counter bug, never clamped into range"
    )]
    RateExceedsUnity {
        key: &'static str,
        num: u64,
        den: u64,
    },
    #[error("counter invariant broken for {key}: {detail}")]
    CounterInvariant {
        key: &'static str,
        detail: &'static str,
    },
    #[error("{counter} has no bound architecture provenance")]
    CounterArchitectureUnbound { counter: &'static str },
    #[error("{counter} architecture mismatch: expected {expected}, got {got}")]
    CounterArchitectureMismatch {
        counter: &'static str,
        expected: ArchId,
        got: ArchId,
    },
}

/// READ-class bypass sectors = all bypass sectors minus the ATOM write-class
/// subset. This is the LDGSTS/bypass-LOAD stream that folds into the `op_ld`
/// surface; the ATOM subset (booked under op_atom/op_red by NCU) stays out.
fn bypass_read_sectors(key: &'static str, l1: &L1Counters) -> Result<u64, EmitError> {
    l1.bypass_sectors
        .checked_sub(l1.bypass_atom_sectors)
        .ok_or(EmitError::CounterInvariant {
            key,
            detail: "bypass_atom_sectors exceeds bypass_sectors (subset invariant broken)",
        })
}

// ---- per-metric emit fns (the rate accounting, moved out of the counters) ----

/// `smsp__inst_executed.sum` — total warp-instructions issued. One issue per
/// traced record across every SM; always available (no mem unit needed).
fn emit_smsp_inst_executed(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    Ok(EmitOutcome::Predicted(MetricValue::Exact(
        snap.instructions_issued,
    )))
}

/// `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — global-load sectors:
/// modeled eligible loads plus the READ-class bypass (LDGSTS) stream.
fn emit_gmem_ld_sectors(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "traffic_gmem_ld_sectors";
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    let sectors = l1.hits + l1.misses + bypass_read_sectors(key, l1)?;
    Ok(EmitOutcome::Predicted(MetricValue::Exact(sectors)))
}

/// `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` — global-store sectors.
fn emit_gmem_st_sectors(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    Ok(EmitOutcome::Predicted(MetricValue::Exact(l1.store_sectors)))
}

/// `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` — the op_ld sector
/// hit rate. Numerator = modeled eligible hits; denominator folds the
/// READ-class bypass (LDGSTS) sectors in (they miss L1 structurally in v0, so
/// they enter the denominator but contribute no hits — the honest residual).
fn emit_l1tex_read_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l1tex_read_sector";
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    let num = l1.hits;
    let den = l1.hits + l1.misses + bypass_read_sectors(key, l1)?;
    Ok(EmitOutcome::Predicted(MetricValue::fraction(
        key, num, den,
    )?))
}

/// `l1tex__t_sector_hit_rate.pct` — the unified L1TEX sector hit rate over all
/// op classes. v0 numerator uses the sector-valid store rule (`store_hits`);
/// the §6.4 line-TAG rule (`store_tag_hits`) is the carried alternative,
/// unresolved until l1tex store accounting is scored. The full bypass stream
/// (atom included) is in the denominator — the unified metric covers
/// global_atomic/global_reduction, so `bypass_atom_sectors` must NOT be added
/// on top (it is already a subset of `bypass_sectors`).
fn emit_l1tex_unified_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l1tex_sector";
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    let num = l1.hits + l1.store_hits;
    let den = l1.hits + l1.misses + l1.store_sectors + l1.bypass_sectors + l1.untagged_sectors;
    Ok(EmitOutcome::Predicted(MetricValue::fraction(
        key, num, den,
    )?))
}

/// `lts__t_sector_hit_rate.pct` — the unified L2 sector hit rate. Every LTS
/// access from all four L2 source lanes is in `l2.hits`/`l2.misses`;
/// `write_tag_hits` is already a subset of `hits` (do not add it).
fn emit_l2_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l2_sector";
    let Some(l2) = snap.l2 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::Lts,
        }));
    };
    Ok(EmitOutcome::Predicted(MetricValue::fraction(
        key,
        l2.hits,
        l2.total(),
    )?))
}

/// `dram__sectors_read.sum` — DRAM read sectors (v0 read-only demand proxy).
fn emit_dram_read_sectors(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let Some(dram) = snap.dram else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::Dram,
        }));
    };
    Ok(EmitOutcome::Predicted(MetricValue::Exact(
        dram.read_sectors,
    )))
}

const SM86_ONLY: &[ArchId] = &[ArchId::Sm86];

/// The v0 emission table — one row per registry contract this layer predicts.
/// Adding a metric is a one-row append plus its emit fn; the lockstep test and
/// the eval join need no edits.
pub const DESCRIPTORS: &[MetricDescriptor] = &[
    MetricDescriptor {
        key: "smsp_inst_executed_sum",
        ncu_metric: "smsp__inst_executed.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        available_on: SM86_ONLY,
        emit: emit_smsp_inst_executed,
    },
    MetricDescriptor {
        key: "traffic_gmem_ld_sectors",
        ncu_metric: "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        available_on: SM86_ONLY,
        emit: emit_gmem_ld_sectors,
    },
    MetricDescriptor {
        key: "traffic_gmem_st_sectors",
        ncu_metric: "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        available_on: SM86_ONLY,
        emit: emit_gmem_st_sectors,
    },
    MetricDescriptor {
        key: "cache_hit_rate_l1tex_read_sector",
        ncu_metric: "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        available_on: SM86_ONLY,
        emit: emit_l1tex_read_hit_rate,
    },
    MetricDescriptor {
        key: "cache_hit_rate_l1tex_sector",
        ncu_metric: "l1tex__t_sector_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        available_on: SM86_ONLY,
        emit: emit_l1tex_unified_hit_rate,
    },
    MetricDescriptor {
        key: "cache_hit_rate_l2_sector",
        ncu_metric: "lts__t_sector_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        available_on: SM86_ONLY,
        emit: emit_l2_hit_rate,
    },
    MetricDescriptor {
        key: "traffic_dram_read_sectors",
        ncu_metric: "dram__sectors_read.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        available_on: SM86_ONLY,
        emit: emit_dram_read_sectors,
    },
];

/// Assert an emit fn returned the numeric shape its descriptor promised.
fn shape_matches(kind: MetricKind, value: &MetricValue) -> bool {
    matches!(
        (kind, value),
        (MetricKind::Count, MetricValue::Exact(_))
            | (MetricKind::Fraction, MetricValue::Fraction { .. })
    )
}

/// Emit every descriptor over one completed replay, in registry-table order.
/// Genuine faults (a rate above unity, a broken subset invariant) return
/// [`EmitError`]; benign missing-unit cases ride the [`Unavailable`] arm.
pub fn emit_all(snap: &CounterSnapshot) -> Result<Vec<MetricPrediction>, EmitError> {
    snap.validate_provenance()?;
    let mut out = Vec::with_capacity(DESCRIPTORS.len());
    for d in DESCRIPTORS {
        let outcome = if d.available_on.contains(&snap.arch) {
            (d.emit)(snap)?
        } else {
            EmitOutcome::Unavailable(Unavailable::ArchitectureUnsupported { arch: snap.arch })
        };
        if let EmitOutcome::Predicted(v) = &outcome {
            debug_assert!(
                shape_matches(d.kind, v),
                "emit fn for {} returned a value shape inconsistent with its declared kind",
                d.key
            );
        }
        out.push(MetricPrediction {
            arch: snap.arch,
            key: d.key,
            ncu_metric: d.ncu_metric,
            unit: d.unit,
            outcome,
        });
    }
    Ok(out)
}

/// A by-key view of a prediction vector for the eval join.
pub fn index_by_key(
    preds: &[MetricPrediction],
) -> std::collections::BTreeMap<&'static str, &MetricPrediction> {
    preds.iter().map(|p| (p.key, p)).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use sim_mem::{L1Counters, L2Counters};
    use sim_sm::DramStats;

    /// A replay result carrying just the counters the emit fns read.
    fn replay(
        instructions_issued: u64,
        l1: Option<L1Counters>,
        l2: Option<L2Counters>,
        dram: Option<DramStats>,
    ) -> KernelReplayResult {
        KernelReplayResult {
            arch: ArchId::Sm86,
            instructions_issued,
            records_seen: instructions_issued,
            per_sm: Default::default(),
            max_cycles: 1,
            l1,
            l2,
            l2_by_source: None,
            dram,
        }
    }

    fn predicted<'a>(preds: &'a [MetricPrediction], key: &str) -> &'a EmitOutcome {
        &preds
            .iter()
            .find(|p| p.key == key)
            .expect("key present")
            .outcome
    }

    /// Emit from owned counters (the `KernelReplayResult` temporary lives
    /// through the `emit_all` call within this expression).
    fn emit(
        instructions_issued: u64,
        l1: Option<L1Counters>,
        l2: Option<L2Counters>,
        dram: Option<DramStats>,
    ) -> Result<Vec<MetricPrediction>, EmitError> {
        emit_all(&CounterSnapshot::from_replay(&replay(
            instructions_issued,
            l1,
            l2,
            dram,
        )))
    }

    #[test]
    fn template_architecture_is_unavailable_not_zero() {
        let mut result = replay(17, None, None, None);
        result.arch = ArchId::Sm120;
        let preds = emit_all(&CounterSnapshot::from_replay(&result)).unwrap();
        assert!(preds.iter().all(|p| {
            p.arch == ArchId::Sm120
                && p.outcome
                    == EmitOutcome::Unavailable(Unavailable::ArchitectureUnsupported {
                        arch: ArchId::Sm120,
                    })
        }));
    }

    #[test]
    fn counter_provenance_is_required_and_cannot_be_relabelled() {
        let unbound = L1Counters::default();
        let snapshot = CounterSnapshot {
            arch: ArchId::Sm86,
            instructions_issued: 1,
            l1: Some(&unbound),
            l2: None,
            dram: None,
        };
        assert_eq!(
            emit_all(&snapshot),
            Err(EmitError::CounterArchitectureUnbound {
                counter: "L1 counters"
            })
        );

        let sm86 = L1Counters::for_arch(ArchId::Sm86);
        let relabelled = CounterSnapshot {
            arch: ArchId::Sm120,
            instructions_issued: 1,
            l1: Some(&sm86),
            l2: None,
            dram: None,
        };
        assert_eq!(
            emit_all(&relabelled),
            Err(EmitError::CounterArchitectureMismatch {
                counter: "L1 counters",
                expected: ArchId::Sm120,
                got: ArchId::Sm86,
            })
        );
    }

    #[test]
    fn fraction_is_none_at_zero_denominator_never_zero() {
        let v = MetricValue::fraction("k", 0, 0).unwrap();
        assert_eq!(
            v,
            MetricValue::Fraction {
                value: None,
                numerator: 0,
                denominator: 0
            }
        );
    }

    #[test]
    fn fraction_real_zero_is_some_zero() {
        let MetricValue::Fraction { value, .. } = MetricValue::fraction("k", 0, 10).unwrap() else {
            panic!("expected fraction");
        };
        assert_eq!(value, Some(0.0));
    }

    #[test]
    fn rate_above_unity_is_a_typed_error_not_a_clamp() {
        assert_eq!(
            MetricValue::fraction("k", 11, 10),
            Err(EmitError::RateExceedsUnity {
                key: "k",
                num: 11,
                den: 10
            })
        );
    }

    #[test]
    fn op_ld_surface_folds_read_class_bypass_ldgsts_into_the_denominator() {
        // Witness shape: a kernel of ONLY LDGSTS loads (all bypass, none
        // eligible, no atoms) — mirrors gemm_hmma_bf16. The op_ld sector
        // count is the bypass-read stream, and the modeled hit rate is 0
        // (v0 has no L1 residency for bypass) — the honest under-prediction.
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.bypass_sectors = 196_608;
        l1.bypass_atom_sectors = 0;
        l1.store_sectors = 16_384;
        let preds = emit(197_120, Some(l1), None, None).unwrap();

        assert_eq!(
            predicted(&preds, "traffic_gmem_ld_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(196_608))
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l1tex_read_sector"),
            &EmitOutcome::Predicted(MetricValue::Fraction {
                value: Some(0.0),
                numerator: 0,
                denominator: 196_608
            })
        );
        assert_eq!(
            predicted(&preds, "traffic_gmem_st_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(16_384))
        );
        // The instruction count needs no mem unit and is always emitted.
        assert_eq!(
            predicted(&preds, "smsp_inst_executed_sum"),
            &EmitOutcome::Predicted(MetricValue::Exact(197_120))
        );
    }

    #[test]
    fn op_ld_denominator_excludes_the_atom_write_class_bypass_subset() {
        // 100 bypass sectors, 40 of them ATOM (write-class -> op_atom/op_red,
        // NOT op_ld). The op_ld surface sees only the 60 READ-class sectors.
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.hits = 10;
        l1.misses = 5;
        l1.bypass_sectors = 100;
        l1.bypass_atom_sectors = 40;
        let preds = emit(1, Some(l1), None, None).unwrap();
        assert_eq!(
            predicted(&preds, "traffic_gmem_ld_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(10 + 5 + 60))
        );
    }

    #[test]
    fn atom_subset_exceeding_bypass_is_a_typed_counter_invariant() {
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.bypass_sectors = 3;
        l1.bypass_atom_sectors = 4;
        assert_eq!(
            emit(1, Some(l1), None, None),
            Err(EmitError::CounterInvariant {
                key: "traffic_gmem_ld_sectors",
                detail: "bypass_atom_sectors exceeds bypass_sectors (subset invariant broken)",
            })
        );
    }

    #[test]
    fn missing_mem_unit_is_unavailable_not_an_error_or_a_zero() {
        // Flat mode: no L1/L2/DRAM. Only the instruction count is produced;
        // every cache metric is typed-unavailable with its needed unit.
        let preds = emit(42, None, None, None).unwrap();
        assert_eq!(
            predicted(&preds, "smsp_inst_executed_sum"),
            &EmitOutcome::Predicted(MetricValue::Exact(42))
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l1tex_read_sector"),
            &EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::L1Tex
            })
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l2_sector"),
            &EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::Lts
            })
        );
        assert_eq!(
            predicted(&preds, "traffic_dram_read_sectors"),
            &EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::Dram
            })
        );
    }

    #[test]
    fn l2_and_dram_metrics_predict_when_their_units_are_present() {
        let mut l2 = L2Counters::for_arch(ArchId::Sm86);
        l2.hits = 3;
        l2.misses = 1;
        let mut dram = DramStats::for_arch(ArchId::Sm86);
        dram.read_sectors = 8415;
        dram.bw_delayed_responses = 0;
        let preds = emit(1, None, Some(l2), Some(dram)).unwrap();
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l2_sector"),
            &EmitOutcome::Predicted(MetricValue::Fraction {
                value: Some(0.75),
                numerator: 3,
                denominator: 4
            })
        );
        assert_eq!(
            predicted(&preds, "traffic_dram_read_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(8415))
        );
    }

    #[test]
    fn index_by_key_covers_every_descriptor() {
        let preds = emit(1, None, None, None).unwrap();
        let idx = index_by_key(&preds);
        assert_eq!(idx.len(), DESCRIPTORS.len());
        for d in DESCRIPTORS {
            assert!(
                idx.contains_key(d.key),
                "descriptor {} missing from index",
                d.key
            );
        }
    }

    #[test]
    fn descriptor_keys_are_unique() {
        let mut keys: Vec<&str> = DESCRIPTORS.iter().map(|d| d.key).collect();
        keys.sort_unstable();
        let n = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), n, "duplicate descriptor keys");
    }
}
