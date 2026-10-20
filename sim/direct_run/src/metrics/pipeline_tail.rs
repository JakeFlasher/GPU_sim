//! Cohesive child mechanism extracted from the parent module.

use super::*;

pub(super) fn finish_sm86(state: Sm86MetricState<'_>) -> NativeMetrics {
    NativeMetrics {
        schema_version: 1,
        metric_surface_version: 2,
        architecture: "sm_86",
        sm_count: state.sm_count,
        active_sm_count: state.active_sm_count,
        reference_clock_hz: state.clock,
        metrics: state.metrics,
    }
}

pub(super) fn project_sm86_metrics(
    result: &FullKernelReplayResult,
    card: &ArchCard,
    sm_count: u32,
    active_sm_count: u32,
    clock: Option<u64>,
    external: Option<&LaunchResourceObservation>,
    m: BTreeMap<String, Metric>,
) -> Result<NativeMetrics, RunError> {
    let mut state = Sm86MetricState {
        result,
        external,
        card,
        sm_count,
        active_sm_count,
        clock,
        metrics: m,
    };
    state.add_sm86_core()?;
    state.add_sm86_readiness()?;
    state.add_sm86_memory()?;
    state.add_sm86_occupancy();
    state.add_sm86_tail()?;
    Ok(finish_sm86(state))
}

#[rustfmt::skip]
pub(crate) fn project(
    result: &FullKernelReplayResult,
    card: &ArchCard,
    external: Option<&LaunchResourceObservation>,
) -> Result<NativeMetrics, RunError> {

    let active_sm_count = u32::try_from(result.per_sm().len())
        .map_err(|_| RunError::new("metric SM count", "SM count exceeds u32"))?;
    let sm_count = card.sm_count().map_err(crate::at("metric SM count"))?;
    if active_sm_count > sm_count {
        return Err(RunError::new(
            "metric SM count",
            "active SM count exceeds card capacity",
        ));
    }
    // Conversion is a declared fixed-clock estimate, not a clock inferred
    // from the scored workload's measured duration.
    let clock = card
        .require_dram_bandwidth()
        .ok()
        .and_then(|bandwidth| bandwidth.locked_sm_clock_hz().ok());
    let mut m = BTreeMap::new();
    if let Some(resources) = external {
        resources.validate_trace_resources(result.trace_metrics().launch_resources.as_ref())?;
        for (key, unit, value, semantics) in [
            (
                "external_registers_per_thread",
                "register/thread",
                u128::from(resources.registers_per_thread),
                "CUPTI launch resource observation",
            ),
            (
                "external_static_shared_memory_bytes",
                "byte",
                u128::from(resources.static_shared_memory_bytes),
                "CUPTI launch resource observation",
            ),
            (
                "external_dynamic_shared_memory_bytes",
                "byte",
                u128::from(resources.dynamic_shared_memory_bytes),
                "CUPTI launch resource observation",
            ),
            (
                "external_carveout_requested",
                "boolean",
                u128::from(resources.carveout_requested),
                "CUPTI launch resource observation; preference presence marker",
            ),
        ] {
            m.insert(key.into(), Metric::observed(None, unit, value, semantics));
        }
        m.insert("external_correlation_id".into(), if resources.unique_correlation {
            Metric::observed(None, "id", u128::from(resources.correlation_id), "correlation ID from the unique matching CUPTI resource record; not an NVBit launch ID")
        } else {
            Metric::unavailable(None, "id", "repeated equal-resource launches cannot be paired with cross-run CUPTI correlation IDs")
        });
        m.insert(
            "external_shared_memory_config_bytes".into(),
            Metric::observed(
                Some("launch__shared_mem_config_size"),
                "byte",
                u128::from(resources.shared_memory_config_bytes),
                "CUPTI sharedMemoryExecuted: driver-selected shared-memory configuration; external observation, never a replay/timing input",
            ),
        );
        m.insert(
            "shared_memory_config_size".into(),
            Metric::observed(
                Some("launch__shared_mem_config_size"),
                "byte",
                u128::from(resources.shared_memory_config_bytes),
                "CUPTI sharedMemoryExecuted joined to this launch; external observation, never a replay/timing input",
            ),
        );
        if let Some(pct) = resources.carveout_pct {
            m.insert(
                "external_shared_memory_carveout_pct".into(),
                Metric::observed(
                    None,
                    "%",
                    u128::from(pct),
                    "CUPTI/CUDA launch carveout preference; preference is not guaranteed to equal driver-selected configuration",
                ),
            );
        }
    }

    if result.arch() == ArchId::Sm120 {
        return project_sm120_alpha_metrics(result, sm_count, active_sm_count, clock, m);
    }
    if result.arch() != ArchId::Sm86 {
        m.insert(
            "native_metrics".to_owned(),
            Metric::unavailable(None, "none", "SM86-only metric definitions"),
        );
        return Ok(NativeMetrics {
            schema_version: 1,
            metric_surface_version: 2,
            architecture: result.arch().token(),
            sm_count,
            active_sm_count,
            reference_clock_hz: clock,
            metrics: m,
        });
    }

    project_sm86_metrics(result, card, sm_count, active_sm_count, clock, external, m)
}
