//! Post-replay report records and serde contracts; report-only with no cache or timing mutation.

use super::*;

#[derive(Debug, Serialize)]
pub struct TimingReport {
    /// Distinguishes the simulator's internal frontier status from physical
    /// completion semantics.  The SM120 alpha path deliberately reports an
    /// approximate internal frontier when it flattens async/TMA demands.
    pub completion_frontier_semantics: &'static str,
    /// Effective SM120 GPC front-end envelope. This is separate from the raw
    /// SM-active frontier and is not a hardware clock conversion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_gpc_front_end_envelope_cycles: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha_approximation: Option<AlphaApproximationReport>,
    pub diagnostic_last_issue_cycles: u64,
    pub last_processed_modeled_system_event_time: Option<u64>,
    pub terminal_modeled_event_time: Option<u64>,
    /// SM-active terminal lower bound with service-acknowledgement events kept
    /// in a separate observation domain.  The complete modeled-event frontier
    /// above remains the authoritative event-queue timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm_active_terminal_frontier: Option<SmActiveTerminalFrontierReport>,
    /// Explicit SM-active physical observation frontier. The raw lower bounds
    /// above remain unchanged and are retained inside this projection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm_active_observation_frontier: Option<SmActiveObservationFrontierReport>,
    /// Architecture-scoped GPC observation frontier. The raw modeled event
    /// lower bound above remains separate so a profiler boundary cannot be
    /// mistaken for physical cache or SM execution time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpc_observation_frontier: Option<GpcObservationFrontierReport>,
    pub completion_frontier: CompletionFrontierReport,
    pub readiness: ReadinessReport,
}

#[derive(Debug, Serialize)]
pub struct Sm120IssueEnvelopeReport {
    pub status: &'static str,
    pub profile_id: &'static str,
    /// Exact values consumed by the flat alpha executor.  These are scenario
    /// inputs, not confidence bounds or identified physical resources.
    pub issue_ports_configured: u32,
    pub dual_issue_configured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scalar_latency_configured_cycles: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scalar_initiation_configured_cycles: Option<u64>,
    pub max_resident_warps_configured: u32,
    pub max_resident_blocks_configured: u32,
    pub configuration_source: &'static str,
    pub non_claim: &'static str,
}

impl Sm120IssueEnvelopeReport {
    pub(super) fn from_issue_params(params: IssueModelParams) -> Self {
        let scalar = params.scalar_execution_pipeline();
        Self {
            status: "configured_alpha_inputs",
            profile_id: Sm120AlphaConfig::PROFILE_ID,
            issue_ports_configured: params.issue_ports(),
            dual_issue_configured: params.dual_issue_supported(),
            scalar_latency_configured_cycles: scalar.map(|value| value.latency_cycles),
            scalar_initiation_configured_cycles: scalar
                .map(|value| value.initiation_interval_cycles),
            max_resident_warps_configured: params.max_resident_warps_per_sm(),
            max_resident_blocks_configured: params.max_resident_blocks_per_sm(),
            configuration_source: "IssueModelParams consumed by executor",
            non_claim: "scenario inputs only; not a confidence interval or physical scheduler topology",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Sm120ServiceEnvelopeReport {
    pub status: &'static str,
    pub source_domains: Vec<&'static str>,
    pub route_policy: &'static str,
    pub residency: &'static str,
    pub outstanding_demand: &'static str,
    pub dram_service: &'static str,
    pub geometry_status: &'static str,
    pub partition_mapping_status: &'static str,
    pub mshr_count: &'static str,
    pub queue_policy: &'static str,
    pub non_claim: &'static str,
}

impl Sm120ServiceEnvelopeReport {
    pub(super) fn observation_only() -> Self {
        Self {
            status: "logical_route_service_no_geometry_fit",
            source_domains: vec!["trace_memory_coherence", "ncu_raw_l1_l2_dram_sectors"],
            route_policy: "trace_route_and_static_width",
            residency: "single_warp_dense_service_arm_only",
            outstanding_demand: "logical_route_frontier",
            dram_service: "logical_global_service_envelope",
            geometry_status: "typed_gap",
            partition_mapping_status: "typed_gap",
            mshr_count: "typed_gap",
            queue_policy: "per_route_logical_fifo",
            non_claim: "logical service envelope only; no physical cache/DRAM topology or native completion",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Sm120CompletionLedgerReport {
    pub status: &'static str,
    pub issue: &'static str,
    pub transaction_complete: &'static str,
    pub barrier_arrival: &'static str,
    pub wait_or_test: &'static str,
    pub destination_visibility: &'static str,
    pub payload_observation: &'static str,
    pub affected_async_records: u64,
    pub non_claim: &'static str,
}

impl Sm120CompletionLedgerReport {
    pub(super) fn provisional(affected_async_records: u64) -> Self {
        Self {
            status: "typed_alpha_completion_ledger",
            issue: "observed_in_trace",
            transaction_complete: "unobserved_sm120",
            barrier_arrival: "model_relative_if_decoded",
            wait_or_test: "model_relative_if_decoded",
            destination_visibility: "unobserved_without_consumer_payload_observation",
            payload_observation: "not_present_in_alpha_trace_contract",
            affected_async_records,
            non_claim: "ledger separates semantic edges but does not supply native completion timing",
        }
    }
}

impl TimingReport {
    /// Project the existing simulator-only timing fields into the factorized
    /// frontier contract. Hardware/profiler endpoints are intentionally left
    /// unknown; this method does not create a CUDA-event or NCU conversion.
    pub fn latent_frontiers(&self) -> Result<FrontierVector, LatentModelError> {
        // The current SM86 LDGSTS source/destination timing is a diagnostic
        // hypothesis. A live same-binary control differential shows that its
        // completion frontier can overshoot the NCU target by a large amount;
        // do not expose that hypothesis as a prediction lower bound until C3
        // closes an independent async completion witness.
        let async_completion_unvalidated = self.readiness.diagnostics.async_groups.groups_committed
            != 0
            || self.alpha_approximation.is_some();
        let modeled_event = if async_completion_unvalidated {
            FrontierBound::unknown()
        } else {
            self.terminal_modeled_event_time
                .map_or_else(FrontierBound::unknown, FrontierBound::exact)
        };
        let typed_tail = match &self.completion_frontier {
            _ if async_completion_unvalidated => FrontierBound::unknown(),
            CompletionFrontierReport::Complete { cycles } => {
                cycles.map_or_else(FrontierBound::unknown, FrontierBound::exact)
            }
            CompletionFrontierReport::IncompleteUnmodeledRetirementTails {
                lower_bound_cycles,
                ..
            } => lower_bound_cycles.map_or_else(FrontierBound::unknown, FrontierBound::lower_bound),
        };
        let vector = FrontierVector {
            issue: if async_completion_unvalidated {
                FrontierBound::unknown()
            } else {
                FrontierBound::exact(self.diagnostic_last_issue_cycles)
            },
            modeled_event,
            resource_service: FrontierBound::unknown(),
            launch_envelope: FrontierBound::unknown(),
            typed_tail,
            cache_history: FrontierBound::unknown(),
            visibility: FrontierBound::unknown(),
            writeback_service: FrontierBound::unknown(),
            // Native replay starts at cycle zero. The completion endpoint
            // includes required tails, not just the last processed event.
            endpoint_start: FrontierBound::exact(0),
            endpoint_end: typed_tail,
            required_completion: [
                sim_semantics::CompletionRequirement::ModeledEvent,
                sim_semantics::CompletionRequirement::TypedTail,
            ]
            .into_iter()
            .collect(),
        };
        vector.validate()?;
        Ok(vector)
    }

    /// Declare the simulator's own endpoint semantics without implying that
    /// it is the NCU, CUPTI, CUDA-event, or host endpoint.
    pub fn simulator_endpoint_descriptor() -> Result<EndpointDescriptor, LatentModelError> {
        EndpointDescriptor::new(
            ClockDomain::SimulatorCycles,
            "simulator_cycle_origin_zero",
            "simulator_completion_including_required_retirement_tails",
            vec!["simulator_issue_event_queue_and_required_retirement_tails".to_owned()],
            vec![
                "host_submission".to_owned(),
                "ncu_frontend_wait_for_idle".to_owned(),
                "profiler_or_host_observer_tail".to_owned(),
            ],
            "deterministic_sim_engine_event_order",
            "simulator_replay_configuration",
            "architecture_card_model_clock",
        )
    }

    /// Declare Nsight Compute's profiler endpoint semantics as an independent
    /// comparison contract. This descriptor carries no measured duration and
    /// authorizes no conversion to simulator cycles.
    pub fn ncu_endpoint_descriptor() -> Result<EndpointDescriptor, LatentModelError> {
        EndpointDescriptor::new(
            ClockDomain::NsightCompute,
            "ncu_frontend_wait_for_idle_then_grid_start",
            "ncu_grid_completion_semaphore",
            vec![
                "profiled_kernel_grid".to_owned(),
                "ncu_profiler_instrumentation".to_owned(),
            ],
            vec!["host_submission".to_owned()],
            "ncu_profiler_replay_application_endpoint",
            "ncu_cache_control_policy",
            "ncu_clock_control_policy",
        )
    }
}

#[derive(Debug, Serialize)]
pub struct GpcObservationFrontierReport {
    pub lower_bound_cycles: u64,
    pub envelope_cycles: u64,
    pub grid_blocks: u64,
    pub grid_log2: u32,
    pub cycles: u64,
    pub role: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SmActiveObservationFrontierReport {
    pub issue_lower_bound_cycles: u64,
    pub issue_cycles: u64,
    pub terminal_lower_bound_cycles: Option<u64>,
    pub terminal_cycles: Option<u64>,
    pub envelope_cycles: u64,
    pub role: &'static str,
}

impl From<SmActiveObservationFrontier> for SmActiveObservationFrontierReport {
    fn from(value: SmActiveObservationFrontier) -> Self {
        Self {
            issue_lower_bound_cycles: value.issue_lower_bound_cycles,
            issue_cycles: value.issue_cycles,
            terminal_lower_bound_cycles: value.terminal_lower_bound_cycles,
            terminal_cycles: value.terminal_cycles,
            envelope_cycles: value.envelope_cycles,
            role: "architecture_scoped_sm_active_observation_boundary_not_raw_lower_bound",
        }
    }
}

impl From<GpcObservationFrontier> for GpcObservationFrontierReport {
    fn from(value: GpcObservationFrontier) -> Self {
        Self {
            lower_bound_cycles: value.lower_bound_cycles,
            envelope_cycles: value.envelope_cycles,
            grid_blocks: value.grid_blocks,
            grid_log2: value.grid_log2,
            cycles: value.cycles,
            role: "architecture_scoped_gpc_observation_boundary_not_timing_crosswalk",
        }
    }
}
