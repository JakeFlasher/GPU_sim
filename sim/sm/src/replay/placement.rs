//! Launch placement and replay configuration; immutable placement inputs.

use super::*;

/// Card-derived spatial-placement parameters. Only the architecture/SKU-local SM
/// range crosses into replay; no label or target value is a simulator input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PlacementConfig {
    pub(super) arch: ArchId,
    pub(super) sku: Sku,
    pub(super) sm_count: u32,
    /// Upper bound of the `%smid` identifier domain.  PTX explicitly permits
    /// non-contiguous identifiers, so this is not used as a dense SM count.
    pub(super) sm_id_upper_bound: u32,
    /// Maximum physical warp slot admitted by the runtime launch contract.
    /// This is deliberately sourced from `runtime_limits` when a card has
    /// that block; it is not an issue-width or scheduler hypothesis.
    pub(super) max_resident_warps_per_sm: u32,
    /// Launch-envelope block/thread ceilings.  These are retained even when
    /// no issue model exists; a trace can be admitted for structural checks
    /// without inventing scheduler throughput.
    pub(super) max_resident_blocks_per_sm: u32,
    pub(super) max_threads_per_sm: u32,
    pub(super) max_threads_per_block: u32,
    pub(super) max_shared_bytes_per_sm: u64,
    pub(super) max_shared_bytes_per_block: u64,
    pub(super) max_shared_bytes_per_block_optin: u64,
}

impl PlacementConfig {
    pub(super) fn from_arch_card(
        card: &ArchCard,
        expected_arch: ArchId,
    ) -> Result<Self, ReplayError> {
        let card_error = |source| ReplayError::ArchitectureCardPlacement { source };
        let arch = card.arch_id().map_err(card_error)?;
        require_replay_arch(expected_arch, arch, "architecture card")?;
        let sku = card.sku_id().map_err(card_error)?;
        let sm_count = card.sm_count().map_err(card_error)?;
        // Placement needs two architecture-local *launch-envelope* facts:
        // physical warp-slot capacity and per-SM shared-memory capacity.  A
        // Blackwell card intentionally leaves L1 geometry as a research gap,
        // so consulting `l1_geometry` here would incorrectly couple placement
        // to an unmeasured cache hypothesis.  Prefer the observed CUDA
        // runtime-limit block when present.  The legacy SM86 card predates
        // that block; retain its validated L1 shared-capacity field only as a
        // compatibility fallback, and never use it to admit SM120.
        let (
            max_resident_warps_per_sm,
            sm_id_upper_bound,
            max_resident_blocks_per_sm,
            max_threads_per_sm,
            max_threads_per_block,
            max_shared_bytes_per_sm,
            max_shared_bytes_per_block,
            max_shared_bytes_per_block_optin,
        ) = match card.require_runtime_limits() {
            Ok(limits) => (
                limits.max_resident_warps_per_sm,
                limits.sm_id_upper_bound,
                limits.max_blocks_per_sm,
                limits.max_threads_per_sm,
                limits.max_threads_per_block,
                limits.max_shared_memory_per_sm_bytes,
                limits.max_shared_memory_per_block_bytes,
                limits.max_shared_memory_per_block_optin_bytes,
            ),
            Err(
                gap @ CalibError::BlockResearchGap {
                    block: "runtime_limits",
                    ..
                },
            ) => {
                if arch != ArchId::Sm86 {
                    return Err(ReplayError::ArchitectureCardSharedMemoryCapacity {
                        arch,
                        source: gap,
                    });
                }
                let l1 = card.require_l1_geometry().map_err(|source| {
                    ReplayError::ArchitectureCardSharedMemoryCapacity { arch, source }
                })?;
                let (legacy_warps, legacy_blocks) = match card.require_occupancy_resources() {
                    Ok(occupancy) => (occupancy.max_resident_warps_per_sm, occupancy.max_blocks),
                    Err(CalibError::BlockResearchGap {
                        block: "occupancy", ..
                    }) => arch
                        .issue_hypotheses()
                        .require("legacy physical warp-slot capacity")
                        .map(|hypotheses| {
                            (
                                hypotheses.max_resident_warps_per_sm,
                                hypotheses.max_resident_blocks_per_sm,
                            )
                        })
                        .map_err(|source| ReplayError::WarpCapacityUnavailable { arch, source })?,
                    Err(source) => {
                        return Err(ReplayError::ArchitectureCardSharedMemoryCapacity {
                            arch,
                            source,
                        });
                    }
                };
                (
                    legacy_warps,
                    sm_count,
                    legacy_blocks,
                    legacy_warps.saturating_mul(32),
                    legacy_warps.saturating_mul(32),
                    l1.max_shared_bytes,
                    l1.max_shared_bytes,
                    l1.max_shared_bytes,
                )
            }
            Err(source) => {
                return Err(ReplayError::ArchitectureCardSharedMemoryCapacity { arch, source });
            }
        };
        if max_shared_bytes_per_sm == 0 {
            return Err(ReplayError::SharedMemoryCapacityInvalid {
                arch,
                bytes: max_shared_bytes_per_sm,
            });
        }
        Ok(Self {
            arch,
            sku,
            sm_count,
            sm_id_upper_bound,
            max_resident_warps_per_sm,
            max_resident_blocks_per_sm,
            max_threads_per_sm,
            max_threads_per_block,
            max_shared_bytes_per_sm,
            max_shared_bytes_per_block,
            max_shared_bytes_per_block_optin,
        })
    }

    pub(super) fn shared_memory_residency(
        self,
        kernel: &pb::Kernel,
    ) -> Result<SharedMemoryResidency, ReplayError> {
        let bytes_per_cta = u64::try_from(kernel.size_shared_memory).map_err(|_| {
            ReplayError::SharedMemoryDemandInvalid {
                kernel_id: kernel.id,
                bytes: kernel.size_shared_memory,
            }
        })?;
        if bytes_per_cta > self.max_shared_bytes_per_sm {
            return Err(ReplayError::CtaSharedMemoryCapacityExceeded {
                kernel_id: kernel.id,
                arch: self.arch,
                bytes_per_cta,
                max_bytes_per_sm: self.max_shared_bytes_per_sm,
            });
        }
        if bytes_per_cta > self.max_shared_bytes_per_block_optin {
            return Err(ReplayError::CtaSharedMemoryBlockLimitExceeded {
                kernel_id: kernel.id,
                arch: self.arch,
                bytes_per_cta,
                max_bytes_per_block: self.max_shared_bytes_per_block_optin,
            });
        }
        let block = kernel
            .block_dim
            .as_ref()
            .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
            .map(|dim| {
                u64::try_from(dim.x)
                    .ok()
                    .and_then(|x| u64::try_from(dim.y).ok().and_then(|y| x.checked_mul(y)))
                    .and_then(|xy| u64::try_from(dim.z).ok().and_then(|z| xy.checked_mul(z)))
            })
            .flatten();
        if let Some(threads_per_cta) = block {
            if threads_per_cta > u64::from(self.max_threads_per_block) {
                return Err(ReplayError::CtaThreadCapacityExceeded {
                    kernel_id: kernel.id,
                    arch: self.arch,
                    threads_per_cta,
                    max_threads_per_block: self.max_threads_per_block,
                });
            }
        }
        Ok(SharedMemoryResidency::new(
            self.max_shared_bytes_per_sm,
            bytes_per_cta,
        ))
    }

    pub(super) fn require_traced_sm(self, path: &Path, sm_id: i32) -> Result<(), ReplayError> {
        if sm_id < 0 || u32::try_from(sm_id).unwrap_or(u32::MAX) >= self.sm_id_upper_bound {
            return Err(ReplayError::SmIdOutOfRange {
                path: path.to_path_buf(),
                sm_id,
                arch: self.arch,
                sku: self.sku,
                sm_id_upper_bound: self.sm_id_upper_bound,
            });
        }
        Ok(())
    }

    #[cfg(feature = "campaign")]
    pub(super) fn require_declared_sm(
        self,
        launch_index: u64,
        sm_id: u64,
    ) -> Result<(), ReplayError> {
        if sm_id >= u64::from(self.sm_id_upper_bound) {
            return Err(ReplayError::SamplingSmIdOutOfRange {
                launch_index,
                sm_id,
                arch: self.arch,
                sku: self.sku,
                sm_id_upper_bound: self.sm_id_upper_bound,
            });
        }
        Ok(())
    }
}

/// Exact structural identity of every issue and memory-hierarchy parameter used
/// for one replay. No digest or caller label substitutes for these values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayConfiguration {
    pub(super) issue_params: IssueModelParams,
    pub(super) memory_mode: ReplayMemMode,
    pub(super) memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    pub(super) co_residency: CoResidency,
}

impl ReplayConfiguration {
    /// Build a declaration-side identity while enforcing architecture lockstep
    /// across all nested memory configurations.
    pub fn checked(
        issue_params: IssueModelParams,
        memory_mode: ReplayMemMode,
    ) -> Result<Self, ReplayError> {
        validate_replay_mode_architecture(issue_params.arch, memory_mode)?;
        issue_params.validate_resource_source()?;
        Ok(Self {
            issue_params,
            memory_mode,
            memory_admission_feedback: None,
            co_residency: CoResidency::Modeled,
        })
    }

    /// Build the structural identity for the timing-active schema-10 lane.
    /// The candidate's operand-visibility floor must equal the issue model's
    /// existing read-barrier latency: feedback replaces queue readiness only
    /// and must not silently introduce (or ignore) a second latency source.
    pub fn checked_with_memory_admission_feedback(
        issue_params: IssueModelParams,
        memory_mode: ReplayMemMode,
        candidate: MemoryAdmissionCandidate,
    ) -> Result<Self, ReplayError> {
        if !matches!(memory_mode, ReplayMemMode::L1SharedL2 { .. }) {
            return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
        }
        candidate.validate_feedback(issue_params.read_barrier_latency)?;
        let mut configuration = Self::checked(issue_params, memory_mode)?;
        configuration.memory_admission_feedback = Some(candidate);
        Ok(configuration)
    }

    pub fn arch(&self) -> ArchId {
        self.issue_params.arch
    }

    pub fn issue_params(&self) -> IssueModelParams {
        self.issue_params
    }

    pub fn memory_mode(&self) -> ReplayMemMode {
        self.memory_mode
    }

    /// Timing-active finite admission candidate, when schema 10 is enabled.
    pub fn memory_admission_feedback(&self) -> Option<MemoryAdmissionCandidate> {
        self.memory_admission_feedback
    }

    pub fn co_residency(&self) -> CoResidency {
        self.co_residency
    }

    /// Return the residual architecture-scoped SM-active observation boundary
    /// used by the explicit frontier projection. Once the causal IMC path is
    /// active, the independent 964-cycle legacy residual is not additive: it
    /// represented the missing constant-bank work and would double-charge the
    /// same silicon interval. The current SM86 causal lane therefore exposes a
    /// zero residual while retaining the historical capability for old data.
    pub fn sm_active_activation_cycles(&self) -> u64 {
        if !matches!(self.memory_mode, ReplayMemMode::L1SharedL2 { .. })
            || !matches!(
                self.issue_params.resource_source(),
                IssueResourceSource::ArchitectureDefaults
            )
        {
            return 0;
        }
        if self.issue_params.immediate_constant_cache().is_some() {
            return 0;
        }
        self.issue_params
            .arch
            .sm_active_observation_envelope()
            .require("sm_active_observation_envelope")
            .map_or(0, |envelope| envelope.cycles())
    }
}
