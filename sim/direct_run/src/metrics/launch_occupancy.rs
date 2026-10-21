//! Launch-resource and occupancy projection. Report-only; no scheduler/cache mutation.

use super::*;

impl<'a> Sm86MetricState<'a> {
    pub(super) fn add_sm86_occupancy(&mut self) {
        let result = self.result;
        let external = self.external;
        let card = self.card;
        let trace = result.trace_metrics();
        let _sm_count = self.sm_count;
        let _active_sm_count = self.active_sm_count;
        let _clock = self.clock;
        let mut m = &mut self.metrics;
        let max_warps = result
            .per_sm()
            .values()
            .next()
            .map(|s| s.activity.max_resident_warps_per_sm)
            .filter(|v| *v != 0)
            .unwrap_or(1);
        if let Some(resources) = trace.launch_resources {
            for (key, ncu, unit, value) in [
                (
                    "registers_per_thread",
                    Some("launch__registers_per_thread"),
                    "register/thread",
                    Some(resources.registers_per_thread),
                ),
                (
                    "shared_memory_bytes_per_block",
                    None,
                    "byte",
                    Some(resources.shared_memory_bytes_per_block),
                ),
                (
                    "static_shared_memory_bytes",
                    Some("launch__shared_mem_per_block_static"),
                    "byte",
                    resources.static_shared_memory_bytes,
                ),
                (
                    "dynamic_shared_memory_bytes",
                    Some("launch__shared_mem_per_block_dynamic"),
                    "byte",
                    resources.dynamic_shared_memory_bytes,
                ),
                (
                    "shared_memory_carveout_preference_pct",
                    None,
                    "%",
                    resources
                        .shared_memory_carveout_preference_pct
                        .filter(|value| *value >= 0)
                        .map(|value| value as u32),
                ),
                (
                    "driver_occupancy_limit_blocks",
                    None,
                    "block/SM",
                    resources.driver_occupancy_limit_blocks,
                ),
            ] {
                m.insert(
                key.into(),
                match value {
                    Some(value) => Metric::count(
                        ncu,
                        unit,
                        u128::from(value),
                        "captured CUDA function/launch resource metadata; not an NCU measurement",
                    ),
                    None => Metric::unavailable(
                        ncu,
                        unit,
                        "legacy trace has shared-memory total but lacks the static/dynamic split",
                    ),
                },
            );
            }
            if resources.block_threads != 0 {
                const WARP_SIZE: u64 = 32;
                const REGISTERS_PER_SM: u64 = 65_536;
                const REGISTER_ALLOCATION_GRANULARITY: u64 = 256;
                const SHARED_ALLOCATION_GRANULARITY: u64 = 128;
                // NVIDIA's Ampere tuning guide documents a 1 KiB per-block
                // system reservation. Keep it in the occupancy projection, not
                // in the trace's requested static+dynamic byte row.
                const SHARED_SYSTEM_RESERVATION_BYTES: u64 = 1024;
                let warps_per_block = u64::from(resources.block_threads).div_ceil(WARP_SIZE);
                let max_blocks = u64::from(
                    card.require_occupancy_resources()
                        .map(|occupancy| occupancy.max_blocks)
                        .unwrap_or(0),
                );
                let regs_per_warp = u64::from(resources.registers_per_thread)
                    .saturating_mul(WARP_SIZE)
                    .div_ceil(REGISTER_ALLOCATION_GRANULARITY)
                    .saturating_mul(REGISTER_ALLOCATION_GRANULARITY);
                let registers_per_thread_allocated = regs_per_warp / WARP_SIZE;
                let regs_per_block = regs_per_warp.saturating_mul(warps_per_block);
                let registers_per_subpartition = REGISTERS_PER_SM / 4;
                let register_warps_per_subpartition = if regs_per_warp == 0 {
                    0
                } else {
                    registers_per_subpartition
                        .checked_div(regs_per_warp)
                        .unwrap_or(0)
                };
                let register_warps_per_sm = register_warps_per_subpartition.saturating_mul(4);
                let register_limited_blocks = if regs_per_block == 0 {
                    max_blocks
                } else {
                    register_warps_per_sm / warps_per_block.max(1)
                };
                let shared_bytes = u64::from(resources.shared_memory_bytes_per_block);
                let shared_allocated = shared_bytes
                    .div_ceil(SHARED_ALLOCATION_GRANULARITY)
                    .saturating_mul(SHARED_ALLOCATION_GRANULARITY)
                    .saturating_add(SHARED_SYSTEM_RESERVATION_BYTES);
                let shared_capacity = external.map_or_else(
                    || {
                        card.require_l1_geometry()
                            .map(|geometry| geometry.max_shared_bytes)
                            .unwrap_or(0)
                    },
                    |observation| u64::from(observation.shared_memory_config_bytes),
                );
                let shared_limited_blocks = if shared_allocated == 0 {
                    max_blocks
                } else {
                    shared_capacity.checked_div(shared_allocated).unwrap_or(0)
                };
                let max_warps_u64 = u64::from(max_warps);
                let warp_limited_blocks = max_warps_u64 / warps_per_block.max(1);
                let theoretical_blocks = register_limited_blocks
                    .min(shared_limited_blocks)
                    .min(max_blocks)
                    .min(warp_limited_blocks);
                let theoretical_warps = theoretical_blocks.saturating_mul(warps_per_block);
                let resource_semantics = "SM86 occupancy resource projection: 64K 32-bit registers partitioned across four SMSPs, 256-register warp allocation, 128-byte shared allocation, carded shared/warp/block limits; launch metadata and architecture constants, not NCU labels";
                add_count_row(
                    &mut m,
                    "block_threads",
                    Some("launch__block_size"),
                    "thread/block",
                    u64::from(resources.block_threads),
                    "captured block geometry metadata",
                );
                add_count_row(
                    &mut m,
                    "grid_blocks",
                    Some("launch__grid_size"),
                    "block",
                    resources.grid_blocks,
                    "captured grid geometry metadata",
                );
                add_count_row(
                    &mut m,
                    "block_size",
                    Some("launch__block_size"),
                    "thread/block",
                    u64::from(resources.block_threads),
                    "captured block geometry metadata; NCU launch-size crosswalk",
                );
                add_count_row(
                    &mut m,
                    "grid_size",
                    Some("launch__grid_size"),
                    "block",
                    resources.grid_blocks,
                    "captured grid geometry metadata; NCU launch-size crosswalk",
                );
                m.insert(
                "registers_per_thread_allocated".into(),
                Metric::approximate_count(
                    Some("launch__registers_per_thread_allocated"),
                    "register/thread",
                    u128::from(registers_per_thread_allocated),
                    "SM86 256-register warp allocation granularity projected from captured registers/thread",
                ),
            );
                m.insert(
                "shared_memory_bytes_per_block_allocated".into(),
                Metric::approximate_count(
                    Some("launch__shared_mem_per_block_allocated"),
                    "byte/block",
                    u128::from(shared_allocated),
                    "SM86 128-byte allocation granularity plus NVIDIA-documented 1 KiB per-block system reservation projected from captured static+dynamic bytes",
                ),
            );
                m.insert(
                "shared_memory_capacity_bytes".into(),
                Metric::approximate_count(
                    Some("launch__shared_mem_config_size"),
                    "byte/SM",
                    u128::from(shared_capacity),
                    if external.is_some() {
                        "CUPTI driver-selected L1/shared-memory configuration used as an external occupancy boundary"
                    } else {
                        "SM86 maximum shared-memory capacity from the architecture card; runtime carveout is not present in this trace"
                    },
                ),
            );
                for (key, ncu, value, semantics) in [
                    (
                        "occupancy_limit_blocks",
                        "launch__occupancy_limit_blocks",
                        max_blocks,
                        "carded SM86 architectural maximum blocks/SM",
                    ),
                    (
                        "occupancy_limit_registers",
                        "launch__occupancy_limit_registers",
                        register_limited_blocks,
                        "register-limited blocks/SM from SM86 allocation granularity",
                    ),
                    (
                        "occupancy_limit_shared_memory",
                        "launch__occupancy_limit_shared_mem",
                        shared_limited_blocks,
                        "shared-memory-limited blocks/SM from SM86 allocation granularity",
                    ),
                    (
                        "occupancy_limit_warps",
                        "launch__occupancy_limit_warps",
                        warp_limited_blocks,
                        "warp-capacity-limited blocks/SM from carded resident-warp capacity",
                    ),
                ] {
                    m.insert(
                        key.into(),
                        Metric::approximate_count(
                            Some(ncu),
                            "block/SM",
                            u128::from(value),
                            semantics,
                        ),
                    );
                }
                add_count_row(
                    &mut m,
                    "warps_per_block_theoretical",
                    None,
                    "warp/block",
                    warps_per_block,
                    resource_semantics,
                );
                add_count_row(
                    &mut m,
                    "register_limited_blocks_per_sm",
                    None,
                    "block/SM",
                    register_limited_blocks,
                    resource_semantics,
                );
                add_count_row(
                    &mut m,
                    "shared_limited_blocks_per_sm",
                    None,
                    "block/SM",
                    shared_limited_blocks,
                    resource_semantics,
                );
                add_count_row(
                    &mut m,
                    "resource_limited_blocks_per_sm",
                    None,
                    "block/SM",
                    theoretical_blocks,
                    resource_semantics,
                );
                m.insert(
                    "theoretical_occupancy_pct".into(),
                    Metric::ratio(
                        None,
                        "%",
                        u128::from(theoretical_warps) * 100,
                        u128::from(max_warps_u64),
                        true,
                        resource_semantics,
                    ),
                );
            } else {
                for (key, unit, reason) in [
                    (
                        "block_threads",
                        "thread/block",
                        "legacy trace lacks validated block geometry",
                    ),
                    (
                        "block_size",
                        "thread/block",
                        "legacy trace lacks validated block geometry",
                    ),
                    (
                        "grid_size",
                        "block",
                        "legacy trace lacks validated grid geometry",
                    ),
                    (
                        "theoretical_occupancy_pct",
                        "%",
                        "legacy trace lacks validated launch resources",
                    ),
                ] {
                    m.insert(key.to_owned(), Metric::unavailable(None, unit, reason));
                }
            }
        }
    }
}
