//! Observation-only Nsight Compute CSV ingestion for direct replay.
//!
//! These values are comparison targets. They never enter issue, memory, or
//! architecture configuration decisions.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct NcuResourceDependency {
    pub issue_instructions: Option<f64>,
    pub load_store_sectors: Option<f64>,
    pub eligible_warps: Option<f64>,
    pub scoreboard_stall_units: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct NcuObservation {
    pub launch_id: u64,
    pub kernel_name: String,
    pub device: String,
    pub block_size: Option<u64>,
    pub grid_size: Option<u64>,
    pub registers_per_thread: Option<u64>,
    pub shared_memory_bytes_per_block: Option<u64>,
    pub static_shared_memory_bytes: Option<u64>,
    pub dynamic_shared_memory_bytes: Option<u64>,
    pub shared_memory_config_size: Option<u64>,
    pub registers_per_thread_allocated: Option<u64>,
    pub shared_memory_bytes_per_block_allocated: Option<u64>,
    pub shared_memory_bytes_per_block_driver: Option<u64>,
    pub occupancy_limit_blocks: Option<u64>,
    pub occupancy_limit_registers: Option<u64>,
    pub occupancy_limit_shared_memory: Option<u64>,
    pub occupancy_limit_warps: Option<u64>,
    pub occupancy_per_block_size: Option<u64>,
    pub occupancy_per_register_count: Option<u64>,
    pub occupancy_per_shared_memory_size: Option<u64>,
    pub gpu_time_ns: Option<f64>,
    /// Sum-duration variant used by the nvprof-scripts/MICRO-style report
    /// extractor. Kept separate from Nsight's average duration field.
    pub gpu_time_sum_ns: Option<f64>,
    pub sm_cycles: Option<f64>,
    pub sm_cycles_active: Option<f64>,
    pub gpc_cycles_max: Option<f64>,
    pub gpc_cycles_avg: Option<f64>,
    pub gpc_cycles_avg_per_second: Option<f64>,
    pub dram_cycles_avg_per_second: Option<f64>,
    pub warp_instructions: Option<f64>,
    /// The .avg roll-up is kept separate because it is a mean across SMSP
    /// instances, not a total instruction count.
    pub warp_instructions_avg: Option<f64>,
    pub fma_instructions: Option<f64>,
    pub alu_instructions: Option<f64>,
    pub adu_instructions: Option<f64>,
    pub cbu_instructions: Option<f64>,
    pub fp64_instructions: Option<f64>,
    pub lsu_instructions: Option<f64>,
    pub tensor_instructions: Option<f64>,
    pub uniform_instructions: Option<f64>,
    pub xu_instructions: Option<f64>,
    /// Number of divergent branch-target events reported by NCU. The metric
    /// unit is `branches` (not threads), so it is kept distinct from the
    /// native trace's unavailable target-participation row.
    pub branch_target_divergent_branches: Option<f64>,
    pub sass_branch_target_divergent_branches: Option<f64>,
    pub branch_target_uniform_branches: Option<f64>,
    pub branch_target_uniform_pct: Option<f64>,
    pub branch_control_instructions: Option<f64>,
    pub sm_ipc: Option<f64>,
    pub mips: Option<f64>,
    /// NCU scheduler samples keyed by the suffix after
    /// `smsp__warps_issue_stalled_`. These are comparison targets only;
    /// unlike the simulator's exclusive readiness causes, NCU samples may
    /// overlap and are not replay inputs.
    pub scheduler_stalls: BTreeMap<String, f64>,
    /// Additional warp/PC/PM sampling families when a capture requests them.
    /// Keys retain the complete NCU metric name because these families use
    /// different sampling domains and cannot be merged with cumulative
    /// `smsp__warps_issue_stalled_*` counters.
    pub scheduler_samples: BTreeMap<String, f64>,
    pub eligible_warps: Option<f64>,
    pub long_scoreboard_stalls: Option<f64>,
    pub short_scoreboard_stalls: Option<f64>,
    pub achieved_warps_per_sm: Option<f64>,
    pub achieved_occupancy_pct: Option<f64>,
    pub smsp_warps_active_per_cycle: Option<f64>,
    /// Optional raw NCU theoretical-warp rows. They are not synthesized from
    /// the carded occupancy projection because some SM86 section versions do
    /// not expose the occupancy-calculator metrics in the counter catalog.
    pub theoretical_max_warps_per_sm: Option<f64>,
    pub theoretical_warps_pct: Option<f64>,
    pub global_load_sectors: Option<f64>,
    pub global_store_sectors: Option<f64>,
    pub global_atomic_sectors: Option<f64>,
    pub global_reduction_sectors: Option<f64>,
    pub global_load_requests: Option<f64>,
    pub global_store_requests: Option<f64>,
    pub global_atomic_requests: Option<f64>,
    pub global_reduction_requests: Option<f64>,
    /// Individual non-global L1TEX request families retained so the unified
    /// HyFiSS-style request aggregate can be audited rather than treated as a
    /// black-box scalar.
    pub local_load_requests: Option<f64>,
    pub local_store_requests: Option<f64>,
    pub surface_load_requests: Option<f64>,
    pub surface_store_requests: Option<f64>,
    pub texture_load_requests: Option<f64>,
    pub surface_reduction_requests: Option<f64>,
    pub surface_atomic_requests: Option<f64>,
    pub shared_load_wavefronts: Option<f64>,
    pub shared_store_wavefronts: Option<f64>,
    pub shared_atomic_wavefronts: Option<f64>,
    pub global_load_lookup_hits: Option<f64>,
    pub global_load_lookup_misses: Option<f64>,
    pub l1_sector_hit_rate_pct: Option<f64>,
    pub l1_global_load_hit_rate_pct: Option<f64>,
    pub l2_sector_hit_rate_pct: Option<f64>,
    /// Nsight's Memory Workload Analysis throughput families, retained as
    /// independent external observations rather than collapsed into one
    /// simulator score.
    pub mem_busy_pct: Option<f64>,
    pub max_bandwidth_pct: Option<f64>,
    pub mem_pipes_busy_pct: Option<f64>,
    pub dram_bytes_per_second: Option<f64>,
    pub dram_read_bytes_per_second: Option<f64>,
    pub dram_throughput_pct: Option<f64>,
    pub l1_read_miss_sectors: Option<f64>,
    pub l1_write_sectors: Option<f64>,
    pub l1_total_requests: Option<f64>,
    pub l2_total_requests: Option<f64>,
    pub dram_read_sectors: Option<f64>,
    pub dram_write_sectors: Option<f64>,
    pub dram_read_bytes: Option<f64>,
    pub dram_write_bytes: Option<f64>,
    pub l2_tex_lookup_sectors: Option<f64>,
    pub l2_tex_lookup_hits: Option<f64>,
    pub resource_dependency: NcuResourceDependency,
}

impl NcuObservation {
    pub fn from_csv(text: &str, selected_launch: Option<u64>) -> Result<Self, NcuError> {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let mut lines = text.lines();
        let header = loop {
            let Some(line) = lines.next() else {
                return Err(NcuError::MissingHeader);
            };
            let row = split_csv_line(line)?;
            if row.first().is_some_and(|field| field == "ID") {
                break row;
            }
        };
        let mut candidates = Vec::new();
        let mut units = None;
        for line in lines {
            let row = split_csv_line(line)?;
            // Wide raw exports carry metric units in the row before launches.
            // Compact, unitless CSV inputs retain their base-unit convention.
            if units.is_none()
                && candidates.is_empty()
                && row.len() == header.len()
                && row.first().is_some_and(String::is_empty)
            {
                units = Some(row);
                continue;
            }
            let Some(id) = row.first().and_then(|field| field.parse::<u64>().ok()) else {
                continue;
            };
            if selected_launch.is_none_or(|wanted| wanted == id) {
                candidates.push((id, row));
            }
        }
        let (launch_id, data) = match candidates.len() {
            0 => {
                return Err(selected_launch.map_or(NcuError::MissingData, NcuError::LaunchNotFound));
            }
            1 => candidates.pop().ok_or(NcuError::MissingData)?,
            _ => return Err(NcuError::MultipleLaunches),
        };
        let value = |name: &str, base_unit: &str| -> Result<Option<f64>, NcuError> {
            let Some(index) = header.iter().position(|field| field == name) else {
                return Ok(None);
            };
            let raw = data.get(index).ok_or(NcuError::ShortRow)?;
            if raw.is_empty() || raw.eq_ignore_ascii_case("N/A") {
                return Ok(None);
            }
            let unit = units.as_ref().map_or("", |row| row[index].as_str());
            let scale = metric_unit_scale(unit, base_unit)
                .ok_or_else(|| NcuError::InvalidUnit(name.to_owned(), unit.to_owned()))?;
            parse_number(raw)
                .map(|value| value * scale)
                .filter(|value| value.is_finite())
                .map(Some)
                .ok_or_else(|| NcuError::InvalidNumber(name.to_owned(), raw.to_owned()))
        };
        let text_value = |names: &[&str]| -> Result<String, NcuError> {
            for name in names {
                let Some(index) = header.iter().position(|field| field == *name) else {
                    continue;
                };
                let raw = data.get(index).ok_or(NcuError::ShortRow)?;
                if !raw.is_empty() && !raw.eq_ignore_ascii_case("N/A") {
                    return Ok(raw.clone());
                }
            }
            Ok(String::new())
        };
        let integer = |name: &str, base_unit: &str| -> Result<Option<u64>, NcuError> {
            let Some(index) = header.iter().position(|field| field == name) else {
                return Ok(None);
            };
            let raw = data.get(index).ok_or(NcuError::ShortRow)?;
            if raw.is_empty() || raw.eq_ignore_ascii_case("N/A") {
                return Ok(None);
            }
            let unit = units.as_ref().map_or("", |row| row[index].as_str());
            let scale_digits = integer_unit_scale(unit, base_unit)
                .ok_or_else(|| NcuError::InvalidUnit(name.to_owned(), unit.to_owned()))?;
            if let Some(integer) = parse_scaled_integer(raw, scale_digits) {
                return Ok(Some(integer));
            }
            // Never route integer fields through f64: values above 2^53 can
            // round before conversion. Exact decimal spellings and explicit
            // decimal Kbyte launch units are handled lexically below.
            Err(NcuError::InvalidInteger(name.to_owned(), raw.to_owned()))
        };
        let global_load_sectors =
            value("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "sector")?;
        let global_store_sectors =
            value("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "sector")?;
        let global_atomic_sectors =
            value("l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum", "sector")?;
        let global_reduction_sectors =
            value("l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum", "sector")?;
        let global_load_requests =
            value("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "request")?;
        let global_store_requests =
            value("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "request")?;
        let global_atomic_requests = value(
            "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
            "request",
        )?;
        let global_reduction_requests = value(
            "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
            "request",
        )?;
        let local_load_requests =
            value("l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum", "request")?;
        let local_store_requests =
            value("l1tex__t_requests_pipe_lsu_mem_local_op_st.sum", "request")?;
        let surface_load_requests = value(
            "l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum",
            "request",
        )?;
        let surface_store_requests = value(
            "l1tex__t_requests_pipe_tex_mem_surface_op_st.sum",
            "request",
        )?;
        let texture_load_requests = value("l1tex__t_requests_pipe_tex_mem_texture.sum", "request")?;
        let surface_reduction_requests = value(
            "l1tex__t_requests_pipe_tex_mem_surface_op_red.sum",
            "request",
        )?;
        let surface_atomic_requests = value(
            "l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum",
            "request",
        )?;
        let shared_load_wavefronts = value(
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
            "wavefront",
        )?;
        let shared_store_wavefronts = value(
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
            "wavefront",
        )?;
        let shared_atomic_wavefronts = value(
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum",
            "wavefront",
        )?;
        let global_load_lookup_hits = value(
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
            "sector",
        )?;
        let global_load_lookup_misses = value(
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
            "sector",
        )?;
        let eligible_warps = value("smsp__warps_eligible.sum", "warp")?;
        let long_scoreboard_stalls =
            value("smsp__warps_issue_stalled_long_scoreboard.sum", "warp")?;
        let short_scoreboard_stalls =
            value("smsp__warps_issue_stalled_short_scoreboard.sum", "warp")?;
        let scheduler_stall_names = [
            "barrier",
            "branch_resolving",
            "dispatch_stall",
            "drain",
            "imc_miss",
            "lg_throttle",
            "long_scoreboard",
            "math_pipe_throttle",
            "membar",
            "mio_throttle",
            "misc",
            "no_instructions",
            "not_selected",
            "selected",
            "short_scoreboard",
            "sleeping",
            "tex_throttle",
            "wait",
        ];
        let mut scheduler_stalls = BTreeMap::new();
        for suffix in scheduler_stall_names {
            if let Some(sample) = value(&format!("smsp__warps_issue_stalled_{suffix}.sum"), "warp")?
            {
                scheduler_stalls.insert(suffix.to_owned(), sample);
            }
        }
        let mut scheduler_samples = BTreeMap::new();
        for name in header.iter().filter(|name| {
            name.starts_with("smsp__")
                && name.contains("warps_issue_stalled_")
                && !name.starts_with("smsp__warps_issue_stalled_")
        }) {
            let Some(index) = header.iter().position(|field| field == name) else {
                continue;
            };
            let unit = units.as_ref().map_or("", |row| row[index].as_str());
            let base_unit = if unit.contains("sample") {
                "sample"
            } else {
                "warp"
            };
            if let Some(sample) = value(name, base_unit)? {
                scheduler_samples.insert(name.clone(), sample);
            }
        }
        // Match the aggregate used by the MICRO/HyFiSS report helper only
        // when the complete L1TEX request decomposition is present. The
        // global-only request fields above are intentionally not mislabeled as
        // a unified L1 total.
        let l1_request_components = [
            local_load_requests,
            global_load_requests,
            surface_load_requests,
            texture_load_requests,
            global_store_requests,
            local_store_requests,
            surface_store_requests,
            global_reduction_requests,
            surface_reduction_requests,
            global_atomic_requests,
            surface_atomic_requests,
        ];
        let l1_component_total = l1_request_components
            .into_iter()
            .try_fold(0.0, |total, component| component.map(|value| total + value));
        // GA102 exposes the canonical unified request counter directly. Use
        // it when present; retain the complete HyFiSS-style class sum as a
        // fallback for older/section-specific exports.
        let l1_total_requests = value("l1tex__t_requests.sum", "request")?.or(l1_component_total);
        let gpc_cycles_avg_per_second = value("gpc__cycles_elapsed.avg.per_second", "cycle/s")?;
        let dram_cycles_avg_per_second = value("dram__cycles_elapsed.avg.per_second", "cycle/s")?;
        Ok(Self {
            launch_id,
            kernel_name: text_value(&["launch__kernel_name", "Kernel Name"])?,
            device: text_value(&["device__attribute_display_name"])?,
            block_size: integer("launch__block_size", "")?,
            grid_size: integer("launch__grid_size", "")?,
            registers_per_thread: integer("launch__registers_per_thread", "register/thread")?,
            shared_memory_bytes_per_block: integer("launch__shared_mem_per_block", "byte/block")?,
            static_shared_memory_bytes: integer(
                "launch__shared_mem_per_block_static",
                "byte/block",
            )?,
            dynamic_shared_memory_bytes: integer(
                "launch__shared_mem_per_block_dynamic",
                "byte/block",
            )?,
            shared_memory_config_size: integer("launch__shared_mem_config_size", "byte")?,
            registers_per_thread_allocated: integer(
                "launch__registers_per_thread_allocated",
                "register/thread",
            )?,
            shared_memory_bytes_per_block_allocated: integer(
                "launch__shared_mem_per_block_allocated",
                "byte/block",
            )?,
            shared_memory_bytes_per_block_driver: integer(
                "launch__shared_mem_per_block_driver",
                "byte/block",
            )?,
            occupancy_limit_blocks: integer("launch__occupancy_limit_blocks", "block")?,
            occupancy_limit_registers: integer("launch__occupancy_limit_registers", "block")?,
            occupancy_limit_shared_memory: integer("launch__occupancy_limit_shared_mem", "block")?,
            occupancy_limit_warps: integer("launch__occupancy_limit_warps", "block")?,
            occupancy_per_block_size: integer("launch__occupancy_per_block_size", "")?,
            occupancy_per_register_count: integer("launch__occupancy_per_register_count", "")?,
            occupancy_per_shared_memory_size: integer("launch__occupancy_per_shared_mem_size", "")?,
            gpu_time_ns: value("gpu__time_duration.avg", "ns")?,
            gpu_time_sum_ns: value("gpu__time_duration.sum", "ns")?,
            sm_cycles: value("sm__cycles_elapsed.avg", "cycle")?,
            sm_cycles_active: value("sm__cycles_active.avg", "cycle")?,
            gpc_cycles_max: value("gpc__cycles_elapsed.max", "cycle")?,
            gpc_cycles_avg: value("gpc__cycles_elapsed.avg", "cycle")?,
            gpc_cycles_avg_per_second,
            dram_cycles_avg_per_second,
            warp_instructions: value("smsp__inst_executed.sum", "inst")?,
            warp_instructions_avg: value("smsp__inst_executed.avg", "inst")?,
            fma_instructions: value("smsp__inst_executed_pipe_fma.sum", "inst")?,
            alu_instructions: value("smsp__inst_executed_pipe_alu.sum", "inst")?,
            adu_instructions: value("smsp__inst_executed_pipe_adu.sum", "inst")?,
            cbu_instructions: value("smsp__inst_executed_pipe_cbu.sum", "inst")?,
            fp64_instructions: value("smsp__inst_executed_pipe_fp64.sum", "inst")?,
            lsu_instructions: value("smsp__inst_executed_pipe_lsu.sum", "inst")?,
            tensor_instructions: value("smsp__inst_executed_pipe_tensor.sum", "inst")?,
            uniform_instructions: value("smsp__inst_executed_pipe_uniform.sum", "inst")?,
            xu_instructions: value("smsp__inst_executed_pipe_xu.sum", "inst")?,
            branch_target_divergent_branches: value(
                "smsp__branch_targets_threads_divergent",
                "branches",
            )?,
            sass_branch_target_divergent_branches: value(
                "smsp__sass_branch_targets_threads_divergent.sum",
                "branches",
            )?,
            branch_target_uniform_branches: value(
                "smsp__sass_branch_targets_threads_uniform.sum",
                "branches",
            )?,
            branch_target_uniform_pct: value(
                "smsp__sass_average_branch_targets_threads_uniform.pct",
                "%",
            )?,
            branch_control_instructions: value("smsp__inst_executed_op_branch.sum", "inst")?,
            sm_ipc: value("sm__inst_executed.avg.per_cycle_elapsed", "inst/cycle")?,
            mips: value("sm__inst_executed.avg.per_cycle_elapsed", "inst/cycle")?
                .zip(gpc_cycles_avg_per_second)
                .map(|(ipc, cycles_per_second)| ipc * cycles_per_second / 1_000_000.0),
            scheduler_stalls,
            scheduler_samples,
            eligible_warps,
            long_scoreboard_stalls,
            short_scoreboard_stalls,
            achieved_warps_per_sm: value("sm__warps_active.avg.per_cycle_active", "warp")?,
            achieved_occupancy_pct: value(
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "%",
            )?,
            smsp_warps_active_per_cycle: value("smsp__warps_active.avg.per_cycle_active", "warp")?,
            theoretical_max_warps_per_sm: value("sm__maximum_warps_avg_per_active_cycle", "warp")?,
            theoretical_warps_pct: value("sm__maximum_warps_per_active_cycle_pct", "%")?,
            global_load_sectors,
            global_store_sectors,
            global_atomic_sectors,
            global_reduction_sectors,
            global_load_requests,
            global_store_requests,
            global_atomic_requests,
            global_reduction_requests,
            local_load_requests,
            local_store_requests,
            surface_load_requests,
            surface_store_requests,
            texture_load_requests,
            surface_reduction_requests,
            surface_atomic_requests,
            shared_load_wavefronts,
            shared_store_wavefronts,
            shared_atomic_wavefronts,
            global_load_lookup_hits,
            global_load_lookup_misses,
            l1_sector_hit_rate_pct: value("l1tex__t_sector_hit_rate.pct", "%")?,
            l1_global_load_hit_rate_pct: value(
                "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
                "%",
            )?,
            l2_sector_hit_rate_pct: value("lts__t_sector_hit_rate.pct", "%")?,
            mem_busy_pct: value(
                "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
                "%",
            )?,
            max_bandwidth_pct: value(
                "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed",
                "%",
            )?,
            mem_pipes_busy_pct: value(
                "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed",
                "%",
            )?,
            dram_bytes_per_second: value("dram__bytes.sum.per_second", "byte/s")?,
            dram_read_bytes_per_second: value("dram__bytes_read.sum.per_second", "byte/s")?,
            dram_throughput_pct: value("dram__throughput.avg.pct_of_peak_sustained_elapsed", "%")?,
            l1_read_miss_sectors: value(
                "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum",
                "sector",
            )?,
            l1_write_sectors: value(
                "l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum",
                "sector",
            )?,
            l1_total_requests,
            l2_total_requests: value("lts__t_requests_srcunit_tex.sum", "request")?,
            dram_read_sectors: value("dram__sectors_read.sum", "sector")?,
            dram_write_sectors: value("dram__sectors_write.sum", "sector")?,
            dram_read_bytes: value("dram__bytes_read.sum", "byte")?,
            dram_write_bytes: value("dram__bytes_write.sum", "byte")?,
            l2_tex_lookup_sectors: value("lts__t_sectors_srcunit_tex.sum", "sector")?,
            l2_tex_lookup_hits: value("lts__t_sectors_srcunit_tex_lookup_hit.sum", "sector")?,
            resource_dependency: NcuResourceDependency {
                issue_instructions: value("smsp__inst_executed.sum", "inst")?,
                load_store_sectors: match (global_load_sectors, global_store_sectors) {
                    (Some(load), Some(store)) => Some(load + store),
                    _ => None,
                },
                eligible_warps,
                scoreboard_stall_units: match (long_scoreboard_stalls, short_scoreboard_stalls) {
                    (Some(long), Some(short)) => Some(long + short),
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                },
            },
        })
    }
}

#[derive(Debug)]
pub enum NcuError {
    MissingHeader,
    MissingData,
    MultipleLaunches,
    LaunchNotFound(u64),
    ShortRow,
    MalformedCsv(String),
    InvalidNumber(String, String),
    InvalidInteger(String, String),
    InvalidUnit(String, String),
}

impl fmt::Display for NcuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingHeader => write!(f, "NCU CSV has no raw metric header"),
            Self::MissingData => write!(f, "NCU CSV has no numeric data row"),
            Self::MultipleLaunches => write!(f, "NCU CSV has multiple launches; select one by ID"),
            Self::LaunchNotFound(id) => write!(f, "NCU launch ID {id} was not found"),
            Self::ShortRow => write!(f, "NCU CSV data row is shorter than its header"),
            Self::MalformedCsv(detail) => write!(f, "NCU CSV is malformed: {detail}"),
            Self::InvalidNumber(name, value) => {
                write!(f, "NCU metric {name} is not numeric: {value}")
            }
            Self::InvalidUnit(name, unit) => {
                write!(f, "NCU metric {name} has unsupported unit: {unit}")
            }
            Self::InvalidInteger(name, value) => {
                write!(
                    f,
                    "NCU metric {name} is not a non-negative integer: {value}"
                )
            }
        }
    }
}

impl std::error::Error for NcuError {}

fn metric_unit_scale(unit: &str, base_unit: &str) -> Option<f64> {
    // NCU has emitted both `/s` and `/second` spellings across section
    // versions. Normalize only this equivalent suffix; all other unit
    // mismatches remain typed errors.
    let unit = unit.replace("/second", "/s");
    let base_unit = base_unit.replace("/second", "/s");
    if unit.is_empty() || unit == base_unit {
        return Some(1.0);
    }
    if base_unit == "ns" {
        return match unit.as_str() {
            "us" | "µs" => Some(1_000.0),
            "ms" => Some(1_000_000.0),
            "s" => Some(1_000_000_000.0),
            _ => None,
        };
    }
    if base_unit == "cycle/s" {
        match unit.as_str() {
            "Hz" | "hz" => return Some(1.0),
            "kHz" | "khz" => return Some(1_000.0),
            "MHz" | "mhz" => return Some(1_000_000.0),
            "GHz" | "ghz" => return Some(1_000_000_000.0),
            _ => {}
        }
    }
    match unit.strip_suffix(&base_unit)? {
        "K" => Some(1_000.0),
        "M" => Some(1_000_000.0),
        "G" => Some(1_000_000_000.0),
        "T" => Some(1_000_000_000_000.0),
        _ => None,
    }
}

fn parse_number(raw: &str) -> Option<f64> {
    let value = raw.trim().replace(',', "").parse::<f64>().ok()?;
    value.is_finite().then_some(value)
}

fn split_csv_line(line: &str) -> Result<Vec<String>, NcuError> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(field.trim().to_owned());
                field.clear();
            }
            _ => field.push(ch),
        }
    }
    if quoted {
        return Err(NcuError::MalformedCsv(
            "unterminated quoted field".to_owned(),
        ));
    }
    fields.push(field.trim().to_owned());
    Ok(fields)
}

fn parse_integer(raw: &str) -> Option<u64> {
    let normalized = raw.trim().replace(',', "");
    if let Ok(integer) = normalized.parse::<u64>() {
        return Some(integer);
    }
    let (whole, fraction) = normalized.split_once('.')?;
    if whole.is_empty() || fraction.is_empty() || !fraction.bytes().all(|byte| byte == b'0') {
        return None;
    }
    whole.parse::<u64>().ok()
}

fn integer_unit_scale(unit: &str, base_unit: &str) -> Option<usize> {
    if unit.is_empty() || unit == base_unit {
        return Some(0);
    }
    // Nsight exports the same occupancy-limit quantity as either `block` or
    // `block/SM` depending on the section/version. Both are integral block
    // counts for this parser and must not be rejected as an unknown unit.
    if base_unit == "block" && unit == "block/SM" {
        return Some(0);
    }
    if base_unit.is_empty() {
        return None;
    }
    match unit.strip_suffix(base_unit)? {
        "K" => Some(3),
        "M" => Some(6),
        "G" => Some(9),
        "T" => Some(12),
        _ => None,
    }
}

/// Decimal SI scaling without rounding through f64. Fractional bytes,
/// exponent spellings, incompatible units and u64 overflow must not alias a
/// different valid launch-resource value.
fn parse_scaled_integer(raw: &str, scale_digits: usize) -> Option<u64> {
    if scale_digits == 0 {
        return parse_integer(raw);
    }
    let multiplier = 10_u128.checked_pow(u32::try_from(scale_digits).ok()?)?;
    let normalized = raw.trim().replace(',', "");
    let (whole, fraction) = match normalized.split_once('.') {
        Some((_, "")) => return None,
        Some(parts) => parts,
        None => (normalized.as_str(), ""),
    };
    if whole.is_empty() || !whole.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let whole = whole.parse::<u128>().ok()?;
    if fraction.len() > scale_digits && fraction[scale_digits..].bytes().any(|byte| byte != b'0') {
        return None;
    }
    let mut fraction_value = fraction[..fraction.len().min(scale_digits)]
        .parse::<u128>()
        .unwrap_or(0);
    for _ in fraction.len().min(scale_digits)..scale_digits {
        fraction_value *= 10;
    }
    let scaled = whole.checked_mul(multiplier)?.checked_add(fraction_value)?;
    u64::try_from(scaled).ok()
}
