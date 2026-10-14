use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(feature = "performance-counters")]
use std::time::Instant;

use sim_direct_run::{
    run, CacheScope, MemoryMode, RunConfig, RunError, Sm120EffectiveL1Config,
    Sm120EffectiveL2Config,
};
use sim_semantics::ServiceStageRole;

const USAGE: &str = "Usage: sim-run --trace DIR --control FILE --arch-card FILE \
[--pchase-card FILE] [--mode flat|l1|l1-l2] \
[--cache-scope launch|stream-l2] [--carded-dram] [--kernel-id ID] \
[--dirty-writeback-capacity-sectors N] \
[--coherence-service-cycles N --coherence-capacity-sectors N] \
[--l1-visibility-protocol] \
[--l1-resident-writeback] \
[--hypothetical-issue-ports N] \
[--architecture-spec FILE] \
[--fast-frontier-only] \
[--sm120-alpha] \
[--sm120-effective-l1 CAPACITY:MSHR|none:HIT:MISS] \
[--sm120-effective-l2 CAPACITY:SLICES:ASSOCIATIVITY] \
[--latent-envelope] \
[--service-stage-role independent_post_replay|residual_closure|replacement_timing] \
[--ncu-shaped-metrics] \
[--ncu-csv FILE] [--ncu-launch ID] \
[--launch-resources FILE] \
[--native-payload-sidecar FILE] \
[--native-payload-events FILE] \
[--predictive-service-profile FILE] \
[--memory-service-candidates FILE] [--memory-admission-candidates FILE] \
[--memory-admission-feedback FILE]\n\
SM120 full replay always uses l1-l2 with effective cache defaults and metric projections.\n\
--mode flat is allowed only with --fast-frontier-only on SM120; l1-only full replay is forbidden.\n\
The SM120 architecture card selects its effective timing model automatically.";

fn write_success_document_to<W: Write>(document: &str, writer: &mut W) -> io::Result<()> {
    writer.write_all(document.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn write_success_document(document: &str) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    write_success_document_to(document, &mut stdout)
}

fn take_value(
    args: &mut impl Iterator<Item = OsString>,
    option: &'static str,
) -> Result<OsString, RunError> {
    args.next()
        .ok_or_else(|| RunError::new("arguments", format!("{option} requires a value")))
}

fn parse_sm120_effective_l1(raw: &str) -> Result<Sm120EffectiveL1Config, RunError> {
    let fields: Vec<_> = raw.split(':').collect();
    if fields.len() != 4 {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l1 expects CAPACITY:MSHR|none:HIT:MISS",
        ));
    }
    let parse_u64 = |field: &str, name: &str| {
        field.parse::<u64>().map_err(|error| {
            RunError::new("arguments", format!("invalid {name} {field:?}: {error}"))
        })
    };
    let line_capacity = parse_u64(fields[0], "effective L1 line capacity")?;
    if line_capacity == 0 {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l1 capacity must be positive",
        ));
    }
    let mshr_count = if fields[1] == "none" {
        None
    } else {
        let value = fields[1].parse::<u32>().map_err(|error| {
            RunError::new(
                "arguments",
                format!("invalid effective L1 MSHR count {:?}: {error}", fields[1]),
            )
        })?;
        if value == 0 {
            return Err(RunError::new(
                "arguments",
                "--sm120-effective-l1 MSHR count must be positive or none",
            ));
        }
        Some(value)
    };
    let hit_latency_cycles = parse_u64(fields[2], "effective L1 hit latency")?;
    let miss_latency_cycles = parse_u64(fields[3], "effective L1 miss latency")?;
    if hit_latency_cycles >= miss_latency_cycles {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l1 hit latency must be below miss latency",
        ));
    }
    Ok(Sm120EffectiveL1Config {
        line_capacity,
        mshr_count,
        hit_latency_cycles,
        miss_latency_cycles,
    })
}

fn parse_sm120_effective_l2(raw: &str) -> Result<Sm120EffectiveL2Config, RunError> {
    let fields: Vec<_> = raw.split(':').collect();
    if fields.len() != 3 {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l2 expects CAPACITY_BYTES:SLICES:ASSOCIATIVITY",
        ));
    }
    let parse_u64 = |field: &str, name: &str| {
        field.parse::<u64>().map_err(|error| {
            RunError::new("arguments", format!("invalid {name} {field:?}: {error}"))
        })
    };
    let capacity_bytes = parse_u64(fields[0], "effective L2 capacity")?;
    let num_slices = fields[1].parse::<u32>().map_err(|error| {
        RunError::new(
            "arguments",
            format!("invalid effective L2 slices {:?}: {error}", fields[1]),
        )
    })?;
    let associativity = fields[2].parse::<u32>().map_err(|error| {
        RunError::new(
            "arguments",
            format!(
                "invalid effective L2 associativity {:?}: {error}",
                fields[2]
            ),
        )
    })?;
    if capacity_bytes == 0 || num_slices == 0 || associativity == 0 {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l2 capacity, slices, and associativity must be positive",
        ));
    }
    Ok(Sm120EffectiveL2Config {
        capacity_bytes,
        line_bytes: 128,
        sector_bytes: 32,
        num_slices,
        associativity,
        replacement: sim_mem::L2Replacement::Lru,
        slice_hash: sim_mem::SliceHash::LineMod,
        write_policy: sim_mem::L2_WRITE_ALLOCATE_POLICY,
    })
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Option<RunConfig>, RunError> {
    let mut trace_dir = None;
    let mut control_bits = None;
    let mut arch_card = None;
    let mut pchase_card = None;
    let mut memory_mode = None;
    let mut cache_scope = CacheScope::Launch;
    let mut carded_dram_bandwidth = false;
    let mut dirty_writeback_capacity_sectors = None;
    let mut coherence_service_cycles = None;
    let mut coherence_capacity_sectors = None;
    let mut l1_visibility_protocol = false;
    let mut l1_resident_writeback = false;
    let mut hypothetical_issue_ports = None;
    let mut architecture_spec = None;
    let mut fast_frontier_only = false;
    let mut sm120_alpha = false;
    let mut sm120_effective_l1 = None;
    let mut sm120_effective_l2 = None;
    let mut latent_envelope = false;
    let mut kernel_id = None;
    let mut ncu_shaped_metrics = false;
    let mut ncu_csv = None;
    let mut ncu_launch = None;
    let mut launch_resources = None;
    let mut native_payload_sidecar = None;
    let mut native_payload_events_path = None;
    let mut predictive_service_profile = None;
    let mut service_stage_role = ServiceStageRole::IndependentPostReplay;
    let mut memory_service_candidates = None;
    let mut memory_admission_candidates = None;
    let mut memory_admission_feedback = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_os_str() {
            value if value == OsStr::new("--help") || value == OsStr::new("-h") => {
                return Ok(None);
            }
            value if value == OsStr::new("--trace") => {
                trace_dir = Some(PathBuf::from(take_value(&mut args, "--trace")?));
            }
            value if value == OsStr::new("--control") => {
                control_bits = Some(PathBuf::from(take_value(&mut args, "--control")?));
            }
            value if value == OsStr::new("--arch-card") => {
                arch_card = Some(PathBuf::from(take_value(&mut args, "--arch-card")?));
            }
            value if value == OsStr::new("--pchase-card") => {
                pchase_card = Some(PathBuf::from(take_value(&mut args, "--pchase-card")?));
            }
            value if value == OsStr::new("--mode") => {
                let raw = take_value(&mut args, "--mode")?;
                let raw = raw
                    .to_str()
                    .ok_or_else(|| RunError::new("arguments", "--mode must be UTF-8"))?;
                memory_mode = Some(raw.parse()?);
            }
            value if value == OsStr::new("--cache-scope") => {
                let raw = take_value(&mut args, "--cache-scope")?;
                let raw = raw
                    .to_str()
                    .ok_or_else(|| RunError::new("arguments", "--cache-scope must be UTF-8"))?;
                cache_scope = raw.parse()?;
            }
            value if value == OsStr::new("--carded-dram") => {
                carded_dram_bandwidth = true;
            }
            value if value == OsStr::new("--dirty-writeback-capacity-sectors") => {
                let raw = take_value(&mut args, "--dirty-writeback-capacity-sectors")?;
                let raw = raw.to_str().ok_or_else(|| {
                    RunError::new(
                        "arguments",
                        "--dirty-writeback-capacity-sectors must be UTF-8",
                    )
                })?;
                let capacity = raw.parse::<u32>().map_err(|error| {
                    RunError::new(
                        "arguments",
                        format!("invalid --dirty-writeback-capacity-sectors {raw:?}: {error}"),
                    )
                })?;
                if capacity == 0 {
                    return Err(RunError::new(
                        "arguments",
                        "--dirty-writeback-capacity-sectors must be positive",
                    ));
                }
                dirty_writeback_capacity_sectors = Some(capacity);
            }
            value if value == OsStr::new("--coherence-service-cycles") => {
                let raw = take_value(&mut args, "--coherence-service-cycles")?;
                let raw = raw.to_str().ok_or_else(|| {
                    RunError::new("arguments", "--coherence-service-cycles must be UTF-8")
                })?;
                let cycles = raw.parse::<u64>().map_err(|error| {
                    RunError::new(
                        "arguments",
                        format!("invalid --coherence-service-cycles {raw:?}: {error}"),
                    )
                })?;
                if cycles == 0 {
                    return Err(RunError::new(
                        "arguments",
                        "--coherence-service-cycles must be positive",
                    ));
                }
                coherence_service_cycles = Some(cycles);
            }
            value if value == OsStr::new("--coherence-capacity-sectors") => {
                let raw = take_value(&mut args, "--coherence-capacity-sectors")?;
                let raw = raw.to_str().ok_or_else(|| {
                    RunError::new("arguments", "--coherence-capacity-sectors must be UTF-8")
                })?;
                let capacity = raw.parse::<u32>().map_err(|error| {
                    RunError::new(
                        "arguments",
                        format!("invalid --coherence-capacity-sectors {raw:?}: {error}"),
                    )
                })?;
                if capacity == 0 {
                    return Err(RunError::new(
                        "arguments",
                        "--coherence-capacity-sectors must be positive",
                    ));
                }
                coherence_capacity_sectors = Some(capacity);
            }
            value if value == OsStr::new("--l1-visibility-protocol") => {
                l1_visibility_protocol = true;
            }
            value if value == OsStr::new("--l1-resident-writeback") => {
                l1_resident_writeback = true;
            }
            value if value == OsStr::new("--kernel-id") => {
                let raw = take_value(&mut args, "--kernel-id")?;
                let raw = raw
                    .to_str()
                    .ok_or_else(|| RunError::new("arguments", "--kernel-id must be UTF-8"))?;
                kernel_id = Some(raw.parse::<i32>().map_err(|error| {
                    RunError::new("arguments", format!("invalid --kernel-id {raw:?}: {error}"))
                })?);
            }
            value if value == OsStr::new("--hypothetical-issue-ports") => {
                let raw = take_value(&mut args, "--hypothetical-issue-ports")?;
                let raw = raw.to_str().ok_or_else(|| {
                    RunError::new("arguments", "--hypothetical-issue-ports must be UTF-8")
                })?;
                let ports = raw.parse::<u32>().map_err(|error| {
                    RunError::new(
                        "arguments",
                        format!("invalid --hypothetical-issue-ports {raw:?}: {error}"),
                    )
                })?;
                if !(1..=4).contains(&ports) {
                    return Err(RunError::new(
                        "arguments",
                        "--hypothetical-issue-ports must be in 1..=4",
                    ));
                }
                hypothetical_issue_ports = Some(ports);
            }
            value if value == OsStr::new("--architecture-spec") => {
                architecture_spec =
                    Some(PathBuf::from(take_value(&mut args, "--architecture-spec")?));
            }
            value if value == OsStr::new("--fast-frontier-only") => {
                fast_frontier_only = true;
            }
            value if value == OsStr::new("--sm120-alpha") => {
                sm120_alpha = true;
            }
            value if value == OsStr::new("--sm120-effective-l1") => {
                let raw = take_value(&mut args, "--sm120-effective-l1")?;
                let raw = raw.to_str().ok_or_else(|| {
                    RunError::new("arguments", "--sm120-effective-l1 must be UTF-8")
                })?;
                sm120_effective_l1 = Some(parse_sm120_effective_l1(raw)?);
            }
            value if value == OsStr::new("--sm120-effective-l2") => {
                let raw = take_value(&mut args, "--sm120-effective-l2")?;
                let raw = raw.to_str().ok_or_else(|| {
                    RunError::new("arguments", "--sm120-effective-l2 must be UTF-8")
                })?;
                sm120_effective_l2 = Some(parse_sm120_effective_l2(raw)?);
            }
            value if value == OsStr::new("--latent-envelope") => {
                latent_envelope = true;
            }
            value if value == OsStr::new("--ncu-shaped-metrics") => {
                ncu_shaped_metrics = true;
            }
            value if value == OsStr::new("--ncu-csv") => {
                ncu_csv = Some(PathBuf::from(take_value(&mut args, "--ncu-csv")?));
            }
            value if value == OsStr::new("--ncu-launch") => {
                let raw = take_value(&mut args, "--ncu-launch")?;
                let raw = raw
                    .to_str()
                    .ok_or_else(|| RunError::new("arguments", "--ncu-launch must be UTF-8"))?;
                ncu_launch = Some(raw.parse::<u64>().map_err(|error| {
                    RunError::new(
                        "arguments",
                        format!("invalid --ncu-launch {raw:?}: {error}"),
                    )
                })?);
            }
            value if value == OsStr::new("--launch-resources") => {
                launch_resources =
                    Some(PathBuf::from(take_value(&mut args, "--launch-resources")?));
            }
            value if value == OsStr::new("--native-payload-sidecar") => {
                native_payload_sidecar = Some(PathBuf::from(take_value(
                    &mut args,
                    "--native-payload-sidecar",
                )?));
            }
            value if value == OsStr::new("--native-payload-events") => {
                native_payload_events_path = Some(PathBuf::from(take_value(
                    &mut args,
                    "--native-payload-events",
                )?));
            }
            value if value == OsStr::new("--predictive-service-profile") => {
                predictive_service_profile = Some(PathBuf::from(take_value(
                    &mut args,
                    "--predictive-service-profile",
                )?));
            }
            value if value == OsStr::new("--service-stage-role") => {
                let raw = take_value(&mut args, "--service-stage-role")?;
                let raw = raw.to_str().ok_or_else(|| {
                    RunError::new("arguments", "--service-stage-role must be UTF-8")
                })?;
                service_stage_role = match raw {
                    "independent_post_replay" => ServiceStageRole::IndependentPostReplay,
                    "residual_closure" => ServiceStageRole::ResidualClosure,
                    "replacement_timing" => ServiceStageRole::ReplacementTiming,
                    _ => {
                        return Err(RunError::new(
                            "arguments",
                            "--service-stage-role must be independent_post_replay, residual_closure, or replacement_timing",
                        ));
                    }
                };
            }
            value if value == OsStr::new("--memory-service-candidates") => {
                memory_service_candidates = Some(PathBuf::from(take_value(
                    &mut args,
                    "--memory-service-candidates",
                )?));
            }
            value if value == OsStr::new("--memory-admission-candidates") => {
                memory_admission_candidates = Some(PathBuf::from(take_value(
                    &mut args,
                    "--memory-admission-candidates",
                )?));
            }
            value if value == OsStr::new("--memory-admission-feedback") => {
                memory_admission_feedback = Some(PathBuf::from(take_value(
                    &mut args,
                    "--memory-admission-feedback",
                )?));
            }
            unknown => {
                return Err(RunError::new(
                    "arguments",
                    format!("unknown option {:?}", unknown.to_string_lossy()),
                ));
            }
        }
    }
    let arch_card = arch_card
        .ok_or_else(|| RunError::new("arguments", "--arch-card is required"))?;
    let memory_mode = match memory_mode {
        Some(mode) => mode,
        None if fast_frontier_only => MemoryMode::Flat,
        None => {
            let card = sim_arch::card::ArchCard::load(&arch_card)
                .map_err(|error| RunError::new("architecture card", error.to_string()))?;
            let arch = card.arch_id()
                .map_err(|error| RunError::new("architecture card", error.to_string()))?;
            if arch == sim_arch::ArchId::Sm120 {
                MemoryMode::L1L2
            } else {
                MemoryMode::Flat
            }
        }
    };
    Ok(Some(RunConfig {
        trace_dir: trace_dir.ok_or_else(|| RunError::new("arguments", "--trace is required"))?,
        control_bits: control_bits
            .ok_or_else(|| RunError::new("arguments", "--control is required"))?,
        arch_card,
        pchase_card,
        memory_mode,
        cache_scope,
        carded_dram_bandwidth,
        dirty_writeback_capacity_sectors,
        coherence_service_cycles,
        coherence_capacity_sectors,
        l1_visibility_protocol,
        l1_resident_writeback,
        hypothetical_issue_ports,
        architecture_spec,
        kernel_id,
        ncu_shaped_metrics,
        fast_frontier_only,
        sm120_alpha,
        sm120_effective_l1,
        sm120_effective_l2,
        ncu_csv,
        ncu_launch,
        launch_resources,
        native_payload_sidecar,
        native_payload_events: None,
        native_payload_events_path,
        latent_envelope,
        predictive_service_profile,
        service_stage_role,
        memory_service_candidates,
        memory_admission_candidates,
        memory_admission_feedback,
    }))
}

fn main() -> ExitCode {
    let config = match parse_args(std::env::args_os().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => {
            return match write_success_document(USAGE) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("output: {error}");
                    ExitCode::FAILURE
                }
            };
        }
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&config) {
        Ok(report) => {
            #[cfg(feature = "performance-counters")]
            let serialization_started = Instant::now();
            match serde_json::to_string_pretty(&report) {
                Ok(json) => {
                    #[cfg(feature = "performance-counters")]
                    if std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS").is_some() {
                        eprintln!(
                            "SIM_PHASE_PERF serialization_ns={}",
                            serialization_started.elapsed().as_nanos()
                        );
                    }
                    match write_success_document(&json) {
                        Ok(()) => ExitCode::SUCCESS,
                        Err(error) => {
                            eprintln!("output: {error}");
                            ExitCode::FAILURE
                        }
                    }
                }
                Err(error) => {
                    eprintln!("output: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(error) => {
            eprintln!("{error}");
            if error.stage() == "arguments" {
                ExitCode::from(2)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}
