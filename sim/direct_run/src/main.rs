use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(feature = "performance-counters")]
use std::time::Instant;

use sim_direct_run::{CacheScope, MemoryMode, RunConfig, RunError, run};
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
[--latent-envelope] \
[--service-stage-role independent_post_replay|residual_closure|replacement_timing] \
[--ncu-shaped-metrics] \
[--ncu-csv FILE] [--ncu-launch ID] \
[--launch-resources FILE] \
[--native-payload-sidecar FILE] \
[--native-payload-events FILE] \
[--predictive-service-profile FILE] \
[--memory-service-candidates FILE] [--memory-admission-candidates FILE] \
[--memory-admission-feedback FILE]";

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

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Option<RunConfig>, RunError> {
    let mut trace_dir = None;
    let mut control_bits = None;
    let mut arch_card = None;
    let mut pchase_card = None;
    let mut memory_mode = MemoryMode::Flat;
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
                memory_mode = raw.parse()?;
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
    Ok(Some(RunConfig {
        trace_dir: trace_dir.ok_or_else(|| RunError::new("arguments", "--trace is required"))?,
        control_bits: control_bits
            .ok_or_else(|| RunError::new("arguments", "--control is required"))?,
        arch_card: arch_card
            .ok_or_else(|| RunError::new("arguments", "--arch-card is required"))?,
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

#[cfg(test)]
mod tests {
    use super::*;

    struct ClosedWriter;

    impl Write for ClosedWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn success_output_propagates_closed_stdout() {
        let mut writer = ClosedWriter;
        assert_eq!(
            write_success_document_to("{}", &mut writer)
                .unwrap_err()
                .kind(),
            io::ErrorKind::BrokenPipe
        );
    }

    #[test]
    fn help_is_standalone() {
        assert!(parse_args([OsString::from("--help")]).unwrap().is_none());
    }

    #[test]
    fn required_paths_are_reported() {
        let error = parse_args(std::iter::empty()).unwrap_err();
        assert!(error.to_string().contains("--trace is required"));
    }

    #[test]
    fn cache_scope_defaults_to_launch_and_parses_stream_l2() {
        let base = [
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
        ];
        let default = parse_args(base.clone()).unwrap().unwrap();
        assert_eq!(default.cache_scope, CacheScope::Launch);

        let configured = parse_args(
            base.into_iter()
                .chain([OsString::from("--cache-scope"), OsString::from("stream-l2")]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(configured.cache_scope, CacheScope::StreamL2);
    }

    #[test]
    fn ncu_shaped_metrics_are_opt_in() {
        let base = [
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
        ];
        assert!(
            !parse_args(base.clone())
                .unwrap()
                .unwrap()
                .ncu_shaped_metrics
        );
        assert!(
            parse_args(
                base.into_iter()
                    .chain([OsString::from("--ncu-shaped-metrics",)])
            )
            .unwrap()
            .unwrap()
            .ncu_shaped_metrics
        );
    }

    #[test]
    fn fast_frontier_only_is_opt_in() {
        let base = [
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
        ];
        assert!(
            !parse_args(base.clone())
                .unwrap()
                .unwrap()
                .fast_frontier_only
        );
        assert!(
            parse_args(
                base.into_iter()
                    .chain([OsString::from("--fast-frontier-only")])
            )
            .unwrap()
            .unwrap()
            .fast_frontier_only
        );
    }

    #[test]
    fn latent_envelope_is_opt_in() {
        let base = [
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
        ];
        assert!(!parse_args(base.clone()).unwrap().unwrap().latent_envelope);
        assert!(
            parse_args(
                base.into_iter()
                    .chain([OsString::from("--latent-envelope")])
            )
            .unwrap()
            .unwrap()
            .latent_envelope
        );
    }

    #[test]
    fn service_stage_role_is_explicit_and_defaults_to_sidecar() {
        let base = [
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
        ];
        assert_eq!(
            parse_args(base.clone())
                .unwrap()
                .unwrap()
                .service_stage_role,
            ServiceStageRole::IndependentPostReplay
        );
        let replacement = parse_args(base.into_iter().chain([
            OsString::from("--service-stage-role"),
            OsString::from("replacement_timing"),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            replacement.service_stage_role,
            ServiceStageRole::ReplacementTiming
        );
    }

    #[test]
    fn memory_service_candidate_path_is_explicit() {
        let parsed = parse_args([
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
            OsString::from("--memory-service-candidates"),
            OsString::from("candidates.json"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            parsed.memory_service_candidates,
            Some(PathBuf::from("candidates.json"))
        );
    }

    #[test]
    fn native_payload_event_path_is_explicit() {
        let parsed = parse_args([
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
            OsString::from("--native-payload-events"),
            OsString::from("events.json"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            parsed.native_payload_events_path,
            Some(PathBuf::from("events.json"))
        );
        assert!(parsed.native_payload_events.is_none());
    }

    #[test]
    fn memory_admission_candidate_path_is_explicit() {
        let parsed = parse_args([
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
            OsString::from("--memory-admission-candidates"),
            OsString::from("admission.json"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            parsed.memory_admission_candidates,
            Some(PathBuf::from("admission.json"))
        );
    }

    #[test]
    fn memory_admission_feedback_path_is_explicit() {
        let parsed = parse_args([
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
            OsString::from("--memory-admission-feedback"),
            OsString::from("feedback.json"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            parsed.memory_admission_feedback,
            Some(PathBuf::from("feedback.json"))
        );
    }

    #[test]
    fn dirty_writeback_capacity_path_is_explicit_and_bounded() {
        let parsed = parse_args([
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
            OsString::from("--dirty-writeback-capacity-sectors"),
            OsString::from("4"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.dirty_writeback_capacity_sectors, Some(4));
        let error = parse_args([
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
            OsString::from("--dirty-writeback-capacity-sectors"),
            OsString::from("0"),
        ])
        .expect_err("zero capacity must fail at argument parsing");
        assert_eq!(error.stage(), "arguments");
    }

    #[test]
    fn coherence_service_pair_is_explicit() {
        let parsed = parse_args([
            OsString::from("--trace"),
            OsString::from("trace"),
            OsString::from("--control"),
            OsString::from("control"),
            OsString::from("--arch-card"),
            OsString::from("card"),
            OsString::from("--coherence-service-cycles"),
            OsString::from("2"),
            OsString::from("--coherence-capacity-sectors"),
            OsString::from("8"),
            OsString::from("--l1-visibility-protocol"),
            OsString::from("--l1-resident-writeback"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.coherence_service_cycles, Some(2));
        assert_eq!(parsed.coherence_capacity_sectors, Some(8));
        assert!(parsed.l1_visibility_protocol);
        assert!(parsed.l1_resident_writeback);
    }
}
