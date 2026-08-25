use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(feature = "performance-counters")]
use std::time::Instant;

use sim_direct_run::{CacheScope, MemoryMode, RunConfig, RunError, run};

const USAGE: &str = "Usage: sim-run --trace DIR --control FILE --arch-card FILE \
[--pchase-card FILE] [--mode flat|l1|l1-l2] \
[--cache-scope launch|stream-l2] [--carded-dram] [--kernel-id ID] \
[--memory-service-candidates FILE] [--memory-admission-candidates FILE] \
[--memory-admission-feedback FILE]";

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
    let mut kernel_id = None;
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
            value if value == OsStr::new("--kernel-id") => {
                let raw = take_value(&mut args, "--kernel-id")?;
                let raw = raw
                    .to_str()
                    .ok_or_else(|| RunError::new("arguments", "--kernel-id must be UTF-8"))?;
                kernel_id = Some(raw.parse::<i32>().map_err(|error| {
                    RunError::new("arguments", format!("invalid --kernel-id {raw:?}: {error}"))
                })?);
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
        kernel_id,
        memory_service_candidates,
        memory_admission_candidates,
        memory_admission_feedback,
    }))
}

fn main() -> ExitCode {
    let config = match parse_args(std::env::args_os().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
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
                    println!("{json}");
                    ExitCode::SUCCESS
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
}
