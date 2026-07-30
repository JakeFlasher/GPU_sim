use std::io::Write;
use std::process::ExitCode;

use sim_runner::{RunError, run_manifest};

fn write_error(error: &RunError) -> ExitCode {
    let (document, exit_code) = match error.canonical_json() {
        Ok(document) => (document, error.exit_code()),
        Err(_serialization_error) => (RunError::canonical_serialization_fallback(), 1),
    };
    let mut stderr = std::io::stderr().lock();
    let _write_result = writeln!(stderr, "{document}");
    ExitCode::from(exit_code)
}

fn run() -> Result<(), RunError> {
    let mut args = std::env::args_os();
    let _program = args.next();
    let manifest = args.next().ok_or_else(|| {
        RunError::cli_usage("expected exactly one sim_run_manifest.v1 path argument")
    })?;
    if manifest.is_empty() {
        return Err(RunError::cli_usage(
            "manifest path argument must be nonempty",
        ));
    }
    if args.next().is_some() {
        return Err(RunError::cli_usage(
            "expected exactly one sim_run_manifest.v1 path argument",
        ));
    }
    let output = run_manifest(manifest)?;
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{}", output.canonical_json()).map_err(|source| {
        RunError::OutputWriteFailed {
            reason: format!("could not write canonical result to stdout: {source}"),
        }
    })?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => write_error(&error),
    }
}
