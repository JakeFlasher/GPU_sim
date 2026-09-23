//! Print a structurally checked custom architecture definition; no timing claim.
use std::env;
use std::path::Path;
use std::process::ExitCode;

use sim_arch::spec::ArchitectureSpec;

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: architecture_spec SPEC.json");
        return ExitCode::from(2);
    }
    match ArchitectureSpec::load(Path::new(&args[0]))
        .map_err(|error| error.to_string())
        .and_then(|spec| serde_json::to_string_pretty(&spec).map_err(|error| error.to_string()))
    {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("architecture_spec: {error}");
            ExitCode::FAILURE
        }
    }
}
