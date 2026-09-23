//! Execute the bounded functional SIMT timing probe.
use std::env;
use std::path::Path;
use std::process::ExitCode;

use sim_arch::simt::{SimtProgram, SimtTimingExecutor};
use sim_arch::spec::ArchitectureSpec;

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: custom_simt_timing SPEC.json PROGRAM.json");
        return ExitCode::from(2);
    }
    let result = (|| {
        let spec =
            ArchitectureSpec::load(Path::new(&args[0])).map_err(|error| error.to_string())?;
        let program = SimtProgram::load(Path::new(&args[1])).map_err(|error| error.to_string())?;
        let result = SimtTimingExecutor::new(&spec, &program)
            .map_err(|error| error.to_string())?
            .run()
            .map_err(|error| error.to_string())?;
        serde_json::to_string_pretty(&result).map_err(|error| error.to_string())
    })();
    match result {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("custom_simt_timing: {error}");
            ExitCode::FAILURE
        }
    }
}
