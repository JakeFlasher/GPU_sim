//! Execute the bounded functional per-lane-PC ITS feasibility program.
use std::env;
use std::path::Path;
use std::process::ExitCode;

use sim_arch::its::{ItsExecutor, ItsProgram};
use sim_arch::spec::ArchitectureSpec;

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: custom_simt_its SPEC.json PROGRAM.json");
        return ExitCode::from(2);
    }
    let result = (|| {
        let spec =
            ArchitectureSpec::load(Path::new(&args[0])).map_err(|error| error.to_string())?;
        let program = ItsProgram::load(Path::new(&args[1])).map_err(|error| error.to_string())?;
        let result = ItsExecutor::new(&spec, &program)
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
            eprintln!("custom_simt_its: {error}");
            ExitCode::FAILURE
        }
    }
}
