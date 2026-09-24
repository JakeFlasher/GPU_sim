//! Run one or more declarative custom SIMT groups through the shared
//! dependency/resource event engine. This is a model-relative backend; it is
//! intentionally separate from native NVBit/SASS replay.

use std::env;
use std::path::Path;
use std::process::ExitCode;

use sim_arch::{simt::SimtProgram, spec::ArchitectureSpec};

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(spec_path) = args.next() else {
        eprintln!("usage: custom-program-run SPEC.json PROGRAM.json...");
        return ExitCode::from(2);
    };
    let programs = args.collect::<Vec<_>>();
    if programs.is_empty() {
        eprintln!("usage: custom-program-run SPEC.json PROGRAM.json...");
        return ExitCode::from(2);
    }
    let result = (|| {
        let spec =
            ArchitectureSpec::load(Path::new(&spec_path)).map_err(|error| error.to_string())?;
        let programs = programs
            .iter()
            .map(|path| SimtProgram::load(Path::new(path)).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let result = sim_sm::custom::run(&spec, &programs).map_err(|error| error.to_string())?;
        serde_json::to_string_pretty(&result).map_err(|error| error.to_string())
    })();
    match result {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("custom-program-run: {error}");
            ExitCode::FAILURE
        }
    }
}
