//! Compile the bounded `ptx_subset_v2` source frontend into a simt_its_v1 JSON
//! program. This is a functional lowering seam, not an assembler, CUDA
//! compiler, or native SASS generator.
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::process::ExitCode;

use sim_arch::json::strict_json_from_slice;
use sim_arch::ptx_subset::{MAX_PROGRAM_BYTES, MAX_SOURCE_BYTES, compile_subset_with_memory};

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() > limit {
        return Err(format!(
            "{} is {} bytes, limit is {limit}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("{} has no UTF-8 filename", path.display()))?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let mut created = false;
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("{}: {error}", temporary.display()))?;
        created = true;
        file.write_all(bytes)
            .map_err(|error| format!("{}: {error}", temporary.display()))?;
        file.sync_all()
            .map_err(|error| format!("{}: {error}", temporary.display()))?;
        fs::rename(&temporary, path)
            .map_err(|error| format!("{} -> {}: {error}", temporary.display(), path.display()))
    })();
    if result.is_err() && created {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn ensure_program_size(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() > MAX_PROGRAM_BYTES {
        return Err(format!(
            "serialized program is {} bytes, limit is {MAX_PROGRAM_BYTES}",
            bytes.len()
        ));
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if !(3..=5).contains(&args.len()) {
        eprintln!(
            "usage: ptx_subset_compile ARCHITECTURE INPUT.ptx OUTPUT.json [MAX_STEPS] [INITIAL_MEMORY.json]"
        );
        return ExitCode::from(2);
    }
    let result = (|| {
        let architecture = args[0]
            .to_str()
            .ok_or_else(|| "architecture must be UTF-8".to_owned())?
            .to_owned();
        let input = Path::new(&args[1]);
        let output = Path::new(&args[2]);
        let max_steps = args
            .get(3)
            .map(|value| {
                value
                    .to_str()
                    .ok_or_else(|| "max steps must be UTF-8".to_owned())?
                    .parse::<u64>()
                    .map_err(|_| "max steps must be an unsigned integer".to_owned())
            })
            .transpose()?
            .unwrap_or(100_000);
        let source_bytes = read_bounded(input, MAX_SOURCE_BYTES)?;
        let source = String::from_utf8(source_bytes)
            .map_err(|error| format!("{} is not UTF-8: {error}", input.display()))?;
        let initial_memory = if let Some(memory_path) = args.get(4) {
            let memory_path = Path::new(memory_path);
            let memory_bytes = read_bounded(memory_path, MAX_SOURCE_BYTES)?;
            strict_json_from_slice::<BTreeMap<u32, u32>>(&memory_bytes).map_err(|error| {
                format!("{}: invalid memory JSON: {error}", memory_path.display())
            })?
        } else {
            BTreeMap::new()
        };
        let program = compile_subset_with_memory(&source, architecture, max_steps, initial_memory)
            .map_err(|error| error.to_string())?;
        let json = serde_json::to_vec_pretty(&program).map_err(|error| error.to_string())?;
        ensure_program_size(&json)?;
        write_atomically(output, &json)?;
        Ok::<(), String>(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ptx_subset_compile: {error}");
            ExitCode::FAILURE
        }
    }
}
