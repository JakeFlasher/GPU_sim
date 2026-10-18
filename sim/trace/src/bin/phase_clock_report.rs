//! Report instrumented clock observations from an existing NVBit capture.
use sim_trace::phase_sidecars::PhaseClockEvidence;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(args.next().ok_or("usage: phase_clock_report CAPTURE")?);
    if args.next().is_some() {
        return Err("expected one capture directory".into());
    }
    let evidence = PhaseClockEvidence::load(&root)?;
    serde_json::to_writer(std::io::stdout().lock(), &evidence)?;
    println!();
    Ok(())
}
