#![allow(clippy::unwrap_used)] // test binary: unwrap is an assertion
//! Registry lockstep: every `sim_emit::DESCRIPTORS` row must bind to a real
//! `tools/local_labels/metric_registry.json` v2 entry with a matching NCU
//! metric string, unit, and numeric shape. This closes the drift surface —
//! a descriptor cannot name a metric that does not exist or misdeclare its
//! contract without failing here.
//!
//! Rule-1 note: this reads the registry (metric NAMES / contract shapes — the
//! output contract, allowed in the emit tree), never a label VALUE artifact
//! from the gitignored machine-local evidence trees.

use std::path::PathBuf;

use serde_json::Value;
use sim_emit::{DESCRIPTORS, MetricKind, Unit};

fn registry() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/local_labels/metric_registry.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read registry {}: {e}", path.display()));
    serde_json::from_str(&text).expect("registry parses")
}

#[test]
fn every_descriptor_matches_its_registry_contract() {
    let reg = registry();
    let metrics = reg["metrics"].as_array().expect("metrics is an array");

    let mut failures: Vec<String> = Vec::new();
    for d in DESCRIPTORS {
        let Some(entry) = metrics.iter().find(|m| m["key"].as_str() == Some(d.key)) else {
            failures.push(format!("{}: no registry entry with this key", d.key));
            continue;
        };

        let ncu = entry["ncu_metric"].as_str().unwrap_or("<none>");
        if ncu != d.ncu_metric {
            failures.push(format!(
                "{}: ncu_metric mismatch — descriptor {:?} vs registry {:?}",
                d.key, d.ncu_metric, ncu
            ));
        }

        let unit = entry["unit"].as_str().unwrap_or("<none>");
        let unit_ok = match d.unit {
            Unit::Count => unit == "count",
            Unit::Pct => unit == "pct",
        };
        if !unit_ok {
            failures.push(format!(
                "{}: unit mismatch — descriptor {:?} vs registry {:?}",
                d.key, d.unit, unit
            ));
        }

        // Numeric shape must agree with the registry's bound + parser:
        // a Count is unbounded above (valid_max null); a Fraction is a
        // percent stored as a [0,1] value (valid_max 1.0, percent_to_fraction).
        let valid_max = &entry["valid_max"];
        let parser = entry["parser_transform"].as_str().unwrap_or("<none>");
        let kind_ok = match d.kind {
            MetricKind::Count => valid_max.is_null(),
            MetricKind::Fraction => {
                valid_max.as_f64() == Some(1.0) && parser == "percent_to_fraction"
            }
        };
        if !kind_ok {
            failures.push(format!(
                "{}: kind {:?} inconsistent with registry valid_max={} parser_transform={:?}",
                d.key, d.kind, valid_max, parser
            ));
        }

        // A metric we EMIT as a prediction must be a target, not an input
        // (Rule 1's model_io_role field; the guard's role assertion).
        let role = entry["model_io_role"].as_str().unwrap_or("<none>");
        if role != "target" {
            failures.push(format!(
                "{}: model_io_role is {:?}, expected \"target\" for an emitted prediction",
                d.key, role
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "descriptor/registry lockstep failures:\n{}",
        failures.join("\n")
    );
}
