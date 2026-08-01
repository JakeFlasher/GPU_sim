#![allow(clippy::unwrap_used)] // test binary: unwrap is an assertion
//! Bidirectional registry lockstep. `DESCRIPTORS` and `TYPED_EXCLUSIONS`
//! partition all 61 active registry rows exactly; both table arms bind the full
//! output contract rather than only naming a key.
//!
//! Rule-1 note: this reads metric names and contract shapes only, never label
//! values or machine-local evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde_json::Value;
use sim_run::emit::{
    DESCRIPTORS, EmittedMetricKey, MetricKind, ModelIoRole, ParserTransform, RegistryValidMax,
    TYPED_EXCLUSIONS, TypedExclusionReason, Unit,
};

fn registry() -> Value {
    let path = PathBuf::from("../../tools/local_labels/metric_registry.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read registry {}: {error}", path.display()));
    serde_json::from_str(&text).expect("registry parses")
}

fn registry_projection() -> Value {
    let path = PathBuf::from("../../eval/contracts/metric_registry.7plus1.v1.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read projection {}: {error}", path.display()));
    serde_json::from_str(&text).expect("projection parses")
}

#[allow(clippy::too_many_arguments)]
fn check_contract(
    failures: &mut Vec<String>,
    entry: &Value,
    key: &str,
    ncu_metric: &str,
    unit: Unit,
    valid_max: RegistryValidMax,
    parser_transform: ParserTransform,
    model_io_role: ModelIoRole,
) {
    let registry_ncu = entry["ncu_metric"].as_str().unwrap_or("<none>");
    if registry_ncu != ncu_metric {
        failures.push(format!(
            "{key}: ncu_metric mismatch — table {ncu_metric:?} vs registry {registry_ncu:?}"
        ));
    }

    let registry_unit = entry["unit"].as_str().unwrap_or("<none>");
    if registry_unit != unit.as_str() {
        failures.push(format!(
            "{key}: unit mismatch — table {:?} vs registry {registry_unit:?}",
            unit
        ));
    }

    let registry_valid_max = &entry["valid_max"];
    let valid_max_matches = match valid_max {
        RegistryValidMax::Unbounded => registry_valid_max.is_null(),
        RegistryValidMax::One => registry_valid_max.as_f64() == Some(1.0),
    };
    if !valid_max_matches {
        failures.push(format!(
            "{key}: valid_max mismatch — table {valid_max:?} vs registry {registry_valid_max}"
        ));
    }

    let registry_parser = entry["parser_transform"].as_str().unwrap_or("<none>");
    if registry_parser != parser_transform.as_str() {
        failures.push(format!(
            "{key}: parser_transform mismatch — table {:?} vs registry {registry_parser:?}",
            parser_transform
        ));
    }

    let registry_role = entry["model_io_role"].as_str().unwrap_or("<none>");
    if registry_role != model_io_role.as_str() {
        failures.push(format!(
            "{key}: model_io_role mismatch — table {:?} vs registry {registry_role:?}",
            model_io_role
        ));
    }
}

#[test]
fn registry_is_exactly_partitioned_in_bidirectional_lockstep() {
    let registry = registry();
    let metrics = registry["metrics"].as_array().expect("metrics is an array");
    assert_eq!(metrics.len(), 61, "active registry cardinality drifted");

    let projection = registry_projection();
    let projected_keys = projection["metrics"]
        .as_array()
        .expect("projection metrics is an array")
        .iter()
        .map(|entry| entry["key"].as_str().expect("projection key is a string"))
        .collect::<Vec<_>>();
    let emitted_v2_keys = EmittedMetricKey::V2_ORDER.map(EmittedMetricKey::as_str);
    assert_eq!(
        projected_keys,
        emitted_v2_keys.to_vec(),
        "emitter v2 order must equal the frozen 7+1 projection order"
    );

    let mut failures = Vec::new();
    let mut registry_by_key = BTreeMap::new();
    for entry in metrics {
        let key = entry["key"].as_str().unwrap_or("<missing-key>");
        if registry_by_key.insert(key, entry).is_some() {
            failures.push(format!("registry contains duplicate key {key:?}"));
        }
    }

    let mut descriptor_keys = BTreeSet::new();
    for descriptor in &DESCRIPTORS {
        let key = descriptor.key.as_str();
        if !descriptor_keys.insert(key) {
            failures.push(format!("duplicate descriptor key {key:?}"));
        }
        let Some(entry) = registry_by_key.get(key) else {
            failures.push(format!("{key}: descriptor has no active registry row"));
            continue;
        };
        check_contract(
            &mut failures,
            entry,
            key,
            descriptor.ncu_metric,
            descriptor.unit,
            descriptor.valid_max,
            descriptor.parser_transform,
            descriptor.model_io_role,
        );
        if descriptor.model_io_role != ModelIoRole::Target {
            failures.push(format!("{key}: a descriptor must bind a target row"));
        }
        let kind_matches = match descriptor.kind {
            MetricKind::Count => {
                descriptor.unit == Unit::Count
                    && descriptor.valid_max == RegistryValidMax::Unbounded
                    && descriptor.parser_transform == ParserTransform::IdentityNonnegative
            }
            MetricKind::Fraction => {
                descriptor.unit == Unit::Pct
                    && descriptor.valid_max == RegistryValidMax::One
                    && descriptor.parser_transform == ParserTransform::PercentToFraction
            }
            MetricKind::BindingCycles => {
                descriptor.unit == Unit::Cycles
                    && descriptor.valid_max == RegistryValidMax::Unbounded
                    && descriptor.parser_transform == ParserTransform::IdentityNonnegative
            }
        };
        if !kind_matches {
            failures.push(format!(
                "{key}: descriptor kind {:?} contradicts its registry shape",
                descriptor.kind
            ));
        }
    }

    let mut exclusion_keys = BTreeSet::new();
    let mut reason_counts = BTreeMap::new();
    for exclusion in TYPED_EXCLUSIONS {
        if !exclusion_keys.insert(exclusion.key) {
            failures.push(format!("duplicate typed-exclusion key {:?}", exclusion.key));
        }
        *reason_counts.entry(exclusion.reason).or_insert(0_usize) += 1;
        let Some(entry) = registry_by_key.get(exclusion.key) else {
            failures.push(format!(
                "{}: typed exclusion has no active registry row",
                exclusion.key
            ));
            continue;
        };
        check_contract(
            &mut failures,
            entry,
            exclusion.key,
            exclusion.ncu_metric,
            exclusion.unit,
            exclusion.valid_max,
            exclusion.parser_transform,
            exclusion.model_io_role,
        );
        let role_matches_reason = match exclusion.reason {
            TypedExclusionReason::InputRoleNotEmitted => {
                exclusion.model_io_role == ModelIoRole::Input
            }
            _ => exclusion.model_io_role == ModelIoRole::Target,
        };
        if !role_matches_reason {
            failures.push(format!(
                "{}: exclusion reason {:?} contradicts role {:?}",
                exclusion.key, exclusion.reason, exclusion.model_io_role
            ));
        }
    }

    for key in descriptor_keys.intersection(&exclusion_keys) {
        failures.push(format!(
            "{key}: appears in both descriptors and typed exclusions"
        ));
    }
    for key in registry_by_key.keys() {
        let arms =
            usize::from(descriptor_keys.contains(key)) + usize::from(exclusion_keys.contains(key));
        if arms != 1 {
            failures.push(format!(
                "{key}: active registry row appears in {arms} partition arms, expected exactly one"
            ));
        }
    }
    for key in descriptor_keys.union(&exclusion_keys) {
        if !registry_by_key.contains_key(key) {
            failures.push(format!("{key}: table row is absent from active registry"));
        }
    }

    assert_eq!(DESCRIPTORS.len(), 8, "descriptor cardinality drifted");
    assert_eq!(TYPED_EXCLUSIONS.len(), 53, "exclusion cardinality drifted");
    assert_eq!(
        DESCRIPTORS.len() + TYPED_EXCLUSIONS.len(),
        metrics.len(),
        "partition cardinality does not equal registry cardinality"
    );
    assert_eq!(
        reason_counts,
        BTreeMap::from([
            (TypedExclusionReason::MemUnitUnmodeled, 7),
            (TypedExclusionReason::OpClassUnmodeled, 9),
            (TypedExclusionReason::LabelOnlyNoComposition, 31),
            (TypedExclusionReason::TypedUnavailableRegistryRow, 2),
            (TypedExclusionReason::InputRoleNotEmitted, 4),
        ]),
        "typed-exclusion classification drifted"
    );

    let cycles = DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.key == EmittedMetricKey::ExecCyclesGpcMax)
        .expect("binding cycle descriptor is mandatory");
    assert_eq!(cycles.ncu_metric, "gpc__cycles_elapsed.max");
    assert_eq!(cycles.unit, Unit::Cycles);
    assert_eq!(cycles.kind, MetricKind::BindingCycles);
    assert!(!exclusion_keys.contains("exec_cycles_gpc_max"));

    assert!(
        failures.is_empty(),
        "descriptor/exclusion registry lockstep failures:\n{}",
        failures.join("\n")
    );
}
