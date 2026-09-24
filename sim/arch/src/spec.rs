//! Experimental architecture-definition input for user-defined GPU models.
//!
//! This module is deliberately separate from [`crate::card::ArchCard`].  The
//! production replay still consumes the closed, evidence-bound SM86/SM120
//! card vocabulary.  `ArchitectureSpec` is the first small seam for the
//! custom/open-architecture path: it validates an opaque architecture and
//! ISA token, a SIMT execution shape, and named timing resources without
//! silently turning a hypothetical example into a silicon claim. Parameters
//! may be chosen by users; validation here checks shape, not physical accuracy.

use crate::json::strict_json_from_slice;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::Path;

pub const ARCHITECTURE_SPEC_SCHEMA_VERSION: u32 = 1;
const MAX_SPEC_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Compute,
    Memory,
    Synchronization,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    Documented,
    Measured,
    Inferred,
    Unvalidated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExecutionSpec {
    simt_width: u8,
    max_resident_groups: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawResourceSpec {
    name: String,
    kind: ResourceKind,
    issue_width: u32,
    #[serde(default)]
    latency_cycles: Option<u64>,
    #[serde(default = "one")]
    initiation_interval: u64,
    #[serde(default)]
    max_in_flight: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArchitectureSpec {
    schema_version: u32,
    architecture: String,
    isa: String,
    #[serde(default)]
    instruction_set: Vec<String>,
    execution: RawExecutionSpec,
    resources: Vec<RawResourceSpec>,
    evidence: EvidenceStatus,
    #[serde(default)]
    operation_resources: BTreeMap<String, String>,
}

/// A structurally validated definition for custom functional/timing backends.
/// It is intentionally not an `ArchCard`: an evidence declaration in this
/// user input is not independent validation of a physical GPU. Hypothetical
/// parameters are valid for execution and design-space experiments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArchitectureSpec {
    schema_version: u32,
    architecture: String,
    isa: String,
    instruction_set: Vec<String>,
    execution: ExecutionSpec,
    resources: Vec<ResourceSpec>,
    evidence: EvidenceStatus,
    operation_resources: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExecutionSpec {
    simt_width: u8,
    max_resident_groups: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceSpec {
    name: String,
    kind: ResourceKind,
    issue_width: u32,
    latency_cycles: Option<u64>,
    initiation_interval: u64,
    max_in_flight: Option<u32>,
}

impl ArchitectureSpec {
    /// Load a bounded JSON definition through the same validation path as
    /// [`Self::parse`].
    pub fn load(path: &Path) -> Result<Self, SpecError> {
        let file = fs::File::open(path).map_err(|source| SpecError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut bytes = Vec::new();
        file.take(MAX_SPEC_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|source| SpecError::Io {
                path: path.display().to_string(),
                source,
            })?;
        Self::parse(path, &bytes)
    }

    /// Parse an architecture definition without granting it production-card
    /// status.  Callers receive a fully validated immutable value or an
    /// explicit error.
    pub fn parse(path: &Path, bytes: &[u8]) -> Result<Self, SpecError> {
        if bytes.len() as u64 > MAX_SPEC_BYTES {
            return Err(SpecError::ResourceLimit {
                path: path.display().to_string(),
                observed: bytes.len() as u64,
                limit: MAX_SPEC_BYTES,
            });
        }
        let raw: RawArchitectureSpec =
            strict_json_from_slice(bytes).map_err(|source| SpecError::Parse {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawArchitectureSpec) -> Result<Self, SpecError> {
        if raw.schema_version != ARCHITECTURE_SPEC_SCHEMA_VERSION {
            return Err(SpecError::UnsupportedVersion {
                found: raw.schema_version,
                expected: ARCHITECTURE_SPEC_SCHEMA_VERSION,
            });
        }
        validate_token("architecture", &raw.architecture)?;
        validate_token("isa", &raw.isa)?;
        let mut instruction_names = BTreeSet::new();
        for instruction in &raw.instruction_set {
            validate_token("instruction_set[]", instruction)?;
            if !instruction_names.insert(instruction.clone()) {
                return Err(SpecError::DuplicateInstruction(instruction.clone()));
            }
        }
        if !(1..=64).contains(&raw.execution.simt_width) {
            return Err(SpecError::InvalidValue {
                field: "execution.simt_width",
                detail: "must be in 1..=64".to_owned(),
            });
        }
        if !(1..=64).contains(&raw.execution.max_resident_groups) {
            return Err(SpecError::InvalidValue {
                field: "execution.max_resident_groups",
                detail: "must be in 1..=64".to_owned(),
            });
        }
        if raw.resources.is_empty() {
            return Err(SpecError::InvalidValue {
                field: "resources",
                detail: "must contain at least one resource".to_owned(),
            });
        }

        let mut names = BTreeSet::new();
        let mut resources = Vec::with_capacity(raw.resources.len());
        for resource in raw.resources {
            validate_token("resources[].name", &resource.name)?;
            if !names.insert(resource.name.clone()) {
                return Err(SpecError::DuplicateResource(resource.name));
            }
            if resource.issue_width == 0 {
                return Err(SpecError::InvalidValue {
                    field: "resources[].issue_width",
                    detail: "must be positive".to_owned(),
                });
            }
            if resource.latency_cycles == Some(0) {
                return Err(SpecError::InvalidValue {
                    field: "resources[].latency_cycles",
                    detail: "must be positive when present".to_owned(),
                });
            }
            if resource.initiation_interval == 0 || resource.max_in_flight == Some(0) {
                return Err(SpecError::InvalidValue {
                    field: "resources[].initiation_interval/max_in_flight",
                    detail: "must be positive".to_owned(),
                });
            }
            resources.push(ResourceSpec {
                name: resource.name,
                kind: resource.kind,
                issue_width: resource.issue_width,
                latency_cycles: resource.latency_cycles,
                initiation_interval: resource.initiation_interval,
                max_in_flight: resource.max_in_flight,
            });
        }

        for (op, resource) in &raw.operation_resources {
            if !instruction_names.contains(op) || !names.contains(resource) {
                return Err(SpecError::InvalidValue {
                    field: "operation_resources",
                    detail: format!("unknown operation/resource binding {op} -> {resource}"),
                });
            }
        }
        Ok(Self {
            schema_version: raw.schema_version,
            architecture: raw.architecture,
            isa: raw.isa,
            instruction_set: raw.instruction_set,
            execution: ExecutionSpec {
                simt_width: raw.execution.simt_width,
                max_resident_groups: raw.execution.max_resident_groups,
            },
            resources,
            evidence: raw.evidence,
            operation_resources: raw.operation_resources,
        })
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn isa(&self) -> &str {
        &self.isa
    }

    pub fn instruction_set(&self) -> &[String] {
        &self.instruction_set
    }

    pub const fn execution(&self) -> ExecutionSpec {
        self.execution
    }

    pub fn resources(&self) -> &[ResourceSpec] {
        &self.resources
    }

    pub fn operation_resources(&self) -> &BTreeMap<String, String> {
        &self.operation_resources
    }

    pub const fn evidence(&self) -> EvidenceStatus {
        self.evidence
    }
}

impl ExecutionSpec {
    pub const fn simt_width(self) -> u8 {
        self.simt_width
    }

    pub const fn max_resident_groups(self) -> u32 {
        self.max_resident_groups
    }
}

impl ResourceSpec {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn kind(&self) -> ResourceKind {
        self.kind
    }

    pub const fn issue_width(&self) -> u32 {
        self.issue_width
    }

    pub const fn initiation_interval(&self) -> u64 {
        self.initiation_interval
    }

    pub const fn max_in_flight(&self) -> Option<u32> {
        self.max_in_flight
    }

    pub const fn latency_cycles(&self) -> Option<u64> {
        self.latency_cycles
    }
}

fn one() -> u64 {
    1
}

fn validate_token(field: &'static str, token: &str) -> Result<(), SpecError> {
    let valid = !token.is_empty()
        && token.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
        && token.as_bytes()[0].is_ascii_lowercase();
    if valid {
        Ok(())
    } else {
        Err(SpecError::InvalidToken {
            field,
            token: token.to_owned(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error("could not read architecture spec {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("architecture spec {path} is {observed} bytes, exceeding limit {limit}")]
    ResourceLimit {
        path: String,
        observed: u64,
        limit: u64,
    },
    #[error("could not parse architecture spec {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("architecture spec schema {found} is unsupported; expected {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
    #[error("architecture spec field {field} has invalid token {token:?}")]
    InvalidToken { field: &'static str, token: String },
    #[error("architecture spec field {field} is invalid: {detail}")]
    InvalidValue { field: &'static str, detail: String },
    #[error("architecture spec repeats resource {0:?}")]
    DuplicateResource(String),
    #[error("architecture spec repeats instruction {0:?}")]
    DuplicateInstruction(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn custom_example_loads_as_unvalidated_spec() {
        let spec = ArchitectureSpec::parse(
            Path::new("simt_u32_minimal.json"),
            include_bytes!("../../../arch_profiles/specs/simt_u32_minimal.json"),
        )
        .expect("example spec should validate");
        assert_eq!(spec.architecture(), "example_simt8");
        assert_eq!(spec.isa(), "simt_u32_v1");
        assert_eq!(spec.execution().simt_width(), 8);
        assert_eq!(spec.resources().len(), 2);
        assert_eq!(spec.evidence(), EvidenceStatus::Unvalidated);
    }

    #[test]
    fn invalid_shape_and_duplicate_resources_fail_closed() {
        let raw = br#"{
          "schema_version": 1,
          "architecture": "toy",
          "isa": "toy_isa",
          "execution": {"simt_width": 0, "max_resident_groups": 1},
          "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
          "evidence": "unvalidated"
        }"#;
        assert!(matches!(
            ArchitectureSpec::parse(Path::new("bad.json"), raw),
            Err(SpecError::InvalidValue {
                field: "execution.simt_width",
                ..
            })
        ));

        let duplicate = br#"{
          "schema_version": 1,
          "architecture": "toy",
          "isa": "toy_isa",
          "execution": {"simt_width": 8, "max_resident_groups": 1},
          "resources": [
            {"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1},
            {"name":"alu","kind":"memory","issue_width":1,"latency_cycles":2}
          ],
          "evidence": "unvalidated"
        }"#;
        assert!(matches!(
            ArchitectureSpec::parse(Path::new("duplicate.json"), duplicate),
            Err(SpecError::DuplicateResource(name)) if name == "alu"
        ));
    }

    #[test]
    fn unknown_fields_are_rejected_by_the_spec_boundary() {
        let raw = br#"{
          "schema_version": 1,
          "architecture": "toy",
          "isa": "toy_isa",
          "execution": {"simt_width": 8, "max_resident_groups": 1},
          "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
          "evidence": "unvalidated",
          "guess": 4
        }"#;
        assert!(matches!(
            ArchitectureSpec::parse(Path::new("unknown.json"), raw),
            Err(SpecError::Parse { .. })
        ));

        let unknown_latency = br#"{
          "schema_version": 1,
          "architecture": "toy",
          "isa": "toy_isa",
          "execution": {"simt_width": 8, "max_resident_groups": 1},
          "resources": [{"name":"alu","kind":"compute","issue_width":1}],
          "evidence": "unvalidated"
        }"#;
        let spec = ArchitectureSpec::parse(Path::new("unknown_latency.json"), unknown_latency)
            .expect("absent latency is an explicit unknown");
        assert_eq!(spec.resources()[0].latency_cycles(), None);
    }

    #[test]
    fn resident_group_capacity_is_bounded_for_pipeline_masks() {
        let raw = br#"{
          "schema_version": 1,
          "architecture": "toy",
          "isa": "toy_isa",
          "execution": {"simt_width": 8, "max_resident_groups": 65},
          "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
          "evidence": "unvalidated"
        }"#;
        assert!(matches!(
            ArchitectureSpec::parse(Path::new("capacity.json"), raw),
            Err(SpecError::InvalidValue {
                field: "execution.max_resident_groups",
                ..
            })
        ));
    }
}
