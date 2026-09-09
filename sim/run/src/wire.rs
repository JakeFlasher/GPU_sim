//! Closed v2 wire contracts staged ahead of the production runner cutover.

use std::collections::BTreeSet;
use std::num::NonZeroU8;

use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sim_arch::{ArchId, strict_json_from_slice};
use sim_sm::{SmActiveObservationFrontier, TraceBundleIdentity};
use sim_trace::Sha256Digest;

use crate::emit::{
    EMITTED_METRIC_COUNT, EmitOutcome, EmittedMetricKey, MemUnit, MetricPrediction, MetricValue,
    RequiredUnimplementedBlocker, Unavailable,
};
use crate::{CanonicalRelativePath, ResolvedConfigurationWire};

const MANIFEST_SCHEMA_VERSION: u64 = 2;
const RESULT_SCHEMA_VERSION: &str = "sim_result.v2";
const ERROR_SCHEMA_VERSION: &str = "sim_error.v2";
const MAX_LAUNCHES_V2: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub(crate) enum WireError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Validation(String),
}

impl WireError {
    fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub(crate) struct NonEmptyString(String);

impl NonEmptyString {
    fn checked(raw: impl Into<String>, field: &'static str) -> Result<Self, WireError> {
        let raw = raw.into();
        if raw.is_empty() {
            return Err(WireError::validation(format!("{field} must be nonempty")));
        }
        Ok(Self(raw))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for NonEmptyString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        if raw.is_empty() {
            return Err(de::Error::custom("string must be nonempty"));
        }
        Ok(Self(raw))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
struct VisibleAsciiScope(String);

impl VisibleAsciiScope {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for VisibleAsciiScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        if raw.is_empty() || !raw.bytes().all(|byte| (b'!'..=b'~').contains(&byte)) {
            return Err(de::Error::custom(
                "scope must be nonempty visible ASCII without spaces",
            ));
        }
        Ok(Self(raw))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
struct GitCommit(String);

impl GitCommit {
    fn checked(raw: impl Into<String>) -> Result<Self, WireError> {
        let raw = raw.into();
        if raw.len() != 40
            || !raw
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(WireError::validation(
                "sim_git_commit must be exactly 40 lowercase hexadecimal characters",
            ));
        }
        Ok(Self(raw))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
struct JsonU64(u64);

impl JsonU64 {
    fn get(self) -> u64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for JsonU64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let number = serde_json::Number::deserialize(deserializer)?;
        parse_integral_u64(&number.to_string())
            .map(Self)
            .ok_or_else(|| de::Error::custom("expected an unsigned 64-bit integer"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
struct JsonU32(u32);

impl JsonU32 {
    fn get(self) -> u32 {
        self.0
    }
}

impl<'de> Deserialize<'de> for JsonU32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = JsonU64::deserialize(deserializer)?.get();
        u32::try_from(value)
            .map(Self)
            .map_err(|_| de::Error::custom("expected an unsigned 32-bit integer"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
struct PositiveU32(u32);

impl PositiveU32 {
    fn get(self) -> u32 {
        self.0
    }
}

impl<'de> Deserialize<'de> for PositiveU32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = JsonU32::deserialize(deserializer)?.get();
        if value == 0 {
            return Err(de::Error::custom("expected a positive 32-bit integer"));
        }
        Ok(Self(value))
    }
}

fn parse_integral_u64(raw: &str) -> Option<u64> {
    let (negative, unsigned) = match raw.strip_prefix('-') {
        Some(value) => (true, value),
        None => (false, raw),
    };
    let (mantissa, exponent_raw) = match unsigned.find(['e', 'E']) {
        Some(index) => (&unsigned[..index], Some(&unsigned[index + 1..])),
        None => (unsigned, None),
    };

    let mut digits = String::with_capacity(mantissa.len());
    let mut fraction_digits = 0_i64;
    let mut seen_dot = false;
    for byte in mantissa.bytes() {
        match byte {
            b'0'..=b'9' => {
                digits.push(char::from(byte));
                if seen_dot {
                    fraction_digits = fraction_digits.checked_add(1)?;
                }
            }
            b'.' if !seen_dot => seen_dot = true,
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }

    let first_nonzero = digits.bytes().position(|byte| byte != b'0');
    let Some(first_nonzero) = first_nonzero else {
        return Some(0);
    };
    if negative {
        return None;
    }
    digits.drain(..first_nonzero);

    let exponent = match exponent_raw {
        Some(raw) => parse_decimal_exponent(raw)?,
        None => 0,
    };
    let decimal_scale = exponent.checked_sub(fraction_digits)?;
    if decimal_scale < 0 {
        let removed = usize::try_from(decimal_scale.checked_neg()?).ok()?;
        if removed > digits.len()
            || !digits.as_bytes()[digits.len() - removed..]
                .iter()
                .all(|byte| *byte == b'0')
        {
            return None;
        }
        digits.truncate(digits.len() - removed);
    } else {
        let appended = usize::try_from(decimal_scale).ok()?;
        if digits.len().checked_add(appended)? > 20 {
            return None;
        }
        digits.extend(std::iter::repeat_n('0', appended));
    }

    if digits.is_empty() {
        return Some(0);
    }
    digits.parse().ok()
}

fn parse_decimal_exponent(raw: &str) -> Option<i64> {
    if raw.is_empty() {
        return None;
    }
    let (negative, digits) = match raw.as_bytes()[0] {
        b'+' => (false, &raw[1..]),
        b'-' => (true, &raw[1..]),
        _ => (false, raw),
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let significant = digits.trim_start_matches('0');
    let magnitude = if significant.is_empty() {
        0
    } else {
        significant.parse::<i64>().ok()?
    };
    if negative {
        magnitude.checked_neg()
    } else {
        Some(magnitude)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum Sm86Token {
    #[serde(rename = "sm_86")]
    Sm86,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Rtx3090TiToken {
    #[serde(rename = "rtx_3090_ti")]
    Rtx3090Ti,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Dim3V1 {
    x: PositiveU32,
    y: PositiveU32,
    z: PositiveU32,
}

impl Dim3V1 {
    pub(crate) fn tuple(&self) -> (u32, u32, u32) {
        (self.x.get(), self.y.get(), self.z.get())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LaunchKeyV1 {
    workload_id: NonEmptyString,
    matched_execution_id: NonEmptyString,
    process_launch_ordinal: JsonU64,
    kernel_mangled_name: NonEmptyString,
    grid: Dim3V1,
    block: Dim3V1,
    device_id: JsonU32,
    stream_id: JsonU64,
    binary_architecture: Sm86Token,
}

impl LaunchKeyV1 {
    #[cfg(test)]
    pub(crate) fn process_launch_ordinal(&self) -> u64 {
        self.process_launch_ordinal.get()
    }

    pub(crate) fn kernel_mangled_name(&self) -> &str {
        self.kernel_mangled_name.as_str()
    }

    pub(crate) fn grid(&self) -> (u32, u32, u32) {
        self.grid.tuple()
    }

    pub(crate) fn block(&self) -> (u32, u32, u32) {
        self.block.tuple()
    }

    pub(crate) fn device_id(&self) -> u32 {
        self.device_id.get()
    }

    pub(crate) fn stream_id(&self) -> u64 {
        self.stream_id.get()
    }

    pub(crate) const fn binary_architecture(&self) -> ArchId {
        match self.binary_architecture {
            Sm86Token::Sm86 => ArchId::Sm86,
        }
    }

    pub(crate) fn replay_binding(&self) -> ReplayBindingV1 {
        ReplayBindingV1 {
            kernel_mangled_name: self.kernel_mangled_name.clone(),
            grid: self.grid.clone(),
            block: self.block.clone(),
            device_id: self.device_id,
            stream_id: self.stream_id,
            binary_architecture: self.binary_architecture,
        }
    }
}

#[cfg(test)]
pub(crate) fn parse_launch_key_v1(bytes: &[u8]) -> Result<LaunchKeyV1, WireError> {
    Ok(strict_json_from_slice(bytes)?)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReplayBindingV1 {
    kernel_mangled_name: NonEmptyString,
    grid: Dim3V1,
    block: Dim3V1,
    device_id: JsonU32,
    stream_id: JsonU64,
    binary_architecture: Sm86Token,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DramBandwidthV2 {
    Carded,
    Unmodeled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum ReplayModeV2 {
    Flat,
    L1Const,
    L1SharedL2 { dram_bandwidth: DramBandwidthV2 },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ClosedReplayModeV2 {
    WithoutSharedL2(ReplayModeWithoutSharedL2V2),
    SharedL2(ReplayModeSharedL2V2),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayModeWithoutSharedL2V2 {
    mode: ReplayModeWithoutSharedL2Token,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReplayModeWithoutSharedL2Token {
    Flat,
    L1Const,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayModeSharedL2V2 {
    mode: ReplayModeSharedL2Token,
    dram_bandwidth: DramBandwidthV2,
}

#[derive(Deserialize)]
enum ReplayModeSharedL2Token {
    #[serde(rename = "l1_shared_l2")]
    L1SharedL2,
}

impl<'de> Deserialize<'de> for ReplayModeV2 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match ClosedReplayModeV2::deserialize(deserializer)? {
            ClosedReplayModeV2::WithoutSharedL2(mode) => Ok(match mode.mode {
                ReplayModeWithoutSharedL2Token::Flat => Self::Flat,
                ReplayModeWithoutSharedL2Token::L1Const => Self::L1Const,
            }),
            ClosedReplayModeV2::SharedL2(mode) => {
                let ReplayModeSharedL2Token::L1SharedL2 = mode.mode;
                Ok(Self::L1SharedL2 {
                    dram_bandwidth: mode.dram_bandwidth,
                })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestArchitectureV2 {
    arch_id: Sm86Token,
    sku: Rtx3090TiToken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifestConfigurationEvidenceV2 {
    arch_card_sha256: Sha256Digest,
    pchase_card_sha256: Sha256Digest,
}

impl ManifestConfigurationEvidenceV2 {
    pub(crate) fn arch_card_sha256(&self) -> Sha256Digest {
        self.arch_card_sha256
    }

    pub(crate) fn pchase_card_sha256(&self) -> Sha256Digest {
        self.pchase_card_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifestLaunchV2 {
    launch_evidence_manifest_path: CanonicalRelativePath,
    expected_identity_sha256: Sha256Digest,
    launch_key: LaunchKeyV1,
}

impl ManifestLaunchV2 {
    pub(crate) fn launch_evidence_manifest_path(&self) -> &str {
        self.launch_evidence_manifest_path.as_str()
    }

    pub(crate) fn expected_identity_sha256(&self) -> Sha256Digest {
        self.expected_identity_sha256
    }

    pub(crate) fn launch_key(&self) -> &LaunchKeyV1 {
        &self.launch_key
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSimRunManifestV2 {
    schema_version: JsonU64,
    scope: VisibleAsciiScope,
    architecture: ManifestArchitectureV2,
    configuration_evidence: ManifestConfigurationEvidenceV2,
    launches: Vec<ManifestLaunchV2>,
    replay_mode: ReplayModeV2,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SimRunManifestV2 {
    schema_version: u32,
    scope: VisibleAsciiScope,
    architecture: ManifestArchitectureV2,
    configuration_evidence: ManifestConfigurationEvidenceV2,
    launches: Vec<ManifestLaunchV2>,
    replay_mode: ReplayModeV2,
}

impl SimRunManifestV2 {
    pub(crate) fn scope(&self) -> &str {
        self.scope.as_str()
    }

    pub(crate) fn configuration_evidence(&self) -> &ManifestConfigurationEvidenceV2 {
        &self.configuration_evidence
    }

    pub(crate) fn launches(&self) -> &[ManifestLaunchV2] {
        &self.launches
    }

    pub(crate) fn replay_mode(&self) -> ReplayModeV2 {
        self.replay_mode
    }
}

pub(crate) fn parse_sim_run_manifest_v2(bytes: &[u8]) -> Result<SimRunManifestV2, WireError> {
    let raw: RawSimRunManifestV2 = strict_json_from_slice(bytes)?;
    if raw.schema_version.get() != MANIFEST_SCHEMA_VERSION {
        return Err(WireError::validation(format!(
            "schema_version must be exactly {MANIFEST_SCHEMA_VERSION}"
        )));
    }
    if raw.launches.is_empty() || raw.launches.len() > MAX_LAUNCHES_V2 {
        return Err(WireError::validation(format!(
            "launches must contain 1..={MAX_LAUNCHES_V2} entries"
        )));
    }

    let mut evidence_paths = BTreeSet::new();
    let mut launch_keys = BTreeSet::new();
    let mut replay_bindings = BTreeSet::new();
    for launch in &raw.launches {
        if !evidence_paths.insert(launch.launch_evidence_manifest_path.clone()) {
            return Err(WireError::validation(format!(
                "duplicate launch evidence path {:?}",
                launch.launch_evidence_manifest_path.as_str()
            )));
        }
        if !launch_keys.insert(launch.launch_key.clone()) {
            return Err(WireError::validation("duplicate complete launch key"));
        }
        if !replay_bindings.insert(launch.launch_key.replay_binding()) {
            return Err(WireError::validation("duplicate six-field replay binding"));
        }
    }

    Ok(SimRunManifestV2 {
        schema_version: MANIFEST_SCHEMA_VERSION as u32,
        scope: raw.scope,
        architecture: raw.architecture,
        configuration_evidence: raw.configuration_evidence,
        launches: raw.launches,
        replay_mode: raw.replay_mode,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum BuildClaimabilityV2 {
    Claimable { basis: ClaimableBasisV2 },
    Unclaimable { reason: UnclaimableReasonV2 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ClaimableBasisV2 {
    CleanBuild,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum UnclaimableReasonV2 {
    DirtyBuild,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuildIdentityV2 {
    sim_git_commit: GitCommit,
    sim_git_dirty: bool,
    binary_sha256: Sha256Digest,
    rustc_version: NonEmptyString,
    cargo_profile: NonEmptyString,
    claimability: BuildClaimabilityV2,
}

impl BuildIdentityV2 {
    pub(crate) fn checked(
        sim_git_commit: impl Into<String>,
        sim_git_dirty: bool,
        binary_sha256: Sha256Digest,
        rustc_version: impl Into<String>,
        cargo_profile: impl Into<String>,
    ) -> Result<Self, WireError> {
        let claimability = if sim_git_dirty {
            BuildClaimabilityV2::Unclaimable {
                reason: UnclaimableReasonV2::DirtyBuild,
            }
        } else {
            BuildClaimabilityV2::Claimable {
                basis: ClaimableBasisV2::CleanBuild,
            }
        };
        Ok(Self {
            sim_git_commit: GitCommit::checked(sim_git_commit)?,
            sim_git_dirty,
            binary_sha256,
            rustc_version: NonEmptyString::checked(rustc_version, "rustc_version")?,
            cargo_profile: NonEmptyString::checked(cargo_profile, "cargo_profile")?,
            claimability,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArchCardIdentityV2 {
    card_id: NonEmptyString,
    card_sha256: Sha256Digest,
    sm_arch: Sm86Token,
    sku: Rtx3090TiToken,
}

impl ArchCardIdentityV2 {
    pub(crate) fn checked(
        card_id: impl Into<String>,
        card_sha256: Sha256Digest,
    ) -> Result<Self, WireError> {
        Ok(Self {
            card_id: NonEmptyString::checked(card_id, "card_id")?,
            card_sha256,
            sm_arch: Sm86Token::Sm86,
            sku: Rtx3090TiToken::Rtx3090Ti,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConfigurationIdentityV2 {
    configuration_id: NonEmptyString,
    run_manifest_sha256: Sha256Digest,
    resolved_configuration_sha256: Sha256Digest,
    arch_card_sha256: Sha256Digest,
    pchase_card_sha256: Sha256Digest,
}

impl ConfigurationIdentityV2 {
    pub(crate) fn checked(
        configuration_id: impl Into<String>,
        run_manifest_sha256: Sha256Digest,
        resolved_configuration_sha256: Sha256Digest,
        arch_card_sha256: Sha256Digest,
        pchase_card_sha256: Sha256Digest,
    ) -> Result<Self, WireError> {
        Ok(Self {
            configuration_id: NonEmptyString::checked(configuration_id, "configuration_id")?,
            run_manifest_sha256,
            resolved_configuration_sha256,
            arch_card_sha256,
            pchase_card_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MetricMapV2 {
    smsp_inst_executed_sum: EmitOutcome,
    traffic_gmem_ld_sectors: EmitOutcome,
    traffic_gmem_st_sectors: EmitOutcome,
    traffic_dram_read_sectors: EmitOutcome,
    cache_hit_rate_l1tex_read_sector: EmitOutcome,
    cache_hit_rate_l1tex_sector: EmitOutcome,
    cache_hit_rate_l2_sector: EmitOutcome,
    exec_cycles_gpc_max: EmitOutcome,
}

impl MetricMapV2 {
    pub(crate) fn from_predictions(predictions: &[MetricPrediction]) -> Result<Self, WireError> {
        if predictions
            .iter()
            .any(|prediction| prediction.arch() != ArchId::Sm86)
        {
            return Err(WireError::validation(
                "sim_result.v2 metric maps require sm_86 predictions",
            ));
        }
        Self::checked_entries(
            predictions
                .iter()
                .map(|prediction| (prediction.emitted_key(), *prediction.outcome())),
        )
    }

    fn checked_entries(
        entries: impl IntoIterator<Item = (EmittedMetricKey, EmitOutcome)>,
    ) -> Result<Self, WireError> {
        let mut outcomes = [None; EMITTED_METRIC_COUNT];
        let mut entry_count = 0_usize;
        for (key, outcome) in entries {
            entry_count = entry_count
                .checked_add(1)
                .ok_or_else(|| WireError::validation("metric entry count overflow"))?;
            validate_metric_outcome(key, &outcome)?;
            let slot = &mut outcomes[metric_index(key)];
            if slot.replace(outcome).is_some() {
                return Err(WireError::validation(format!(
                    "duplicate metric map key {}",
                    key.as_str()
                )));
            }
        }
        if entry_count != EMITTED_METRIC_COUNT {
            return Err(WireError::validation(format!(
                "metric map must contain exactly {EMITTED_METRIC_COUNT} entries"
            )));
        }
        let [
            Some(smsp_inst_executed_sum),
            Some(traffic_gmem_ld_sectors),
            Some(traffic_gmem_st_sectors),
            Some(traffic_dram_read_sectors),
            Some(cache_hit_rate_l1tex_read_sector),
            Some(cache_hit_rate_l1tex_sector),
            Some(cache_hit_rate_l2_sector),
            Some(exec_cycles_gpc_max),
        ] = outcomes
        else {
            return Err(WireError::validation(
                "metric map does not contain the exact retained 7+1 key set",
            ));
        };

        Ok(Self {
            smsp_inst_executed_sum,
            traffic_gmem_ld_sectors,
            traffic_gmem_st_sectors,
            traffic_dram_read_sectors,
            cache_hit_rate_l1tex_read_sector,
            cache_hit_rate_l1tex_sector,
            cache_hit_rate_l2_sector,
            exec_cycles_gpc_max,
        })
    }
}

const fn metric_index(key: EmittedMetricKey) -> usize {
    match key {
        EmittedMetricKey::SmspInstExecutedSum => 0,
        EmittedMetricKey::TrafficGmemLdSectors => 1,
        EmittedMetricKey::TrafficGmemStSectors => 2,
        EmittedMetricKey::TrafficDramReadSectors => 3,
        EmittedMetricKey::CacheHitRateL1TexReadSector => 4,
        EmittedMetricKey::CacheHitRateL1TexSector => 5,
        EmittedMetricKey::CacheHitRateL2Sector => 6,
        EmittedMetricKey::ExecCyclesGpcMax => 7,
    }
}

fn validate_metric_outcome(key: EmittedMetricKey, outcome: &EmitOutcome) -> Result<(), WireError> {
    let valid = match (key, outcome) {
        (EmittedMetricKey::SmspInstExecutedSum, EmitOutcome::Predicted(MetricValue::Exact(_))) => {
            true
        }
        (
            EmittedMetricKey::TrafficGmemLdSectors | EmittedMetricKey::TrafficGmemStSectors,
            EmitOutcome::Predicted(MetricValue::Exact(_))
            | EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::L1Tex,
            }),
        ) => true,
        (
            EmittedMetricKey::TrafficDramReadSectors,
            EmitOutcome::Predicted(MetricValue::Exact(_))
            | EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::Dram,
            }),
        ) => true,
        (
            EmittedMetricKey::CacheHitRateL1TexReadSector
            | EmittedMetricKey::CacheHitRateL1TexSector,
            EmitOutcome::Predicted(MetricValue::Fraction {
                value,
                numerator,
                denominator,
            }),
        ) => valid_fraction(*value, *numerator, *denominator),
        (
            EmittedMetricKey::CacheHitRateL1TexReadSector
            | EmittedMetricKey::CacheHitRateL1TexSector,
            EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::L1Tex,
            }),
        ) => true,
        (
            EmittedMetricKey::CacheHitRateL2Sector,
            EmitOutcome::Predicted(MetricValue::Fraction {
                value,
                numerator,
                denominator,
            }),
        ) => valid_fraction(*value, *numerator, *denominator),
        (
            EmittedMetricKey::CacheHitRateL2Sector,
            EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::Lts,
            }),
        ) => true,
        (
            EmittedMetricKey::ExecCyclesGpcMax,
            EmitOutcome::RequiredUnimplemented {
                blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
            },
        ) => true,
        _ => false,
    };
    if !valid {
        return Err(WireError::validation(format!(
            "metric {} has an outcome forbidden by sim_result.v2",
            key.as_str()
        )));
    }
    Ok(())
}

fn valid_fraction(value: Option<f64>, numerator: u64, denominator: u64) -> bool {
    if numerator > denominator {
        return false;
    }
    match (denominator, value) {
        (0, None) => numerator == 0,
        (0, Some(_)) | (_, None) => false,
        (_, Some(value)) => {
            exact_fraction_to_f64(numerator, denominator).is_some_and(|expected| value == expected)
        }
    }
}

fn exact_fraction_to_f64(numerator: u64, denominator: u64) -> Option<f64> {
    const FRACTION_BITS: u32 = 52;
    const HIDDEN_BIT: u128 = 1_u128 << FRACTION_BITS;
    const CARRY_BIT: u128 = HIDDEN_BIT << 1;
    const EXPONENT_BIAS: u64 = 1023;

    if denominator == 0 || numerator > denominator {
        return None;
    }
    if numerator == 0 {
        return Some(0.0);
    }
    if numerator == denominator {
        return Some(1.0);
    }

    let bit_gap = denominator.ilog2().checked_sub(numerator.ilog2())?;
    let normalized_numerator = u128::from(numerator).checked_shl(bit_gap)?;
    let mut negative_exponent =
        bit_gap.checked_add(u32::from(normalized_numerator < u128::from(denominator)))?;
    let scale = FRACTION_BITS.checked_add(negative_exponent)?;
    let scaled_numerator = u128::from(numerator).checked_shl(scale)?;
    let wide_denominator = u128::from(denominator);
    let quotient = scaled_numerator / wide_denominator;
    let remainder = scaled_numerator % wide_denominator;
    let twice_remainder = remainder.checked_mul(2)?;
    let round_up = twice_remainder > wide_denominator
        || (twice_remainder == wide_denominator && quotient & 1 == 1);
    let mut rounded = if round_up {
        quotient.checked_add(1)?
    } else {
        quotient
    };

    if rounded == CARRY_BIT {
        rounded >>= 1;
        negative_exponent = negative_exponent.checked_sub(1)?;
    }
    if !(HIDDEN_BIT..CARRY_BIT).contains(&rounded) {
        return None;
    }

    let significand = u64::try_from(rounded).ok()?;
    let fraction = significand.checked_sub(1_u64 << FRACTION_BITS)?;
    let biased_exponent = EXPONENT_BIAS.checked_sub(u64::from(negative_exponent))?;
    let bits = biased_exponent
        .checked_shl(FRACTION_BITS)?
        .checked_add(fraction)?;
    let value = f64::from_bits(bits);
    (value.is_finite() && (0.0..=1.0).contains(&value)).then_some(value)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PerLaunchResultV2 {
    launch_key: LaunchKeyV1,
    metrics: MetricMapV2,
}

impl PerLaunchResultV2 {
    pub(crate) fn new(launch_key: LaunchKeyV1, metrics: MetricMapV2) -> Self {
        Self {
            launch_key,
            metrics,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AggregateV2 {
    scope: &'static str,
    launch_count: u32,
    metrics: MetricMapV2,
}

impl AggregateV2 {
    pub(crate) fn checked(launch_count: usize, metrics: MetricMapV2) -> Result<Self, WireError> {
        if launch_count == 0 || launch_count > MAX_LAUNCHES_V2 {
            return Err(WireError::validation(format!(
                "aggregate launch_count must be in 1..={MAX_LAUNCHES_V2}"
            )));
        }
        Ok(Self {
            scope: "all_launches",
            launch_count: launch_count as u32,
            metrics,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum TerminalModeledEventTimeV2 {
    Modeled { value: u64 },
    Unavailable { reason: TerminalUnavailableReasonV2 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TerminalUnavailableReasonV2 {
    NoModeledIssueOrSystemEvent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PerLaunchDiagnosticV2 {
    launch_key: LaunchKeyV1,
    diagnostic_last_issue_cycles: u64,
    terminal_modeled_event_time: TerminalModeledEventTimeV2,
    #[serde(skip_serializing_if = "Option::is_none")]
    sm_active_observation_frontier: Option<SmActiveObservationFrontierWire>,
    binding_cycles_role: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SmActiveObservationFrontierWire {
    issue_lower_bound_cycles: u64,
    issue_cycles: u64,
    terminal_lower_bound_cycles: Option<u64>,
    terminal_cycles: Option<u64>,
    envelope_cycles: u64,
    role: &'static str,
}

impl From<SmActiveObservationFrontier> for SmActiveObservationFrontierWire {
    fn from(value: SmActiveObservationFrontier) -> Self {
        Self {
            issue_lower_bound_cycles: value.issue_lower_bound_cycles,
            issue_cycles: value.issue_cycles,
            terminal_lower_bound_cycles: value.terminal_lower_bound_cycles,
            terminal_cycles: value.terminal_cycles,
            envelope_cycles: value.envelope_cycles,
            role: "architecture_scoped_sm_active_observation_boundary_not_raw_lower_bound",
        }
    }
}

impl PerLaunchDiagnosticV2 {
    #[allow(dead_code)]
    pub(crate) fn new(
        launch_key: LaunchKeyV1,
        diagnostic_last_issue_cycles: u64,
        terminal_modeled_event_time: Option<u64>,
    ) -> Self {
        Self::with_sm_active_frontier(
            launch_key,
            diagnostic_last_issue_cycles,
            terminal_modeled_event_time,
            None,
        )
    }

    pub(crate) fn with_sm_active_frontier(
        launch_key: LaunchKeyV1,
        diagnostic_last_issue_cycles: u64,
        terminal_modeled_event_time: Option<u64>,
        sm_active_observation_frontier: Option<SmActiveObservationFrontier>,
    ) -> Self {
        let terminal_modeled_event_time = terminal_modeled_event_time.map_or(
            TerminalModeledEventTimeV2::Unavailable {
                reason: TerminalUnavailableReasonV2::NoModeledIssueOrSystemEvent,
            },
            |value| TerminalModeledEventTimeV2::Modeled { value },
        );
        Self {
            launch_key,
            diagnostic_last_issue_cycles,
            terminal_modeled_event_time,
            sm_active_observation_frontier: sm_active_observation_frontier.map(Into::into),
            binding_cycles_role: "diagnostic_only_not_binding_prediction",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InternalPhaseTimingsNsV2 {
    trace_open_ns: u64,
    trace_read_decode_ns: u64,
    initialization_ns: u64,
    replay_event_memory_ns: u64,
    emission_ns: u64,
    result_serialization_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DiagnosticsV2 {
    admissibility: &'static str,
    external_runtime_role: &'static str,
    per_launch: Vec<PerLaunchDiagnosticV2>,
    #[serde(skip_serializing_if = "Option::is_none")]
    internal_phase_timings_ns: Option<InternalPhaseTimingsNsV2>,
}

impl DiagnosticsV2 {
    pub(crate) fn checked(
        per_launch: Vec<PerLaunchDiagnosticV2>,
        internal_phase_timings_ns: Option<InternalPhaseTimingsNsV2>,
    ) -> Result<Self, WireError> {
        if per_launch.is_empty() || per_launch.len() > MAX_LAUNCHES_V2 {
            return Err(WireError::validation(format!(
                "diagnostics.per_launch must contain 1..={MAX_LAUNCHES_V2} entries"
            )));
        }
        let unique: BTreeSet<_> = per_launch
            .iter()
            .map(|diagnostic| &diagnostic.launch_key)
            .collect();
        if unique.len() != per_launch.len() {
            return Err(WireError::validation("duplicate diagnostics launch key"));
        }
        Ok(Self {
            admissibility: "non_admissible",
            external_runtime_role: "internal_timings_do_not_replace_spawn_to_exit_wall_time",
            per_launch,
            internal_phase_timings_ns,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SuccessExitStatusV2 {
    status: &'static str,
    exit_code: u8,
}

impl SuccessExitStatusV2 {
    const fn new() -> Self {
        Self {
            status: "success",
            exit_code: 0,
        }
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SimResultV2 {
    schema_version: &'static str,
    result_sha256: Sha256Digest,
    build_identity: BuildIdentityV2,
    trace_identity: TraceBundleIdentity,
    arch_card_identity: ArchCardIdentityV2,
    configuration_identity: ConfigurationIdentityV2,
    resolved_configuration: ResolvedConfigurationWire,
    launches: Vec<PerLaunchResultV2>,
    aggregate: AggregateV2,
    diagnostics: DiagnosticsV2,
    exit_status: SuccessExitStatusV2,
}

impl SimResultV2 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn checked(
        result_sha256: Sha256Digest,
        build_identity: BuildIdentityV2,
        trace_identity: TraceBundleIdentity,
        arch_card_identity: ArchCardIdentityV2,
        configuration_identity: ConfigurationIdentityV2,
        resolved_configuration: ResolvedConfigurationWire,
        launches: Vec<PerLaunchResultV2>,
        aggregate: AggregateV2,
        diagnostics: DiagnosticsV2,
    ) -> Result<Self, WireError> {
        validate_result_launch_sets(&launches, &aggregate, &diagnostics)?;
        Ok(Self {
            schema_version: RESULT_SCHEMA_VERSION,
            result_sha256,
            build_identity,
            trace_identity,
            arch_card_identity,
            configuration_identity,
            resolved_configuration,
            launches,
            aggregate,
            diagnostics,
            exit_status: SuccessExitStatusV2::new(),
        })
    }
}

fn validate_result_launch_sets(
    launches: &[PerLaunchResultV2],
    aggregate: &AggregateV2,
    diagnostics: &DiagnosticsV2,
) -> Result<(), WireError> {
    if launches.is_empty() || launches.len() > MAX_LAUNCHES_V2 {
        return Err(WireError::validation(format!(
            "result launches must contain 1..={MAX_LAUNCHES_V2} entries"
        )));
    }
    let launch_keys: BTreeSet<_> = launches.iter().map(|launch| &launch.launch_key).collect();
    if launch_keys.len() != launches.len() {
        return Err(WireError::validation("duplicate result launch key"));
    }
    let replay_bindings: BTreeSet<_> = launches
        .iter()
        .map(|launch| launch.launch_key.replay_binding())
        .collect();
    if replay_bindings.len() != launches.len() {
        return Err(WireError::validation(
            "duplicate result six-field replay binding",
        ));
    }
    if usize::try_from(aggregate.launch_count).ok() != Some(launches.len()) {
        return Err(WireError::validation(
            "aggregate launch_count does not equal result launch count",
        ));
    }
    let diagnostic_keys: BTreeSet<_> = diagnostics
        .per_launch
        .iter()
        .map(|diagnostic| &diagnostic.launch_key)
        .collect();
    if launch_keys != diagnostic_keys {
        return Err(WireError::validation(
            "diagnostic launch-key set does not equal result launch-key set",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CliUsageEvidenceV2 {
    provided_positional_argument_count: u32,
    expected_positional_argument_count: u32,
}

impl CliUsageEvidenceV2 {
    pub(crate) const fn new(provided_positional_argument_count: u32) -> Self {
        Self {
            provided_positional_argument_count,
            expected_positional_argument_count: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifestAdmissionEvidenceV2 {
    manifest_path: CanonicalRelativePath,
    manifest_sha256: Sha256Digest,
}

#[cfg(test)]
impl ManifestAdmissionEvidenceV2 {
    pub(crate) fn checked(
        manifest_path: impl Into<String>,
        manifest_sha256: Sha256Digest,
    ) -> Result<Self, WireError> {
        let manifest_path =
            CanonicalRelativePath::checked(manifest_path.into()).map_err(WireError::validation)?;
        Ok(Self {
            manifest_path,
            manifest_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TraceAdmissionEvidenceV2 {
    bundle_id: NonEmptyString,
    bundle_sha256: Sha256Digest,
}

#[cfg(test)]
impl TraceAdmissionEvidenceV2 {
    pub(crate) fn checked(
        bundle_id: impl Into<String>,
        bundle_sha256: Sha256Digest,
    ) -> Result<Self, WireError> {
        Ok(Self {
            bundle_id: NonEmptyString::checked(bundle_id, "bundle_id")?,
            bundle_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArchCardAdmissionEvidenceV2 {
    card_id: NonEmptyString,
    card_sha256: Sha256Digest,
}

#[cfg(test)]
impl ArchCardAdmissionEvidenceV2 {
    pub(crate) fn checked(
        card_id: impl Into<String>,
        card_sha256: Sha256Digest,
    ) -> Result<Self, WireError> {
        Ok(Self {
            card_id: NonEmptyString::checked(card_id, "card_id")?,
            card_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConfigurationResolutionEvidenceV2 {
    configuration_id: NonEmptyString,
    run_manifest_sha256: Sha256Digest,
}

#[cfg(test)]
impl ConfigurationResolutionEvidenceV2 {
    pub(crate) fn checked(
        configuration_id: impl Into<String>,
        run_manifest_sha256: Sha256Digest,
    ) -> Result<Self, WireError> {
        Ok(Self {
            configuration_id: NonEmptyString::checked(configuration_id, "configuration_id")?,
            run_manifest_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReplayFailureEvidenceV2 {
    launch_key: LaunchKeyV1,
}

#[cfg(test)]
impl ReplayFailureEvidenceV2 {
    pub(crate) fn new(launch_key: LaunchKeyV1) -> Self {
        Self { launch_key }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EmissionFailureEvidenceV2 {
    metric_key: EmittedMetricKey,
}

#[cfg(test)]
impl EmissionFailureEvidenceV2 {
    pub(crate) const fn new(metric_key: EmittedMetricKey) -> Self {
        Self { metric_key }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResultSerializationEvidenceV2 {
    unsigned_result_sha256: Sha256Digest,
}

#[cfg(test)]
impl ResultSerializationEvidenceV2 {
    pub(crate) const fn new(unsigned_result_sha256: Sha256Digest) -> Self {
        Self {
            unsigned_result_sha256,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputWriteEvidenceV2 {
    output_path: CanonicalRelativePath,
}

#[cfg(test)]
impl OutputWriteEvidenceV2 {
    pub(crate) fn checked(output_path: impl Into<String>) -> Result<Self, WireError> {
        let output_path =
            CanonicalRelativePath::checked(output_path.into()).map_err(WireError::validation)?;
        Ok(Self { output_path })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SimErrorKindV2 {
    CliUsage {
        detail: NonEmptyString,
        evidence: CliUsageEvidenceV2,
    },
    ManifestAdmission {
        detail: NonEmptyString,
        evidence: Option<ManifestAdmissionEvidenceV2>,
    },
    TraceAdmission {
        detail: NonEmptyString,
        evidence: Option<TraceAdmissionEvidenceV2>,
    },
    ArchCardAdmission {
        detail: NonEmptyString,
        evidence: Option<ArchCardAdmissionEvidenceV2>,
    },
    ConfigurationResolution {
        detail: NonEmptyString,
        evidence: Option<ConfigurationResolutionEvidenceV2>,
    },
    Replay {
        detail: NonEmptyString,
        evidence: Option<ReplayFailureEvidenceV2>,
    },
    Emission {
        detail: NonEmptyString,
        evidence: Option<EmissionFailureEvidenceV2>,
    },
    ResultSerialization {
        detail: NonEmptyString,
        evidence: Option<ResultSerializationEvidenceV2>,
    },
    OutputWrite {
        detail: NonEmptyString,
        evidence: Option<OutputWriteEvidenceV2>,
    },
}

impl SimErrorKindV2 {
    pub(crate) fn cli_usage(
        detail: impl Into<String>,
        provided_positional_argument_count: u32,
    ) -> Result<Self, WireError> {
        Ok(Self::CliUsage {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence: CliUsageEvidenceV2::new(provided_positional_argument_count),
        })
    }

    pub(crate) fn manifest_admission(
        detail: impl Into<String>,
        evidence: Option<ManifestAdmissionEvidenceV2>,
    ) -> Result<Self, WireError> {
        Ok(Self::ManifestAdmission {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence,
        })
    }

    pub(crate) fn trace_admission(
        detail: impl Into<String>,
        evidence: Option<TraceAdmissionEvidenceV2>,
    ) -> Result<Self, WireError> {
        Ok(Self::TraceAdmission {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence,
        })
    }

    pub(crate) fn arch_card_admission(
        detail: impl Into<String>,
        evidence: Option<ArchCardAdmissionEvidenceV2>,
    ) -> Result<Self, WireError> {
        Ok(Self::ArchCardAdmission {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence,
        })
    }

    pub(crate) fn configuration_resolution(
        detail: impl Into<String>,
        evidence: Option<ConfigurationResolutionEvidenceV2>,
    ) -> Result<Self, WireError> {
        Ok(Self::ConfigurationResolution {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence,
        })
    }

    pub(crate) fn replay(
        detail: impl Into<String>,
        evidence: Option<ReplayFailureEvidenceV2>,
    ) -> Result<Self, WireError> {
        Ok(Self::Replay {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence,
        })
    }

    pub(crate) fn emission(
        detail: impl Into<String>,
        evidence: Option<EmissionFailureEvidenceV2>,
    ) -> Result<Self, WireError> {
        Ok(Self::Emission {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence,
        })
    }

    pub(crate) fn result_serialization(
        detail: impl Into<String>,
        evidence: Option<ResultSerializationEvidenceV2>,
    ) -> Result<Self, WireError> {
        Ok(Self::ResultSerialization {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence,
        })
    }

    pub(crate) fn output_write(
        detail: impl Into<String>,
        evidence: Option<OutputWriteEvidenceV2>,
    ) -> Result<Self, WireError> {
        Ok(Self::OutputWrite {
            detail: NonEmptyString::checked(detail, "error.detail")?,
            evidence,
        })
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct RequiredErrorArm<'a, E> {
    failure_type: &'static str,
    stage: &'static str,
    detail: &'a NonEmptyString,
    evidence: &'a E,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct OptionalErrorArm<'a, E> {
    failure_type: &'static str,
    stage: &'static str,
    detail: &'a NonEmptyString,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence: Option<&'a E>,
}

impl Serialize for SimErrorKindV2 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::CliUsage { detail, evidence } => RequiredErrorArm {
                failure_type: "CLI_USAGE_FAILED",
                stage: "cli_usage",
                detail,
                evidence,
            }
            .serialize(serializer),
            Self::ManifestAdmission { detail, evidence } => OptionalErrorArm {
                failure_type: "MANIFEST_ADMISSION_FAILED",
                stage: "manifest_admission",
                detail,
                evidence: evidence.as_ref(),
            }
            .serialize(serializer),
            Self::TraceAdmission { detail, evidence } => OptionalErrorArm {
                failure_type: "TRACE_ADMISSION_FAILED",
                stage: "trace_admission",
                detail,
                evidence: evidence.as_ref(),
            }
            .serialize(serializer),
            Self::ArchCardAdmission { detail, evidence } => OptionalErrorArm {
                failure_type: "ARCH_CARD_ADMISSION_FAILED",
                stage: "arch_card_admission",
                detail,
                evidence: evidence.as_ref(),
            }
            .serialize(serializer),
            Self::ConfigurationResolution { detail, evidence } => OptionalErrorArm {
                failure_type: "CONFIGURATION_RESOLUTION_FAILED",
                stage: "configuration_resolution",
                detail,
                evidence: evidence.as_ref(),
            }
            .serialize(serializer),
            Self::Replay { detail, evidence } => OptionalErrorArm {
                failure_type: "REPLAY_FAILED",
                stage: "replay",
                detail,
                evidence: evidence.as_ref(),
            }
            .serialize(serializer),
            Self::Emission { detail, evidence } => OptionalErrorArm {
                failure_type: "EMISSION_FAILED",
                stage: "emission",
                detail,
                evidence: evidence.as_ref(),
            }
            .serialize(serializer),
            Self::ResultSerialization { detail, evidence } => OptionalErrorArm {
                failure_type: "RESULT_SERIALIZATION_FAILED",
                stage: "result_serialization",
                detail,
                evidence: evidence.as_ref(),
            }
            .serialize(serializer),
            Self::OutputWrite { detail, evidence } => OptionalErrorArm {
                failure_type: "OUTPUT_WRITE_FAILED",
                stage: "output_write",
                detail,
                evidence: evidence.as_ref(),
            }
            .serialize(serializer),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FailureExitStatusV2 {
    status: &'static str,
    exit_code: u8,
}

impl FailureExitStatusV2 {
    pub(crate) const fn new(exit_code: NonZeroU8) -> Self {
        Self {
            status: "failure",
            exit_code: exit_code.get(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SimErrorV2 {
    schema_version: &'static str,
    error_sha256: Sha256Digest,
    #[serde(skip_serializing_if = "Option::is_none")]
    build_identity: Option<BuildIdentityV2>,
    error: SimErrorKindV2,
    exit_status: FailureExitStatusV2,
}

impl SimErrorV2 {
    pub(crate) fn new(
        error_sha256: Sha256Digest,
        build_identity: Option<BuildIdentityV2>,
        error: SimErrorKindV2,
        exit_code: NonZeroU8,
    ) -> Self {
        Self {
            schema_version: ERROR_SCHEMA_VERSION,
            error_sha256,
            build_identity,
            error,
            exit_status: FailureExitStatusV2::new(exit_code),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde::Serialize;

    use super::*;

    const ZERO_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    const ONE_SHA: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    fn digest(raw: &str) -> Sha256Digest {
        Sha256Digest::parse(raw).expect("test digest must be canonical")
    }

    fn launch_json(
        path: &str,
        workload: &str,
        execution: &str,
        ordinal: &str,
        kernel: &str,
        device: &str,
        stream: &str,
    ) -> String {
        format!(
            r#"{{
                "launch_evidence_manifest_path":"{path}",
                "expected_identity_sha256":"{ZERO_SHA}",
                "launch_key":{{
                    "workload_id":"{workload}",
                    "matched_execution_id":"{execution}",
                    "process_launch_ordinal":{ordinal},
                    "kernel_mangled_name":"{kernel}",
                    "grid":{{"x":1,"y":2,"z":3}},
                    "block":{{"x":32,"y":1,"z":1}},
                    "device_id":{device},
                    "stream_id":{stream},
                    "binary_architecture":"sm_86"
                }}
            }}"#
        )
    }

    fn manifest_json(launches: &[String], replay_mode: &str) -> String {
        format!(
            r#"{{
                "schema_version":2,
                "scope":"all_launches",
                "architecture":{{"arch_id":"sm_86","sku":"rtx_3090_ti"}},
                "configuration_evidence":{{
                    "arch_card_sha256":"{ZERO_SHA}",
                    "pchase_card_sha256":"{ONE_SHA}"
                }},
                "launches":[{}],
                "replay_mode":{replay_mode}
            }}"#,
            launches.join(",")
        )
    }

    fn one_launch_manifest() -> String {
        manifest_json(
            &[launch_json(
                "evidence/launch.json",
                "workload",
                "execution",
                "0",
                "_Z6kernelv",
                "0",
                "7",
            )],
            r#"{"mode":"flat"}"#,
        )
    }

    fn parse_key(workload: &str, execution: &str, ordinal: &str, stream: &str) -> LaunchKeyV1 {
        let launch = launch_json(
            "unused.json",
            workload,
            execution,
            ordinal,
            "_Z6kernelv",
            "0",
            stream,
        );
        let value: serde_json::Value = serde_json::from_str(&launch).expect("test launch JSON");
        let key = serde_json::to_vec(&value["launch_key"]).expect("serialize test launch key");
        parse_launch_key_v1(&key).expect("test launch key must parse")
    }

    #[test]
    fn manifest_consumer_accepts_exact_closed_v2() {
        let manifest = parse_sim_run_manifest_v2(one_launch_manifest().as_bytes())
            .expect("exact manifest must parse");
        assert_eq!(manifest.scope(), "all_launches");
        assert_eq!(manifest.launches().len(), 1);
        assert_eq!(manifest.launches()[0].launch_key().grid(), (1, 2, 3));
        assert_eq!(manifest.launches()[0].launch_key().block(), (32, 1, 1));
        assert_eq!(manifest.launches()[0].launch_key().device_id(), 0);
        assert_eq!(manifest.launches()[0].launch_key().stream_id(), 7);
        assert_eq!(manifest.replay_mode(), ReplayModeV2::Flat);
    }

    #[test]
    fn manifest_consumer_accepts_all_closed_replay_modes() {
        let launch = launch_json(
            "evidence/launch.json",
            "workload",
            "execution",
            "0",
            "_Z6kernelv",
            "0",
            "7",
        );
        let cases = [
            (r#"{"mode":"flat"}"#, ReplayModeV2::Flat),
            (r#"{"mode":"l1_const"}"#, ReplayModeV2::L1Const),
            (
                r#"{"mode":"l1_shared_l2","dram_bandwidth":"carded"}"#,
                ReplayModeV2::L1SharedL2 {
                    dram_bandwidth: DramBandwidthV2::Carded,
                },
            ),
            (
                r#"{"mode":"l1_shared_l2","dram_bandwidth":"unmodeled"}"#,
                ReplayModeV2::L1SharedL2 {
                    dram_bandwidth: DramBandwidthV2::Unmodeled,
                },
            ),
        ];
        for (wire, expected) in cases {
            let manifest = manifest_json(std::slice::from_ref(&launch), wire);
            assert_eq!(
                parse_sim_run_manifest_v2(manifest.as_bytes())
                    .expect("closed replay mode must parse")
                    .replay_mode(),
                expected
            );
        }
    }

    #[test]
    fn launch_key_accepts_full_unsigned_bounds_and_integral_json_numbers() {
        let launch = launch_json(
            "evidence/launch.json",
            "workload",
            "execution",
            "18446744073709551615.0",
            "_Z6kernelv",
            "4294967295.0",
            "18446744073709551615",
        );
        let manifest = manifest_json(&[launch], r#"{"mode":"flat"}"#);
        let key = parse_sim_run_manifest_v2(manifest.as_bytes())
            .expect("exact unsigned maxima must parse")
            .launches()[0]
            .launch_key()
            .clone();
        assert_eq!(key.process_launch_ordinal(), u64::MAX);
        assert_eq!(key.device_id(), u32::MAX);
        assert_eq!(key.stream_id(), u64::MAX);

        let extreme_zero = one_launch_manifest().replacen(
            "\"process_launch_ordinal\":0",
            "\"process_launch_ordinal\":0e9223372036854775808",
            1,
        );
        assert_eq!(
            parse_sim_run_manifest_v2(extreme_zero.as_bytes())
                .expect("every finite-exponent spelling of zero is integral")
                .launches()[0]
                .launch_key()
                .process_launch_ordinal(),
            0
        );
        let padded_exponent = one_launch_manifest().replacen(
            "\"process_launch_ordinal\":0",
            "\"process_launch_ordinal\":1e000000000000000000000000000000000000000000",
            1,
        );
        assert_eq!(
            parse_sim_run_manifest_v2(padded_exponent.as_bytes())
                .expect("zero-padded integral exponent must parse")
                .launches()[0]
                .launch_key()
                .process_launch_ordinal(),
            1
        );

        let maximum_axis = one_launch_manifest().replacen("\"x\":1", "\"x\":4294967295.0", 1);
        assert_eq!(
            parse_sim_run_manifest_v2(maximum_axis.as_bytes())
                .expect("positive u32 axis maximum must parse")
                .launches()[0]
                .launch_key()
                .grid()
                .0,
            u32::MAX
        );

        for invalid in [
            one_launch_manifest().replacen(
                "\"process_launch_ordinal\":0",
                "\"process_launch_ordinal\":18446744073709551616",
                1,
            ),
            one_launch_manifest().replacen("\"device_id\":0", "\"device_id\":4294967296", 1),
            one_launch_manifest().replacen(
                "\"stream_id\":7",
                "\"stream_id\":18446744073709551616",
                1,
            ),
            one_launch_manifest().replacen(
                "\"process_launch_ordinal\":0",
                "\"process_launch_ordinal\":0.5",
                1,
            ),
            one_launch_manifest().replacen("\"x\":1", "\"x\":4294967296", 1),
            one_launch_manifest().replacen("\"x\":1", "\"x\":1.5", 1),
        ] {
            assert!(parse_sim_run_manifest_v2(invalid.as_bytes()).is_err());
        }
    }

    #[test]
    fn v2_consumer_enforces_canonical_relative_paths() {
        for invalid_path in [
            "",
            "/absolute.json",
            "trailing/",
            "double//separator.json",
            "dot/./segment.json",
            "parent/../segment.json",
            "back\\slash.json",
            "space in-name.json",
            "nonascii-λ.json",
        ] {
            let escaped_path = invalid_path.replace('\\', "\\\\");
            let manifest = manifest_json(
                &[launch_json(
                    &escaped_path,
                    "workload",
                    "execution",
                    "0",
                    "kernel",
                    "0",
                    "0",
                )],
                r#"{"mode":"flat"}"#,
            );
            assert!(
                parse_sim_run_manifest_v2(manifest.as_bytes()).is_err(),
                "noncanonical path was accepted: {invalid_path:?}"
            );
        }
    }

    #[test]
    fn launch_key_is_the_complete_closed_nine_field_identity() {
        let key = parse_key("w", "e", "0", "0");
        let value = serde_json::to_value(key).expect("launch key serializes");
        assert_eq!(
            value
                .as_object()
                .expect("launch key object")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "workload_id",
                "matched_execution_id",
                "process_launch_ordinal",
                "kernel_mangled_name",
                "grid",
                "block",
                "device_id",
                "stream_id",
                "binary_architecture",
            ])
        );
        assert_eq!(
            value["grid"]
                .as_object()
                .expect("dim3 object")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["x", "y", "z"])
        );
        let replay_binding = serde_json::to_value(parse_key("w", "e", "0", "0").replay_binding())
            .expect("replay binding serializes");
        assert_eq!(
            replay_binding
                .as_object()
                .expect("replay binding object")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "kernel_mangled_name",
                "grid",
                "block",
                "device_id",
                "stream_id",
                "binary_architecture",
            ])
        );

        let missing = serde_json::to_vec(&serde_json::json!({
            "workload_id":"w",
            "matched_execution_id":"e",
            "process_launch_ordinal":0,
            "kernel_mangled_name":"k",
            "grid":{"x":1,"y":1,"z":1},
            "block":{"x":1,"y":1,"z":1},
            "device_id":0,
            "stream_id":0
        }))
        .expect("missing-field fixture serializes");
        assert!(parse_launch_key_v1(&missing).is_err());
    }

    #[test]
    fn manifest_consumer_rejects_invalid_scalars_and_open_variants() {
        let valid = one_launch_manifest();
        let invalid = [
            valid.replacen("\"schema_version\":2", "\"schema_version\":1", 1),
            valid.replacen(
                "\"scope\":\"all_launches\"",
                "\"scope\":\"all launches\"",
                1,
            ),
            valid.replacen("\"scope\":\"all_launches\"", "\"scope\":\"\"", 1),
            valid.replacen("\"arch_id\":\"sm_86\"", "\"arch_id\":\"sm_120\"", 1),
            valid.replacen("\"sku\":\"rtx_3090_ti\"", "\"sku\":\"rtx_3090\"", 1),
            valid.replacen(ZERO_SHA, "ABC", 1),
            valid.replacen("evidence/launch.json", "evidence/../launch.json", 1),
            valid.replacen("\"workload_id\":\"workload\"", "\"workload_id\":\"\"", 1),
            valid.replacen(
                "\"matched_execution_id\":\"execution\"",
                "\"matched_execution_id\":\"\"",
                1,
            ),
            valid.replacen(
                "\"kernel_mangled_name\":\"_Z6kernelv\"",
                "\"kernel_mangled_name\":\"\"",
                1,
            ),
            valid.replacen("\"x\":1", "\"x\":0", 1),
            valid.replacen(
                "\"block\":{\"x\":32,\"y\":1",
                "\"block\":{\"x\":32,\"y\":0",
                1,
            ),
            valid.replacen(
                "\"process_launch_ordinal\":0",
                "\"process_launch_ordinal\":-1",
                1,
            ),
            valid.replacen("\"device_id\":0", "\"device_id\":-1", 1),
            valid.replacen("\"stream_id\":7", "\"stream_id\":-1", 1),
            valid.replacen(
                "\"binary_architecture\":\"sm_86\"",
                "\"binary_architecture\":\"sm_120\"",
                1,
            ),
            valid.replacen("{\"mode\":\"flat\"}", "{\"mode\":\"typo\"}", 1),
            valid.replacen(
                "{\"mode\":\"flat\"}",
                "{\"mode\":\"flat\",\"extra\":true}",
                1,
            ),
            valid.replacen("{\"mode\":\"flat\"}", "{\"mode\":\"l1_shared_l2\"}", 1),
            valid.replacen(
                "{\"mode\":\"flat\"}",
                "{\"mode\":\"l1_shared_l2\",\"dram_bandwidth\":\"typo\"}",
                1,
            ),
            valid.replacen(
                "\"scope\":\"all_launches\"",
                "\"scope\":\"all_launches\",\"target\":1",
                1,
            ),
        ];
        for document in invalid {
            assert!(
                parse_sim_run_manifest_v2(document.as_bytes()).is_err(),
                "invalid manifest was accepted: {document}"
            );
        }
    }

    #[test]
    fn strict_consumer_rejects_duplicate_members_at_every_manifest_nesting() {
        let valid = one_launch_manifest();
        let duplicates = [
            valid.replacen("\"schema_version\":2", "\"schema_version\":2,\"schema_version\":2", 1),
            valid.replacen("\"arch_id\":\"sm_86\"", "\"arch_id\":\"sm_86\",\"arch_id\":\"sm_86\"", 1),
            valid.replacen("\"arch_card_sha256\"", "\"arch_card_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\",\"arch_card_sha256\"", 1),
            valid.replacen("\"expected_identity_sha256\"", "\"expected_identity_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\",\"expected_identity_sha256\"", 1),
            valid.replacen("\"workload_id\":\"workload\"", "\"workload_id\":\"workload\",\"workload_id\":\"workload\"", 1),
            valid.replacen("\"grid\":{\"x\":1", "\"grid\":{\"x\":1,\"x\":1", 1),
            valid.replacen("{\"mode\":\"flat\"}", "{\"mode\":\"flat\",\"mode\":\"flat\"}", 1),
        ];
        for document in duplicates {
            let error = parse_sim_run_manifest_v2(document.as_bytes())
                .expect_err("duplicate member must fail on the consumer path");
            assert!(
                error.to_string().contains("duplicate JSON member"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn strict_launch_key_consumer_rejects_top_and_dim_duplicates() {
        let key = r#"{
            "workload_id":"w","matched_execution_id":"e","process_launch_ordinal":0,
            "kernel_mangled_name":"k","grid":{"x":1,"x":1,"y":1,"z":1},
            "block":{"x":1,"y":1,"z":1},"device_id":0,"stream_id":0,
            "binary_architecture":"sm_86"
        }"#;
        assert!(
            parse_launch_key_v1(key.as_bytes())
                .expect_err("nested duplicate must fail")
                .to_string()
                .contains("duplicate JSON member")
        );
        let top = key.replace(
            "\"workload_id\":\"w\"",
            "\"workload_id\":\"w\",\"workload_id\":\"w\"",
        );
        assert!(
            parse_launch_key_v1(top.as_bytes())
                .expect_err("top duplicate must fail")
                .to_string()
                .contains("duplicate JSON member")
        );
    }

    #[test]
    fn manifest_semantics_reject_all_three_duplicate_identity_classes() {
        let duplicate_path = manifest_json(
            &[
                launch_json("same.json", "w0", "e0", "0", "k0", "0", "0"),
                launch_json("same.json", "w1", "e1", "1", "k1", "1", "1"),
            ],
            r#"{"mode":"flat"}"#,
        );
        assert!(
            parse_sim_run_manifest_v2(duplicate_path.as_bytes())
                .expect_err("duplicate path")
                .to_string()
                .contains("duplicate launch evidence path")
        );

        let launch = launch_json("a.json", "w", "e", "0", "k", "0", "0");
        let duplicate_key = manifest_json(
            &[launch.clone(), launch.replacen("a.json", "b.json", 1)],
            r#"{"mode":"flat"}"#,
        );
        assert!(
            parse_sim_run_manifest_v2(duplicate_key.as_bytes())
                .expect_err("duplicate key")
                .to_string()
                .contains("duplicate complete launch key")
        );

        let duplicate_binding = manifest_json(
            &[
                launch_json("a.json", "w0", "e0", "0", "k", "0", "0"),
                launch_json("b.json", "w1", "e1", "1.0", "k", "0.0", "0.0"),
            ],
            r#"{"mode":"flat"}"#,
        );
        assert!(
            parse_sim_run_manifest_v2(duplicate_binding.as_bytes())
                .expect_err("duplicate replay binding")
                .to_string()
                .contains("duplicate six-field replay binding")
        );
    }

    #[test]
    fn launch_cardinality_is_nonvacuous_and_bounded() {
        let empty = manifest_json(&[], r#"{"mode":"flat"}"#);
        assert!(
            parse_sim_run_manifest_v2(empty.as_bytes())
                .expect_err("zero launches must fail")
                .to_string()
                .contains("1..=4096")
        );

        let launches: Vec<_> = (0..=MAX_LAUNCHES_V2)
            .map(|index| {
                launch_json(
                    &format!("launch/{index}.json"),
                    &format!("w{index}"),
                    &format!("e{index}"),
                    &index.to_string(),
                    "k",
                    "0",
                    &index.to_string(),
                )
            })
            .collect();
        let maximum = manifest_json(&launches[..MAX_LAUNCHES_V2], r#"{"mode":"flat"}"#);
        assert_eq!(
            parse_sim_run_manifest_v2(maximum.as_bytes())
                .expect("4096 launches must be accepted")
                .launches()
                .len(),
            MAX_LAUNCHES_V2
        );
        let over = manifest_json(&launches, r#"{"mode":"flat"}"#);
        assert!(
            parse_sim_run_manifest_v2(over.as_bytes())
                .expect_err("4097 launches must fail")
                .to_string()
                .contains("1..=4096")
        );
    }

    fn valid_metric_entries() -> [(EmittedMetricKey, EmitOutcome); EMITTED_METRIC_COUNT] {
        let exact = EmitOutcome::Predicted(MetricValue::Exact(1));
        let fraction = EmitOutcome::Predicted(MetricValue::Fraction {
            value: None,
            numerator: 0,
            denominator: 0,
        });
        [
            (EmittedMetricKey::SmspInstExecutedSum, exact),
            (EmittedMetricKey::TrafficGmemLdSectors, exact),
            (EmittedMetricKey::TrafficGmemStSectors, exact),
            (EmittedMetricKey::TrafficDramReadSectors, exact),
            (EmittedMetricKey::CacheHitRateL1TexReadSector, fraction),
            (EmittedMetricKey::CacheHitRateL1TexSector, fraction),
            (EmittedMetricKey::CacheHitRateL2Sector, fraction),
            (
                EmittedMetricKey::ExecCyclesGpcMax,
                EmitOutcome::RequiredUnimplemented {
                    blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
                },
            ),
        ]
    }

    #[test]
    fn metric_map_is_exact_ordered_seven_plus_one() {
        let map = MetricMapV2::checked_entries(valid_metric_entries())
            .expect("exact metric entries must pass");
        let value = serde_json::to_value(map).expect("metric map must serialize");
        let keys: BTreeSet<_> = value
            .as_object()
            .expect("metric map object")
            .keys()
            .map(String::as_str)
            .collect();
        let expected: BTreeSet<_> = EmittedMetricKey::V2_ORDER
            .iter()
            .map(|key| key.as_str())
            .collect();
        assert_eq!(keys, expected);
        assert_eq!(
            value["exec_cycles_gpc_max"]["status"],
            "required_unimplemented"
        );
    }

    #[test]
    fn metric_map_accepts_both_producer_orders_and_rejects_invalid_sets() {
        assert!(MetricMapV2::from_predictions(&[]).is_err());

        let mut missing = valid_metric_entries().to_vec();
        missing.pop();
        assert!(MetricMapV2::checked_entries(missing).is_err());

        let v2_order = valid_metric_entries();
        let descriptor_order = [
            v2_order[0],
            v2_order[1],
            v2_order[2],
            v2_order[4],
            v2_order[5],
            v2_order[6],
            v2_order[3],
            v2_order[7],
        ];
        MetricMapV2::checked_entries(descriptor_order)
            .expect("aggregate descriptor order must build the exact map");

        let mut duplicate = valid_metric_entries();
        duplicate[6] = duplicate[5];
        assert!(MetricMapV2::checked_entries(duplicate).is_err());

        let mut promoted_cycles = valid_metric_entries();
        promoted_cycles[7].1 = EmitOutcome::Predicted(MetricValue::Exact(7));
        assert!(MetricMapV2::checked_entries(promoted_cycles).is_err());

        let mut wrong_unit = valid_metric_entries();
        wrong_unit[6].1 = EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::Dram,
        });
        assert!(MetricMapV2::checked_entries(wrong_unit).is_err());
    }

    #[test]
    fn metric_map_rejects_fraction_payload_contradictions() {
        let mut contradictory = valid_metric_entries();
        contradictory[4].1 = EmitOutcome::Predicted(MetricValue::Fraction {
            value: Some(0.25),
            numerator: 1,
            denominator: 2,
        });
        assert!(MetricMapV2::checked_entries(contradictory).is_err());

        let mut nonnull_zero_denominator = valid_metric_entries();
        nonnull_zero_denominator[4].1 = EmitOutcome::Predicted(MetricValue::Fraction {
            value: Some(0.0),
            numerator: 0,
            denominator: 0,
        });
        assert!(MetricMapV2::checked_entries(nonnull_zero_denominator).is_err());

        let mut exact = valid_metric_entries();
        exact[4].1 = EmitOutcome::Predicted(MetricValue::Fraction {
            value: Some(exact_fraction_to_f64(1, u64::MAX).expect("exact ratio")),
            numerator: 1,
            denominator: u64::MAX,
        });
        MetricMapV2::checked_entries(exact).expect("exact rounded fraction must pass");
    }

    #[test]
    fn result_substructures_are_closed_and_launch_sets_match() {
        let metrics = MetricMapV2::checked_entries(valid_metric_entries()).expect("metric map");
        let key = parse_key("w", "e", "0", "0");
        let launch = PerLaunchResultV2::new(key.clone(), metrics.clone());
        let aggregate = AggregateV2::checked(1, metrics).expect("aggregate");
        let diagnostics = DiagnosticsV2::checked(
            vec![PerLaunchDiagnosticV2::new(key.clone(), 7, Some(11))],
            None,
        )
        .expect("diagnostics");
        validate_result_launch_sets(&[launch.clone()], &aggregate, &diagnostics)
            .expect("matching launch sets");

        let launch_value = serde_json::to_value(launch).expect("launch serializes");
        assert_eq!(
            launch_value
                .as_object()
                .expect("launch object")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["launch_key", "metrics"])
        );
        let diagnostics_value = serde_json::to_value(diagnostics).expect("diagnostics serialize");
        assert!(diagnostics_value.get("internal_phase_timings_ns").is_none());
        assert_eq!(diagnostics_value["admissibility"], "non_admissible");

        let duplicate = vec![
            PerLaunchResultV2::new(key.clone(), metrics_clone()),
            PerLaunchResultV2::new(key, metrics_clone()),
        ];
        let aggregate_two = AggregateV2::checked(2, metrics_clone()).expect("aggregate");
        let diagnostics_two = DiagnosticsV2::checked(
            vec![PerLaunchDiagnosticV2::new(
                parse_key("w", "e", "0", "0"),
                0,
                None,
            )],
            None,
        )
        .expect("diagnostics");
        assert!(validate_result_launch_sets(&duplicate, &aggregate_two, &diagnostics_two).is_err());

        let count_key = parse_key("count-w", "count-e", "0", "2");
        let count_launch = PerLaunchResultV2::new(count_key.clone(), metrics_clone());
        let wrong_count = AggregateV2::checked(2, metrics_clone()).expect("aggregate");
        let count_diagnostics =
            DiagnosticsV2::checked(vec![PerLaunchDiagnosticV2::new(count_key, 0, None)], None)
                .expect("diagnostics");
        assert!(
            validate_result_launch_sets(&[count_launch], &wrong_count, &count_diagnostics).is_err()
        );

        let result_key = parse_key("result-w", "result-e", "0", "3");
        let diagnostic_key = parse_key("diagnostic-w", "diagnostic-e", "0", "4");
        let result_launch = PerLaunchResultV2::new(result_key, metrics_clone());
        let one_aggregate = AggregateV2::checked(1, metrics_clone()).expect("aggregate");
        let mismatched_diagnostics = DiagnosticsV2::checked(
            vec![PerLaunchDiagnosticV2::new(diagnostic_key, 0, None)],
            None,
        )
        .expect("diagnostics");
        assert!(
            validate_result_launch_sets(&[result_launch], &one_aggregate, &mismatched_diagnostics,)
                .is_err()
        );

        let binding_key_a = parse_key("binding-w0", "binding-e0", "0", "5");
        let binding_key_b = parse_key("binding-w1", "binding-e1", "1", "5");
        let binding_launches = vec![
            PerLaunchResultV2::new(binding_key_a.clone(), metrics_clone()),
            PerLaunchResultV2::new(binding_key_b.clone(), metrics_clone()),
        ];
        let binding_aggregate = AggregateV2::checked(2, metrics_clone()).expect("aggregate");
        let binding_diagnostics = DiagnosticsV2::checked(
            vec![
                PerLaunchDiagnosticV2::new(binding_key_a, 0, None),
                PerLaunchDiagnosticV2::new(binding_key_b, 0, None),
            ],
            None,
        )
        .expect("diagnostics");
        assert!(
            validate_result_launch_sets(
                &binding_launches,
                &binding_aggregate,
                &binding_diagnostics,
            )
            .expect_err("duplicate replay binding must fail")
            .to_string()
            .contains("six-field replay binding")
        );

        let duplicate_diagnostic_key = parse_key("diag-w", "diag-e", "0", "6");
        assert!(
            DiagnosticsV2::checked(
                vec![
                    PerLaunchDiagnosticV2::new(duplicate_diagnostic_key.clone(), 0, None),
                    PerLaunchDiagnosticV2::new(duplicate_diagnostic_key, 1, None),
                ],
                None,
            )
            .is_err()
        );
    }

    fn metrics_clone() -> MetricMapV2 {
        MetricMapV2::checked_entries(valid_metric_entries()).expect("metric map")
    }

    #[test]
    fn identities_fix_architecture_and_claimability() {
        let clean = BuildIdentityV2::checked(
            "0123456789abcdef0123456789abcdef01234567",
            false,
            digest(ZERO_SHA),
            "rustc 1.90.0",
            "release",
        )
        .expect("clean identity");
        let dirty = BuildIdentityV2::checked(
            "0123456789abcdef0123456789abcdef01234567",
            true,
            digest(ZERO_SHA),
            "rustc 1.90.0",
            "debug",
        )
        .expect("dirty identity");
        assert_eq!(
            serde_json::to_value(clean).expect("clean serializes")["claimability"],
            serde_json::json!({"status":"claimable","basis":"clean_build"})
        );
        assert_eq!(
            serde_json::to_value(dirty).expect("dirty serializes")["claimability"],
            serde_json::json!({"status":"unclaimable","reason":"dirty_build"})
        );
        assert!(
            BuildIdentityV2::checked("ABC", false, digest(ZERO_SHA), "rustc", "release").is_err()
        );

        let card = ArchCardIdentityV2::checked("card", digest(ZERO_SHA)).expect("card");
        let card = serde_json::to_value(card).expect("card serializes");
        assert_eq!(card["sm_arch"], "sm_86");
        assert_eq!(card["sku"], "rtx_3090_ti");
    }

    #[test]
    fn all_nine_error_arms_have_fixed_failure_stage_pairs() {
        let key = parse_key("w", "e", "0", "0");
        let cases = vec![
            SimErrorKindV2::cli_usage("bad cli", 0).expect("CLI error"),
            SimErrorKindV2::manifest_admission(
                "bad manifest",
                Some(
                    ManifestAdmissionEvidenceV2::checked("run/manifest.json", digest(ZERO_SHA))
                        .expect("manifest evidence"),
                ),
            )
            .expect("manifest error"),
            SimErrorKindV2::trace_admission("bad trace", None).expect("trace error"),
            SimErrorKindV2::arch_card_admission(
                "bad card",
                Some(
                    ArchCardAdmissionEvidenceV2::checked("card", digest(ZERO_SHA))
                        .expect("card evidence"),
                ),
            )
            .expect("card error"),
            SimErrorKindV2::configuration_resolution(
                "bad config",
                Some(
                    ConfigurationResolutionEvidenceV2::checked("config", digest(ZERO_SHA))
                        .expect("configuration evidence"),
                ),
            )
            .expect("configuration error"),
            SimErrorKindV2::replay("bad replay", Some(ReplayFailureEvidenceV2::new(key)))
                .expect("replay error"),
            SimErrorKindV2::emission(
                "bad emission",
                Some(EmissionFailureEvidenceV2::new(
                    EmittedMetricKey::TrafficDramReadSectors,
                )),
            )
            .expect("emission error"),
            SimErrorKindV2::result_serialization(
                "bad serialization",
                Some(ResultSerializationEvidenceV2::new(digest(ZERO_SHA))),
            )
            .expect("serialization error"),
            SimErrorKindV2::output_write(
                "bad write",
                Some(
                    OutputWriteEvidenceV2::checked("output/result.json").expect("output evidence"),
                ),
            )
            .expect("output error"),
        ];
        let expected = [
            ("CLI_USAGE_FAILED", "cli_usage"),
            ("MANIFEST_ADMISSION_FAILED", "manifest_admission"),
            ("TRACE_ADMISSION_FAILED", "trace_admission"),
            ("ARCH_CARD_ADMISSION_FAILED", "arch_card_admission"),
            (
                "CONFIGURATION_RESOLUTION_FAILED",
                "configuration_resolution",
            ),
            ("REPLAY_FAILED", "replay"),
            ("EMISSION_FAILED", "emission"),
            ("RESULT_SERIALIZATION_FAILED", "result_serialization"),
            ("OUTPUT_WRITE_FAILED", "output_write"),
        ];
        for (error, (failure_type, stage)) in cases.into_iter().zip(expected) {
            let value = serde_json::to_value(error).expect("error arm serializes");
            assert_eq!(value["failure_type"], failure_type);
            assert_eq!(value["stage"], stage);
            assert!(
                value["detail"]
                    .as_str()
                    .is_some_and(|detail| !detail.is_empty())
            );
        }
    }

    #[test]
    fn error_evidence_is_required_for_cli_and_grouped_elsewhere() {
        let cli = serde_json::to_value(
            SimErrorKindV2::cli_usage("usage", 2).expect("CLI arm must build"),
        )
        .expect("CLI serializes");
        assert_eq!(
            cli["evidence"],
            serde_json::json!({
                "provided_positional_argument_count": 2,
                "expected_positional_argument_count": 1
            })
        );

        let absent = serde_json::to_value(
            SimErrorKindV2::manifest_admission("manifest", None)
                .expect("optional evidence may be absent"),
        )
        .expect("manifest arm serializes");
        assert!(absent.get("evidence").is_none());

        let present = serde_json::to_value(
            SimErrorKindV2::trace_admission(
                "trace",
                Some(
                    TraceAdmissionEvidenceV2::checked("bundle", digest(ZERO_SHA))
                        .expect("trace evidence"),
                ),
            )
            .expect("trace arm"),
        )
        .expect("trace arm serializes");
        assert_eq!(
            present["evidence"]
                .as_object()
                .expect("evidence object")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["bundle_id", "bundle_sha256"])
        );
        assert!(SimErrorKindV2::output_write("", None).is_err());
    }

    #[test]
    fn top_level_result_and_error_types_are_serializable_without_values() {
        fn assert_serializable<T: Serialize>() {}
        assert_serializable::<SimResultV2>();
        assert_serializable::<SimErrorV2>();

        let error = SimErrorV2::new(
            digest(ZERO_SHA),
            None,
            SimErrorKindV2::cli_usage("usage", 0).expect("CLI error"),
            NonZeroU8::new(2).expect("nonzero test exit"),
        );
        let value = serde_json::to_value(error).expect("top-level error serializes");
        assert_eq!(value["schema_version"], ERROR_SCHEMA_VERSION);
        assert_eq!(
            value["exit_status"],
            serde_json::json!({"status":"failure","exit_code":2})
        );
        assert!(value.get("build_identity").is_none());
    }
}
