//! Native TMA and synchronization observations, owned by one loaded trace.
//! Optional producer clock fields are carried as evidence only; absent clocks
//! never become inferred completion or visibility timestamps.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// Runtime resource bounds, not corpus-selection or measurement gates.
const MAX_LINE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_INSTRUCTION_KEYS: usize = 4_000_000;

/// Open one generated sidecar without materialising the compressed stream.
///
/// A compressed sibling takes precedence over the legacy plain file.  This is
/// intentional: the archive migration can leave both names during a staged
/// copy, and silently choosing the stale plain stream would make the reader
/// report a different execution.  Decoder errors are returned to the caller,
/// rather than being treated as an absent sidecar.
fn open_sidecar_reader(plain: &Path) -> Result<Option<(PathBuf, Box<dyn BufRead>)>, io::Error> {
    let compressed = plain.with_extension("jsonl.zst");
    let path = if compressed.is_file() {
        compressed
    } else if plain.is_file() {
        plain.to_path_buf()
    } else {
        return Ok(None);
    };
    if path.extension().is_some_and(|extension| extension == "zst") {
        let file = File::open(&path)?;
        let decoder = zstd::stream::read::Decoder::new(file)?;
        Ok(Some((path, Box::new(BufReader::new(decoder)))))
    } else {
        Ok(Some((
            path.clone(),
            Box::new(BufReader::new(File::open(path)?)),
        )))
    }
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("native async sidecar {path}: {message}")]
pub struct AsyncSidecarError {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ObservationKey {
    pub cta: [u32; 3],
    pub warp: i32,
    pub instruction: usize,
    pub pc: u32,
    pub function: i32,
}

#[derive(Debug, Clone)]
pub struct TmaObservation {
    pub bytes: Option<u64>,
    pub is_load: bool,
    pub is_store: bool,
    pub zero_byte: bool,
    pub oob_or_empty: bool,
    /// Active lanes observed at the producer site. This is a dynamic
    /// execution mask, not a claim about a physical transaction granule.
    pub executed_lane_mask: Option<u32>,
    pub destination_barrier: Option<u32>,
    /// Raw destination mbarrier address.  The offset alone cannot distinguish
    /// per-CTA barriers when the tracer exposes cluster rank bits.
    pub destination_barrier_raw: Option<u32>,
    pub destination_cluster_rank: Option<u32>,
    /// Optional device-clock observations emitted by a clock-instrumented
    /// producer. They are intentionally independent of NCU and are never
    /// synthesized from payload bytes or profiler counters.
    pub issue_clock: Option<u64>,
    pub completion_clock: Option<u64>,
    /// Distinct 32-byte source sectors projected from the retained TMA source
    /// addresses. This is an address-derived observation for the conditional
    /// TMA/L2 shadow metric; it is not a claim about the engine's request
    /// granularity or completion timing.
    pub source_sector_ids: Box<[i64]>,
    /// Distinct 32-byte destination sectors projected from retained TMA
    /// destination addresses. Used only by the conditional source-specific
    /// L2 shadow observer for store-direction transfers.
    pub destination_sector_ids: Box<[i64]>,
}

#[derive(Debug, Default)]
pub struct SyncObservation {
    pub records: u64,
    pub before_records: u64,
    pub after_records: u64,
    pub value_incomplete_records: u64,
    pub mbarrier_address_records: u64,
    pub mbarrier_register_records: u64,
    pub arrival_records: u64,
    pub phase_check_records: u64,
    pub wait_records: u64,
    pub exchange_records: u64,
    pub exchange_token_records: u64,
    /// Instruction-form semantics retained from the dynamic opcode.  The
    /// sidecar may aggregate several operand rows for one dynamic
    /// instruction, so this is a set rather than a guessed single physical
    /// operation.
    pub barrier_operations: BTreeSet<BarrierOperation>,
    pub query_kind: Option<PhaseQueryKind>,
    /// More than one incompatible opcode/query interpretation was associated
    /// with the same dynamic key. Such a key is retained as evidence but must
    /// not be treated as a decisive frontier.
    pub interpretation_conflict: bool,
    pub executed_lane_mask: u32,
    pub arrival_lane_mask: u32,
    pub transaction_lane_mask: u32,
    pub query_lane_mask: u32,
    pub exchange_token_lane_mask: u32,
    /// Preserve the raw barrier operand alongside the legacy offset
    /// projection. Address equality and shared-memory offset equality are not
    /// the same lifetime/ownership relation, especially for cluster targets.
    pub barrier_address_raw: Option<u32>,
    pub barrier_cluster_rank: Option<u32>,
    /// `ARRIVES.LDGSTSBAR` belongs to the async-copy/group namespace and is
    /// intentionally not folded into ordinary transaction-barrier arrivals.
    pub is_async_copy_arrival: bool,
    /// Signed observation-side transaction accounting delta.  This is a
    /// semantic ledger contribution, not a physical completion timestamp or
    /// a claim that the encoded operand equals a hardware byte granule.
    pub transaction_delta_bytes: Option<i64>,
    /// A decoded unsigned operand existed, but its signed ledger delta could
    /// not be represented exactly.  This is different from an absent
    /// operand: both are unresolved, but the former must remain visible as a
    /// precision loss rather than silently becoming zero/no-op.
    pub operand_precision_lost: bool,
    /// The operand is retained separately from its interpreted delta so lane
    /// multiplicity and an elected/counting lowering cannot be mistaken for a
    /// population count.
    pub transaction_operand_bytes: Option<u64>,
    pub arrival_count_operand: Option<u64>,
    // Distinct within ONE instruction. Summing the lengths is an occurrence
    // count, not distinct barrier objects or addresses across a whole launch.
    pub barrier_addresses: BTreeSet<u32>,
    pub is_arrival: bool,
    pub is_phase_query: bool,
    pub expected_tx_bytes: Option<u64>,
    pub requested_phase: Option<u64>,
    /// All uniform phase values observed for this dynamic query key. A set of
    /// more than one value is retained as an alternative, not guessed into a
    /// single generation.
    pub requested_phases: BTreeSet<u64>,
    pub successful_wait_lanes: u32,
    pub unsuccessful_wait_lanes: u32,
    /// Result sampled from the predicate row associated with this dynamic
    /// query.  The result is an evaluation witness; it is not a persistent
    /// state sample at the later callback clock.
    pub query_result_evaluated: bool,
    pub query_result_lane_mask: u32,
    pub query_result_mixed: bool,
    /// Optional device-clock observation at an explicitly identified acquire
    /// visibility point. Legacy `visibility_clock` rows are parsed as
    /// `query_clock` instead; the tracer cannot infer acquire semantics from a
    /// lowered PHASECHK mnemonic.
    pub visibility_clock: Option<u64>,
    pub query_clock: Option<u64>,
    pub last_unsuccessful_query_clock: Option<u64>,
    pub first_successful_query_clock: Option<u64>,
}

/// Architectural operation family visible in the captured SASS spelling.
/// These names describe the software/barrier obligation only; they are not
/// hardware queue, engine, or completion-port identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BarrierOperation {
    ArriveExpectTransaction,
    ArriveOnly,
    ExpectTransactionOnly,
    CompleteTransaction,
    CountedArrival,
    ArrivalDrop,
    AsyncCopyBarrierArrival,
    UnknownArrivalForm,
    PhaseQuery,
    BarrierExchange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseQueryKind {
    /// A potentially suspending phase wait.  A false result is not itself a
    /// proof that the issuing warp continued issuing instructions.
    TryWait,
    /// A nonblocking phase test/query.
    TestWait,
    Unknown,
}

/// Classify only documented/observed opcode-form distinctions.  A missing or
/// unrecognised suffix is kept as an unknown form rather than silently
/// assigned a transaction effect.
pub fn barrier_operation(opcode: &str) -> Option<BarrierOperation> {
    let op = opcode.trim();
    if op.starts_with("ARRIVES.LDGSTSBAR") {
        return Some(BarrierOperation::AsyncCopyBarrierArrival);
    }
    if op.starts_with("SYNCS.PHASECHK")
        || op.starts_with("SYNCS.TRYWAIT")
        || op.starts_with("SYNCS.TESTWAIT")
    {
        return Some(BarrierOperation::PhaseQuery);
    }
    if op.starts_with("SYNCS.EXCH") {
        return Some(BarrierOperation::BarrierExchange);
    }
    if !op.starts_with("SYNCS.ARRIVE") {
        return None;
    }
    let fields: BTreeSet<&str> = op.split('.').collect();
    if fields.contains("OPTOUT") {
        return Some(BarrierOperation::ArrivalDrop);
    }
    if fields.contains("A1T0") {
        return Some(BarrierOperation::ArriveOnly);
    }
    if fields.contains("A0TR") {
        return Some(BarrierOperation::ExpectTransactionOnly);
    }
    if fields.contains("A0TX") {
        return Some(BarrierOperation::CompleteTransaction);
    }
    if fields.contains("ART0") {
        return Some(BarrierOperation::CountedArrival);
    }
    if fields.contains("A1TR") {
        return Some(BarrierOperation::ArriveExpectTransaction);
    }
    // Other A*/T* combinations are preserved as unknown rather than
    // accidentally receiving the default transaction effect.
    if fields.iter().any(|field| {
        field.starts_with('A')
            && field.len() == 4
            && field
                .as_bytes()
                .get(1)
                .is_some_and(|byte| byte.is_ascii_digit())
            && field.as_bytes().get(2) == Some(&b'T')
    }) {
        return Some(BarrierOperation::UnknownArrivalForm);
    }
    // The live corpus contains both the explicit default spelling and
    // `.RED` variants.  Both use the default arrive+expected-transaction
    // form; RED only suppresses the returned token.
    if fields.contains("TRANS64") {
        return Some(BarrierOperation::ArriveExpectTransaction);
    }
    Some(BarrierOperation::UnknownArrivalForm)
}

pub fn phase_query_kind(opcode: &str) -> Option<PhaseQueryKind> {
    if !opcode.starts_with("SYNCS.PHASECHK")
        && !opcode.starts_with("SYNCS.TRYWAIT")
        && !opcode.starts_with("SYNCS.TESTWAIT")
    {
        return None;
    }
    if opcode.split('.').any(|field| field == "TRYWAIT") {
        Some(PhaseQueryKind::TryWait)
    } else if opcode.starts_with("SYNCS.PHASECHK") || opcode.starts_with("SYNCS.TESTWAIT") {
        Some(PhaseQueryKind::TestWait)
    } else {
        Some(PhaseQueryKind::Unknown)
    }
}

fn merge_phase_query_kind(
    current: Option<PhaseQueryKind>,
    incoming: PhaseQueryKind,
) -> Option<PhaseQueryKind> {
    match current {
        None => Some(incoming),
        Some(existing) if existing == incoming => Some(existing),
        Some(_) => Some(PhaseQueryKind::Unknown),
    }
}

fn merge_operand_value(current: &mut Option<u64>, incoming: Option<u64>, conflict: &mut bool) {
    let Some(incoming) = incoming else {
        return;
    };
    if current.is_some_and(|prior| prior != incoming) {
        *conflict = true;
    } else {
        *current = Some(incoming);
    }
}

#[derive(Debug, Default)]
pub struct LaunchObservations {
    pub tma: BTreeMap<ObservationKey, TmaObservation>,
    /// First-seen order of unique TMA records in the sidecar stream. The map
    /// remains the identity index; this vector is an optional conditional
    /// shadow-order coordinate and is not asserted to be engine chronology.
    pub tma_order: Vec<ObservationKey>,
    pub sync: BTreeMap<ObservationKey, SyncObservation>,
}

/// How strongly a producer/query relationship is identified by the captured
/// sidecars.  A protocol-scoped candidate is useful for follow-up analysis,
/// but must never be counted as an established satisfaction witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseAssociation {
    EstablishedWitness,
    ProtocolScopedCandidate,
    Unresolved,
}

/// A phase-readiness observation is deliberately weaker than an engine
/// completion timestamp.  It joins a producer/query only when the captured
/// barrier identity and one-to-one generation evidence justify that claim;
/// otherwise it is retained as a protocol-scoped candidate.  Query ordering
/// is never inferred from file order, parity, or an exchange token alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PhaseReadinessObservation {
    pub association: PhaseAssociation,
    pub producer: ObservationKey,
    pub query: ObservationKey,
    pub barrier_cta: [u32; 3],
    pub barrier_offset: u32,
    pub barrier_address_raw: Option<u32>,
    pub destination_cluster_rank: Option<u32>,
    pub payload_bytes: u64,
    pub producer_issue_clock: Option<u64>,
    pub last_unsuccessful_query_clock: Option<u64>,
    pub first_successful_query_clock: Option<u64>,
    pub successful_wait_lanes: u32,
    pub observation_window: Option<[u64; 2]>,
    pub upper_only: Option<u64>,
    /// NVBit query clocks are sampled after the instrumented query. They are
    /// therefore a span between post-query observations, not an execution
    /// bracket around the query itself. Native probes may report a stronger
    /// bracket separately in their compact reduction.
    pub window_basis: &'static str,
}

#[derive(Debug, Default)]
pub struct AsyncSidecars {
    launches: BTreeMap<(i64, i64, i32), LaunchObservations>,
    /// Optional launch-scoped `%clock64` evidence. This is kept beside, but
    /// not mixed into, TMA/sync relation timing or replay cycles.
    pub phase_clock: crate::phase_sidecars::PhaseClockEvidence,
}

#[derive(Deserialize)]
struct TmaSourceAddress {
    address: u64,
    oob: bool,
}

fn compact_sector_ids(
    values: Option<&[TmaSourceAddress]>,
) -> Result<Box<[i64]>, &'static str> {
    let Some(values) = values else {
        return Ok(Box::new([]));
    };
    let mut seen = BTreeSet::new();
    let mut sectors = Vec::new();
    for value in values {
        if value.oob {
            continue;
        }
        let sector = value.address >> 5;
        let sector = i64::try_from(sector).map_err(|_| "TMA source address exceeds the signed sector-id domain")?;
        if seen.insert(sector) {
            sectors.push(sector);
        }
    }
    Ok(sectors.into_boxed_slice())
}

// Do not flatten a serde map here: it would materialize the enormous address
// arrays. Only TMA source addresses are decoded into compact distinct sectors;
// other unused fields remain skipped without constructing Values.
#[derive(Deserialize)]
struct Row {
    device_id: i64,
    kernel_id: i32,
    stream_id: Option<i64>,
    threadblock: Option<String>,
    cta: [u32; 3],
    warp_id: i32,
    warp_instruction_index: usize,
    function_id: i32,
    pc: u32,
    opcode: String,
    byte_count: Option<u64>,
    zero_byte_disposition: Option<String>,
    operand_kind: Option<String>,
    phase: Option<String>,
    value_complete: Option<bool>,
    barrier_address_present: Option<bool>,
    barrier_address_u32: Option<u32>,
    dst_mbar_valid: Option<bool>,
    dst_mbar_address: Option<u32>,
    dst_mbar_address_offset: Option<u32>,
    dst_mbar_cluster_cta_id: Option<u32>,
    issue_clock: Option<u64>,
    completion_clock: Option<u64>,
    visibility_clock: Option<u64>,
    query_clock: Option<u64>,
    acquire_visibility_clock: Option<u64>,
    sync_role: Option<u32>,
    values: Option<Vec<u64>>,
    lane_mask: Option<u32>,
    value_mask_complete: Option<bool>,
    src_addresses: Option<Vec<TmaSourceAddress>>,
    dst_addresses: Option<Vec<TmaSourceAddress>>,
}

impl AsyncSidecars {
    pub fn launch(&self, device: i64, stream: i64, kernel: i32) -> Option<&LaunchObservations> {
        self.launches.get(&(device, stream, kernel))
    }

    pub(crate) fn load(
        root: &Path,
        streams: &BTreeMap<(i64, i32), BTreeSet<i64>>,
    ) -> Result<Self, AsyncSidecarError> {
        let mut result = Self::default();
        let mut keys = 0;
        for (name, tma) in [("tma_trace.jsonl", true), ("sync_trace.jsonl", false)] {
            let plain = root.join("extra_info").join(name);
            let Some((path, mut reader)) =
                open_sidecar_reader(&plain).map_err(|e| AsyncSidecarError {
                    path: plain.clone(),
                    message: e.to_string(),
                })?
            else {
                continue;
            };
            let error = |message: String| AsyncSidecarError {
                path: path.clone(),
                message,
            };
            let mut line = Vec::new();
            let mut number = 0_u64;
            loop {
                line.clear();
                // `take` is recreated for each line to retain the bound;
                // decoder errors propagate instead of looking like EOF.
                let read = reader
                    .by_ref()
                    .take(MAX_LINE_BYTES + 1)
                    .read_until(b'\n', &mut line)
                    .map_err(|e| error(e.to_string()))?;
                if read == 0 {
                    break;
                }
                number += 1;
                if read as u64 > MAX_LINE_BYTES {
                    return Err(error(format!(
                        "line {number} exceeds {MAX_LINE_BYTES} bytes"
                    )));
                }
                let row: Row = serde_json::from_slice(&line)
                    .map_err(|e| error(format!("line {number}: {e}")))?;
                let launch_streams = streams.get(&(row.device_id, row.kernel_id));
                let stream = row
                    .stream_id
                    .or_else(|| {
                        // Producer identity: d_<device>_s_<stream>_k_<kernel>_<CTA>.
                        let tb = row.threadblock.as_deref()?;
                        let prefix = format!("d_{}_s_", row.device_id);
                        let (stream, rest) = tb.strip_prefix(&prefix)?.split_once("_k_")?;
                        rest.strip_prefix(&format!("{}_", row.kernel_id))?;
                        stream.parse().ok()
                    })
                    .or_else(|| launch_streams.filter(|s| s.len() == 1)?.first().copied())
                    .ok_or_else(|| {
                        error(format!(
                            "line {number} has ambiguous/missing stream identity"
                        ))
                    })?;
                if !launch_streams.is_some_and(|s| s.contains(&stream)) {
                    return Err(error(format!(
                        "line {number} refers to an absent traced launch"
                    )));
                }
                let key = ObservationKey {
                    cta: row.cta,
                    warp: row.warp_id,
                    instruction: row.warp_instruction_index,
                    pc: row.pc,
                    function: row.function_id,
                };
                let launch = result
                    .launches
                    .entry((row.device_id, stream, row.kernel_id))
                    .or_default();
                let is_new = if tma {
                    !launch.tma.contains_key(&key)
                } else {
                    !launch.sync.contains_key(&key)
                };
                if is_new {
                    keys += 1;
                    if keys > MAX_INSTRUCTION_KEYS {
                        return Err(error(format!(
                            "instruction index exceeds {MAX_INSTRUCTION_KEYS} keys"
                        )));
                    }
                }
                if tma {
                    let head = row.opcode.split('.').next().unwrap_or("");
                    if !matches!(head, "UTMALDG" | "UTMASTG" | "UBLKCP") {
                        return Err(error(format!(
                            "line {number} unsupported TMA opcode {}",
                            row.opcode
                        )));
                    }
                    let is_load = matches!(head, "UTMALDG" | "UBLKCP");
                    let source_sector_ids = if is_load {
                        compact_sector_ids(row.src_addresses.as_deref())
                            .map_err(|message| error(message.to_owned()))?
                    } else {
                        Box::new([])
                    };
                    let destination_sector_ids = if head == "UTMASTG" {
                        compact_sector_ids(row.dst_addresses.as_deref())
                            .map_err(|message| error(message.to_owned()))?
                    } else {
                        Box::new([])
                    };
                    let obs = TmaObservation {
                        bytes: row.byte_count,
                        // Blackwell lowers the bulk tensor/mbarrier transfer
                        // used by the live probe to UBLKCP.S.G. Treat that
                        // spelling as a producer-side load while retaining
                        // the opcode in the sidecar identity; it is not
                        // silently reclassified as ordinary LDGSTS traffic.
                        is_load,
                        is_store: head == "UTMASTG",
                        zero_byte: row.byte_count == Some(0),
                        oob_or_empty: row.zero_byte_disposition.as_deref()
                            == Some("oob_or_empty_transfer"),
                        executed_lane_mask: row.lane_mask,
                        destination_barrier: (row.dst_mbar_valid == Some(true))
                            .then_some(row.dst_mbar_address_offset.or(row.dst_mbar_address))
                            .flatten(),
                        destination_barrier_raw: (row.dst_mbar_valid == Some(true))
                            .then_some(row.dst_mbar_address)
                            .flatten(),
                        destination_cluster_rank: row.dst_mbar_cluster_cta_id,
                        issue_clock: row.issue_clock,
                        completion_clock: row.completion_clock,
                        source_sector_ids,
                        destination_sector_ids,
                    };
                    if launch.tma.insert(key, obs).is_some() {
                        return Err(error(format!(
                            "line {number} duplicates TMA instruction identity"
                        )));
                    }
                    launch.tma_order.push(key);
                } else {
                    let obs = launch.sync.entry(key).or_default();
                    let operation = barrier_operation(&row.opcode);
                    if let Some(operation) = operation {
                        if !obs.barrier_operations.is_empty()
                            && !obs.barrier_operations.contains(&operation)
                        {
                            obs.interpretation_conflict = true;
                        }
                        obs.barrier_operations.insert(operation);
                    }
                    if let Some(kind) = phase_query_kind(&row.opcode) {
                        if obs.query_kind.is_some_and(|prior| prior != kind) {
                            obs.interpretation_conflict = true;
                        }
                        obs.query_kind = merge_phase_query_kind(obs.query_kind, kind);
                    }
                    if let Some(mask) = row.lane_mask {
                        obs.executed_lane_mask |= mask;
                        if operation == Some(BarrierOperation::PhaseQuery) {
                            obs.query_lane_mask |= mask;
                        }
                        if matches!(
                            operation,
                            Some(
                                BarrierOperation::ArriveExpectTransaction
                                    | BarrierOperation::ArriveOnly
                                    | BarrierOperation::CountedArrival
                                    | BarrierOperation::ArrivalDrop
                                    | BarrierOperation::AsyncCopyBarrierArrival
                            )
                        ) {
                            obs.arrival_lane_mask |= mask;
                        }
                        if row.sync_role == Some(2)
                            && matches!(
                                operation,
                                Some(
                                    BarrierOperation::ArriveExpectTransaction
                                        | BarrierOperation::ExpectTransactionOnly
                                        | BarrierOperation::CompleteTransaction
                                )
                            )
                        {
                            obs.transaction_lane_mask |= mask;
                        }
                        if operation == Some(BarrierOperation::BarrierExchange)
                            && row.operand_kind.as_deref() == Some("mbarrier_register")
                        {
                            obs.exchange_token_lane_mask |= mask;
                        }
                    }
                    if row.acquire_visibility_clock.is_some() {
                        obs.visibility_clock = row.acquire_visibility_clock;
                    }
                    let query_clock = row.query_clock.or(row.visibility_clock);
                    if query_clock.is_some() {
                        obs.query_clock = query_clock;
                    }
                    obs.is_arrival |= matches!(
                        operation,
                        Some(
                            BarrierOperation::ArriveExpectTransaction
                                | BarrierOperation::ArriveOnly
                                | BarrierOperation::CountedArrival
                                | BarrierOperation::ArrivalDrop
                        )
                    );
                    obs.is_async_copy_arrival |=
                        operation == Some(BarrierOperation::AsyncCopyBarrierArrival);
                    obs.is_phase_query |= matches!(operation, Some(BarrierOperation::PhaseQuery));
                    if row.value_complete == Some(true) && row.value_mask_complete == Some(true) {
                        if let Some(values) = row.values.as_ref().filter(|v| !v.is_empty()) {
                            let uniform =
                                values.iter().all(|v| *v == values[0]).then_some(values[0]);
                            if row.phase.as_deref() == Some("before") {
                                if row.sync_role == Some(2) {
                                    match operation {
                                        Some(
                                            BarrierOperation::ArriveExpectTransaction
                                            | BarrierOperation::ExpectTransactionOnly,
                                        ) => {
                                            if let Some(bytes) = uniform {
                                                merge_operand_value(
                                                    &mut obs.transaction_operand_bytes,
                                                    Some(bytes),
                                                    &mut obs.interpretation_conflict,
                                                );
                                                if obs
                                                    .expected_tx_bytes
                                                    .is_some_and(|prior| prior != bytes)
                                                {
                                                    obs.interpretation_conflict = true;
                                                } else {
                                                    obs.expected_tx_bytes = Some(bytes);
                                                }
                                                let delta = i64::try_from(bytes).ok();
                                                if delta.is_none() {
                                                    obs.operand_precision_lost = true;
                                                }
                                                if obs.transaction_delta_bytes.is_some_and(
                                                    |prior| Some(prior) != delta.as_ref().copied(),
                                                ) {
                                                    obs.interpretation_conflict = true;
                                                } else if let Some(delta) = delta {
                                                    obs.transaction_delta_bytes = Some(delta);
                                                }
                                            }
                                        }
                                        Some(BarrierOperation::CompleteTransaction) => {
                                            if let Some(bytes) = uniform {
                                                merge_operand_value(
                                                    &mut obs.transaction_operand_bytes,
                                                    Some(bytes),
                                                    &mut obs.interpretation_conflict,
                                                );
                                                let delta = i64::try_from(bytes)
                                                    .ok()
                                                    .and_then(|value| value.checked_neg());
                                                if delta.is_none() {
                                                    obs.operand_precision_lost = true;
                                                }
                                                if obs.transaction_delta_bytes.is_some_and(
                                                    |prior| Some(prior) != delta.as_ref().copied(),
                                                ) {
                                                    obs.interpretation_conflict = true;
                                                } else if let Some(delta) = delta {
                                                    obs.transaction_delta_bytes = Some(delta);
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                    if operation == Some(BarrierOperation::CountedArrival)
                                        || operation == Some(BarrierOperation::ArrivalDrop)
                                    {
                                        merge_operand_value(
                                            &mut obs.arrival_count_operand,
                                            uniform,
                                            &mut obs.interpretation_conflict,
                                        );
                                        if uniform
                                            .is_some_and(|value| i64::try_from(value).is_err())
                                        {
                                            obs.operand_precision_lost = true;
                                        }
                                    }
                                }
                                if row.sync_role == Some(3) && obs.is_phase_query {
                                    if let Some(phase) = uniform {
                                        obs.requested_phases.insert(phase);
                                        if obs.requested_phase.is_none() {
                                            obs.requested_phase = Some(phase);
                                        } else if obs.requested_phase != Some(phase) {
                                            obs.requested_phase = None;
                                        }
                                    }
                                }
                            } else if row.phase.as_deref() == Some("after")
                                && row.sync_role == Some(4)
                                && obs.is_phase_query
                            {
                                obs.query_result_evaluated = true;
                                if let Some(mask) = row.lane_mask {
                                    obs.query_result_lane_mask |= mask;
                                }
                                let has_false = values.iter().any(|value| *value == 0);
                                let has_true = values.iter().any(|value| *value == 1);
                                obs.query_result_mixed |= has_false && has_true;
                                if let Some(clock) = query_clock {
                                    if has_false {
                                        obs.last_unsuccessful_query_clock = Some(
                                            obs.last_unsuccessful_query_clock
                                                .map_or(clock, |prior| prior.max(clock)),
                                        );
                                    }
                                    if has_true {
                                        obs.first_successful_query_clock = Some(
                                            obs.first_successful_query_clock
                                                .map_or(clock, |prior| prior.min(clock)),
                                        );
                                    }
                                }
                                if let Some(mask) = row.lane_mask {
                                    if values.len() == mask.count_ones() as usize {
                                        let mut index = 0;
                                        for lane in 0..32 {
                                            let bit = 1u32 << lane;
                                            if mask & bit != 0 {
                                                match values[index] {
                                                    0 => obs.unsuccessful_wait_lanes |= bit,
                                                    1 => obs.successful_wait_lanes |= bit,
                                                    _ => {}
                                                }
                                                index += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let increment = |counter: &mut u64, amount: u64| {
                        *counter = counter.checked_add(amount).ok_or_else(|| {
                            error(format!("line {number} observation count overflow"))
                        })?;
                        Ok::<(), AsyncSidecarError>(())
                    };
                    increment(&mut obs.records, 1)?;
                    increment(
                        &mut obs.before_records,
                        u64::from(row.phase.as_deref() == Some("before")),
                    )?;
                    increment(
                        &mut obs.after_records,
                        u64::from(row.phase.as_deref() == Some("after")),
                    )?;
                    increment(
                        &mut obs.value_incomplete_records,
                        u64::from(row.value_complete != Some(true)),
                    )?;
                    increment(
                        &mut obs.mbarrier_address_records,
                        u64::from(row.operand_kind.as_deref() == Some("mbarrier_address")),
                    )?;
                    increment(
                        &mut obs.mbarrier_register_records,
                        u64::from(row.operand_kind.as_deref() == Some("mbarrier_register")),
                    )?;
                    increment(
                        &mut obs.arrival_records,
                        u64::from(
                            row.opcode.starts_with("SYNCS.ARRIVE")
                                || row.opcode.starts_with("ARRIVES."),
                        ),
                    )?;
                    increment(
                        &mut obs.phase_check_records,
                        u64::from(row.opcode.starts_with("SYNCS.PHASECHK")),
                    )?;
                    increment(
                        &mut obs.wait_records,
                        u64::from(row.opcode.contains("TRYWAIT")),
                    )?;
                    increment(
                        &mut obs.exchange_records,
                        u64::from(row.opcode.starts_with("SYNCS.EXCH")),
                    )?;
                    increment(
                        &mut obs.exchange_token_records,
                        u64::from(
                            operation == Some(BarrierOperation::BarrierExchange)
                                && row.operand_kind.as_deref() == Some("mbarrier_register")
                                && row.value_complete == Some(true),
                        ),
                    )?;
                    if row.barrier_address_present == Some(true)
                        && let Some(address) = row.barrier_address_u32
                    {
                        obs.barrier_addresses.insert(address);
                        obs.barrier_address_raw = Some(address);
                    }
                    if let Some(rank) = row.dst_mbar_cluster_cta_id {
                        obs.barrier_cluster_rank = Some(rank);
                    }
                }
            }
        }
        result.phase_clock =
            crate::phase_sidecars::PhaseClockEvidence::load(root).map_err(|error| {
                AsyncSidecarError {
                    path: error.path,
                    message: error.message,
                }
            })?;
        Ok(result)
    }
}

/// Actual instruction-to-instruction relations. Optional clock fields remain
/// trace-local evidence and are only surfaced when the producer supplied them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationAssociation {
    EstablishedWitness,
    ProtocolScopedCandidate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AsyncRelation {
    pub association: RelationAssociation,
    pub producer: ObservationKey,
    pub wait: ObservationKey,
    pub barrier_cta: [u32; 3],
    pub barrier_offset: u32,
    pub barrier_address_raw: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destination_cluster_rank: Option<u32>,
    pub payload_bytes: u64,
    pub requested_phase: Option<u64>,
    pub successful_wait_lanes: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer_issue_clock: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer_completion_clock: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wait_visibility_clock: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct AsyncRelations {
    pub tma_loads: u64,
    pub tma_stores: u64,
    pub barriers_observed: u64,
    pub one_shot_relations: u64,
    /// Relations retained for analysis but not counted as established
    /// witnesses because raw scope/lifetime or generation identity is open.
    pub protocol_scoped_candidates: u64,
    pub unresolved_generation_barriers: u64,
    pub unresolved_payload_barriers: u64,
    pub missing_destination_barriers: u64,
    pub phase_query_instructions: u64,
    pub successful_query_instructions: u64,
    pub unsuccessful_query_instructions: u64,
    /// Evidence class for the producer-to-completion interval. `Unavailable`
    /// is the normal state for the current corpus because NVBit sidecars do
    /// not export a device completion clock; it is not a zero-cycle estimate.
    pub completion_timing: AsyncTimingEvidence,
    /// Evidence class for completion-to-consumer visibility. A successful
    /// phase query establishes ordering, not a silicon latency timestamp.
    pub visibility_timing: AsyncTimingEvidence,
    /// Evidence class for the directly observed producer-call to successful
    /// consumer-query interval. This remains distinct from completion-to-
    /// visibility because the producer completion clock may be absent.
    pub issue_visibility_timing: AsyncTimingEvidence,
    /// Kept as scalar compatibility projections for existing consumers. They
    /// are populated only for a unique interval and otherwise remain absent.
    pub native_completion_cycles: Option<u64>,
    pub native_visibility_cycles: Option<u64>,
    pub native_issue_visibility_cycles: Option<u64>,
    pub relation_samples: Vec<AsyncRelation>,
    #[serde(skip)]
    observed_completion_intervals: BTreeSet<u64>,
    #[serde(skip)]
    observed_visibility_intervals: BTreeSet<u64>,
    #[serde(skip)]
    observed_issue_visibility_intervals: BTreeSet<u64>,
}

/// Trace-local timing evidence for one async interval. This deliberately
/// distinguishes missing clocks from conflicting observed intervals so a
/// consumer cannot mistake `None` for a measured zero or a fitted constant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum AsyncTimingEvidence {
    Unavailable { reason: &'static str },
    UniqueInterval { cycles: u64 },
    AmbiguousIntervals { distinct_intervals: u64 },
}

impl Default for AsyncTimingEvidence {
    fn default() -> Self {
        Self::Unavailable {
            reason: "producer sidecar did not expose the required device clock",
        }
    }
}

fn timing_evidence(intervals: &BTreeSet<u64>, reason: &'static str) -> AsyncTimingEvidence {
    match intervals.len() {
        0 => AsyncTimingEvidence::Unavailable { reason },
        1 => intervals
            .first()
            .copied()
            .map_or(AsyncTimingEvidence::Unavailable { reason }, |cycles| {
                AsyncTimingEvidence::UniqueInterval { cycles }
            }),
        count => AsyncTimingEvidence::AmbiguousIntervals {
            distinct_intervals: u64::try_from(count).unwrap_or(u64::MAX),
        },
    }
}

fn record_relation(
    result: &mut AsyncRelations,
    cta: [u32; 3],
    address: u32,
    raw_address: Option<u32>,
    producer: ObservationKey,
    load: &TmaObservation,
    wait: ObservationKey,
    observation: &SyncObservation,
    association: RelationAssociation,
) {
    if association == RelationAssociation::EstablishedWitness {
        result.one_shot_relations += 1;
    } else {
        result.protocol_scoped_candidates += 1;
    }
    if result.relation_samples.len() < 64 {
        result.relation_samples.push(AsyncRelation {
            association,
            producer,
            wait,
            barrier_cta: cta,
            barrier_offset: load.destination_barrier.unwrap_or(address),
            barrier_address_raw: raw_address,
            destination_cluster_rank: load.destination_cluster_rank,
            payload_bytes: load.bytes.unwrap_or(0),
            requested_phase: observation.requested_phase,
            successful_wait_lanes: observation.successful_wait_lanes,
            producer_issue_clock: load.issue_clock,
            producer_completion_clock: load.completion_clock,
            wait_visibility_clock: observation.visibility_clock,
        });
    }
    if let (Some(issue), Some(completion)) = (load.issue_clock, load.completion_clock)
        && let Some(interval) = completion.checked_sub(issue)
    {
        result.observed_completion_intervals.insert(interval);
    }
    if let (Some(completion), Some(visibility)) =
        (load.completion_clock, observation.visibility_clock)
        && let Some(interval) = visibility.checked_sub(completion)
    {
        result.observed_visibility_intervals.insert(interval);
    }
    if let (Some(issue), Some(visibility)) = (load.issue_clock, observation.visibility_clock)
        && let Some(interval) = visibility.checked_sub(issue)
    {
        result.observed_issue_visibility_intervals.insert(interval);
    }
}

/// Join repeated producer/arrival/wait generations only when dynamic
/// instruction order, same-warp scope, and per-arrival byte accounting all
/// agree. This is stronger than parity/file order and deliberately refuses
/// cross-warp or incomplete windows.
fn relate_ordered_generations(
    result: &mut AsyncRelations,
    cta: [u32; 3],
    address: u32,
    raw_address: Option<u32>,
    loads: &[(ObservationKey, &TmaObservation)],
    sync: &[(ObservationKey, &SyncObservation)],
) -> bool {
    let arrivals: Vec<_> = sync
        .iter()
        .filter(|(_, observation)| {
            observation.is_arrival && observation.expected_tx_bytes.is_some()
        })
        .map(|(key, observation)| (*key, observation))
        .collect();
    let waits: Vec<_> = sync
        .iter()
        .filter(|(_, observation)| {
            observation.is_phase_query && observation.successful_wait_lanes != 0
        })
        .map(|(key, observation)| (*key, observation))
        .collect();
    if arrivals.len() < 2 || arrivals.len() != waits.len() {
        return false;
    }
    let mut matched = Vec::new();
    let mut consumed = BTreeSet::new();
    for (index, ((arrival, arrival_observation), (wait, wait_observation))) in
        arrivals.iter().zip(waits.iter()).enumerate()
    {
        if arrival.cta != wait.cta
            || arrival.warp != wait.warp
            || arrival.instruction >= wait.instruction
        {
            return false;
        }
        let previous_wait = index
            .checked_sub(1)
            .and_then(|previous| waits.get(previous))
            .map(|(key, _)| key.instruction);
        let mut window = Vec::new();
        for &(producer, load) in loads {
            if producer.cta == arrival.cta
                && producer.warp == arrival.warp
                && previous_wait.is_none_or(|previous| producer.instruction > previous)
                && producer.instruction < arrival.instruction
            {
                window.push((producer, load));
            }
        }
        if window.is_empty() {
            return false;
        }
        let Some(expected) = arrival_observation.expected_tx_bytes else {
            return false;
        };
        let total = window
            .iter()
            .try_fold(0_u64, |sum, (_, load)| sum.checked_add(load.bytes?));
        if total != Some(expected) {
            return false;
        }
        for &(producer, load) in &window {
            if !consumed.insert(producer) {
                return false;
            }
            matched.push((producer, load, *wait, wait_observation));
        }
    }
    if consumed.len() != loads.len() {
        return false;
    }
    let association = if raw_address.is_some()
        && sync
            .iter()
            .all(|(_, observation)| observation.exchange_records == 0)
    {
        RelationAssociation::EstablishedWitness
    } else {
        RelationAssociation::ProtocolScopedCandidate
    };
    for (producer, load, wait, observation) in matched {
        record_relation(
            result,
            cta,
            address,
            raw_address,
            producer,
            load,
            wait,
            observation,
            association,
        );
    }
    true
}

fn relate_barrier(
    result: &mut AsyncRelations,
    cta: [u32; 3],
    address: u32,
    raw_address: Option<u32>,
    loads: &[(ObservationKey, &TmaObservation)],
    sync: &[(ObservationKey, &SyncObservation)],
) {
    let arrivals: Vec<_> = sync.iter().filter(|(_, x)| x.is_arrival).collect();
    let initializations = sync.iter().filter(|(_, x)| x.exchange_records != 0).count();
    if (arrivals.len() != 1 || initializations != 1)
        && (arrivals.len() < 2
            || !relate_ordered_generations(result, cta, address, raw_address, loads, sync))
    {
        result.unresolved_generation_barriers += 1;
        return;
    }
    if arrivals.len() != 1 || initializations != 1 {
        return;
    }
    let expected = arrivals[0].1.expected_tx_bytes;
    let total = loads
        .iter()
        .try_fold(0u64, |sum, (_, x)| sum.checked_add(x.bytes?));
    if expected.is_none() || total != expected {
        result.unresolved_payload_barriers += 1;
        return;
    }
    // This path has one exchange witness by construction.  An exchange
    // token is useful for scoped correlation but does not establish a unique
    // initialization lifetime, so relations remain candidates.
    let association = RelationAssociation::ProtocolScopedCandidate;
    for &(wait, observation) in sync {
        if !observation.is_phase_query || observation.successful_wait_lanes == 0 {
            continue;
        }
        for &(producer, load) in loads {
            // Same-warp order is known. Different-warp order follows only
            // from the one-shot barrier and byte accounting; no timestamp is
            // inferred.
            if producer.cta == wait.cta
                && producer.warp == wait.warp
                && producer.instruction >= wait.instruction
            {
                continue;
            }
            record_relation(
                result,
                cta,
                address,
                raw_address,
                producer,
                load,
                wait,
                observation,
                association,
            );
        }
    }
}

impl LaunchObservations {
    /// Build a conservative cross-warp phase-readiness view for a bounded
    /// single-barrier cohort.  This path is evidence-only and is intentionally
    /// separate from replay relations: it requires one producer warp, exact
    /// per-generation expected-byte accounting, and one successful phase query
    /// sequence per consumer warp.  It never infers an engine completion clock.
    pub fn phase_readiness_observations(
        &self,
        cluster: [u32; 3],
    ) -> Vec<PhaseReadinessObservation> {
        type Barrier = ([u32; 3], u32);
        let mut loads_by_barrier: BTreeMap<Barrier, Vec<(ObservationKey, &TmaObservation)>> =
            BTreeMap::new();
        let mut arrivals_by_barrier: BTreeMap<Barrier, Vec<(ObservationKey, &SyncObservation)>> =
            BTreeMap::new();
        let mut queries_by_barrier: BTreeMap<
            (Barrier, i32),
            Vec<(ObservationKey, &SyncObservation)>,
        > = BTreeMap::new();

        for (&key, observation) in &self.tma {
            if observation.is_store {
                continue;
            }
            if let Some(offset) = observation.destination_barrier {
                loads_by_barrier
                    .entry((key.cta, offset))
                    .or_default()
                    .push((key, observation));
            }
        }
        for (&key, observation) in &self.sync {
            if observation.barrier_addresses.len() != 1 {
                continue;
            }
            let offset = *observation
                .barrier_addresses
                .first()
                .expect("set length checked");
            let barrier = (key.cta, offset);
            if observation.is_arrival && observation.expected_tx_bytes.is_some() {
                arrivals_by_barrier
                    .entry(barrier)
                    .or_default()
                    .push((key, observation));
            }
            if observation.is_phase_query {
                queries_by_barrier
                    .entry((barrier, key.warp))
                    .or_default()
                    .push((key, observation));
            }
        }

        let mut result = Vec::new();
        for (barrier, mut arrivals) in arrivals_by_barrier {
            let Some(loads) = loads_by_barrier.get(&barrier) else {
                continue;
            };
            arrivals.sort_by_key(|(key, _)| key.instruction);
            let producer_warps: BTreeSet<i32> = arrivals.iter().map(|(key, _)| key.warp).collect();
            if producer_warps.len() != 1 || arrivals.is_empty() {
                continue;
            }
            let producer_warp = *producer_warps.first().expect("non-empty producer set");

            // A repeated sequence needs an explicit generation token or a
            // retained lifetime/phase relation.  The sidecar projection has
            // neither.  In particular, do not pair the nth successful query
            // with the nth producer: a successful query may observe any
            // already-completed phase, and the same producer may otherwise be
            // reused for multiple arrivals.  Keep only the one-shot case in
            // this compact witness path; repeated cases remain unresolved in
            // the obligation ledger.
            if arrivals.len() != 1 {
                continue;
            }
            let (arrival, arrival_observation) = arrivals[0];
            let mut producer_candidates: Vec<_> = loads
                .iter()
                .filter(|(producer, load)| {
                    producer.cta == arrival.cta
                        && producer.warp == producer_warp
                        && producer.instruction < arrival.instruction
                        && load.bytes == arrival_observation.expected_tx_bytes
                })
                .collect();
            // Some captures place the expected-byte marker before the copy;
            // accept that ordering only when it still leaves one producer.
            if producer_candidates.is_empty() {
                producer_candidates = loads
                    .iter()
                    .filter(|(producer, load)| {
                        producer.cta == arrival.cta
                            && producer.warp == producer_warp
                            && producer.instruction > arrival.instruction
                            && load.bytes == arrival_observation.expected_tx_bytes
                    })
                    .collect();
            }
            if producer_candidates.len() != 1 {
                continue;
            }
            let (producer, load) = *producer_candidates[0];

            for ((query_barrier, _warp), queries) in queries_by_barrier.iter() {
                if *query_barrier != barrier {
                    continue;
                }
                let mut successes: Vec<_> = queries
                    .iter()
                    .copied()
                    .filter(|(_, observation)| {
                        observation.successful_wait_lanes != 0 && !observation.query_result_mixed
                    })
                    .collect();
                successes.sort_by_key(|(key, _)| key.instruction);
                // A single success is associated with the one-shot producer;
                // multiple successes at one warp are retained as ambiguous
                // rather than selected by file/instruction order.
                if successes.len() != 1 {
                    continue;
                }
                let (query, observation) = successes[0];
                let last_false = queries
                    .iter()
                    .filter(|(key, candidate)| {
                        candidate.unsuccessful_wait_lanes != 0
                            && key.instruction < query.instruction
                    })
                    .filter_map(|(_, candidate)| candidate.last_unsuccessful_query_clock)
                    .max();
                let first_true = observation.first_successful_query_clock;
                let observation_window = match (last_false, first_true) {
                    (Some(last), Some(first)) if last <= first => Some([last, first]),
                    _ => None,
                };
                let upper_only = observation_window.is_none().then_some(first_true).flatten();
                let strong_scope = load.destination_barrier_raw.is_some()
                    && observation.barrier_address_raw == load.destination_barrier_raw
                    && (cluster == [1, 1, 1]
                        || (load.destination_cluster_rank.is_some()
                            && observation.barrier_cluster_rank == load.destination_cluster_rank));
                // An exchange row is not an initialization/lifetime proof;
                // retain the relation as a candidate even when all other
                // fields happen to line up.
                let association = if strong_scope
                    && arrival_observation.exchange_records == 0
                    && observation.exchange_records == 0
                    && !arrival_observation.interpretation_conflict
                    && !observation.interpretation_conflict
                    && !arrival_observation.operand_precision_lost
                {
                    PhaseAssociation::EstablishedWitness
                } else {
                    // The byte-qualified one-shot relation is still useful
                    // as a protocol-scoped candidate, but raw-address,
                    // destination-scope, or lifetime ambiguity prevents an
                    // established witness.
                    PhaseAssociation::ProtocolScopedCandidate
                };
                result.push(PhaseReadinessObservation {
                    association,
                    producer,
                    query,
                    barrier_cta: barrier.0,
                    barrier_offset: barrier.1,
                    barrier_address_raw: load.destination_barrier_raw,
                    destination_cluster_rank: load.destination_cluster_rank,
                    payload_bytes: load.bytes.unwrap_or(0),
                    producer_issue_clock: load.issue_clock,
                    last_unsuccessful_query_clock: last_false,
                    first_successful_query_clock: first_true,
                    successful_wait_lanes: observation.successful_wait_lanes,
                    observation_window,
                    upper_only,
                    window_basis: "post_query_sample_span",
                });
            }
        }
        result
    }

    /// Join a barrier with exact one-shot accounting, or with an ordered
    /// repeated-generation sequence when each arrival/query window has
    /// same-CTA/warp instruction order and exact byte accounting. Multiple
    /// phases are never inferred from parity or file/channel order. Caller
    /// supplies captured cluster dims.
    pub fn relations(&self, _cluster: [u32; 3]) -> AsyncRelations {
        type Barrier = ([u32; 3], u32);
        let mut result = AsyncRelations::default();
        let mut raw_producers: BTreeMap<(u32, [u32; 3]), Vec<(ObservationKey, &TmaObservation)>> =
            BTreeMap::new();
        let mut fallback_producers: BTreeMap<Barrier, Vec<(ObservationKey, &TmaObservation)>> =
            BTreeMap::new();
        let mut raw_consumers: BTreeMap<(u32, [u32; 3]), Vec<(ObservationKey, &SyncObservation)>> =
            BTreeMap::new();
        let mut raw_targets: BTreeMap<u32, BTreeSet<[u32; 3]>> = BTreeMap::new();
        let mut fallback_consumers: BTreeMap<Barrier, Vec<(ObservationKey, &SyncObservation)>> =
            BTreeMap::new();

        for (&key, observation) in &self.tma {
            if observation.is_store {
                result.tma_stores += 1;
                continue;
            }
            result.tma_loads += 1;
            if let Some(raw) = observation.destination_barrier_raw {
                raw_producers
                    .entry((raw, key.cta))
                    .or_default()
                    .push((key, observation));
            } else if let Some(offset) = observation.destination_barrier {
                // Legacy rows without a raw address are safe only for a
                // single-CTA launch.  Do not reconstruct a cluster mapping
                // from a caller-supplied geometry that the sidecar did not
                // actually observe.
                if _cluster == [1, 1, 1] {
                    fallback_producers
                        .entry((key.cta, offset))
                        .or_default()
                        .push((key, observation));
                } else {
                    result.missing_destination_barriers += 1;
                }
            } else {
                result.missing_destination_barriers += 1;
            }
        }

        for (&key, observation) in &self.sync {
            result.phase_query_instructions += u64::from(observation.is_phase_query);
            result.successful_query_instructions +=
                u64::from(observation.successful_wait_lanes != 0);
            result.unsuccessful_query_instructions +=
                u64::from(observation.unsuccessful_wait_lanes != 0);
            // Divergent barrier addresses need lane-specific correlation; do
            // not associate every producer with all addresses in that warp.
            if observation.barrier_addresses.len() == 1 {
                let address = *observation.barrier_addresses.first().unwrap();
                fallback_consumers
                    .entry((key.cta, address))
                    .or_default()
                    .push((key, observation));
                raw_consumers
                    .entry((address, key.cta))
                    .or_default()
                    .push((key, observation));
                raw_targets.entry(address).or_default().insert(key.cta);
            }
        }

        // A raw address is stronger than an inferred cluster rank, but shared
        // memory offsets are commonly reused by every CTA. Prefer the
        // producer's own CTA; only use a cross-CTA target when the raw address
        // resolves to exactly one other CTA.
        result.barriers_observed = (raw_producers.len() + fallback_producers.len()) as u64;
        for ((raw, producer_cta), loads) in raw_producers {
            if let Some(sync) = raw_consumers.get(&(raw, producer_cta)) {
                relate_barrier(&mut result, producer_cta, raw, Some(raw), &loads, sync);
                continue;
            }
            let Some(targets) = raw_targets.get(&raw) else {
                result.missing_destination_barriers += 1;
                continue;
            };
            if targets.len() != 1 {
                result.unresolved_generation_barriers += 1;
                continue;
            }
            let target_cta = *targets.iter().next().unwrap();
            let Some(sync) = raw_consumers.get(&(raw, target_cta)) else {
                result.missing_destination_barriers += 1;
                continue;
            };
            relate_barrier(&mut result, target_cta, raw, Some(raw), &loads, sync);
        }

        // Legacy offset-only observations retain the conservative same-CTA
        // behavior used before raw-address transport was available.
        for ((cta, offset), loads) in fallback_producers {
            if let Some(sync) = fallback_consumers.get(&(cta, offset)) {
                relate_barrier(&mut result, cta, offset, None, &loads, sync);
            }
        }
        result.native_completion_cycles = if result.observed_completion_intervals.len() == 1 {
            result.observed_completion_intervals.first().copied()
        } else {
            None
        };
        result.native_visibility_cycles = if result.observed_visibility_intervals.len() == 1 {
            result.observed_visibility_intervals.first().copied()
        } else {
            None
        };
        result.native_issue_visibility_cycles =
            if result.observed_issue_visibility_intervals.len() == 1 {
                result.observed_issue_visibility_intervals.first().copied()
            } else {
                None
            };
        result.completion_timing = timing_evidence(
            &result.observed_completion_intervals,
            "producer sidecar did not expose issue and completion clocks",
        );
        result.visibility_timing = timing_evidence(
            &result.observed_visibility_intervals,
            "producer sidecar did not expose completion and visibility clocks",
        );
        result.issue_visibility_timing = timing_evidence(
            &result.observed_issue_visibility_intervals,
            "producer or consumer call-site clocks are unavailable",
        );
        result
    }
}
