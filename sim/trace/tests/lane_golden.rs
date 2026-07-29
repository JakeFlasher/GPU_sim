//! Golden-fixture lockstep test: the Rust side of the lane-reconstruction
//! LOCKSTEP PAIR. The Python twin is
//! `tools/nvbit_tracer/test_lane_reconstruction_golden.py`; both consume
//! `tools/nvbit_tracer/lane_reconstruction_golden.json`. Extend the fixture
//! in the same commit as any semantic change to either decoder.

#![allow(clippy::unwrap_used)] // test binary: unwrap is an assertion

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;
use sim_trace::pb;
use sim_trace::{LANE_INACTIVE_SENTINEL, LANES_PER_WARP, LaneError, reconstruct_lane_vector};

#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    compression_format: i32,
    base_address: u64,
    stride: i32,
    #[serde(default)]
    addrs: Option<Vec<i64>>,
    #[serde(default)]
    addrs_list_all: Option<BTreeMap<String, i64>>,
    #[serde(default)]
    addrs_raw_len: Option<usize>,
    and_mask: u32,
    #[serde(default)]
    expect_lanes: Option<BTreeMap<String, serde_json::Value>>,
    #[serde(default)]
    error: Option<String>,
}

fn golden_path() -> PathBuf {
    PathBuf::from("../../tools/nvbit_tracer/lane_reconstruction_golden.json")
        .canonicalize()
        .expect("golden fixture exists in-repo")
}

fn wire_addrs(case: &Case) -> Vec<i64> {
    if let Some(len) = case.addrs_raw_len {
        return vec![LANE_INACTIVE_SENTINEL; len];
    }
    if let Some(list_all) = &case.addrs_list_all {
        let mut lanes = vec![LANE_INACTIVE_SENTINEL; LANES_PER_WARP];
        for (lane, value) in list_all {
            lanes[lane.parse::<usize>().unwrap()] = *value;
        }
        return lanes;
    }
    case.addrs.clone().expect("case carries addrs")
}

fn expected_vector(case: &Case) -> [i64; LANES_PER_WARP] {
    let spec = case.expect_lanes.as_ref().unwrap();
    let mut lanes = [LANE_INACTIVE_SENTINEL; LANES_PER_WARP];
    if let Some(sf) = spec.get("__stride_from_base") {
        let pair: Vec<i64> = serde_json::from_value(sf.clone()).unwrap();
        let (base, stride) = (pair[0], pair[1]);
        for (k, lane) in lanes.iter_mut().enumerate() {
            if (case.and_mask >> k) & 1 == 1 {
                *lane = base + (k as i64) * stride;
            }
        }
        return lanes;
    }
    for (lane, value) in spec {
        lanes[lane.parse::<usize>().unwrap()] = value.as_i64().unwrap();
    }
    lanes
}

/// Shared outcome tokens <-> Rust error variants.
fn token(err: &LaneError) -> &'static str {
    match err {
        LaneError::ListAllLengthInvalid { .. } => "list_all_length",
        LaneError::LanePositionMismatch { .. } => "position_mismatch",
        LaneError::BaseStrideAddrsNonEmpty { .. } => "stride_addrs_nonempty",
        LaneError::BaseStrideSingleLane => "stride_single_lane",
        LaneError::BaseStrideNonContiguous { .. } => "stride_non_contiguous",
        LaneError::BaseDeltaCountInvalid { .. } => "delta_count",
        LaneError::AddressOverflow { .. } => "address_overflow",
        LaneError::UnknownFormat(_) => "unknown_format",
    }
}

#[test]
fn golden_cases_match() {
    let golden: Golden =
        serde_json::from_str(&std::fs::read_to_string(golden_path()).unwrap()).unwrap();
    assert!(!golden.cases.is_empty());
    for case in &golden.cases {
        let addr = pb::Address {
            data_width: 4,
            udesc_value: 0,
            compression_format: case.compression_format,
            base_address: case.base_address,
            stride: case.stride,
            addrs: wire_addrs(case),
            allocation_id: None,
            address_space: None,
        };
        let result = reconstruct_lane_vector(&addr, case.and_mask);
        match (&case.error, result) {
            (Some(expected), Err(err)) => {
                assert_eq!(
                    token(&err),
                    expected,
                    "{}: wrong error token ({err})",
                    case.name
                );
            }
            (Some(expected), Ok(v)) => {
                panic!("{}: expected error {expected}, got vector {v:?}", case.name)
            }
            (None, Ok(vector)) => {
                assert_eq!(
                    vector,
                    expected_vector(case),
                    "{}: vector mismatch",
                    case.name
                );
            }
            (None, Err(err)) => panic!("{}: unexpected error {err}", case.name),
        }
    }
}
