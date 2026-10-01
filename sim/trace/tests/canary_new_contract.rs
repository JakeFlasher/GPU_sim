//! Real new-contract payload: the address-carrying-contract canary trace, captured from
//! `tools/local_labels/canary/sm86_addr_carry_canary.cu` on the RTX 3090 Ti under
//! the address-carrying contract (`trace_contract_id = sm86_addr_carry_v1`). This
//! exercises the Rust reader's allocation-identity gate + record decode on REAL
//! admitted ATOMIC / ASYNC_COPY payload with a producer-emitted allocations
//! sidecar — not synthetic fixtures. A regression that a future producer/reader
//! change silently stops admitting (or mis-validates) the flipped classes.

use std::path::Path;

use sim_mem::{
    NativePayloadIdentityV1, NativePayloadRecordV1, SectorPayloadV1, join_native_payload_records,
};
use sim_trace::{TraceDir, pb, visit_memory_access};

#[test]
fn canary_new_contract_trace_passes_reader_identity_and_carries_atomic_async() {
    let fixture = Path::new("tests/fixtures/addr_carry_canary");
    let td = TraceDir::open_direct(&fixture).expect("open canary trace dir");

    // The new contract id + capability list are on the wire.
    assert_eq!(td.trace().trace_contract_id, "sm86_addr_carry_v1");
    let caps = td.trace().memory_payload_classes.clone();
    assert!(
        caps.contains(&(pb::MemoryOpClass::Atomic as i32)),
        "capability advertises ATOMIC: {caps:?}"
    );
    assert!(
        caps.contains(&(pb::MemoryOpClass::AsyncCopy as i32)),
        "capability advertises ASYNC_COPY: {caps:?}"
    );
    // The producer emitted the allocations sidecar; the reader parsed it under the
    // exact-shape parser (would have failed TraceDir::open otherwise).
    assert!(td.allocations().is_some(), "allocations sidecar loaded");

    let mut atomic = 0usize;
    let mut async_copy = 0usize;
    for loc in td.kernels() {
        // The allocation-identity gate runs here (fail-closed) on the REAL
        // captured payload — a malformed record would error out.
        let blocks = td
            .read_threadblocks_in_launch_order(&loc)
            .expect("read threadblocks + pass the identity gate on real payload");
        for (_index, tb) in blocks {
            for warp in tb.warps.values() {
                for inst in &warp.instructions {
                    if let Some(meta) = visit_memory_access(inst, || "canary".into(), |_| {})
                        .expect("decode memory record")
                    {
                        match meta.op_class {
                            pb::MemoryOpClass::Atomic => atomic += 1,
                            pb::MemoryOpClass::AsyncCopy => async_copy += 1,
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    // The flip admitted real atomic + async-copy records with resolvable identity.
    assert!(atomic > 0, "admitted ATOMIC records present (got {atomic})");
    assert!(
        async_copy > 0,
        "admitted ASYNC_COPY (LDGSTS global source) records present (got {async_copy})"
    );

    let loc = td.kernels().next().expect("canary has one launch");
    let identities = td
        .dynamic_memory_join_identities(&loc)
        .expect("dynamic global memory identity projection");
    assert!(
        !identities.is_empty(),
        "global memory join identities present"
    );
    assert!(
        identities
            .iter()
            .all(|identity| identity.launch_id == u64::try_from(loc.kernel().id).unwrap())
    );
    let locations = td
        .dynamic_memory_join_locations(&loc)
        .expect("dynamic memory location projection");
    assert_eq!(locations.len(), identities.len());
    assert!(locations.values().all(|location| {
        location.key.cta == location.block_id
            && location.key.warp == location.warp
            && location.access_kind.is_some()
            && location.sm_id.is_some_and(|sm| sm >= 0)
    }));

    let key = *identities.iter().next().unwrap();
    let record = NativePayloadRecordV1::new(
        SectorPayloadV1::new_with_generation(9, 1, 0x100, 0, 1, [0x5a; 32], u32::MAX).unwrap(),
        NativePayloadIdentityV1 {
            launch_id: Some(key.launch_id),
            function_name: Some("canary".to_owned()),
            function_unique_id: Some(key.function_unique_id),
            pc: Some(key.pc),
            guard: None,
            occurrence: Some(key.occurrence),
            cta: Some(key.cta),
            warp: Some(key.warp),
            lane: Some(key.lane),
        },
    )
    .unwrap();
    assert_eq!(
        join_native_payload_records(&[record], &identities)
            .unwrap()
            .len(),
        1
    );
}
