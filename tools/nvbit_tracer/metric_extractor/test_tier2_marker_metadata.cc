// nvbit_tracer/metric_extractor/test_tier2_marker_metadata.cc
//
// Behavioral regression for the Tier-2 truncation-marker boundary metadata.
//
// Background. Codex's Round-14 review found the producer emitting
// TruncationRecordV1.next_sequence_id_not_emitted as ``last_sequence_id + 1``.
// That formula is correct only when at least one compact record has been
// streamed before emission stops. When the first record in the staging
// buffer is a lossy TMA (``TMA_EXPANSION_FAILED`` /
// ``TMA_EXPANSION_SECTOR_OVERFLOW``) at sequence 0, the producer reported
// next_sequence_id_not_emitted=1 even though sequence 0 was the actual
// first omitted event. The same drift surfaces on any host-budget truncation
// that happens before the first compact record is emitted.
//
// This binary drives the shared host-side serializer in
// ``tier2_frame_serializer.h`` with synthetic record arrays + synthetic
// TmaExpansionResult arrays, parses the on-wire truncation marker, and
// asserts (a) the diagnostic-field value returned by the serializer and
// (b) the actual wire bytes both report the correct first omitted sequence
// id for every scenario.
//
// Build via the Makefile target ``test_tier2_marker_metadata``; runs as a
// standalone host binary with no NVBit / CUDA dependency.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common.h"
#include "tier2_frame_serializer.h"

namespace {

CompactAccessRecord MakeNormalRecord(
    uint64_t sequence_id, uint8_t op_kind, uint8_t address_space,
    uint16_t bytes_per_access, uint16_t sector_count) {
    CompactAccessRecord r{};
    r.sequence_id = sequence_id;
    r.pc_or_instruction_id = sequence_id + 0x1000ull;
    r.cta_x = 0; r.cta_y = 0; r.cta_z = 0;
    r.warp_id = 0;
    r.bytes_per_access = bytes_per_access;
    r.lane_mask = 0xffffffffu;
    r.active_mask = 0xffffffffu;
    r.op_kind = op_kind;
    r.cache_modifier = 0;
    r.address_space = address_space;
    r.record_flags = 0;
    r.sector_count = sector_count;
    for (uint32_t k = 0; k < sector_count && k < 64; ++k) {
        r.sector_ids[k] = sequence_id * 4 + k;
    }
    for (int i = 0; i < 32; ++i) {
        r.lane_addresses[i] = 0x10000ull + sequence_id * 128ull + (uint64_t)i * 4ull;
    }
    r.tma_handle_size = 0;
    memset(r.tma_handle, 0, sizeof(r.tma_handle));
    return r;
}

CompactAccessRecord MakeTmaRecord(uint64_t sequence_id) {
    CompactAccessRecord r{};
    r.sequence_id = sequence_id;
    r.pc_or_instruction_id = 7u;  // synthetic opcode id; serializer does not use it
    r.cta_x = 0; r.cta_y = 0; r.cta_z = 0;
    r.warp_id = 0;
    r.bytes_per_access = 32;
    r.lane_mask = 0x1u;
    r.active_mask = 0x1u;
    r.op_kind = 7;
    r.cache_modifier = 0;
    r.address_space = 1;
    r.record_flags = 0;
    r.sector_count = 0;
    for (int i = 0; i < 32; ++i) r.lane_addresses[i] = 0;
    r.tma_handle_size = 16;
    memset(r.tma_handle, 0x77, 16);
    return r;
}

TmaExpansionResult MakeExpansion(
    TmaExpansionStatus status, uint16_t bytes, uint16_t sector_count,
    uint64_t base_sector) {
    TmaExpansionResult e{};
    e.status = status;
    e.bytes_per_access = bytes;
    e.sector_count = sector_count;
    for (uint32_t k = 0; k < sector_count && k < 64; ++k) {
        e.sector_ids[k] = base_sector + k;
    }
    return e;
}

// Helper to peek a little-endian unsigned value from a byte buffer.
uint64_t ReadU64LE(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (i * 8);
    return v;
}
uint32_t ReadU32LE(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (i * 8);
    return v;
}
uint16_t ReadU16LE(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

struct WireParse {
    bool valid;
    uint32_t header_flag_value;
    uint64_t trailer_offset;
    uint8_t  header_binary_sha256[32];
    bool marker_present;
    uint16_t marker_typed_reason;
    uint64_t marker_next_sequence_id;
    uint64_t marker_dropped_count;
    uint64_t marker_deterministic_sampling_seed;
    uint16_t marker_record_len;
    uint8_t  marker_record_kind;
    uint64_t trailer_access_record_count;
    uint64_t trailer_last_sequence_id;
    uint32_t trailer_flags;
    uint32_t trailer_typed_reason_code;
};

bool ParseFrame(FILE* f, WireParse* p) {
    memset(p, 0, sizeof(*p));
    fseek(f, 0, SEEK_END);
    const long fsize = ftell(f);
    if (fsize < 160 + 80) {
        fprintf(stderr, "ParseFrame: file too small (%ld)\n", fsize);
        return false;
    }
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(static_cast<size_t>(fsize));
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) return false;
    // Header magic.
    if (buf[0] != 'T' || buf[1] != 'I' || buf[2] != '2' || buf[3] != 'A') {
        fprintf(stderr, "ParseFrame: header magic mismatch\n");
        return false;
    }
    p->header_flag_value = ReadU32LE(&buf[12]);
    for (int i = 0; i < 32; ++i) p->header_binary_sha256[i] = buf[16 + i];
    p->trailer_offset = ReadU64LE(&buf[104]);
    if (p->trailer_offset + 80 > buf.size()) {
        fprintf(stderr, "ParseFrame: trailer_offset %llu past file end %zu\n",
                static_cast<unsigned long long>(p->trailer_offset), buf.size());
        return false;
    }
    // Trailer magic.
    const uint8_t* tp = &buf[p->trailer_offset];
    if (tp[0] != 'T' || tp[1] != '2' || tp[2] != 'A' || tp[3] != 'T') {
        fprintf(stderr, "ParseFrame: trailer magic mismatch\n");
        return false;
    }
    p->trailer_access_record_count = ReadU64LE(tp + 16);
    p->trailer_flags = ReadU32LE(tp + 40);
    p->trailer_typed_reason_code = ReadU32LE(tp + 44);
    p->trailer_last_sequence_id = ReadU64LE(tp + 48);
    // Marker iff flag bit 1 set.
    if (p->header_flag_value & 0x2u) {
        const uint64_t marker_off = p->trailer_offset - 56;
        if (marker_off + 56 > p->trailer_offset) return false;
        const uint8_t* mp = &buf[marker_off];
        p->marker_record_len = ReadU16LE(mp);
        p->marker_record_kind = mp[2];
        p->marker_typed_reason = ReadU16LE(mp + 4);
        p->marker_next_sequence_id = ReadU64LE(mp + 8);
        p->marker_dropped_count = ReadU64LE(mp + 32);
        p->marker_deterministic_sampling_seed = ReadU64LE(mp + 40);
        p->marker_present = true;
    }
    p->valid = true;
    return true;
}

struct ScenarioInput {
    const char* name;
    uint64_t launch_uid;
    uint8_t  binary_sha256[32];
    std::vector<CompactAccessRecord> records;
    std::vector<TmaExpansionResult> expansions;
    uint64_t device_dropped_count;
    bool device_truncated;
    uint64_t budget_limit;
    bool per_lane_on;
};

struct ScenarioExpect {
    bool marker_emitted;
    uint64_t next_sequence_id_not_emitted;
    uint64_t dropped_count;
    uint32_t marker_typed_reason_code;  // 1 = trace_budget_exceeded
    uint64_t record_count_written;
    uint64_t trailer_last_sequence_id;
    bool has_emitted_records;
    const char* typed_reason_substr;   // serializer-output typed_reason
};

int g_fail_count = 0;
int g_pass_count = 0;

template <typename T>
bool Expect(const char* sc, const char* label, T got, T expected) {
    if (got != expected) {
        fprintf(stderr,
                "FAIL %s: %s expected %llu got %llu\n",
                sc, label,
                static_cast<unsigned long long>(expected),
                static_cast<unsigned long long>(got));
        return false;
    }
    return true;
}

// Captured seed values for the cross-scenario uniqueness assertions
// pinned at the end of main(). Round 16: scenarios C, I, and J all
// use the same record shape (first-record TMA FAILED) but differ in
// launch_uid (C vs I) or binary_sha256 (C vs J); the seed MUST differ.
uint64_t g_seed_C = 0;
uint64_t g_seed_I = 0;
uint64_t g_seed_J = 0;
bool g_have_seed_C = false;
bool g_have_seed_I = false;
bool g_have_seed_J = false;

uint64_t RunScenario(const ScenarioInput& in, const ScenarioExpect& ex) {
    FILE* f = tmpfile();
    if (f == nullptr) {
        fprintf(stderr, "FAIL %s: tmpfile() failed\n", in.name);
        ++g_fail_count;
        return 0;
    }
    const uint32_t grid_dim[3] = {1, 1, 1};
    Tier2FrameSerializeOutput body;
    tier2_serialize_frame_body(
        f,
        in.launch_uid,
        in.binary_sha256,
        /*stream_ordinal=*/0,
        grid_dim,
        in.records.data(),
        static_cast<uint64_t>(in.records.size()),
        in.expansions.data(),
        in.device_dropped_count,
        in.device_truncated,
        in.budget_limit,
        in.per_lane_on,
        &body);

    WireParse wp;
    if (!ParseFrame(f, &wp)) {
        fprintf(stderr, "FAIL %s: ParseFrame failed\n", in.name);
        fclose(f);
        ++g_fail_count;
        return 0;
    }
    fclose(f);

    bool ok = true;
    // Diagnostic-side assertions on the in-memory return struct.
    if (body.marker_emitted != ex.marker_emitted) {
        fprintf(stderr, "FAIL %s: body.marker_emitted=%d expected=%d\n",
                in.name, body.marker_emitted, ex.marker_emitted);
        ok = false;
    }
    if (ex.marker_emitted) {
        ok &= Expect<uint64_t>(in.name, "body.next_sequence_id_not_emitted_value",
                               body.next_sequence_id_not_emitted_value,
                               ex.next_sequence_id_not_emitted);
        ok &= Expect<uint64_t>(in.name, "body.marker_dropped_count",
                               body.marker_dropped_count,
                               ex.dropped_count);
        ok &= Expect<uint32_t>(in.name, "body.marker_typed_reason_code",
                               body.marker_typed_reason_code,
                               ex.marker_typed_reason_code);
    }
    ok &= Expect<uint64_t>(in.name, "body.record_count_written",
                           body.record_count_written,
                           ex.record_count_written);
    ok &= Expect<uint64_t>(in.name, "body.trailer_last_sequence_id",
                           body.trailer_last_sequence_id,
                           ex.trailer_last_sequence_id);
    if (body.has_emitted_records != ex.has_emitted_records) {
        fprintf(stderr, "FAIL %s: body.has_emitted_records=%d expected=%d\n",
                in.name, body.has_emitted_records, ex.has_emitted_records);
        ok = false;
    }
    if (ex.typed_reason_substr != nullptr) {
        const char* got = body.typed_reason ? body.typed_reason : "(null)";
        if (strstr(got, ex.typed_reason_substr) == nullptr) {
            fprintf(stderr, "FAIL %s: typed_reason '%s' missing '%s'\n",
                    in.name, got, ex.typed_reason_substr);
            ok = false;
        }
    }

    // Wire-format assertions. The diagnostic field and the on-wire field
    // are computed at the same site, but asserting both pins the wiring
    // and catches any future divergence.
    const bool wire_marker_expected = ex.marker_emitted;
    if (wp.marker_present != wire_marker_expected) {
        fprintf(stderr, "FAIL %s: wire marker_present=%d expected=%d\n",
                in.name, wp.marker_present, wire_marker_expected);
        ok = false;
    }
    if (wire_marker_expected) {
        ok &= Expect<uint64_t>(in.name, "wire marker.next_sequence_id_not_emitted",
                               wp.marker_next_sequence_id,
                               ex.next_sequence_id_not_emitted);
        ok &= Expect<uint64_t>(in.name, "wire marker.dropped_record_count_estimate",
                               wp.marker_dropped_count,
                               ex.dropped_count);
        ok &= Expect<uint16_t>(in.name, "wire marker.truncation_reason",
                               wp.marker_typed_reason,
                               static_cast<uint16_t>(ex.marker_typed_reason_code));
        ok &= Expect<uint16_t>(in.name, "wire marker.record_len",
                               wp.marker_record_len, static_cast<uint16_t>(56));
        ok &= Expect<uint8_t>(in.name, "wire marker.record_kind",
                              wp.marker_record_kind, static_cast<uint8_t>(0x7e));
    }
    ok &= Expect<uint64_t>(in.name, "wire trailer.access_record_count",
                           wp.trailer_access_record_count,
                           ex.record_count_written);
    ok &= Expect<uint64_t>(in.name, "wire trailer.last_sequence_id",
                           wp.trailer_last_sequence_id,
                           ex.trailer_last_sequence_id);

    // Round 16: deterministic_sampling_seed derivation. The serializer
    // computes the seed from (launch_uid, binary_sha256); the test
    // computes the same value via the public helper and asserts BOTH the
    // diagnostic field and the on-wire bytes round-trip the expected
    // value. binary_sha256 is also round-tripped through the header.
    const uint64_t expected_seed =
        tier2_compute_deterministic_sampling_seed(
            in.launch_uid, in.binary_sha256);
    if (ex.marker_emitted) {
        ok &= Expect<uint64_t>(in.name, "body.marker_deterministic_sampling_seed",
                               body.marker_deterministic_sampling_seed,
                               expected_seed);
        ok &= Expect<uint64_t>(in.name, "wire marker.deterministic_sampling_seed",
                               wp.marker_deterministic_sampling_seed,
                               expected_seed);
    }
    // Header binary_sha256 round-trip (regardless of marker presence).
    for (int i = 0; i < 32; ++i) {
        if (wp.header_binary_sha256[i] != in.binary_sha256[i]) {
            fprintf(stderr,
                    "FAIL %s: header.binary_sha256[%d] = 0x%02x expected 0x%02x\n",
                    in.name, i, wp.header_binary_sha256[i], in.binary_sha256[i]);
            ok = false;
            break;
        }
    }

    if (ok) {
        printf("PASS %s\n", in.name);
        ++g_pass_count;
    } else {
        ++g_fail_count;
    }
    return expected_seed;
}

}  // namespace

int main() {
    // ---- Scenario A: five OK records, all fit, no marker ----
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;
        // binary_sha256 stays all-zero by value-initialization.
        in.name = "A_all_fit_no_marker";
        for (uint64_t s = 0; s < 5; ++s) {
            in.records.push_back(MakeNormalRecord(s, 1, 1, 4, 1));
        }
        in.expansions.assign(in.records.size(), TmaExpansionResult{});
        in.device_dropped_count = 0;
        in.device_truncated = false;
        in.budget_limit = 1024;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = false;
        ex.record_count_written = 5;
        ex.trailer_last_sequence_id = 4;
        ex.has_emitted_records = true;
        ex.typed_reason_substr = nullptr;
        RunScenario(in, ex);
    }

    // ---- Scenario B: budget too tight, host_budget_truncation at i=3 ----
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;
        // binary_sha256 stays all-zero by value-initialization.
        in.name = "B_budget_truncation_after_3_records";
        for (uint64_t s = 0; s < 5; ++s) {
            in.records.push_back(MakeNormalRecord(s, 1, 1, 4, 1));
        }
        in.expansions.assign(in.records.size(), TmaExpansionResult{});
        in.device_dropped_count = 0;
        in.device_truncated = false;
        // 5 records * 64 B each = 320 > 250; records_byte_ceiling = 250-56 = 194;
        // 3 records (192 B) fit, 4th (256 B) does not -> truncation at i=3.
        in.budget_limit = 250;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = true;
        ex.next_sequence_id_not_emitted = 3;
        ex.dropped_count = 2;  // host_budget_omitted = 5 - 3 = 2
        ex.marker_typed_reason_code = 1;
        ex.record_count_written = 3;
        ex.trailer_last_sequence_id = 2;
        ex.has_emitted_records = true;
        ex.typed_reason_substr = "trace_budget_exceeded";
        RunScenario(in, ex);
    }

    // ---- Scenario C: first record is op_kind=7 with TMA_EXPANSION_FAILED ----
    // This is the Round-14 review's Finding 1 head: marker MUST report
    // next_sequence_id_not_emitted = 0, NOT 1.
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;
        // binary_sha256 stays all-zero by value-initialization.
        in.name = "C_first_record_tma_failed_marker_reports_0";
        in.records.push_back(MakeTmaRecord(0));
        in.expansions.push_back(MakeExpansion(TMA_EXPANSION_FAILED, 0, 0, 0));
        in.device_dropped_count = 0;
        in.device_truncated = false;
        in.budget_limit = 1024;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = true;
        ex.next_sequence_id_not_emitted = 0;  // NOT 1
        ex.dropped_count = 1;
        ex.marker_typed_reason_code = 1;
        ex.record_count_written = 0;
        ex.trailer_last_sequence_id = 0;
        ex.has_emitted_records = false;
        ex.typed_reason_substr = "trace_budget_exceeded";
        g_seed_C = RunScenario(in, ex);
        g_have_seed_C = true;
    }

    // ---- Scenario D: first record op_kind=7 with TMA_EXPANSION_SECTOR_OVERFLOW ----
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;
        // binary_sha256 stays all-zero by value-initialization.
        in.name = "D_first_record_tma_sector_overflow_marker_reports_0";
        in.records.push_back(MakeTmaRecord(0));
        in.expansions.push_back(MakeExpansion(TMA_EXPANSION_SECTOR_OVERFLOW, 32, 64, 0x1000));
        in.device_dropped_count = 0;
        in.device_truncated = false;
        in.budget_limit = 1024;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = true;
        ex.next_sequence_id_not_emitted = 0;  // NOT 1
        ex.dropped_count = 1;
        ex.marker_typed_reason_code = 1;
        ex.record_count_written = 0;
        ex.trailer_last_sequence_id = 0;
        ex.has_emitted_records = false;
        ex.typed_reason_substr = "trace_budget_exceeded";
        RunScenario(in, ex);
    }

    // ---- Scenario E: 3 OK records then a lossy TMA at sequence 3 ----
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;
        // binary_sha256 stays all-zero by value-initialization.
        in.name = "E_three_ok_then_tma_failed_at_seq_3";
        for (uint64_t s = 0; s < 3; ++s) {
            in.records.push_back(MakeNormalRecord(s, 1, 1, 4, 1));
        }
        in.records.push_back(MakeTmaRecord(3));
        in.expansions.assign(in.records.size(), TmaExpansionResult{});
        in.expansions[3] = MakeExpansion(TMA_EXPANSION_FAILED, 0, 0, 0);
        in.device_dropped_count = 0;
        in.device_truncated = false;
        in.budget_limit = 1024;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = true;
        ex.next_sequence_id_not_emitted = 3;
        ex.dropped_count = 1;  // prefix_records - i = 4 - 3 = 1
        ex.marker_typed_reason_code = 1;
        ex.record_count_written = 3;
        ex.trailer_last_sequence_id = 2;
        ex.has_emitted_records = true;
        ex.typed_reason_substr = "trace_budget_exceeded";
        RunScenario(in, ex);
    }

    // ---- Scenario F: NO_MODELED_GLOBAL_SIDE is honest clean-skip; no marker ----
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;
        // binary_sha256 stays all-zero by value-initialization.
        in.name = "F_no_modeled_global_side_clean_skip_no_marker";
        in.records.push_back(MakeNormalRecord(0, 1, 1, 4, 1));
        in.records.push_back(MakeNormalRecord(1, 1, 1, 4, 1));
        in.records.push_back(MakeTmaRecord(2));
        in.records.push_back(MakeNormalRecord(3, 1, 1, 4, 1));
        in.expansions.assign(in.records.size(), TmaExpansionResult{});
        in.expansions[2] =
            MakeExpansion(TMA_EXPANSION_NO_MODELED_GLOBAL_SIDE, 0, 0, 0);
        in.device_dropped_count = 0;
        in.device_truncated = false;
        in.budget_limit = 1024;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = false;
        ex.record_count_written = 3;  // 0, 1, 3 emitted; 2 cleanly skipped
        ex.trailer_last_sequence_id = 3;
        ex.has_emitted_records = true;
        ex.typed_reason_substr = nullptr;
        RunScenario(in, ex);
    }

    // ---- Scenario G: device-capacity truncation, no host-side break ----
    // device_truncated=true; 4 OK records all fit budget. Marker reports
    // next_sequence_id_not_emitted = prefix_records = 4.
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;
        // binary_sha256 stays all-zero by value-initialization.
        in.name = "G_device_truncated_marker_reports_prefix_records";
        for (uint64_t s = 0; s < 4; ++s) {
            in.records.push_back(MakeNormalRecord(s, 1, 1, 4, 1));
        }
        in.expansions.assign(in.records.size(), TmaExpansionResult{});
        in.device_dropped_count = 7;
        in.device_truncated = true;
        in.budget_limit = 4096;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = true;
        ex.next_sequence_id_not_emitted = 4;
        ex.dropped_count = 7;
        ex.marker_typed_reason_code = 1;
        ex.record_count_written = 4;
        ex.trailer_last_sequence_id = 3;
        ex.has_emitted_records = true;
        ex.typed_reason_substr = "trace_budget_exceeded";
        RunScenario(in, ex);
    }

    // ---- Scenario H: budget below marker floor; no marker emitted ----
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;
        // binary_sha256 stays all-zero by value-initialization.
        in.name = "H_budget_below_marker_floor_no_marker";
        for (uint64_t s = 0; s < 5; ++s) {
            in.records.push_back(MakeNormalRecord(s, 1, 1, 4, 1));
        }
        in.expansions.assign(in.records.size(), TmaExpansionResult{});
        in.device_dropped_count = 0;
        in.device_truncated = false;
        in.budget_limit = 32;  // below 56-byte marker floor
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = false;
        ex.record_count_written = 0;
        ex.trailer_last_sequence_id = 0;
        ex.has_emitted_records = false;
        ex.typed_reason_substr = "tier2_trace_budget_below_marker_floor";
        RunScenario(in, ex);
    }

    // ---- Scenario I: same shape as C, but different launch_uid ----
    // Round 16 / task5: the deterministic_sampling_seed must vary with
    // launch_uid. Same record set + binary_sha256 as C; only launch_uid
    // differs. Cross-scenario assertion below checks seed_I != seed_C.
    {
        ScenarioInput in{};
        in.launch_uid = 0x0123456789ABCDEFull;  // distinct from scenario C
        // binary_sha256 stays all-zero (matching scenario C).
        in.name = "I_seed_varies_with_launch_uid";
        in.records.push_back(MakeTmaRecord(0));
        in.expansions.push_back(MakeExpansion(TMA_EXPANSION_FAILED, 0, 0, 0));
        in.device_dropped_count = 0;
        in.device_truncated = false;
        in.budget_limit = 1024;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = true;
        ex.next_sequence_id_not_emitted = 0;
        ex.dropped_count = 1;
        ex.marker_typed_reason_code = 1;
        ex.record_count_written = 0;
        ex.trailer_last_sequence_id = 0;
        ex.has_emitted_records = false;
        ex.typed_reason_substr = "trace_budget_exceeded";
        g_seed_I = RunScenario(in, ex);
        g_have_seed_I = true;
    }

    // ---- Scenario J: same shape as C, but different binary_sha256 ----
    // Round 16 / task5: the deterministic_sampling_seed must also vary
    // with binary_sha256. Same record set + launch_uid as C; only
    // binary_sha256 differs. Cross-scenario assertion below checks
    // seed_J != seed_C and seed_J != seed_I.
    {
        ScenarioInput in{};
        in.launch_uid = 0xCAFEBABEDEADBEEFull;  // matches C
        for (int i = 0; i < 32; ++i) {
            in.binary_sha256[i] = static_cast<uint8_t>(0xA5);  // != 0 (C's value)
        }
        in.name = "J_seed_varies_with_binary_sha256";
        in.records.push_back(MakeTmaRecord(0));
        in.expansions.push_back(MakeExpansion(TMA_EXPANSION_FAILED, 0, 0, 0));
        in.device_dropped_count = 0;
        in.device_truncated = false;
        in.budget_limit = 1024;
        in.per_lane_on = false;
        ScenarioExpect ex{};
        ex.marker_emitted = true;
        ex.next_sequence_id_not_emitted = 0;
        ex.dropped_count = 1;
        ex.marker_typed_reason_code = 1;
        ex.record_count_written = 0;
        ex.trailer_last_sequence_id = 0;
        ex.has_emitted_records = false;
        ex.typed_reason_substr = "trace_budget_exceeded";
        g_seed_J = RunScenario(in, ex);
        g_have_seed_J = true;
    }

    // ---- Scenario K: same-input determinism ----
    // Codex Round-16-review Finding 4: serialize two frames with
    // byte-identical inputs and assert byte-identical frame bytes +
    // identical sidecar SHA-256. The producer must be a pure function
    // of its inputs; any time-of-day or address-based nondeterminism
    // here would make sidecars non-replayable across runs.
    {
        ScenarioInput k_in{};
        k_in.launch_uid = 0xDEADC0DEDEADC0DEull;
        for (int i = 0; i < 32; ++i) {
            k_in.binary_sha256[i] = static_cast<uint8_t>(0x5A);
        }
        k_in.name = "K_same_input_twice_identical_frame";
        for (uint64_t s = 0; s < 3; ++s) {
            k_in.records.push_back(MakeNormalRecord(s, 1, 1, 4, 1));
        }
        k_in.records.push_back(MakeTmaRecord(3));
        k_in.expansions.assign(k_in.records.size(), TmaExpansionResult{});
        k_in.expansions[3] = MakeExpansion(TMA_EXPANSION_FAILED, 0, 0, 0);
        k_in.device_dropped_count = 0;
        k_in.device_truncated = false;
        k_in.budget_limit = 1024;
        k_in.per_lane_on = false;

        auto write_once = [&](std::vector<uint8_t>* out_bytes,
                              Tier2FrameSerializeOutput* out_body) -> bool {
            FILE* f = tmpfile();
            if (f == nullptr) return false;
            const uint32_t grid_dim[3] = {1, 1, 1};
            tier2_serialize_frame_body(
                f, k_in.launch_uid, k_in.binary_sha256,
                /*stream_ordinal=*/0, grid_dim,
                k_in.records.data(),
                static_cast<uint64_t>(k_in.records.size()),
                k_in.expansions.data(),
                k_in.device_dropped_count, k_in.device_truncated,
                k_in.budget_limit, k_in.per_lane_on, out_body);
            fseek(f, 0, SEEK_END);
            const long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            out_bytes->resize(static_cast<size_t>(sz));
            const size_t got = fread(out_bytes->data(), 1, out_bytes->size(), f);
            fclose(f);
            return got == out_bytes->size();
        };

        Tier2FrameSerializeOutput body_a;
        Tier2FrameSerializeOutput body_b;
        std::vector<uint8_t> bytes_a;
        std::vector<uint8_t> bytes_b;
        bool ok = write_once(&bytes_a, &body_a)
                  && write_once(&bytes_b, &body_b);
        if (!ok) {
            fprintf(stderr,
                    "FAIL K_same_input_twice_identical_frame: "
                    "write_once setup failed\n");
            ++g_fail_count;
        } else if (bytes_a != bytes_b) {
            fprintf(stderr,
                    "FAIL K_same_input_twice_identical_frame: "
                    "frame bytes differ between runs "
                    "(%zu vs %zu bytes)\n",
                    bytes_a.size(), bytes_b.size());
            ++g_fail_count;
        } else if (strcmp(body_a.sha256_hex, body_b.sha256_hex) != 0) {
            fprintf(stderr,
                    "FAIL K_same_input_twice_identical_frame: "
                    "sidecar SHA-256 hex differs (%s vs %s)\n",
                    body_a.sha256_hex, body_b.sha256_hex);
            ++g_fail_count;
        } else if (body_a.marker_deterministic_sampling_seed
                   != body_b.marker_deterministic_sampling_seed) {
            fprintf(stderr,
                    "FAIL K_same_input_twice_identical_frame: "
                    "marker_deterministic_sampling_seed differs "
                    "(0x%016llx vs 0x%016llx)\n",
                    static_cast<unsigned long long>(
                        body_a.marker_deterministic_sampling_seed),
                    static_cast<unsigned long long>(
                        body_b.marker_deterministic_sampling_seed));
            ++g_fail_count;
        } else {
            printf("PASS K_same_input_twice_identical_frame\n");
            ++g_pass_count;
        }
    }

    // ---- Cross-scenario uniqueness: seeds for C, I, J MUST differ ----
    // This is the positive "varies-with-inputs" test. Determinism (same
    // inputs -> same seed) is already verified inside RunScenario via the
    // body.marker_deterministic_sampling_seed == wire seed ==
    // tier2_compute_deterministic_sampling_seed(launch_uid, binary_sha256)
    // chain of equalities.
    if (g_have_seed_C && g_have_seed_I) {
        if (g_seed_C == g_seed_I) {
            fprintf(stderr,
                    "FAIL cross_scenario_seed_uniqueness: "
                    "seed_C == seed_I == 0x%016llx (launch_uid varied; "
                    "seeds MUST differ)\n",
                    static_cast<unsigned long long>(g_seed_C));
            ++g_fail_count;
        } else {
            printf("PASS cross_scenario_seed_varies_with_launch_uid\n");
            ++g_pass_count;
        }
    }
    if (g_have_seed_C && g_have_seed_J) {
        if (g_seed_C == g_seed_J) {
            fprintf(stderr,
                    "FAIL cross_scenario_seed_uniqueness: "
                    "seed_C == seed_J == 0x%016llx (binary_sha256 varied; "
                    "seeds MUST differ)\n",
                    static_cast<unsigned long long>(g_seed_C));
            ++g_fail_count;
        } else {
            printf("PASS cross_scenario_seed_varies_with_binary_sha256\n");
            ++g_pass_count;
        }
    }

    fprintf(stderr, "\nsummary: %d passed, %d failed\n",
            g_pass_count, g_fail_count);
    return g_fail_count == 0 ? 0 : 1;
}
