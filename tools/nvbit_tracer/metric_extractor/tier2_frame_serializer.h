// nvbit_tracer/metric_extractor/tier2_frame_serializer.h
//
// Header-only Tier-2 sidecar frame writer. Hosts the file-write portion of the
// per-launch producer. The runtime caller is responsible for opening the
// file, computing the per-launch UID, and running NVBit-side TMA descriptor
// expansion; the serializer accepts a precomputed TmaExpansionResult array
// and never invokes NVBit.

#ifndef NVBIT_TRACER_METRIC_EXTRACTOR_TIER2_FRAME_SERIALIZER_H_
#define NVBIT_TRACER_METRIC_EXTRACTOR_TIER2_FRAME_SERIALIZER_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "common.h"
#include "tier2_sha256.h"

enum TmaExpansionStatus {
    TMA_EXPANSION_OK = 0,
    TMA_EXPANSION_NO_MODELED_GLOBAL_SIDE = 1,
    TMA_EXPANSION_FAILED = 2,
    TMA_EXPANSION_SECTOR_OVERFLOW = 3,
};

struct TmaExpansionResult {
    TmaExpansionStatus status;
    uint16_t bytes_per_access;
    uint16_t sector_count;
    uint64_t sector_ids[METRIC_EXTRACTOR_TIER2_SECTORS_PER_RECORD];
};

// Per-frame serializer output. The runtime caller projects this onto its own
// Tier2EmitResult shape used by the runtime writer.
struct Tier2FrameSerializeOutput {
    bool emitted;
    bool truncated;
    const char* typed_reason;
    uint64_t trace_bytes;
    uint8_t  digest[32];
    char     sha256_hex[65];
    bool     marker_emitted;
    uint64_t next_sequence_id_not_emitted_value;
    uint64_t marker_dropped_count;
    uint64_t marker_deterministic_sampling_seed;
    uint32_t marker_typed_reason_code;
    uint64_t record_count_written;
    uint64_t per_lane_records_written;
    uint64_t trailer_last_sequence_id;
    bool     has_emitted_records;
};

static inline void tier2_put_u16_le(
    FILE* f, uint16_t v, tier2::Sha256Ctx* h, bool digest) {
    uint8_t buf[2] = {
        static_cast<uint8_t>(v & 0xff),
        static_cast<uint8_t>((v >> 8) & 0xff),
    };
    fwrite(buf, 1, 2, f);
    if (digest && h) h->Update(buf, 2);
}

static inline void tier2_put_u32_le(
    FILE* f, uint32_t v, tier2::Sha256Ctx* h, bool digest) {
    uint8_t buf[4];
    for (int i = 0; i < 4; ++i) {
        buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
    }
    fwrite(buf, 1, 4, f);
    if (digest && h) h->Update(buf, 4);
}

static inline void tier2_put_u64_le(
    FILE* f, uint64_t v, tier2::Sha256Ctx* h, bool digest) {
    uint8_t buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
    }
    fwrite(buf, 1, 8, f);
    if (digest && h) h->Update(buf, 8);
}

static inline void tier2_put_zeros(
    FILE* f, size_t n, tier2::Sha256Ctx* h, bool digest) {
    static const uint8_t z[64] = {0};
    while (n > 0) {
        const size_t take = (n < sizeof(z)) ? n : sizeof(z);
        fwrite(z, 1, take, f);
        if (digest && h) h->Update(z, take);
        n -= take;
    }
}

static inline void tier2_put_bytes(
    FILE* f, const uint8_t* p, size_t n,
    tier2::Sha256Ctx* h, bool digest) {
    if (n == 0) return;
    fwrite(p, 1, n, f);
    if (digest && h) h->Update(p, n);
}

static inline size_t tier2_write_compact_record(
    FILE* f, const CompactAccessRecord& src, tier2::Sha256Ctx& h) {
    const uint32_t sector_count = src.sector_count;
    const uint32_t fixed_prefix = 56;
    const uint32_t sector_bytes = sector_count * 8u;
    uint32_t total = fixed_prefix + sector_bytes;
    if ((total & 7u) != 0) total += 8u - (total & 7u);
    tier2_put_u16_le(f, static_cast<uint16_t>(total), &h, true);
    const uint8_t kind_and_hdr[2] = {0x01u, 0x38u};
    tier2_put_bytes(f, kind_and_hdr, 2, &h, true);
    const uint8_t enums[4] = {
        src.op_kind, src.cache_modifier, src.address_space, src.record_flags,
    };
    tier2_put_bytes(f, enums, 4, &h, true);
    tier2_put_u64_le(f, src.sequence_id, &h, true);
    tier2_put_u64_le(f, src.pc_or_instruction_id, &h, true);
    tier2_put_u32_le(f, src.cta_x, &h, true);
    tier2_put_u32_le(f, src.cta_y, &h, true);
    tier2_put_u32_le(f, src.cta_z, &h, true);
    tier2_put_u16_le(f, src.warp_id, &h, true);
    tier2_put_u16_le(f, src.bytes_per_access, &h, true);
    tier2_put_u32_le(f, src.lane_mask, &h, true);
    tier2_put_u32_le(f, src.active_mask, &h, true);
    tier2_put_u16_le(f, static_cast<uint16_t>(sector_count), &h, true);
    // Low byte: l1_access_policy enum. High byte: reserved-zero.
    // The u32 at wire offset 52..55 stays reserved-zero.
    const uint8_t pad_pair[2] = {src.l1_access_policy, 0};
    tier2_put_bytes(f, pad_pair, 2, &h, true);
    tier2_put_u32_le(f, 0, &h, true);
    for (uint32_t i = 0; i < sector_count; ++i) {
        tier2_put_u64_le(f, src.sector_ids[i], &h, true);
    }
    const uint32_t aligned_used = fixed_prefix + sector_bytes;
    if (total > aligned_used) {
        tier2_put_zeros(f, total - aligned_used, &h, true);
    }
    return total;
}

static inline size_t tier2_write_per_lane_record(
    FILE* f, const CompactAccessRecord& src, tier2::Sha256Ctx& h) {
    const uint16_t record_len = 288u;
    tier2_put_u16_le(f, record_len, &h, true);
    const uint8_t kind_and_hdr[2] = {0x02u, 0x20u};
    tier2_put_bytes(f, kind_and_hdr, 2, &h, true);
    tier2_put_u32_le(f, src.active_mask, &h, true);   // present_mask mirror
    tier2_put_u64_le(f, src.sequence_id, &h, true);
    tier2_put_u32_le(f, src.lane_mask, &h, true);
    tier2_put_u32_le(f, src.active_mask, &h, true);
    tier2_put_zeros(f, 8, &h, true);
    for (int i = 0; i < 32; ++i) {
        tier2_put_u64_le(f, src.lane_addresses[i], &h, true);
    }
    return record_len;
}

// Deterministic per-launch sampling seed stored in TruncationRecordV1's
// ``deterministic_sampling_seed`` field. Round 16 closes Codex Round-15
// review Finding 3 / task5: the field was previously written as the
// literal 0; the schema (Section 5) calls for the first 64 bits little-
// endian of SHA-256("solbench-tier2-sampling-v1\0" || uint64_le(launch_uid)
// || binary_sha256). The seed is deterministic for a given (launch_uid,
// binary_sha256) pair so repeated runs of the same kernel against the same
// binary produce the same seed, which downstream consumers can rely on for
// reproducibility of any future sampling/sub-sampling decisions.
static inline uint64_t tier2_compute_deterministic_sampling_seed(
    uint64_t launch_uid, const uint8_t binary_sha256[32]) {
    tier2::Sha256Ctx ctx;
    static const char prefix[] = "solbench-tier2-sampling-v1";
    ctx.Update(prefix, sizeof(prefix));  // includes trailing NUL
    uint8_t buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<uint8_t>((launch_uid >> (i * 8)) & 0xff);
    }
    ctx.Update(buf, 8);
    ctx.Update(binary_sha256, 32);
    uint8_t digest[32];
    ctx.Finalize(digest);
    uint64_t seed = 0;
    for (int i = 0; i < 8; ++i) {
        seed |= static_cast<uint64_t>(digest[i]) << (i * 8);
    }
    return seed;
}

// Writes a complete Tier-2 sidecar frame to ``f`` starting at the current
// file position. Output bytes are: LaunchHeaderV1 (160 B), variable-length
// CompactAccessRecordV1 records (each optionally followed by a 288 B
// PerLaneAddressRecordV1 when per_lane_on), optional 56 B TruncationRecordV1,
// LaunchTrailerV1 (80 B). Counter/digest header fields are seek-patched in
// place once the trailer is streamed.
//
// The serializer is NVBit-free: TMA descriptor expansion is the caller's
// responsibility. expansions[i] is read only when records[i].op_kind == 7.
static inline void tier2_serialize_frame_body(
    FILE* f,
    uint64_t launch_uid,
    const uint8_t binary_sha256[32],
    int64_t  stream_ordinal,
    const uint32_t grid_dim[3],
    const CompactAccessRecord* records,
    uint64_t prefix_records,
    const TmaExpansionResult* expansions,
    uint64_t device_dropped_count,
    bool     device_truncated,
    uint64_t budget_limit,
    bool     per_lane_on,
    Tier2FrameSerializeOutput* out) {
    memset(out, 0, sizeof(*out));
    out->typed_reason = nullptr;

    const uint64_t frame_start = static_cast<uint64_t>(ftell(f));

    static const uint8_t header_magic[4] = {'T', 'I', '2', 'A'};
    fwrite(header_magic, 1, 4, f);
    {
        uint8_t hdr[156];
        memset(hdr, 0, sizeof(hdr));
        hdr[0] = 1; hdr[1] = 0;
        hdr[2] = 0; hdr[3] = 0;
        hdr[4] = 160 & 0xff;
        hdr[5] = (160 >> 8) & 0xff;
        hdr[6] = 0x02; hdr[7] = 0x01;
        // binary_sha256 at file offset 16 (hdr index 12), 32 bytes.
        // Copy the caller-supplied binary identity. Seed derivation consumes
        // this field along with the launch identity.
        for (int i = 0; i < 32; ++i) hdr[12 + i] = binary_sha256[i];
        const uint32_t launch_uid_off = 48 - 4;
        for (int i = 0; i < 8; ++i) {
            hdr[launch_uid_off + i] =
                static_cast<uint8_t>((launch_uid >> (i * 8)) & 0xff);
        }
        const uint32_t stream_off = 56 - 4;
        const uint32_t stream_raw =
            static_cast<uint32_t>(static_cast<uint64_t>(stream_ordinal));
        for (int i = 0; i < 4; ++i) {
            hdr[stream_off + i] =
                static_cast<uint8_t>((stream_raw >> (i * 8)) & 0xff);
        }
        for (int g = 0; g < 3; ++g) {
            const uint32_t off = (60 - 4) + (g * 4);
            for (int i = 0; i < 4; ++i) {
                hdr[off + i] =
                    static_cast<uint8_t>((grid_dim[g] >> (i * 8)) & 0xff);
            }
        }
        fwrite(hdr, 1, 156, f);
    }
    const uint64_t records_start_offset = frame_start + 160;
    fseek(f, records_start_offset, SEEK_SET);

    tier2::Sha256Ctx digest_ctx;
    uint64_t record_count_written = 0;
    uint64_t budget_bytes_used = 0;
    const uint64_t marker_floor = 56;
    const uint64_t per_lane_record_size = 288;
    const bool budget_below_marker_floor = budget_limit < marker_floor;

    uint64_t prefix_total_bytes = 0;
    bool tma_forces_truncation = false;
    for (uint64_t i = 0; i < prefix_records; ++i) {
        const CompactAccessRecord& rec = records[i];
        uint16_t effective_sector_count = rec.sector_count;
        if (rec.op_kind == 7u) {
            const TmaExpansionStatus s = expansions[i].status;
            if (s == TMA_EXPANSION_OK) {
                effective_sector_count = expansions[i].sector_count;
            } else if (s == TMA_EXPANSION_NO_MODELED_GLOBAL_SIDE) {
                continue;
            } else {
                tma_forces_truncation = true;
                break;
            }
        }
        uint32_t prospective_len =
            56u + static_cast<uint32_t>(effective_sector_count) * 8u;
        if ((prospective_len & 7u) != 0) {
            prospective_len += 8u - (prospective_len & 7u);
        }
        prefix_total_bytes += prospective_len;
        if (per_lane_on) {
            prefix_total_bytes += per_lane_record_size;
        }
    }

    const bool marker_required =
        device_truncated ||
        tma_forces_truncation ||
        (prefix_total_bytes > budget_limit);

    bool budget_starves_marker = false;
    uint64_t records_byte_ceiling;
    if (marker_required) {
        if (budget_below_marker_floor) {
            records_byte_ceiling = 0;
            budget_starves_marker = true;
        } else {
            records_byte_ceiling = budget_limit - marker_floor;
        }
    } else {
        records_byte_ceiling = budget_limit;
    }

    bool host_budget_truncation = false;
    uint64_t host_budget_omitted = 0;
    uint64_t per_lane_records_written = 0;
    // Round 15: explicit first-omitted-sequence tracking. The
    // Round-14 producer derived ``next_sequence_id_not_emitted`` as
    // ``last_sequence_id + 1``, which is wrong when the first emitted
    // boundary is a lossy TMA at sequence 0, when host-budget
    // truncation fires before any compact record is emitted, or when
    // ``NO_MODELED_GLOBAL_SIDE`` skips push the stream boundary away
    // from ``last_sequence_id + 1``. Track the value explicitly on
    // every host-side break instead.
    bool has_next_sequence_id_not_emitted = false;
    uint64_t next_sequence_id_not_emitted = 0;
    bool has_emitted_records = false;
    uint64_t last_emitted_sequence_id = 0;

    for (uint64_t i = 0; i < prefix_records; ++i) {
        const CompactAccessRecord& rec_src = records[i];
        CompactAccessRecord rec = rec_src;
        if (rec_src.op_kind == 7u) {
            const TmaExpansionResult& exp = expansions[i];
            switch (exp.status) {
                case TMA_EXPANSION_OK:
                    rec.bytes_per_access = exp.bytes_per_access;
                    rec.sector_count = exp.sector_count;
                    for (uint32_t k = 0; k < exp.sector_count; ++k) {
                        rec.sector_ids[k] = exp.sector_ids[k];
                    }
                    break;
                case TMA_EXPANSION_NO_MODELED_GLOBAL_SIDE:
                    continue;
                case TMA_EXPANSION_FAILED:
                case TMA_EXPANSION_SECTOR_OVERFLOW:
                    host_budget_truncation = true;
                    host_budget_omitted = prefix_records - i;
                    if (!has_next_sequence_id_not_emitted) {
                        has_next_sequence_id_not_emitted = true;
                        next_sequence_id_not_emitted = rec_src.sequence_id;
                    }
                    goto emit_loop_done;
            }
        }
        const uint32_t fixed_prefix = 56;
        const uint32_t sector_bytes =
            static_cast<uint32_t>(rec.sector_count) * 8u;
        uint32_t prospective_len = fixed_prefix + sector_bytes;
        if ((prospective_len & 7u) != 0) {
            prospective_len += 8u - (prospective_len & 7u);
        }
        const uint64_t pair_len =
            per_lane_on
                ? static_cast<uint64_t>(prospective_len) + per_lane_record_size
                : static_cast<uint64_t>(prospective_len);
        if (budget_bytes_used + pair_len > records_byte_ceiling) {
            host_budget_truncation = true;
            host_budget_omitted = prefix_records - i;
            if (!has_next_sequence_id_not_emitted) {
                has_next_sequence_id_not_emitted = true;
                next_sequence_id_not_emitted = rec_src.sequence_id;
            }
            break;
        }
        if (per_lane_on) {
            rec.record_flags = static_cast<uint8_t>(rec.record_flags | 0x1u);
        }
        const size_t compact_bytes =
            tier2_write_compact_record(f, rec, digest_ctx);
        budget_bytes_used += compact_bytes;
        record_count_written += 1;
        has_emitted_records = true;
        last_emitted_sequence_id = rec.sequence_id;
        if (per_lane_on) {
            const size_t per_lane_bytes =
                tier2_write_per_lane_record(f, rec, digest_ctx);
            budget_bytes_used += per_lane_bytes;
            per_lane_records_written += 1;
        }
    }
emit_loop_done:;
    uint32_t trailer_typed_reason_code = 0;
    const bool truncated_now = device_truncated || host_budget_truncation;
    if (budget_starves_marker) {
        out->truncated = true;
        out->typed_reason = "tier2_trace_budget_below_marker_floor";
    } else if (truncated_now) {
        out->truncated = true;
        out->typed_reason = "trace_budget_exceeded";
        // Resolve marker next_sequence_id_not_emitted from three sources
        // in priority order: host-side break (the explicit per-break
        // assignment above), device-capacity overflow with no host-side
        // break (the first sequence id past the staging capacity), or
        // empty frame with marker-only emission (0 is correct because
        // the trailer's access_record_count==0 disambiguates).
        uint64_t marker_next_seq;
        if (has_next_sequence_id_not_emitted) {
            marker_next_seq = next_sequence_id_not_emitted;
        } else if (device_truncated) {
            marker_next_seq = prefix_records;
        } else {
            marker_next_seq = 0;
        }
        const uint16_t trunc_len = 56;
        tier2_put_u16_le(f, trunc_len, &digest_ctx, true);
        const uint8_t trunc_hdr[2] = {0x7e, 0x38};
        tier2_put_bytes(f, trunc_hdr, 2, &digest_ctx, true);
        tier2_put_u16_le(f, 1, &digest_ctx, true);
        tier2_put_u16_le(f, 0, &digest_ctx, true);
        tier2_put_u64_le(f, marker_next_seq, &digest_ctx, true);
        tier2_put_u64_le(f, budget_limit, &digest_ctx, true);
        tier2_put_u64_le(f, budget_bytes_used, &digest_ctx, true);
        const uint64_t dropped = device_dropped_count + host_budget_omitted;
        tier2_put_u64_le(f, dropped, &digest_ctx, true);
        const uint64_t det_seed =
            tier2_compute_deterministic_sampling_seed(launch_uid, binary_sha256);
        tier2_put_u64_le(f, det_seed, &digest_ctx, true);
        tier2_put_u64_le(f, 0, &digest_ctx, true);  // reserved_zero
        budget_bytes_used += trunc_len;
        trailer_typed_reason_code = 1;
        out->next_sequence_id_not_emitted_value = marker_next_seq;
        out->marker_dropped_count = dropped;
        out->marker_deterministic_sampling_seed = det_seed;
        out->marker_typed_reason_code = 1;
        out->marker_emitted = true;
    }

    const uint64_t trailer_offset = records_start_offset + budget_bytes_used;
    const bool marker_emitted_flag = truncated_now && !budget_starves_marker;
    uint32_t flag_value = 0u;
    if (per_lane_records_written > 0) flag_value |= 0x1u;
    if (marker_emitted_flag)          flag_value |= 0x2u;

    static const uint8_t trailer_magic[4] = {'T', '2', 'A', 'T'};
    tier2_put_bytes(f, trailer_magic, 4, &digest_ctx, true);
    tier2_put_u16_le(f, 80, &digest_ctx, true);
    tier2_put_u16_le(f, 1, &digest_ctx, true);
    tier2_put_u64_le(f, launch_uid, &digest_ctx, true);
    tier2_put_u64_le(f, record_count_written, &digest_ctx, true);
    tier2_put_u64_le(f, budget_bytes_used, &digest_ctx, true);
    tier2_put_u64_le(f, budget_limit, &digest_ctx, true);
    tier2_put_u32_le(f, flag_value, &digest_ctx, true);
    tier2_put_u32_le(f, trailer_typed_reason_code, &digest_ctx, true);
    // Trailer last_sequence_id: only meaningful when a compact record was
    // emitted. Empty-record truncated frames carry 0 and the parser
    // disambiguates via the matching access_record_count==0 in the same
    // trailer. Tracking has_emitted_records explicitly keeps the writer's
    // intent honest.
    const uint64_t trailer_last_seq =
        has_emitted_records ? last_emitted_sequence_id : 0;
    tier2_put_u64_le(f, trailer_last_seq, &digest_ctx, true);
    tier2_put_zeros(f, 24, &digest_ctx, true);

    uint8_t digest[32];
    digest_ctx.Finalize(digest);
    auto patch_u32 = [&](uint64_t off, uint32_t v) {
        fseek(f, frame_start + off, SEEK_SET);
        uint8_t buf[4];
        for (int i = 0; i < 4; ++i) {
            buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
        }
        fwrite(buf, 1, 4, f);
    };
    auto patch_u64 = [&](uint64_t off, uint64_t v) {
        fseek(f, frame_start + off, SEEK_SET);
        uint8_t buf[8];
        for (int i = 0; i < 8; ++i) {
            buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
        }
        fwrite(buf, 1, 8, f);
    };
    patch_u32(12, flag_value);
    patch_u64(80, budget_bytes_used);
    patch_u64(88, budget_limit);
    patch_u64(96, record_count_written);
    patch_u64(104, trailer_offset);
    fseek(f, frame_start + 128, SEEK_SET);
    fwrite(digest, 1, 32, f);
    fseek(f, 0, SEEK_END);
    fflush(f);

    out->emitted = true;
    out->trace_bytes = budget_bytes_used;
    memcpy(out->digest, digest, 32);
    tier2::HexEncode32(digest, out->sha256_hex);
    out->record_count_written = record_count_written;
    out->per_lane_records_written = per_lane_records_written;
    out->trailer_last_sequence_id = trailer_last_seq;
    out->has_emitted_records = has_emitted_records;
}

#endif  // NVBIT_TRACER_METRIC_EXTRACTOR_TIER2_FRAME_SERIALIZER_H_
