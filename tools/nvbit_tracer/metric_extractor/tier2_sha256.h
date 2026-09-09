// nvbit_tracer/metric_extractor/tier2_sha256.h
//
// Minimal SHA-256 implementation used by the Tier-2 address-dump
// producer to compute the per-launch sidecar digest the schema places
// in ``LaunchHeaderV1.sidecar_sha256``. The producer streams the
// canonical bytes (every byte from the first byte after the 160-byte
// header through the final trailer byte, exclusive of the digest
// field itself) through ``Tier2Sha256Ctx`` while writing them to
// disk, then seek-patches the digest into the header placeholder.
//
// The implementation is header-only and dependency-free so it can be linked
// directly into the C++ host writer in ``metric_extractor.cu``.

#ifndef NVBIT_TRACER_METRIC_EXTRACTOR_TIER2_SHA256_H_
#define NVBIT_TRACER_METRIC_EXTRACTOR_TIER2_SHA256_H_

#include <stdint.h>
#include <string.h>

namespace tier2 {

inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// FIPS-180-4 round constants (first 32 bits of fractional parts of
// cube roots of the first 64 primes).
static const uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

// FIPS-180-4 initial hash values (first 32 bits of fractional parts
// of square roots of the first 8 primes).
static const uint32_t kSha256H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

class Sha256Ctx {
public:
    Sha256Ctx() {
        Reset();
    }

    void Reset() {
        for (int i = 0; i < 8; ++i) h_[i] = kSha256H0[i];
        bit_len_ = 0;
        buffer_used_ = 0;
    }

    void Update(const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        bit_len_ += static_cast<uint64_t>(len) * 8ULL;
        // If there are pending bytes, fill the buffer first.
        if (buffer_used_ > 0) {
            size_t need = 64 - buffer_used_;
            size_t take = (len < need) ? len : need;
            memcpy(buffer_ + buffer_used_, p, take);
            buffer_used_ += take;
            p += take;
            len -= take;
            if (buffer_used_ == 64) {
                Transform(buffer_);
                buffer_used_ = 0;
            }
        }
        // Consume whole blocks directly.
        while (len >= 64) {
            Transform(p);
            p += 64;
            len -= 64;
        }
        // Stash the tail.
        if (len > 0) {
            memcpy(buffer_ + buffer_used_, p, len);
            buffer_used_ += len;
        }
    }

    // Finalize the digest into a 32-byte output buffer. After Finalize,
    // Reset must be called before reusing the context.
    void Finalize(uint8_t out[32]) {
        // FIPS-180-4 padding: append a single 0x80 byte, then zero bytes
        // until the buffer length is 56 (mod 64), then the 64-bit
        // big-endian message length in bits.
        buffer_[buffer_used_++] = 0x80;
        if (buffer_used_ > 56) {
            while (buffer_used_ < 64) buffer_[buffer_used_++] = 0;
            Transform(buffer_);
            buffer_used_ = 0;
        }
        while (buffer_used_ < 56) buffer_[buffer_used_++] = 0;
        // Big-endian 64-bit bit-length.
        for (int i = 7; i >= 0; --i) {
            buffer_[buffer_used_++] =
                static_cast<uint8_t>((bit_len_ >> (i * 8)) & 0xffu);
        }
        Transform(buffer_);
        buffer_used_ = 0;
        // Emit the digest big-endian per FIPS-180-4.
        for (int i = 0; i < 8; ++i) {
            out[i * 4 + 0] = static_cast<uint8_t>((h_[i] >> 24) & 0xffu);
            out[i * 4 + 1] = static_cast<uint8_t>((h_[i] >> 16) & 0xffu);
            out[i * 4 + 2] = static_cast<uint8_t>((h_[i] >> 8) & 0xffu);
            out[i * 4 + 3] = static_cast<uint8_t>(h_[i] & 0xffu);
        }
    }

    // One-shot convenience: compute SHA-256 over a contiguous buffer.
    static void OneShot(const void* data, size_t len, uint8_t out[32]) {
        Sha256Ctx c;
        c.Update(data, len);
        c.Finalize(out);
    }

private:
    void Transform(const uint8_t* block) {
        uint32_t w[64];
        // Load big-endian 32-bit words from the block.
        for (int i = 0; i < 16; ++i) {
            w[i] =
                (static_cast<uint32_t>(block[i * 4 + 0]) << 24) |
                (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
                static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 =
                rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 =
                rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = h + S1 + ch + kSha256K[i] + w[i];
            uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }

    uint32_t h_[8];
    uint64_t bit_len_;
    uint8_t  buffer_[64];
    size_t   buffer_used_;
};

// Convenience: hex-encode a 32-byte digest as 64 lowercase chars
// (null-terminated). Caller must provide a 65-byte buffer.
inline void HexEncode32(const uint8_t digest[32], char out[65]) {
    static const char k_hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out[i * 2 + 0] = k_hex[(digest[i] >> 4) & 0xfu];
        out[i * 2 + 1] = k_hex[digest[i] & 0xfu];
    }
    out[64] = '\0';
}

}  // namespace tier2

#endif  // NVBIT_TRACER_METRIC_EXTRACTOR_TIER2_SHA256_H_
