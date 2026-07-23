// nvbit_tracer/metric_extractor/test_tier2_sha256.cc
//
// Known-vector tests for the SHA-256 implementation in
// ``tier2_sha256.h``. Mirrors the existing standalone test convention
// (``test_classifier`` / ``test_saturating_counter``). Build via the
// Makefile target; run produces a single-line PASS / FAIL summary.
//
// Vectors:
//   * FIPS-180-4 Appendix B.1: empty string.
//   * FIPS-180-4 Appendix B.1: "abc".
//   * FIPS-180-4 Appendix B.3: 1,000,000 'a' bytes.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "tier2_sha256.h"

namespace {

void ToHex(const uint8_t digest[32], char out[65]) {
    tier2::HexEncode32(digest, out);
}

bool ExpectDigest(const char* label, const char* data, size_t len,
                  const char* expected_hex) {
    uint8_t digest[32];
    tier2::Sha256Ctx::OneShot(data, len, digest);
    char hex[65];
    ToHex(digest, hex);
    if (strcmp(hex, expected_hex) != 0) {
        fprintf(stderr,
                "FAIL %s: got %s expected %s\n", label, hex, expected_hex);
        return false;
    }
    return true;
}

bool TestEmpty() {
    return ExpectDigest(
        "empty", "", 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

bool TestAbc() {
    return ExpectDigest(
        "abc", "abc", 3,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

bool TestOneMillionA() {
    // FIPS-180-4 Appendix B.3 expects the SHA-256 of one million 'a' bytes
    // to be ``cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0``.
    // Feed via several Update calls to also exercise the buffer / boundary
    // path.
    const size_t total = 1000000;
    std::string chunk(1024, 'a');
    tier2::Sha256Ctx ctx;
    size_t remaining = total;
    while (remaining > 0) {
        size_t take = (remaining < chunk.size()) ? remaining : chunk.size();
        ctx.Update(chunk.data(), take);
        remaining -= take;
    }
    uint8_t digest[32];
    ctx.Finalize(digest);
    char hex[65];
    ToHex(digest, hex);
    const char* expected =
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    if (strcmp(hex, expected) != 0) {
        fprintf(stderr,
                "FAIL million_a: got %s expected %s\n", hex, expected);
        return false;
    }
    return true;
}

bool TestStreamingMatchesOneShot() {
    // Feeding the same input in arbitrary chunks must yield the same
    // digest as the one-shot path.
    const char* input =
        "The quick brown fox jumps over the lazy dog. ";
    const size_t input_len = strlen(input);
    std::string buf;
    for (int rep = 0; rep < 100; ++rep) buf.append(input, input_len);
    uint8_t one_shot[32];
    tier2::Sha256Ctx::OneShot(buf.data(), buf.size(), one_shot);

    tier2::Sha256Ctx ctx;
    size_t offset = 0;
    size_t step = 17;  // intentionally awkward chunk size
    while (offset < buf.size()) {
        size_t take = (offset + step <= buf.size()) ? step : buf.size() - offset;
        ctx.Update(buf.data() + offset, take);
        offset += take;
    }
    uint8_t streamed[32];
    ctx.Finalize(streamed);
    if (memcmp(one_shot, streamed, 32) != 0) {
        fprintf(stderr,
                "FAIL streaming_matches_one_shot: digests differ\n");
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= TestEmpty();
    ok &= TestAbc();
    ok &= TestOneMillionA();
    ok &= TestStreamingMatchesOneShot();
    if (ok) {
        fprintf(stdout, "test_tier2_sha256: PASS\n");
        return 0;
    }
    fprintf(stdout, "test_tier2_sha256: FAIL\n");
    return 1;
}
