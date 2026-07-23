// workloads/portable_lcd/runner_common.h
//
// Shared host-runner scaffolding for the portable_lcd suite. Every
// per-slug `main.cu` has the same boilerplate: CLI parsing, RNG
// helpers, the schema-v2 measurement-manifest emitter, gpu-sku /
// uuid / sha256 helpers, and `mkdir -p`. This header collects the
// boilerplate into a single place so per-slug runners only need to
// own slug-specific behavior (correctness reference, allocation,
// kernel launch, manifest field overrides).
//
// Keep this header header-only and inline-only — every consumer is a
// `.cu` translation unit compiled by `nvcc`, and we never want
// duplicate symbol errors from a single-file suite where each slug
// links one runner.
//
// ALL functions and types live in the `runner_common` namespace.
// Slug runners include this header and use `runner_common::*` to
// reach the helpers.

#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

namespace runner_common {

enum class Mode { Unset, Correctness, Timing };

struct Args {
    Mode        mode               = Mode::Unset;
    uint32_t    seed               = 42u;
    int         warmup             = 0;
    int         reps               = 1;
    const char *output_dir         = "results";
    bool        allow_any_device   = false;
    // Optional timing-mode shape overrides. Presence is tracked
    // via the four has_override_* booleans (set by parse_args
    // when the corresponding --BH/--Sq/--Sk/--D flag is
    // consumed). Value-positivity is NOT a proxy for presence:
    // an explicit ``--BH 0`` is a presence event that the
    // consuming runner MUST surface to its validation predicate
    // (the runner rejects the zero value because the kernels
    // require positive dimensions). Consuming runners MUST
    // honor the overrides only in Mode::Timing and reject them
    // in Mode::Correctness.
    int         override_bh        = -1;
    int         override_sq        = -1;
    int         override_sk        = -1;
    int         override_d         = -1;
    bool        has_override_bh    = false;
    bool        has_override_sq    = false;
    bool        has_override_sk    = false;
    bool        has_override_d     = false;
    // Optional timing-mode GEMM shape overrides. Same presence-
    // boolean discipline as the flash_attn overrides above: an
    // explicit ``--m 0`` is a presence event that the consuming
    // runner MUST surface to its validation predicate (zero is
    // rejected because the kernels require positive dimensions).
    // Consuming runners MUST honor the overrides only in
    // Mode::Timing and reject them in Mode::Correctness.
    int         override_m         = -1;
    int         override_n         = -1;
    int         override_k         = -1;
    bool        has_override_m     = false;
    bool        has_override_n     = false;
    bool        has_override_k     = false;
    // Optional generic shape-point identity. When the local-label
    // driver passes `--shape-point trace_l2_accuracy_xl` or
    // `--shape-point trace_l2_accuracy_lg`, the parser captures the
    // requested string here so the per-run main.cu can emit it
    // verbatim into the measurement manifest + stdout (instead of
    // the hard-coded "timing_00" that the manifest validator would
    // reject as a shape-identity mismatch). Stays nullptr when the
    // operator uses the legacy correctness_00 / timing_00 /
    // --correctness_shape / --timing_shape forms (manifest emission
    // falls back to the canonical "correctness_00" / "timing_00"
    // strings in that case).
    const char *shape_point        = nullptr;
};

// Per-runner opt-in for the dimension-override flag families.
// `parse_args` parses every flag indiscriminately (so unknown
// flags surface as parse errors instead of being silently
// dropped); each runner's `main` MUST call
// `require_accepted_overrides` immediately after `parse_args`
// to reject the flag sets it does not consume. Without this
// gate, a runner that hard-codes its kernel dimensions but
// silently ignores `--m / --n / --k` would emit a manifest at
// its hard-coded shape while claiming the operator-requested
// shape_point — exactly the label-poisoning failure mode the
// header-only opt-in pattern defends against.
struct AcceptedOverrides {
    // Runner consumes `--m / --n / --k` (GEMM-shape dimensions).
    bool gemm_mnk           = false;
    // Runner consumes `--BH / --Sq / --Sk / --D`
    // (flash_attn-shape dimensions).
    bool flash_attn_bhsqsk  = false;
    // Runner consumes ONLY `--n` (an activation/element-count override
    // that repurposes the GEMM `--n` flag as a 1-D count, e.g. swiglu_fwd);
    // `--m` / `--k` are NOT consumed and are rejected. This is narrower
    // than `gemm_mnk`: opting into the full GEMM family for an `--n`-only
    // runner would silently accept-and-ignore `--m` / `--k`, letting a run
    // claim a generic shape while launching the `--n`-only count.
    bool activation_n       = false;
};

inline void require_accepted_overrides(
    const Args &a,
    const char *prog,
    const AcceptedOverrides &accepted
) {
    if (!accepted.gemm_mnk) {
        if (accepted.activation_n) {
            // The runner consumes ONLY `--n` (a 1-D element count); `--m`
            // and `--k` are not consumed, so reject them rather than
            // silently ignoring them (an accepted-but-ignored `--m`/`--k`,
            // including `--m 0`, would let a run claim a generic shape while
            // launching the `--n`-only count).
            if (a.has_override_m || a.has_override_k) {
                fprintf(stderr,
                    "%s: --m / --k overrides are not supported by this "
                    "runner; it consumes only the --n element-count "
                    "override\n", prog);
                exit(2);
            }
        } else if (a.has_override_m || a.has_override_n || a.has_override_k) {
            fprintf(stderr,
                "%s: --m / --n / --k overrides are not supported by "
                "this runner; the runner does not consume the "
                "GEMM dimension override surface\n", prog);
            exit(2);
        }
    }
    if (!accepted.flash_attn_bhsqsk) {
        if (a.has_override_bh || a.has_override_sq
            || a.has_override_sk || a.has_override_d) {
            fprintf(stderr,
                "%s: --BH / --Sq / --Sk / --D overrides are not "
                "supported by this runner; the runner does not "
                "consume the flash_attn dimension override surface\n",
                prog);
            exit(2);
        }
    }
}

inline int parse_int_or_die(const char *prog, const char *flag, const char *val) {
    char *end = nullptr;
    long v = strtol(val, &end, 10);
    if (end == val || *end != '\0') {
        fprintf(stderr, "%s: %s requires an integer argument; got %s\n", prog, flag, val);
        exit(2);
    }
    if (v < 0 || v > 1000000000) {
        fprintf(stderr, "%s: %s out of range\n", prog, flag);
        exit(2);
    }
    return (int)v;
}

inline void usage(FILE *f, const char *prog) {
    fprintf(f,
        "usage: %s \\\n"
        "  [--correctness_shape | --timing_shape] \\\n"
        "  [--seed N] [--warmup N] [--reps N] \\\n"
        "  [--output_dir PATH] [--allow-any-device] \\\n"
        "  [--BH N] [--Sq N] [--Sk N] [--D N]\n"
        "  [--m N] [--n N] [--k N]\n"
        "\n"
        "Timing-mode shape overrides (--BH / --Sq / --Sk / --D for\n"
        "flash_attn-shaped runners; --m / --n / --k for GEMM-shaped\n"
        "runners) are consumed only when the per-slug runner declares\n"
        "it accepts them; correctness mode rejects every override.\n", prog);
}

inline Args parse_args(int argc, char **argv, const char *prog) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char *s = argv[i];
        if      (!strcmp(s, "--correctness_shape"))   a.mode = Mode::Correctness;
        else if (!strcmp(s, "--timing_shape"))        a.mode = Mode::Timing;
        // R40 generic --shape-point alias: the canonical local-label
        // driver passes --shape-point <id> for non-{correctness,timing}
        // shape_points (the generic-shape branch in label_driver.sh).
        // The portable_lcd binaries map known generic shape_points to
        // the canonical Mode they share kernel-launch semantics with.
        // Currently the only such generic shape is trace_l2_accuracy_xl,
        // which the GEMM / activation binaries treat as a Mode::Timing
        // variant whose actual axes come from --m / --n / --k overrides.
        // Unknown --shape-point values are NOT silently accepted; the
        // runner exits with `unknown_shape_point_arg` so a future generic
        // shape addition is forced through the runner_common.h surface.
        else if (!strcmp(s, "--shape-point") && i+1 < argc) {
            const char *sp = argv[++i];
            if      (!strcmp(sp, "correctness_00") || !strcmp(sp, "correctness_shape")) a.mode = Mode::Correctness;
            else if (!strcmp(sp, "timing_00") || !strcmp(sp, "timing_shape"))           a.mode = Mode::Timing;
            else if (!strcmp(sp, "trace_l2_accuracy_xl"))                               a.mode = Mode::Timing;
            else if (!strcmp(sp, "trace_l2_accuracy_lg"))                               a.mode = Mode::Timing;
            else {
                fprintf(stderr, "%s: unknown_shape_point_arg %s\n", prog, sp);
                exit(2);
            }
            // Capture the requested shape-point identity ONLY for
            // generic shape names. Legacy correctness_00 / timing_00
            // (and their _shape aliases) keep a.shape_point == nullptr
            // so per-run main.cu fall through to the canonical
            // "correctness_00" / "timing_00" literals (preserves
            // byte-equality with existing committed manifests).
            if (!strcmp(sp, "trace_l2_accuracy_xl") ||
                !strcmp(sp, "trace_l2_accuracy_lg")) {
                a.shape_point = sp;
            }
        }
        else if (!strcmp(s, "--seed") && i+1 < argc)  a.seed   = (uint32_t)parse_int_or_die(prog, s, argv[++i]);
        else if (!strcmp(s, "--warmup") && i+1 < argc) a.warmup = parse_int_or_die(prog, s, argv[++i]);
        else if (!strcmp(s, "--reps") && i+1 < argc)  a.reps   = parse_int_or_die(prog, s, argv[++i]);
        else if (!strcmp(s, "--output_dir") && i+1 < argc) a.output_dir = argv[++i];
        else if (!strcmp(s, "--allow-any-device"))    a.allow_any_device = true;
        else if (!strcmp(s, "--BH") && i+1 < argc) {
            a.override_bh = parse_int_or_die(prog, s, argv[++i]);
            a.has_override_bh = true;
        }
        else if (!strcmp(s, "--Sq") && i+1 < argc) {
            a.override_sq = parse_int_or_die(prog, s, argv[++i]);
            a.has_override_sq = true;
        }
        else if (!strcmp(s, "--Sk") && i+1 < argc) {
            a.override_sk = parse_int_or_die(prog, s, argv[++i]);
            a.has_override_sk = true;
        }
        else if (!strcmp(s, "--D") && i+1 < argc) {
            a.override_d  = parse_int_or_die(prog, s, argv[++i]);
            a.has_override_d = true;
        }
        else if (!strcmp(s, "--m") && i+1 < argc) {
            a.override_m  = parse_int_or_die(prog, s, argv[++i]);
            a.has_override_m = true;
        }
        else if (!strcmp(s, "--n") && i+1 < argc) {
            a.override_n  = parse_int_or_die(prog, s, argv[++i]);
            a.has_override_n = true;
        }
        else if (!strcmp(s, "--k") && i+1 < argc) {
            a.override_k  = parse_int_or_die(prog, s, argv[++i]);
            a.has_override_k = true;
        }
        else if (!strcmp(s, "-h") || !strcmp(s, "--help")) { usage(stdout, prog); exit(0); }
        else { fprintf(stderr, "%s: unknown flag %s\n", prog, s); usage(stderr, prog); exit(2); }
    }
    if (a.mode == Mode::Unset) a.mode = Mode::Correctness;
    if (a.reps < 1) {
        fprintf(stderr, "%s: repeat_iters_must_be_positive\n", prog);
        exit(2);
    }
    return a;
}

inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }

inline void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}

inline void mkdir_p(const char *prog, const char *path) {
    char buf[1024]; size_t n = strlen(path);
    if (n >= sizeof(buf)) { fprintf(stderr, "%s: output_dir too long\n", prog); exit(2); }
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i < n; ++i) {
        if (buf[i] == '/') { buf[i] = '\0'; mkdir(buf, 0755); buf[i] = '/'; }
    }
    mkdir(buf, 0755);
}

inline void compute_self_sha256(char out[65]) {
    char cmd[256]; snprintf(cmd, sizeof(cmd), "sha256sum /proc/%d/exe 2>/dev/null", (int)getpid());
    FILE *p = popen(cmd, "r");
    if (!p) { strcpy(out, "0000000000000000000000000000000000000000000000000000000000000000"); return; }
    if (fscanf(p, "%64s", out) != 1) {
        strcpy(out, "0000000000000000000000000000000000000000000000000000000000000000");
    }
    pclose(p);
}

inline void infer_gpu_sku(const cudaDeviceProp &dp, char out[64]) {
    if (dp.major == 8 && dp.minor == 6) snprintf(out, 64, "rtx_3090_ti");
    else if (dp.major == 9 && dp.minor == 0) snprintf(out, 64, "h100");
    else if (dp.major == 12 && dp.minor == 0) {
        // CC 12.0 covers multiple Blackwell SKUs; disambiguate by
        // device name and fall back to a generic tag so future
        // variants do not silently masquerade as a specific SKU.
        if (strstr(dp.name, "5060 Laptop") != nullptr) {
            snprintf(out, 64, "rtx_5060_laptop");
        } else if (strstr(dp.name, "RTX PRO 6000") != nullptr ||
                   strstr(dp.name, "RTX_PRO_6000") != nullptr) {
            snprintf(out, 64, "rtx_pro_6000_blackwell");
        } else {
            snprintf(out, 64, "cc_12_0_unknown");
        }
    }
    else snprintf(out, 64, "cc_%d_%d", dp.major, dp.minor);
}

inline void uuid_to_hex(const cudaUUID_t &u, char out[33]) {
    static const char *hex = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) {
        unsigned char b = (unsigned char)u.bytes[i];
        out[i*2] = hex[b >> 4]; out[i*2 + 1] = hex[b & 0xf];
    }
    out[32] = '\0';
}

// Acquire device properties + validate target arch. Returns 0 on
// success, or a non-zero exit code the runner should propagate.
inline int acquire_device(const char *prog, const Args &a, cudaDeviceProp &dp) {
    int dev = 0;
    cudaError_t rc = cudaGetDevice(&dev);
    if (rc != cudaSuccess) {
        fprintf(stderr, "%s: cudaGetDevice failed (%s)\n", prog, cudaGetErrorString(rc));
        return 1;
    }
    rc = cudaGetDeviceProperties(&dp, dev);
    if (rc != cudaSuccess) {
        fprintf(stderr, "%s: cudaGetDeviceProperties failed (%s)\n", prog, cudaGetErrorString(rc));
        return 1;
    }
    if (!a.allow_any_device && dp.major == 0) {
        fprintf(stderr, "%s: unknown_target_device (compute_capability_major == 0)\n", prog);
        return 1;
    }
    return 0;
}

// Run a timed-rep loop with two selectable cudaEvent measurement
// shapes, gated by the `SIMBENCH_BRACKET_REPS` environment variable.
//
// `SIMBENCH_BRACKET_REPS` unset / empty / "0": legacy per-rep
// brackets — one cudaEventRecord pair per launch, with
// `samples_ms[r]` carrying the per-rep value. This is the original
// cudaEventElapsedTime shape every per-slug runner used by default.
//
// `SIMBENCH_BRACKET_REPS` set to anything else: bracket-all-reps
// mode — a single outer cudaEventRecord pair around the whole loop;
// `samples_ms[r]` carries the per-rep average so the runner's
// mean_ms / cv_pct reporting stays consistent.
template <typename LaunchCallable>
inline void run_timed_reps(
    const Args &a, double *samples_ms, LaunchCallable &&launch)
{
    cudaEvent_t evt_start, evt_stop;
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_stop);

    const char *bracket = std::getenv("SIMBENCH_BRACKET_REPS");
    bool bracket_all = (bracket && bracket[0] != '\0' && bracket[0] != '0');

    if (bracket_all) {
        cudaEventRecord(evt_start, 0);
        for (int r = 0; r < a.reps; ++r) {
            launch();
        }
        cudaEventRecord(evt_stop, 0);
        cudaEventSynchronize(evt_stop);
        float total_ms = 0.f;
        cudaEventElapsedTime(&total_ms, evt_start, evt_stop);
        double per_rep = (a.reps > 0)
            ? (double)total_ms / (double)a.reps
            : (double)total_ms;
        for (int r = 0; r < a.reps; ++r) samples_ms[r] = per_rep;
    } else {
        for (int r = 0; r < a.reps; ++r) {
            cudaEventRecord(evt_start, 0);
            launch();
            cudaEventRecord(evt_stop, 0);
            cudaEventSynchronize(evt_stop);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, evt_start, evt_stop);
            samples_ms[r] = (double)ms;
        }
    }
    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_stop);
}

// Aggregate per-rep timing samples into mean_ms + cv_pct.
inline void summarize_reps(const double *samples_ms, int reps, double *mean_ms_out, double *cv_pct_out) {
    double mean_ms = 0.0, cv_pct = 0.0;
    if (reps > 0) {
        for (int r = 0; r < reps; ++r) mean_ms += samples_ms[r];
        mean_ms /= reps;
        if (reps > 1 && mean_ms > 0.0) {
            double var = 0.0;
            for (int r = 0; r < reps; ++r) { double d = samples_ms[r] - mean_ms; var += d * d; }
            var /= (reps - 1);
            cv_pct = 100.0 * sqrt(var) / mean_ms;
        }
    }
    *mean_ms_out = mean_ms;
    *cv_pct_out  = cv_pct;
}

// Schema-v2 measurement manifest emission. Writes to
// <output_dir>/<slug>/manifest.json and returns the path via
// `manifest_path_out` (caller-provided buffer >= 1280 bytes).
inline void emit_manifest_v2(
    const char *prog, const char *slug, const Args &a,
    const cudaDeviceProp &dp, const char *mode_str, const char *shape_point,
    double cv_pct, char *manifest_path_out)
{
    int rt_ver = 0, drv_ver = 0;
    cudaRuntimeGetVersion(&rt_ver);
    cudaDriverGetVersion(&drv_ver);

    char gpu_sku[64];   infer_gpu_sku(dp, gpu_sku);
    char gpu_uuid[33];  uuid_to_hex(dp.uuid, gpu_uuid);
    char self_sha[65];  compute_self_sha256(self_sha);
    char timestamp[40];
    {
        time_t t = time(nullptr);
        struct tm tm_utc; gmtime_r(&t, &tm_utc);
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    }

    char dir_full[1024];
    snprintf(dir_full, sizeof(dir_full), "%s/%s", a.output_dir, slug);
    mkdir_p(prog, dir_full);
    snprintf(manifest_path_out, 1280, "%s/manifest.json", dir_full);

    FILE *mf = fopen(manifest_path_out, "w");
    if (!mf) { fprintf(stderr, "%s: cannot open %s for write\n", prog, manifest_path_out); exit(1); }
    fprintf(mf, "{\n");
    fprintf(mf, "  \"schema_version\": 2,\n");
    fprintf(mf, "  \"slug\": \"%s\",\n", slug);
    fprintf(mf, "  \"mode\": \"%s\",\n", mode_str);
    fprintf(mf, "  \"warm_up_iters\": %d,\n", a.warmup);
    fprintf(mf, "  \"repeat_iters\": %d,\n", a.reps);
    fprintf(mf, "  \"cv_pct\": %.6f,\n", cv_pct);
    fprintf(mf, "  \"cuda_version\": \"%d.%d\",\n", rt_ver / 1000, (rt_ver % 1000) / 10);
    fprintf(mf, "  \"driver_version\": \"%d.%d\",\n", drv_ver / 1000, (drv_ver % 1000) / 10);
    fprintf(mf, "  \"gpu_sku\": \"%s\",\n", gpu_sku);
    fprintf(mf, "  \"gpu_uuid\": \"%s\",\n", gpu_uuid);
    fprintf(mf, "  \"sm_count\": %d,\n", dp.multiProcessorCount);
    fprintf(mf, "  \"total_global_mem_bytes\": %zu,\n", (size_t)dp.totalGlobalMem);
    fprintf(mf, "  \"device_name\": \"%s\",\n", dp.name);
    fprintf(mf, "  \"compute_capability_major\": %d,\n", dp.major);
    fprintf(mf, "  \"compute_capability_minor\": %d,\n", dp.minor);
    fprintf(mf, "  \"binary_sha256\": \"%s\",\n", self_sha);
    fprintf(mf, "  \"compiler_flags\": [\"-std=c++17\", \"-Xptxas=-v\", \"-lineinfo\","
                " \"-gencode=arch=compute_80,code=sm_86\","
                " \"-gencode=arch=compute_80,code=sm_90\","
                " \"-gencode=arch=compute_80,code=sm_120\","
                " \"-gencode=arch=compute_80,code=compute_80\"],\n");
    fprintf(mf, "  \"kb_capability_card\": \"references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md\",\n");
    fprintf(mf, "  \"timestamp_utc\": \"%s\",\n", timestamp);
    fprintf(mf, "  \"shape_point\": \"%s\"\n", shape_point);
    fprintf(mf, "}\n");
    fclose(mf);
}

} // namespace runner_common
