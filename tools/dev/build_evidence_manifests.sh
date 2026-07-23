#!/usr/bin/env bash
# Build deterministic, per-file evidence manifests without modifying evidence.

set -euo pipefail

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
manifest_dir="$repo_root/evidence/manifests"
humanize_archive=${HUMANIZE_ARCHIVE_ROOT:-/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/.humanize}
round28="$humanize_archive/rlcr/2026-07-08_11-39-13"
external_capture=${SM86_REFERENCE_CAPTURE_ROOT:-/home/jakeshea/sm86_reference_capture_r27}

mkdir -p "$manifest_dir"

hash_tree() {
    local root=$1
    local output=$2
    local temporary

    if [[ ! -d "$root" ]]; then
        echo "ERROR: evidence root is unavailable: $root" >&2
        exit 2
    fi

    temporary=$(mktemp "$manifest_dir/.manifest.XXXXXX")
    (
        cd "$root"
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
    ) >"$temporary"
    mv "$temporary" "$manifest_dir/$output"
}

hash_tree "$humanize_archive" humanize.sha256
hash_tree "$repo_root/data/annex" data_annex.sha256
hash_tree "$repo_root/data/local_labels/campaign" campaign.sha256
hash_tree "$repo_root/tools/sol_execbench/data" sol_execbench_data.sha256
hash_tree "$repo_root/references" references.sha256
hash_tree "$repo_root/.toolchain" toolchain.sha256
hash_tree "$external_capture" external_capture.sha256

(
    cd "$repo_root/.toolchain"
    find . -type l -printf '%p\t%l\n' | LC_ALL=C sort
) >"$manifest_dir/toolchain.symlinks"

migration_manifest="$manifest_dir/humanize_migration_baseline.sha256"
frozen_manifest="$repo_root/evidence/MANIFEST.sha256"
temporary_migration=$(mktemp "$manifest_dir/.migration.XXXXXX")
temporary_frozen=$(mktemp "$manifest_dir/.frozen.XXXXXX")

emit_mapping() {
    local source=$1
    local destination=$2
    local frozen=${3:-true}
    local digest

    if [[ ! -f "$source" ]]; then
        echo "ERROR: migration source is unavailable: $source" >&2
        exit 2
    fi
    digest=$(sha256sum "$source" | cut -d ' ' -f 1)
    printf '%s  %s\n' "$digest" "$destination" >>"$temporary_migration"
    if [[ "$frozen" == true ]]; then
        printf '%s  %s\n' "$digest" "$destination" >>"$temporary_frozen"
    fi
}

emit_mapping \
    "$humanize_archive/bitlesson.md" \
    "docs/lessons.md" false
emit_mapping \
    "$humanize_archive/plans/sol_pattern_mapping.md" \
    "tools/sol_baselines_scaffold/sol_pattern_mapping.md" false

for name in \
    sol_definition_equivalence_audit_r28_v1_blocking.json \
    sol_definition_equivalence_audit_r28_precommit_h1.json \
    sol_definition_equivalence_audit_r28.json
do
    emit_mapping \
        "$round28/$name" \
        "evidence/round28/sol_equivalence/$name"
done

for source in "$round28"/adjudication_r28_*.md
do
    emit_mapping \
        "$source" \
        "evidence/round28/preparation_failures/problem-070/$(basename "$source")"
done

while IFS= read -r -d '' source
do
    relative=${source#"$round28/prep070_reconstruction/"}
    emit_mapping \
        "$source" \
        "evidence/round28/preparation_failures/problem-070/prep070_reconstruction/$relative"
done < <(find "$round28/prep070_reconstruction" -type f -print0 | LC_ALL=C sort -z)

emit_mapping \
    "$round28/session_log.jsonl" \
    "evidence/round28/preparation_failures/problem-070/session_log.jsonl"
emit_mapping \
    "$round28/fable5_prompt_r28_sol_capture_integrity_false_positive_and_h3_attempt.md" \
    "evidence/round28/pending/fable5_prompt_r28_sol_capture_integrity_false_positive_and_h3_attempt.md"

if [[ $(wc -l <"$temporary_migration") -ne 22 ]]; then
    echo "ERROR: expected 22 Humanize migration files" >&2
    exit 2
fi
if [[ $(wc -l <"$temporary_frozen") -ne 20 ]]; then
    echo "ERROR: expected 20 permanently frozen Round-28 files" >&2
    exit 2
fi

LC_ALL=C sort -k2,2 "$temporary_migration" >"$migration_manifest"
LC_ALL=C sort -k2,2 "$temporary_frozen" >"$frozen_manifest"
rm -f "$temporary_migration" "$temporary_frozen"

echo "Evidence manifests written under $manifest_dir"
