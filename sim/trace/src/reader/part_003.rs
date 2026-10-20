// Lexical implementation section 4/5; included by the parent facade.
#[cfg(feature = "campaign")]
fn validate_subset_producer_contract(
    root: &serde_json::Map<String, serde_json::Value>,
    original_grid: (u32, u32, u32),
    original_block: (u32, u32, u32),
    captured_block: (u32, u32, u32),
    selected_segments: &BTreeSet<u64>,
    certainty_segments: &BTreeSet<u64>,
    segment_blocks: &BTreeMap<u64, BTreeSet<u64>>,
) -> Result<ValidatedPerSmSlotMap, String> {
    let reason = string(required(root, "reason", "sampled subset")?, "reason")?;
    if ![
        "byte_budget",
        "walltime_budget",
        "gate_forced",
        "not_breaching",
    ]
    .contains(&reason)
    {
        return Err("reason is not a schema-v2 value".to_string());
    }
    if original_block != captured_block {
        return Err(
            "shape.captured_block must equal shape.original_block for a subset".to_string(),
        );
    }
    let shape = object(required(root, "shape", "sampled subset")?, "shape")?;
    if !shape
        .get("reduced_shape_variant")
        .is_some_and(serde_json::Value::is_null)
    {
        return Err("sampled subset requires reduced_shape_variant=null".to_string());
    }

    let gates = object(required(root, "gates", "sampled subset")?, "gates")?;
    const GATE_FIELDS: &[&str] = &[
        "coop",
        "wave_count",
        "n_measured_waves",
        "tail_measured_waves",
        "atomic_density",
        "hotline_share",
        "skew_top1pct_share",
        "gini",
        "per_sm_block_count_cv",
        "n_phases",
        "skew_heavy_block_count",
        "certainty_bytes",
        "max_trace_bytes",
        "certainty_budget_exceeded",
        "verdict",
    ];
    exact_keys(gates, GATE_FIELDS, GATE_FIELDS, "gates")?;
    if boolean(required(gates, "coop", "gates")?, "gates.coop")? {
        return Err("a cooperative launch cannot sanction a sampled subset".to_string());
    }
    let gate_wave_count = unsigned(required(gates, "wave_count", "gates")?, "gates.wave_count")?;
    let gate_measured = unsigned(
        required(gates, "n_measured_waves", "gates")?,
        "gates.n_measured_waves",
    )?;
    let gate_tail_measured = unsigned(
        required(gates, "tail_measured_waves", "gates")?,
        "gates.tail_measured_waves",
    )?;
    for field in [
        "atomic_density",
        "hotline_share",
        "skew_top1pct_share",
        "gini",
        "per_sm_block_count_cv",
    ] {
        nonnegative_number(required(gates, field, "gates")?, &format!("gates.{field}"))?;
    }
    positive(required(gates, "n_phases", "gates")?, "gates.n_phases")?;
    let skew_heavy_count = unsigned(
        required(gates, "skew_heavy_block_count", "gates")?,
        "gates.skew_heavy_block_count",
    )?;
    let gate_certainty_bytes = unsigned(
        required(gates, "certainty_bytes", "gates")?,
        "gates.certainty_bytes",
    )?;
    let gate_max_bytes = positive(
        required(gates, "max_trace_bytes", "gates")?,
        "gates.max_trace_bytes",
    )?;
    if boolean(
        required(gates, "certainty_budget_exceeded", "gates")?,
        "gates.certainty_budget_exceeded",
    )? {
        return Err("a certainty-budget-exceeded gate cannot sanction a subset".to_string());
    }
    if string(required(gates, "verdict", "gates")?, "gates.verdict")? != "subset_admissible" {
        return Err("a sampled subset requires gates.verdict=subset_admissible".to_string());
    }

    let tracer = object(required(root, "tracer", "sampled subset")?, "tracer")?;
    exact_keys(
        tracer,
        &["version", "features", "stats_pass_version"],
        &["version", "features", "stats_pass_version"],
        "tracer",
    )?;
    if unsigned(required(tracer, "version", "tracer")?, "tracer.version")? != 5
        || unsigned(
            required(tracer, "stats_pass_version", "tracer")?,
            "tracer.stats_pass_version",
        )? != 2
    {
        return Err("sampled subset requires tracer v5 and stats pass v2".to_string());
    }
    let features = object(required(tracer, "features", "tracer")?, "tracer.features")?;
    exact_keys(
        features,
        &["atomic_addr", "ldgsts_addr"],
        &["atomic_addr", "ldgsts_addr"],
        "tracer.features",
    )?;
    if !boolean(
        required(features, "atomic_addr", "tracer.features")?,
        "tracer.features.atomic_addr",
    )? || !boolean(
        required(features, "ldgsts_addr", "tracer.features")?,
        "tracer.features.ldgsts_addr",
    )? {
        return Err("sampled subset requires atomic_addr and ldgsts_addr features".to_string());
    }

    let rng = object(required(root, "rng", "sampled subset")?, "rng")?;
    exact_keys(rng, &["algo", "seed"], &["algo", "seed"], "rng")?;
    if string(required(rng, "algo", "rng")?, "rng.algo")? != "pcg64"
        || required(rng, "seed", "rng")?.as_i64().is_none()
    {
        return Err("sampled subset requires pcg64 with an integer seed".to_string());
    }
    if string(
        required(root, "label_pairing", "sampled subset")?,
        "label_pairing",
    )? != "full_shape_label"
    {
        return Err("sampled subset requires full_shape_label pairing".to_string());
    }

    let wave = object(
        required(root, "wave_model", "sampled subset")?,
        "wave_model",
    )?;
    const WAVE_FIELDS: &[&str] = &[
        "reconstruction",
        "per_sm_resident_blocks",
        "n_waves",
        "segment_len",
        "ghost_depth",
    ];
    exact_keys(wave, WAVE_FIELDS, WAVE_FIELDS, "wave_model")?;
    if string(
        required(wave, "reconstruction", "wave_model")?,
        "wave_model.reconstruction",
    )? != "launch_index"
    {
        return Err("wave_model.reconstruction must be launch_index".to_string());
    }
    let declared_n_waves = unsigned(
        required(wave, "n_waves", "wave_model")?,
        "wave_model.n_waves",
    )?;
    let segment_len = positive(
        required(wave, "segment_len", "wave_model")?,
        "wave_model.segment_len",
    )?;
    let ghost_depth = unsigned(
        required(wave, "ghost_depth", "wave_model")?,
        "wave_model.ghost_depth",
    )?;
    if segment_len != 3 || ghost_depth != 1 {
        return Err(
            "current schema-v2 producer requires segment_len=3 and ghost_depth=1".to_string(),
        );
    }

    let residency_values = required(wave, "per_sm_resident_blocks", "wave_model")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "wave_model.per_sm_resident_blocks must be a nonempty array".to_string())?;
    let mut resident_by_sm = BTreeMap::new();
    for (index, value) in residency_values.iter().enumerate() {
        let context = format!("wave_model.per_sm_resident_blocks[{index}]");
        let row = object(value, &context)?;
        exact_keys(
            row,
            &["sm_id", "resident_blocks"],
            &["sm_id", "resident_blocks"],
            &context,
        )?;
        let sm = unsigned(
            required(row, "sm_id", &context)?,
            &format!("{context}.sm_id"),
        )?;
        let resident = positive(
            required(row, "resident_blocks", &context)?,
            &format!("{context}.resident_blocks"),
        )?;
        if resident_by_sm.insert(sm, resident).is_some() {
            return Err(format!("duplicate resident-block entry for sm {sm}"));
        }
    }

    let slot_values = required(root, "per_sm_slot_map", "sampled subset")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "per_sm_slot_map must be a nonempty array".to_string())?;
    let mut roster_by_sm: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut roster = BTreeSet::new();
    for (index, value) in slot_values.iter().enumerate() {
        let context = format!("per_sm_slot_map[{index}]");
        let row = object(value, &context)?;
        exact_keys(
            row,
            &["sm_id", "block_ids"],
            &["sm_id", "block_ids"],
            &context,
        )?;
        let sm = unsigned(
            required(row, "sm_id", &context)?,
            &format!("{context}.sm_id"),
        )?;
        let values = required(row, "block_ids", &context)?
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| format!("{context}.block_ids must be a nonempty array"))?;
        let mut block_ids = Vec::with_capacity(values.len());
        for value in values {
            let block = unsigned(value, &format!("{context}.block_ids entry"))?;
            if block > i64::MAX as u64 {
                return Err(format!(
                    "sampled block id {block} exceeds the replay index range"
                ));
            }
            if block_ids.last().is_some_and(|previous| *previous >= block) {
                return Err(format!(
                    "{context}.block_ids must be strictly launch-index sorted"
                ));
            }
            if !roster.insert(block) {
                return Err(format!(
                    "block {block} occurs more than once in per_sm_slot_map"
                ));
            }
            block_ids.push(block);
        }
        if roster_by_sm.insert(sm, block_ids).is_some() {
            return Err(format!("duplicate per_sm_slot_map entry for sm {sm}"));
        }
    }
    if resident_by_sm.keys().ne(roster_by_sm.keys()) {
        return Err("resident-block SM keys must exactly equal per_sm_slot_map keys".to_string());
    }
    let grid_total = u64::from(original_grid.0)
        .checked_mul(u64::from(original_grid.1))
        .and_then(|value| value.checked_mul(u64::from(original_grid.2)))
        .ok_or_else(|| "original grid size overflows u64".to_string())?;
    if roster.len() as u64 != grid_total || roster.iter().copied().ne(0..grid_total) {
        return Err("per_sm_slot_map must exhaust launch indices 0..original_grid".to_string());
    }

    let reconstructed_n_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| {
            let resident = resident_by_sm[sm];
            (blocks.len() as u64).div_ceil(resident)
        })
        .max()
        .unwrap_or(0);
    if reconstructed_n_waves != declared_n_waves || gate_wave_count != declared_n_waves {
        return Err("wave count disagrees with reconstructed per-SM cohorts".to_string());
    }
    let complete_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| blocks.len() as u64 / resident_by_sm[sm])
        .min()
        .unwrap_or(0);
    let complete_cutoff = (complete_waves / segment_len) * segment_len;
    let residual_tail = reconstructed_n_waves > complete_cutoff;
    let tail_wave_count = if residual_tail {
        reconstructed_n_waves - complete_cutoff
    } else {
        segment_len.min(complete_cutoff)
    };
    // Deliberate semantic clamp: the Python producer defines ghost removal as
    // max(0, waves - ghosts), so the saturating subtractions here are contract
    // arithmetic rather than overflow recovery. The ghost product itself must
    // still be checked.
    let expected_tail_measured = tail_wave_count.saturating_sub(ghost_depth);
    let segment_count = complete_cutoff / segment_len + u64::from(residual_tail);
    let ghost_waves = ghost_depth
        .checked_mul(segment_count)
        .ok_or_else(|| "ghost-wave accounting overflows u64".to_string())?;
    let expected_measured = reconstructed_n_waves.saturating_sub(ghost_waves);
    if gate_tail_measured != expected_tail_measured || gate_measured != expected_measured {
        return Err("measured-wave gates disagree with reconstructed ghost accounting".to_string());
    }

    let mut expected_segments: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (sm, blocks) in &roster_by_sm {
        let resident = resident_by_sm[sm];
        for (sequence, block) in blocks.iter().copied().enumerate() {
            let wave_index = sequence as u64 / resident;
            let segment_id = if wave_index < complete_cutoff {
                wave_index / segment_len
            } else {
                complete_cutoff / segment_len
            };
            expected_segments
                .entry(segment_id)
                .or_default()
                .insert(block);
        }
    }
    for (segment_id, blocks) in segment_blocks {
        if expected_segments.get(segment_id) != Some(blocks) {
            return Err(format!(
                "segment {segment_id} is not a complete reconstructed wave cohort"
            ));
        }
    }
    let cold_segment = *expected_segments
        .keys()
        .next()
        .ok_or_else(|| "wave reconstruction produced no segments".to_string())?;
    let candidate_tail = complete_cutoff / segment_len;
    let tail_segment = if expected_segments.contains_key(&candidate_tail) {
        candidate_tail
    } else {
        *expected_segments
            .keys()
            .next_back()
            .expect("nonempty above")
    };
    if !certainty_segments.contains(&cold_segment) || !certainty_segments.contains(&tail_segment) {
        return Err(
            "certainty_segments must contain reconstructed cold and tail cohorts".to_string(),
        );
    }

    let force_blocks = parse_unique_ids(
        required(root, "force_include_blocks", "sampled subset")?,
        "force_include_blocks",
        true,
    )?;
    if force_blocks.len() as u64 != skew_heavy_count {
        return Err("skew_heavy_block_count disagrees with force_include_blocks".to_string());
    }
    let certainty_blocks: BTreeSet<u64> = certainty_segments
        .iter()
        .flat_map(|segment| segment_blocks[segment].iter().copied())
        .collect();
    if !force_blocks.is_subset(&certainty_blocks) {
        return Err("force_include_blocks must be contained in certainty segments".to_string());
    }

    let strata_values = required(root, "strata", "sampled subset")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "strata must be a nonempty array".to_string())?;
    let mut stratum_ids = BTreeSet::new();
    let mut stratum_keys = BTreeSet::new();
    let mut sampled_ordinary = BTreeSet::new();
    const STRATUM_FIELDS: &[&str] = &[
        "id",
        "phase_id",
        "launch_tercile",
        "segment_ids",
        "N_h",
        "n_h",
        "pi",
        "sector_stddev",
        "neyman_weight",
    ];
    for (index, value) in strata_values.iter().enumerate() {
        let context = format!("strata[{index}]");
        let stratum = object(value, &context)?;
        exact_keys(stratum, STRATUM_FIELDS, STRATUM_FIELDS, &context)?;
        let id = unsigned(required(stratum, "id", &context)?, &format!("{context}.id"))?;
        let phase = unsigned(
            required(stratum, "phase_id", &context)?,
            &format!("{context}.phase_id"),
        )?;
        let tercile = unsigned(
            required(stratum, "launch_tercile", &context)?,
            &format!("{context}.launch_tercile"),
        )?;
        if tercile > 2 {
            return Err(format!("{context} id does not encode phase/launch tercile"));
        }
        let encoded_id = phase
            .checked_mul(3)
            .and_then(|value| value.checked_add(tercile))
            .ok_or_else(|| format!("{context} phase/launch tercile encoding overflows u64"))?;
        if id != encoded_id {
            return Err(format!("{context} id does not encode phase/launch tercile"));
        }
        if !stratum_ids.insert(id) || !stratum_keys.insert((phase, tercile)) {
            return Err(format!("{context} duplicates a stratum id or key"));
        }
        let ids = parse_unique_ids(
            required(stratum, "segment_ids", &context)?,
            &format!("{context}.segment_ids"),
            false,
        )?;
        if ids.len() < 2 {
            return Err(format!(
                "{context}.segment_ids must contain at least two ids"
            ));
        }
        let population = positive(
            required(stratum, "N_h", &context)?,
            &format!("{context}.N_h"),
        )?;
        let sample = positive(
            required(stratum, "n_h", &context)?,
            &format!("{context}.n_h"),
        )?;
        let ids_len = u64::try_from(ids.len())
            .map_err(|_| format!("{context}.segment_ids cardinality exceeds u64"))?;
        if sample < 2 || sample > population || !sampling_cardinalities_match(sample, ids_len) {
            return Err(format!("{context} has inconsistent N_h/n_h/segment_ids"));
        }
        let pi = nonnegative_number(required(stratum, "pi", &context)?, &format!("{context}.pi"))?;
        let stddev = nonnegative_number(
            required(stratum, "sector_stddev", &context)?,
            &format!("{context}.sector_stddev"),
        )?;
        let weight = nonnegative_number(
            required(stratum, "neyman_weight", &context)?,
            &format!("{context}.neyman_weight"),
        )?;
        if pi <= 0.0 || pi > 1.0 {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        let population_f64 = sampling_u64_as_f64(population);
        let expected_pi = sampling_u64_as_f64(sample) / population_f64;
        // An absolute floor would admit arbitrarily wrong probabilities as N_h
        // grows. Keep only a relative allowance for cross-runtime float spelling.
        if !relative_close_enough(pi, expected_pi) {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        let expected_weight = population_f64 * stddev;
        if !expected_weight.is_finite() {
            return Err(format!("{context} derived neyman_weight is not finite"));
        }
        if !close_enough(weight, expected_weight) {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        if ids.iter().any(|id| !sampled_ordinary.insert(*id)) {
            return Err("ordinary segment occurs in multiple strata".to_string());
        }
    }
    if !sampled_ordinary.is_disjoint(certainty_segments) {
        return Err("certainty segments cannot also be sampled ordinary segments".to_string());
    }
    let selected_partition: BTreeSet<u64> = sampled_ordinary
        .union(certainty_segments)
        .copied()
        .collect();
    if &selected_partition != selected_segments {
        return Err(
            "strata sampled ids plus certainty ids must partition selected ids".to_string(),
        );
    }

    let budget = object(
        required(root, "selection_budget", "sampled subset")?,
        "selection_budget",
    )?;
    const BUDGET_FIELDS: &[&str] = &[
        "max_trace_bytes",
        "basis",
        "certainty_fraction_numerator",
        "certainty_fraction_denominator",
        "certainty_bytes",
    ];
    exact_keys(budget, BUDGET_FIELDS, BUDGET_FIELDS, "selection_budget")?;
    let max_bytes = positive(
        required(budget, "max_trace_bytes", "selection_budget")?,
        "selection_budget.max_trace_bytes",
    )?;
    let certainty_bytes = unsigned(
        required(budget, "certainty_bytes", "selection_budget")?,
        "selection_budget.certainty_bytes",
    )?;
    if string(
        required(budget, "basis", "selection_budget")?,
        "selection_budget.basis",
    )? != "raw_threadblock_pb"
        || unsigned(
            required(budget, "certainty_fraction_numerator", "selection_budget")?,
            "selection_budget.certainty_fraction_numerator",
        )? != 2
        || unsigned(
            required(budget, "certainty_fraction_denominator", "selection_budget")?,
            "selection_budget.certainty_fraction_denominator",
        )? != 5
        || max_bytes != gate_max_bytes
        || certainty_bytes != gate_certainty_bytes
        || 5_u128 * u128::from(certainty_bytes) > 2_u128 * u128::from(max_bytes)
    {
        return Err("selection budget disagrees with the exact raw-byte 2/5 contract".to_string());
    }

    let auxiliary = object(
        required(root, "selection_auxiliary", "sampled subset")?,
        "selection_auxiliary",
    )?;
    const AUX_FIELDS: &[&str] = &["name", "sector_bytes", "scope", "expected_full", "selected"];
    exact_keys(auxiliary, AUX_FIELDS, AUX_FIELDS, "selection_auxiliary")?;
    let expected_aux = unsigned(
        required(auxiliary, "expected_full", "selection_auxiliary")?,
        "selection_auxiliary.expected_full",
    )?;
    let selected_aux = unsigned(
        required(auxiliary, "selected", "selection_auxiliary")?,
        "selection_auxiliary.selected",
    )?;
    if string(
        required(auxiliary, "name", "selection_auxiliary")?,
        "selection_auxiliary.name",
    )? != "l2_sector_transactions_32b"
        || unsigned(
            required(auxiliary, "sector_bytes", "selection_auxiliary")?,
            "selection_auxiliary.sector_bytes",
        )? != 32
        || string(
            required(auxiliary, "scope", "selection_auxiliary")?,
            "selection_auxiliary.scope",
        )? != "pass2_all_blocks"
        || selected_aux > expected_aux
    {
        return Err("selection auxiliary envelope is inconsistent".to_string());
    }

    let bytes = object(required(root, "bytes", "sampled subset")?, "bytes")?;
    const BYTE_FIELDS: &[&str] = &["raw", "compressed", "codec", "level", "long", "dict_id"];
    exact_keys(bytes, BYTE_FIELDS, BYTE_FIELDS, "bytes")?;
    let raw_bytes = unsigned(required(bytes, "raw", "bytes")?, "bytes.raw")?;
    unsigned(required(bytes, "compressed", "bytes")?, "bytes.compressed")?;
    if raw_bytes > max_bytes
        || string(required(bytes, "codec", "bytes")?, "bytes.codec")? != "zstd"
        || required(bytes, "level", "bytes")?.as_i64() != Some(19)
        || required(bytes, "long", "bytes")?.as_i64() != Some(27)
        || !required(bytes, "dict_id", "bytes")?.is_null()
    {
        return Err("bytes envelope is not the current bounded zstd contract".to_string());
    }

    let estimator = object(required(root, "estimator", "sampled subset")?, "estimator")?;
    exact_keys(
        estimator,
        &["mode", "targets", "not_recoverable", "convergence"],
        &[
            "mode",
            "form",
            "ci",
            "targets",
            "not_recoverable",
            "convergence",
        ],
        "estimator",
    )?;
    if string(required(estimator, "mode", "estimator")?, "estimator.mode")?
        != "weighted_full_shape_estimate"
    {
        return Err("subset estimator mode must be weighted_full_shape_estimate".to_string());
    }
    let targets = required(estimator, "targets", "estimator")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "estimator.targets must be a nonempty array".to_string())?;
    for target in targets {
        string(target, "estimator.targets entry")?;
    }
    string_array(
        required(estimator, "not_recoverable", "estimator")?,
        "estimator.not_recoverable",
    )?;
    if let Some(form) = estimator.get("form") {
        string(form, "estimator.form")?;
    }
    if let Some(ci_value) = estimator.get("ci") {
        let ci = object(ci_value, "estimator.ci")?;
        exact_keys(ci, &[], &["method", "B", "alpha"], "estimator.ci")?;
        if let Some(method) = ci.get("method") {
            string(method, "estimator.ci.method")?;
        }
        if let Some(samples) = ci.get("B") {
            positive(samples, "estimator.ci.B")?;
        }
        if let Some(alpha) = ci.get("alpha") {
            nonnegative_number(alpha, "estimator.ci.alpha")?;
        }
    }
    let convergence = object(
        required(estimator, "convergence", "estimator")?,
        "estimator.convergence",
    )?;
    exact_keys(
        convergence,
        &["status"],
        &["status", "per_target"],
        "estimator.convergence",
    )?;
    let status = string(
        required(convergence, "status", "estimator.convergence")?,
        "estimator.convergence.status",
    )?;
    if !["converged", "budget_limited_wide_ci", "unavailable"].contains(&status) {
        return Err("estimator convergence status is invalid".to_string());
    }
    if convergence
        .get("per_target")
        .is_some_and(|value| !value.is_object())
    {
        return Err("estimator.convergence.per_target must be an object".to_string());
    }

    let records = object(required(root, "records", "sampled subset")?, "records")?;
    let expected_records = unsigned(
        required(records, "expected_full", "records")?,
        "records.expected_full",
    )?;
    let captured_records = unsigned(
        required(records, "captured", "records")?,
        "records.captured",
    )?;
    if captured_records > expected_records {
        return Err("records.captured cannot exceed records.expected_full".to_string());
    }
    ValidatedPerSmSlotMap::retain(&roster_by_sm, grid_total)
}

#[cfg(feature = "campaign")]
fn validate_sampling_payload_snapshot(
    root: &Path,
    sidecar: &TraceSamplingSidecar,
    snapshot: &VerifiedSamplingPayload,
) -> Result<(), TraceError> {
    let sidecar_path = root.join("trace_sampling.json");
    let malformed = |message: String| TraceError::SamplingSidecarMalformed {
        path: sidecar_path.clone(),
        message,
    };
    let mut actual_entries = BTreeMap::new();
    for (path, raw) in &snapshot.by_path {
        let relative = path
            .strip_prefix(root)
            .ok()
            .and_then(Path::to_str)
            .ok_or_else(|| {
                malformed(format!(
                    "snapshotted payload path is not canonical under the trace root: {}",
                    path.display()
                ))
            })?;
        actual_entries.insert(
            relative,
            (
                u64::try_from(raw.len())
                    .map_err(|_| malformed(format!("payload {relative:?} length exceeds u64")))?,
                Sha256Digest::of(raw),
            ),
        );
    }
    let actual_paths: BTreeSet<_> = actual_entries.keys().copied().collect();
    let declared_paths: BTreeSet<_> = sidecar
        .payload_file_manifest
        .keys()
        .map(String::as_str)
        .collect();
    if actual_paths != declared_paths {
        let missing: Vec<_> = declared_paths.difference(&actual_paths).take(4).collect();
        let unexpected: Vec<_> = actual_paths.difference(&declared_paths).take(4).collect();
        return Err(malformed(format!(
            "per_block_raw_sha256_manifest path set disagrees with threadblock payload: missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    for (relative, (size, digest)) in actual_entries {
        let declared = &sidecar.payload_file_manifest[relative];
        if size != declared.size_bytes {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest size for {relative:?} is {}, actual {size}",
                declared.size_bytes
            )));
        }
        if digest.to_string() != declared.sha256 {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest hash for {relative:?} does not match raw payload (actual {digest})"
            )));
        }
    }
    if snapshot.identity.raw_bytes != sidecar.raw_payload_bytes {
        return Err(malformed(format!(
            "bytes.raw {} does not match threadblock payload bytes {}",
            sidecar.raw_payload_bytes, snapshot.identity.raw_bytes
        )));
    }
    if snapshot.identity.sorted_path_stream_sha256.to_string()
        != sidecar.decompressed_payload_sha256
    {
        return Err(malformed(format!(
            "decompressed_payload_sha256 does not match sorted threadblock bytes (actual {})",
            snapshot.identity.sorted_path_stream_sha256
        )));
    }
    Ok(())
}

#[cfg(feature = "campaign")]
impl TraceSamplingSidecar {
    /// Parse `<root>/trace_sampling.json` if present (`Ok(None)` if absent).
    /// This internal stage validates and retains the JSON envelope; callers only
    /// receive it through [`TraceDir::sampling_sidecar`] after `TraceDir::open`
    /// has also bound the decompressed threadblock bytes.

    fn parse(path: &Path, raw: &[u8]) -> Result<Self, TraceError> {
        Self::parse_with_limits(
            path,
            raw,
            TRACE_SAMPLING_JSON_MAX_BYTES,
            TRACE_SAMPLING_JSON_MAX_NODES,
        )
    }

    fn parse_with_limits(
        path: &Path,
        raw: &[u8],
        maximum_bytes: u64,
        maximum_nodes: u64,
    ) -> Result<Self, TraceError> {
        sidecar_byte_limit(path, "trace_sampling_json", raw, maximum_bytes)?;
        match preflight_json_shape(raw, maximum_nodes, None) {
            Ok(_) => {}
            Err(JsonPreflightError::Malformed(message)) => {
                return Err(TraceError::SamplingSidecarMalformed {
                    path: path.to_path_buf(),
                    message,
                });
            }
            Err(JsonPreflightError::ResourceLimit(error)) => {
                return Err(map_json_preflight_limit(path, "trace_sampling_json", error));
            }
        }
        let parsed: serde_json::Value =
            serde_json::from_slice(raw).map_err(|e| TraceError::SamplingSidecarMalformed {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        let malformed = |message: String| TraceError::SamplingSidecarMalformed {
            path: path.to_path_buf(),
            message,
        };
        macro_rules! get {
            ($object:expr, $field:expr, $context:expr) => {
                required($object, $field, $context).map_err(&malformed)?
            };
        }
        let root = object(&parsed, "trace_sampling.json root").map_err(&malformed)?;
        const TOP_LEVEL_FIELDS: &[&str] = &[
            "schema_version",
            "cell_id",
            "kernel",
            "method",
            "reason",
            "shape",
            "gates",
            "wave_model",
            "segments",
            "selected_segment_ids",
            "certainty_segments",
            "force_include_blocks",
            "per_sm_slot_map",
            "strata",
            "selection_reconciliation",
            "selection_budget",
            "selection_auxiliary",
            "rng",
            "estimator",
            "label_pairing",
            "tracer",
            "bytes",
            "records",
            "hashes",
        ];
        const BASE_REQUIRED: &[&str] = &[
            "schema_version",
            "cell_id",
            "kernel",
            "method",
            "reason",
            "shape",
            "gates",
            "records",
            "tracer",
            "hashes",
        ];
        exact_keys(
            root,
            BASE_REQUIRED,
            TOP_LEVEL_FIELDS,
            "trace_sampling.json root",
        )
        .map_err(&malformed)?;
        if get!(root, "schema_version", "trace_sampling.json root").as_u64() != Some(2) {
            return Err(malformed("schema_version must be 2".to_string()));
        }
        let method = get!(root, "method", "trace_sampling.json root")
            .as_str()
            .ok_or_else(|| malformed("method must be a string".to_string()))?;

        let cell_id = get!(root, "cell_id", "trace_sampling.json root")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("cell_id must be a non-empty string".to_string()))?
            .to_string();
        let kernel = object(get!(root, "kernel", "trace_sampling.json root"), "kernel")
            .map_err(&malformed)?;
        exact_keys(
            kernel,
            &["mangled_name", "args_hash"],
            &["mangled_name", "args_hash"],
            "kernel",
        )
        .map_err(&malformed)?;
        let kernel_mangled_name = get!(kernel, "mangled_name", "kernel")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("kernel.mangled_name must be nonempty".to_string()))?
            .to_string();
        let kernel_args_hash = get!(kernel, "args_hash", "kernel")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("kernel.args_hash must be nonempty".to_string()))?
            .to_string();
        get!(root, "reason", "trace_sampling.json root")
            .as_str()
            .ok_or_else(|| malformed("reason must be a string".to_string()))?;
        object(get!(root, "gates", "trace_sampling.json root"), "gates").map_err(&malformed)?;
        object(get!(root, "tracer", "trace_sampling.json root"), "tracer").map_err(&malformed)?;

        let shape =
            object(get!(root, "shape", "trace_sampling.json root"), "shape").map_err(&malformed)?;
        exact_keys(
            shape,
            &[
                "original_grid",
                "original_block",
                "captured_grid",
                "captured_block",
            ],
            &[
                "original_grid",
                "original_block",
                "captured_grid",
                "captured_block",
                "reduced_shape_variant",
            ],
            "shape",
        )
        .map_err(&malformed)?;
        let original_grid =
            parse_dim3(get!(shape, "original_grid", "shape"), "shape.original_grid")
                .map_err(&malformed)?;
        let captured_grid =
            parse_dim3(get!(shape, "captured_grid", "shape"), "shape.captured_grid")
                .map_err(&malformed)?;
        let original_block = parse_dim3(
            get!(shape, "original_block", "shape"),
            "shape.original_block",
        )
        .map_err(&malformed)?;
        let captured_block = parse_dim3(
            get!(shape, "captured_block", "shape"),
            "shape.captured_block",
        )
        .map_err(&malformed)?;

        let records = object(get!(root, "records", "trace_sampling.json root"), "records")
            .map_err(&malformed)?;
        exact_keys(
            records,
            &["expected_full", "captured"],
            &["expected_full", "captured"],
            "records",
        )
        .map_err(&malformed)?;
        let expected_full_records = get!(records, "expected_full", "records")
            .as_u64()
            .ok_or_else(|| malformed("records.expected_full must be a u64".to_string()))?;
        let captured_records = get!(records, "captured", "records")
            .as_u64()
            .ok_or_else(|| malformed("records.captured must be a u64".to_string()))?;

        let hashes = object(get!(root, "hashes", "trace_sampling.json root"), "hashes")
            .map_err(&malformed)?;
        exact_keys(
            hashes,
            &[
                "selected_block_ids_sha256",
                "decompressed_payload_sha256",
                "per_block_raw_sha256_manifest",
                "sidecar_sha256",
            ],
            &[
                "selected_block_ids_sha256",
                "decompressed_payload_sha256",
                "per_block_raw_sha256_manifest",
                "sidecar_sha256",
            ],
            "hashes",
        )
        .map_err(&malformed)?;
        for field in [
            "selected_block_ids_sha256",
            "decompressed_payload_sha256",
            "sidecar_sha256",
        ] {
            let value = get!(hashes, field, "hashes")
                .as_str()
                .ok_or_else(|| malformed(format!("hashes.{field} must be a string")))?;
            if !is_lower_hex_sha256(value) {
                return Err(malformed(format!(
                    "hashes.{field} must be lowercase SHA-256"
                )));
            }
        }
        let selected_block_ids_sha256 = get!(hashes, "selected_block_ids_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let decompressed_payload_sha256 = get!(hashes, "decompressed_payload_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let payload_file_manifest = parse_payload_file_manifest(
            get!(hashes, "per_block_raw_sha256_manifest", "hashes")
                .as_str()
                .ok_or_else(|| {
                    malformed("hashes.per_block_raw_sha256_manifest must be a string".to_string())
                })?,
        )
        .map_err(&malformed)?;
        let sidecar_sha256 = get!(hashes, "sidecar_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let expected_seal = canonical_sidecar_sha256(&parsed).map_err(&malformed)?;
        if sidecar_sha256 != expected_seal {
            return Err(malformed(
                "hashes.sidecar_sha256 does not match Python-canonical sidecar".to_string(),
            ));
        }

        let (method, raw_payload_bytes, per_sm_slot_map) = match method {
            "wave_segment_stratified_subset" => {
                const SUBSET_REQUIRED: &[&str] = &[
                    "wave_model",
                    "segments",
                    "selected_segment_ids",
                    "certainty_segments",
                    "force_include_blocks",
                    "per_sm_slot_map",
                    "strata",
                    "selection_reconciliation",
                    "selection_budget",
                    "selection_auxiliary",
                    "rng",
                    "estimator",
                    "label_pairing",
                    "bytes",
                ];
                exact_keys(root, SUBSET_REQUIRED, TOP_LEVEL_FIELDS, "sampled subset")
                    .map_err(&malformed)?;
                for field in [
                    "wave_model",
                    "selection_budget",
                    "selection_auxiliary",
                    "rng",
                    "estimator",
                    "bytes",
                ] {
                    object(get!(root, field, "sampled subset"), field).map_err(&malformed)?;
                }
                for field in ["force_include_blocks", "per_sm_slot_map", "strata"] {
                    get!(root, field, "sampled subset")
                        .as_array()
                        .ok_or_else(|| malformed(format!("{field} must be an array")))?;
                }
                get!(root, "label_pairing", "sampled subset")
                    .as_str()
                    .ok_or_else(|| malformed("label_pairing must be a string".to_string()))?;
                if captured_grid != original_grid {
                    return Err(malformed(
                        "shape.captured_grid must equal shape.original_grid for a subset"
                            .to_string(),
                    ));
                }

                let reconciliation = object(
                    get!(root, "selection_reconciliation", "sampled subset"),
                    "selection_reconciliation",
                )
                .map_err(&malformed)?;
                exact_keys(
                    reconciliation,
                    &["status", "attempts", "max_attempts", "capture_stats_sha256"],
                    &["status", "attempts", "max_attempts", "capture_stats_sha256"],
                    "selection_reconciliation",
                )
                .map_err(&malformed)?;
                let status = get!(reconciliation, "status", "selection_reconciliation").as_str();
                let attempts =
                    get!(reconciliation, "attempts", "selection_reconciliation").as_u64();
                let max_attempts =
                    get!(reconciliation, "max_attempts", "selection_reconciliation").as_u64();
                let capture_hash = get!(
                    reconciliation,
                    "capture_stats_sha256",
                    "selection_reconciliation"
                )
                .as_str();
                if status != Some("fixed_point")
                    || attempts.is_none_or(|value| value == 0)
                    || max_attempts.is_none_or(|value| value == 0)
                    || attempts > max_attempts
                    || capture_hash.is_none_or(|value| !is_lower_hex_sha256(value))
                {
                    return Err(malformed(
                        "invalid fixed-point reconciliation envelope".to_string(),
                    ));
                }

                let selected_segments = parse_unique_ids(
                    get!(root, "selected_segment_ids", "sampled subset"),
                    "selected_segment_ids",
                    false,
                )
                .map_err(&malformed)?;
                let certainty_segments = parse_unique_ids(
                    get!(root, "certainty_segments", "sampled subset"),
                    "certainty_segments",
                    false,
                )
                .map_err(&malformed)?;
                if !certainty_segments.is_subset(&selected_segments) {
                    return Err(malformed(
                        "certainty_segments must be a subset of selected_segment_ids".to_string(),
                    ));
                }

                let segment_values = get!(root, "segments", "sampled subset")
                    .as_array()
                    .ok_or_else(|| malformed("segments must be an array".to_string()))?;
                let mut segment_ids = BTreeSet::new();
                let mut segment_blocks_by_id = BTreeMap::new();
                let mut blocks = BTreeSet::new();
                for (index, segment_value) in segment_values.iter().enumerate() {
                    let context = format!("segments[{index}]");
                    let segment = object(segment_value, &context).map_err(&malformed)?;
                    exact_keys(
                        segment,
                        &["id", "block_ids"],
                        &["id", "block_ids"],
                        &context,
                    )
                    .map_err(&malformed)?;
                    let segment_id = get!(segment, "id", &context).as_u64().ok_or_else(|| {
                        malformed(format!("{context}.id must be a nonnegative integer"))
                    })?;
                    if !segment_ids.insert(segment_id) {
                        return Err(malformed(format!(
                            "segments contains duplicate id {segment_id}"
                        )));
                    }
                    let blocks_for_segment = parse_unique_ids(
                        get!(segment, "block_ids", &context),
                        &format!("{context}.block_ids"),
                        false,
                    )
                    .map_err(&malformed)?;
                    for block_id in &blocks_for_segment {
                        if *block_id > i64::MAX as u64 {
                            return Err(malformed(format!(
                                "selected block id {block_id} exceeds the replay index range"
                            )));
                        }
                        if !blocks.insert(*block_id) {
                            return Err(malformed(format!(
                                "selected block id {block_id} occurs in multiple segments"
                            )));
                        }
                    }
                    segment_blocks_by_id.insert(segment_id, blocks_for_segment);
                }
                if segment_ids != selected_segments {
                    return Err(malformed(
                        "segments ids must exactly equal selected_segment_ids".to_string(),
                    ));
                }

                let per_sm_slot_map = validate_subset_producer_contract(
                    root,
                    original_grid,
                    original_block,
                    captured_block,
                    &selected_segments,
                    &certainty_segments,
                    &segment_blocks_by_id,
                )
                .map_err(&malformed)?;

                let sorted_blocks: Vec<u64> = blocks.iter().copied().collect();
                let block_json = serde_json::to_string(&sorted_blocks).map_err(|error| {
                    malformed(format!(
                        "could not canonicalize selected block ids: {error}"
                    ))
                })?;
                let expected_block_hash = sha256_hex(block_json.as_bytes());
                if selected_block_ids_sha256 != expected_block_hash {
                    return Err(malformed(
                        "selected_block_ids_sha256 does not match segment membership".to_string(),
                    ));
                }

                (
                    TraceSamplingMethod::WaveSegmentStratifiedSubset(WaveSegmentSubsetProvenance {
                        selected_block_ids: blocks.into_iter().map(|value| value as i64).collect(),
                        selected_block_ids_sha256,
                        sidecar_sha256,
                    }),
                    unsigned(
                        required(
                            object(get!(root, "bytes", "sampled subset"), "bytes")
                                .map_err(&malformed)?,
                            "raw",
                            "bytes",
                        )
                        .map_err(&malformed)?,
                        "bytes.raw",
                    )
                    .map_err(&malformed)?,
                    per_sm_slot_map,
                )
            }
            // A lossless-compression declaration is sealed and payload-bound,
            // but replay must still observe complete launch coverage itself.
            "lossless_compressed_full" => {
                let (raw, per_sm_slot_map) = validate_lossless_producer_contract(
                    root,
                    original_grid,
                    original_block,
                    captured_grid,
                    captured_block,
                    &selected_block_ids_sha256,
                )
                .map_err(&malformed)?;
                (
                    TraceSamplingMethod::LosslessCompressedFull(LosslessFullProvenance {
                        selected_block_ids_sha256,
                        sidecar_sha256,
                    }),
                    raw,
                    per_sm_slot_map,
                )
            }
            _ => {
                return Err(malformed(
                    "method must be a sampled subset or lossless compressed full".to_string(),
                ));
            }
        };
        let manifest_bytes = payload_file_manifest
            .values()
            .try_fold(0_u64, |total, entry| {
                total.checked_add(entry.size_bytes).ok_or_else(|| {
                    malformed("per_block_raw_sha256_manifest byte total overflows u64".to_string())
                })
            })?;
        if manifest_bytes != raw_payload_bytes {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest byte total {manifest_bytes} disagrees with bytes.raw {raw_payload_bytes}"
            )));
        }
        Ok(TraceSamplingSidecar {
            cell_id,
            kernel_mangled_name,
            kernel_args_hash,
            original_grid,
            original_block,
            captured_grid,
            captured_block,
            expected_full_records,
            captured_records,
            decompressed_payload_sha256,
            payload_file_manifest,
            raw_payload_bytes,
            per_sm_slot_map,
            method,
        })
    }
}

pub fn block_launch_index(
    kernel: &pb::Kernel,
    tb: &pb::Threadblock,
    path: &Path,
) -> Result<i64, TraceError> {
    let grid = kernel
        .grid_dim
        .as_ref()
        .filter(|g| g.x > 0 && g.y > 0 && g.z > 0)
        .ok_or(TraceError::GridDimMissing {
            kernel_id: kernel.id,
        })?;
    let (gx, gy, gz) = (i64::from(grid.x), i64::from(grid.y), i64::from(grid.z));
    let grid_plane = gx
        .checked_mul(gy)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "grid plane size",
        })?;
    let grid_total = grid_plane
        .checked_mul(gz)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "grid volume",
        })?;
    let Some(bid) = tb.block_id.as_ref() else {
        return Err(TraceError::BlockIdMissing {
            path: path.to_path_buf(),
        });
    };
    let (x, y, z) = (bid.x, bid.y, bid.z);
    // Reject invalid coordinates before signed launch-index arithmetic. A malformed
    // out-of-axis coordinate must remain a range error even when its flattened
    // value exceeds i64; i128 covers every i32-coordinate/i32-grid product.
    if x < 0 || y < 0 || z < 0 || i64::from(x) >= gx || i64::from(y) >= gy || i64::from(z) >= gz {
        let index =
            i128::from(x) + i128::from(y) * i128::from(gx) + i128::from(z) * i128::from(grid_plane);
        return Err(TraceError::BlockIdOutOfRange {
            path: path.to_path_buf(),
            x,
            y,
            z,
            index,
            grid: grid_total,
        });
    }
    let y_offset = i64::from(y)
        .checked_mul(gx)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "block y offset",
        })?;
    let z_offset =
        i64::from(z)
            .checked_mul(grid_plane)
            .ok_or(TraceError::GridArithmeticOverflow {
                kernel_id: kernel.id,
                operation: "block z offset",
            })?;
    let index = i64::from(x)
        .checked_add(y_offset)
        .and_then(|value| value.checked_add(z_offset))
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "flattened block index",
        })?;
    // The per-axis checks above prevent an invalid CTA from flattening into a
    // valid launch index. Retain the flattened range check as a fail-closed
    // invariant over the checked arithmetic.
    if index >= grid_total {
        return Err(TraceError::BlockIdOutOfRange {
            path: path.to_path_buf(),
            x,
            y,
            z,
            index: i128::from(index),
            grid: grid_total,
        });
    }
    Ok(index)
}

/// Read one per-threadblock `.pb` file.
fn decode_threadblock(raw: &[u8], path: &Path) -> Result<pb::Threadblock, TraceError> {
    wire_preflight::preflight_threadblock(raw)
        .map_err(|error| map_wire_preflight_error(path, "dynamic_trace.threadblock", error))?;
    pb::Threadblock::decode(raw).map_err(|source| TraceError::Decode {
        path: path.to_path_buf(),
        message_type: "dynamic_trace.threadblock",
        source,
    })
}

/// One wire address record decoded for the memory hierarchy: the
/// reconstructed 32-lane vector plus the per-lane access width in bytes.
/// Width resolution mirrors the canonical Python consumer
/// (`l1_production_helper`): the address record's own `data_width` wins
/// when strictly positive, else the instruction-level width.
#[derive(Debug, Clone, Copy)]
pub struct AddressAccess {
    pub lane_vector: [i64; lanes::LANES_PER_WARP],
    /// Operand-specific effective lanes. An absent wire field falls back to
    /// the instruction mask. Async-copy sources narrow this for ignore-src
    /// zero-fill lanes without hiding the SHARED destination writes.
    pub lane_mask: u32,
    pub data_width_bytes: i32,
    /// Per-address memory space (address-carrying contract). Absent on legacy
    /// pre-transition traces (the field is unset), where consumers fall back to
    /// the instruction-level [`MemoryAccessMeta::op_class`]. A record carrying
    /// an explicit space lets a consumer route the global source of an atomic /
    /// async-copy to the L2 stream while exempting a shared destination.
    pub address_space: Option<pb::MemorySpace>,
    /// Backing global allocation id (absent/0 = unresolved). The identity gate
    /// requires a non-zero id resolving to a live sidecar allocation for a
    /// GLOBAL/GENERIC-space active record; SHARED/LOCAL are exempt.
    pub allocation_id: Option<u32>,
}

/// The instruction-level memory identity, without the per-record payloads —
/// what [`visit_memory_access`] returns after streaming the payloads into
/// the caller's sink.
#[derive(Debug, Clone, Copy)]
pub struct MemoryAccessMeta {
    pub op_class: pb::MemoryOpClass,
    /// Absent on legacy traces (pre-tag-12); consumers route absent to their
    /// UNSPECIFIED bucket, mirroring the Python consumers.
    pub access_kind: Option<pb::MemoryOpAccessKind>,
    /// Absent or UNSPECIFIED means "tracer did not classify the cache
    /// operator" — downstream L1 modeling must treat that as typed missing
    /// provenance, not as a default operator. (A wire value UNKNOWN to the
    /// committed schema is a typed reader error instead — schema skew.)
    pub cache_op: Option<pb::InstructionCacheOp>,
    /// `active_mask & predicate_mask`. May be 0: a predicated-off memory
    /// instruction still ISSUES (the SM model sees it); it just touches no
    /// lanes — the sink receives no records in that case, mirroring the
    /// canonical consumers, which skip lane reconstruction on zero masks.
    pub and_mask: u32,
}

/// A successful full structural preflight tied to one immutable instruction.
/// Consumers may apply semantic gates to [`Self::meta`] before invoking
/// [`Self::visit_addresses`], which preserves structural-error precedence without
/// decoding the complete wire record again.
#[derive(Debug, Clone, Copy)]
pub struct PreflightedMemoryAccess<'a> {
    inst: &'a pb::Instruction,
    meta: MemoryAccessMeta,
    inst_width: i32,
}

impl<'a> PreflightedMemoryAccess<'a> {
    pub fn instruction(self) -> &'a pb::Instruction {
        self.inst
    }

    pub fn meta(self) -> MemoryAccessMeta {
        self.meta
    }

    /// Reconstruct each already-validated address into stack storage in wire
    /// order. The instruction borrow prevents mutation between preflight and
    /// this pass, making a second decode failure an unreachable invariant.
    pub fn visit_addresses<F>(self, mut sink: F)
    where
        F: FnMut(&AddressAccess),
    {
        self.visit_addresses_enumerated(|_, access| sink(access));
    }

    fn visit_addresses_enumerated<F>(self, mut sink: F)
    where
        F: FnMut(usize, &AddressAccess),
    {
        if self.meta.and_mask == 0 {
            return;
        }
        let no_error_context = || String::new();
        for (operand, addr) in self.inst.addresses.iter().enumerate() {
            let access = match decode_address_access(
                addr,
                self.meta.and_mask,
                self.inst_width,
                &no_error_context,
            ) {
                Ok(access) => access,
                Err(_) => unreachable!("preflighted immutable address failed a second decode"),
            };
            sink(operand, &access);
        }
    }

    fn decoded_address(self, operand: usize) -> Option<AddressAccess> {
        let addr = self.inst.addresses.get(operand)?;
        let no_error_context = || String::new();
        Some(
            decode_address_access(addr, self.meta.and_mask, self.inst_width, &no_error_context)
                .unwrap_or_else(|_| {
                    unreachable!("preflighted immutable address failed a second decode")
                }),
        )
    }
}

/// Sum each decoded address operand's per-lane width over its effective lanes.
/// This preserves payload volume independently of source-sector coalescing.
pub fn project_effective_data_bytes(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u64, EffectiveDataBytesError> {
    let mut bytes = 0_u64;
    let mut overflowed = false;
    preflight.visit_addresses(|access| {
        if overflowed {
            return;
        }
        match effective_data_bytes_for_access(access) {
            Ok(operand_bytes) => match bytes.checked_add(operand_bytes) {
                Some(total) => bytes = total,
                None => overflowed = true,
            },
            Err(EffectiveDataBytesError::Overflow) => overflowed = true,
        }
    });
    if overflowed {
        Err(EffectiveDataBytesError::Overflow)
    } else {
        Ok(bytes)
    }
}

/// Count sectors whose decoded byte coverage is partial for one global store
/// instruction. This is a trace-side RMW-demand bound: it does not assert that
/// the line missed in L2, and callers must intersect it with their modeled
/// store-miss population before charging any lower-level read.
pub fn project_partial_store_sectors(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u64, EffectiveDataBytesError> {
    let mut coverage = BTreeMap::<i64, u32>::new();
    let mut overflowed = false;
    preflight.visit_addresses(|access| {
        if overflowed {
            return;
        }
        let width = match u64::try_from(access.data_width_bytes) {
            Ok(width) => width,
            Err(_) => {
                overflowed = true;
                return;
            }
        };
        for &address in access.lane_vector.iter() {
            if address == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            for byte in 0..width {
                let byte_address = match i128::from(address).checked_add(i128::from(byte)) {
                    Some(value) => value,
                    None => {
                        overflowed = true;
                        break;
                    }
                };
                let sector = byte_address.div_euclid(32);
                let offset = byte_address.rem_euclid(32) as u32;
                let Ok(sector) = i64::try_from(sector) else {
                    overflowed = true;
                    break;
                };
                let entry = coverage.entry(sector).or_default();
                *entry |= 1_u32 << offset;
            }
            if overflowed {
                break;
            }
        }
    });
    if overflowed {
        return Err(EffectiveDataBytesError::Overflow);
    }
    Ok(coverage.values().filter(|mask| **mask != u32::MAX).count() as u64)
}

fn project_operand_effective_data_bytes(
    preflight: PreflightedMemoryAccess<'_>,
    operand: usize,
) -> Result<u64, EffectiveDataBytesError> {
    let access = preflight
        .decoded_address(operand)
        .unwrap_or_else(|| unreachable!("validated async-copy source operand disappeared"));
    effective_data_bytes_for_access(&access)
}

fn effective_data_bytes_for_access(access: &AddressAccess) -> Result<u64, EffectiveDataBytesError> {
    let active_lanes = access
        .lane_vector
        .iter()
        .filter(|&&address| address != lanes::LANE_INACTIVE_SENTINEL)
        .count() as u64;
    let width =
        u64::try_from(access.data_width_bytes).map_err(|_| EffectiveDataBytesError::Overflow)?;
    width
        .checked_mul(active_lanes)
        .ok_or(EffectiveDataBytesError::Overflow)
}

const SHARED_BANK_COUNT_32X4: i64 = 32;

const SHARED_BANK_WORD_BYTES_32X4: i128 = 4;

const MAX_SHARED_WORDS_PER_LANE: u64 = 1_024;

const MAX_SHARED_WORDS_PER_INSTRUCTION: u64 = 65_536;

/// Reduce effective shared addresses to the exact bank-wavefront observable for
/// 32 banks with 4-byte words.
///
/// Ordinary loads/stores use the broadcast rule: repeated references to one
/// 32-bit word are serviced by one bank word.  Shared atomics are different:
/// each active lane performs an independent read-modify-write, so even lanes
/// targeting the same word must be serialized by the bank's atomic unit.  The
/// access-kind metadata is part of the trace contract and lets us keep those
/// two observers separate without treating global atomics as shared traffic
/// (this function is only called after the route has resolved to `Shared`).
pub fn project_shared_bank_wavefronts_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, SharedBankProjectionError> {
    word_scratch.clear();
    let mut projected_words = 0_u64;
    let mut max_data_width_bytes = 0_i32;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        max_data_width_bytes = max_data_width_bytes.max(access.data_width_bytes);
        let width = i128::from(access.data_width_bytes);
        for (lane, &address) in access.lane_vector.iter().enumerate() {
            if address == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let first_word = i128::from(address).div_euclid(SHARED_BANK_WORD_BYTES_32X4);
            let last_byte = i128::from(address) + width - 1;
            let last_word = last_byte.div_euclid(SHARED_BANK_WORD_BYTES_32X4);
            let words = u64::try_from(last_word - first_word + 1)
                .unwrap_or_else(|_| unreachable!("positive data width produced a negative span"));
            if words > MAX_SHARED_WORDS_PER_LANE {
                failure = Some(SharedBankProjectionError::LaneWordLimit {
                    lane,
                    words,
                    limit: MAX_SHARED_WORDS_PER_LANE,
                });
                return;
            }
            let Some(next_projected_words) = projected_words.checked_add(words) else {
                failure = Some(SharedBankProjectionError::InstructionWordLimit {
                    words: u64::MAX,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
                return;
            };
            if next_projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
                failure = Some(SharedBankProjectionError::InstructionWordLimit {
                    words: next_projected_words,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
                return;
            }
            let additional = usize::try_from(words)
                .unwrap_or_else(|_| unreachable!("bounded word span escaped usize"));
            if word_scratch.try_reserve(additional).is_err() {
                failure = Some(SharedBankProjectionError::CapacityUnavailable { additional });
                return;
            }
            for word in first_word..=last_word {
                word_scratch.push(
                    i64::try_from(word)
                        .unwrap_or_else(|_| unreachable!("i64 byte address word escaped i64")),
                );
            }
            projected_words = next_projected_words;
        }
    });
    if let Some(failure) = failure {
        return Err(failure);
    }

    let is_shared_atomic = preflight.meta().access_kind == Some(pb::MemoryOpAccessKind::Atomic)
        || preflight.meta().op_class == pb::MemoryOpClass::Atomic;
    if !is_shared_atomic {
        // SM120 services a 64-bit shared access in two independent 16-lane
        // phases and a 128-bit access in four independent 8-lane phases.
        // Broadcast collapse is local to each phase; deduplicating the whole
        // warp would incorrectly let one phase satisfy another.  Keep widths
        // above 128 bits on the historical whole-warp path because their
        // packetization is not represented by this compact contract.
        let lanes_per_phase = match max_data_width_bytes {
            5..=8 => 16_usize,
            9..=16 => 8_usize,
            _ => 0_usize,
        };
        let pack_sparse_active_loads = lanes_per_phase != 0
            && max_data_width_bytes <= 8
            && preflight.meta().access_kind == Some(pb::MemoryOpAccessKind::Load);
        if lanes_per_phase != 0 {
            // A whole-warp broadcast is still one service word even when the
            // operand is wide.  CUTLASS epilogues in particular issue a
            // vector load/store with every lane naming the same shared
            // fragment; forcing that through four quarter-warps would turn a
            // true broadcast into four fabricated wavefronts.  Detect this
            // source-level topology and retain the historical whole-warp
            // collapse for it.  Nonuniform lane spans use the phase model.
            let mut lane_spans: [Vec<(i128, i128)>; 32] = std::array::from_fn(|_| Vec::new());
            preflight.visit_addresses(|access| {
                let width = i128::from(access.data_width_bytes);
                for (lane, &address) in access.lane_vector.iter().enumerate() {
                    if address == lanes::LANE_INACTIVE_SENTINEL {
                        continue;
                    }
                    let first_word = i128::from(address).div_euclid(SHARED_BANK_WORD_BYTES_32X4);
                    let last_byte = i128::from(address) + width - 1;
                    let last_word = last_byte.div_euclid(SHARED_BANK_WORD_BYTES_32X4);
                    lane_spans[lane].push((first_word, last_word));
                }
            });
            let broadcast_signature = lane_spans.iter().find(|spans| !spans.is_empty());
            let all_active_lanes_broadcast = broadcast_signature.is_some()
                && lane_spans
                    .iter()
                    .filter(|spans| !spans.is_empty())
                    .all(|spans| Some(spans) == broadcast_signature);
            if all_active_lanes_broadcast {
                // Fall through to the existing whole-warp sort/dedup path.
                // This preserves the measured broadcast behavior while the
                // phased path below handles genuinely lane-varying accesses.
            } else {
                // Load requests packetize active lane spans by payload
                // capacity, not by physical lane-number windows. Stores
                // retain the established fixed half/quarter-warp packets;
                // their baseline is already exact on the real corpus.
                let active_lanes: Vec<usize> = lane_spans
                    .iter()
                    .enumerate()
                    .filter_map(|(lane, spans)| (!spans.is_empty()).then_some(lane))
                    .collect();
                let mut total_wavefronts = 0_u64;
                let phase_count = 32_usize / lanes_per_phase;
                for phase in 0..phase_count {
                    word_scratch.clear();
                    preflight.visit_addresses(|access| {
                        let width = i128::from(access.data_width_bytes);
                        for (lane, &address) in access.lane_vector.iter().enumerate() {
                            let in_phase = if pack_sparse_active_loads {
                                active_lanes
                                    .chunks(lanes_per_phase)
                                    .nth(phase)
                                    .is_some_and(|phase_lanes| phase_lanes.contains(&lane))
                            } else {
                                let lane_lo = phase * lanes_per_phase;
                                lane >= lane_lo && lane < lane_lo + lanes_per_phase
                            };
                            if !in_phase || address == lanes::LANE_INACTIVE_SENTINEL {
                                continue;
                            }
                            let first_word =
                                i128::from(address).div_euclid(SHARED_BANK_WORD_BYTES_32X4);
                            let last_byte = i128::from(address) + width - 1;
                            let last_word = last_byte.div_euclid(SHARED_BANK_WORD_BYTES_32X4);
                            for word in first_word..=last_word {
                                word_scratch.push(i64::try_from(word).unwrap_or_else(|_| {
                                    unreachable!("i64 byte address word escaped i64")
                                }));
                            }
                        }
                    });
                    if word_scratch.is_empty() {
                        continue;
                    }
                    word_scratch.sort_unstable();
                    word_scratch.dedup();
                    let mut bank_words = [0_u64; SHARED_BANK_COUNT_32X4 as usize];
                    let mut bank_overflow = None;
                    for &word in word_scratch.iter() {
                        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
                        let Some(next) = bank_words[bank].checked_add(1) else {
                            bank_overflow = Some(bank as u8);
                            break;
                        };
                        bank_words[bank] = next;
                    }
                    if let Some(bank) = bank_overflow {
                        return Err(SharedBankProjectionError::CountOverflow { bank });
                    }
                    total_wavefronts = total_wavefronts
                        .checked_add(bank_words.into_iter().max().unwrap_or(0))
                        .ok_or(SharedBankProjectionError::CountOverflow { bank: 0 })?;
                }
                // On SM120, a conflict-free full-warp `LDS.128` packet is
                // observed as two shared-data wavefronts rather than the
                // four quarter-warp scheduler phases implied by the byte
                // width.  Keep this correction on the decoupled observer
                // only; scheduler timing retains the legacy projection.
                let observed_wavefronts = if max_data_width_bytes == 16
                    && preflight.meta().access_kind == Some(pb::MemoryOpAccessKind::Load)
                    && active_lanes.len() == lanes::LANES_PER_WARP
                    && total_wavefronts == 4
                {
                    total_wavefronts / 2
                } else {
                    total_wavefronts
                };
                return u32::try_from(observed_wavefronts)
                    .map_err(|_| SharedBankProjectionError::CountOverflow { bank: 0 });
            }
        }

        // Shared LDS/STS broadcasts collapse repeated references to one
        // 32-bit word.  Keep this legacy path byte-for-byte identical for
        // ordinary shared accesses.
        word_scratch.sort_unstable();
        word_scratch.dedup();
    }
    let mut bank_words = [0_u64; SHARED_BANK_COUNT_32X4 as usize];
    for &word in word_scratch.iter() {
        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
        bank_words[bank] = bank_words[bank]
            .checked_add(1)
            .ok_or(SharedBankProjectionError::CountOverflow { bank: bank as u8 })?;
    }
    let wavefronts = bank_words.into_iter().max().unwrap_or(0);
    Ok(
        u32::try_from(wavefronts)
            .unwrap_or_else(|_| unreachable!("bounded projection escaped u32")),
    )
}

/// Whole-warp shared service projection retained for scheduler timing. This
/// deliberately excludes the phase-local observer refinement so changing the
/// L1TEX counter model cannot change SM issue or cycle state.
fn project_shared_bank_wavefronts_32x4_legacy_with_policy(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
    dedup_atomic: bool,
) -> Result<u32, SharedBankProjectionError> {
    word_scratch.clear();
    let mut projected_words = 0_u64;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        let width = i128::from(access.data_width_bytes);
        for (lane, &address) in access.lane_vector.iter().enumerate() {
            if address == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let first_word = i128::from(address).div_euclid(SHARED_BANK_WORD_BYTES_32X4);
            let last_byte = i128::from(address) + width - 1;
            let last_word = last_byte.div_euclid(SHARED_BANK_WORD_BYTES_32X4);
            let words = u64::try_from(last_word - first_word + 1)
                .unwrap_or_else(|_| unreachable!("positive data width produced a negative span"));
            if words > MAX_SHARED_WORDS_PER_LANE {
                failure = Some(SharedBankProjectionError::LaneWordLimit {
                    lane,
                    words,
                    limit: MAX_SHARED_WORDS_PER_LANE,
                });
                return;
            }
            let Some(next_projected_words) = projected_words.checked_add(words) else {
                failure = Some(SharedBankProjectionError::InstructionWordLimit {
                    words: u64::MAX,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
                return;
            };
            if next_projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
                failure = Some(SharedBankProjectionError::InstructionWordLimit {
                    words: next_projected_words,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
                return;
            }
            let additional = usize::try_from(words)
                .unwrap_or_else(|_| unreachable!("bounded word span escaped usize"));
            if word_scratch.try_reserve(additional).is_err() {
                failure = Some(SharedBankProjectionError::CapacityUnavailable { additional });
                return;
            }
            for word in first_word..=last_word {
                word_scratch.push(
                    i64::try_from(word)
                        .unwrap_or_else(|_| unreachable!("i64 byte address word escaped i64")),
                );
            }
            projected_words = next_projected_words;
        }
    });
    if let Some(failure) = failure {
        return Err(failure);
    }
    let is_shared_atomic = preflight.meta().access_kind == Some(pb::MemoryOpAccessKind::Atomic)
        || preflight.meta().op_class == pb::MemoryOpClass::Atomic;
    if dedup_atomic || !is_shared_atomic {
        word_scratch.sort_unstable();
        word_scratch.dedup();
    }
    let mut bank_words = [0_u64; SHARED_BANK_COUNT_32X4 as usize];
    for &word in word_scratch.iter() {
        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
        bank_words[bank] = bank_words[bank]
            .checked_add(1)
            .ok_or(SharedBankProjectionError::CountOverflow { bank: bank as u8 })?;
    }
    u32::try_from(bank_words.into_iter().max().unwrap_or(0))
        .map_err(|_| SharedBankProjectionError::CountOverflow { bank: 0 })
}

/// Whole-warp shared service projection retained for SM120 scheduler timing.
pub fn project_shared_bank_wavefronts_32x4_legacy(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, SharedBankProjectionError> {
    project_shared_bank_wavefronts_32x4_legacy_with_policy(preflight, word_scratch, false)
}

/// SM86 compatibility projection from the original branch. It deduplicates
/// broadcasts for every shared record, including records whose producer
/// metadata carries an atomic class.
pub fn project_shared_bank_wavefronts_32x4_sm86(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, SharedBankProjectionError> {
    project_shared_bank_wavefronts_32x4_legacy_with_policy(preflight, word_scratch, true)
}

/// Whole-warp width-only ideal paired with the scheduler projection.
pub fn project_shared_bank_ideal_wavefronts_32x4_legacy(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u32, SharedBankProjectionError> {
    let mut ideal_words = 0_u64;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        if access.data_width_bytes <= 0 {
            failure = Some(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words: 0,
                limit: MAX_SHARED_WORDS_PER_LANE,
            });
            return;
        }
        let width = u64::try_from(access.data_width_bytes)
            .unwrap_or_else(|_| unreachable!("validated width escaped u64"));
        let words = width
            .checked_add((SHARED_BANK_WORD_BYTES_32X4 as u64) - 1)
            .and_then(|value| value.checked_div(SHARED_BANK_WORD_BYTES_32X4 as u64))
            .unwrap_or(u64::MAX);
        if words > MAX_SHARED_WORDS_PER_LANE {
            failure = Some(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words,
                limit: MAX_SHARED_WORDS_PER_LANE,
            });
            return;
        }
        ideal_words = ideal_words.max(words);
    });
    if let Some(failure) = failure {
        return Err(failure);
    }
    u32::try_from(ideal_words).map_err(|_| SharedBankProjectionError::CountOverflow { bank: 0 })
}

/// Compute the ideal shared-memory wavefront count for one instruction under
/// the same 32-bank/4-byte-word contract as
/// [`project_shared_bank_wavefronts_32x4`].  The ideal count is the maximum
/// number of 4-byte words any active lane must transfer.  It therefore
/// preserves the unavoidable multi-wavefront cost of a wide per-lane access
/// (for example, an 8-byte load needs at least two wavefronts) while ignoring
/// inter-lane bank conflicts.  Nsight's shared-memory bank-conflict surface
/// measures excess wavefronts above this ideal, not excess above one request.
pub fn project_shared_bank_ideal_wavefronts_32x4(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u32, SharedBankProjectionError> {
    let mut ideal_words = 0_u64;
    let mut max_data_width_bytes = 0_i32;
    let mut active_lane_mask = 0_u32;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        max_data_width_bytes = max_data_width_bytes.max(access.data_width_bytes);
        for (lane, &address) in access.lane_vector.iter().enumerate() {
            if address != lanes::LANE_INACTIVE_SENTINEL {
                active_lane_mask |= 1_u32 << lane;
            }
        }
        if access.data_width_bytes <= 0 {
            failure = Some(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words: 0,
                limit: MAX_SHARED_WORDS_PER_LANE,
            });
            return;
        }
        let width = u64::try_from(access.data_width_bytes).unwrap_or_else(|_| unreachable!());
        let words = width
            .checked_add((SHARED_BANK_WORD_BYTES_32X4 as u64) - 1)
            .and_then(|value| value.checked_div(SHARED_BANK_WORD_BYTES_32X4 as u64))
            .unwrap_or(u64::MAX);
        if words > MAX_SHARED_WORDS_PER_LANE {
            failure = Some(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words,
                limit: MAX_SHARED_WORDS_PER_LANE,
            });
            return;
        }
        ideal_words = ideal_words.max(words);
    });
    if let Some(failure) = failure {
        return Err(failure);
    }

    // Match the phased service model used by
    // `project_shared_bank_wavefronts_32x4`: one ideal wavefront per active
    // half-warp (64-bit) or quarter-warp (128-bit).  The <=32-bit path and
    // widths above 128 bits retain the historical width-only ideal.
    let lanes_per_phase = match max_data_width_bytes {
        5..=8 => 16_u32,
        9..=16 => 8_u32,
        _ => 0_u32,
    };
    if lanes_per_phase != 0 {
        let mut phases = 0_u32;
        let phase_count = 32_u32 / lanes_per_phase;
        for phase in 0..phase_count {
            let phase_mask = ((1_u32 << lanes_per_phase) - 1) << (phase * lanes_per_phase);
            if active_lane_mask & phase_mask != 0 {
                phases = phases.saturating_add(1);
            }
        }
        return Ok(phases);
    }
    Ok(u32::try_from(ideal_words).unwrap_or_else(|_| unreachable!()))
}

#[derive(Debug, Clone, Copy)]
struct LdgstsLaneTransfer {
    source_start: u64,
    source_end: u64,
    source_first_sector: i64,
    source_last_sector: i64,
    destination_first_word: i64,
    destination_last_word: i64,
    lane: u8,
    cohort: u8,
}

struct ProjectedLdgstsSourceCohorts {
    topology: LdgstsDestinationTopology,
    cohorts: Vec<LdgstsSourceCohort>,
    source_sector_cohort_masks: Vec<u32>,
}

const LDGSTS_DESTINATION_LINE_WORDS_128B: i64 = 128 / 4;

fn finish_ldgsts_destination_cohort(
    word_scratch: &mut Vec<i64>,
    active_transfers: u32,
    payload_bytes: u32,
    cohorts: &mut Vec<LdgstsSourceCohort>,
) -> Result<(), LdgstsDestinationProjectionError> {
    if active_transfers == 0 {
        debug_assert!(word_scratch.is_empty());
        debug_assert_eq!(payload_bytes, 0);
        return Ok(());
    }
    word_scratch.sort_unstable();
    word_scratch.dedup();
    let mut bank_words = [0_u32; SHARED_BANK_COUNT_32X4 as usize];
    let mut destination_lines = 0_u32;
    let mut previous_line = None;
    for &word in word_scratch.iter() {
        let line = word.div_euclid(LDGSTS_DESTINATION_LINE_WORDS_128B);
        if previous_line != Some(line) {
            destination_lines = destination_lines
                .checked_add(1)
                .ok_or(LdgstsDestinationProjectionError::DestinationLineFanoutOverflow)?;
            previous_line = Some(line);
        }
        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
        bank_words[bank] = bank_words[bank].checked_add(1).ok_or(
            LdgstsDestinationProjectionError::BankWavefrontCountOverflow { bank: bank as u8 },
        )?;
    }
    let bank_wavefronts = bank_words.into_iter().max().unwrap_or(0);
    let bypass_product = destination_lines.checked_mul(bank_wavefronts).ok_or(
        LdgstsDestinationProjectionError::BypassWavefrontProductOverflow {
            destination_lines,
            bank_wavefronts,
        },
    )?;
    let bypass_wavefronts = active_transfers.min(bypass_product);
    cohorts
        .try_reserve(1)
        .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable { additional: 1 })?;
    cohorts.push(LdgstsSourceCohort {
        payload_bytes,
        destination_lines,
        bank_wavefronts,
        bypass_wavefronts,
    });
    word_scratch.clear();
    Ok(())
}

fn recompose_ldgsts_destination_topology(
    cohorts: &[LdgstsSourceCohort],
) -> Result<LdgstsDestinationTopology, LdgstsDestinationProjectionError> {
    let mut topology = LdgstsDestinationTopology {
        source_cohorts: u32::try_from(cohorts.len())
            .map_err(|_| LdgstsDestinationProjectionError::SourceCohortCountOverflow)?,
        ..LdgstsDestinationTopology::default()
    };
    for cohort in cohorts {
        topology.destination_line_fanout_total = topology
            .destination_line_fanout_total
            .checked_add(cohort.destination_lines)
            .ok_or(LdgstsDestinationProjectionError::DestinationLineFanoutOverflow)?;
        topology.destination_line_fanout_max = topology
            .destination_line_fanout_max
            .max(cohort.destination_lines);
        topology.cache_all_bank_wavefronts_total = topology
            .cache_all_bank_wavefronts_total
            .checked_add(cohort.bank_wavefronts)
            .ok_or(LdgstsDestinationProjectionError::WavefrontCountOverflow)?;
        topology.cache_all_bank_wavefronts_max = topology
            .cache_all_bank_wavefronts_max
            .max(cohort.bank_wavefronts);
        topology.bypass_wavefronts_total = topology
            .bypass_wavefronts_total
            .checked_add(cohort.bypass_wavefronts)
            .ok_or(LdgstsDestinationProjectionError::BypassWavefrontCountOverflow)?;
        topology.bypass_wavefronts_max =
            topology.bypass_wavefronts_max.max(cohort.bypass_wavefronts);
    }
    Ok(topology)
}

