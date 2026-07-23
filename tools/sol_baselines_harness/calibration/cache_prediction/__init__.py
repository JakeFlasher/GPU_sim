"""Cache-prediction NCU label harvester + parser.

Reserves the calibration-only NCU surface for the Tier-2 cache
predictor's offline ground-truth labels per the CLAUDE.md Rule 1
carveout: NCU may be invoked on sm_86 / sm_120 hosts to gather
read-only sector hit-rate metrics — L1 global-load family
(`l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` plus
the underlying
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` /
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum`)
and L2 op_read family
(`lts__t_sector_op_read_hit_rate.pct` plus
`lts__t_sectors_op_read.sum` /
`lts__t_sectors_op_read_lookup_hit.sum`) for calibration. The
deployed framework path and every B200 invocation MUST NOT
import from this subtree.

Importing this package does NOT invoke NCU. NCU is launched
only when a caller explicitly invokes `harvest_ncu_label(...)`.
"""

from baselines_harness.calibration.cache_prediction.label_parser import (
    NcuLabelParserError,
    NcuLabelRecord,
    NCU_LABEL_PARSER_TYPED_REASONS,
    parse_ncu_csv,
)
from baselines_harness.calibration.cache_prediction.label_smoke import (
    LABEL_SMOKE_SCHEMA_VERSION,
    LABEL_SMOKE_TYPED_REASONS,
    LabelSmokeArtifact,
    LabelSmokeCell,
    LabelSmokeError,
    LabelSmokeRecord,
    dump_label_smoke_artifact_to_json,
    load_label_smoke_artifact_from_json,
    run_label_smoke,
    validate_label_smoke_artifact,
)
from baselines_harness.calibration.cache_prediction.ncu_harvester import (
    NCU_HARVESTER_TYPED_REASONS,
    NCU_INVOCATION_TEMPLATE,
    NCU_REQUIRED_METRICS,
    NcuHarvesterError,
    harvest_ncu_label,
)
from baselines_harness.calibration.cache_prediction.corpus_manifest import (
    CALIBRATION_CORPUS_SCHEMA_VERSION,
    CORPUS_MANIFEST_MEASURED_CACHE_INITIAL_STATES,
    CORPUS_MANIFEST_PREDICTOR_BUCKETS,
    CORPUS_MANIFEST_TYPED_REASONS,
    CalibrationCorpusCell,
    CalibrationCorpusManifest,
    CalibrationCorpusMeta,
    CorpusManifestError,
    load_calibration_corpus_manifest,
    validate_calibration_corpus_manifest,
)

__all__ = [
    "CALIBRATION_CORPUS_SCHEMA_VERSION",
    "CORPUS_MANIFEST_MEASURED_CACHE_INITIAL_STATES",
    "CORPUS_MANIFEST_PREDICTOR_BUCKETS",
    "CORPUS_MANIFEST_TYPED_REASONS",
    "CalibrationCorpusCell",
    "CalibrationCorpusManifest",
    "CalibrationCorpusMeta",
    "CorpusManifestError",
    "LABEL_SMOKE_SCHEMA_VERSION",
    "LABEL_SMOKE_TYPED_REASONS",
    "LabelSmokeArtifact",
    "LabelSmokeCell",
    "LabelSmokeError",
    "LabelSmokeRecord",
    "NCU_HARVESTER_TYPED_REASONS",
    "NCU_INVOCATION_TEMPLATE",
    "NCU_LABEL_PARSER_TYPED_REASONS",
    "NCU_REQUIRED_METRICS",
    "NcuHarvesterError",
    "NcuLabelParserError",
    "NcuLabelRecord",
    "dump_label_smoke_artifact_to_json",
    "harvest_ncu_label",
    "load_calibration_corpus_manifest",
    "load_label_smoke_artifact_from_json",
    "parse_ncu_csv",
    "run_label_smoke",
    "validate_calibration_corpus_manifest",
    "validate_label_smoke_artifact",
]
