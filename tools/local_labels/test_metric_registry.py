"""Tests for the committed local-label metric registry and its loader."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from metric_registry import (
    AVAILABILITY_TOKENS,
    CARRIER_KINDS,
    DEPLOYABILITY_KINDS,
    LABEL_SCHEMA_VERSION,
    PARSER_TRANSFORMS,
    REPEATABILITY_RULES,
    UNITS,
    MetricRegistry,
    MetricRegistryError,
    load_registry,
    load_registry_raw,
    parse_registry,
)

_REGISTRY_PATH = Path(__file__).resolve().parent / "metric_registry.json"


class MetricRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reg: MetricRegistry = load_registry()
        self.raw = load_registry_raw()

    def test_registry_loads_and_aligns_with_label_schema_version_5(self) -> None:
        self.assertEqual(self.reg.label_schema_version, 5)
        self.assertEqual(LABEL_SCHEMA_VERSION, 5)
        self.assertEqual(self.raw["label_schema_version"], 5)

    def test_parse_registry_uses_only_a_snapshot_of_caller_data(self) -> None:
        raw = load_registry_raw()
        parsed = parse_registry(raw)
        first_key = parsed.entries[0].key
        raw["metrics"][0]["key"] = "caller_mutated_after_parse"
        self.assertEqual(parsed.entries[0].key, first_key)

    def test_parse_registry_rejects_non_object_root(self) -> None:
        with self.assertRaisesRegex(MetricRegistryError, "root must be a JSON object"):
            parse_registry([])

    def test_parse_registry_normalizes_missing_row_field(self) -> None:
        raw = load_registry_raw()
        del raw["metrics"][0]["sidecar_field"]
        with self.assertRaisesRegex(MetricRegistryError, "registry is malformed"):
            parse_registry(raw)

    def test_on_disk_vocabularies_match_module_constants(self) -> None:
        self.assertEqual(tuple(self.raw["units"]), UNITS)
        self.assertEqual(tuple(self.raw["parser_transforms"]), PARSER_TRANSFORMS)
        self.assertEqual(
            tuple(self.raw["repeatability_rules"]), REPEATABILITY_RULES
        )
        self.assertEqual(
            tuple(self.raw["deployability_kinds"]), DEPLOYABILITY_KINDS
        )
        self.assertEqual(tuple(self.raw["carrier_kinds"]), CARRIER_KINDS)

    def test_repeatability_thresholds_match_provisional_per_unit_values(self) -> None:
        # Per the per-unit repeatability decision: fractions keep stddev<=0.03;
        # cycles/ns use CV<=0.05; counts use relative-tolerance<=0.05.
        self.assertAlmostEqual(
            self.reg.repeatability_thresholds["stddev_abs"], 0.03
        )
        self.assertAlmostEqual(
            self.reg.repeatability_thresholds["coefficient_of_variation"], 0.05
        )
        self.assertAlmostEqual(
            self.reg.repeatability_thresholds["relative_tolerance"], 0.05
        )

    def test_every_metric_resolves_to_unit_transform_aggregation_nullable_deploy(
        self,
    ) -> None:
        for e in self.reg.entries:
            self.assertIn(e.unit, UNITS, e.key)
            self.assertIn(e.parser_transform, PARSER_TRANSFORMS, e.key)
            self.assertIn(e.repeatability_rule, REPEATABILITY_RULES, e.key)
            self.assertIn(e.deployability, DEPLOYABILITY_KINDS, e.key)
            self.assertIn(e.carrier, CARRIER_KINDS, e.key)
            self.assertIn(e.availability, AVAILABILITY_TOKENS, e.key)
            self.assertIsInstance(e.nullable, bool, e.key)
            self.assertTrue(e.aggregation, e.key)

    def test_unit_transform_coherence(self) -> None:
        for e in self.reg.entries:
            self.assertEqual(
                e.unit == "pct",
                e.parser_transform == "percent_to_fraction",
                f"{e.key}: pct<->percent_to_fraction must agree",
            )

    def test_frozen_l2_metric_keeps_flat_field_and_percentage_path(self) -> None:
        e = self.reg.by_ncu_metric("lts__t_sector_hit_rate.pct")
        self.assertEqual(e.unit, "pct")
        self.assertEqual(e.carrier, "flat_field")
        self.assertEqual(e.sidecar_field, "cache_hit_rate_l2_sector")
        self.assertEqual(e.admitted_field, "cache_hit_rate_l2_sector_pct")
        self.assertEqual(e.parser_transform, "percent_to_fraction")
        self.assertFalse(e.nullable)
        self.assertEqual(e.deployability, "label_only")

    def test_l1_hit_rate_present_and_flat_field(self) -> None:
        e = self.reg.by_ncu_metric("l1tex__t_sector_hit_rate.pct")
        self.assertEqual(e.unit, "pct")
        self.assertEqual(e.carrier, "flat_field")
        self.assertEqual(e.sidecar_field, "cache_hit_rate_l1tex_sector")
        self.assertTrue(e.nullable)

    def test_execution_time_metrics_are_unit_tagged_nonpercentage(self) -> None:
        cyc = self.reg.by_ncu_metric("gpc__cycles_elapsed.max")
        self.assertEqual(cyc.unit, "cycles")
        self.assertEqual(cyc.carrier, "unit_tagged_metric")
        self.assertEqual(cyc.parser_transform, "identity_nonneg")
        self.assertEqual(cyc.repeatability_rule, "coefficient_of_variation")
        ns = self.reg.by_ncu_metric("gpu__time_duration.sum")
        self.assertEqual(ns.unit, "ns")
        self.assertEqual(ns.carrier, "unit_tagged_metric")
        self.assertEqual(ns.parser_transform, "identity_nonneg")

    def test_mandatory_traffic_subset_present_with_count_unit(self) -> None:
        mandatory = [
            e
            for e in self.reg.entries
            if e.family == "traffic" and e.traffic_subset == "mandatory"
        ]
        ncu = {e.ncu_metric for e in mandatory}
        # The decided mandatory subset: GMEM load/store requests + sectors,
        # DRAM sectors, L2 total requests.
        self.assertIn("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", ncu)
        self.assertIn("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", ncu)
        self.assertIn("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", ncu)
        self.assertIn("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", ncu)
        self.assertIn("dram__sectors_read.sum", ncu)
        self.assertIn("dram__sectors_write.sum", ncu)
        self.assertIn("lts__t_requests_srcunit_tex.sum", ncu)
        for e in mandatory:
            self.assertEqual(e.unit, "count", e.key)
            self.assertEqual(e.parser_transform, "identity_nonneg", e.key)
            self.assertEqual(e.repeatability_rule, "relative_tolerance", e.key)

    def test_achieved_occupancy_and_ipc_are_label_only(self) -> None:
        # No-deployability-leakage precondition: counter-derived achieved
        # occupancy + IPC may only be labels, never deployable features.
        achieved = self.reg.by_ncu_metric(
            "sm__warps_active.avg.pct_of_peak_sustained_active"
        )
        ipc = self.reg.by_ncu_metric("sm__inst_executed.avg.per_cycle_elapsed")
        self.assertEqual(achieved.deployability, "label_only")
        self.assertEqual(ipc.deployability, "label_only")
        self.assertNotIn(achieved.ncu_metric, self.reg.feature_eligible_ncu_metrics())
        self.assertNotIn(ipc.ncu_metric, self.reg.feature_eligible_ncu_metrics())

    def test_theoretical_occupancy_is_feature_eligible(self) -> None:
        e = self.reg.by_ncu_metric("launch__occupancy_limit_warps")
        self.assertEqual(e.deployability, "feature_eligible")
        self.assertIn(e.ncu_metric, self.reg.feature_eligible_ncu_metrics())

    def test_l1_total_requests_resolved_by_probe(self) -> None:
        # The NCU metric-availability probe resolved get_L1_Total_Requests to
        # l1tex__t_requests.sum and confirmed it on this card, so the registry
        # availability moved from the placeholder "unresolved" to "available".
        e = self.reg.by_key("traffic_l1_total_requests")
        self.assertEqual(e.ncu_metric, "l1tex__t_requests.sum")
        self.assertEqual(e.availability, "available")

    def test_label_field_names_avoid_import_label_denylist_prefixes(self) -> None:
        forbidden_prefixes = (
            "hw_counter",
            "raw_hw_counter",
            "pm_counter",
            "pm_sample",
            "pm_sampling",
            "counter_id",
            "counter_name",
            "cupti_event",
            "cupti_metric",
            "cupti_pc_sampling",
            "range_profiler",
            "sass_metric",
            "pc_sampling",
            "instruction_execution",
            "instruction_correlation",
            "event_id",
            "event_name",
            "metric_id",
            "metric_name",
            "signal_id",
            "signal_name",
            "nvperf",
            "perfworks",
        )
        for e in self.reg.entries:
            for fld in (e.sidecar_field, e.admitted_field):
                for pref in forbidden_prefixes:
                    self.assertFalse(
                        fld == pref or fld.startswith(pref + "_"),
                        f"label field {fld!r} for {e.key!r} trips the denylist "
                        f"prefix {pref!r}",
                    )

    def test_keys_and_ncu_metrics_are_unique(self) -> None:
        keys = [e.key for e in self.reg.entries]
        ncu = [e.ncu_metric for e in self.reg.entries]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(ncu), len(set(ncu)))

    def test_lookup_helpers(self) -> None:
        e = self.reg.by_key("exec_cycles_gpc_max")
        self.assertIs(self.reg.by_ncu_metric(e.ncu_metric), e)
        self.assertIs(self.reg.by_sidecar_field(e.sidecar_field), e)
        self.assertIs(self.reg.by_admitted_field(e.admitted_field), e)
        self.assertIsNone(self.reg.by_sidecar_field("does_not_exist"))
        with self.assertRaises(MetricRegistryError):
            self.reg.by_key("does_not_exist")

    def test_bad_registry_rejected(self) -> None:
        raw = load_registry_raw()
        raw["metrics"][3]["unit"] = "furlongs"
        tmp = Path(self.id())  # unused; build inline below
        del tmp
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            json.dump(raw, fh)
            bad_path = fh.name
        with self.assertRaises(MetricRegistryError):
            load_registry(bad_path)


class StrictOntologyParserTest(unittest.TestCase):
    """Negative cases for the strict registry_version>=2 ontology parser.

    Each case builds a minimal in-memory registry (registry_version 2), writes
    it to a temp file, and asserts load_registry rejects it with
    MetricRegistryError. A control case (a complete v2 registry) must load.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        # Cache misses on path => give each registry a distinct path.
        self._counter = 0

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write(self, registry: dict) -> str:
        self._counter += 1
        path = Path(self._tmpdir) / f"reg_{self._counter}.json"
        path.write_text(json.dumps(registry), encoding="utf-8")
        return str(path)

    def _obs_contract(self) -> dict:
        return {
            "counting_unit": "L1TEX",
            "rollup": "sum",
            "numerator": "l1tex__t_requests_pipe_lsu_mem_global_op_ld",
            "denominator": None,
            "op_classes": ["global_load"],
            "unsupported_op_behavior": (
                "kUnspecified->unsupported_memory_path typed fallback"
            ),
        }

    def _per_arch(self) -> dict:
        return {"sm_86": "available", "sm_120": "sm_120_deferred_no_live_evidence"}

    def _count_entry(self, **overrides) -> dict:
        entry = {
            "key": "probe_count",
            "ncu_metric": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            "unit": "count",
            "family": "traffic",
            "metric_class": "replay_count",
            "risk_tier": "T1",
            "carrier": "unit_tagged_metric",
            "sidecar_field": "probe_count",
            "admitted_field": "probe_count",
            "valid_min": 0.0,
            "valid_max": None,
            "parser_transform": "identity_nonneg",
            "aggregation": "mean_nonneg",
            "repeatability_rule": "relative_tolerance",
            "nullable": True,
            "deployability": "label_only",
            "availability": "available",
            "ncu_observation_contract": self._obs_contract(),
            "per_arch_availability": self._per_arch(),
        }
        entry.update(overrides)
        return entry

    def _derived_entry(self, **overrides) -> dict:
        entry = {
            "key": "probe_ratio",
            "ncu_metric": "lts__t_sector_hit_rate.pct",
            "unit": "pct",
            "family": "cache_hit_rate",
            "metric_class": "derived_ratio",
            "risk_tier": "T2",
            "carrier": "flat_field",
            "sidecar_field": "probe_ratio",
            "admitted_field": "probe_ratio_pct",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "parser_transform": "percent_to_fraction",
            "aggregation": "mean_fraction",
            "repeatability_rule": "stddev_abs",
            "nullable": True,
            "deployability": "label_only",
            "availability": "available",
            "ncu_observation_contract": {
                "counting_unit": "LTS",
                "rollup": "pct",
                "numerator": "lts__t_sector_hit_count",
                "denominator": "lts__t_sector_count",
                "op_classes": ["global_load"],
                "unsupported_op_behavior": (
                    "kUnspecified->unsupported_memory_path typed fallback"
                ),
            },
            "composition": {
                "formula": "label_only_no_predictor_composition",
                "dependencies": [],
                "device_constants": [],
            },
            "per_arch_availability": self._per_arch(),
        }
        entry.update(overrides)
        return entry

    def _registry(self, metrics: list) -> dict:
        return {
            "registry_version": 2,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "repeatability_thresholds": {
                "stddev_abs": 0.03,
                "coefficient_of_variation": 0.05,
                "relative_tolerance": 0.05,
            },
            "metrics": metrics,
        }

    def test_control_complete_v2_registry_loads(self) -> None:
        reg = load_registry(
            self._write(self._registry([self._count_entry(), self._derived_entry()]))
        )
        self.assertEqual(reg.registry_version, 2)
        self.assertEqual(len(reg.entries), 2)

    def test_v2_rejects_entry_missing_metric_class(self) -> None:
        entry = self._count_entry()
        del entry["metric_class"]
        with self.assertRaises(MetricRegistryError):
            load_registry(self._write(self._registry([entry])))

    def test_v2_rejects_entry_missing_risk_tier(self) -> None:
        entry = self._count_entry()
        del entry["risk_tier"]
        with self.assertRaises(MetricRegistryError):
            load_registry(self._write(self._registry([entry])))

    def test_v2_rejects_entry_missing_ncu_observation_contract(self) -> None:
        entry = self._count_entry()
        del entry["ncu_observation_contract"]
        with self.assertRaises(MetricRegistryError):
            load_registry(self._write(self._registry([entry])))

    def test_v2_rejects_observation_contract_missing_subkey(self) -> None:
        entry = self._count_entry()
        del entry["ncu_observation_contract"]["denominator"]
        with self.assertRaises(MetricRegistryError):
            load_registry(self._write(self._registry([entry])))

    def test_v2_rejects_entry_missing_per_arch_availability(self) -> None:
        entry = self._count_entry()
        del entry["per_arch_availability"]
        with self.assertRaises(MetricRegistryError):
            load_registry(self._write(self._registry([entry])))

    def test_v2_rejects_per_arch_availability_missing_arch(self) -> None:
        entry = self._count_entry()
        del entry["per_arch_availability"]["sm_120"]
        with self.assertRaises(MetricRegistryError):
            load_registry(self._write(self._registry([entry])))

    def test_v2_rejects_derived_entry_missing_composition(self) -> None:
        entry = self._derived_entry()
        del entry["composition"]
        with self.assertRaises(MetricRegistryError):
            load_registry(self._write(self._registry([entry])))

    def test_v2_rejects_composition_missing_subkey(self) -> None:
        entry = self._derived_entry()
        del entry["composition"]["dependencies"]
        with self.assertRaises(MetricRegistryError):
            load_registry(self._write(self._registry([entry])))

    def test_pre_v2_keeps_ontology_fields_optional(self) -> None:
        # Backward-compat: a registry_version 1 snapshot need NOT carry the
        # ontology fields, and still loads.
        entry = self._count_entry()
        for f in (
            "metric_class",
            "risk_tier",
            "ncu_observation_contract",
            "per_arch_availability",
        ):
            del entry[f]
        reg_dict = self._registry([entry])
        reg_dict["registry_version"] = 1
        reg = load_registry(self._write(reg_dict))
        self.assertEqual(reg.registry_version, 1)
        self.assertIsNone(reg.entries[0].metric_class)


class NoLabelLeakageGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = load_registry()

    def test_assert_no_label_leakage_flags_label_only_key(self) -> None:
        # ipc_inst_per_cycle is label_only; admitting it as a feature must fail.
        with self.assertRaises(MetricRegistryError):
            self.reg.assert_no_label_leakage(
                {"occupancy_limit_warps_theoretical", "ipc_inst_per_cycle"}
            )

    def test_assert_no_label_leakage_flags_label_only_ncu_id(self) -> None:
        with self.assertRaises(MetricRegistryError):
            self.reg.assert_no_label_leakage(
                {"sm__inst_executed.avg.per_cycle_elapsed"}
            )

    def test_assert_no_label_leakage_allows_feature_eligible(self) -> None:
        # The static theoretical-occupancy limits are feature_eligible and must
        # pass the guard (no exception).
        self.reg.assert_no_label_leakage(
            {
                "occupancy_limit_warps_theoretical",
                "occupancy_limit_blocks_theoretical",
                "occupancy_limit_registers_theoretical",
                "occupancy_limit_shared_mem_theoretical",
            }
        )


if __name__ == "__main__":
    unittest.main()
