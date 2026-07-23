"""Paired-control tests for the L1 cache-op classifier.

These are failing-test-first witnesses: they enumerate the
12-underscore-vocabulary closed enum, the per-arch SASS mnemonic
divergence (Ampere live vs. Blackwell canned fixtures per
``BL-20260505-cross-arch-sass-mnemonic-divergence``), the
sass_provenance_status fail-closed arms, and the
memory_op_class-vs-instruction_cache_op orthogonality
discipline.

The classifier's output schema is
``tools/cache_predictor/cache_op_map.schema.json``. Each test
exercises both the in-memory classifier API and the schema's
closure (so a future maintainer cannot silently widen the
12-token enum without also widening the classifier's mapping
table).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Cache-predictor support package import; per BL-20260505 the
# sibling-import sys.path discipline avoids collisions on bare
# module names across `tools/<subdir>/` directories. The classifier
# module is unique to this subtree.
from l1_cache_op_classifier import (  # noqa: E402
    INSTRUCTION_CACHE_OP_TOKENS,
    L1_ELIGIBLE_TOKENS,
    L1_BYPASS_TOKENS,
    classify_sass_text,
    derive_l1_eligible,
    OpMapEntry,
    SassProvenanceStatus,
)

_SCHEMA_PATH = _HERE / "cache_op_map.schema.json"


def _load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate_manifest(manifest: dict[str, Any]) -> None:
    jsonschema.validate(instance=manifest, schema=_load_schema())


_AMPERE_SAMPLE_SASS_LDG_E_CA = """
code for sm_86
                Function : _Z11test_kernelPi
        .headerflags    @\"EF_CUDA_TEXMODE_UNIFIED\"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.CA R5, [R4] ;
        /*0020*/                   EXIT ;
"""


_BLACKWELL_SAMPLE_SASS_STRONG_SM = """
code for sm_120
                Function : _Z11test_kernelPi
        .headerflags    @\"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS\"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.STRONG.SM R5, [R4] ;
        /*0020*/                   EXIT ;
"""


_BLACKWELL_SAMPLE_SASS_STRONG_GPU = """
code for sm_120
                Function : _Z11test_kernelPi
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.STRONG.GPU R5, [R4] ;
        /*0020*/                   EXIT ;
"""


_AMPERE_SAMPLE_SASS_LDGSTS = """
code for sm_86
                Function : _Z11async_kernelPi
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDGSTS.E [R0], [R2] ;
        /*0020*/                   EXIT ;
"""


_AMPERE_SAMPLE_SASS_LDL_STL = """
code for sm_86
                Function : _Z14spill_kernelPi
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDL R5, [R1] ;
        /*0020*/                   STL [R1+0x4], R5 ;
        /*0030*/                   EXIT ;
"""


_AMPERE_SAMPLE_SASS_LDG_E_VARIANTS = """
code for sm_86
                Function : _Z16variants_kernelPi
        /*0000*/                   LDG.E.CA R4, [R2] ;
        /*0010*/                   LDG.E.CG R5, [R2+0x4] ;
        /*0020*/                   LDG.E.CS R6, [R2+0x8] ;
        /*0030*/                   LDG.E.EF R7, [R2+0xc] ;
        /*0040*/                   EXIT ;
"""


_AMPERE_SAMPLE_SASS_ATOM_TEX = """
code for sm_86
                Function : _Z11misc_kernelPi
        /*0000*/                   ATOM.E.ADD R5, [R2], R3 ;
        /*0010*/                   TEX.SCR.LL R7, R8, R10, R12, 0x0, 2D, 0xf ;
        /*0020*/                   EXIT ;
"""


_AMPERE_SAMPLE_SASS_DEFAULT_LDG = """
code for sm_86
                Function : _Z14default_kernelPi
        /*0000*/                   LDG.E R4, [R2] ;
        /*0010*/                   EXIT ;
"""


_AMPERE_SAMPLE_SASS_CORRUPT = """
code for sm_86
this is not valid SASS output and contains no Function : marker
"""


_AMPERE_SAMPLE_SASS_EMPTY_KERNEL = """
code for sm_86
                Function : _Z11test_kernelPi
"""


_SAMPLE_SASS_NO_MATCHING_KERNEL = """
code for sm_86
                Function : _Z18different_kernelPi
        /*0000*/                   LDG.E.CA R5, [R4] ;
        /*0010*/                   EXIT ;
"""


_AMPERE_SAMPLE_SASS_NO_LOAD_CLASS = """
code for sm_86
                Function : _Z11pure_alu_kernelPi
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   IADD R2, R1, 0x4 ;
        /*0020*/                   EXIT ;
"""


class TwelveTokenEnumClosureTest(unittest.TestCase):
    """The 12 underscore-vocabulary tokens are fully enumerated and
    no token outside the set may appear in a classifier manifest."""

    def test_classifier_tokens_match_documented_vocabulary(self) -> None:
        documented = frozenset({
            "LDG_E_CA", "LDG_E_CG", "LDG_E_CS", "LDG_E_EF",
            "LDG_E_STRONG_SM", "LDG_E_STRONG_GPU",
            "LDL", "STL", "LDGSTS", "ATOM", "TEX", "OTHER",
        })
        self.assertEqual(set(INSTRUCTION_CACHE_OP_TOKENS), documented)
        self.assertEqual(len(INSTRUCTION_CACHE_OP_TOKENS), 12)

    def test_schema_enum_matches_classifier_tokens(self) -> None:
        schema = _load_schema()
        entry_props = (
            schema["properties"]["per_pc_instruction_cache_op"]
                  ["items"]["properties"]
        )
        schema_enum = entry_props["instruction_cache_op"]["enum"]
        self.assertEqual(
            sorted(schema_enum), sorted(INSTRUCTION_CACHE_OP_TOKENS),
        )

    def test_l1_eligibility_split(self) -> None:
        # L1-eligible tokens (the per-SM walker fills line state on
        # these).
        self.assertEqual(
            set(L1_ELIGIBLE_TOKENS),
            {"LDG_E_CA", "LDG_E_EF", "LDL", "STL"},
        )
        # L1-bypass tokens (no L1 line state participation; the
        # downstream predictor's L1 head filters these out before
        # walking).
        self.assertEqual(
            set(L1_BYPASS_TOKENS),
            {
                "LDG_E_CG", "LDG_E_CS",
                "LDG_E_STRONG_SM", "LDG_E_STRONG_GPU",
                "LDGSTS", "ATOM", "TEX", "OTHER",
            },
        )
        self.assertEqual(
            set(L1_ELIGIBLE_TOKENS) | set(L1_BYPASS_TOKENS),
            set(INSTRUCTION_CACHE_OP_TOKENS),
        )
        self.assertEqual(
            set(L1_ELIGIBLE_TOKENS) & set(L1_BYPASS_TOKENS),
            set(),
        )

    def test_derive_l1_eligible_round_trip(self) -> None:
        for token in INSTRUCTION_CACHE_OP_TOKENS:
            eligible = derive_l1_eligible(token)
            self.assertIsInstance(eligible, bool)
            if eligible:
                self.assertIn(token, L1_ELIGIBLE_TOKENS)
            else:
                self.assertIn(token, L1_BYPASS_TOKENS)


class AmpereLiveSassTokenRecognitionTest(unittest.TestCase):
    """Live sm_86 cuobjdump-shaped SASS text resolves the 10
    Ampere-recognizable tokens to underscore-vocabulary form."""

    def test_ldg_e_ca_is_l1_eligible(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_LDG_E_CA,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        self.assertEqual(
            manifest["sass_provenance_status"], "matched",
        )
        entries = manifest["per_pc_instruction_cache_op"]
        loads = [e for e in entries if e["pc"] == "0x10"]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0]["instruction_cache_op"], "LDG_E_CA")
        self.assertTrue(loads[0]["l1_eligible"])

    def test_ldg_e_variants_split_correctly(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_LDG_E_VARIANTS,
            kernel_unmangled_name="variants_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        self.assertEqual(
            entries_by_pc["0x0"]["instruction_cache_op"], "LDG_E_CA",
        )
        self.assertTrue(entries_by_pc["0x0"]["l1_eligible"])
        self.assertEqual(
            entries_by_pc["0x10"]["instruction_cache_op"], "LDG_E_CG",
        )
        self.assertFalse(entries_by_pc["0x10"]["l1_eligible"])
        self.assertEqual(
            entries_by_pc["0x20"]["instruction_cache_op"], "LDG_E_CS",
        )
        self.assertFalse(entries_by_pc["0x20"]["l1_eligible"])
        self.assertEqual(
            entries_by_pc["0x30"]["instruction_cache_op"], "LDG_E_EF",
        )
        self.assertTrue(entries_by_pc["0x30"]["l1_eligible"])

    def test_ldgsts_routes_to_bypass(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_LDGSTS,
            kernel_unmangled_name="async_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        load_entries = [
            e for e in manifest["per_pc_instruction_cache_op"]
            if e["instruction_cache_op"] == "LDGSTS"
        ]
        self.assertEqual(len(load_entries), 1)
        self.assertEqual(load_entries[0]["pc"], "0x10")
        self.assertFalse(load_entries[0]["l1_eligible"])

    def test_ldl_stl_routes_l1_eligible(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_LDL_STL,
            kernel_unmangled_name="spill_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        self.assertEqual(
            entries_by_pc["0x10"]["instruction_cache_op"], "LDL",
        )
        self.assertTrue(entries_by_pc["0x10"]["l1_eligible"])
        self.assertEqual(
            entries_by_pc["0x20"]["instruction_cache_op"], "STL",
        )
        self.assertTrue(entries_by_pc["0x20"]["l1_eligible"])

    def test_atom_and_tex_route_to_bypass(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_ATOM_TEX,
            kernel_unmangled_name="misc_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        self.assertEqual(
            entries_by_pc["0x0"]["instruction_cache_op"], "ATOM",
        )
        self.assertFalse(entries_by_pc["0x0"]["l1_eligible"])
        self.assertEqual(
            entries_by_pc["0x10"]["instruction_cache_op"], "TEX",
        )
        self.assertFalse(entries_by_pc["0x10"]["l1_eligible"])

    def test_bare_ldg_e_on_sm_86_maps_to_ldg_e_ca(self) -> None:
        # cuobjdump on Ampere (sm_86) emits the PTX ISA default
        # cache modifier as bare ``LDG.E`` rather than ``LDG.E.CA``.
        # The classifier resolves the bare form to ``LDG_E_CA``
        # because that IS the ISA-documented default on this
        # arch; bare ``LDG.E`` on sm_86 routes through L1.
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_DEFAULT_LDG,
            kernel_unmangled_name="default_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        self.assertEqual(
            entries_by_pc["0x0"]["instruction_cache_op"], "LDG_E_CA",
        )
        self.assertTrue(entries_by_pc["0x0"]["l1_eligible"])

    def test_bare_ldg_e_on_unknown_arch_stays_other(self) -> None:
        # On arches where the bare-LDG.E default-modifier mapping
        # is not yet documented in the classifier, the conservative
        # bare-LDG.E -> ``OTHER`` mapping holds. This is the
        # arch-coupling-discipline guard from CLAUDE.md: do not
        # assume an opcode set or default modifier is constant
        # across arches without explicit documentation + tests.
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_DEFAULT_LDG,
            kernel_unmangled_name="default_kernel",
            arch_token="sm_120",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        self.assertEqual(
            entries_by_pc["0x0"]["instruction_cache_op"], "OTHER",
        )
        self.assertFalse(entries_by_pc["0x0"]["l1_eligible"])

    def test_explicit_ldg_e_ca_still_matches_on_sm_86(self) -> None:
        # Precedence regression guard: when the cuobjdump output
        # carries an EXPLICIT ``LDG.E.CA``, the explicit-modifier
        # rule must continue to match first; the bare-LDG.E
        # arch-conditional resolution kicks in only when no
        # explicit modifier is present.
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_LDG_E_CA,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        loads = [
            e for e in manifest["per_pc_instruction_cache_op"]
            if e["instruction_cache_op"].startswith("LDG_E_")
        ]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0]["instruction_cache_op"], "LDG_E_CA")
        self.assertTrue(loads[0]["l1_eligible"])

    def test_bare_ldg_no_e_still_routes_to_other(self) -> None:
        # The arch-conditional bare-LDG.E -> LDG_E_CA mapping is
        # for the ``LDG.E`` form ONLY. A bare ``LDG`` (no ``.E``,
        # which would be a legacy non-extended global load) does
        # NOT upgrade. This pins the arch-conditional scope
        # narrowly so a future cuobjdump regression that emits
        # bare ``LDG`` cannot silently inherit the CA mapping.
        bare_ldg_sass = """
code for sm_86
                Function : _Z11test_kernelPi
        /*0000*/                   LDG R4, [R2] ;
        /*0010*/                   EXIT ;
"""
        manifest = classify_sass_text(
            bare_ldg_sass,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        self.assertEqual(
            entries_by_pc["0x0"]["instruction_cache_op"], "OTHER",
        )
        self.assertFalse(entries_by_pc["0x0"]["l1_eligible"])


class BlackwellCannedFixtureRecognitionTest(unittest.TestCase):
    """Blackwell-only tokens (`LDG_E_STRONG_SM`, `LDG_E_STRONG_GPU`)
    are covered via CANNED cuobjdump TEXT FIXTURES — live sm_120
    binary coverage defers to the Tier-2-cross-arch follow-up loop
    per the plan."""

    def test_ldg_e_strong_sm_is_l1_bypass(self) -> None:
        manifest = classify_sass_text(
            _BLACKWELL_SAMPLE_SASS_STRONG_SM,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_120",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        loads = [
            e for e in manifest["per_pc_instruction_cache_op"]
            if e["pc"] == "0x10"
        ]
        self.assertEqual(len(loads), 1)
        self.assertEqual(
            loads[0]["instruction_cache_op"], "LDG_E_STRONG_SM",
        )
        self.assertFalse(loads[0]["l1_eligible"])

    def test_ldg_e_strong_gpu_is_l1_bypass(self) -> None:
        manifest = classify_sass_text(
            _BLACKWELL_SAMPLE_SASS_STRONG_GPU,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_120",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        loads = [
            e for e in manifest["per_pc_instruction_cache_op"]
            if e["pc"] == "0x10"
        ]
        self.assertEqual(len(loads), 1)
        self.assertEqual(
            loads[0]["instruction_cache_op"], "LDG_E_STRONG_GPU",
        )
        self.assertFalse(loads[0]["l1_eligible"])


class FailClosedSassProvenanceTest(unittest.TestCase):
    """sass_provenance_status fail-closed arms produce empty
    per-PC mappings and a diagnostic field, never a populated
    array with a fake hit-rate scalar attached upstream."""

    def test_corrupt_sass_returns_no_kernel_section(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_CORRUPT,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        self.assertEqual(
            manifest["sass_provenance_status"],
            SassProvenanceStatus.NO_KERNEL_SECTION.value,
        )
        self.assertEqual(manifest["per_pc_instruction_cache_op"], [])
        self.assertIn("diagnostic", manifest)

    def test_empty_kernel_section_fires(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_EMPTY_KERNEL,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        self.assertEqual(
            manifest["sass_provenance_status"],
            SassProvenanceStatus.KERNEL_SECTION_EMPTY.value,
        )
        self.assertEqual(manifest["per_pc_instruction_cache_op"], [])

    def test_missing_kernel_returns_no_section(self) -> None:
        manifest = classify_sass_text(
            _SAMPLE_SASS_NO_MATCHING_KERNEL,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        self.assertEqual(
            manifest["sass_provenance_status"],
            SassProvenanceStatus.NO_KERNEL_SECTION.value,
        )

    def test_binary_not_parseable_status_emits_typed_diagnostic(self) -> None:
        # The classifier's pure-text API does not own the binary-load
        # path (that's `classify_binary`); the binary_not_parseable
        # status is set by the live caller when cuobjdump exits
        # non-zero. We exercise the schema admission for the value
        # so the future binary-loader integration can emit the same
        # shape.
        manifest = {
            "schema_version": 1,
            "binary_source": "live_cuobjdump",
            "arch_token": "sm_86",
            "sass_provenance_status": "binary_not_parseable",
            "per_pc_instruction_cache_op": [],
            "diagnostic": "cuobjdump exited with rc=1",
        }
        _validate_manifest(manifest)


class SchemaClosureNegativeTest(unittest.TestCase):
    """Invalid wire-format payloads are rejected by the schema."""

    def test_dotted_form_is_rejected(self) -> None:
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "sass_provenance_status": "matched",
            "per_pc_instruction_cache_op": [
                {
                    "pc": "0x10",
                    "instruction_cache_op": "LDG.E.CA",
                    "l1_eligible": True,
                },
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_unknown_token_is_rejected(self) -> None:
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "sass_provenance_status": "matched",
            "per_pc_instruction_cache_op": [
                {
                    "pc": "0x10",
                    "instruction_cache_op": "FOO_BAR",
                    "l1_eligible": True,
                },
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_pc_must_be_hex(self) -> None:
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "sass_provenance_status": "matched",
            "per_pc_instruction_cache_op": [
                {
                    "pc": "16",  # decimal, not hex
                    "instruction_cache_op": "LDG_E_CA",
                    "l1_eligible": True,
                },
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_unknown_status_is_rejected(self) -> None:
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "sass_provenance_status": "made_up_status",
            "per_pc_instruction_cache_op": [],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)


class StatusMappingContractTest(unittest.TestCase):
    """Conditional schema arms tie sass_provenance_status to the
    required shape of per_pc_instruction_cache_op + diagnostic +
    kernel_mangled_name. The contract:

      - matched: per_pc has >=1 entry; kernel_mangled_name required.
      - matched_no_load_class_instructions: per_pc empty;
        kernel_mangled_name + diagnostic required.
      - binary_not_parseable / no_kernel_section /
        kernel_section_empty: per_pc empty; diagnostic required.

    Each clause has one positive arm (the contract is satisfied)
    and one negative arm (the contract is violated and the schema
    rejects the manifest). The negative arms are the regression
    guard against a future maintainer relaxing the conditional
    constraint.
    """

    def test_matched_with_empty_per_pc_is_rejected(self) -> None:
        # Negative arm: matched status MUST carry at least one
        # per-PC entry. An empty array on a matched manifest is
        # the "contradictory provenance" case Codex's R0 review
        # flagged.
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "kernel_mangled_name": "_Z11test_kernelPi",
            "sass_provenance_status": "matched",
            "per_pc_instruction_cache_op": [],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_matched_without_kernel_mangled_name_is_rejected(self) -> None:
        # Negative arm: matched status MUST carry kernel_mangled_name
        # (the matched section's identity).
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "sass_provenance_status": "matched",
            "per_pc_instruction_cache_op": [
                {
                    "pc": "0x0",
                    "instruction_cache_op": "LDG_E_CA",
                    "l1_eligible": True,
                },
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_matched_with_one_entry_and_name_validates(self) -> None:
        # Positive arm: matched with 1 entry + kernel_mangled_name
        # validates.
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "kernel_mangled_name": "_Z11test_kernelPi",
            "sass_provenance_status": "matched",
            "per_pc_instruction_cache_op": [
                {
                    "pc": "0x0",
                    "instruction_cache_op": "LDG_E_CA",
                    "l1_eligible": True,
                },
            ],
        }
        _validate_manifest(manifest)

    def test_binary_not_parseable_with_non_empty_per_pc_is_rejected(
        self,
    ) -> None:
        # Negative arm: binary_not_parseable MUST carry an empty
        # per-PC list. Non-empty is the "contradictory provenance"
        # case Codex flagged.
        manifest = {
            "schema_version": 1,
            "binary_source": "live_cuobjdump",
            "arch_token": "sm_86",
            "sass_provenance_status": "binary_not_parseable",
            "diagnostic": "cuobjdump exited rc=1",
            "per_pc_instruction_cache_op": [
                {
                    "pc": "0x0",
                    "instruction_cache_op": "LDG_E_CA",
                    "l1_eligible": True,
                },
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_binary_not_parseable_without_diagnostic_is_rejected(self) -> None:
        # Negative arm: binary_not_parseable MUST carry a
        # diagnostic. Without it the operator has no audit trail.
        manifest = {
            "schema_version": 1,
            "binary_source": "live_cuobjdump",
            "arch_token": "sm_86",
            "sass_provenance_status": "binary_not_parseable",
            "per_pc_instruction_cache_op": [],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_binary_not_parseable_with_empty_and_diagnostic_validates(
        self,
    ) -> None:
        # Positive arm.
        manifest = {
            "schema_version": 1,
            "binary_source": "live_cuobjdump",
            "arch_token": "sm_86",
            "sass_provenance_status": "binary_not_parseable",
            "diagnostic": "cuobjdump exited rc=1",
            "per_pc_instruction_cache_op": [],
        }
        _validate_manifest(manifest)

    def test_no_kernel_section_without_diagnostic_is_rejected(self) -> None:
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "sass_provenance_status": "no_kernel_section",
            "per_pc_instruction_cache_op": [],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_kernel_section_empty_without_diagnostic_is_rejected(self) -> None:
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "sass_provenance_status": "kernel_section_empty",
            "per_pc_instruction_cache_op": [],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_matched_no_load_class_validates_with_full_required(self) -> None:
        # Positive arm: matched_no_load_class_instructions with
        # empty per-PC + kernel_mangled_name + diagnostic
        # validates.
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "kernel_mangled_name": "_Z11pure_alu_kernelPi",
            "sass_provenance_status": "matched_no_load_class_instructions",
            "diagnostic": "section had no load-class opcodes",
            "per_pc_instruction_cache_op": [],
        }
        _validate_manifest(manifest)

    def test_matched_no_load_class_with_non_empty_per_pc_is_rejected(
        self,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "kernel_mangled_name": "_Z11pure_alu_kernelPi",
            "sass_provenance_status": "matched_no_load_class_instructions",
            "diagnostic": "section had no load-class opcodes",
            "per_pc_instruction_cache_op": [
                {
                    "pc": "0x0",
                    "instruction_cache_op": "LDG_E_CA",
                    "l1_eligible": True,
                },
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)

    def test_matched_no_load_class_without_kernel_name_is_rejected(
        self,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "sass_provenance_status": "matched_no_load_class_instructions",
            "diagnostic": "section had no load-class opcodes",
            "per_pc_instruction_cache_op": [],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(manifest)


class TokenToEligibilityPartitionTest(unittest.TestCase):
    """Per-PC entries must satisfy the token-to-eligibility
    partition: LDG_E_CA / LDG_E_EF / LDL / STL require
    l1_eligible=true; the eight other tokens require
    l1_eligible=false. The schema's per-item oneOf rejects any
    pairing that disagrees with the partition.

    Codex reproducer (which previously validated): a
    matched manifest with instruction_cache_op="LDG_E_CG" +
    l1_eligible=true. This test class proves the reproducer is
    now rejected at the schema boundary.
    """

    @staticmethod
    def _manifest_with_entry(token: str, eligible: bool) -> dict:
        return {
            "schema_version": 1,
            "binary_source": "canned_fixture",
            "arch_token": "sm_86",
            "kernel_mangled_name": "_Z11test_kernelPi",
            "sass_provenance_status": "matched",
            "per_pc_instruction_cache_op": [
                {
                    "pc": "0x10",
                    "instruction_cache_op": token,
                    "l1_eligible": eligible,
                },
            ],
        }

    def test_eligible_tokens_with_true_validate(self) -> None:
        for token in ("LDG_E_CA", "LDG_E_EF", "LDL", "STL"):
            with self.subTest(token=token):
                _validate_manifest(
                    self._manifest_with_entry(token, True)
                )

    def test_bypass_tokens_with_false_validate(self) -> None:
        for token in (
            "LDG_E_CG", "LDG_E_CS",
            "LDG_E_STRONG_SM", "LDG_E_STRONG_GPU",
            "LDGSTS", "ATOM", "TEX", "OTHER",
        ):
            with self.subTest(token=token):
                _validate_manifest(
                    self._manifest_with_entry(token, False)
                )

    def test_eligible_tokens_with_false_are_rejected(self) -> None:
        for token in ("LDG_E_CA", "LDG_E_EF", "LDL", "STL"):
            with self.subTest(token=token):
                with self.assertRaises(jsonschema.ValidationError):
                    _validate_manifest(
                        self._manifest_with_entry(token, False)
                    )

    def test_bypass_tokens_with_true_are_rejected(self) -> None:
        for token in (
            "LDG_E_CG", "LDG_E_CS",
            "LDG_E_STRONG_SM", "LDG_E_STRONG_GPU",
            "LDGSTS", "ATOM", "TEX", "OTHER",
        ):
            with self.subTest(token=token):
                with self.assertRaises(jsonschema.ValidationError):
                    _validate_manifest(
                        self._manifest_with_entry(token, True)
                    )

    def test_codex_reproducer_ldg_e_cg_with_eligible_true_rejected(
        self,
    ) -> None:
        # The exact reproducer Codex's review pinned:
        # LDG_E_CG (bypass) + l1_eligible=true must now fail.
        with self.assertRaises(jsonschema.ValidationError):
            _validate_manifest(
                self._manifest_with_entry("LDG_E_CG", True)
            )

    def test_schema_partition_mirrors_classifier_frozensets(
        self,
    ) -> None:
        # Structural lockstep: the schema's oneOf partition mirrors
        # the runtime L1_ELIGIBLE_TOKENS / L1_BYPASS_TOKENS
        # frozensets in l1_cache_op_classifier.py.
        schema = _load_schema()
        items = schema["properties"]["per_pc_instruction_cache_op"]["items"]
        arms = items["oneOf"]
        schema_eligible: set[str] = set()
        schema_bypass: set[str] = set()
        for arm in arms:
            tokens = arm["properties"]["instruction_cache_op"]["enum"]
            const = arm["properties"]["l1_eligible"]["const"]
            if const is True:
                schema_eligible.update(tokens)
            elif const is False:
                schema_bypass.update(tokens)
        self.assertEqual(schema_eligible, set(L1_ELIGIBLE_TOKENS))
        self.assertEqual(schema_bypass, set(L1_BYPASS_TOKENS))
        # Sanity: the two arms partition the 12-token vocabulary.
        self.assertEqual(
            schema_eligible | schema_bypass,
            set(INSTRUCTION_CACHE_OP_TOKENS),
        )
        self.assertEqual(schema_eligible & schema_bypass, set())


class ClassifierMatchedNoLoadClassTest(unittest.TestCase):
    """The classifier emits matched_no_load_class_instructions when
    the kernel section is found and non-empty but contains no
    load-class opcodes (a pure-ALU or shared-only kernel)."""

    def test_pure_alu_section_emits_no_load_class_status(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_NO_LOAD_CLASS,
            kernel_unmangled_name="pure_alu_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        _validate_manifest(manifest)
        self.assertEqual(
            manifest["sass_provenance_status"],
            SassProvenanceStatus.MATCHED_NO_LOAD_CLASS_INSTRUCTIONS.value,
        )
        self.assertEqual(manifest["per_pc_instruction_cache_op"], [])
        self.assertIn("diagnostic", manifest)
        self.assertEqual(
            manifest["kernel_mangled_name"], "_Z11pure_alu_kernelPi",
        )


class OpMapEntryTypedShapeTest(unittest.TestCase):
    """The classifier's in-memory NamedTuple shape mirrors the
    schema closure; the to_dict / from_dict round-trip preserves
    every field."""

    def test_entry_round_trip(self) -> None:
        entry = OpMapEntry(
            pc="0x10",
            instruction_cache_op="LDG_E_CA",
            l1_eligible=True,
            raw_mnemonic="LDG.E.CA.U8",
        )
        d = entry.to_dict()
        self.assertEqual(d["pc"], "0x10")
        self.assertEqual(d["instruction_cache_op"], "LDG_E_CA")
        self.assertTrue(d["l1_eligible"])
        self.assertEqual(d["raw_mnemonic"], "LDG.E.CA.U8")


class MemoryOpClassOrthogonalityTest(unittest.TestCase):
    """`memory_op_class` (memory SPACE: GLOBAL / SHARED / LOCAL /
    CLUSTER_DSMEM / ATOMIC) is ORTHOGONAL to `instruction_cache_op`
    (cache OPERATOR). Impossible combinations (a SHARED-space access
    paired with an LDG-vocabulary cache-op token) fire a defect."""

    def test_ldg_e_ca_pairs_with_global(self) -> None:
        # Helper checks the orthogonality at the (memory_op_class,
        # instruction_cache_op) pair boundary.
        from l1_cache_op_classifier import (  # noqa: E402
            is_orthogonality_defect,
        )
        self.assertFalse(
            is_orthogonality_defect(
                memory_op_class="GLOBAL",
                instruction_cache_op="LDG_E_CA",
            ),
        )

    def test_ldl_pairs_with_local(self) -> None:
        from l1_cache_op_classifier import is_orthogonality_defect  # noqa: E402
        self.assertFalse(
            is_orthogonality_defect(
                memory_op_class="LOCAL",
                instruction_cache_op="LDL",
            ),
        )

    def test_atom_pairs_with_atomic(self) -> None:
        from l1_cache_op_classifier import is_orthogonality_defect  # noqa: E402
        self.assertFalse(
            is_orthogonality_defect(
                memory_op_class="ATOMIC",
                instruction_cache_op="ATOM",
            ),
        )

    def test_shared_plus_ldg_e_is_a_defect(self) -> None:
        from l1_cache_op_classifier import is_orthogonality_defect  # noqa: E402
        # SHARED memory uses LDS / LDSM mnemonics. Any
        # `LDG_E_*` cache-op against a SHARED memory_op_class
        # is impossible by construction.
        for token in (
            "LDG_E_CA", "LDG_E_CG", "LDG_E_CS", "LDG_E_EF",
            "LDG_E_STRONG_SM", "LDG_E_STRONG_GPU",
        ):
            self.assertTrue(
                is_orthogonality_defect(
                    memory_op_class="SHARED",
                    instruction_cache_op=token,
                ),
                f"SHARED + {token} must be a defect",
            )

    def test_cluster_dsmem_plus_ldl_is_a_defect(self) -> None:
        from l1_cache_op_classifier import is_orthogonality_defect  # noqa: E402
        self.assertTrue(
            is_orthogonality_defect(
                memory_op_class="CLUSTER_DSMEM",
                instruction_cache_op="LDL",
            ),
        )

    def test_global_plus_ldl_is_a_defect(self) -> None:
        from l1_cache_op_classifier import is_orthogonality_defect  # noqa: E402
        self.assertTrue(
            is_orthogonality_defect(
                memory_op_class="GLOBAL",
                instruction_cache_op="LDL",
            ),
        )


class ClassifierMisTagNegativeControlTest(unittest.TestCase):
    """If a fixture is constructed where the classifier mis-tags
    LDG.E.CA as LDG_E_CG (or vice versa), the assertion against the
    expected mapping fires. This is the per-shape negative-control
    discipline the L1 paired-control test suite requires (one
    negative control per supported wire-token classification path)."""

    def test_ldg_e_ca_must_not_be_classified_as_cg(self) -> None:
        manifest = classify_sass_text(
            _AMPERE_SAMPLE_SASS_LDG_E_CA,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        loads = [
            e for e in manifest["per_pc_instruction_cache_op"]
            if e["pc"] == "0x10"
        ]
        self.assertEqual(len(loads), 1)
        # Negative control: if the classifier returned LDG_E_CG
        # for an LDG.E.CA line, the per-arch SASS mnemonic
        # divergence discipline would be violated. The assertion
        # below is what catches a mis-tag.
        self.assertNotEqual(
            loads[0]["instruction_cache_op"], "LDG_E_CG",
        )

    def test_strong_sm_must_not_be_classified_as_ca(self) -> None:
        # The plan's wire vocabulary treats LDG.E.STRONG.SM as a
        # DIFFERENT token from LDG.E.CA — the strong-scope
        # variant is L1-bypass on Blackwell even though it
        # encodes the same .ca PTX intent. The classifier MUST
        # NOT collapse them into the same wire token.
        manifest = classify_sass_text(
            _BLACKWELL_SAMPLE_SASS_STRONG_SM,
            kernel_unmangled_name="test_kernel",
            arch_token="sm_120",
            binary_source="canned_fixture",
        )
        loads = [
            e for e in manifest["per_pc_instruction_cache_op"]
            if e["pc"] == "0x10"
        ]
        self.assertEqual(len(loads), 1)
        self.assertNotEqual(
            loads[0]["instruction_cache_op"], "LDG_E_CA",
        )


class PcRegexFirstCharCoverageTest(unittest.TestCase):
    """Paired-control regression for the `_PC_RE` regex repair.

    The pre-fix regex `0x?(...)` required PC literals to start with
    the literal character `0` (e.g., `/*0010*/`). PCs with a first
    hex character other than `0` (e.g., `/*1234*/`, `/*0x1234*/`)
    were silently dropped from the manifest. The post-fix regex
    `(?:0x)?(...)` treats the entire `0x` prefix as optional, so
    all PC literals are admitted regardless of the first hex
    character.

    The carveout_sensitivity_probe binary's LDG.E PC literals span
    past 0x1000; the pre-fix regex dropped 905 of 930 unique trace
    PCs (1,623,168 records) for that cell, producing
    `predicted_hit_rate=0.005228` vs NCU truth 0.9751. See the
    R14 dual-root-cause diagnostic at
    `data/sm_86_predictor_accuracy_recovery_v2/carveout_l1_head_diagnostic.json`.
    """

    def test_pc_starting_with_nonzero_admits(self) -> None:
        """Positive control: a PC literal whose first hex character
        is NOT `0` (e.g., `0x1234`) is now admitted to the
        manifest. The pre-fix regex dropped this PC silently."""
        sass = """
code for sm_86
                Function : _Z12start_at_1kPi
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*1234*/                   LDG.E.CA R5, [R4] ;
        /*1240*/                   EXIT ;
"""
        manifest = classify_sass_text(
            sass,
            kernel_unmangled_name="start_at_1k",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        self.assertIn(
            "0x1234",
            entries_by_pc,
            msg=(
                "PC 0x1234 not admitted; the _PC_RE regex must "
                "treat the `0x` prefix as fully optional via "
                "`(?:0x)?` instead of `0x?` which requires a "
                "literal leading `0`."
            ),
        )
        self.assertEqual(
            entries_by_pc["0x1234"]["instruction_cache_op"],
            "LDG_E_CA",
        )
        self.assertTrue(entries_by_pc["0x1234"]["l1_eligible"])

    def test_pc_with_explicit_0x_prefix_admits(self) -> None:
        """Positive control: a PC literal with an explicit `0x`
        prefix on a non-zero first character (e.g.,
        `/*0x1234*/`) is admitted. The pre-fix regex matched
        this only by accident (the literal `0` was the prefix's
        own `0`, not the value's first character)."""
        sass = """
code for sm_86
                Function : _Z13explicit_0x1kPi
        /*0x0000*/                   MOV R1, c[0x0][0x28] ;
        /*0x1234*/                   LDG.E.CA R5, [R4] ;
        /*0x1240*/                   EXIT ;
"""
        manifest = classify_sass_text(
            sass,
            kernel_unmangled_name="explicit_0x1k",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        self.assertIn("0x1234", entries_by_pc)
        self.assertEqual(
            entries_by_pc["0x1234"]["instruction_cache_op"],
            "LDG_E_CA",
        )

    def test_pc_with_non_hex_first_character_does_not_admit(
        self,
    ) -> None:
        """Negative control: the regex must not over-match. A PC
        literal that contains a non-hex character (e.g., `/*GG*/`)
        does not produce a manifest entry."""
        sass = """
code for sm_86
                Function : _Z14non_hex_kernelPi
        /*0010*/                   LDG.E.CA R5, [R4] ;
        /*GG*/                     LDG.E.CA R6, [R5] ;
        /*0020*/                   LDG.E.CA R7, [R6] ;
"""
        manifest = classify_sass_text(
            sass,
            kernel_unmangled_name="non_hex_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        pcs = {
            e["pc"]
            for e in manifest["per_pc_instruction_cache_op"]
        }
        # `0x10` and `0x20` valid LDG.E.CA; `GG` invalid hex and rejected.
        self.assertIn("0x10", pcs)
        self.assertIn("0x20", pcs)
        for pc in pcs:
            # Sanity: every admitted PC parses as hex.
            int(pc, 16)

    def test_mixed_pcs_high_and_low_addresses_all_admit(
        self,
    ) -> None:
        """Positive control: a SASS sample with PCs at low (0x...),
        mid (0x1...), and high (0xc...0xf...) addresses admits
        ALL of them. This is the exact failure mode the
        carveout_sensitivity_probe binary triggered: 25 of 26
        LDG.E PCs were at addresses past 0x1000 and were silently
        dropped by the pre-fix regex."""
        sass = """
code for sm_86
                Function : _Z16mixed_addr_kernelPi
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0130*/                   LDG.E.CA R5, [R4] ;
        /*1240*/                   LDG.E.CA R6, [R5] ;
        /*c090*/                   LDG.E.CA R7, [R6] ;
        /*fc00*/                   LDG.E.CA R8, [R7] ;
        /*fc10*/                   EXIT ;
"""
        manifest = classify_sass_text(
            sass,
            kernel_unmangled_name="mixed_addr_kernel",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )
        entries_by_pc = {
            e["pc"]: e for e in manifest["per_pc_instruction_cache_op"]
        }
        for pc in ("0x130", "0x1240", "0xc090", "0xfc00"):
            self.assertIn(
                pc,
                entries_by_pc,
                msg=(
                    f"PC {pc} not admitted; the regex repair "
                    "must admit all hex PC literals regardless "
                    "of leading character."
                ),
            )
            self.assertEqual(
                entries_by_pc[pc]["instruction_cache_op"],
                "LDG_E_CA",
            )


if __name__ == "__main__":
    unittest.main()
