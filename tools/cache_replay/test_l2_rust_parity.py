"""Cross-language lockstep for the shared-L2 model (BL-20260507 pattern).

`sim/mem/src/l2.rs` is a lockstep port of THIS package's cache core
(`sector_cache_replay.replay_access_stream` + `_install_way`). Like the
control-layout and cache-op parity tests, this file pins the shared
surfaces from the Python side while the Rust unit tests pin the very
same values (`sim/mem/src/l2.rs::tests::python_parity_corpus_pins`), so
either side drifting breaks a test in the SAME commit:

- structural pins: policy constants (SRRIP RRPV / BIP period), the
  slice-hash registry names, and the fixed 32/128 granularity;
- behavioral pins: a deterministic LCG corpus (all three policies, six
  hashes, negative line ids) replayed through the PYTHON engine must
  reproduce the pinned (hits, total) counters bit-exactly.

Regenerating the pins is deliberate work: rerun the corpus through
`replay_access_stream` and update BOTH files in one commit.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import sector_cache_replay as m

_REPO_ROOT = Path(__file__).resolve().parents[2]
_L2_RS = _REPO_ROOT / "sim" / "mem" / "src" / "l2.rs"

pytestmark = pytest.mark.skipif(not _L2_RS.is_file(), reason="sim tree absent")

# One entry per corpus case; the identical table is pinned in l2.rs.
# (policy, hash, seed, capacity, assoc, slices, n, line_space, signed,
#  hits, total)
PARITY_CORPUS = [
    ("LRU", "line_mod", 1, 16384, 4, 3, 500, 128, False, 159, 500),
    ("LRU", "knuth_mult", 2, 16384, 4, 3, 500, 128, False, 164, 500),
    ("SRRIP", "xor_folded", 3, 16384, 4, 3, 500, 128, False, 150, 500),
    ("SRRIP", "bit_reverse", 4, 4096, 2, 2, 500, 256, False, 6, 500),
    ("BIP", "sector_mod", 5, 16384, 4, 3, 500, 128, False, 181, 500),
    ("BIP", "shift_mod", 6, 4096, 2, 2, 500, 256, False, 11, 500),
    ("LRU", "line_mod", 7, 4096, 2, 3, 400, 96, True, 14, 400),
]

_MASK = (1 << 64) - 1


def _lcg_stream(seed: int, n: int, line_space: int, signed: bool):
    x = seed
    lines, sectors = [], []
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & _MASK
        line = (x >> 20) % line_space
        if signed and (x & 1):
            line = -line
        lines.append(line)
        sectors.append((x >> 10) & 3)
    return lines, sectors


@pytest.mark.parametrize(
    "case", PARITY_CORPUS, ids=lambda c: f"{c[0]}_{c[1]}_seed{c[2]}"
)
def test_python_engine_reproduces_pinned_corpus(case) -> None:
    policy, hash_id, seed, cap, assoc, slices, n, space, signed, hits, total = case
    cfg = m.CacheReplayConfig(
        capacity_bytes=cap,
        associativity=assoc,
        replacement_policy=policy,
        num_slices=slices,
        slice_hash_id=hash_id,
    )
    lines, sectors = _lcg_stream(seed, n, space, signed)
    r = m.replay_access_stream(lines, sectors, cfg, classify_misses=False)
    assert (r.hits, r.total_accesses) == (hits, total)


def test_rust_side_pins_the_same_corpus() -> None:
    """The identical rows must be pinned in l2.rs (same-commit lockstep)."""
    src = _L2_RS.read_text()
    assert "python_parity_corpus_pins" in src, "Rust corpus-pin test missing"
    for case in PARITY_CORPUS:
        policy, hash_id, seed, cap, assoc, slices, n, space, signed, hits, total = case
        needle = f"{n}, {space}, {str(signed).lower()}, {hits}, {total})"
        assert needle in src, (
            f"corpus row (seed={seed}, hits={hits}, total={total}) not pinned "
            f"in l2.rs — regenerate BOTH sides in one commit"
        )


def test_policy_constants_match() -> None:
    src = _L2_RS.read_text()

    def rust_const(name: str) -> int:
        match = re.search(rf"const {name}: \w+ = (\d+);", src)
        assert match, f"{name} not found in l2.rs"
        return int(match.group(1))

    assert rust_const("SRRIP_RRPV_MAX") == m._SRRIP_RRPV_MAX
    assert rust_const("SRRIP_INSERT_RRPV") == m._SRRIP_INSERT_RRPV
    assert rust_const("BIP_MRU_INSERT_PERIOD") == m._BIP_MRU_INSERT_PERIOD
    assert m.SECTOR_BYTES == 32 and m.LINE_BYTES == 128


def test_slice_hash_registry_matches_rust_enum() -> None:
    """Every Python hash id must have a Rust variant and vice versa."""
    src = _L2_RS.read_text()
    variant_of = {
        "line_mod": "LineMod",
        "sector_mod": "SectorMod",
        "xor_folded": "XorFolded",
        "knuth_mult": "KnuthMult",
        "shift_mod": "ShiftMod",
        "bit_reverse": "BitReverse",
    }
    assert set(variant_of) == set(m.SLICE_HASH_IDS)
    enum_body = re.search(r"pub enum SliceHash \{(.*?)\}", src, re.DOTALL)
    assert enum_body, "SliceHash enum not found"
    variants = set(re.findall(r"^\s*(\w+),", enum_body.group(1), re.MULTILINE))
    assert variants == set(variant_of.values())
