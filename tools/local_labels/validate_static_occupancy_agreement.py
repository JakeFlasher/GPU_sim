"""Exact-agreement validator: static occupancy-limit predictions vs NCU.

Compares the four ``launch__occupancy_limit_*`` static predictions (produced by
``emit_static_occupancy_prediction.py``) against the launch-section values NCU
records in a committed ``.ncu-rep`` capture. ALL FOUR limits are gated for EXACT
integer agreement whenever NCU exposes a value. The static occupancy producer
emits ``launch__occupancy_limit_shared_mem`` as ``typed_unavailable`` unless the
launch's configured shared-memory carveout is supplied (NCU's value depends on
that runtime carveout plus the driver-reserved per-block shared memory), so a
pure-static shared_mem prediction is excluded from the comparison by the
STANDARD ``typed_unavailable`` rule -- never a per-metric carve-out -- and a
``typed_unavailable`` or mismatching metric can never yield a false
``overall_agree=True``.

Where a prediction is ``typed_unavailable`` (e.g. every limit on the deferred
sm_120) or NCU exposes no launch-section value, the metric is recorded with a
typed reason and excluded from the exact-agreement gate -- absence is never
treated as a numeric 0.

This module reads an existing ``.ncu-rep`` (read-only ``ncu --import``; no
profiling run, no replay, no clock control); it lives under
``tools/local_labels/`` as an authorized label-evidence consumer. The comparison
core is GPU-free and text-driven.

Public surface:

- ``compare_occupancy_agreement(predictions, ncu_values)`` -> agreement record.
- ``parse_ncu_launch_section_csv(csv_text)`` -> list of per-launch value dicts.
- ``read_ncu_launch_section(rep_path)`` -> per-launch value dicts (shells ncu).
- ``build_agreement_artifact(prediction_artifact, ncu_values, ...)`` + ``emit``.
- CLI: ``--prediction ... (--ncu-rep ... | --ncu-csv-file ...) --out ...``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

from occupancy_calculator import LAUNCH_OCCUPANCY_LIMIT_METRICS

_AGREEMENT_SCHEMA_VERSION = 1
# ALL four launch occupancy limits are gated for exact agreement whenever NCU
# exposes a value. A prediction that is typed_unavailable (e.g.
# launch__occupancy_limit_shared_mem without a configured carveout, or every
# limit on the deferred sm_120) is excluded from the gate by the STANDARD
# typed_unavailable rule below -- NOT a per-metric carve-out -- so a
# typed_unavailable or mismatching metric can never produce a false
# overall_agree=True.
_GATED_METRICS = LAUNCH_OCCUPANCY_LIMIT_METRICS


class OccupancyAgreementError(ValueError):
    """Raised when NCU launch-section values cannot be read/parsed."""


def compare_occupancy_agreement(predictions: dict, ncu_values: dict) -> dict:
    """Compare a prediction block to NCU launch-section values.

    ``predictions`` is the artifact's ``predictions`` block (4 records each with
    ``value``/``status``). ``ncu_values`` maps each launch__occupancy_limit_*
    name to an int (the NCU launch-section value) or ``None`` when NCU exposes
    none. Returns an agreement record; ``overall_agree`` is True only if at
    least one gated metric was comparable and every comparable gated metric
    matched exactly.
    """
    per_metric: list[dict] = []
    for name in LAUNCH_OCCUPANCY_LIMIT_METRICS:
        if name not in predictions:
            raise OccupancyAgreementError(
                f"prediction artifact is missing required limit {name!r}"
            )
        pred = predictions[name]
        predicted_value = pred["value"]
        predicted_status = pred["status"]
        ncu_value = ncu_values.get(name)

        if predicted_status != "available":
            # Honest exclusion: the producer admits it cannot compute this limit
            # statically (e.g. shared_mem without a runtime carveout). Not gated
            # away -- simply not comparable.
            agree: bool | None = None
            reason = pred.get("reason") or "prediction_typed_unavailable"
        elif ncu_value is None:
            agree = None
            reason = "ncu_launch_section_not_collected"
        else:
            agree = predicted_value == ncu_value
            reason = None if agree else "value_mismatch"

        per_metric.append(
            {
                "ncu_metric": name,
                "predicted_value": predicted_value,
                "predicted_status": predicted_status,
                "ncu_value": ncu_value,
                "gated": name in _GATED_METRICS,
                "agree": agree,
                "reason": reason,
            }
        )

    comparable = [
        m for m in per_metric if m["gated"] and m["agree"] is not None
    ]
    overall_agree = bool(comparable) and all(m["agree"] for m in comparable)
    return {
        "schema_version": _AGREEMENT_SCHEMA_VERSION,
        "gated_metrics": list(_GATED_METRICS),
        "comparable_gated_count": len(comparable),
        "overall_agree": overall_agree,
        "per_metric": per_metric,
    }


def _coerce_ncu_int(raw: str) -> int | None:
    """Parse one NCU CSV cell into an int, or None for an empty/absent cell.

    Strips US-locale thousands separators (NCU renders large counters as
    ``2,481.00``) before conversion. An empty cell is None (distinct from a
    real 0), so 'absent' is never confused with 'zero'. A non-numeric cell
    (NCU's units-row token ``block``, or an ``inf``/``nan`` rendering) raises
    ``ValueError``; the parser uses that to tell a units row from launch data.
    Occupancy values are integral in practice (NCU renders ``X.000000``).
    """
    text = (raw or "").strip().replace(",", "")
    if text == "":
        return None
    try:
        return int(round(float(text)))
    except (ValueError, OverflowError) as exc:  # 'block', 'inf', 'nan', ...
        raise ValueError(f"non-numeric NCU cell {raw!r}") from exc


def parse_ncu_launch_section_csv(csv_text: str) -> list[dict]:
    """Parse ``ncu --import <rep> --page raw --csv`` text into per-launch dicts.

    Returns one dict per data row, each mapping every launch__occupancy_limit_*
    name present as a column to its int value (or None for an empty cell).
    Metric names absent from the header are simply not keyed.
    """
    reader = csv.reader(io.StringIO(csv_text))
    rows = [r for r in reader if r]
    if not rows:
        raise OccupancyAgreementError("empty NCU CSV text")
    # The header is the first row whose cells include at least one of our metric
    # names (NCU may emit banner lines before the table).
    header_idx = None
    for i, row in enumerate(rows):
        if any(cell.strip() in LAUNCH_OCCUPANCY_LIMIT_METRICS for cell in row):
            header_idx = i
            break
    if header_idx is None:
        raise OccupancyAgreementError(
            "no launch__occupancy_limit_* column found in NCU CSV header"
        )
    header = [c.strip() for c in rows[header_idx]]
    col_of = {
        name: header.index(name)
        for name in LAUNCH_OCCUPANCY_LIMIT_METRICS
        if name in header
    }
    launches: list[dict] = []
    for row in rows[header_idx + 1 :]:
        cells = {
            name: (row[idx] if idx < len(row) else "")
            for name, idx in col_of.items()
        }
        if not any(c.strip() for c in cells.values()):
            continue  # blank in every occupancy column -> not a launch row
        parsed: dict[str, int | None] = {}
        numeric = 0
        nonnumeric = 0
        for name, cell in cells.items():
            try:
                parsed[name] = _coerce_ncu_int(cell)
            except ValueError:
                parsed[name] = None
                nonnumeric += 1
            else:
                if parsed[name] is not None:
                    numeric += 1
        if nonnumeric and numeric == 0:
            # NCU's --csv units row ("block", "register/thread", ...) right
            # after the metric-name header: every occupancy cell is non-numeric.
            # Not launch data -- skip rather than fabricate values.
            continue
        if nonnumeric:
            # A genuine launch row with a corrupt occupancy cell is NOT silently
            # dropped (which could shift which launch the caller compares); fail
            # closed so the caller sees the malformed input.
            raise OccupancyAgreementError(
                f"malformed NCU launch row (mixed numeric / non-numeric "
                f"occupancy cells): {cells}"
            )
        launches.append(parsed)
    return launches


def read_ncu_launch_section(rep_path: str | Path) -> list[dict]:
    """Read launch__occupancy_limit_* from an existing .ncu-rep (read-only).

    Invokes ``ncu --import <rep> --page raw --csv`` (the launch metrics live on
    the raw page, not the section-organized details page) and parses it. No
    profiling run, replay, or clock control occurs.
    """
    cmd = ["ncu", "--import", str(rep_path), "--page", "raw", "--csv"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise OccupancyAgreementError("ncu not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise OccupancyAgreementError(
            f"ncu --import failed (rc={exc.returncode}): {exc.stderr.strip()[:200]}"
        ) from exc
    return parse_ncu_launch_section_csv(proc.stdout)


def build_agreement_artifact(
    prediction_artifact: dict,
    ncu_values: dict,
    *,
    ncu_rep: str | None = None,
    launch_index: int = 0,
) -> dict:
    """Wrap ``compare_occupancy_agreement`` with the cell identity for auditing."""
    comparison = compare_occupancy_agreement(
        prediction_artifact["predictions"], ncu_values
    )
    artifact = {
        "schema_version": _AGREEMENT_SCHEMA_VERSION,
        "artifact_kind": "static_occupancy_agreement",
        "arch": prediction_artifact.get("arch"),
        "kernel": prediction_artifact.get("kernel"),
        "launch_index": launch_index,
        "ncu_rep": ncu_rep,
        "overall_agree": comparison["overall_agree"],
        "comparable_gated_count": comparison["comparable_gated_count"],
        "gated_metrics": comparison["gated_metrics"],
        "per_metric": comparison["per_metric"],
    }
    for key in ("slug", "shape_point", "binary_sha256", "kernel_signature_sha256"):
        if key in prediction_artifact:
            artifact[key] = prediction_artifact[key]
    return artifact


def emit(artifact: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return out_path


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prediction", required=True, help="prediction artifact JSON")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ncu-rep", help="committed .ncu-rep to import (read-only)")
    src.add_argument("--ncu-csv-file", help="pre-captured ncu --page raw --csv text")
    ap.add_argument("--launch-index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    if args.ncu_rep is not None:
        launches = read_ncu_launch_section(args.ncu_rep)
    else:
        launches = parse_ncu_launch_section_csv(
            Path(args.ncu_csv_file).read_text(encoding="utf-8")
        )
    if not launches:
        raise OccupancyAgreementError("no launches in NCU CSV")
    if args.launch_index >= len(launches):
        raise OccupancyAgreementError(
            f"launch_index {args.launch_index} out of range ({len(launches)} launches)"
        )
    ncu_values = launches[args.launch_index]
    artifact = build_agreement_artifact(
        prediction, ncu_values, ncu_rep=args.ncu_rep, launch_index=args.launch_index
    )
    written = emit(artifact, Path(args.out))
    print(
        f"wrote {written} (overall_agree={artifact['overall_agree']}, "
        f"comparable_gated={artifact['comparable_gated_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
