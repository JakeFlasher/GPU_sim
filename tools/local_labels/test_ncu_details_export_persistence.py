from __future__ import annotations

import copy
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import metric_registry
from tools.local_labels import ncu_action_executor
from tools.local_labels import ncu_details_export_persistence as persistence
from tools.local_labels import ncu_details_export_receipt
from tools.local_labels import ncu_launch_documents
from tools.local_labels import parse_ncu_metrics


_REPORT_BYTES = b"synthetic retained NCU report"
_CSV_BYTES = (
    b'"Metric Value","Stream","ID","Grid Size","Metric Name","Context",'
    b'"Device","Block Size","Process ID","Kernel Name"\n'
    b'"88","2","9","(4, 1, 1)","lts__t_sector_hit_rate.pct",'
    b'"1","0","(128, 1, 1)","100","_Z7kernelAv"\n'
    b'"123","2","9","(4, 1, 1)","gpc__cycles_elapsed.max",'
    b'"1","0","(128, 1, 1)","100","_Z7kernelAv"\n'
)
_RECEIPT_RELATIVE = "raw/run_0_launch_set.details_export_receipt.json"


def _identity() -> ncu_action_executor.ExpectedNcuIdentity:
    return ncu_action_executor.build_expected_ncu_identity(
        resolved_path="/opt/nvidia/ncu",
        executable_sha256="a" * 64,
        version_line="Version 2026.1.0.0 (build 1) (public-release)",
    )


def _protocol() -> dict[str, object]:
    return {
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_replay_mode": "kernel",
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": 0,
        "ncu_launch_count": 2,
    }


def _prepare(tmp_path: Path):
    root = tmp_path / "artifact-root"
    raw = root / "raw"
    raw.mkdir(parents=True)
    report = raw / "run_0_launch_set.ncu-rep"
    report.write_bytes(_REPORT_BYTES)
    identity = _identity()
    imported = parse_ncu_metrics.ImportedReportCsv(
        text=_CSV_BYTES.decode("utf-8"),
        report_sha256=immutable.sha256_bytes(_REPORT_BYTES),
        report_size_bytes=len(_REPORT_BYTES),
        csv_bytes=_CSV_BYTES,
        stderr_bytes=b"",
        normalized_arguments=ncu_details_export_receipt.EXPECTED_IMPORT_ARGUMENTS,
        action_environment=(
            ncu_details_export_receipt.EXPECTED_ACTION_ENVIRONMENT
        ),
        working_directory="/",
        exit_status=0,
        ncu_identity=identity,
    )
    products = ncu_launch_documents.build_details_export_products(
        imported=imported,
        preflight_sha256="0" * 64,
        source_commit="1" * 40,
        capture_identity_sha256="2" * 64,
        expected_ncu_identity=identity,
        report_relative_path="raw/run_0_launch_set.ncu-rep",
        csv_relative_path="raw/run_0_launch_set.details.csv",
        stderr_relative_path="raw/run_0_launch_set.details.stderr",
        inventory_relative_path="raw/run_0_launch_set.launch_inventory.json",
        metrics_relative_path="raw/run_0_launch_set.launch_metrics.json",
        expected_launch_count=1,
        protocol=_protocol(),
        registry=metric_registry.load_registry(),
        percentage_metric_names=("lts__t_sector_hit_rate.pct",),
        unit_metric_names=("gpc__cycles_elapsed.max",),
    )
    return root, raw, report, products


def _persistence_authority() -> dict[str, object]:
    return {
        "expected_preflight_sha256": "0" * 64,
        "expected_source_commit": "1" * 40,
        "expected_capture_identity_sha256": "2" * 64,
        "expected_ncu_identity": _identity(),
        "expected_launch_count": 1,
        "protocol": _protocol(),
        "registry": metric_registry.load_registry(),
        "percentage_metric_names": ("lts__t_sector_hit_rate.pct",),
        "unit_metric_names": ("gpc__cycles_elapsed.max",),
    }


def _persist(root: Path, products):
    return persistence.persist_details_export_products(
        artifact_root=root,
        receipt_relative_path=_RECEIPT_RELATIVE,
        products=products,
        **_persistence_authority(),
    )


def test_persistence_publishes_read_only_products_and_empty_stderr_last(
    tmp_path, monkeypatch,
):
    root, raw, report, products = _prepare(tmp_path)
    published_names: list[str] = []
    real_publish = persistence.immutable.publish_regular_file

    def unexpected_hardware_action(*_args, **_kwargs):
        pytest.fail("offline Details persistence reached NCU execution")

    monkeypatch.setattr(
        persistence.ncu_action_executor,
        "execute_ncu_action",
        unexpected_hardware_action,
    )

    def observe(parent, name, payload, **kwargs):
        published_names.append(name)
        return real_publish(parent, name, payload, **kwargs)

    monkeypatch.setattr(
        persistence.immutable, "publish_regular_file", observe,
    )
    receipt_path = _persist(root, products)

    assert receipt_path == raw / "run_0_launch_set.details_export_receipt.json"
    assert published_names == [
        "run_0_launch_set.details.csv",
        "run_0_launch_set.details.stderr",
        "run_0_launch_set.launch_inventory.json",
        "run_0_launch_set.launch_metrics.json",
        "run_0_launch_set.details_export_receipt.json",
    ]
    expected = {
        "run_0_launch_set.details.csv": products.csv_bytes,
        "run_0_launch_set.details.stderr": b"",
        "run_0_launch_set.launch_inventory.json": products.inventory_bytes,
        "run_0_launch_set.launch_metrics.json": products.metrics_bytes,
        "run_0_launch_set.details_export_receipt.json": products.receipt_bytes,
    }
    for name, payload in expected.items():
        path = raw / name
        assert path.read_bytes() == payload
        assert path.stat().st_nlink == 1
        assert path.stat().st_mode & 0o222 == 0
    assert report.read_bytes() == _REPORT_BYTES


def test_report_identity_is_verified_without_buffering_report_bytes(
    tmp_path, monkeypatch,
):
    root, _raw, report, products = _prepare(tmp_path)
    report_identity = (report.stat().st_dev, report.stat().st_ino)
    real_pread = persistence.os.pread

    def reject_report_read(descriptor, size, offset):
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == report_identity:
            pytest.fail("persistence buffered the retained NCU report")
        return real_pread(descriptor, size, offset)

    monkeypatch.setattr(persistence.os, "pread", reject_report_read)
    _persist(root, products)


def test_report_drift_refuses_before_first_output(tmp_path):
    root, raw, report, products = _prepare(tmp_path)
    report.write_bytes(b"replacement report")

    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="identity differs from receipt",
    ):
        _persist(root, products)
    assert not (raw / "run_0_launch_set.details.csv").exists()
    assert not (raw / "run_0_launch_set.details_export_receipt.json").exists()


def test_existing_destination_is_unchanged_and_receipt_is_absent(tmp_path):
    root, raw, _report, products = _prepare(tmp_path)
    destination = raw / "run_0_launch_set.details.csv"
    destination.write_bytes(b"unique existing evidence")

    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="publication already exists",
    ):
        _persist(root, products)
    assert destination.read_bytes() == b"unique existing evidence"
    assert not (raw / "run_0_launch_set.details_export_receipt.json").exists()


def test_mismatched_partial_is_preserved(tmp_path):
    root, raw, _report, products = _prepare(tmp_path)
    partial = raw / ".run_0_launch_set.details.csv.ncu-details-partial"
    partial.write_bytes(b"incomplete unique evidence")

    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="partial is preserved",
    ):
        _persist(root, products)
    assert partial.read_bytes() == b"incomplete unique evidence"
    assert not (raw / "run_0_launch_set.details.csv").exists()


def test_raced_partial_fifo_fails_closed_without_blocking(tmp_path, monkeypatch):
    root, raw, _report, products = _prepare(tmp_path)
    partial = raw / ".run_0_launch_set.details.csv.ncu-details-partial"
    partial.write_bytes(products.csv_bytes)
    partial.chmod(0o444)
    real_entry_mode = persistence.immutable._entry_mode
    raced = False

    def replace_partial_after_stat(parent_descriptor, name):
        nonlocal raced
        mode = real_entry_mode(parent_descriptor, name)
        if name == partial.name and mode is not None and not raced:
            raced = True
            partial.unlink()
            os.mkfifo(partial, 0o400)
        return mode

    monkeypatch.setattr(
        persistence.immutable, "_entry_mode", replace_partial_after_stat,
    )
    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="unrecognized partial stage entry",
    ):
        _persist(root, products)
    assert raced
    assert partial.exists()
    assert not (raw / "run_0_launch_set.details.csv").exists()
    assert not (raw / "run_0_launch_set.details_export_receipt.json").exists()


@pytest.mark.parametrize("kind", ["leaf_symlink", "ancestor_symlink"])
def test_symlinks_are_refused_without_touching_targets(tmp_path, kind):
    if kind == "leaf_symlink":
        root, raw, report, products = _prepare(tmp_path)
        target = tmp_path / "report-target"
        report.rename(target)
        report.symlink_to(target)
    else:
        root = tmp_path / "artifact-root"
        root.mkdir()
        target_parent = tmp_path / "target-raw"
        target_parent.mkdir()
        target = target_parent / "run_0_launch_set.ncu-rep"
        target.write_bytes(_REPORT_BYTES)
        (root / "raw").symlink_to(target_parent, target_is_directory=True)
        identity = _identity()
        imported = parse_ncu_metrics.ImportedReportCsv(
            text=_CSV_BYTES.decode(),
            report_sha256=immutable.sha256_bytes(_REPORT_BYTES),
            report_size_bytes=len(_REPORT_BYTES),
            csv_bytes=_CSV_BYTES,
            stderr_bytes=b"",
            normalized_arguments=(
                ncu_details_export_receipt.EXPECTED_IMPORT_ARGUMENTS
            ),
            action_environment=(
                ncu_details_export_receipt.EXPECTED_ACTION_ENVIRONMENT
            ),
            working_directory="/",
            exit_status=0,
            ncu_identity=identity,
        )
        products = ncu_launch_documents.build_details_export_products(
            imported=imported,
            preflight_sha256="0" * 64,
            source_commit="1" * 40,
            capture_identity_sha256="2" * 64,
            expected_ncu_identity=identity,
            report_relative_path="raw/run_0_launch_set.ncu-rep",
            csv_relative_path="raw/run_0_launch_set.details.csv",
            stderr_relative_path="raw/run_0_launch_set.details.stderr",
            inventory_relative_path="raw/run_0_launch_set.launch_inventory.json",
            metrics_relative_path="raw/run_0_launch_set.launch_metrics.json",
            expected_launch_count=1,
            protocol=_protocol(),
            registry=metric_registry.load_registry(),
            percentage_metric_names=("lts__t_sector_hit_rate.pct",),
            unit_metric_names=("gpc__cycles_elapsed.max",),
        )

    before = target.read_bytes()
    with pytest.raises(persistence.DetailsExportPersistenceError):
        _persist(root, products)
    assert target.read_bytes() == before
    assert not (target.parent / "run_0_launch_set.details.csv").exists()


def test_hard_linked_report_is_refused(tmp_path):
    root, raw, report, products = _prepare(tmp_path)
    os.link(report, tmp_path / "second-report-link")

    with pytest.raises(
        persistence.DetailsExportPersistenceError, match="hard-linked",
    ):
        _persist(root, products)
    assert not (raw / "run_0_launch_set.details.csv").exists()


def test_injected_middle_failure_preserves_prefix_without_receipt(
    tmp_path, monkeypatch,
):
    root, raw, _report, products = _prepare(tmp_path)
    real_publish = persistence.immutable.publish_regular_file

    def fail_inventory(parent, name, payload, **kwargs):
        if name == "run_0_launch_set.launch_inventory.json":
            raise immutable.ImmutableArtifactError("synthetic middle failure")
        return real_publish(parent, name, payload, **kwargs)

    monkeypatch.setattr(
        persistence.immutable, "publish_regular_file", fail_inventory,
    )
    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="synthetic middle failure",
    ):
        _persist(root, products)
    assert (raw / "run_0_launch_set.details.csv").is_file()
    assert (raw / "run_0_launch_set.details.stderr").is_file()
    assert not (raw / "run_0_launch_set.launch_inventory.json").exists()
    assert not (raw / "run_0_launch_set.details_export_receipt.json").exists()


def test_post_publish_substitution_is_detected_before_receipt(
    tmp_path, monkeypatch,
):
    root, raw, _report, products = _prepare(tmp_path)
    real_open = persistence.immutable.open_retained_regular
    target = raw / "run_0_launch_set.launch_metrics.json"

    def substitute(path):
        if path == target:
            path.chmod(0o600)
            path.write_bytes(b"replacement metrics")
        return real_open(path)

    monkeypatch.setattr(
        persistence.immutable, "open_retained_regular", substitute,
    )
    with pytest.raises(persistence.DetailsExportPersistenceError):
        _persist(root, products)
    assert target.read_bytes() == b"replacement metrics"
    assert not (raw / "run_0_launch_set.details_export_receipt.json").exists()


def test_generated_hardlink_is_detected_before_receipt(tmp_path, monkeypatch):
    root, raw, _report, products = _prepare(tmp_path)
    real_publish = persistence.immutable.publish_regular_file
    link = tmp_path / "hostile-hardlink"

    def link_after_publish(parent, name, payload, **kwargs):
        path = real_publish(parent, name, payload, **kwargs)
        if name == "run_0_launch_set.details.csv":
            os.link(path, link)
        return path

    monkeypatch.setattr(
        persistence.immutable, "publish_regular_file", link_after_publish,
    )
    with pytest.raises(
        persistence.DetailsExportPersistenceError, match="hard-linked",
    ):
        _persist(root, products)
    assert link.is_file()
    assert not (raw / "run_0_launch_set.details_export_receipt.json").exists()


def test_self_resealed_non_authoritative_receipt_is_refused(tmp_path):
    root, raw, _report, products = _prepare(tmp_path)
    document = copy.deepcopy(products.receipt_document)
    document["import_arguments"].append("--forged")
    document["import_arguments_sha256"] = immutable.sha256_bytes(
        immutable.canonical_json_bytes(document["import_arguments"])
    )
    unsigned = dict(document)
    unsigned.pop("receipt_seal_sha256")
    document = immutable.seal_model(unsigned, "receipt_seal_sha256")
    products = replace(
        products,
        receipt_document=document,
        receipt_bytes=immutable.pretty_json_bytes(document),
    )

    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="semantic contract is invalid",
    ):
        _persist(root, products)
    assert not (raw / "run_0_launch_set.details.csv").exists()


def test_self_resealed_metrics_forgery_is_refused(tmp_path):
    root, raw, _report, products = _prepare(tmp_path)
    metrics = copy.deepcopy(products.metrics_document)
    metrics[0]["unit_tagged_metrics"]["gpc__cycles_elapsed.max"][
        "value"
    ] = 999999.0
    metrics_bytes = immutable.pretty_json_bytes(metrics)
    receipt = copy.deepcopy(products.receipt_document)
    receipt["products"][4]["sha256"] = immutable.sha256_bytes(metrics_bytes)
    receipt["products"][4]["size_bytes"] = len(metrics_bytes)
    unsigned = dict(receipt)
    unsigned.pop("receipt_seal_sha256")
    receipt = immutable.seal_model(unsigned, "receipt_seal_sha256")
    products = replace(
        products,
        metrics_document=metrics,
        metrics_bytes=metrics_bytes,
        receipt_document=receipt,
        receipt_bytes=immutable.pretty_json_bytes(receipt),
    )

    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="launch semantics are invalid",
    ):
        _persist(root, products)
    assert not (raw / "run_0_launch_set.details.csv").exists()


def test_cyclic_model_is_translated_to_persistence_error(tmp_path):
    root, raw, _report, products = _prepare(tmp_path)
    cyclic: dict[str, object] = {}
    cyclic["cycle"] = cyclic
    products = replace(products, inventory_document=cyclic)

    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="cyclic or excessively nested",
    ):
        _persist(root, products)
    assert not (raw / "run_0_launch_set.details.csv").exists()


def test_contract_refuses_noncanonical_root_and_wrong_fixed_names(tmp_path):
    root, _raw, _report, products = _prepare(tmp_path)
    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="canonical absolute path",
    ):
        persistence.persist_details_export_products(
            artifact_root=Path("relative"),
            receipt_relative_path=_RECEIPT_RELATIVE,
            products=products,
            **_persistence_authority(),
        )
    with pytest.raises(
        persistence.DetailsExportPersistenceError,
        match="receipt basename differs",
    ):
        persistence.persist_details_export_products(
            artifact_root=root,
            receipt_relative_path="raw/forged.json",
            products=products,
            **_persistence_authority(),
        )
