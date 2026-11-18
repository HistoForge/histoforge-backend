from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import uuid
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.datasets import service as service_module
from histoforge.cloud.training.datasets.control import TrainingDatasetControl
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.datasets.service import TrainingDatasetService
from histoforge.cloud.training.datasets.state import TrainingDatasetState
from tests.test_training_contract import training_capability
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import _dataset, _object_feature, _rectangle
from tests.test_training_dataset_state import (
    DATASET_ID,
    SECOND_DATASET_ID,
    service_limits,
)


class _Control(TrainingDatasetControl):
    def __init__(
        self,
        state: TrainingDatasetState,
        files: TrainingDatasetFiles,
        api_limits: Any,
    ) -> None:
        super().__init__(state, files, api_limits)
        self.state = state
        self.fail_next_seal_commit = False

    def commit_seal(
        self,
        dataset_id: str,
        start_digest: str,
        claim_token: str,
        manifest: Any,
        retained_bytes: int,
        task_display_name: str,
    ) -> dict[str, Any]:
        if self.fail_next_seal_commit:
            self.fail_next_seal_commit = False
            raise GatewayFailure("SERVICE_UNAVAILABLE", "injected commit failure")
        return super().commit_seal(
            dataset_id,
            start_digest,
            claim_token,
            manifest,
            retained_bytes,
            task_display_name,
        )


def test_service_upload_seal_reuse_list_and_delete(tmp_path: Path) -> None:
    service, control, files = _service(tmp_path)
    archive, manifest = _archive(tmp_path)
    request = _start_request(archive, manifest["dataset_digest"], chunk_size=211)

    accepted = service.start(request)
    assert accepted["type"] == "upload"
    for index, body in _chunks(archive, request["chunk_size_bytes"]):
        receipt = service.put_chunk(
            DATASET_ID, index, body, hashlib.sha256(body).hexdigest()
        )
        assert receipt["chunk"]["chunk_index"] == index
    replay = service.put_chunk(
        DATASET_ID,
        0,
        archive[: request["chunk_size_bytes"]],
        hashlib.sha256(archive[: request["chunk_size_bytes"]]).hexdigest(),
    )
    assert replay["chunk"]["chunk_index"] == 0

    sealed = service.seal(
        DATASET_ID,
        {
            "schema_version": 1,
            "start_request_digest": accepted["start_request_digest"],
        },
    )
    assert sealed["type"] == "retained"
    assert sealed["dataset_digest"] == manifest["dataset_digest"]
    assert service.start(request) == {"type": "existing", "dataset": sealed}
    assert service.get(DATASET_ID)["record"] == sealed
    assert service.page(10)["items"][0]["record"] == sealed

    service.delete(DATASET_ID)
    assert service.page(10) == {"items": []}
    assert files.dataset_remaining_bytes(DATASET_ID) == 0


def test_dataset_delete_failure_remains_visible_and_retries_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _control, files = _service(tmp_path)
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    service.seal(
        DATASET_ID,
        {
            "schema_version": 1,
            "start_request_digest": accepted["start_request_digest"],
        },
    )
    monkeypatch.setattr(
        files,
        "delete_dataset",
        lambda _dataset_id: (_ for _ in ()).throw(OSError("injected delete loss")),
    )

    with pytest.raises(GatewayFailure) as raised:
        service.delete(DATASET_ID)
    assert raised.value.error["code"] == "SERVICE_UNAVAILABLE"

    restarted, _restarted_control, restarted_files = _service(tmp_path)
    visible = restarted.get(DATASET_ID)["record"]
    assert visible["type"] == "cleanup-required"
    assert visible["last_error"]["code"] == "SERVICE_UNAVAILABLE"
    restarted.delete(DATASET_ID)
    assert restarted.page(10) == {"items": []}
    assert restarted_files.dataset_remaining_bytes(DATASET_ID) == 0


def test_dataset_delete_resumes_after_process_loss_during_byte_cleanup(
    tmp_path: Path,
) -> None:
    service, control, files = _service(tmp_path)
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    service.seal(
        DATASET_ID,
        {
            "schema_version": 1,
            "start_request_digest": accepted["start_request_digest"],
        },
    )
    upload_root = tmp_path / "volume" / "training" / "uploads" / DATASET_ID / "leftover"
    upload_root.mkdir(parents=True)
    (upload_root / "bytes.bin").write_bytes(b"leftover upload bytes")

    cleanup = control.begin_delete(DATASET_ID)
    assert cleanup is not None
    assert cleanup["type"] == "cleanup-required"
    assert "last_error" not in cleanup
    files.delete_dataset(DATASET_ID)

    restarted, _restarted_control, restarted_files = _service(tmp_path)
    visible = restarted.get(DATASET_ID)["record"]
    assert visible == cleanup
    assert restarted_files.dataset_remaining_bytes(DATASET_ID) == 0
    assert restarted_files.upload_remaining_bytes(DATASET_ID) > 0

    restarted.delete(DATASET_ID)
    assert restarted.page(10) == {"items": []}
    assert restarted_files.dataset_remaining_bytes(DATASET_ID) == 0
    assert restarted_files.upload_remaining_bytes(DATASET_ID) == 0


def test_java_project_package_uploads_and_seals_against_backend_contract(
    tmp_path: Path,
) -> None:
    encoded = (
        Path(__file__).parent
        / "fixtures"
        / "contracts"
        / "project-training-dataset-v1.zip.b64"
    ).read_bytes()
    archive = base64.b64decode(encoded)
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        assert package.testzip() is None
        manifest = json.loads(package.read("dataset.json"))

    service, _control, _files = _service(tmp_path)
    request = _start_request(
        archive,
        manifest["dataset_digest"],
        chunk_size=len(archive),
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    sealed = service.seal(
        DATASET_ID,
        {
            "schema_version": 1,
            "start_request_digest": accepted["start_request_digest"],
        },
    )

    assert sealed["dataset_digest"] == manifest["dataset_digest"]
    assert sealed["task_display_name"] == "Leydig cell segmentation"
    assert sealed["image_count"] == 1
    assert sealed["roi_count"] == 1


def test_invalid_archive_never_becomes_a_dataset(tmp_path: Path) -> None:
    service, control, files = _service(tmp_path)
    archive = b"not a zip"
    request = _start_request(archive, "1" * 64, chunk_size=len(archive))
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())

    with pytest.raises(GatewayFailure) as raised:
        service.seal(
            DATASET_ID,
            {
                "schema_version": 1,
                "start_request_digest": accepted["start_request_digest"],
            },
        )
    assert raised.value.error["code"] == "VALIDATION_ERROR"
    assert control.state.upload_records()[0]["dataset_id"] == DATASET_ID
    assert files.dataset_remaining_bytes(DATASET_ID) == 0


def test_seal_rejects_declared_dataset_digest_before_publication(
    tmp_path: Path,
) -> None:
    service, control, files = _service(tmp_path)
    archive, _manifest = _archive(tmp_path)
    request = _start_request(archive, "f" * 64, chunk_size=len(archive))
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())

    with pytest.raises(GatewayFailure) as raised:
        service.seal(
            DATASET_ID,
            {
                "schema_version": 1,
                "start_request_digest": accepted["start_request_digest"],
            },
        )

    assert raised.value.error["code"] == "VALIDATION_ERROR"
    assert files.dataset_remaining_bytes(DATASET_ID) == 0
    assert control.state.upload_records()[0]["dataset_id"] == DATASET_ID


def test_seal_identifies_roi_and_pixel_when_training_objects_overlap(
    tmp_path: Path,
) -> None:
    service, _control, files = _service(tmp_path)
    archive, manifest = _archive(
        tmp_path,
        objects=[
            _object_feature(
                "11111111-1111-4111-8111-111111111111", _rectangle(12, 12, 20, 20)
            ),
            _object_feature(
                "22222222-2222-4222-8222-222222222222", _rectangle(16, 16, 24, 24)
            ),
        ],
    )
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())

    with pytest.raises(GatewayFailure) as raised:
        service.seal(
            DATASET_ID,
            {
                "schema_version": 1,
                "start_request_digest": accepted["start_request_digest"],
            },
        )
    error = raised.value.error
    assert error["code"] == "VALIDATION_ERROR"
    assert error["resource_id"] == "rois/0/roi.geojson"
    assert "Two training objects overlap near x=16.50, y=16.50" in error["message"]
    assert "click Refresh in Train" in error["message"]
    assert files.dataset_remaining_bytes(DATASET_ID) == 0


def test_seal_retry_finishes_after_publication_preceded_control_failure(
    tmp_path: Path,
) -> None:
    service, control, files = _service(tmp_path)
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    seal = {
        "schema_version": 1,
        "start_request_digest": accepted["start_request_digest"],
    }
    control.fail_next_seal_commit = True

    with pytest.raises(GatewayFailure, match="injected"):
        service.seal(DATASET_ID, seal)
    assert files.dataset_remaining_bytes(DATASET_ID) > 0
    assert control.state.upload_records()[0]["dataset_id"] == DATASET_ID

    sealed = service.seal(DATASET_ID, seal)
    assert sealed["dataset_digest"] == manifest["dataset_digest"]
    assert control.state.upload_records() == []


def test_seal_commit_survives_control_owner_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, control, files = _service(tmp_path)
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    limits = training_api_limits()
    private_limits = service_limits()

    def commit_with_reconstructed_owner(
        dataset_id: str,
        start_digest: str,
        claim_token: str,
        sealed_manifest: Any,
        retained_bytes: int,
        task_display_name: str,
    ) -> dict[str, Any]:
        records = TrainingControlRecords(tmp_path / "volume", limits, private_limits)
        state = TrainingDatasetState(records, limits, private_limits)
        reconstructed = TrainingDatasetControl(state, files, limits)
        return reconstructed.commit_seal(
            dataset_id,
            start_digest,
            claim_token,
            sealed_manifest,
            retained_bytes,
            task_display_name,
        )

    monkeypatch.setattr(control, "commit_seal", commit_with_reconstructed_owner)
    sealed = service.seal(
        DATASET_ID,
        {
            "schema_version": 1,
            "start_request_digest": accepted["start_request_digest"],
        },
    )

    assert sealed["dataset_digest"] == manifest["dataset_digest"]
    assert (
        TrainingControlRecords(tmp_path / "volume", limits, private_limits).get_dataset(
            DATASET_ID
        )
        == sealed
    )
    files.require_dataset(DATASET_ID, manifest, limits)


def test_dataset_reuse_removes_chunks_left_after_successful_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, control, files = _service(tmp_path)
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    delete_upload = files.delete_upload
    monkeypatch.setattr(
        files,
        "delete_upload",
        lambda _dataset_id: (_ for _ in ()).throw(OSError("injected cleanup loss")),
    )

    with pytest.raises(GatewayFailure) as raised:
        service.seal(
            DATASET_ID,
            {
                "schema_version": 1,
                "start_request_digest": accepted["start_request_digest"],
            },
        )
    assert raised.value.error["code"] == "SERVICE_UNAVAILABLE"
    assert control.state.upload_records() == []
    assert files.upload_remaining_bytes(DATASET_ID) > 0

    monkeypatch.setattr(files, "delete_upload", delete_upload)
    assert service.start(request)["type"] == "existing"
    assert files.upload_remaining_bytes(DATASET_ID) == 0


def test_unrelated_start_reclaims_upload_root_left_after_successful_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, control, files = _service(
        tmp_path, dataset_ids=(DATASET_ID, SECOND_DATASET_ID)
    )
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    delete_upload = files.delete_upload
    monkeypatch.setattr(
        files,
        "delete_upload",
        lambda _dataset_id: (_ for _ in ()).throw(OSError("injected cleanup loss")),
    )

    with pytest.raises(GatewayFailure) as raised:
        service.seal(
            DATASET_ID,
            {
                "schema_version": 1,
                "start_request_digest": accepted["start_request_digest"],
            },
        )
    assert raised.value.error["code"] == "SERVICE_UNAVAILABLE"
    assert control.state.upload_records() == []
    assert files.upload_remaining_bytes(DATASET_ID) > 0

    monkeypatch.setattr(files, "delete_upload", delete_upload)
    other = _start_request(archive, "f" * 64, chunk_size=len(archive))
    assert service.start(other)["dataset_id"] == SECOND_DATASET_ID
    assert files.upload_remaining_bytes(DATASET_ID) == 0


def test_unowned_cleanup_never_deletes_a_live_upload(tmp_path: Path) -> None:
    service, _control, files = _service(
        tmp_path, dataset_ids=(DATASET_ID, SECOND_DATASET_ID)
    )
    body = b"abcd"
    first = _start_request(body, "1" * 64, chunk_size=len(body))
    service.start(first)
    service.put_chunk(DATASET_ID, 0, body, hashlib.sha256(body).hexdigest())

    second = _start_request(body, "2" * 64, chunk_size=len(body))
    assert service.start(second)["dataset_id"] == SECOND_DATASET_ID
    assert files.upload_remaining_bytes(DATASET_ID) > 0


def test_durable_chunk_survives_worker_loss_before_lane_commit(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)]
    volume = tmp_path / "volume"
    limits = training_api_limits()
    private_limits = service_limits(dataset_upload_idle_timeout_seconds=10)
    records = TrainingControlRecords(volume, limits, private_limits)
    state = TrainingDatasetState(
        records,
        limits,
        private_limits,
        clock=lambda: now[0],
        id_factory=lambda: uuid.UUID(DATASET_ID),
    )
    files = TrainingDatasetFiles(volume)
    control = _Control(state, files, limits)
    body = b"abcd"
    start = _start_request(body, "1" * 64, chunk_size=len(body))
    control.reserve(start)

    now[0] += timedelta(seconds=9)
    claim = control.claim_chunk(
        DATASET_ID, 0, len(body), hashlib.sha256(body).hexdigest()
    )
    files.publish_chunk(DATASET_ID, body, claim["commit"])

    now[0] += timedelta(seconds=2)
    assert control.claim_expired_upload() is None
    replay = control.claim_chunk(
        DATASET_ID, 0, len(body), hashlib.sha256(body).hexdigest()
    )
    assert replay["accepted"] == claim["commit"]


def test_chunk_failure_before_pair_publication_leaves_a_clean_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, control, files = _service(tmp_path)
    body = b"abcd"
    request = _start_request(body, "1" * 64, chunk_size=len(body))
    service.start(request)
    write_file = files._write_file

    def fail_before_receipt(path: Path, value: bytes) -> None:
        if path.name == "receipt.json":
            raise OSError("simulated publication interruption")
        write_file(path, value)

    monkeypatch.setattr(files, "_write_file", fail_before_receipt)
    with pytest.raises(OSError, match="simulated publication interruption"):
        service.put_chunk(DATASET_ID, 0, body, hashlib.sha256(body).hexdigest())

    assert files.upload_remaining_bytes(DATASET_ID) == 0
    assert control.state.upload_records()[0]["dataset_id"] == DATASET_ID

    monkeypatch.setattr(files, "_write_file", write_file)
    receipt = service.put_chunk(DATASET_ID, 0, body, hashlib.sha256(body).hexdigest())
    assert files.require_chunk(DATASET_ID, receipt["chunk"]) == receipt["chunk"]


def test_expired_upload_cleanup_reclaims_partial_cleanup_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)]
    volume = tmp_path / "volume"
    limits = training_api_limits()
    private_limits = service_limits(dataset_upload_idle_timeout_seconds=10)
    records = TrainingControlRecords(volume, limits, private_limits)
    state = TrainingDatasetState(
        records,
        limits,
        private_limits,
        clock=lambda: now[0],
        id_factory=lambda: uuid.UUID(DATASET_ID),
    )
    files = TrainingDatasetFiles(volume)
    control = _Control(state, files, limits)
    service = TrainingDatasetService(
        files,
        control,
        limits,
        private_limits,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        lambda _capability_id: None,
    )
    body = b"abcd"
    service.start(_start_request(body, "1" * 64, chunk_size=len(body)))
    service.put_chunk(DATASET_ID, 0, body, hashlib.sha256(body).hexdigest())
    unpublished = volume / "training" / "datasets" / DATASET_ID / "orphan.bin"
    unpublished.parent.mkdir(parents=True)
    unpublished.write_bytes(b"unpublished validated bytes")
    now[0] += timedelta(seconds=10)

    monkeypatch.setattr(
        files,
        "delete_dataset",
        lambda _dataset_id: (_ for _ in ()).throw(
            OSError("simulated cleanup interruption")
        ),
    )
    with pytest.raises(OSError, match="simulated cleanup interruption"):
        service.cleanup_stale_artifacts()

    assert files.upload_remaining_bytes(DATASET_ID) == 0
    assert files.dataset_remaining_bytes(DATASET_ID) > 0
    assert control.state.upload_records()[0]["dataset_id"] == DATASET_ID

    restarted_records = TrainingControlRecords(volume, limits, private_limits)
    restarted_state = TrainingDatasetState(
        restarted_records,
        limits,
        private_limits,
        clock=lambda: now[0],
    )
    restarted_files = TrainingDatasetFiles(volume)
    restarted_control = _Control(restarted_state, restarted_files, limits)
    restarted = TrainingDatasetService(
        restarted_files,
        restarted_control,
        limits,
        private_limits,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        lambda _capability_id: None,
    )
    restarted.cleanup_stale_artifacts()
    assert restarted_control.state.upload_records() == []
    assert restarted_files.dataset_remaining_bytes(DATASET_ID) == 0


def test_late_chunk_after_expiry_is_discarded_without_poisoning_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)]
    volume = tmp_path / "volume"
    limits = training_api_limits()
    private_limits = service_limits(dataset_upload_idle_timeout_seconds=10)
    generated_ids = iter((DATASET_ID, SECOND_DATASET_ID))
    records = TrainingControlRecords(volume, limits, private_limits)
    state = TrainingDatasetState(
        records,
        limits,
        private_limits,
        clock=lambda: now[0],
        id_factory=lambda: uuid.UUID(next(generated_ids)),
    )
    files = TrainingDatasetFiles(volume)
    control = _Control(state, files, limits)
    service = TrainingDatasetService(
        files,
        control,
        limits,
        private_limits,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        lambda _capability_id: None,
    )
    body = b"abcd"
    first = _start_request(body, "1" * 64, chunk_size=len(body))
    service.start(first)
    publish_chunk = files.publish_chunk

    def publish_after_expiry_cleanup(dataset_id, value, commit):
        now[0] += timedelta(seconds=private_limits["dataset_writer_claim_seconds"] + 1)
        expired = control.claim_expired_upload()
        assert expired is not None
        files.delete_upload(dataset_id)
        files.delete_dataset(dataset_id)
        control.finish_expired_upload_cleanup(
            dataset_id, expired["start_request_digest"]
        )
        return publish_chunk(dataset_id, value, commit)

    monkeypatch.setattr(files, "publish_chunk", publish_after_expiry_cleanup)

    with pytest.raises(GatewayFailure) as raised:
        service.put_chunk(DATASET_ID, 0, body, hashlib.sha256(body).hexdigest())

    assert raised.value.error["code"] == "TRAINING_UPLOAD_NOT_FOUND"
    assert files.upload_remaining_bytes(DATASET_ID) == 0
    second = _start_request(body, "3" * 64, chunk_size=len(body))
    assert service.start(second)["dataset_id"] == SECOND_DATASET_ID


def test_process_seal_gate_rejects_parallel_heavy_work_and_releases_claim(
    tmp_path: Path,
) -> None:
    service, control, _ = _service(tmp_path)
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    seal = {
        "schema_version": 1,
        "start_request_digest": accepted["start_request_digest"],
    }

    assert service_module._SEAL_GATE.acquire(blocking=False)
    try:
        with pytest.raises(GatewayFailure) as raised:
            service.seal(DATASET_ID, seal)
        assert raised.value.error["code"] == "RATE_LIMITED"
    finally:
        service_module._SEAL_GATE.release()

    sealed = service.seal(DATASET_ID, seal)
    assert sealed["dataset_digest"] == manifest["dataset_digest"]
    assert control.state.upload_records() == []


@pytest.mark.parametrize(
    "blocked_stage",
    ("_copy_archive", "_extract_and_admit", "_publish_dataset"),
)
def test_blocked_seal_file_work_does_not_block_unrelated_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_stage: str,
) -> None:
    service, control, _files = _service(
        tmp_path, dataset_ids=(DATASET_ID, SECOND_DATASET_ID)
    )
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    seal = {
        "schema_version": 1,
        "start_request_digest": accepted["start_request_digest"],
    }
    entered = threading.Event()
    release = threading.Event()
    original = getattr(service, blocked_stage)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, blocked_stage, blocked)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(service.seal, DATASET_ID, seal)
        try:
            assert entered.wait(timeout=5)
            assert control.upload_root_cleanup_permissions([DATASET_ID]) == {
                DATASET_ID: False
            }
            other = _start_request(b"other", "2" * 64, chunk_size=5)
            assert service.start(other)["dataset_id"] == SECOND_DATASET_ID
        finally:
            release.set()
        sealed = pending.result(timeout=5)

    assert sealed["dataset_digest"] == manifest["dataset_digest"]


def test_blocked_dataset_deletion_does_not_block_unrelated_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, control, files = _service(
        tmp_path, dataset_ids=(DATASET_ID, SECOND_DATASET_ID)
    )
    archive, manifest = _archive(tmp_path)
    request = _start_request(
        archive, manifest["dataset_digest"], chunk_size=len(archive)
    )
    accepted = service.start(request)
    service.put_chunk(DATASET_ID, 0, archive, hashlib.sha256(archive).hexdigest())
    service.seal(
        DATASET_ID,
        {
            "schema_version": 1,
            "start_request_digest": accepted["start_request_digest"],
        },
    )
    entered = threading.Event()
    release = threading.Event()
    delete_dataset = files.delete_dataset

    def blocked_delete(dataset_id: str) -> None:
        entered.set()
        assert release.wait(timeout=5)
        delete_dataset(dataset_id)

    monkeypatch.setattr(files, "delete_dataset", blocked_delete)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(service.delete, DATASET_ID)
        try:
            assert entered.wait(timeout=5)
            records = service.page(10)["items"]
            assert records[0]["record"]["type"] == "cleanup-required"
            other = _start_request(b"other", "2" * 64, chunk_size=5)
            assert service.start(other)["dataset_id"] == SECOND_DATASET_ID
        finally:
            release.set()
        pending.result(timeout=5)

    assert control.state.upload_records()[0]["dataset_id"] == SECOND_DATASET_ID


def _service(
    root: Path,
    *,
    dataset_ids: tuple[str, ...] = (DATASET_ID,),
) -> tuple[TrainingDatasetService, _Control, TrainingDatasetFiles]:
    volume = root / "volume"
    limits = training_api_limits()
    private_limits = service_limits()
    records = TrainingControlRecords(volume, limits, private_limits)
    generated_ids = iter(dataset_ids)
    state = TrainingDatasetState(
        records,
        limits,
        private_limits,
        id_factory=lambda: uuid.UUID(next(generated_ids)),
    )
    files = TrainingDatasetFiles(volume)
    control = _Control(state, files, limits)
    service = TrainingDatasetService(
        files,
        control,
        limits,
        private_limits,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        lambda capability_id: (
            training_capability()
            if capability_id == training_capability()["capability_id"]
            else None
        ),
    )
    return service, control, files


def _archive(
    root: Path, *, objects: list[dict] | None = None
) -> tuple[bytes, Mapping[str, Any]]:
    extracted = _dataset(root / "source-dataset", objects=objects)
    output = root / "dataset.zip"
    with zipfile.ZipFile(output, "w", allowZip64=True) as package:
        paths = sorted(item for item in extracted.root.rglob("*") if item.is_file())
        for path in paths:
            package.write(path, path.relative_to(extracted.root).as_posix())
    return output.read_bytes(), extracted.manifest


def _start_request(
    archive: bytes,
    dataset_digest: str,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_digest": dataset_digest,
        "archive_bytes_digest": hashlib.sha256(archive).hexdigest(),
        "total_bytes": len(archive),
        "chunk_size_bytes": chunk_size,
    }


def _chunks(value: bytes, size: int):
    for offset in range(0, len(value), size):
        yield offset // size, value[offset : offset + size]
