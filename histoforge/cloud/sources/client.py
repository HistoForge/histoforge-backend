"""Authenticated Modal adapter for the source-artifact lifecycle owner."""

import modal

from histoforge.cloud._config import DEPLOYMENT_CONFIG


class SourceArtifactLifecycleClient:
    """Bind one authenticated request to the serialized lifecycle lane."""

    def __init__(self, state: dict) -> None:
        self._lane = modal.Cls.from_name(
            DEPLOYMENT_CONFIG.gateway_app_name,
            DEPLOYMENT_CONFIG.artifact_lifecycle_class_name,
        )()
        self._deployment_id = state["deployment_id"]
        self._auth_revision = state["auth_revision"]

    def reserve(self, start_request: dict) -> dict:
        return self._lane.reserve.remote(
            self._deployment_id,
            self._auth_revision,
            start_request,
        )

    def release(self, artifact_id: str) -> None:
        self._lane.release.remote(
            self._deployment_id,
            self._auth_revision,
            artifact_id,
        )

    def claim_chunk(
        self,
        artifact_id: str,
        chunk_id: int,
        body_size: int,
        digest: str,
    ) -> dict:
        return self._lane.claim_chunk.remote(
            self._deployment_id,
            self._auth_revision,
            artifact_id,
            chunk_id,
            body_size,
            digest,
        )

    def commit_chunk(
        self,
        artifact_id: str,
        chunk_id: int,
        digest: str,
        start_request_digest: str,
    ) -> dict:
        return self._lane.commit_chunk.remote(
            self._deployment_id,
            self._auth_revision,
            artifact_id,
            chunk_id,
            digest,
            start_request_digest,
        )

    def claim_seal(self, artifact_id: str, seal_request: dict) -> dict:
        return self._lane.claim_seal.remote(
            self._deployment_id,
            self._auth_revision,
            artifact_id,
            seal_request,
        )

    def commit_seal(
        self,
        artifact_id: str,
        seal_request: dict,
        start_request_digest: str,
        manifest: dict,
    ) -> dict:
        return self._lane.commit_seal.remote(
            self._deployment_id,
            self._auth_revision,
            artifact_id,
            seal_request,
            start_request_digest,
            manifest,
        )
