"""The `histoforge` command — provision and manage a private Modal gateway.

Runs on the researcher's laptop; orchestration plus exact local/wire codecs.
The plugin invokes these behind buttons and parses `--json` output. The gateway
token is written to ~/.histoforge/deployment.json (0600). Hugging Face credentials
arrive only through stdin and persist only in the administration Modal Secret.
"""

import argparse
import contextlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import warnings

from histoforge import __version__, _cloud_update, _deployment, _modal_setup
from histoforge.cloud import _auth
from histoforge.cloud._config import (
    DEPLOYMENT_CONFIG,
    validate_gateway_health,
)
from histoforge.cloud.capabilities import require_supported_gateway_capabilities

_WEB_FN = "fastapi_app"


class CliFailure(RuntimeError):
    """One stable local-helper failure that the QuPath UI may branch on."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _gateway_url() -> str:
    import modal

    return modal.Function.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name, _WEB_FN
    ).get_web_url()


def _required_gateway_url() -> str:
    """Resolve an existing gateway or expose only its confirmed absence."""

    import modal

    try:
        return _gateway_url()
    except modal.exception.NotFoundError as error:
        raise CliFailure(
            "GATEWAY_NOT_FOUND",
            "No HistoForge gateway exists for this Modal profile.",
        ) from error


def _public(data: dict) -> dict:
    # Keep the small CLI result stable and never echo the raw token or complete
    # private credential to stdout/logs.
    return {
        "deployment_id": data["deployment_id"],
        "gateway_url": data["gateway_url"],
        "profile": data["modal_profile"],
        "app_name": data["modal_app_name"],
        "auth_revision": data["auth_revision"],
    }


def _emit(args, payload: dict) -> None:
    if args.json:
        print(json.dumps(payload))
    else:
        for k, v in payload.items():
            print(f"{k:14}: {v}")


def _health(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/v1/health", timeout=10) as r:
            return validate_gateway_health(json.loads(r.read()))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _gateway_state_dict(*, create: bool = False):
    import modal

    return modal.Dict.from_name(
        DEPLOYMENT_CONFIG.gateway_state_dict_name, create_if_missing=create
    )


def _await_module_mutation_lane() -> None:
    """Wait until every mutation already admitted to the deployed lane has finished."""

    import modal

    lane = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.module_mutation_class_name,
    )()
    lane.deployment_barrier.remote()


def _require_artifact_lifecycle_lane() -> None:
    """Reject an unsupported deployment cutover before changing credentials."""

    import modal

    try:
        modal.Cls.from_name(
            DEPLOYMENT_CONFIG.gateway_app_name,
            DEPLOYMENT_CONFIG.artifact_lifecycle_class_name,
        ).hydrate()
    except modal.exception.NotFoundError as error:
        raise CliFailure(
            "DEPLOYMENT_CUTOVER_REQUIRED",
            "This development gateway predates the unified artifact lifecycle. "
            "Use a fresh Modal environment or complete the documented one-time "
            "cutover before upgrading. Credentials were not changed.",
        ) from error


def _await_artifact_lifecycle_lane() -> None:
    """Wait until every earlier source/run lifecycle control cut has finished."""

    import modal

    lane = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.artifact_lifecycle_class_name,
    )()
    lane.deployment_barrier.remote()


def _await_training_state_lane() -> None:
    """Drain the installed deployment's training-state writer."""

    import modal

    lane = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.training_state_class_name,
    )()
    lane.deployment_barrier.remote()


def _deployed_gateway_exists() -> bool:
    """Return whether the active workspace already contains the gateway app."""

    import modal

    try:
        return _gateway_url() is not None
    except modal.exception.NotFoundError:
        return False


def _capabilities(url: str, token: str) -> dict:
    return require_supported_gateway_capabilities(
        _cloud_update.gateway_request(url, token, "/v1/capabilities")
    )


def _capabilities_after_deploy(url: str, token: str) -> dict:
    """Wait for a recreated gateway's first authenticated readiness response."""

    for attempt in range(4):
        try:
            return _capabilities(url, token)
        except urllib.error.HTTPError as error:
            if error.code not in {500, 502, 503, 504} or attempt == 3:
                raise
        except urllib.error.URLError:
            if attempt == 3:
                raise
        time.sleep(2)
    raise AssertionError("unreachable readiness retry state")


def _publish_local_credential(
    url: str,
    state: dict,
    token: str,
    modal_profile: str,
    *,
    expected_artifact_digest: str | None = None,
) -> dict:
    capabilities = (
        _capabilities_after_deploy(url, token)
        if expected_artifact_digest is not None
        else _capabilities(url, token)
    )
    if capabilities["deployment_id"] != state["deployment_id"]:
        raise RuntimeError("gateway capabilities returned a different deployment ID")
    if capabilities["auth_revision"] != state["auth_revision"]:
        raise RuntimeError("gateway capabilities returned a different auth revision")
    if expected_artifact_digest is not None:
        build = capabilities["gateway_build"]
        if (
            build["version"] != __version__
            or build["artifact_digest"] != expected_artifact_digest
        ):
            raise RuntimeError(
                "the gateway is not running the expected bundled backend; "
                "update was not verified and the local credential was not replaced"
            )
    return _deployment.write(
        _deployment.build(
            deployment_id=state["deployment_id"],
            gateway_url=url,
            modal_profile=modal_profile,
            modal_app_name=DEPLOYMENT_CONFIG.gateway_app_name,
            auth_revision=state["auth_revision"],
            token=token,
        )
    )


def _validated_deployment_app():
    """Construct the graph and validate Modal's synthesized class signatures."""

    from modal.cls import _get_class_constructor_signature

    from histoforge.cloud.deploy import app

    for name, deployed_class in app.registered_classes.items():
        try:
            _get_class_constructor_signature(deployed_class._get_user_cls())
        except ValueError as error:
            raise RuntimeError(
                f"Modal class {name!r} has an invalid parameter signature: {error}"
            ) from error
    return app


def _deploy(args, *, fresh: bool) -> None:
    if not fresh and not args.cancel_work:
        raise CliFailure(
            "UPDATE_CONFIRMATION_REQUIRED",
            "Updating stops unfinished work in this Modal workspace. "
            "Use --cancel-work to confirm, and keep other clients from submitting "
            "or reconnecting until the update finishes.",
        )
    # ensure_auth sets MODAL_PROFILE; it must run BEFORE `import modal`, since modal
    # resolves its active profile once at import (importing first ignores --profile).
    modal_profile = _modal_setup.ensure_auth(args.profile)
    import modal

    # Validate the complete local graph before rotating authentication or
    # touching any cloud-owned state. Modal otherwise discovers malformed
    # parameter ordering only while resolving the deployment.
    app = _validated_deployment_app()
    from histoforge.cloud.runtime_identity import bundle_artifact_digest

    expected_artifact = bundle_artifact_digest()
    existing_gateway = _deployed_gateway_exists()
    if fresh and existing_gateway:
        raise CliFailure(
            "GATEWAY_EXISTS",
            "A gateway already exists. Use Update cloud backend, not setup.",
        )
    if not fresh and not existing_gateway:
        raise CliFailure(
            "GATEWAY_NOT_FOUND", "No HistoForge gateway exists for this Modal profile."
        )
    if existing_gateway:
        _require_artifact_lifecycle_lane()

    state_store = _gateway_state_dict(create=fresh)
    previous = _auth.current_state(state_store)
    if existing_gateway:
        url = _required_gateway_url()
        credential = _deployment.read()
        if (
            previous is None
            or credential is None
            or credential["deployment_id"] != previous["deployment_id"]
            or credential["modal_profile"] != modal_profile
            or credential["gateway_url"] != url
            or not _auth.matches_local_token(previous, credential["token"])
        ):
            raise CliFailure(
                "RECONNECT_REQUIRED",
                "Reconnect to this Modal workspace before updating; its current "
                "credential is required to stop work safely. Nothing was changed.",
            )
        _cloud_update.stop_cloud_work(
            url, credential["token"], wait_for_module_writer=_await_module_mutation_lane
        )
        if _auth.current_state(state_store) != previous:
            raise RuntimeError(
                "gateway access changed during update preparation; "
                "retry without concurrent reconnects"
            )

    # Establish an unpublished auth generation before draining. Requests that
    # authenticated under the previous generation are rejected again at admin
    # admission and at the serialized mutation boundary, closing the gap
    # between the final idle observation and recreate.
    if previous is None:
        state, token = _auth.provision_new(state_store)
    else:
        state, token = _auth.rotate(state_store)
    if existing_gateway:
        # Synchronous delete has no admin-job row. Entering the same serialized
        # lane waits behind every already-running mutation. A stale request may
        # still publish a failed transient row, so check active rows once more.
        _await_module_mutation_lane()
        _await_artifact_lifecycle_lane()
        _await_training_state_lane()
        _cloud_update.stop_cloud_work(
            url, token, wait_for_module_writer=_await_module_mutation_lane
        )
    from histoforge.cloud._app import IMAGE_BUILDER_VERSION

    # Modal streams its build logs to stdout; the plugin parses stdout as the
    # result JSON, so route all of Modal's output to stderr to keep stdout clean.
    # Modal resolves this setting when hydrating images, not when constructing
    # them. Keep it active through deploy without changing the workspace default
    # or leaving an override behind in the caller's process.
    previous_builder = os.environ.get("MODAL_IMAGE_BUILDER_VERSION")
    os.environ["MODAL_IMAGE_BUILDER_VERSION"] = IMAGE_BUILDER_VERSION
    print("Deploying the bundled cloud backend…", file=sys.stderr)
    try:
        with (
            contextlib.redirect_stdout(sys.stderr),
            modal.enable_output(),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "error",
                message=r".*containers did not all terminate.*",
                category=UserWarning,
            )
            app.deploy(strategy="recreate")
    except UserWarning as error:
        raise CliFailure(
            "DEPLOYMENT_INCOMPLETE",
            "Modal could not confirm that the old containers stopped. The cloud "
            "update is incomplete; reconnect and retry. Stored data was not reset.",
        ) from error
    finally:
        if previous_builder is None:
            os.environ.pop("MODAL_IMAGE_BUILDER_VERSION", None)
        else:
            os.environ["MODAL_IMAGE_BUILDER_VERSION"] = previous_builder
    url = _gateway_url()
    print("Verifying the updated cloud backend…", file=sys.stderr)
    data = _publish_local_credential(
        url, state, token, modal_profile, expected_artifact_digest=expected_artifact
    )
    _emit(args, {"status": "deployed" if fresh else "upgraded", **_public(data)})


def cmd_deploy(args) -> None:
    _deploy(args, fresh=True)


def cmd_upgrade(args) -> None:
    _deploy(args, fresh=False)


def cmd_connect(args) -> None:
    modal_profile = _modal_setup.ensure_auth(args.profile)
    url = _required_gateway_url()
    state, token = _auth.rotate(_gateway_state_dict())
    data = _publish_local_credential(url, state, token, modal_profile)
    _emit(args, {"status": "connected", **_public(data)})


def cmd_login(args) -> None:
    """Complete Modal login before accepting a separate sensitive stdin payload."""
    profile = _modal_setup.ensure_auth(args.profile)
    _emit(args, {"status": "authenticated", "profile": profile})


def cmd_huggingface_token(args) -> None:
    from histoforge._huggingface_setup import TokenSetupFailure, save_token

    try:
        profile = save_token(args.profile, sys.stdin)
    except TokenSetupFailure as error:
        raise CliFailure(error.code, str(error)) from None
    except Exception:
        raise CliFailure(
            "HF_SETUP_FAILED", "Could not save the Hugging Face token. Try again."
        ) from None
    _emit(args, {"status": "saved", "profile": profile})


def cmd_self_check(args) -> None:
    """Construct the complete deployment graph without mutating cloud state."""

    _validated_deployment_app()

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    _emit(
        args,
        {
            "status": "ok",
            "version": __version__,
            "python": python_version,
        },
    )


def cmd_status(args) -> None:
    try:
        data = _deployment.read()
    except (OSError, ValueError):
        _emit(
            args,
            {
                "status": "not_configured",
                "hint": (
                    "the local deployment credential is invalid; "
                    "run `histoforge connect` to replace it"
                ),
            },
        )
        return
    if not data:
        _emit(args, {"status": "not_configured", "hint": "run `histoforge deploy`"})
        return
    health = _health(data["gateway_url"])
    capabilities = None
    if health is not None:
        try:
            capabilities = _capabilities(data["gateway_url"], data["token"])
        except (urllib.error.URLError, OSError, ValueError):
            pass
    connected = bool(
        capabilities
        and health["api_major"] == capabilities["api_major"]
        and capabilities["deployment_id"] == data["deployment_id"]
        and capabilities["auth_revision"] == data["auth_revision"]
    )
    _emit(
        args,
        {
            "status": "configured",
            "gateway_url": data["gateway_url"],
            "profile": data["modal_profile"],
            "app_name": data["modal_app_name"],
            "reachable": health is not None,
            "connected": connected,
            "api_major": (health or {}).get("api_major"),
            "auth_revision": data["auth_revision"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="histoforge",
        description="Manage a private HistoForge inference gateway on Modal.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("-p", "--profile", help="local Modal profile to use")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "deploy", parents=[common], help="provision the gateway in your workspace"
    ).set_defaults(func=cmd_deploy)
    sub.add_parser(
        "connect", parents=[common], help="join an existing gateway"
    ).set_defaults(func=cmd_connect)
    upgrade = sub.add_parser(
        "upgrade",
        parents=[common],
        help="stop unfinished work and update the cloud backend",
    )
    upgrade.add_argument(
        "--cancel-work",
        action="store_true",
        help="confirm cancellation; do not submit or reconnect from another client "
        "during the update",
    )
    upgrade.set_defaults(func=cmd_upgrade)
    sub.add_parser(
        "status", parents=[common], help="show deployment + connection state"
    ).set_defaults(func=cmd_status)
    sub.add_parser(
        "self-check", parents=[common], help="verify the bundled deployment runtime"
    ).set_defaults(func=cmd_self_check)
    sub.add_parser(
        "login", parents=[common], help="authenticate your local Modal profile"
    ).set_defaults(func=cmd_login)
    sub.add_parser(
        "huggingface-token",
        parents=[common],
        help="verify the token on stdin and save it to your Modal workspace",
    ).set_defaults(func=cmd_huggingface_token)

    args = parser.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as e:  # clean error contract for the plugin
        code = e.code if isinstance(e, CliFailure) else "CLI_ERROR"
        if getattr(args, "json", False):
            print(json.dumps({"status": "error", "code": code, "error": str(e)}))
        else:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
