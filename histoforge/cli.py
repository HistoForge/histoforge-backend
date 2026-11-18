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
from histoforge._https import client_ssl_context
from histoforge.cloud import _auth
from histoforge.cloud._config import (
    DEPLOYMENT_CONFIG,
    validate_gateway_capabilities,
    validate_gateway_health,
)
from histoforge.cloud.capabilities import (
    gateway_protocol_mismatches,
    require_supported_gateway_capabilities,
)
from histoforge.cloud.protocol_versions import GATEWAY_API_MAJOR

_WEB_FN = "fastapi_app"
_CONNECTION_HELP = "Email histoforge.contact@gmail.com with Technical details for help."


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


def _public(data: dict, *, environment: str | None = None) -> dict:
    # Keep the small CLI result stable and never echo the raw token or complete
    # private credential to stdout/logs.
    result = {
        "deployment_id": data["deployment_id"],
        "gateway_url": data["gateway_url"],
        "profile": data["modal_profile"],
        "app_name": data["modal_app_name"],
        "auth_revision": data["auth_revision"],
    }
    selected = environment or data.get("modal_environment")
    if selected is not None:
        result["environment"] = selected
    return result


def _emit(args, payload: dict) -> None:
    if args.json:
        print(json.dumps(payload))
    else:
        for k, v in payload.items():
            print(f"{k:14}: {v}")


def _health(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"{url}/v1/health", timeout=10, context=client_ssl_context()
        ) as r:
            return validate_gateway_health(json.loads(r.read()))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _require_gateway_identity(state: dict | None) -> dict:
    if state is None:
        raise CliFailure(
            "IDENTITY_RECOVERY_REQUIRED",
            "This backend's saved connection record is unavailable. "
            "Your saved connection was not changed. " + _CONNECTION_HELP,
        )
    return state


def _expected_connection(args, *, required: bool = False) -> dict | None:
    """Capture the saved target for reconnect or update, never its old revision."""

    expected_url = getattr(args, "expected_gateway", None)
    expected_id = getattr(args, "expected_deployment_id", None)
    if (expected_url is None) != (expected_id is None):
        raise ValueError(
            "--expected-gateway and --expected-deployment-id must be used together"
        )
    if not required and expected_url is None:
        return None
    try:
        credential = _deployment.read()
    except (OSError, ValueError) as error:
        raise CliFailure(
            "RECONNECT_REQUIRED",
            "The saved connection could not be read. Open cloud setup to "
            "restore access; no backend was changed.",
        ) from error
    if credential is None:
        raise CliFailure(
            "RECONNECT_REQUIRED",
            "The saved connection is unavailable. Open cloud setup to connect "
            "this computer. No backend was changed.",
        )
    if expected_url is not None and (
        credential["gateway_url"] != expected_url
        or credential["deployment_id"] != expected_id
    ):
        raise CliFailure(
            "GATEWAY_TARGET_MISMATCH",
            "The saved cloud connection changed. Reopen Settings and try again. "
            "No backend was changed.",
        )
    return credential


def _requested_environment(args, credential: dict | None = None) -> str | None:
    """Use the saved location; an explicit name may repair an older credential."""

    requested = getattr(args, "environment", None)
    saved = None if credential is None else credential.get("modal_environment")
    if saved is not None and requested is not None and requested != saved:
        raise CliFailure(
            "GATEWAY_TARGET_MISMATCH",
            "The selected Modal environment does not match the saved backend. "
            "Your saved connection was not changed.",
        )
    return saved or requested


def _resolve_environment(args, credential: dict | None = None) -> str:
    """Keep confirmed missing locations recoverable without hiding access errors."""

    requested = _requested_environment(args, credential)
    import modal

    try:
        return _modal_setup.ensure_environment(requested)
    except modal.exception.NotFoundError as error:
        if credential is None:
            raise
        guidance = (
            "Use Locate saved backend in Settings to enter the environment "
            "used during setup. "
            if credential.get("modal_environment") is None
            else "Check the saved environment in Modal. " + _CONNECTION_HELP + " "
        )
        raise CliFailure(
            "GATEWAY_TARGET_MISMATCH",
            "The selected Modal environment could not be found. "
            + guidance
            + "Your saved connection was not changed.",
        ) from error


def _require_gateway_target(modal_profile: str, credential: dict) -> str:
    """Reject another Modal target before state lookup or credential rotation."""

    if credential["modal_app_name"] != DEPLOYMENT_CONFIG.gateway_app_name:
        raise CliFailure(
            "GATEWAY_TARGET_MISMATCH",
            "The saved connection names an unsupported cloud app. "
            "Nothing was changed. " + _CONNECTION_HELP,
        )
    if credential["modal_profile"] != modal_profile:
        raise CliFailure(
            "GATEWAY_TARGET_MISMATCH",
            "The selected Modal profile does not match the saved connection. "
            "Use the Modal profile selected during setup. Nothing was changed.",
        )
    try:
        url = _required_gateway_url()
    except CliFailure as error:
        if error.code != "GATEWAY_NOT_FOUND":
            raise
        raise CliFailure(
            "GATEWAY_TARGET_MISMATCH",
            "This Modal profile or environment cannot find your saved backend. "
            "Check the location used during setup. Nothing was changed. "
            + _CONNECTION_HELP,
        ) from error
    if url != credential["gateway_url"]:
        print(
            f"Saved backend: {credential['gateway_url']}; "
            f"selected backend: {url}; Modal profile: {modal_profile}; "
            f"environment: {os.environ.get('MODAL_ENVIRONMENT')}.",
            file=sys.stderr,
        )
        raise CliFailure(
            "GATEWAY_TARGET_MISMATCH",
            "This Modal profile or environment reaches a different backend. "
            "Your saved connection was not changed. " + _CONNECTION_HELP,
        )
    return url


def _require_same_deployment(state: dict, credential: dict) -> None:
    if state["deployment_id"] != credential["deployment_id"]:
        raise CliFailure(
            "GATEWAY_TARGET_MISMATCH",
            "The backend at the saved address has a different deployment "
            "identity. Your saved connection was not changed. " + _CONNECTION_HELP,
        )


def _prepare_upgrade(args, modal_profile: str) -> dict:
    """Read and validate the update target; do not stop work or replace access."""

    from histoforge._gateway_state_setup import GatewayStateSetup
    from histoforge.cloud.gateway_state_client import GatewayStateClient

    credential = _expected_connection(args, required=True)
    url = _require_gateway_target(modal_profile, credential)
    _require_artifact_lifecycle_lane()
    client = GatewayStateClient()
    setup = None
    staged = None
    if client.available():
        previous = _require_gateway_identity(client.current())
        _require_same_deployment(previous, credential)
    else:
        setup = GatewayStateSetup()
        previous = _require_gateway_identity(setup.legacy_current())
        _require_same_deployment(previous, credential)
        staged = setup.prepare_cutover(previous)
    if credential["auth_revision"] != previous["auth_revision"] or not (
        _auth.matches_local_token(previous, credential["token"])
    ):
        raise CliFailure(
            "RECONNECT_REQUIRED",
            "Access to this backend has changed. Click Reconnect, then try "
            "the update again. Nothing was changed.",
        )
    return {
        "credential": credential,
        "url": url,
        "client": client,
        "previous": previous,
        "legacy_setup": setup,
        "staged": staged,
    }


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
        except (urllib.error.URLError, TimeoutError):
            # A response read can time out without urllib wrapping the exception.
            if attempt == 3:
                raise
        time.sleep(2)
    raise AssertionError("unreachable readiness retry state")


def _credential_from_gateway_capabilities(
    url: str,
    state: dict,
    token: str,
    modal_profile: str,
    capabilities: dict,
    *,
    environment: str,
) -> dict:
    """Bind a local login to its authenticated deployment and current revision."""

    if capabilities["deployment_id"] != state["deployment_id"]:
        raise RuntimeError("gateway capabilities returned a different deployment ID")
    if capabilities["auth_revision"] != state["auth_revision"]:
        raise RuntimeError("gateway capabilities returned a different auth revision")
    return _deployment.build(
        deployment_id=state["deployment_id"],
        gateway_url=url,
        modal_profile=modal_profile,
        modal_app_name=DEPLOYMENT_CONFIG.gateway_app_name,
        auth_revision=state["auth_revision"],
        token=token,
        modal_environment=environment,
    )


def _publish_deployed_credential(
    url: str,
    state: dict,
    token: str,
    modal_profile: str,
    *,
    expected_artifact_digest: str,
    environment: str,
) -> dict:
    """Publish an updated login only after verifying the exact bundled backend."""

    capabilities = _capabilities_after_deploy(url, token)
    credential = _credential_from_gateway_capabilities(
        url, state, token, modal_profile, capabilities, environment=environment
    )
    build = capabilities["gateway_build"]
    if (
        build["version"] != __version__
        or build["artifact_digest"] != expected_artifact_digest
    ):
        raise RuntimeError(
            "the gateway is not running the expected bundled backend; "
            "update was not verified and the local credential was not replaced"
        )
    return _deployment.write(credential)


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
    saved = None if fresh else _expected_connection(args, required=True)
    modal_profile = _modal_setup.ensure_auth(
        args.profile or (saved["modal_profile"] if saved is not None else None)
    )
    environment = _resolve_environment(args, saved)
    import modal

    from histoforge._gateway_state_setup import GatewayStateSetup
    from histoforge.cloud.gateway_state_client import GatewayStateClient

    # Validate the complete local graph before rotating authentication or
    # touching any cloud-owned state. Modal otherwise discovers malformed
    # parameter ordering only while resolving the deployment.
    app = _validated_deployment_app()
    from histoforge.cloud.runtime_identity import bundle_artifact_digest

    expected_artifact = bundle_artifact_digest()
    if fresh:
        if _deployed_gateway_exists():
            raise CliFailure(
                "GATEWAY_EXISTS",
                "A gateway already exists. Use Update cloud backend, not setup.",
            )
        # The app is confirmed absent. A retry preserves the bootstrap identity.
        GatewayStateSetup(fresh=True).bootstrap()
        client = GatewayStateClient()
    else:
        prepared = _prepare_upgrade(args, modal_profile)
        client = prepared["client"]
        previous = prepared["previous"]
        setup = prepared["legacy_setup"]
        url = prepared["url"]
        credential = prepared["credential"]
        _cloud_update.stop_cloud_work(
            url, credential["token"], wait_for_module_writer=_await_module_mutation_lane
        )
        # Withhold the new credential while old admitted work drains. Every
        # mutation boundary still checks its original deployment generation.
        token = _auth.new_token()
        if setup is not None:
            state = setup.cut_over(
                previous, prepared["staged"], _auth.token_digest(token)
            )
        else:
            state = client.rotate(
                _auth.token_digest(token),
                expected_deployment_id=previous["deployment_id"],
                expected_revision=previous["auth_revision"],
            )
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
    if fresh:
        token = _auth.new_token()
        state = client.rotate(_auth.token_digest(token))
    print("Verifying the updated cloud backend…", file=sys.stderr)
    data = _publish_deployed_credential(
        url,
        state,
        token,
        modal_profile,
        expected_artifact_digest=expected_artifact,
        environment=environment,
    )
    _emit(args, {"status": "deployed" if fresh else "upgraded", **_public(data)})


def cmd_deploy(args) -> None:
    _deploy(args, fresh=True)


def cmd_upgrade(args) -> None:
    if not getattr(args, "check", False):
        _deploy(args, fresh=False)
        return
    saved = _expected_connection(args, required=True)
    modal_profile = _modal_setup.ensure_auth(args.profile or saved["modal_profile"])
    environment = _resolve_environment(args, saved)
    _validated_deployment_app()
    prepared = _prepare_upgrade(args, modal_profile)
    _emit(
        args,
        {
            "status": "ready_to_update",
            **_public(prepared["credential"], environment=environment),
        },
    )


def cmd_connect(args) -> None:
    expected = _expected_connection(args)
    modal_profile = _modal_setup.ensure_auth(
        args.profile or (expected["modal_profile"] if expected is not None else None)
    )
    environment = _resolve_environment(args, expected)
    url = (
        _required_gateway_url()
        if expected is None
        else _require_gateway_target(modal_profile, expected)
    )
    from histoforge._gateway_state_setup import GatewayStateSetup
    from histoforge.cloud.gateway_state_client import GatewayStateClient

    client = GatewayStateClient()
    if client.available():
        previous = _require_gateway_identity(client.current())
        if expected is not None:
            _require_same_deployment(previous, expected)
        token = _auth.new_token()
        if expected is None:
            state = client.rotate(_auth.token_digest(token))
        else:
            state = client.rotate(
                _auth.token_digest(token),
                expected_deployment_id=expected["deployment_id"],
            )
    else:
        # Only confirmed absence of the new class permits the old helper path.
        # A missing identity never becomes an automatic fresh deployment.
        setup = GatewayStateSetup()
        previous = _require_gateway_identity(setup.legacy_current())
        if expected is not None:
            _require_same_deployment(previous, expected)
        state, token = setup.legacy_rotate(
            expected_deployment_id=(
                None if expected is None else expected["deployment_id"]
            )
        )
    if expected is not None:
        _require_same_deployment(state, expected)
    capabilities = validate_gateway_capabilities(
        _cloud_update.gateway_request(url, token, "/v1/capabilities")
    )
    if capabilities["api_major"] != GATEWAY_API_MAJOR:
        raise ValueError("the cloud backend uses an unsupported gateway API major")
    mismatches = gateway_protocol_mismatches(capabilities)
    credential = _credential_from_gateway_capabilities(
        url, state, token, modal_profile, capabilities, environment=environment
    )
    data = _deployment.write(credential)
    payload = {"status": "connected", **_public(data)}
    if mismatches:
        payload["protocol_mismatches"] = mismatches
    _emit(args, payload)


def cmd_login(args) -> None:
    """Complete Modal login before accepting a separate sensitive stdin payload."""
    expected = _expected_connection(args)
    profile = _modal_setup.ensure_auth(
        args.profile or (None if expected is None else expected["modal_profile"])
    )
    environment = _resolve_environment(args, expected)
    if expected is not None:
        _verify_secret_target(profile, expected)
    _emit(
        args,
        {
            "status": "authenticated",
            "profile": profile,
            "environment": environment,
            **({} if expected is None else _public(expected, environment=environment)),
        },
    )


def _verify_secret_target(profile: str, expected: dict) -> None:
    """Check the saved identity before replacing an environment-local Secret."""

    _require_gateway_target(profile, expected)
    from histoforge._gateway_state_setup import GatewayStateSetup
    from histoforge.cloud.gateway_state_client import GatewayStateClient

    client = GatewayStateClient()
    state = (
        client.current() if client.available() else GatewayStateSetup().legacy_current()
    )
    _require_same_deployment(_require_gateway_identity(state), expected)


def cmd_huggingface_token(args) -> None:
    from histoforge._huggingface_setup import TokenSetupFailure, save_token

    try:
        expected = _expected_connection(args, required=_deployment.PATH.exists())
        requested = _requested_environment(args, expected)
        profile, environment = save_token(
            args.profile or (None if expected is None else expected["modal_profile"]),
            sys.stdin,
            environment=requested,
            verify_target=(
                None
                if expected is None
                else lambda selected: _verify_secret_target(selected, expected)
            ),
        )
    except CliFailure:
        raise
    except TokenSetupFailure as error:
        raise CliFailure(error.code, str(error)) from None
    except Exception:
        raise CliFailure(
            "HF_SETUP_FAILED", "Could not save the Hugging Face token. Try again."
        ) from None
    _emit(
        args,
        {
            "status": "saved",
            "profile": profile,
            "environment": environment,
            **({} if expected is None else _public(expected, environment=environment)),
        },
    )


def cmd_self_check(args) -> None:
    """Check local TLS and deployment imports without contacting the cloud."""

    client_ssl_context()
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
    common.add_argument(
        "--environment",
        help="Modal environment for setup or verified recovery of an older connection",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "deploy", parents=[common], help="provision the gateway in your workspace"
    ).set_defaults(func=cmd_deploy)
    connect = sub.add_parser(
        "connect", parents=[common], help="join an existing gateway"
    )
    connect.set_defaults(func=cmd_connect)
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
    upgrade.add_argument(
        "--check",
        action="store_true",
        help="check the saved update target without stopping work or deploying",
    )
    login = sub.add_parser(
        "login", parents=[common], help="authenticate your local Modal profile"
    )
    login.set_defaults(func=cmd_login)
    token = sub.add_parser(
        "huggingface-token",
        parents=[common],
        help="verify the token on stdin and save it to your Modal workspace",
    )
    token.set_defaults(func=cmd_huggingface_token)
    for command in (connect, upgrade, login, token):
        command.add_argument(
            "--expected-gateway", help="require the gateway saved by this client"
        )
        command.add_argument(
            "--expected-deployment-id", help="require the saved deployment identity"
        )
    upgrade.set_defaults(func=cmd_upgrade)
    sub.add_parser(
        "status", parents=[common], help="show deployment + connection state"
    ).set_defaults(func=cmd_status)
    sub.add_parser(
        "self-check", parents=[common], help="verify the bundled deployment runtime"
    ).set_defaults(func=cmd_self_check)

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
