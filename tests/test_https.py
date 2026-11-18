"""Exercise real TLS handshakes without changing system trust or cloud state."""

import http.server
import io
import ipaddress
import os
import ssl
import threading
import urllib.error
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import truststore
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from histoforge import _cloud_update, _deployment, _https, _modal_setup, cli
from histoforge import _huggingface_setup as hf
from histoforge.cloud import gateway_state_client

TOKEN = "synthetic-token-must-not-leak"


@pytest.fixture(autouse=True)
def certificate_environment(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")


def test_native_context_keeps_verification_and_does_not_patch_ssl():
    original_context = ssl.SSLContext
    original_factory = ssl.create_default_context
    environment = dict(os.environ)
    context = _https.client_ssl_context()
    assert isinstance(context, truststore.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert ssl.SSLContext is original_context
    assert ssl.create_default_context is original_factory
    assert dict(os.environ) == environment


@pytest.mark.parametrize("variable", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
@pytest.mark.parametrize("value", ["", "configured-path"])
def test_explicit_certificate_paths_preserve_python_policy(
    monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)
    expected = object()
    monkeypatch.setattr(ssl, "create_default_context", lambda: expected)
    monkeypatch.setattr(
        truststore, "SSLContext", lambda *_: pytest.fail("native policy substituted")
    )
    assert _https.client_ssl_context() is expected


@pytest.fixture
def certificates(tmp_path):
    """Create a private test issuer; never add it to the operating system store."""
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Local TLS test CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=2))
        .not_valid_after(now + timedelta(days=100))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(True, False, False, False, False, True, True, False, False),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    (tmp_path / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    (tmp_path / "key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    for kind in ("valid", "wrong-host", "expired"):
        host = (
            x509.DNSName("wrong.invalid")
            if kind == "wrong-host"
            else x509.IPAddress(ipaddress.ip_address("127.0.0.1"))
        )
        certificate = (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name(
                    [x509.NameAttribute(NameOID.COMMON_NAME, "Local TLS fixture")]
                )
            )
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=2))
            .not_valid_after(now + timedelta(days=-1 if kind == "expired" else 30))
            .add_extension(x509.SubjectAlternativeName([host]), critical=False)
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None), critical=True
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
            .add_extension(
                x509.KeyUsage(
                    True, False, True, False, False, False, False, False, False
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )
        (tmp_path / f"{kind}.pem").write_bytes(
            certificate.public_bytes(serialization.Encoding.PEM)
        )
    return tmp_path


@pytest.fixture
def tls_server(request, certificates):
    kind = request.param
    received = []
    connections = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            received.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b'{"status":"ok","api_major":1}'
                if self.path == "/v1/health"
                else b'{"status":"ok","name":"researcher"}'
            )

        def log_message(self, *_args):
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    certificate = "valid" if kind == "untrusted" else kind
    context.load_cert_chain(
        certificates / f"{certificate}.pem", certificates / "key.pem"
    )

    class Server(http.server.ThreadingHTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connections.append(address)
            try:
                return context.wrap_socket(connection, server_side=True), address
            except BaseException:
                connection.close()
                raise

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_port}", received, connections, kind
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    "tls_server", ["valid", "untrusted", "wrong-host", "expired"], indirect=True
)
@pytest.mark.parametrize("consumer", ["health", "gateway", "huggingface"])
@pytest.mark.parametrize("policy", ["native", "configured"])
def test_https_consumers_verify_before_sending_requests(
    monkeypatch, tls_server, consumer, policy, capsys, certificates
):
    url, received, connections, kind = tls_server
    if policy == "configured":
        # A nonexistent CA file retains Python's standard policy without trusting
        # our issuer, while still exercising the explicitly configured branch.
        ca_file = "absent.pem" if kind == "untrusted" else "ca.pem"
        monkeypatch.setenv("SSL_CERT_FILE", str(certificates / ca_file))
    context = _https.client_ssl_context()
    if policy == "native" and kind != "untrusted":
        context.load_verify_locations(cafile=str(certificates / "ca.pem"))
    selected = []

    def make_context():
        selected.append(True)
        return context

    owner = {"health": cli, "gateway": _cloud_update, "huggingface": hf}[consumer]
    monkeypatch.setattr(owner, "client_ssl_context", make_context)
    monkeypatch.setattr(hf, "_IDENTITY_URL", url + "/identity")
    if consumer == "health":
        result = cli._health(url)
        assert (result is not None) == (kind == "valid")
    elif consumer == "gateway":
        if kind == "valid":
            assert (
                _cloud_update.gateway_request(url, TOKEN, "/control")["status"] == "ok"
            )
        else:
            with pytest.raises(urllib.error.URLError) as failure:
                _cloud_update.gateway_request(url, TOKEN, "/control")
            assert isinstance(failure.value.reason, ssl.SSLCertVerificationError)
    else:
        if kind == "valid":
            hf._verify_token(TOKEN)
        else:
            monkeypatch.setattr(
                hf, "_save_secret", lambda _: pytest.fail("Secret changed")
            )
            monkeypatch.setattr(
                hf._modal_setup,
                "ensure_auth",
                lambda *_a, **_k: pytest.fail("Modal called"),
            )
            with pytest.raises(hf.TokenSetupFailure) as failure:
                hf.save_token("test", io.StringIO(TOKEN))
            assert failure.value.code == "HF_VERIFICATION_FAILED"
            assert TOKEN not in str(failure.value)
    assert selected == [True], (
        "Verification failure must not select a second trust policy"
    )
    assert len(connections) == 1, "Certificate failure must not trigger another attempt"
    assert len(received) == (1 if kind == "valid" else 0)
    if received:
        assert received[0][1] == (None if consumer == "health" else f"Bearer {TOKEN}")
    assert TOKEN not in "".join(capsys.readouterr())


@pytest.mark.parametrize("tls_server", ["untrusted"], indirect=True)
@pytest.mark.parametrize("workflow", ["connect", "post_deploy"])
def test_tls_failure_preserves_previous_local_credential(
    monkeypatch, tls_server, tmp_path, workflow
):
    url, received, connections, _ = tls_server
    path = tmp_path / "deployment.json"
    path.write_text("previous credential")
    monkeypatch.setattr(_deployment, "PATH", path)
    monkeypatch.setattr(_deployment, "DIR", tmp_path)
    waits, rotations = [], []
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: waits.append(seconds))

    class Client:
        def available(self):
            return True

        def current(self):
            return {
                "deployment_id": "deployment-a",
                "auth_revision": 0,
                "token_sha256": "a" * 64,
            }

        def rotate(self, verifier):
            rotations.append(verifier)
            return {
                "deployment_id": "deployment-a",
                "auth_revision": 1,
                "token_sha256": verifier,
            }

    monkeypatch.setattr(gateway_state_client, "GatewayStateClient", Client)
    monkeypatch.setattr(_modal_setup, "ensure_auth", lambda profile: profile)
    monkeypatch.setattr(
        _modal_setup, "ensure_environment", lambda value=None: "research"
    )
    monkeypatch.setattr(cli, "_gateway_url", lambda: url)
    with pytest.raises(urllib.error.URLError) as failure:
        if workflow == "connect":
            cli.cmd_connect(SimpleNamespace(profile="test", json=True))
        else:
            cli._publish_deployed_credential(
                url,
                {},
                TOKEN,
                "test",
                expected_artifact_digest="a" * 64,
                environment="research",
            )
    assert isinstance(failure.value.reason, ssl.SSLCertVerificationError)
    assert path.read_text() == "previous credential"
    assert received == []
    assert len(connections) == (1 if workflow == "connect" else 4)
    assert len(rotations) == int(workflow == "connect")
    assert waits == ([] if workflow == "connect" else [2, 2, 2])


def test_context_configuration_failure_stays_inside_hf_redaction(monkeypatch, capsys):
    def fail():
        raise OSError(TOKEN)

    monkeypatch.setattr(hf, "client_ssl_context", fail)
    with pytest.raises(hf.TokenSetupFailure) as failure:
        hf.save_token("test", io.StringIO(TOKEN))
    assert failure.value.code == "HF_VERIFICATION_FAILED"
    assert TOKEN not in str(failure.value)
    assert capsys.readouterr() == ("", "")


def test_self_check_rejects_unusable_tls_before_reporting_ready(monkeypatch, capsys):
    def fail():
        raise OSError("TLS context unavailable")

    monkeypatch.setattr(cli, "client_ssl_context", fail)
    with pytest.raises(OSError, match="TLS context unavailable"):
        cli.cmd_self_check(SimpleNamespace(json=True))
    assert capsys.readouterr() == ("", "")
