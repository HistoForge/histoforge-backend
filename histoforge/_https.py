"""Certificate verification for the local helper's HTTPS requests."""

import os
import ssl


def client_ssl_context() -> ssl.SSLContext:
    """Use native trust unless the user configured Python's certificate paths.

    Keep explicit SSL_CERT_FILE/SSL_CERT_DIR behavior unchanged. Otherwise use
    the platform's chain validation, rather than copying its current roots into
    OpenSSL. A failed verification is never retried with another trust source.
    """
    if "SSL_CERT_FILE" in os.environ or "SSL_CERT_DIR" in os.environ:
        return ssl.create_default_context()

    import truststore

    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
