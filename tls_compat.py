"""
tls_compat.py — TLS 1.2 compatibility patch for pybit on macOS with V2Ray/TUN VPNs.

V2BOX / Outline in TUN mode resets TLS 1.3 handshakes from Python OpenSSL.
curl is unaffected because it uses macOS SecureTransport, not OpenSSL.

Fix: monkey-patch ssl.create_default_context so that every SSL context
created anywhere in the process (urllib3, requests, pybit) disables TLS 1.3.
"""

import ssl
import logging
import urllib3.util.ssl_ as _urllib3_ssl

logger = logging.getLogger(__name__)

_patched = False


def apply_tls12_global_patch() -> None:
    """
    Globally disable TLS 1.3 for urllib3 / requests / pybit.
    Safe to call multiple times — only patches once.

    Root cause: V2BOX / Outline in TUN mode resets TLS-1.3 handshakes from
    Python OpenSSL.  urllib3 creates SSL contexts via create_urllib3_context()
    and ssl.create_default_context(); patching both ensures every HTTPS
    connection in the process uses TLS ≤ 1.2.
    """
    global _patched
    if _patched:
        return

    # 1. Patch urllib3's own context factory (used for every pybit request)
    _orig_create_urllib3_context = _urllib3_ssl.create_urllib3_context

    def _patched_create_urllib3_context(*args, **kwargs):
        ctx = _orig_create_urllib3_context(*args, **kwargs)
        try:
            ctx.options |= ssl.OP_NO_TLSv1_3
        except Exception:
            pass
        return ctx

    _urllib3_ssl.create_urllib3_context = _patched_create_urllib3_context

    # 2. Patch ssl.create_default_context for anything else
    _orig_create_default_context = ssl.create_default_context

    def _patched_create_default_context(*args, **kwargs):
        ctx = _orig_create_default_context(*args, **kwargs)
        try:
            ctx.options |= ssl.OP_NO_TLSv1_3
        except Exception:
            pass
        return ctx

    ssl.create_default_context = _patched_create_default_context

    _patched = True
    logger.debug("tls_compat: TLS-1.3 globally disabled (V2Ray TUN workaround)")


# Apply patch on import so any pybit session created afterwards is covered.
apply_tls12_global_patch()


def patch_pybit_session(http_session) -> None:  # noqa: ARG001
    """
    No-op kept for backward compatibility.
    The global patch in apply_tls12_global_patch() covers pybit sessions.
    """
    logger.debug("tls_compat: patch_pybit_session called (global patch already active)")
