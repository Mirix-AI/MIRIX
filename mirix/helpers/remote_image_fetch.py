"""Server-side fetch of a caller-supplied remote image URL, guarded against SSRF.

A memory-add payload may reference an image by URL. The conversation store only
records the address (never downloaded), and the OpenAI and Anthropic
clients hand the URL to the provider, which fetches it from its own network.
The Google client, however, must inline the bytes, so the MIRIX server itself
would GET a URL chosen by any API caller — e.g. a cloud metadata endpoint or an
internal service. Every such fetch goes through :func:`fetch_remote_image`:

* ``https`` only, no credentials in the URL;
* the host is resolved once and EVERY resolved address must be globally
  routable (no loopback, private, link-local — which covers
  ``169.254.169.254`` —, shared, reserved, multicast or unspecified
  addresses). IPv6 forms that carry an IPv4 address are judged by that IPv4
  address — IPv4-mapped ``::ffff:0:0/96``, IPv4-translated
  ``::ffff:0:0:0/96`` and the NAT64 well-known prefix ``64:ff9b::/96``, where
  a DNS64/NAT64 gateway would forward ``64:ff9b::a9fe:a9fe`` to
  ``169.254.169.254`` — and forms whose IPv4 cannot be located reliably or
  that are deprecated are refused outright: the local-use NAT64 prefix
  ``64:ff9b:1::/48`` (operator-defined layout), IPv4-compatible ``::/96``,
  6to4 ``2002::/16`` and Teredo ``2001::/32``;
* the connection goes to the vetted address itself (the Host header and TLS SNI
  / certificate check still use the host name), so a second DNS answer cannot
  redirect it (DNS rebinding); proxies from the environment are not used;
* redirects are not followed; the response must be ``200`` with an ``image/*``
  content type; at most ``MAX_IMAGE_BYTES`` are read. Every connect/read/write
  has a short timeout, and the whole fetch (resolution included) a total
  deadline, so a server trickling one byte at a time cannot hold the request
  (and the meta-agent step waiting on it) for longer than the deadline;
* the bytes must decode as a supported image (``inspect_image_bytes``).

Any violation raises :class:`RemoteImageRejected`; callers degrade to a text
placeholder rather than failing the whole request.
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Awaitable, Callable, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import httpx

from mirix.log import get_logger

logger = get_logger(__name__)

Resolver = Callable[[str, int], Awaitable[List[str]]]


class RemoteImageRejected(Exception):
    """The URL may not be fetched, or did not yield an acceptable image."""


# IPv6 prefixes whose last 32 bits are an IPv4 address the packet reaches.
_IPV4_TRANSLATED = ipaddress.ip_network("::ffff:0:0:0/96")  # RFC 2765 (SIIT)
_NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")  # RFC 6052
# Refused outright: the IPv4 position is operator-defined (RFC 8215 local-use
# NAT64) or the form is deprecated (RFC 4291 IPv4-compatible, RFC 7526 6to4).
# Teredo (2001::/32) and 6to4 are also caught by ``teredo``/``sixtofour``.
_REFUSED_IPV6_NETWORKS = (
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("2002::/16"),
    ipaddress.ip_network("2001::/32"),
)


def _is_public_ipv4(ip: ipaddress.IPv4Address) -> bool:
    return bool(ip.is_global) and not ip.is_multicast


def is_public_address(address: str) -> bool:
    """True only for a globally routable unicast address.

    IPv6 addresses that embed an IPv4 destination (IPv4-mapped, IPv4-translated,
    NAT64 well-known prefix) are judged by that IPv4 address; embedded forms
    whose IPv4 cannot be located reliably, and deprecated ones, are refused
    (see the module docstring). This does not rely on the running Python's
    ``is_global`` tables, which differ between versions for these prefixes.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv4Address):
        return _is_public_ipv4(ip)
    if ip.ipv4_mapped is not None:
        return _is_public_ipv4(ip.ipv4_mapped)
    if ip in _IPV4_TRANSLATED or ip in _NAT64_WELL_KNOWN:
        return _is_public_ipv4(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    if (
        any(ip in network for network in _REFUSED_IPV6_NETWORKS)
        or ip.sixtofour is not None
        or ip.teredo is not None
    ):
        return False
    return bool(ip.is_global) and not ip.is_multicast


async def _system_resolver(host: str, port: int) -> List[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=0, proto=0)
    return list(dict.fromkeys(info[4][0] for info in infos))


def _netloc(address: str, port: Optional[int]) -> str:
    host = f"[{address}]" if ":" in address else address
    return f"{host}:{port}" if port else host


async def fetch_remote_image(
    url: str,
    *,
    max_bytes: Optional[int] = None,
    timeout: Optional[float] = None,
    total_timeout: Optional[float] = None,
    resolver: Optional[Resolver] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Tuple[bytes, str]:
    """GET a caller-supplied image URL under the SSRF rules above.

    Returns ``(bytes, mime_type)``. ``timeout`` bounds each connect/read/write
    (``MIRIX_REMOTE_IMAGE_FETCH_TIMEOUT_SECONDS``); ``total_timeout`` bounds the
    whole resolution + download (``MIRIX_REMOTE_IMAGE_FETCH_TOTAL_TIMEOUT_SECONDS``).
    ``resolver`` and ``transport`` are injectable for tests (no network).
    """
    from mirix.constants import (
        MAX_IMAGE_BYTES,
        REMOTE_IMAGE_FETCH_TIMEOUT_SECONDS,
        REMOTE_IMAGE_FETCH_TOTAL_TIMEOUT_SECONDS,
    )
    from mirix.services.image_store import ImageDecodeError, inspect_image_bytes

    max_bytes = MAX_IMAGE_BYTES if max_bytes is None else int(max_bytes)
    timeout = REMOTE_IMAGE_FETCH_TIMEOUT_SECONDS if timeout is None else float(timeout)
    total_timeout = (
        REMOTE_IMAGE_FETCH_TOTAL_TIMEOUT_SECONDS
        if total_timeout is None
        else float(total_timeout)
    )

    try:
        data, content_type = await asyncio.wait_for(
            _download(
                url,
                max_bytes=max_bytes,
                timeout=timeout,
                resolver=resolver,
                transport=transport,
            ),
            timeout=total_timeout,
        )
    except asyncio.TimeoutError as exc:
        raise RemoteImageRejected(
            f"fetching the image took longer than {total_timeout:g}s"
        ) from exc

    try:
        prepared = await asyncio.to_thread(inspect_image_bytes, data, content_type)
    except ImageDecodeError as exc:
        raise RemoteImageRejected(str(exc)) from exc
    return prepared.data, prepared.mime_type


async def _download(
    url: str,
    *,
    max_bytes: int,
    timeout: float,
    resolver: Optional[Resolver],
    transport: Optional[httpx.AsyncBaseTransport],
) -> Tuple[bytes, str]:
    """Validate, resolve, and read the image bytes (the part under the deadline)."""

    parts = urlsplit(url or "")
    if parts.scheme.lower() != "https":
        raise RemoteImageRejected("only https image URLs are fetched")
    if parts.username or parts.password:
        raise RemoteImageRejected("image URLs with credentials are not fetched")
    host = parts.hostname
    if not host:
        raise RemoteImageRejected("the image URL has no host")
    try:
        port = parts.port
    except ValueError as exc:
        raise RemoteImageRejected("the image URL has an invalid port") from exc

    try:
        addresses = await (resolver or _system_resolver)(host, port or 443)
    except Exception as exc:  # noqa: BLE001 — resolution failure = not fetchable
        raise RemoteImageRejected(f"cannot resolve {host}") from exc
    if not addresses:
        raise RemoteImageRejected(f"cannot resolve {host}")
    blocked = [a for a in addresses if not is_public_address(a)]
    if blocked:
        raise RemoteImageRejected(
            f"{host} resolves to a non-public address; not fetched"
        )

    pinned = urlunsplit(
        ("https", _netloc(addresses[0], port), parts.path or "/", parts.query, "")
    )
    headers = {"Host": _netloc(host, port), "Accept": "image/*"}
    try:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream(
                "GET", pinned, headers=headers, extensions={"sni_hostname": host}
            ) as response:
                if response.status_code != 200:
                    raise RemoteImageRejected(
                        f"image URL answered HTTP {response.status_code}"
                    )
                content_type = (
                    response.headers.get("content-type", "").split(";", 1)[0].strip()
                ).lower()
                if not content_type.startswith("image/"):
                    raise RemoteImageRejected(
                        f"image URL returned {content_type or 'no content type'}"
                    )
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise RemoteImageRejected("remote image is too large")
                chunks: List[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise RemoteImageRejected("remote image is too large")
                    chunks.append(chunk)
    except RemoteImageRejected:
        raise
    except (httpx.HTTPError, OSError) as exc:
        raise RemoteImageRejected(
            f"fetching the image failed ({type(exc).__name__})"
        ) from exc

    return b"".join(chunks), content_type
