"""SSRF guard for server-side remote image fetches and image reads
off the event loop in the LLM clients.

Hermetic: no DNS lookup and no network — the resolver and the HTTP transport
are injected (``httpx.MockTransport``), and a guard fails the test if any
request reaches a transport it should never have reached.
"""

import asyncio
import base64
import io
import threading
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from mirix.helpers.remote_image_fetch import (
    RemoteImageRejected,
    fetch_remote_image,
    is_public_address,
)

pytestmark = pytest.mark.asyncio

PUBLIC_IP = "93.184.215.14"


def _png(size=(3, 2)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "green").save(buf, "PNG")
    return buf.getvalue()


def _resolver(*addresses):
    calls = []

    async def resolve(host, port):
        calls.append((host, port))
        return list(addresses)

    resolve.calls = calls
    return resolve


def _never_called():
    def handler(request):
        raise AssertionError(f"no request may be sent, got {request.url}")

    return httpx.MockTransport(handler)


def _serving(body=b"", status=200, headers=None, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, content=body, headers=headers or {})

    return httpx.MockTransport(handler)


class TestAddressClassification:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.3.4",
            "192.168.1.1",
            "169.254.169.254",  # cloud metadata
            "100.64.0.1",  # shared address space
            "0.0.0.0",
            "224.0.0.1",
            "::1",
            "fe80::1%eth0",
            "fc00::1",
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
            "::ffff:169.254.169.254",
            "not-an-ip",
        ],
    )
    async def test_non_public_addresses(self, address):
        assert not is_public_address(address)

    @pytest.mark.parametrize("address", [PUBLIC_IP, "8.8.8.8", "2606:4700::1111"])
    async def test_public_addresses(self, address):
        assert is_public_address(address)

    # IPv6 forms that carry an IPv4 destination are judged by it.
    @pytest.mark.parametrize(
        "address",
        [
            "64:ff9b::a9fe:a9fe",  # NAT64 well-known prefix -> 169.254.169.254
            "64:ff9b::7f00:1",  # -> 127.0.0.1
            "64:ff9b::a00:5",  # -> 10.0.0.5
            "64:ff9b::c0a8:101",  # -> 192.168.1.1
            "::ffff:0:a9fe:a9fe",  # IPv4-translated (SIIT) -> 169.254.169.254
            "::ffff:0:7f00:1",
        ],
    )
    async def test_embedded_private_ipv4_is_not_public(self, address):
        assert not is_public_address(address)

    @pytest.mark.parametrize(
        "address",
        [
            "64:ff9b::808:808",  # NAT64 to 8.8.8.8: a public destination
            "::ffff:0:808:808",
            "::ffff:8.8.8.8",
        ],
    )
    async def test_embedded_public_ipv4_is_public(self, address):
        assert is_public_address(address)

    @pytest.mark.parametrize(
        "address",
        [
            "64:ff9b:1::a9fe:a9fe",  # local-use NAT64: IPv4 position unknown
            "64:ff9b:1:ffff::808:808",  # refused even when it looks public
            "::a9fe:a9fe",  # deprecated IPv4-compatible
            "::808:808",  # refused even when the IPv4 is public
            "2002:a9fe:a9fe::1",  # 6to4 embedding 169.254.169.254
            "2002:808:808::1",  # 6to4 is deprecated: refused outright
            "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo
        ],
    )
    async def test_unreliable_or_deprecated_embeddings_are_refused(self, address):
        assert not is_public_address(address)

    async def test_nat64_resolution_is_never_fetched(self):
        """A DNS64 answer for an internal host must not be dialed."""
        with pytest.raises(RemoteImageRejected, match="non-public"):
            await fetch_remote_image(
                "https://metadata.example.com/latest/",
                resolver=_resolver("64:ff9b::a9fe:a9fe"),
                transport=_never_called(),
            )


class TestRefusedBeforeConnecting:
    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/a.png",
            "ftp://example.com/a.png",
            "file:///etc/passwd",
            "https://user:pw@example.com/a.png",
            "https:///a.png",
            "https://example.com:99999/a.png",
            "",
        ],
    )
    async def test_bad_urls(self, url):
        resolve = _resolver(PUBLIC_IP)
        with pytest.raises(RemoteImageRejected):
            await fetch_remote_image(url, resolver=resolve, transport=_never_called())

    @pytest.mark.parametrize(
        "addresses",
        [
            ["169.254.169.254"],
            ["127.0.0.1"],
            ["10.1.2.3"],
            [PUBLIC_IP, "192.168.0.10"],  # any private answer taints the host
            [],
        ],
    )
    async def test_non_public_resolution(self, addresses):
        with pytest.raises(RemoteImageRejected):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                resolver=_resolver(*addresses),
                transport=_never_called(),
            )

    async def test_literal_metadata_ip(self):
        async def system_like(host, port):
            return [host]

        with pytest.raises(RemoteImageRejected, match="non-public"):
            await fetch_remote_image(
                "https://169.254.169.254/latest/meta-data/",
                resolver=system_like,
                transport=_never_called(),
            )

    async def test_resolution_failure(self):
        async def failing(host, port):
            raise OSError("NXDOMAIN")

        with pytest.raises(RemoteImageRejected, match="cannot resolve"):
            await fetch_remote_image(
                "https://nx.example.com/a.png",
                resolver=failing,
                transport=_never_called(),
            )


class TestResponses:
    async def test_pinned_to_the_vetted_address_with_the_real_host(self):
        seen = []
        data, mime = await fetch_remote_image(
            "https://images.example.com:8443/p/a.png?v=1",
            resolver=_resolver(PUBLIC_IP),
            transport=_serving(
                _png(), headers={"content-type": "image/png"}, seen=seen
            ),
        )
        assert data == _png() and mime == "image/png"
        [request] = seen
        assert request.url.host == PUBLIC_IP and request.url.port == 8443
        assert request.url.path == "/p/a.png" and request.url.query == b"v=1"
        assert request.headers["host"] == "images.example.com:8443"
        assert request.extensions["sni_hostname"] == "images.example.com"

    @pytest.mark.parametrize("status", [301, 302, 307, 308])
    async def test_redirects_are_not_followed(self, status):
        seen = []
        with pytest.raises(RemoteImageRejected, match=str(status)):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                resolver=_resolver(PUBLIC_IP),
                transport=_serving(
                    status=status,
                    headers={"location": "http://169.254.169.254/"},
                    seen=seen,
                ),
            )
        assert len(seen) == 1

    async def test_non_image_content_type(self):
        with pytest.raises(RemoteImageRejected, match="text/html"):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                resolver=_resolver(PUBLIC_IP),
                transport=_serving(b"<html>", headers={"content-type": "text/html"}),
            )

    async def test_declared_oversize(self):
        with pytest.raises(RemoteImageRejected, match="too large"):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                max_bytes=10,
                resolver=_resolver(PUBLIC_IP),
                transport=_serving(
                    _png(),
                    headers={"content-type": "image/png"},
                ),
            )

    async def test_streamed_oversize_without_content_length(self):
        big = b"x" * 5000

        async def body():
            for i in range(0, len(big), 1000):
                yield big[i : i + 1000]

        def handler(request):
            return httpx.Response(
                200, content=body(), headers={"content-type": "image/png"}
            )

        with pytest.raises(RemoteImageRejected, match="too large"):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                max_bytes=2500,
                resolver=_resolver(PUBLIC_IP),
                transport=httpx.MockTransport(handler),
            )

    async def test_bytes_that_are_not_a_supported_image(self):
        with pytest.raises(RemoteImageRejected):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                resolver=_resolver(PUBLIC_IP),
                transport=_serving(
                    b"GIF89a-garbage", headers={"content-type": "image/gif"}
                ),
            )

    async def test_a_trickling_server_hits_the_total_deadline(self):
        """Every read arrives inside the per-read timeout, but the
        whole fetch is bounded (asyncio.wait_for) — no slow-drip hold."""
        reads = []

        async def trickle():
            while True:  # one byte every 0.05 s, forever
                await asyncio.sleep(0.05)
                reads.append(1)
                yield b"x"

        def handler(request):
            return httpx.Response(
                200, content=trickle(), headers={"content-type": "image/png"}
            )

        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(RemoteImageRejected, match="longer than 0.3s"):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                timeout=5,
                total_timeout=0.3,
                resolver=_resolver(PUBLIC_IP),
                transport=httpx.MockTransport(handler),
            )
        elapsed = loop.time() - started
        assert 0.25 <= elapsed < 2.0
        assert reads  # it really was streaming when the deadline fired

    async def test_the_deadline_covers_resolution(self):
        async def hanging(host, port):
            await asyncio.sleep(30)
            return [PUBLIC_IP]

        with pytest.raises(RemoteImageRejected, match="longer than"):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                total_timeout=0.1,
                resolver=hanging,
                transport=_never_called(),
            )

    async def test_the_default_deadline_is_configured(self, monkeypatch):
        import mirix.constants as constants

        monkeypatch.setattr(constants, "REMOTE_IMAGE_FETCH_TOTAL_TIMEOUT_SECONDS", 0.1)

        async def hanging(host, port):
            await asyncio.sleep(30)
            return [PUBLIC_IP]

        with pytest.raises(RemoteImageRejected, match="longer than 0.1s"):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                resolver=hanging,
                transport=_never_called(),
            )

    async def test_transport_errors_become_rejections(self):
        def handler(request):
            raise httpx.ConnectTimeout("slow", request=request)

        with pytest.raises(RemoteImageRejected, match="ConnectTimeout"):
            await fetch_remote_image(
                "https://images.example.com/a.png",
                resolver=_resolver(PUBLIC_IP),
                transport=httpx.MockTransport(handler),
            )


# --------------------------------------------------------------------------- #
# LLM clients                                                                  #
# --------------------------------------------------------------------------- #


class _Files:
    def __init__(self, **files):
        self.files = files

    async def get_file_metadata_by_id(self, file_id):
        return self.files[file_id]


def _client(cls, files):
    client = object.__new__(cls)
    client.file_manager = files
    return client


class TestGoogleClientRemoteImages:
    async def test_private_url_is_never_fetched_and_becomes_a_placeholder(
        self, monkeypatch
    ):
        from mirix.helpers import remote_image_fetch
        from mirix.llm_api.google_ai_client import GoogleAIClient

        attempted = []

        async def resolver(host, port):
            attempted.append(host)
            return ["169.254.169.254"]

        monkeypatch.setattr(remote_image_fetch, "_system_resolver", resolver)

        def no_client(*args, **kwargs):
            raise AssertionError("no HTTP client may be created")

        monkeypatch.setattr(remote_image_fetch.httpx, "AsyncClient", no_client)
        url = "https://metadata.internal/latest/meta-data/iam"
        client = _client(
            GoogleAIClient,
            _Files(img=SimpleNamespace(source_url=url, file_path=None, file_type=None)),
        )
        [message] = await client.fill_image_content_in_messages(
            [{"role": "user", "parts": [{"image_id": "img"}]}]
        )
        assert attempted == ["metadata.internal"]
        texts = [p.get("text") for p in message["parts"]]
        assert f"[Remote image {url} could not be loaded]" in texts
        assert not any("inline_data" in p for p in message["parts"])

    async def test_public_url_is_inlined(self, monkeypatch):
        from mirix.llm_api.google_ai_client import GoogleAIClient

        async def fake_fetch(url, **kwargs):
            return _png(), "image/png"

        import mirix.helpers.remote_image_fetch as remote_image_fetch

        monkeypatch.setattr(remote_image_fetch, "fetch_remote_image", fake_fetch)
        client = _client(
            GoogleAIClient,
            _Files(
                img=SimpleNamespace(
                    source_url="https://images.example.com/a.png",
                    file_path=None,
                    file_type=None,
                )
            ),
        )
        [message] = await client.fill_image_content_in_messages(
            [{"role": "user", "parts": [{"image_id": "img"}]}]
        )
        [inline] = [p["inline_data"] for p in message["parts"] if "inline_data" in p]
        assert inline["mime_type"] == "image/png"
        assert base64.b64decode(inline["data"]) == _png()


class TestImageReadsOffTheLoop:
    """Local image files are read in a worker thread, not on the loop."""

    @pytest.fixture
    def image_file(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(_png())
        return str(path)

    def _recording(self, real):
        threads = []

        def wrapper(*args, **kwargs):
            threads.append(threading.current_thread())
            return real(*args, **kwargs)

        wrapper.threads = threads
        return wrapper

    async def test_google(self, monkeypatch, image_file):
        from mirix.llm_api import google_ai_client
        from mirix.llm_api.google_ai_client import GoogleAIClient

        rec = self._recording(google_ai_client.read_file_base64)
        monkeypatch.setattr(google_ai_client, "read_file_base64", rec)
        client = _client(
            GoogleAIClient,
            _Files(
                img=SimpleNamespace(
                    source_url=None, file_path=image_file, file_type="image/png"
                )
            ),
        )
        await client.fill_image_content_in_messages(
            [{"role": "user", "parts": [{"image_id": "img"}]}]
        )
        assert rec.threads and threading.main_thread() not in rec.threads

    async def test_anthropic(self, monkeypatch, image_file):
        from mirix.llm_api import anthropic_client
        from mirix.llm_api.anthropic_client import AnthropicClient

        rec = self._recording(anthropic_client.read_file_base64)
        monkeypatch.setattr(anthropic_client, "read_file_base64", rec)
        client = _client(
            AnthropicClient,
            _Files(img=SimpleNamespace(source_url=None, file_path=image_file)),
        )
        [message] = await client.fill_image_content_in_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_id": "img"}]}]
        )
        assert rec.threads and threading.main_thread() not in rec.threads
        [image] = [c for c in message["content"] if c["type"] == "image"]
        assert base64.b64decode(image["source"]["data"]) == _png()

    async def test_openai(self, monkeypatch, image_file):
        from mirix.llm_api import openai_client
        from mirix.llm_api.openai_client import OpenAIClient

        rec = self._recording(openai_client.encode_image)
        monkeypatch.setattr(openai_client, "encode_image", rec)
        client = _client(
            OpenAIClient,
            _Files(img=SimpleNamespace(source_url=None, file_path=image_file)),
        )
        message = SimpleNamespace(
            role="user",
            content=[{"type": "image_url", "image_id": "img", "detail": "auto"}],
        )
        await client.fill_image_content_in_messages([message])
        assert rec.threads and threading.main_thread() not in rec.threads

    async def test_the_loop_keeps_running_during_a_slow_read(
        self, monkeypatch, image_file
    ):
        from mirix.llm_api import anthropic_client
        from mirix.llm_api.anthropic_client import AnthropicClient

        release = threading.Event()
        real = anthropic_client.read_file_base64

        def slow_read(path):
            assert release.wait(5)
            return real(path)

        monkeypatch.setattr(anthropic_client, "read_file_base64", slow_read)
        client = _client(
            AnthropicClient,
            _Files(img=SimpleNamespace(source_url=None, file_path=image_file)),
        )
        task = asyncio.create_task(
            client.fill_image_content_in_messages(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_id": "img"}],
                    }
                ]
            )
        )
        await asyncio.sleep(0.05)  # would never return if the read blocked the loop
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, 5)
