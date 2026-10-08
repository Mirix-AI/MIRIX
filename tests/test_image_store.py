"""Image storage for memory-add payloads.

Hermetic: a throwaway SQLite database behind ``db_context`` and a tmp images
directory. Covers decoding and inspection, the content-addressed per-user
layout, ``files`` rows carrying ``user_id``, remote URLs recorded without any
download, release semantics (a disk file goes only with its last row, and only
files this module wrote are ever unlinked), user erasure, and that every disk
and decode step runs off the event loop.
"""

import asyncio
import base64
import io
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
from mirix.orm.base import Base
from mirix.orm.file import FileMetadata as FileModel
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.user import User as UserORM
from mirix.services import image_store
from mirix.services.conversation_content import ImagePart, parse_messages
from mirix.services.image_store import (
    ImageDecodeError,
    decode_image_payload,
    delete_user_image_files,
    discard_image_refs,
    is_managed_image_path,
    prepare_images,
    release_image_files,
    scope_dir_name,
    store_image_part,
    store_remote_image_ref,
    unlink_image_paths,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _image_bytes(fmt="PNG", size=(4, 3), color="blue") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("image_store") / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    local = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _ctx():
        async with local() as session:
            try:
                yield session
            finally:
                await session.close()

    yield _ctx
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def env(sm, monkeypatch, tmp_path):
    import mirix.server.server as server_module
    from mirix.settings import settings

    monkeypatch.setattr(server_module, "db_context", sm)
    monkeypatch.setattr(settings, "images_dir", tmp_path / "images")
    org_id = f"org-{uuid.uuid4().hex[:8]}"
    users = [f"user-{uuid.uuid4().hex[:8]}" for _ in range(2)]
    async with sm() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        for uid in users:
            db.add(
                UserORM(
                    id=uid,
                    name=uid,
                    organization_id=org_id,
                    status="active",
                    timezone="UTC",
                )
            )
        await db.commit()

    class Env:
        pass

    e = Env()
    e.sm, e.org, e.user, e.other_user, e.root = (
        sm,
        org_id,
        users[0],
        users[1],
        tmp_path / "images",
    )
    return e


async def _files(sm, **filters):
    async with sm() as db:
        stmt = select(FileModel)
        for key, value in filters.items():
            stmt = stmt.where(getattr(FileModel, key) == value)
        return list((await db.execute(stmt)).scalars().all())


# --------------------------------------------------------------------------- #
# Decoding                                                                     #
# --------------------------------------------------------------------------- #


class TestDecode:
    async def test_png_and_jpeg_are_inspected(self):
        png = decode_image_payload(_b64(_image_bytes("PNG", (7, 5))))
        assert (png.mime_type, png.width, png.height, png.extension) == (
            "image/png",
            7,
            5,
            "png",
        )
        jpeg = decode_image_payload(_b64(_image_bytes("JPEG", (2, 9))), "image/png")
        # The real format wins over a wrong declared mime type.
        assert (jpeg.mime_type, jpeg.width, jpeg.height, jpeg.extension) == (
            "image/jpeg",
            2,
            9,
            "jpg",
        )

    async def test_sha256_and_size_describe_the_bytes(self):
        import hashlib

        raw = _image_bytes()
        prepared = decode_image_payload(_b64(raw))
        assert prepared.sha256 == hashlib.sha256(raw).hexdigest()
        assert prepared.size == len(raw)

    async def test_whitespace_and_missing_padding_are_tolerated(self):
        encoded = _b64(_image_bytes()).rstrip("=")
        wrapped = "\n".join(encoded[i : i + 20] for i in range(0, len(encoded), 20))
        assert decode_image_payload(wrapped).width == 4

    @pytest.mark.parametrize(
        "payload, fragment",
        [
            ("!!!not base64!!!", "base64"),
            (_b64(b"plain text, not an image"), "decodable"),
            ("", "empty"),
        ],
    )
    async def test_bad_payloads_raise_decode_errors(self, payload, fragment):
        with pytest.raises(ImageDecodeError) as exc:
            decode_image_payload(payload)
        assert fragment in str(exc.value)

    async def test_prepare_images_decodes_once_off_the_loop(self, monkeypatch):
        calls = []
        real_to_thread = asyncio.to_thread

        async def spy_to_thread(fn, *args, **kwargs):
            calls.append(fn.__name__)
            return await real_to_thread(fn, *args, **kwargs)

        monkeypatch.setattr(image_store.asyncio, "to_thread", spy_to_thread)
        part = ImagePart(data_b64=_b64(_image_bytes()))
        remote = ImagePart(url="https://x.test/a.png")
        await prepare_images([part, remote])
        await prepare_images([part])
        assert calls == ["decode_image_payload"]
        assert part.prepared.width == 4 and remote.prepared is None

    async def test_prepare_images_fails_before_anything_is_written(self, env):
        good = ImagePart(data_b64=_b64(_image_bytes()))
        bad = ImagePart(data_b64=_b64(b"nope"))
        with pytest.raises(ImageDecodeError):
            await prepare_images([good, bad])
        assert not env.root.exists()
        assert await _files(env.sm, organization_id=env.org) == []


# --------------------------------------------------------------------------- #
# Store                                                                        #
# --------------------------------------------------------------------------- #


class TestStore:
    async def test_inline_image_is_written_and_recorded_with_user_id(self, env):
        raw = _image_bytes("PNG", (6, 4))
        ref = await store_image_part(
            ImagePart(data_b64=_b64(raw), detail="high"),
            organization_id=env.org,
            user_id=env.user,
        )
        assert ref["type"] == "image_ref"
        assert (ref["mime_type"], ref["width"], ref["height"], ref["bytes"]) == (
            "image/png",
            6,
            4,
            len(raw),
        )
        assert ref["detail"] == "high"
        [row] = await _files(env.sm, id=ref["file_id"])
        assert row.user_id == env.user and row.organization_id == env.org
        path = Path(row.file_path)
        assert path.read_bytes() == raw
        assert path.name == f"{ref['sha256']}.png"
        assert path.parent.name == scope_dir_name(env.org, env.user)
        assert path.parent.parent == env.root
        assert is_managed_image_path(row.file_path)

    async def test_each_store_owns_a_row_but_same_bytes_share_a_file(self, env):
        raw = _image_bytes(color="green")
        first = await store_image_part(
            ImagePart(data_b64=_b64(raw)), organization_id=env.org, user_id=env.user
        )
        second = await store_image_part(
            ImagePart(data_b64=_b64(raw)), organization_id=env.org, user_id=env.user
        )
        assert first["file_id"] != second["file_id"]
        [a] = await _files(env.sm, id=first["file_id"])
        [b] = await _files(env.sm, id=second["file_id"])
        assert a.file_path == b.file_path

    async def test_users_never_share_a_disk_file(self, env):
        raw = _image_bytes(color="yellow")
        mine = await store_image_part(
            ImagePart(data_b64=_b64(raw)), organization_id=env.org, user_id=env.user
        )
        theirs = await store_image_part(
            ImagePart(data_b64=_b64(raw)),
            organization_id=env.org,
            user_id=env.other_user,
        )
        [a] = await _files(env.sm, id=mine["file_id"])
        [b] = await _files(env.sm, id=theirs["file_id"])
        assert a.file_path != b.file_path

    async def test_remote_url_is_recorded_never_downloaded(self, env, monkeypatch):
        import httpx

        def no_network(*args, **kwargs):
            raise AssertionError("remote images must not be downloaded")

        monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
        monkeypatch.setattr(httpx.Client, "send", no_network)
        [msg] = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/p/chart.png"},
                        }
                    ],
                }
            ]
        )
        ref = await store_image_part(
            msg.parts[0], organization_id=env.org, user_id=env.user
        )
        assert ref == {
            "type": "image_ref",
            "file_id": ref["file_id"],
            "url": "https://x.test/p/chart.png",
            "mime_type": "image/png",
        }
        [row] = await _files(env.sm, id=ref["file_id"])
        assert row.source_url == "https://x.test/p/chart.png" and row.file_path is None
        assert row.user_id == env.user

    async def test_remote_url_without_image_extension_has_no_mime(self, env):
        ref = await store_remote_image_ref(
            "https://x.test/render?id=3", organization_id=env.org, user_id=env.user
        )
        assert "mime_type" not in ref

    async def test_disk_writes_run_in_threads(self, env, monkeypatch):
        calls = []
        real_to_thread = asyncio.to_thread

        async def spy_to_thread(fn, *args, **kwargs):
            calls.append(fn.__name__)
            return await real_to_thread(fn, *args, **kwargs)

        monkeypatch.setattr(image_store.asyncio, "to_thread", spy_to_thread)
        await store_image_part(
            ImagePart(data_b64=_b64(_image_bytes(color="purple"))),
            organization_id=env.org,
            user_id=env.user,
        )
        assert calls.count("decode_image_payload") == 1
        assert (
            calls.count("_write_image_file") == 2
        )  # write, then re-check after the row exists

    async def test_file_is_restored_if_released_between_write_and_row(
        self, env, monkeypatch
    ):
        """A concurrent release of the same bytes may unlink the file after our
        write; once our row exists the file is written again."""
        from mirix.services.file_manager import FileManager

        raw = _image_bytes(color="orange")
        real_create = FileManager.create_file_metadata

        async def create_then_lose_the_file(self, pydantic_file):
            os.unlink(pydantic_file.file_path)
            return await real_create(self, pydantic_file)

        monkeypatch.setattr(
            FileManager, "create_file_metadata", create_then_lose_the_file
        )
        ref = await store_image_part(
            ImagePart(data_b64=_b64(raw)), organization_id=env.org, user_id=env.user
        )
        [row] = await _files(env.sm, id=ref["file_id"])
        assert Path(row.file_path).read_bytes() == raw


# --------------------------------------------------------------------------- #
# Release                                                                      #
# --------------------------------------------------------------------------- #


class TestRelease:
    async def test_file_goes_only_with_its_last_row(self, env):
        raw = _image_bytes(color="cyan")
        a = await store_image_part(
            ImagePart(data_b64=_b64(raw)), organization_id=env.org, user_id=env.user
        )
        b = await store_image_part(
            ImagePart(data_b64=_b64(raw)), organization_id=env.org, user_id=env.user
        )
        [row] = await _files(env.sm, id=a["file_id"])
        path = Path(row.file_path)

        async with env.sm() as db:
            paths = await release_image_files(
                db, organization_id=env.org, file_ids=[a["file_id"]]
            )
            await db.commit()
        assert paths == []
        await unlink_image_paths(paths)
        assert path.exists() and await _files(env.sm, id=a["file_id"]) == []

        async with env.sm() as db:
            paths = await release_image_files(
                db, organization_id=env.org, file_ids=[b["file_id"]]
            )
            await db.commit()
        assert paths == [str(path)]
        assert await unlink_image_paths(paths) == 1
        assert not path.exists()

    async def test_release_is_scoped_to_the_organization(self, env):
        ref = await store_image_part(
            ImagePart(data_b64=_b64(_image_bytes(color="white"))),
            organization_id=env.org,
            user_id=env.user,
        )
        async with env.sm() as db:
            assert (
                await release_image_files(
                    db, organization_id="org-other", file_ids=[ref["file_id"]]
                )
                == []
            )
            await db.commit()
        assert len(await _files(env.sm, id=ref["file_id"])) == 1

    async def test_unmanaged_paths_are_never_unlinked(self, env, tmp_path):
        """A legacy files row can point at a caller's own file (file_uri); release
        drops the row but must never delete that file."""
        precious = tmp_path / "report.pdf"
        precious.write_bytes(b"%PDF")
        async with env.sm() as db:
            row = FileModel(
                id=f"file-{uuid.uuid4()}",
                organization_id=env.org,
                user_id=env.user,
                file_path=str(precious),
                file_name="report.pdf",
            )
            db.add(row)
            await db.commit()
            file_id = row.id
        async with env.sm() as db:
            paths = await release_image_files(
                db, organization_id=env.org, file_ids=[file_id]
            )
            await db.commit()
        assert paths == []
        assert await unlink_image_paths([str(precious)]) == 0
        assert precious.exists()
        assert not is_managed_image_path(str(precious))
        assert not is_managed_image_path(None)

    async def test_discard_undoes_a_failed_write(self, env):
        ref = await store_image_part(
            ImagePart(data_b64=_b64(_image_bytes(color="black"))),
            organization_id=env.org,
            user_id=env.user,
        )
        [row] = await _files(env.sm, id=ref["file_id"])
        await discard_image_refs([ref], organization_id=env.org)
        assert await _files(env.sm, id=ref["file_id"]) == []
        assert not Path(row.file_path).exists()

    async def test_discard_never_raises(self, env, monkeypatch):
        async def broken(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(image_store, "release_image_files", broken)
        await discard_image_refs([{"file_id": "file-x"}], organization_id=env.org)

    async def test_user_erasure_deletes_only_that_users_images(self, env):
        mine = await store_image_part(
            ImagePart(data_b64=_b64(_image_bytes(color="gray"))),
            organization_id=env.org,
            user_id=env.user,
        )
        theirs = await store_image_part(
            ImagePart(data_b64=_b64(_image_bytes(color="gray"))),
            organization_id=env.org,
            user_id=env.other_user,
        )
        [mine_row] = await _files(env.sm, id=mine["file_id"])
        [their_row] = await _files(env.sm, id=theirs["file_id"])
        deleted = await delete_user_image_files(env.user)
        assert deleted >= 1
        assert await _files(env.sm, user_id=env.user) == []
        assert not Path(mine_row.file_path).exists()
        assert Path(their_row.file_path).exists()
        assert len(await _files(env.sm, id=theirs["file_id"])) == 1


# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #


class TestIntakeBounds:
    """provider-supported formats only, bounded bytes and pixels."""

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "GIF", "WEBP"])
    async def test_supported_formats_pass(self, fmt):
        prepared = decode_image_payload(_b64(_image_bytes(fmt=fmt)))
        assert prepared.mime_type.startswith("image/")

    @pytest.mark.parametrize("fmt", ["TIFF", "BMP", "ICO"])
    async def test_other_formats_are_refused(self, fmt):
        with pytest.raises(ImageDecodeError, match="unsupported image format"):
            decode_image_payload(_b64(_image_bytes(fmt=fmt, size=(16, 16))))

    async def test_oversized_payload_is_refused_before_decoding(self, monkeypatch):
        import mirix.constants as constants

        monkeypatch.setattr(constants, "MAX_IMAGE_BYTES", 10)

        def forbidden(*args, **kwargs):
            raise AssertionError("an oversized payload must not be decoded")

        monkeypatch.setattr(image_store.base64, "b64decode", forbidden)
        with pytest.raises(ImageDecodeError, match="limit is 10 bytes"):
            decode_image_payload(_b64(b"x" * 64))

    async def test_pixel_cap(self, monkeypatch):
        import mirix.constants as constants

        monkeypatch.setattr(constants, "MAX_IMAGE_PIXELS", 11)
        with pytest.raises(ImageDecodeError, match="pixels"):
            decode_image_payload(_b64(_image_bytes(size=(4, 3))))


class TestRaceSafeUnlink:
    """A release racing a store of the same bytes never leaves a files row
    pointing at a deleted file."""

    async def test_a_reference_created_during_the_release_keeps_the_file(
        self, env, monkeypatch
    ):
        raw = _image_bytes(color="purple")
        a = await store_image_part(
            ImagePart(data_b64=_b64(raw)), organization_id=env.org, user_id=env.user
        )
        [row] = await _files(env.sm, id=a["file_id"])
        path = Path(row.file_path)
        async with env.sm() as db:
            paths = await release_image_files(
                db, organization_id=env.org, file_ids=[a["file_id"]]
            )
            await db.commit()
        assert paths == [str(path)]

        # The concurrent store of the same bytes runs while the file is moved
        # aside, i.e. after the release's own in-transaction check: it rewrites
        # the file and commits a new files row before the re-check.
        real_referenced = image_store._referenced_paths
        stored = {}

        async def store_then_recheck(paths_, session_maker):
            assert not path.exists()  # tombstoned
            stored["ref"] = await store_image_part(
                ImagePart(data_b64=_b64(raw)),
                organization_id=env.org,
                user_id=env.user,
            )
            return await real_referenced(paths_, session_maker)

        monkeypatch.setattr(image_store, "_referenced_paths", store_then_recheck)
        assert await unlink_image_paths(paths) == 0
        [new_row] = await _files(env.sm, id=stored["ref"]["file_id"])
        assert new_row.file_path == str(path)
        assert path.read_bytes() == raw  # the file is there for the new row
        assert not list(path.parent.glob("*.released-*"))

    async def test_an_unsure_recheck_keeps_every_file(self, env, monkeypatch):
        ref = await store_image_part(
            ImagePart(data_b64=_b64(_image_bytes(color="teal"))),
            organization_id=env.org,
            user_id=env.user,
        )
        [row] = await _files(env.sm, id=ref["file_id"])
        path = Path(row.file_path)

        async def broken(paths_, session_maker):
            raise RuntimeError("db down")

        monkeypatch.setattr(image_store, "_referenced_paths", broken)
        assert await unlink_image_paths([str(path)]) == 0
        assert path.exists() and not list(path.parent.glob("*.released-*"))

    async def test_unreferenced_files_are_deleted_and_no_tombstone_remains(
        self, env
    ):
        ref = await store_image_part(
            ImagePart(data_b64=_b64(_image_bytes(color="olive"))),
            organization_id=env.org,
            user_id=env.user,
        )
        [row] = await _files(env.sm, id=ref["file_id"])
        path = Path(row.file_path)
        async with env.sm() as db:
            paths = await release_image_files(
                db, organization_id=env.org, file_ids=[ref["file_id"]]
            )
            await db.commit()
        assert await unlink_image_paths(paths) == 1
        assert not path.exists() and not list(path.parent.glob("*.released-*"))


class TestLegacyImages:
    """Main-era images (<images root>/image_<16 hex>.<ext>) are managed too, so
    releasing their last row removes them; anything else never is."""

    async def test_legacy_layout_is_managed_only_inside_the_images_root(
        self, env, tmp_path
    ):
        env.root.mkdir(parents=True, exist_ok=True)
        inside = env.root / "image_0123456789abcdef.png"
        outside = tmp_path / "elsewhere" / "image_0123456789abcdef.png"
        assert is_managed_image_path(str(inside))
        assert not is_managed_image_path(str(outside))
        assert not is_managed_image_path(str(env.root / "image_nothex.png"))


class TestMessageImageIds:
    async def test_collects_image_ids_from_objects_and_dicts(self):
        from mirix.schemas.mirix_message_content import (
            FileContent,
            ImageContent,
            TextContent,
        )

        ids = image_store.message_image_ids(
            [
                [TextContent(text="x"), ImageContent(image_id="file-a")],
                [{"type": "image_url", "image_id": "file-b"}, {"type": "text"}],
                [FileContent(file_id="file-doc")],
                None,
                [ImageContent(image_id="file-a")],
            ]
        )
        assert ids == ["file-a", "file-b"]
