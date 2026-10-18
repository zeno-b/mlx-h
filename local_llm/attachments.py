"""Per-user uploads (images, audio, documents) that a chat turn can attach.

The chat route used to accept only text, so "what is in this screenshot?" or
"summarise this PDF" meant pasting, and an image could not be sent at all. An
upload is now a row in the `attachments` table plus a file under
DATA_DIR/attachments/<user_id>/ (or ATTACHMENTS_DIR), and a chat request names
it by id. content_parts_for() turns ids into OpenAI content parts: images as
data: URLs (the agent sends them only when the served model can see, see
vision.served_vision), documents as text extracted by the same readers the
read_file tool uses, audio as its transcript (audio.py).

Two rules shape everything here:
  * The client's Content-Type and file name are never trusted. The type is
    sniffed from magic bytes and only a short allowlist is stored; anything else
    is refused, so an "image" that is really HTML can never be served back.
  * Every read is owner-checked, and a miss is a 404 whether the id does not
    exist or belongs to someone else, so ids cannot be probed. Served files carry
    nosniff, a sandboxing CSP, and `attachment` disposition for anything that is
    not an image or audio.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
import re
import secrets
import struct
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403

# Endpoint annotations are strings (from __future__ import annotations) that
# FastAPI resolves against this module's globals, so the Request type must live
# here under a name of its own. Guarded so --selftest imports without FastAPI.
try:
    from fastapi import Request as _AttRequest  # noqa: F401
except Exception:  # pragma: no cover
    _AttRequest = Any  # type: ignore


ATTACHMENTS_SCHEMA_NAME = "attachments"

# Sniffed type -> stored file extension. The keys are the only types stored.
ATTACHMENT_IMAGE_TYPES: dict[str, str] = {
    "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif",
}
ATTACHMENT_AUDIO_TYPES: dict[str, str] = {
    "audio/wav": "wav", "audio/mpeg": "mp3", "audio/mp4": "m4a",
    "audio/webm": "webm", "audio/ogg": "ogg",
}
_ATT_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_ATT_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ATTACHMENT_DOCUMENT_TYPES: dict[str, str] = {
    "application/pdf": "pdf", _ATT_DOCX: "docx", _ATT_XLSX: "xlsx",
    "text/plain": "txt", "text/markdown": "md", "text/csv": "csv",
    "text/tab-separated-values": "tsv", "application/json": "json",
}
# Text documents keep their own extension (the reader is picked by it), mapped
# to the stored type. Anything else that is valid UTF-8 is text/plain.
_ATT_TEXT_EXTS: dict[str, str] = {
    "md": "text/markdown", "markdown": "text/markdown", "csv": "text/csv",
    "tsv": "text/tab-separated-values", "json": "application/json",
    "ipynb": "application/json",
}
# ISO base media brands that are audio containers (m4a and friends). Image
# brands (heic, avif, mif1) are deliberately absent.
_ATT_M4A_BRANDS = frozenset({b"M4A ", b"M4B ", b"mp42", b"mp41", b"isom", b"iso2",
                             b"iso5", b"iso6", b"dash", b"3gp4", b"3gp5"})
# A decompression bomb is a small file with a huge canvas; the model server
# decodes it, not us, so refuse the canvas up front.
_ATT_MAX_PIXELS = 50_000_000
_ATT_ID_RE = re.compile(r"^att_[0-9a-f]{24}$")
_ATT_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]")

# Set by register_attachments: content_parts_for() is called with (db, user,
# ids) only, so it reads limits from here. Mutated in place, never rebound,
# because star imports copy the binding.
_ATT_RUNTIME: dict[str, Any] = {"config": None}
# audio.py installs its transcriber here (an attachments -> audio import would
# be circular). fn(store, row) -> transcript text.
ATTACHMENT_HOOKS: dict[str, Any] = {"transcribe": None}

_att_logger = get_logger("attachments")


class AttachmentError(ValueError):
    """A refused upload or read, with the HTTP status it maps to."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class SniffedFile:
    kind: str                   # image | audio | document
    mime: str
    ext: str
    width: int | None = None
    height: int | None = None


@dataclass
class UploadBody:
    data: bytes
    filename: str = ""
    fields: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def _att_schema(conn) -> None:
    """Idempotent. `path` is relative to the attachments root, so moving
    DATA_DIR (or setting ATTACHMENTS_DIR to it) keeps every row valid."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            conversation_id TEXT,
            kind TEXT NOT NULL,
            mime TEXT NOT NULL,
            filename TEXT NOT NULL DEFAULT '',
            path TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            transcript TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_attachments_user "
                 "ON attachments(user_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_attachments_conversation "
                 "ON attachments(conversation_id)")


register_schema(ATTACHMENTS_SCHEMA_NAME, _att_schema)


# --------------------------------------------------------------------------- #
# Sniffing
# --------------------------------------------------------------------------- #
def _att_png_size(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    return None


def _att_gif_size(data: bytes) -> tuple[int, int] | None:
    return struct.unpack("<HH", data[6:10]) if len(data) >= 10 else None


def _att_webp_size(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X" and len(data) >= 30:
        return (int.from_bytes(data[24:27], "little") + 1,
                int.from_bytes(data[27:30], "little") + 1)
    return None


_ATT_JPEG_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                           0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def _att_jpeg_size(data: bytes) -> tuple[int, int] | None:
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:          # fill byte
            i += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image / start of scan: no SOF seen
            return None
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in _ATT_JPEG_SOF and i + 9 <= n:
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + length
    return None


def _att_is_mp3(data: bytes) -> bool:
    if data[:3] == b"ID3":
        return True
    # MPEG audio frame sync; layer bits 00 are reserved (that is AAC ADTS).
    return (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
            and ((data[1] >> 1) & 0x03) != 0)


def _att_text_type(data: bytes, ext: str) -> str | None:
    head = data[:8192]
    if b"\x00" in head:
        return None
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    return _ATT_TEXT_EXTS.get(ext, "text/plain")


def _att_ext_of(filename: str) -> str:
    name = str(filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def sniff_attachment(data: bytes, filename: str = "") -> SniffedFile:
    """The real type of an upload from its bytes, or AttachmentError(415).

    The file name only picks between text formats (csv vs markdown), never
    between binary types: a .png that is really a PDF is stored as a PDF.
    """
    if not data:
        raise AttachmentError("the upload is empty", 400)
    ext = _att_ext_of(filename)
    size: tuple[int, int] | None = None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        mime, size = "image/png", _att_png_size(data)
    elif data[:3] == b"\xff\xd8\xff":
        mime, size = "image/jpeg", _att_jpeg_size(data)
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        mime, size = "image/gif", _att_gif_size(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime, size = "image/webp", _att_webp_size(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return SniffedFile("audio", "audio/wav", "wav")
    elif data[:4] == b"OggS":
        return SniffedFile("audio", "audio/ogg", "ogg")
    elif data[:4] == b"\x1a\x45\xdf\xa3":
        return SniffedFile("audio", "audio/webm", "webm")
    elif data[4:8] == b"ftyp" and data[8:12] in _ATT_M4A_BRANDS:
        return SniffedFile("audio", "audio/mp4", "m4a")
    elif _att_is_mp3(data):
        return SniffedFile("audio", "audio/mpeg", "mp3")
    elif data[:5] == b"%PDF-":
        return SniffedFile("document", "application/pdf", "pdf")
    elif data[:4] == b"PK\x03\x04":
        try:
            names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        except (zipfile.BadZipFile, ValueError, OSError):
            names = set()
        if "word/document.xml" in names:
            return SniffedFile("document", _ATT_DOCX, "docx")
        if "xl/workbook.xml" in names:
            return SniffedFile("document", _ATT_XLSX, "xlsx")
        raise AttachmentError("zip archives are not accepted (only .docx and .xlsx)", 415)
    else:
        text_type = _att_text_type(data, ext)
        if text_type is None:
            raise AttachmentError(
                "unsupported file type: images (png, jpeg, webp, gif), audio (wav, mp3, "
                "m4a, webm, ogg), pdf, docx, xlsx and UTF-8 text are accepted", 415)
        keep = ext if ext in _ATT_TEXT_EXTS or re.fullmatch(r"[a-z0-9]{1,8}", ext) else "txt"
        return SniffedFile("document", text_type, keep)
    if size is None:
        raise AttachmentError(f"could not read the {mime} header (corrupt image?)", 400)
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise AttachmentError("the image has no pixels", 400)
    if width * height > _ATT_MAX_PIXELS:
        raise AttachmentError(f"the image is too large ({width}x{height} pixels)", 413)
    return SniffedFile("image", mime, ATTACHMENT_IMAGE_TYPES[mime], width, height)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def _att_cfg(name: str, default: Any) -> Any:
    cfg = _ATT_RUNTIME.get("config")
    value = getattr(cfg, name, None) if cfg is not None else None
    return default if value is None else value


def _att_clean_name(filename: str) -> str:
    name = unquote(str(filename or "")).replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:200]


def _att_public(row: dict) -> dict:
    return {
        "id": row["id"], "kind": row["kind"], "mime": row["mime"], "bytes": row["bytes"],
        "filename": row.get("filename") or "", "width": row.get("width"),
        "height": row.get("height"), "sha256": row["sha256"],
        "conversation_id": row.get("conversation_id"), "created_at": row["created_at"],
        "transcribed": bool(row.get("transcript")),
    }


class AttachmentStore:
    """The attachments table plus its files. Stateless apart from the handles."""

    def __init__(self, db: Any, config: Any = None):
        self.db = db
        self.config = config

    def _setting(self, name: str, default: Any) -> Any:
        value = getattr(self.config, name, None) if self.config is not None else None
        return _att_cfg(name, default) if value is None else value

    def _ready(self) -> None:
        self.db.ensure_schema(ATTACHMENTS_SCHEMA_NAME)

    def root(self) -> Path:
        configured = str(self._setting("attachments_dir", "") or "").strip()
        if configured:
            return Path(configured).expanduser()
        # Next to the database, which is DATA_DIR in production and a temp
        # directory in tests: uploads never leak out of the data they index.
        return Path(self.db.db_path).resolve().parent / "attachments"

    def _user_dir(self, user_id: str) -> Path:
        segment = _ATT_SAFE_SEGMENT.sub("_", str(user_id))[:64].strip(".")
        if not segment:
            segment = "u" + hashlib.sha256(str(user_id).encode()).hexdigest()[:16]
        return self.root() / segment

    def path_of(self, row: dict) -> Path:
        root = self.root().resolve()
        target = (root / str(row["path"])).resolve()
        if root not in target.parents:
            raise LookupError("attachment file is outside the attachments root")
        return target

    def usage_bytes(self, user_id: str) -> int:
        self._ready()
        row = self.db.execute("SELECT COALESCE(SUM(bytes), 0) AS n FROM attachments "
                              "WHERE user_id = ?", (str(user_id),)).fetchone()
        return int(row["n"] or 0)

    def save(self, user_id: str, data: bytes, filename: str = "",
             conversation_id: str | None = None) -> dict:
        cap = int(self._setting("attachment_max_bytes", 20 * 1024 * 1024))
        if len(data) > cap:
            raise AttachmentError(f"upload exceeds ATTACHMENT_MAX_BYTES ({cap})", 413)
        sniffed = sniff_attachment(data, filename)
        quota_mb = int(self._setting("attachment_quota_mb", 0) or 0)
        if quota_mb > 0 and self.usage_bytes(user_id) + len(data) > quota_mb * 1024 * 1024:
            raise AttachmentError(
                f"attachment storage quota reached (ATTACHMENT_QUOTA_MB={quota_mb}); "
                "delete old attachments first", 413)
        self._ready()
        att_id = "att_" + secrets.token_hex(12)
        folder = self._user_dir(user_id)
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        final = folder / f"{att_id}.{sniffed.ext}"
        tmp = folder / f".{att_id}.part"
        tmp.write_bytes(data)
        os.replace(tmp, final)
        row = {
            "id": att_id, "user_id": str(user_id), "conversation_id": conversation_id or None,
            "kind": sniffed.kind, "mime": sniffed.mime,
            "filename": _att_clean_name(filename) or final.name,
            "path": str(final.relative_to(self.root())), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "width": sniffed.width, "height": sniffed.height, "transcript": None,
            "created_at": time.time(),
        }
        try:
            self.db.execute(
                "INSERT INTO attachments (id, user_id, conversation_id, kind, mime, filename, "
                "path, bytes, sha256, width, height, transcript, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(row[k] for k in ("id", "user_id", "conversation_id", "kind", "mime",
                                       "filename", "path", "bytes", "sha256", "width",
                                       "height", "transcript", "created_at")))
            self.db.commit()
        except Exception:
            final.unlink(missing_ok=True)
            raise
        return row

    def get(self, att_id: str, user_id: str | None) -> dict | None:
        """The row if it exists AND belongs to user_id (None = unscoped)."""
        if not _ATT_ID_RE.match(str(att_id or "")):
            return None
        self._ready()
        row = self.db.execute("SELECT * FROM attachments WHERE id = ?", (att_id,)).fetchone()
        if row is None:
            return None
        row = dict(row)
        if user_id is not None and row["user_id"] != str(user_id):
            return None
        return row

    def read(self, row: dict) -> bytes:
        try:
            return self.path_of(row).read_bytes()
        except FileNotFoundError:
            raise LookupError(f"attachment {row['id']} has no file") from None

    def list(self, user_id: str, conversation_id: str | None = None,
             limit: int = 100) -> list[dict]:
        self._ready()
        sql = "SELECT * FROM attachments WHERE user_id = ?"
        args: list[Any] = [str(user_id)]
        if conversation_id:
            sql += " AND conversation_id = ?"
            args.append(conversation_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, min(500, int(limit))))
        return [dict(r) for r in self.db.execute(sql, tuple(args)).fetchall()]

    def set_transcript(self, att_id: str, user_id: str, text: str) -> None:
        self._ready()
        self.db.execute("UPDATE attachments SET transcript = ? WHERE id = ? AND user_id = ?",
                        (text, att_id, str(user_id)))
        self.db.commit()

    def delete(self, att_id: str, user_id: str) -> bool:
        row = self.get(att_id, user_id)
        if row is None:
            return False
        self.db.execute("DELETE FROM attachments WHERE id = ? AND user_id = ?",
                        (att_id, str(user_id)))
        self.db.commit()
        try:
            self.path_of(row).unlink(missing_ok=True)
        except (LookupError, OSError) as exc:
            log(f"Attachment {att_id}: file not removed ({exc}).", logging.WARNING)
        return True


# --------------------------------------------------------------------------- #
# Content parts for a chat turn
# --------------------------------------------------------------------------- #
def attachment_text(path: Path, ext: str, cap: int) -> str:
    """Document text through the read_file tool's own readers.

    Those readers are ToolRegistry methods that never touch `self` beyond the
    PDF helper (a staticmethod), so they are called unbound rather than
    duplicated: an attachment and a read_file of the same file read the same.
    """
    readers = {
        "pdf": lambda: ToolRegistry._extract_pdf(str(path), cap),
        "docx": lambda: ToolRegistry._read_docx(None, path, cap),
        "xlsx": lambda: ToolRegistry._read_xlsx(None, path, cap),
        "ipynb": lambda: ToolRegistry._read_notebook(None, path, cap),
        "csv": lambda: ToolRegistry._read_tabular(None, path, cap, ","),
        "tsv": lambda: ToolRegistry._read_tabular(None, path, cap, "\t"),
        "json": lambda: ToolRegistry._read_json(None, path, cap),
    }
    reader = readers.get(ext)
    if reader is not None:
        try:
            return str(reader())
        except Exception as exc:  # noqa: BLE001
            if ext in ("pdf", "docx", "xlsx"):
                return f"[could not read {ext}: {exc}]"
    return path.read_bytes().decode("utf-8-sig", errors="replace")[:cap]


def content_parts_for(db: Any, user_id: str, ids: list[str]) -> list[dict]:
    """OpenAI content parts for attachment ids the user owns, in request order.

    LookupError for an id that is missing or not theirs (the route's 404), and
    ValueError when an audio attachment cannot be transcribed here (a 400 that
    says what to install), so a turn never runs on half its input.
    """
    store = AttachmentStore(db, _ATT_RUNTIME.get("config"))
    cap = int(_att_cfg("tool_raw_chars", 20000) or 20000)
    parts: list[dict] = []
    seen: set[str] = set()
    for att_id in ids or ():
        att_id = str(att_id).strip()
        if not att_id or att_id in seen:
            continue
        seen.add(att_id)
        row = store.get(att_id, user_id)
        if row is None:
            raise LookupError(f"attachment {att_id} not found")
        name = row.get("filename") or att_id
        if row["kind"] == "image":
            data = base64.b64encode(store.read(row)).decode("ascii")
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{row['mime']};base64,{data}"}})
        elif row["kind"] == "document":
            ext = Path(str(row["path"])).suffix.lstrip(".").lower()
            text = attachment_text(store.path_of(row), ext, cap)
            parts.append({"type": "text", "text": f"[attached file: {name}]\n{text}"})
        elif row["kind"] == "audio":
            text = row.get("transcript")
            if not text:
                transcribe = ATTACHMENT_HOOKS.get("transcribe")
                if not callable(transcribe):
                    raise ValueError("audio attachments need speech-to-text, which is not "
                                     "available here (pip install mlx-whisper)")
                text = transcribe(store, row)
            parts.append({"type": "text",
                          "text": f"[voice message from the user, transcribed: {name}]\n{text}"})
    return parts


# --------------------------------------------------------------------------- #
# Upload reading (shared with audio.py)
# --------------------------------------------------------------------------- #
def _att_multipart_available() -> bool:
    import importlib.util  # noqa: PLC0415
    return any(importlib.util.find_spec(n) is not None
               for n in ("python_multipart", "multipart"))


async def read_upload(request: Any, cap: int, *, file_field: str = "file") -> UploadBody:
    """The uploaded bytes from a raw body (X-Filename header) or multipart form.

    Streamed with a hard cap like /api/import, so a missing or lying
    Content-Length cannot make the server buffer an unbounded body. Multipart
    gets a little headroom for its boundaries and form fields.
    """
    ctype = (request.headers.get("content-type") or "").lower()
    multipart = ctype.startswith("multipart/form-data")
    limit = cap + (256 * 1024 if multipart else 0)
    try:
        declared = int(request.headers.get("content-length") or 0)
    except ValueError:
        declared = 0
    too_big = AttachmentError(f"upload exceeds ATTACHMENT_MAX_BYTES ({cap})", 413)
    if declared and declared > limit:
        raise too_big
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise too_big
        chunks.append(chunk)
    body = b"".join(chunks)
    if not multipart:
        if len(body) > cap:
            raise too_big
        return UploadBody(body, _att_clean_name(request.headers.get("x-filename", "")))
    if not _att_multipart_available():
        raise AttachmentError("multipart uploads need python-multipart; send the raw file "
                              "as the body with an X-Filename header instead", 415)
    from starlette.formparsers import MultiPartParser  # noqa: PLC0415

    async def _att_body():
        yield body
        yield b""

    try:
        form = await MultiPartParser(request.headers, _att_body(), max_files=1,
                                     max_fields=32).parse()
    except Exception as exc:  # noqa: BLE001
        raise AttachmentError(f"malformed multipart body: {redact_text(str(exc))}", 400) from None
    try:
        upload = form.get(file_field)
        fields = {k: v for k, v in form.multi_items() if isinstance(v, str)}
        if upload is None or isinstance(upload, str) or not hasattr(upload, "read"):
            raise AttachmentError(f"multipart upload needs a '{file_field}' file field", 400)
        data = await upload.read()
        filename = _att_clean_name(getattr(upload, "filename", "") or "")
    finally:
        await form.close()
    if len(data) > cap:
        raise too_big
    return UploadBody(data, filename, fields)


def attachment_headers(row: dict) -> dict:
    """Headers for serving a stored file: inline media, everything else downloads."""
    inline = row["kind"] in ("image", "audio")
    name = row.get("filename") or row["id"]
    ascii_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120] or row["id"]
    return {
        "Content-Disposition": (f"{'inline' if inline else 'attachment'}; "
                                f"filename=\"{ascii_name}\"; "
                                f"filename*=UTF-8''{quote(name, safe='')}"),
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; sandbox",
        "Cache-Control": "private, max-age=3600",
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def register_attachments(app: Any, ctx: Any) -> None:
    from fastapi.responses import JSONResponse, Response  # noqa: PLC0415

    config, db = ctx.config, ctx.db
    _ATT_RUNTIME["config"] = config
    store = AttachmentStore(db, config)
    ctx.extras["attachments"] = store
    USER = ctx.USER

    def _att_refused(exc: AttachmentError) -> Any:
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.post("/api/attachments")
    async def attachment_upload(request: _AttRequest, user: dict = USER):
        if not getattr(config, "attachments_enabled", True):
            return JSONResponse({"error": "attachments are switched off "
                                          "(ATTACHMENTS_ENABLED=0)"}, status_code=403)
        uid = user["id"]
        try:
            upload = await read_upload(request, int(config.attachment_max_bytes))
        except AttachmentError as exc:
            return _att_refused(exc)
        cid = (upload.fields.get("conversation_id")
               or request.query_params.get("conversation_id")
               or request.headers.get("x-conversation-id") or "").strip() or None
        if cid is not None:
            if len(cid) > 64:
                return JSONResponse({"error": "conversation_id is at most 64 characters"},
                                    status_code=400)
            if not await asyncio.to_thread(db.can_access_conversation, cid, uid):
                return JSONResponse({"error": "conversation not found"}, status_code=404)
        try:
            row = await asyncio.to_thread(store.save, uid, upload.data, upload.filename, cid)
        except AttachmentError as exc:
            return _att_refused(exc)
        except Exception as exc:  # noqa: BLE001
            log(f"Attachment upload failed: {exc}", logging.WARNING)
            return JSONResponse({"error": redact_text(str(exc))}, status_code=500)
        log_event(_att_logger, logging.INFO, "attachments.uploaded", attachment_id=row["id"],
                  user_id=uid, kind=row["kind"], mime=row["mime"], size_bytes=row["bytes"])
        return _att_public(row)

    @app.get("/api/attachments")
    async def attachment_list(conversation_id: str | None = None, user: dict = USER):
        rows = await asyncio.to_thread(store.list, user["id"], conversation_id)
        return {"attachments": [_att_public(r) for r in rows]}

    @app.get("/api/attachments/{attachment_id}")
    async def attachment_get(attachment_id: str, user: dict = USER):
        row = await asyncio.to_thread(store.get, attachment_id, user["id"])
        if row is None:
            return JSONResponse({"error": "attachment not found"}, status_code=404)
        try:
            data = await asyncio.to_thread(store.read, row)
        except LookupError:
            return JSONResponse({"error": "attachment not found"}, status_code=404)
        media = (f"{row['mime']}; charset=utf-8" if row["mime"].startswith("text/")
                 else row["mime"])
        return Response(content=data, media_type=media, headers=attachment_headers(row))

    @app.delete("/api/attachments/{attachment_id}")
    async def attachment_delete(attachment_id: str, user: dict = USER):
        removed = await asyncio.to_thread(store.delete, attachment_id, user["id"])
        if not removed:
            return JSONResponse({"error": "attachment not found"}, status_code=404)
        log_event(_att_logger, logging.INFO, "attachments.deleted",
                  attachment_id=attachment_id, user_id=user["id"])
        return {"deleted": True, "id": attachment_id}


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ATTACHMENTS_SCHEMA_NAME',
    'ATTACHMENT_AUDIO_TYPES',
    'ATTACHMENT_DOCUMENT_TYPES',
    'ATTACHMENT_HOOKS',
    'ATTACHMENT_IMAGE_TYPES',
    'AttachmentError',
    'AttachmentStore',
    'SniffedFile',
    'UploadBody',
    '_ATT_MAX_PIXELS',
    '_ATT_RUNTIME',
    '_AttRequest',
    '_att_cfg',
    '_att_clean_name',
    '_att_multipart_available',
    '_att_public',
    '_att_schema',
    'attachment_headers',
    'attachment_text',
    'content_parts_for',
    'read_upload',
    'register_attachments',
    'sniff_attachment',
]
