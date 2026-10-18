"""Tests for multimodal input: attachments, audio routes and the mlx-vlm backend."""
from __future__ import annotations

import importlib.util
import io
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-attachments-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.attachments as att_mod  # noqa: E402
import local_llm.audio as audio_mod  # noqa: E402
import local_llm.vision as vision_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.backends import get_backend, request_model_for  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import current_turn  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    """Same construction as tests/test_app.py:build_app."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"),
                agent_enabled=True, knowledge_triage=False)
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    mm.status = "ready"
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")
    return app, cfg, db


def _cookie(c: TestClient, username: str, password: str) -> str:
    c.cookies.clear()
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    value = "; ".join(f"{k}={v}" for k, v in c.cookies.items())
    c.cookies.clear()
    return value


def _png(w: int = 4, h: int = 3) -> bytes:
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
            + struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00" + b"\x00" * 4
            + struct.pack(">I", 0) + b"IEND" + b"\xaeB`\x82")


def _jpeg(w: int = 640, h: int = 480) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", h, w) + b"\x03" + b"\x00" * 9
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def _webp_vp8x(w: int = 800, h: int = 600) -> bytes:
    payload = b"VP8X" + struct.pack("<I", 10) + b"\x00" * 4 + (w - 1).to_bytes(3, "little") \
        + (h - 1).to_bytes(3, "little")
    return b"RIFF" + struct.pack("<I", 4 + len(payload)) + b"WEBP" + payload


def _wav(seconds: float = 0.1, rate: int = 16000) -> bytes:
    return audio_mod.encode_wav([0.0] * int(seconds * rate), rate)


def _docx() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", "<w:document/>")
    return buf.getvalue()


def _final(answer: str = "ok") -> dict:
    return {"type": "final", "answer": answer, "steps": 1, "trace": [], "tools_used": [],
            "elapsed_ms": 1, "prompt_tokens": 1, "completion_tokens": 1, "truncated": False,
            "changed_files": [], "diff": ""}


def _upload(c, cookie, data: bytes, name: str, **headers):
    return c.post("/api/attachments", content=data,
                  headers={"cookie": cookie, "X-Filename": name, **headers})


# --------------------------------------------------------------------------- #
# Sniffing
# --------------------------------------------------------------------------- #
def test_sniffing_trusts_bytes_not_names():
    s = att_mod.sniff_attachment
    png = s(_png(4, 3), "x.png")
    assert (png.kind, png.mime, png.width, png.height) == ("image", "image/png", 4, 3)
    jpg = s(_jpeg(640, 480), "photo.bin")
    assert (jpg.mime, jpg.width, jpg.height) == ("image/jpeg", 640, 480)
    gif = s(b"GIF89a" + struct.pack("<HH", 10, 20) + b"\x00" * 10)
    assert (gif.mime, gif.width, gif.height) == ("image/gif", 10, 20)
    webp = s(_webp_vp8x(800, 600))
    assert (webp.mime, webp.width, webp.height) == ("image/webp", 800, 600)
    assert s(_wav(), "a.wav").mime == "audio/wav"
    assert s(b"OggS" + b"\x00" * 40).mime == "audio/ogg"
    assert s(b"\x1a\x45\xdf\xa3" + b"\x00" * 40).mime == "audio/webm"
    assert s(b"ID3\x04\x00" + b"\x00" * 40).mime == "audio/mpeg"
    assert s(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 40).mime == "audio/mp4"
    assert s(b"%PDF-1.4\n%%EOF\n").mime == "application/pdf"
    assert s(_docx(), "report.docx").ext == "docx"
    csv = s(b"a,b\n1,2\n", "table.csv")
    assert (csv.kind, csv.mime, csv.ext) == ("document", "text/csv", "csv")
    # An "image" that is really HTML is text, never an image, whatever its name.
    html = s(b"<html><script>alert(1)</script></html>", "cat.png")
    assert html.kind == "document" and html.mime == "text/plain"
    for bad, status in ((b"\x00\x01\x02\x03binary", 415), (b"", 400),
                        (_png(20000, 20000), 413), (b"\x89PNG\r\n\x1a\nshort", 400)):
        try:
            s(bad, "x")
        except att_mod.AttachmentError as exc:
            assert exc.status == status, (bad[:10], exc.status)
        else:
            raise AssertionError(f"accepted {bad[:12]!r}")
    plain_zip = io.BytesIO()
    with zipfile.ZipFile(plain_zip, "w") as z:
        z.writestr("evil.sh", "rm -rf /")
    try:
        s(plain_zip.getvalue(), "x.docx")
    except att_mod.AttachmentError as exc:
        assert exc.status == 415
    else:
        raise AssertionError("accepted a plain zip")


# --------------------------------------------------------------------------- #
# Upload, fetch, delete, isolation
# --------------------------------------------------------------------------- #
def test_upload_limits_serving_and_owner_isolation():
    app, cfg, db = build_app(attachment_max_bytes=4096)
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        bob = _cookie(c, "bob", "bobpw12345")
        r = _upload(c, admin, _png(4, 3), "shot.png")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["kind"] == "image" and body["mime"] == "image/png"
        assert body["bytes"] == len(_png(4, 3)) and body["width"] == 4
        att_id = body["id"]
        assert att_id.startswith("att_")
        # Stored next to the database (DATA_DIR in production), per user.
        files = list((Path(db.db_path).parent / "attachments").rglob(f"{att_id}.png"))
        assert len(files) == 1

        assert _upload(c, admin, b"x" * 5000, "big.txt").status_code == 413
        assert _upload(c, admin, b"\x00\x01binary\x00", "a.bin").status_code == 415

        r = c.get(f"/api/attachments/{att_id}", headers={"cookie": admin})
        assert r.status_code == 200 and r.content == _png(4, 3)
        assert r.headers["content-type"] == "image/png"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["content-disposition"].startswith("inline")
        assert "sandbox" in r.headers["content-security-policy"]

        # Bob can neither read, list nor delete it: 404, never 403.
        assert c.get(f"/api/attachments/{att_id}", headers={"cookie": bob}).status_code == 404
        assert c.delete(f"/api/attachments/{att_id}", headers={"cookie": bob}).status_code == 404
        assert c.get("/api/attachments", headers={"cookie": bob}).json()["attachments"] == []
        assert [a["id"] for a in c.get("/api/attachments",
                                       headers={"cookie": admin}).json()["attachments"]] == [att_id]

        # Documents download rather than render, as text.
        r = _upload(c, admin, b"<h1>hi</h1>", "page.html")
        doc_id = r.json()["id"]
        r = c.get(f"/api/attachments/{doc_id}", headers={"cookie": admin})
        assert r.headers["content-type"].startswith("text/plain")
        assert r.headers["content-disposition"].startswith("attachment")

        # Multipart works too (python-multipart is installed here).
        if att_mod._att_multipart_available():
            r = c.post("/api/attachments", headers={"cookie": admin},
                       files={"file": ("notes.md", b"# notes\n", "image/png")},
                       data={"conversation_id": "conv-mm"})
            assert r.status_code == 200, r.text
            assert r.json()["mime"] == "text/markdown"           # the client's mime is ignored
            assert r.json()["conversation_id"] == "conv-mm"

        r = c.delete(f"/api/attachments/{att_id}", headers={"cookie": admin})
        assert r.status_code == 200 and r.json()["deleted"] is True
        assert c.get(f"/api/attachments/{att_id}", headers={"cookie": admin}).status_code == 404
        assert not files[0].exists()
        assert c.get("/api/attachments/../../etc/passwd",
                     headers={"cookie": admin}).status_code == 404


def test_quota_and_switch_off():
    app, cfg, db = build_app(attachment_quota_mb=1, attachment_max_bytes=900 * 1024)
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        assert _upload(c, admin, b"a" * (600 * 1024), "a.txt").status_code == 200
        r = _upload(c, admin, b"b" * (600 * 1024), "b.txt")
        assert r.status_code == 413 and "quota" in r.json()["error"]
        cfg.apply({"attachments_enabled": False})
        assert _upload(c, admin, b"hello", "c.txt").status_code == 403


# --------------------------------------------------------------------------- #
# Content parts
# --------------------------------------------------------------------------- #
def test_content_parts_for_images_documents_and_audio():
    app, cfg, db = build_app()
    store = att_mod.AttachmentStore(db, cfg)
    uid = "user-a"
    img = store.save(uid, _png(2, 2), "dot.png")
    csv = store.save(uid, b"name,qty\napple,3\npear,5\n", "fruit.csv")
    voice = store.save(uid, _wav(), "note.wav")
    other = store.save("user-b", _png(2, 2), "theirs.png")

    parts = att_mod.content_parts_for(db, uid, [img["id"], csv["id"], img["id"]])
    assert len(parts) == 2                                       # duplicates collapse
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["type"] == "text" and "fruit.csv" in parts[1]["text"]
    assert "columns: name, qty" in parts[1]["text"]              # the read_file tool's reader

    for ids in ([other["id"]], ["att_" + "0" * 24], ["../etc"]):
        try:
            att_mod.content_parts_for(db, uid, ids)
        except LookupError:
            pass
        else:
            raise AssertionError(f"resolved {ids}")

    previous = audio_mod.set_audio_engine("stt", None)
    try:
        if not audio_mod.stt_status()[0]:
            try:
                att_mod.content_parts_for(db, uid, [voice["id"]])
            except ValueError as exc:
                assert "mlx-whisper" in str(exc)
            else:
                raise AssertionError("audio resolved without speech-to-text")
        calls = []
        audio_mod.set_audio_engine(
            "stt", lambda data, mime, model, lang: calls.append((mime, model)) or
            {"text": " buy milk ", "language": "en"})
        parts = att_mod.content_parts_for(db, uid, [voice["id"]])
        assert parts[0]["type"] == "text" and parts[0]["text"].endswith("buy milk")
        assert calls == [("audio/wav", cfg.whisper_model)]
        att_mod.content_parts_for(db, uid, [voice["id"]])
        assert len(calls) == 1                                   # the transcript is cached
    finally:
        audio_mod.set_audio_engine("stt", previous)


# --------------------------------------------------------------------------- #
# Chat route
# --------------------------------------------------------------------------- #
def _capture_agent(app, seen: list):
    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, attachments=None, **kw):
        turn = current_turn()
        seen.append({"attachments": attachments,
                     "vision": (turn.extras.get("vision") if turn is not None else "no-turn")})
        yield _final("seen")
    app.state.agent.run_iterating = fake_run


def test_chat_route_vision_vs_text_backend():
    # Default install: mlx-lm serving a text model, so images are not shown to it.
    app, cfg, db = build_app()
    seen: list = []
    _capture_agent(app, seen)
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        img = _upload(c, admin, _png(), "a.png").json()["id"]
        doc = _upload(c, admin, b"hello from a file", "a.txt").json()["id"]
        r = c.post("/api/chat", json={"message": "what is this?", "attachments": [img, doc]},
                   headers={"cookie": admin})
        assert r.status_code == 200, r.text
        assert seen[-1]["vision"] is False
        kinds = [p["type"] for p in seen[-1]["attachments"]]
        assert kinds == ["image_url", "text"]
        # Documents alone: no vision decision is needed or made.
        r = c.post("/api/chat", json={"message": "summarise", "attachments": [doc]},
                   headers={"cookie": admin})
        assert r.status_code == 200 and seen[-1]["vision"] is None
        # Someone else's id is a 404 before anything streams.
        bob = _cookie(c, "bob", "bobpw12345")
        r = c.post("/api/chat/stream", json={"message": "x", "attachments": [img]},
                   headers={"cookie": bob})
        assert r.status_code == 404

    # VISION_MODE=on (or an mlx-vlm engine, below) lets the image through.
    app, cfg, db = build_app(vision_mode="on")
    seen = []
    _capture_agent(app, seen)
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        img = _upload(c, admin, _png(), "a.png").json()["id"]
        r = c.post("/api/chat/stream", json={"message": "describe", "attachments": [img]},
                   headers={"cookie": admin})
        assert r.status_code == 200, r.text
        assert seen[-1]["vision"] is True
        assert seen[-1]["attachments"][0]["type"] == "image_url"


def test_served_vision_asks_engine_and_model():
    def cfg(model, backend="mlx-lm", mode="auto"):
        return SimpleNamespace(model=model, model_backend=backend, vision_mode=mode)

    text_model = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
    vl_model = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    assert vision_mod.served_vision(None, cfg(text_model))[0] is False
    # mlx-lm declares "vision" but cannot run Gemma 3's vision tower.
    assert vision_mod.served_vision(None, cfg("mlx-community/gemma-3-4b-it-4bit"))[0] is False
    assert vision_mod.served_vision(None, cfg(vl_model, "mlx-vlm"))[0] is True
    assert vision_mod.served_vision(None, cfg(text_model, "mlx-vlm"))[0] is False
    assert vision_mod.served_vision(None, cfg(text_model, mode="on"))[0] is True
    assert vision_mod.served_vision(None, cfg(vl_model, "mlx-vlm", "off"))[0] is False
    manager = SimpleNamespace(backend=get_backend("mlx-vlm"), model_id=vl_model,
                              decode_concurrency=0, kv_bits=0)
    assert vision_mod.served_vision(manager, cfg(text_model))[0] is True


# --------------------------------------------------------------------------- #
# Audio routes
# --------------------------------------------------------------------------- #
def test_audio_routes_are_provider_shaped_503_without_engines():
    if importlib.util.find_spec("mlx_whisper") or importlib.util.find_spec("mlx_audio"):
        return  # the real engines are installed here; the 503 path is not reachable
    app, cfg, db = build_app()
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        r = c.post("/v1/audio/transcriptions", content=_wav(),
                   headers={"cookie": admin, "X-Filename": "a.wav"})
        assert r.status_code == 503
        err = r.json()["error"]
        assert err["code"] == "stt_unavailable" and "mlx-whisper" in err["message"]
        r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "alloy"},
                   headers={"cookie": admin})
        assert r.status_code == 503 and r.json()["error"]["code"] == "tts_unavailable"
        assert c.post("/v1/audio/transcriptions", content=_wav()).status_code == 401


def test_audio_routes_with_installed_engines():
    stt_calls: list = []
    tts_calls: list = []

    def fake_stt(data, mime, model, language):
        stt_calls.append((mime, model, language))
        return {"text": "hello world", "language": language or "en",
                "segments": [{"start": 0.0, "end": 1.5, "text": "hello world"}]}

    def fake_tts(text, voice, model, speed):
        tts_calls.append((text, voice, model, speed))
        return [0.0] * 1600, 16000

    prev_stt = audio_mod.set_audio_engine("stt", fake_stt)
    prev_tts = audio_mod.set_audio_engine("tts", fake_tts)
    try:
        app, cfg, db = build_app()
        with TestClient(app) as c:
            admin = _cookie(c, "admin", "adminpw123")
            r = c.post("/v1/audio/transcriptions", headers={"cookie": admin},
                       files={"file": ("a.wav", _wav(), "audio/wav")},
                       data={"model": "some-org/huge-model", "language": "nl"})
            assert r.status_code == 200, r.text
            assert r.json() == {"text": "hello world"}
            # The request's model is a label: only WHISPER_MODEL is ever loaded.
            assert stt_calls[-1] == ("audio/wav", cfg.whisper_model, "nl")
            r = c.post("/v1/audio/transcriptions?response_format=text", content=_wav(),
                       headers={"cookie": admin, "X-Filename": "a.wav"})
            assert r.status_code == 200 and r.text == "hello world"
            r = c.post("/v1/audio/transcriptions", headers={"cookie": admin},
                       files={"file": ("a.wav", _wav(), "audio/wav")},
                       data={"response_format": "verbose_json"})
            body = r.json()
            assert body["task"] == "transcribe" and body["duration"] == 1.5
            assert body["segments"][0]["text"] == "hello world"
            r = c.post("/v1/audio/transcriptions", headers={"cookie": admin},
                       files={"file": ("a.png", _png(), "audio/wav")})
            assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_file"
            r = c.post("/v1/audio/transcriptions?response_format=srt", content=_wav(),
                       headers={"cookie": admin})
            assert r.status_code == 400

            r = c.post("/v1/audio/speech", headers={"cookie": admin},
                       json={"model": "tts-1", "input": "Hello there", "voice": "alloy",
                             "response_format": "wav"})
            assert r.status_code == 200, r.text
            assert r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
            assert tts_calls[-1] == ("Hello there", cfg.tts_voice, cfg.tts_model, 1.0)
            r = c.post("/v1/audio/speech", headers={"cookie": admin},
                       json={"input": "x" * 5000})
            assert r.status_code == 400
            r = c.post("/v1/audio/speech", headers={"cookie": admin},
                       json={"input": "hi", "response_format": "flac"})
            assert r.status_code == 400

            # The composer's path: upload a voice note, transcribe it once.
            voice = _upload(c, admin, _wav(), "note.wav").json()["id"]
            r = c.post(f"/api/attachments/{voice}/transcribe", headers={"cookie": admin})
            assert r.status_code == 200 and r.json()["text"] == "hello world"
            assert r.json()["cached"] is False
            r = c.post(f"/api/attachments/{voice}/transcribe", headers={"cookie": admin})
            assert r.json()["cached"] is True
            bob = _cookie(c, "bob", "bobpw12345")
            assert c.post(f"/api/attachments/{voice}/transcribe",
                          headers={"cookie": bob}).status_code == 404
    finally:
        audio_mod.set_audio_engine("stt", prev_stt)
        audio_mod.set_audio_engine("tts", prev_tts)


# --------------------------------------------------------------------------- #
# mlx-vlm backend and config
# --------------------------------------------------------------------------- #
# The option names of `python -m mlx_vlm.server --help` in mlx-vlm 0.7.8 (read
# from the wheel's server/cli.py), as the manager's help sniffing sees them.
_VLM_078_HELP = " ".join([
    "--host", "--port", "--trust-remote-code", "--model", "--image-model", "--tts-model",
    "--stt-model", "--embedding-model", "--decision-model", "--reranker-model",
    "--model-dir", "--adapter-path", "--vision-cache-size", "--prefill-step-size",
    "--log-progress-interval", "--max-tokens", "--enable-thinking", "--thinking-budget",
    "--thinking-start-token", "--thinking-end-token", "--kv-bits", "--kv-key-bits",
    "--kv-value-bits", "--kv-key-scheme", "--kv-value-scheme", "--kv-quant-scheme",
    "--kv-group-size", "--max-kv-size", "--quantized-kv-start", "--expert-cache-gb",
    "--draft-model", "--draft-kind", "--draft-block-size", "--max-num-seqs",
    "--top-logprobs-k", "--api-key", "--reload", "--log-level",
])


def test_mlx_vlm_spec_matches_the_real_cli():
    spec = get_backend("mlx-vlm")
    assert spec.name == "mlx-vlm" and spec.help_module == "mlx_vlm.server"
    assert {"vision", "batching"} <= spec.capabilities
    assert spec.serves_adapter() is False
    m = SimpleNamespace(model_id="mlx-community/Qwen2.5-VL-3B-Instruct-4bit", model_port=9555,
                        max_kv_size=8192, kv_bits=8, kv_group_size=64, quantized_kv_start=0,
                        prefill_step_size=512, decode_concurrency=2,
                        max_concurrent_generations=4, _server_help=_VLM_078_HELP)
    cmd = spec.launcher(m, True)
    assert cmd[1:3] == ["-m", "mlx_vlm.server"]
    args = cmd[3:]
    assert args[:6] == ["--model", m.model_id, "--host", "127.0.0.1", "--port", "9555"]
    pairs = dict(zip(args[6::2], args[7::2]))
    assert pairs == {"--max-kv-size": "8192", "--kv-bits": "8", "--kv-group-size": "64",
                     "--prefill-step-size": "512", "--max-num-seqs": "2"}
    assert "--adapter-path" not in cmd                       # mlx-lm adapters are not served
    # An older build whose --help lists none of the levers still starts.
    m._server_help = ""
    assert spec.launcher(m, False)[3:] == args[:6]
    caps = spec.caps_for(m)
    assert {"vision", "batching", "kv_quant"} <= caps
    m.decode_concurrency, m.max_concurrent_generations, m.kv_bits = 1, 0, 0
    assert "batching" not in spec.caps_for(m) and "kv_quant" not in spec.caps_for(m)
    # Every request carries the exact --model id (mlx-vlm loads anything else).
    cfg = SimpleNamespace(model="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                          model_backend="mlx-vlm")
    assert request_model_for(cfg) == cfg.model
    assert vision_mod.VLM_BACKEND in [b["name"] for b in
                                      __import__("local_llm.backends", fromlist=["x"])
                                      .public_backends(cfg)]


def test_config_clamps_and_settings_coverage():
    cfg = Config(vision_mode="sideways", whisper_model="../../etc/passwd",
                 tts_voice="af heart; rm", attachment_max_bytes=10**12, attachment_quota_mb=-5)
    assert cfg.vision_mode == "auto"
    assert cfg.whisper_model == "mlx-community/whisper-tiny"
    assert cfg.tts_voice == "af_heart"
    assert cfg.attachment_max_bytes == 512 * 1024 * 1024 and cfg.attachment_quota_mb == 0
    cfg.apply({"whisper_model": "mlx-community/whisper-large-v3-turbo", "vision_mode": "on"})
    assert cfg.whisper_model == "mlx-community/whisper-large-v3-turbo"
    assert cfg.vision_mode == "on"
    for key in ("attachments_enabled", "attachment_max_bytes", "vision_mode",
                "whisper_model", "tts_model", "tts_voice"):
        assert key in Config.MUTABLE
    assert "attachments_dir" not in Config.MUTABLE


def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
