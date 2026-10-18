"""Speech in and out: OpenAI-shaped /v1/audio/transcriptions and /v1/audio/speech.

Voice was the one input a phone or a meeting recording produces that the app
could not take. Speech-to-text runs mlx-whisper and text-to-speech runs
mlx-audio, both in-process and both optional: without them the routes answer
503 in the provider's error shape with the install hint, so an OpenAI client
gets a readable reason instead of a 404 and nothing else changes.

Facts these calls rest on (read from the wheels, not executed):
  * mlx-whisper 0.4.3: mlx_whisper.transcribe(audio, path_or_hf_repo=...,
    verbose=..., **decode_options) where audio is a path or a float32 array at
    16 kHz; a path is decoded by the ffmpeg CLI. Default repo
    "mlx-community/whisper-tiny". Returns {"text", "segments", "language"}.
  * mlx-audio 0.5.9: mlx_audio.tts.utils.load(repo) and
    model.generate(text, voice=..., speed=...) yielding results with .audio and
    .sample_rate (as its own server's /v1/audio/speech does); mp3 encoding there
    shells out to ffmpeg. README voices for Kokoro include "af_heart".

Only the configured models (WHISPER_MODEL, TTS_MODEL) are ever loaded: a
request's "model" is accepted as a label (whisper-1, tts-1, ...) and never used
as a repo id, because loading what a client names would let any user make the
server download arbitrary repositories. Both run under the generation
admission gate (they share the GPU and the memory with the chat model) and
behind a lock (one Metal inference of each kind at a time).
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import logging
import re
import shutil
import struct
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .attachments import *  # noqa: F401,F403


AUDIO_STT_HINT = "pip install mlx-whisper (and ffmpeg on PATH for mp3, m4a, webm and ogg)"
AUDIO_TTS_HINT = "pip install mlx-audio (and ffmpeg on PATH for mp3 output)"
AUDIO_TRANSCRIPT_FORMATS = ("json", "text", "verbose_json")
AUDIO_SPEECH_FORMATS = ("wav", "mp3")
AUDIO_SPEECH_MAX_CHARS = 4096          # OpenAI's limit for /v1/audio/speech input
# OpenAI voice names have no meaning for a local TTS model; they map to TTS_VOICE.
_AUD_OPENAI_VOICES = frozenset({"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx",
                                "nova", "sage", "shimmer", "verse"})
_AUD_VOICE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_AUD_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,4})?$")
_AUD_WHISPER_RATE = 16000

# Engine overrides: tests and extensions install a callable here instead of
# the mlx packages. stt: fn(data, mime, model, language) -> {"text", ...};
# tts: fn(text, voice, model, speed) -> (float samples, sample_rate).
# Mutated in place, never rebound (star imports copy bindings).
_AUD_ENGINES: dict[str, Callable | None] = {"stt": None, "tts": None}
_AUD_STT_LOCK = threading.Lock()
_AUD_TTS_LOCK = threading.Lock()
_AUD_TTS_CACHE: dict[str, Any] = {}

_aud_logger = get_logger("audio")


class AudioUnavailable(RuntimeError):
    """The engine (or a decoder it needs) is not installed here."""


def set_audio_engine(kind: str, fn: Callable | None) -> Callable | None:
    """Install (or with None remove) an stt/tts engine; returns the previous one."""
    if kind not in _AUD_ENGINES:
        raise ValueError("kind must be 'stt' or 'tts'")
    previous = _AUD_ENGINES[kind]
    _AUD_ENGINES[kind] = fn
    return previous


def stt_status() -> tuple[bool, str]:
    if _AUD_ENGINES["stt"] is not None:
        return True, ""
    if importlib.util.find_spec("mlx_whisper") is None:
        return False, f"speech-to-text is not installed: {AUDIO_STT_HINT}"
    return True, ""


def tts_status() -> tuple[bool, str]:
    if _AUD_ENGINES["tts"] is not None:
        return True, ""
    if importlib.util.find_spec("mlx_audio") is None:
        return False, f"text-to-speech is not installed: {AUDIO_TTS_HINT}"
    return True, ""


# --------------------------------------------------------------------------- #
# Decoding and encoding
# --------------------------------------------------------------------------- #
def _aud_wav_samples(data: bytes) -> tuple[Any, float] | None:
    """(float32 mono samples at 16 kHz, seconds) for a PCM16 WAV, else None.

    WAV is the one format decoded here, so a voice note from a client that
    records WAV works without ffmpeg; everything else goes through ffmpeg.
    """
    try:
        import numpy as np  # noqa: PLC0415
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except Exception:  # noqa: BLE001 (not PCM, truncated, or numpy missing)
        return None
    if width != 2 or channels < 1 or rate <= 0:
        return None
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples[: len(samples) - len(samples) % channels]
        samples = samples.reshape(-1, channels).mean(axis=1)
    seconds = len(samples) / float(rate)
    if rate != _AUD_WHISPER_RATE and len(samples):
        target = max(1, int(round(len(samples) * _AUD_WHISPER_RATE / rate)))
        samples = np.interp(np.linspace(0, len(samples) - 1, target),
                            np.arange(len(samples)), samples).astype(np.float32)
    return samples, seconds


def encode_wav(samples: Any, rate: int) -> bytes:
    """PCM16 mono WAV from float samples in [-1, 1] (a list or an array)."""
    try:
        import numpy as np  # noqa: PLC0415
        arr = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
        pcm = (arr * 32767.0).astype("<i2").tobytes()
    except ImportError:
        pcm = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, float(s))) * 32767))
                       for s in samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(rate))
        wav.writeframes(pcm)
    return buf.getvalue()


def _aud_encode(samples: Any, rate: int, fmt: str) -> tuple[bytes, str]:
    if fmt == "wav":
        return encode_wav(samples, rate), "audio/wav"
    if shutil.which("ffmpeg") is None or importlib.util.find_spec("mlx_audio") is None:
        raise AudioUnavailable("mp3 output needs mlx-audio and ffmpeg on PATH; "
                               "ask for response_format=wav")
    import numpy as np  # noqa: PLC0415
    from mlx_audio.audio_io import write as _mlx_audio_write  # noqa: PLC0415
    buf = io.BytesIO()
    _mlx_audio_write(buf, np.asarray(samples, dtype=np.float32), int(rate), format="mp3")
    return buf.getvalue(), "audio/mpeg"


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
def _aud_whisper(data: bytes, mime: str, model: str, language: str | None) -> dict:
    import mlx_whisper  # noqa: PLC0415 (optional, heavy)
    decoded = _aud_wav_samples(data) if mime == "audio/wav" else None
    tmp_path: Path | None = None
    if decoded is not None:
        source: Any = decoded[0]
    else:
        if shutil.which("ffmpeg") is None:
            raise AudioUnavailable(f"decoding {mime} needs ffmpeg on PATH (brew install ffmpeg)")
        suffix = "." + ATTACHMENT_AUDIO_TYPES.get(mime, "bin")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp_path = Path(fh.name)
        source = str(tmp_path)
    options: dict[str, Any] = {"path_or_hf_repo": model, "verbose": None}
    if language:
        options["language"] = language
    try:
        with _AUD_STT_LOCK:
            out = mlx_whisper.transcribe(source, **options)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    if decoded is not None and isinstance(out, dict):
        out.setdefault("duration", decoded[1])
    return out if isinstance(out, dict) else {"text": str(out or "")}


def transcribe_audio(data: bytes, mime: str, *, model: str,
                     language: str | None = None) -> dict:
    """{"text", "language", "duration", "segments"} for audio bytes."""
    engine = _AUD_ENGINES["stt"]
    if engine is not None:
        raw = engine(data, mime, model, language)
    else:
        ok, reason = stt_status()
        if not ok:
            raise AudioUnavailable(reason)
        raw = _aud_whisper(data, mime, model, language)
    raw = raw if isinstance(raw, dict) else {"text": str(raw or "")}
    segments = [
        {"id": i, "start": float(s.get("start", 0.0) or 0.0),
         "end": float(s.get("end", 0.0) or 0.0), "text": str(s.get("text", ""))}
        for i, s in enumerate(raw.get("segments") or []) if isinstance(s, dict)
    ]
    duration = raw.get("duration")
    if duration is None and segments:
        duration = segments[-1]["end"]
    return {"text": str(raw.get("text") or "").strip(),
            "language": str(raw.get("language") or language or ""),
            "duration": float(duration or 0.0), "segments": segments}


def _aud_mlx_audio(text: str, voice: str, model: str, speed: float) -> tuple[Any, int]:
    import numpy as np  # noqa: PLC0415
    from mlx_audio.tts.utils import load as _mlx_audio_load  # noqa: PLC0415
    with _AUD_TTS_LOCK:
        tts = _AUD_TTS_CACHE.get(model)
        if tts is None:
            # One TTS model resident at a time: on a small Mac the chat model
            # owns the memory, so a changed TTS_MODEL replaces, never adds.
            _AUD_TTS_CACHE.clear()
            tts = _AUD_TTS_CACHE[model] = _mlx_audio_load(model)
        chunks, rate = [], None
        for result in tts.generate(text, voice=voice, speed=speed):
            chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            rate = rate or getattr(result, "sample_rate", None)
    if not chunks:
        raise RuntimeError("the TTS model produced no audio")
    return np.concatenate(chunks), int(rate or getattr(tts, "sample_rate", 24000) or 24000)


def synthesize_speech(text: str, *, voice: str, model: str, fmt: str = "wav",
                      speed: float = 1.0) -> tuple[bytes, str]:
    """(encoded audio, content type) for text."""
    engine = _AUD_ENGINES["tts"]
    if engine is not None:
        samples, rate = engine(text, voice, model, speed)
    else:
        ok, reason = tts_status()
        if not ok:
            raise AudioUnavailable(reason)
        samples, rate = _aud_mlx_audio(text, voice, model, speed)
    return _aud_encode(samples, int(rate), fmt)


def _aud_transcribe_attachment(store: Any, row: dict) -> str:
    """ATTACHMENT_HOOKS["transcribe"]: transcript of a stored audio attachment."""
    if row.get("transcript"):
        return str(row["transcript"])
    ok, reason = stt_status()
    if not ok:
        raise ValueError(reason)
    model = str(_att_cfg("whisper_model", "") or "mlx-community/whisper-tiny")
    try:
        result = transcribe_audio(store.read(row), row["mime"], model=model)
    except AudioUnavailable as exc:
        raise ValueError(str(exc)) from None
    store.set_transcript(row["id"], row["user_id"], result["text"])
    return result["text"]


ATTACHMENT_HOOKS["transcribe"] = _aud_transcribe_attachment


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _aud_oai_error(status: int, message: str, code: str,
                   kind: str = "invalid_request_error", headers: dict | None = None) -> Any:
    from fastapi.responses import JSONResponse  # noqa: PLC0415
    return JSONResponse({"error": {"message": message, "type": kind, "param": None,
                                   "code": code}}, status_code=status, headers=headers)


def _aud_gate(ctx: Any) -> Any:
    """The generation gate's slot() context manager, or a no-op without a gate."""
    gate = getattr(ctx, "generation_gate", None)
    slot = getattr(gate, "slot", None)
    if callable(slot):
        return slot()
    return _AudAsyncNull()


class _AudAsyncNull:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def register_audio(app: Any, ctx: Any) -> None:
    from fastapi.responses import JSONResponse, PlainTextResponse, Response  # noqa: PLC0415

    config = ctx.config
    _ATT_RUNTIME["config"] = config
    USER = ctx.USER

    def _stt_model() -> str:
        return str(getattr(config, "whisper_model", "") or "mlx-community/whisper-tiny")

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: _AttRequest, user: dict = USER):
        ok, reason = stt_status()
        if not ok:
            return _aud_oai_error(503, reason, "stt_unavailable", "server_error")
        try:
            upload = await read_upload(request, int(config.attachment_max_bytes))
        except AttachmentError as exc:
            return _aud_oai_error(exc.status, str(exc), "invalid_file")
        fields = {**dict(request.query_params), **upload.fields}
        fmt = str(fields.get("response_format") or "json").strip().lower()
        if fmt not in AUDIO_TRANSCRIPT_FORMATS:
            return _aud_oai_error(400, f"response_format must be one of "
                                       f"{', '.join(AUDIO_TRANSCRIPT_FORMATS)}",
                                  "invalid_response_format")
        language = str(fields.get("language") or "").strip() or None
        if language is not None and not _AUD_LANG_RE.match(language):
            return _aud_oai_error(400, "language must be an ISO-639 code such as 'en'",
                                  "invalid_language")
        try:
            sniffed = sniff_attachment(upload.data, upload.filename)
        except AttachmentError as exc:
            return _aud_oai_error(exc.status, str(exc), "invalid_file")
        if sniffed.kind != "audio":
            return _aud_oai_error(400, "the file is not audio (wav, mp3, m4a, webm or ogg)",
                                  "invalid_file")
        started = time.time()
        try:
            async with _aud_gate(ctx):
                result = await asyncio.to_thread(transcribe_audio, upload.data, sniffed.mime,
                                                 model=_stt_model(), language=language)
        except GenerationBusy as exc:
            return _aud_oai_error(503, str(exc), "busy", "server_error",
                                  headers={"Retry-After": "5"})
        except AudioUnavailable as exc:
            return _aud_oai_error(503, str(exc), "stt_unavailable", "server_error")
        except Exception as exc:  # noqa: BLE001
            log(f"Transcription failed: {exc}", logging.WARNING)
            return _aud_oai_error(500, redact_text(str(exc)), "transcription_failed",
                                  "server_error")
        log_event(_aud_logger, logging.INFO, "audio.transcribed", user_id=user["id"],
                  mime=sniffed.mime, size_bytes=len(upload.data),
                  duration_s=round(result["duration"], 2),
                  elapsed_ms=int((time.time() - started) * 1000))
        if fmt == "text":
            return PlainTextResponse(result["text"])
        if fmt == "verbose_json":
            return {"task": "transcribe", "language": result["language"],
                    "duration": result["duration"], "text": result["text"],
                    "segments": result["segments"]}
        return {"text": result["text"]}

    @app.post("/v1/audio/speech")
    async def audio_speech(body: dict, user: dict = USER):
        ok, reason = tts_status()
        if not ok:
            return _aud_oai_error(503, reason, "tts_unavailable", "server_error")
        text = body.get("input") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return _aud_oai_error(400, "input must be a non-empty string", "invalid_input")
        if len(text) > AUDIO_SPEECH_MAX_CHARS:
            return _aud_oai_error(400, f"input is at most {AUDIO_SPEECH_MAX_CHARS} characters",
                                  "invalid_input")
        voice = str(body.get("voice") or "").strip()
        if not voice or voice.lower() in _AUD_OPENAI_VOICES:
            voice = str(getattr(config, "tts_voice", "") or "af_heart")
        if not _AUD_VOICE_RE.match(voice):
            return _aud_oai_error(400, "voice must be a short voice name", "invalid_voice")
        # OpenAI's default is mp3; without an mp3 encoder here the default
        # becomes wav (the Content-Type says which), an explicit mp3 is a 400.
        default_fmt = ("mp3" if shutil.which("ffmpeg") and
                       importlib.util.find_spec("mlx_audio") else "wav")
        fmt = str(body.get("response_format") or default_fmt).strip().lower()
        if fmt not in AUDIO_SPEECH_FORMATS:
            return _aud_oai_error(400, "response_format must be wav or mp3",
                                  "invalid_response_format")
        try:
            speed = float(body.get("speed", 1.0) or 1.0)
        except (TypeError, ValueError):
            return _aud_oai_error(400, "speed must be a number", "invalid_speed")
        speed = min(4.0, max(0.25, speed))
        model = str(getattr(config, "tts_model", "") or "mlx-community/Kokoro-82M-bf16")
        started = time.time()
        try:
            async with _aud_gate(ctx):
                audio, ctype = await asyncio.to_thread(
                    synthesize_speech, text, voice=voice, model=model, fmt=fmt, speed=speed)
        except GenerationBusy as exc:
            return _aud_oai_error(503, str(exc), "busy", "server_error",
                                  headers={"Retry-After": "5"})
        except AudioUnavailable as exc:
            status = 400 if fmt == "mp3" and "mp3" in str(exc) else 503
            return _aud_oai_error(status, str(exc), "tts_unavailable",
                                  "invalid_request_error" if status == 400 else "server_error")
        except Exception as exc:  # noqa: BLE001
            log(f"Speech synthesis failed: {exc}", logging.WARNING)
            return _aud_oai_error(500, redact_text(str(exc)), "speech_failed", "server_error")
        log_event(_aud_logger, logging.INFO, "audio.spoken", user_id=user["id"],
                  chars=len(text), format=fmt, size_bytes=len(audio),
                  elapsed_ms=int((time.time() - started) * 1000))
        return Response(content=audio, media_type=ctype,
                        headers={"X-Content-Type-Options": "nosniff"})

    @app.post("/api/attachments/{attachment_id}/transcribe")
    async def attachment_transcribe(attachment_id: str, user: dict = USER):
        store = AttachmentStore(ctx.db, config)
        row = await asyncio.to_thread(store.get, attachment_id, user["id"])
        if row is None:
            return JSONResponse({"error": "attachment not found"}, status_code=404)
        if row["kind"] != "audio":
            return JSONResponse({"error": "only audio attachments can be transcribed"},
                                status_code=400)
        if row.get("transcript"):
            return {"id": row["id"], "text": row["transcript"], "cached": True}
        ok, reason = stt_status()
        if not ok:
            return JSONResponse({"error": reason, "hint": AUDIO_STT_HINT}, status_code=503)
        try:
            data = await asyncio.to_thread(store.read, row)
            async with _aud_gate(ctx):
                result = await asyncio.to_thread(transcribe_audio, data, row["mime"],
                                                 model=_stt_model())
        except LookupError:
            return JSONResponse({"error": "attachment not found"}, status_code=404)
        except GenerationBusy as exc:
            return JSONResponse({"error": str(exc)}, status_code=503,
                                headers={"Retry-After": "5"})
        except AudioUnavailable as exc:
            return JSONResponse({"error": str(exc), "hint": AUDIO_STT_HINT}, status_code=503)
        except Exception as exc:  # noqa: BLE001
            log(f"Attachment transcription failed: {exc}", logging.WARNING)
            return JSONResponse({"error": redact_text(str(exc))}, status_code=500)
        await asyncio.to_thread(store.set_transcript, row["id"], user["id"], result["text"])
        return {"id": row["id"], "text": result["text"], "language": result["language"],
                "duration": result["duration"], "cached": False}


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'AUDIO_SPEECH_FORMATS',
    'AUDIO_SPEECH_MAX_CHARS',
    'AUDIO_STT_HINT',
    'AUDIO_TRANSCRIPT_FORMATS',
    'AUDIO_TTS_HINT',
    'AudioUnavailable',
    '_AUD_ENGINES',
    '_AudAsyncNull',
    '_aud_encode',
    '_aud_gate',
    '_aud_oai_error',
    '_aud_transcribe_attachment',
    '_aud_wav_samples',
    'encode_wav',
    'register_audio',
    'set_audio_engine',
    'stt_status',
    'synthesize_speech',
    'transcribe_audio',
    'tts_status',
]
